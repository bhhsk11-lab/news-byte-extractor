"""
Google News redirect-URL resolver.

Hybrid strategy:
  1. Direct 'url=' query parameter (fast).
  2. Offline base64/protobuf decode (old-style IDs).
  3. batchexecute RPC (requires proxy, often rate‑limited).
  4. Playwright browser fallback (reliable, no proxy needed).

Playwright uses a pooled browser to load the redirect page, click consent,
and follow the redirect to the real publisher URL.
"""

import asyncio
import base64
import json
import logging
import re
import time
from collections import OrderedDict
from urllib.parse import parse_qs, urlparse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from curl_cffi import requests as cffi_requests
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

from config import settings

logger = logging.getLogger("gnews-resolver")

# ── batchexecute constants ──────────────────────────────────────────
_BATCHEXECUTE_URL = "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je"
_RESOLVE_CACHE_MAX = 2000
_resolve_cache: OrderedDict = OrderedDict()
_resolve_lock = asyncio.Lock()
_last_call_ts = 0.0
_MIN_INTERVAL_S = 0.75

# ── Playwright browser pool ─────────────────────────────────────────
_BROWSER_POOL_SIZE = 3          # adjust based on your instance memory
_browser_pool: asyncio.Queue = None
_playwright_instance = None
_PW_INITIALIZED = False


def is_google_news_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return False
    return host == "news.google.com" or host.endswith(".news.google.com")


def _cache_get(url: str) -> str | None:
    if url in _resolve_cache:
        _resolve_cache.move_to_end(url)
        return _resolve_cache[url]
    return None


def _cache_put(url: str, resolved: str):
    _resolve_cache[url] = resolved
    _resolve_cache.move_to_end(url)
    while len(_resolve_cache) > _RESOLVE_CACHE_MAX:
        _resolve_cache.popitem(last=False)


def _proxies() -> dict | None:
    if settings.proxy_url:
        return {"http": settings.proxy_url, "https": settings.proxy_url}
    return None


# ── Offline decode (old IDs) ────────────────────────────────────────
def _decode_offline(b64: str) -> str | None:
    try:
        s = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4))
    except Exception:
        return None
    if s.startswith(b"\x08\x13\x22"):
        s = s[3:]
    if s.endswith(b"\xd2\x01\x00"):
        s = s[:-3]
    if not s:
        return None
    first = s[0]
    if first >= 0x80:
        s = s[2:first + 2] if len(s) >= first + 2 else s[2:]
    else:
        s = s[1:first + 1] if len(s) >= first + 1 else s[1:]
    try:
        out = s.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return out if out.startswith("http") else None


# ── batchexecute RPC (legacy) ──────────────────────────────────────
def _build_freq(gn_art_id: str, sig: str = "", ts: str = "") -> str:
    from urllib.parse import quote
    inner_req = [
        "garturlreq",
        [
            [
                ["en-US", "US", ["FINANCE_TOP_INDICES", "WEB_TEST_1_0_0"], None,
                 None, 1, 1, "US:en", None, 180, None, None, None, None, 0,
                 None, None, [1608992183, 723341000]],
                "en-US", "US", 1, [2, 3, 4, 8], 1, 0, "655000234", 0, 0,
                None, 0
            ]
        ],
        gn_art_id,
    ]
    if sig and ts:
        inner_req += [ts, sig, False]
    freq = json.dumps([[["Fbv4je", json.dumps(inner_req), None, "generic"]]])
    return "f.req=" + quote(freq)


def _get_redirect_page(url: str):
    r = cffi_requests.get(
        url,
        impersonate="chrome124",
        timeout=20,
        allow_redirects=True,
        proxies=_proxies(),
        headers={
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        },
    )
    return r.text, r.cookies


def _parse_garturlres(text: str) -> str | None:
    try:
        payload = text[text.index("["):]
        data = json.loads(payload)
        for row in data:
            try:
                inner = json.loads(row[2])
                if (isinstance(inner, list) and len(inner) > 1
                        and inner[0] == "garturlres"
                        and isinstance(inner[1], str)
                        and inner[1].startswith("http")):
                    return inner[1]
            except (json.JSONDecodeError, TypeError, IndexError, ValueError):
                continue
    except (ValueError, json.JSONDecodeError):
        pass
    m = re.search(r'garturlres\\",\\"(https?://[^\\",]+)', text)
    return m.group(1) if m else None


def _is_retryable(exception):
    if isinstance(exception, Exception):
        msg = str(exception).lower()
        return any(x in msg for x in ("timeout", "429", "rate", "too many", "500", "502", "503"))
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _post_batchexecute(freq_body: str, cookies: dict) -> str:
    r = cffi_requests.post(
        _BATCHEXECUTE_URL,
        data=freq_body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Referer": "https://news.google.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        },
        cookies=cookies,
        impersonate="chrome124",
        timeout=25,
        allow_redirects=True,
        proxies=_proxies(),
    )
    if r.status_code != 200:
        raise RuntimeError(f"batchexecute HTTP {r.status_code}")
    return r.text


# ── Playwright fallback resolver ────────────────────────────────────
async def _resolve_with_playwright(url: str, timeout: int = 20) -> str | None:
    """
    Use a real browser to load the Google News redirect page,
    click consent if needed, and follow the redirect to the publisher URL.
    Returns the final URL, or None on failure.
    """
    global _browser_pool, _playwright_instance, _PW_INITIALIZED

    if not _PW_INITIALIZED:
        logger.warning("Playwright not initialized – cannot use browser fallback")
        return None

    browser = await _browser_pool.get()
    page = None
    try:
        if not browser.is_connected():
            logger.warning("Browser disconnected – recreating")
            try:
                await browser.close()
            except Exception:
                pass
            browser = await _playwright_instance.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
            )

        page = await browser.new_page(
            locale="en-US",
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )

        # Navigate with a timeout
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        except PlaywrightTimeoutError:
            logger.warning(f"Playwright goto timeout for {url}")
            return None

        # Try to click "Accept all" consent button (common on Google News)
        consent_selectors = [
            'text="Zaakceptuj wszystko"',
            'text="Akceptuj wszystko"',
            'button:has-text("Zaakceptuj wszystko")',
            'text="Accept all"',
            'button:has-text("Accept all")',
            'button:has-text("Accept All")',
        ]
        clicked = False
        for selector in consent_selectors:
            try:
                btn = await page.wait_for_selector(selector, timeout=500)
                if btn:
                    await asyncio.gather(
                        page.wait_for_load_state("domcontentloaded", timeout=5000),
                        btn.click(timeout=1000),
                    )
                    clicked = True
                    logger.info(f"Consent clicked with selector: {selector}")
                    break
            except (PlaywrightTimeoutError, Exception):
                continue

        # Wait for redirection away from google.com
        start = time.time()
        while time.time() - start < 8:  # wait up to 8 seconds for redirect
            current_url = page.url
            if "google.com" not in current_url:
                final_url = current_url
                await page.close()
                await _browser_pool.put(browser)
                return final_url
            await page.wait_for_timeout(300)

        # If still on google.com, try to extract from page source as fallback
        html = await page.content()
        # Look for data-n-a-uc or data-n-au attributes
        m = re.search(r'data-n-a-uc="([^"]+)"', html) or re.search(r'data-n-au="([^"]+)"', html)
        if m:
            extracted = m.group(1)
            if extracted.startswith("http"):
                await page.close()
                await _browser_pool.put(browser)
                return extracted

        # If nothing found, return the current URL (may still be google.com)
        final_url = page.url
        await page.close()
        await _browser_pool.put(browser)
        return final_url if "google.com" not in final_url else None

    except Exception as e:
        logger.warning(f"Playwright resolver error for {url}: {e}")
        if page:
            try:
                await page.close()
            except Exception:
                pass
        # Put browser back even on error
        try:
            await _browser_pool.put(browser)
        except Exception:
            pass
        return None


# ── Public interface ──────────────────────────────────────────────
async def resolve_google_news(url: str) -> tuple[str | None, str]:
    if not is_google_news_url(url):
        return url, "not-a-gnews-url"

    cached = _cache_get(url)
    if cached:
        return cached, "cache"

    # 1. Fast path: direct 'url=' parameter
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "url" in qs and qs["url"][0].startswith("http"):
        direct = qs["url"][0]
        _cache_put(url, direct)
        return direct, "url-param"

    path = parsed.path.rstrip("/")
    b64 = path.rsplit("/", 1)[-1]
    if not b64:
        return None, "no-article-id"

    # 2. Offline decode
    offline = _decode_offline(b64)
    if offline:
        _cache_put(url, offline)
        return offline, "b64-decode"

    # 3. batchexecute (try with sig/ts if possible)
    sig = ts = ""
    cookies = {}
    try:
        html, cookies = await asyncio.to_thread(_get_redirect_page, url)
        m_sig = re.search(r'data-n-a-sg="([^"]*)"', html)
        m_ts = re.search(r'data-n-a-ts="([^"]*)"', html)
        sig = m_sig.group(1) if m_sig else ""
        ts = m_ts.group(1) if m_ts else ""
    except Exception as e:
        logger.debug(f"Failed to fetch redirect page for batchexecute: {e}")

    global _last_call_ts
    async with _resolve_lock:
        wait = _last_call_ts + _MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_ts = time.monotonic()

        attempts = []
        if sig and ts:
            attempts.append(("batchexecute+sig", _build_freq(b64, sig, ts)))
        attempts.append(("batchexecute-legacy", _build_freq(b64)))

        for method, body in attempts:
            try:
                resp = await asyncio.to_thread(_post_batchexecute, body, cookies)
                resolved = _parse_garturlres(resp)
                if resolved:
                    _cache_put(url, resolved)
                    return resolved, method
            except Exception as e:
                logger.warning(f"batchexecute {method} failed: {e}")

    # 4. Playwright fallback (browser)
    logger.info(f"batchexecute failed, trying Playwright fallback for {url}")
    resolved = await _resolve_with_playwright(url)
    if resolved:
        _cache_put(url, resolved)
        return resolved, "playwright"
    return None, "playwright-failed"


# ── Lifecycle management (called from app.py) ──────────────────────
async def init_playwright_pool(pool_size: int = 3):
    """Initialize the Playwright browser pool. Call this on app startup."""
    global _playwright_instance, _browser_pool, _PW_INITIALIZED
    if _PW_INITIALIZED:
        return
    _playwright_instance = await async_playwright().start()
    _browser_pool = asyncio.Queue(maxsize=pool_size)
    for _ in range(pool_size):
        browser = await _playwright_instance.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        await _browser_pool.put(browser)
    _PW_INITIALIZED = True
    logger.info(f"Playwright pool of {pool_size} browsers initialized")


async def shutdown_playwright_pool():
    """Clean up the browser pool. Call on app shutdown."""
    global _playwright_instance, _browser_pool, _PW_INITIALIZED
    if not _PW_INITIALIZED:
        return
    if _browser_pool:
        while not _browser_pool.empty():
            try:
                browser = _browser_pool.get_nowait()
                await browser.close()
            except Exception:
                pass
    if _playwright_instance:
        await _playwright_instance.stop()
    _PW_INITIALIZED = False
    logger.info("Playwright pool shut down")
