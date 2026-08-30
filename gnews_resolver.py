"""Google News redirect resolver.

Resolution order:
1. direct ``url=`` query parameter
2. offline base64/protobuf decoding (older IDs)
3. Google HTML payload inspection (data-n-a-uc / data-n-au + external links)
4. Playwright Chromium navigation (real browser redirect / consent handling)
5. batchexecute Fbv4je RPC (legacy/current fallback)

The resolver is intentionally isolated so the article extractor can call it
without knowing which strategy succeeded.
"""
import asyncio
import base64
import json
import logging
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, quote, unquote, urlparse

from curl_cffi import requests as cffi_requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger("gnews-resolver")

_BATCHEXECUTE_URL = "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je"
_CACHE_MAX = 2000
_CACHE = OrderedDict()
_LOCK = asyncio.Lock()
_LAST_CALL_TS = 0.0
_MIN_INTERVAL_S = 0.75

BROWSER_POOL_SIZE = max(1, int(os.getenv("GNEWS_BROWSER_POOL_SIZE", "3")))
PLAYWRIGHT_TIMEOUT_MS = max(3000, int(os.getenv("GNEWS_PLAYWRIGHT_TIMEOUT_MS", "12000")))
PLAYWRIGHT_REDIRECT_WAIT_MS = max(500, int(os.getenv("GNEWS_REDIRECT_WAIT_MS", "5000")))

_playwright = None
_browser_pool = None
_browser_start_lock = asyncio.Lock()


def is_google_news_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
        path = (urlparse(url).path or "").lower()
        return host == "news.google.com" and (
            path.startswith("/rss/articles/")
            or path.startswith("/articles/")
            or path.startswith("/read/")
        )
    except Exception:
        return False


def _cache_get(url: str):
    value = _CACHE.get(url)
    if value:
        _CACHE.move_to_end(url)
    return value


def _cache_put(url: str, resolved: str):
    if not resolved or resolved == url or "news.google.com" in resolved:
        return
    _CACHE[url] = resolved
    _CACHE.move_to_end(url)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def _proxy_dict():
    proxy = os.getenv("GNEWS_PROXY_URL", "").strip()
    return {"http": proxy, "https": proxy} if proxy else None


def _is_external(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
        return bool(host) and not (
            host == "google.com"
            or host.endswith(".google.com")
            or host.endswith("gstatic.com")
            or host.endswith("googleapis.com")
            or host == "w3.org"
            or host.endswith(".w3.org")
            or host == "schema.org"
            or host.endswith(".schema.org")
        )
    except Exception:
        return False


def _clean_candidate(value: str) -> str | None:
    if not value:
        return None
    value = unquote(value).replace("&amp;", "&").strip().strip('\\"\'<>')
    if value.startswith("\\u003d"):
        value = value[6:]
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return None


# ---------- Strategy 1: offline protobuf/base64 ----------
def _decode_offline(b64: str) -> str | None:
    try:
        raw = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4))
    except Exception:
        return None
    candidates = [raw]
    if raw.startswith(b"\x08\x13\x22"):
        candidates.append(raw[3:])
    for data in candidates:
        if data.endswith(b"\xd2\x01\x00"):
            data = data[:-3]
        if not data:
            continue
        for offset in (0, 1, 2):
            if len(data) <= offset:
                continue
            tail = data[offset:]
            # Try protobuf length-delimited field at the beginning.
            if tail and tail[0] < 0x80:
                end = 1 + tail[0]
                if end <= len(tail):
                    out = _clean_candidate(tail[1:end].decode("utf-8", errors="ignore"))
                    if out and _is_external(out):
                        return out
            text = tail.decode("utf-8", errors="ignore")
            m = re.search(r"https?://[^\x00\"'<>\s]+", text)
            if m:
                out = _clean_candidate(m.group(0))
                if out and _is_external(out):
                    return out
    return None


# ---------- Strategy 2: HTML payload inspection ----------
def _get_redirect_page(url: str):
    return cffi_requests.get(
        url,
        impersonate="chrome124",
        timeout=15,
        allow_redirects=True,
        proxies=_proxy_dict(),
        headers={
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        },
    )


def _extract_html_target(html: str) -> str | None:
    # Mirrors the useful fallback from the supplied JS resolver.
    for pattern in (
        r'data-n-a-uc="([^"]+)"',
        r'data-n-au="([^"]+)"',
        r"data-n-a-uc='([^']+)'",
        r"data-n-au='([^']+)'",
    ):
        m = re.search(pattern, html, re.I)
        if m:
            target = _clean_candidate(m.group(1))
            if target and _is_external(target):
                return target

    # JSON/escaped variants sometimes appear in Google's HTML.
    for pattern in (
        r'data-n-a-uc\\?[:=]\\?["\'](https?://[^"\'\\]+)',
        r'data-n-au\\?[:=]\\?["\'](https?://[^"\'\\]+)',
    ):
        m = re.search(pattern, html, re.I)
        if m:
            target = _clean_candidate(m.group(1))
            if target and _is_external(target):
                return target

    # Last HTML fallback: first external absolute URL.
    for candidate in re.findall(r"https?://[^\s\"'<>\{\}\[\]`]+", html, re.I):
        target = _clean_candidate(candidate)
        if target and _is_external(target):
            return target
    return None


async def _html_resolve(url: str):
    try:
        response = await asyncio.to_thread(_get_redirect_page, url)
        final = str(getattr(response, "url", "") or "")
        if final and _is_external(final):
            return final, "html-redirect"
        target = _extract_html_target(response.text)
        if target:
            return target, "html-payload"
    except Exception as exc:
        logger.debug("HTML resolver failed: %s", exc)
    return None, "html-failed"


# ---------- Strategy 3: Playwright ----------
async def _create_browser():
    return await _playwright.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
    )


async def _ensure_browser_pool():
    global _playwright, _browser_pool
    if _browser_pool is not None:
        return
    async with _browser_start_lock:
        if _browser_pool is not None:
            return
        _playwright = await async_playwright().start()
        _browser_pool = asyncio.Queue(maxsize=BROWSER_POOL_SIZE)
        for _ in range(BROWSER_POOL_SIZE):
            await _browser_pool.put(await _create_browser())
        logger.info("Google News Playwright pool ready: %s browsers", BROWSER_POOL_SIZE)


@asynccontextmanager
async def _get_browser():
    await _ensure_browser_pool()
    browser = await _browser_pool.get()
    try:
        if not browser.is_connected():
            try:
                await browser.close()
            except Exception:
                pass
            browser = await _create_browser()
        yield browser
    finally:
        try:
            if browser.is_connected():
                await _browser_pool.put(browser)
            else:
                await _browser_pool.put(await _create_browser())
        except Exception:
            pass


async def _playwright_resolve(url: str):
    try:
        async with _get_browser() as browser:
            page = await browser.new_page(
                locale="en-US",
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT_MS)

                selectors = [
                    'button:has-text("Accept all")',
                    'button:has-text("I agree")',
                    'text="Accept all"',
                    'text="Zaakceptuj wszystko"',
                    'text="Akceptuj wszystko"',
                ]
                for selector in selectors:
                    try:
                        btn = await page.wait_for_selector(selector, timeout=400)
                        if btn:
                            await btn.click(timeout=1000)
                            await page.wait_for_timeout(300)
                            break
                    except (PlaywrightTimeoutError, Exception):
                        continue

                started = time.monotonic()
                while time.monotonic() - started < PLAYWRIGHT_REDIRECT_WAIT_MS / 1000:
                    current = page.url
                    if _is_external(current):
                        return current, "playwright"
                    await page.wait_for_timeout(250)

                current = page.url
                if _is_external(current):
                    return current, "playwright"

                # Inspect page HTML after JS has run.
                try:
                    target = _extract_html_target(await page.content())
                    if target:
                        return target, "playwright-html"
                except Exception:
                    pass
            finally:
                await page.close()
    except Exception as exc:
        logger.warning("Playwright Google News resolver failed: %s", exc)
    return None, "playwright-failed"


# ---------- Strategy 4: batchexecute ----------
def _build_freq(gn_art_id: str, sig: str = "", ts: str = "") -> str:
    inner_req = [
        "garturlreq",
        [[
            [["en-US", "US", ["FINANCE_TOP_INDICES", "WEB_TEST_1_0_0"], None,
              None, 1, 1, "US:en", None, 180, None, None, None, None, 0,
              None, None, [1608992183, 723341000]],
             "en-US", "US", 1, [2, 3, 4, 8], 1, 0, "655000234", 0, 0,
             None, 0]
        ]],
        gn_art_id,
    ]
    if sig and ts:
        inner_req += [ts, sig, False]
    freq = json.dumps([[['Fbv4je', json.dumps(inner_req), None, 'generic']]])
    return "f.req=" + quote(freq)


def _parse_garturlres(text: str) -> str | None:
    try:
        payload = text[text.index("["):]
        data = json.loads(payload)
        for row in data:
            try:
                inner = json.loads(row[2])
                if (isinstance(inner, list) and len(inner) > 1
                        and inner[0] == "garturlres"
                        and isinstance(inner[1], str)):
                    target = _clean_candidate(inner[1])
                    if target and _is_external(target):
                        return target
            except (json.JSONDecodeError, TypeError, IndexError, ValueError):
                continue
    except (ValueError, json.JSONDecodeError, TypeError):
        pass
    m = re.search(r'garturlres\\",\\"(https?://[^\\",]+)', text)
    if m:
        target = _clean_candidate(m.group(1))
        return target if target and _is_external(target) else None
    return None


def _is_retryable(exc):
    msg = str(exc).lower()
    return any(x in msg for x in ("timeout", "429", "rate", "too many", "500", "502", "503"))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _post_batchexecute(body: str, cookies: dict) -> str:
    r = cffi_requests.post(
        _BATCHEXECUTE_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Referer": "https://news.google.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        },
        cookies=cookies,
        impersonate="chrome124",
        timeout=15,
        allow_redirects=True,
        proxies=_proxy_dict(),
    )
    if r.status_code != 200:
        raise RuntimeError(f"batchexecute HTTP {r.status_code}")
    return r.text


async def _batchexecute_resolve(url: str, b64: str):
    global _LAST_CALL_TS
    sig = ts = ""
    cookies = {}
    try:
        response = await asyncio.to_thread(_get_redirect_page, url)
        html = response.text
        cookies = response.cookies
        m_sig = re.search(r'data-n-a-sg="([^"]*)"', html)
        m_ts = re.search(r'data-n-a-ts="([^"]*)"', html)
        sig = m_sig.group(1) if m_sig else ""
        ts = m_ts.group(1) if m_ts else ""
    except Exception:
        pass

    async with _LOCK:
        delay = _LAST_CALL_TS + _MIN_INTERVAL_S - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        _LAST_CALL_TS = time.monotonic()
        attempts = []
        if sig and ts:
            attempts.append(("batchexecute+sig", _build_freq(b64, sig, ts)))
        attempts.append(("batchexecute-legacy", _build_freq(b64)))
        last_err = "batchexecute-failed"
        for method, body in attempts:
            try:
                text = await asyncio.to_thread(_post_batchexecute, body, cookies)
                target = _parse_garturlres(text)
                if target:
                    return target, method
                last_err = f"{method}:no-garturlres"
            except Exception as exc:
                last_err = f"{method}:{type(exc).__name__}"
        return None, last_err


async def resolve_google_news(url: str) -> tuple[str | None, str]:
    """Return ``(publisher_url, method)``. Non-Google-News URLs pass through."""
    if not is_google_news_url(url):
        return url, "not-a-gnews-url"

    cached = _cache_get(url)
    if cached:
        return cached, "cache"

    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if qs.get("url"):
        direct = _clean_candidate(qs["url"][0])
        if direct and _is_external(direct):
            _cache_put(url, direct)
            return direct, "url-param"

    b64 = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if not b64:
        return None, "no-article-id"

    offline = _decode_offline(b64)
    if offline:
        _cache_put(url, offline)
        return offline, "b64-decode"

    # The supplied JS fallback is cheap, so use it before launching Chromium.
    target, method = await _html_resolve(url)
    if target:
        _cache_put(url, target)
        return target, method

    # Browser fallback for redirects that require JavaScript/cookies/consent.
    target, method = await _playwright_resolve(url)
    if target:
        _cache_put(url, target)
        return target, method

    # Final RPC fallback.
    target, method = await _batchexecute_resolve(url, b64)
    if target:
        _cache_put(url, target)
        return target, method

    return None, method


async def shutdown_resolver():
    """Close Playwright resources during application shutdown."""
    global _playwright, _browser_pool
    if _browser_pool is not None:
        while not _browser_pool.empty():
            try:
                browser = _browser_pool.get_nowait()
                await browser.close()
            except Exception:
                pass
    _browser_pool = None
    if _playwright is not None:
        try:
            await _playwright.stop()
        except Exception:
            pass
    _playwright = None
