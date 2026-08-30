"""
Google News redirect-URL resolver.

news.google.com/rss/articles/CBMi...?oc=5 URLs are protobuf-encoded
click-tracking redirects, NOT real article URLs. Resolution strategy:

  1. Direct 'url=' query parameter (some redirects include it).
  2. Offline base64/protobuf decode (old‑style IDs embed the URL).
  3. batchexecute RPC (Fbv4je / garturlreq) with signature+timestamp
     scraped from the redirect page — required for new‑style AU_yqL
     IDs (July 2024+).
  4. batchexecute without sig/ts (legacy fallback).

Google rate‑limits batchexecute hard from datacenter IPs (429s), so all
calls route through PROXY_URL when configured, results are LRU‑cached,
and calls are serialized with a minimum interval. Retries are implemented
for transient errors.
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

from config import settings

logger = logging.getLogger("gnews-resolver")

_BATCHEXECUTE_URL = "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je"

# ── LRU cache for resolved URLs ──────────────────────────────────────
_RESOLVE_CACHE_MAX = 2000
_resolve_cache: OrderedDict = OrderedDict()

# Serialize batchexecute calls + politeness delay (429 hotspot)
_resolve_lock = asyncio.Lock()
_last_call_ts = 0.0
_MIN_INTERVAL_S = 0.75


def is_google_news_url(url: str) -> bool:
    """Check if the URL is a Google News redirect (rss/articles/...)."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return False
    # Some URLs may have host as news.google.com or subdomains
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


# ── Strategy 1: offline protobuf base64 decode (old‑style IDs) ──────
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


# ── batchexecute plumbing ────────────────────────────────────────────
def _build_freq(gn_art_id: str, sig: str = "", ts: str = "") -> str:
    """
    Build the URL‑encoded f.req body for the Fbv4je 'garturlreq' RPC.

    Structure is based on traffic captured from news.google.com in Chrome.
    The nested list layout is critical for the server to accept the request.
    """
    # The base request structure
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
    # The outer wrapper: [ [ [ RPC_ID, JSON_string, None, "generic" ] ] ]
    freq = json.dumps([[["Fbv4je", json.dumps(inner_req), None, "generic"]]])
    # URL‑encode the whole thing as 'f.req=...'
    return "f.req=" + quote(freq)


# We need urllib.parse.quote for the encoding
from urllib.parse import quote


def _get_redirect_page(url: str):
    """Fetch the redirect page and return (html, cookies)."""
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
    """Extract the resolved URL from a batchexecute response."""
    try:
        # The response is a JSON with a leading `)]}'` sometimes
        payload = text[text.index("["):]
        data = json.loads(payload)
        # data is typically a list of lists
        for row in data:
            try:
                # The third element is a JSON string containing the actual result
                inner = json.loads(row[2])  # row[2] is the string
                if (isinstance(inner, list) and len(inner) > 1
                        and inner[0] == "garturlres"
                        and isinstance(inner[1], str)
                        and inner[1].startswith("http")):
                    return inner[1]
            except (json.JSONDecodeError, TypeError, IndexError, ValueError):
                continue
    except (ValueError, json.JSONDecodeError):
        pass
    # Fallback regex in case parsing fails
    m = re.search(r'garturlres\\",\\"(https?://[^\\",]+)', text)
    return m.group(1) if m else None


# Retry decorator for batchexecute (transient errors)
def _is_retryable(exception):
    """Return True if we should retry the batchexecute call."""
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
    """Make the batchexecute POST request with retries."""
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


async def resolve_google_news(url: str) -> tuple[str | None, str]:
    """
    Resolve a Google News redirect URL to the original publisher URL.
    Returns (resolved_url, method_or_error). resolved_url is None on failure.
    Non‑Google‑News URLs pass through unchanged.
    """
    if not is_google_news_url(url):
        return url, "not-a-gnews-url"

    cached = _cache_get(url)
    if cached:
        return cached, "cache"

    # ── Fast path: extract direct 'url=' parameter ──────────────────
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "url" in qs and qs["url"][0].startswith("http"):
        direct = qs["url"][0]
        _cache_put(url, direct)
        return direct, "url-param"

    global _last_call_ts

    # ── Extract article ID from path ────────────────────────────────
    path = parsed.path.rstrip("/")
    b64 = path.rsplit("/", 1)[-1]
    if not b64:
        return None, "no-article-id"

    # ── Offline decode (works for older IDs) ────────────────────────
    offline = _decode_offline(b64)
    if offline:
        _cache_put(url, offline)
        return offline, "b64-decode"

    # ── Fetch redirect page to get sig/ts and cookies ──────────────
    sig = ts = ""
    cookies = {}
    try:
        html, cookies = await asyncio.to_thread(_get_redirect_page, url)
        # Extract data-n-a-sg and data-n-a-ts
        m_sig = re.search(r'data-n-a-sg="([^"]*)"', html)
        m_ts = re.search(r'data-n-a-ts="([^"]*)"', html)
        sig = m_sig.group(1) if m_sig else ""
        ts = m_ts.group(1) if m_ts else ""
        logger.debug(f"Extracted sig={sig[:10]}..., ts={ts}")
    except Exception as e:
        logger.warning(f"Failed to fetch redirect page: {e}")

    # ── Serialize calls to avoid hitting rate limits ────────────────
    async with _resolve_lock:
        wait = _last_call_ts + _MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_ts = time.monotonic()

        # Build two attempts: with sig/ts, then without
        attempts = []
        if sig and ts:
            attempts.append(("batchexecute+sig", _build_freq(b64, sig, ts)))
        attempts.append(("batchexecute-legacy", _build_freq(b64)))

        last_err = "unknown"
        for method, body in attempts:
            try:
                resp = await asyncio.to_thread(_post_batchexecute, body, cookies)
                resolved = _parse_garturlres(resp)
                if resolved:
                    _cache_put(url, resolved)
                    return resolved, method
                last_err = f"no garturlres (resp len={len(resp)})"
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:120]}"
                logger.warning(f"batchexecute {method} failed: {last_err}")
        return None, last_err