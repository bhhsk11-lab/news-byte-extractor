"""
Robust Google News article URL resolver.

The important rule is: NEVER treat a Google-hosted asset such as
lh3.googleusercontent.com, gstatic.com, googleapis.com, etc. as the
publisher URL.

Resolution strategy:
  1. Explicit ?url= parameter when present.
  2. Legacy protobuf/base64 URLs that still contain the source URL.
  3. Current Google News page parameters:
       - c-wiz[data-p] payload
       - data-n-a-id / data-n-a-sg / data-n-a-ts
       - equivalent attributes on c-wiz/div
  4. Fbv4je / garturlreq batchexecute RPC using Google's current payload.
  5. Browser navigation as a last resort.
  6. A second batchexecute attempt using the legacy parameter shape.

This module is intentionally conservative: a URL is returned only when it
passes publisher validation. Google CDN URLs are never accepted as articles.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from html import unescape
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from curl_cffi import requests as cffi_requests

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
except Exception:
    async_playwright = None
    PlaywrightTimeoutError = Exception


log = logging.getLogger("gnews-resolver")

GOOGLE_NEWS_HOSTS = {
    "news.google.com",
    "news.googleusercontent.com",
}

GOOGLE_ASSET_SUFFIXES = (
    "gstatic.com",
    "googleapis.com",
    "googleusercontent.com",
    "googlevideo.com",
    "ggpht.com",
)

BATCHEXECUTE_URL = (
    "https://news.google.com/_/DotsSplashUi/data/batchexecute"
    "?rpcids=Fbv4je"
)

CACHE_MAX = max(100, int(os.getenv("GNEWS_CACHE_MAX", "4000")))
CACHE_TTL = max(60, int(os.getenv("GNEWS_CACHE_TTL", "86400")))
MIN_RPC_INTERVAL = max(0.2, float(os.getenv("GNEWS_MIN_RPC_INTERVAL", "0.8")))

HTTP_TIMEOUT = max(5, float(os.getenv("GNEWS_HTTP_TIMEOUT", "15")))
BROWSER_TIMEOUT_MS = max(3000, int(os.getenv("GNEWS_PLAYWRIGHT_TIMEOUT_MS", "12000")))
BROWSER_WAIT_MS = max(500, int(os.getenv("GNEWS_REDIRECT_WAIT_MS", "4500")))
BROWSER_POOL_SIZE = max(1, int(os.getenv("GNEWS_BROWSER_POOL_SIZE", "2")))

USER_AGENT = os.getenv(
    "GNEWS_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Safari/537.36",
)

_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
_rpc_lock = asyncio.Lock()
_last_rpc = 0.0

_playwright = None
_browser_pool = None
_browser_lock = asyncio.Lock()


def is_google_news_url(url: str) -> bool:
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        path = (p.path or "").lower()
        return host == "news.google.com" and (
            path.startswith("/rss/articles/")
            or path.startswith("/articles/")
            or path.startswith("/read/")
            or path.startswith("/__i/rss/rd/articles/")
        )
    except Exception:
        return False


def _cache_get(key: str) -> str | None:
    item = _cache.get(key)
    if not item:
        return None
    ts, value = item
    if time.monotonic() - ts > CACHE_TTL:
        _cache.pop(key, None)
        return None
    _cache.move_to_end(key)
    return value


def _cache_put(key: str, value: str) -> None:
    if not value or not _is_publisher_url(value):
        return
    _cache[key] = (time.monotonic(), value)
    _cache.move_to_end(key)
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)


def _proxy() -> dict[str, str] | None:
    value = os.getenv("GNEWS_PROXY_URL", "").strip()
    return {"http": value, "https": value} if value else None


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().rstrip(".")
    except Exception:
        return ""


def _is_google_host(host: str) -> bool:
    if not host:
        return True
    return (
        host == "google.com"
        or host.endswith(".google.com")
        or host.endswith(".google.co.uk")
        or host.endswith(".google.co.in")
        or any(host.endswith("." + suffix) or host == suffix
               for suffix in GOOGLE_ASSET_SUFFIXES)
    )


def _is_publisher_url(url: str) -> bool:
    """Strict target validation. Google CDN/assets can never be a target."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            return False

        host = p.hostname.lower().rstrip(".")
        if _is_google_host(host):
            return False

        if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
            return False

        # Reject obvious non-page assets.
        path = (p.path or "").lower()
        if path.endswith((
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
            ".ico", ".css", ".js", ".woff", ".woff2", ".mp4", ".webm",
        )):
            return False

        return True
    except Exception:
        return False


def _clean_url(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")

    value = unescape(str(value))
    value = value.replace("\\/", "/")
    value = value.replace("\\u0026", "&")
    value = value.replace("\\u003d", "=")
    value = value.replace("\\u002F", "/")
    value = unquote(value)
    value = value.strip().strip("\"'<>[](),")

    # Some Google payloads contain JSON-escaped URLs.
    if "\\x3a" in value:
        value = value.replace("\\x3a", ":")
    if "\\x2f" in value:
        value = value.replace("\\x2f", "/")

    if value.startswith(("http://", "https://")) and _is_publisher_url(value):
        return value
    return None


def _article_id(url: str) -> str | None:
    try:
        path = urlparse(url).path.rstrip("/")
        marker = "/articles/"
        if marker in path:
            return path.split(marker, 1)[1].split("/", 1)[0]
        marker = "/rd/articles/"
        if marker in path:
            return path.split(marker, 1)[1].split("/", 1)[0]
        marker = "/rss/articles/"
        if marker in path:
            return path.split(marker, 1)[1].split("/", 1)[0]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Legacy URL format
# ---------------------------------------------------------------------------

def _decode_legacy_base64(article_id: str) -> str | None:
    try:
        raw = base64.urlsafe_b64decode(article_id + "=" * (-len(article_id) % 4))
    except Exception:
        return None

    # Old format commonly starts with protobuf bytes 08 13 22 and ends d2 01 00.
    blobs = [raw]
    if raw.startswith(b"\x08\x13\x22"):
        blobs.append(raw[3:])

    for blob in blobs:
        if blob.endswith(b"\xd2\x01\x00"):
            blob = blob[:-3]

        # Scan for a real URL rather than assuming a fixed protobuf offset.
        text = blob.decode("utf-8", "ignore")
        for match in re.finditer(r"https?://[^\x00\s\"'<>]+", text):
            target = _clean_url(match.group(0))
            if target:
                return target

        # Also try length-delimited protobuf strings near the beginning.
        for offset in range(min(8, len(blob))):
            if offset >= len(blob):
                break
            first = blob[offset]
            if first < 0x80:
                end = offset + 1 + first
                if end <= len(blob):
                    target = _clean_url(
                        blob[offset + 1:end].decode("utf-8", "ignore")
                    )
                    if target:
                        return target
    return None


# ---------------------------------------------------------------------------
# HTTP session / Google page
# ---------------------------------------------------------------------------

def _google_get(url: str):
    return cffi_requests.get(
        url,
        impersonate="chrome124",
        timeout=HTTP_TIMEOUT,
        allow_redirects=True,
        proxies=_proxy(),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        },
    )


def _get_page(url: str):
    """Try /articles first because current Google implementations expose
    decoding metadata there more reliably; fall back to the original URL."""
    aid = _article_id(url)
    urls = []
    if aid:
        urls.append(f"https://news.google.com/articles/{aid}")
        urls.append(f"https://news.google.com/rss/articles/{aid}")
    urls.append(url)

    seen = set()
    last = None

    for candidate in urls:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            response = _google_get(candidate)
            last = response
            if response.status_code == 200 and response.text:
                return response
        except Exception as exc:
            log.debug("Google GET failed for %s: %s", candidate, exc)

    return last


def _extract_attr(html: str, name: str) -> str | None:
    patterns = [
        rf'\b{name}\s*=\s*"([^"]+)"',
        rf"\b{name}\s*=\s*'([^']+)'",
        rf'\b{name}\s*=\s*\\?"([^"\\]+)\\?"',
    ]
    for pattern in patterns:
        m = re.search(pattern, html, re.I)
        if m:
            return unescape(m.group(1))
    return None


def _extract_current_params(html: str, article_id: str) -> dict[str, Any]:
    """
    Current Google pages have used two useful representations:
      c-wiz[data-p] with a serialized garturlreq structure
      c-wiz/div data-n-a-id + data-n-a-ts + data-n-a-sg
    """
    result: dict[str, Any] = {
        "id": article_id,
        "timestamp": None,
        "signature": None,
        "data_p": None,
    }

    # First: explicit attributes.
    for key, attr in (
        ("timestamp", "data-n-a-ts"),
        ("signature", "data-n-a-sg"),
        ("id", "data-n-a-id"),
    ):
        value = _extract_attr(html, attr)
        if value:
            result[key] = value

    # Second: data-p. This is especially important for the newer format.
    for pattern in (
        r'<c-wiz\b[^>]*\bdata-p\s*=\s*"([^"]+)"',
        r"<c-wiz\b[^>]*\bdata-p\s*=\s*'([^']+)'",
        r'\bdata-p\s*=\s*"([^"]+)"',
        r"\bdata-p\s*=\s*'([^']+)'",
    ):
        m = re.search(pattern, html, re.I | re.S)
        if m:
            result["data_p"] = unescape(m.group(1))
            break

    return result


def _decode_data_p(data_p: str) -> list[Any] | None:
    if not data_p:
        return None

    candidates = [
        data_p,
        data_p.replace("%.@.", '["garturlreq",'),
    ]

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    # Sometimes the attribute itself is escaped.
    try:
        candidate = bytes(data_p, "utf-8").decode("unicode_escape")
        obj = json.loads(candidate)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    return None


def _build_rpc_from_data_p(data_p: str) -> str | None:
    """
    Current/observed format:
      c-wiz[data-p] contains a serialized garturlreq request.

    The public reverse-engineered clients send that request to Fbv4je after
    removing the browser-only tail fields.
    """
    obj = _decode_data_p(data_p)
    if not obj:
        return None

    try:
        # The data-p object itself normally begins with garturlreq.
        if obj[0] == "garturlreq":
            # Keep the request and remove browser-only fields at the end.
            # This mirrors the current working implementations while avoiding
            # hard-coding a locale-specific request.
            if len(obj) >= 3:
                core = obj[:-6] + obj[-2:]
            else:
                core = obj

            inner = json.dumps(core, separators=(",", ":"))
            outer = [[[
                "Fbv4je",
                inner,
                None,
                "generic",
            ]]]

            return "f.req=" + quote(
                json.dumps(outer, separators=(",", ":")),
                safe="",
            )
    except Exception:
        log.exception("Failed to build RPC from data-p")

    return None


def _build_rpc_legacy(article_id: str, timestamp: str, signature: str) -> str:
    """
    Legacy/current fallback shape used by several working Google News
    decoders. Locale is intentionally fixed to en-US/US for deterministic
    server-side resolution.
    """
    inner = [
        "garturlreq",
        [[
            ["X", "X", ["FINANCE_TOP_INDICES", "WEB_TEST_1_0_0"],
             None, None, 1, 1, "US:en", None, 1, None, None, None,
             None, None, 0, 1],
            "en-US", "US", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0
        ]],
        article_id,
        timestamp,
        signature,
    ]

    outer = [[[
        "Fbv4je",
        json.dumps(inner, separators=(",", ":")),
        None,
        "generic",
    ]]]

    return "f.req=" + quote(
        json.dumps(outer, separators=(",", ":")),
        safe="",
    )


def _build_rpc_legacy_unsigned(article_id: str) -> str:
    inner = [
        "garturlreq",
        [[
            [["en-US", "US", ["FINANCE_TOP_INDICES", "WEB_TEST_1_0_0"],
              None, None, 1, 1, "US:en", None, 1, None, None,
              None, None, None, 0, 1],
             "en-US", "US", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0]
        ]],
        article_id,
    ]

    outer = [[[
        "Fbv4je",
        json.dumps(inner, separators=(",", ":")),
        None,
        "generic",
    ]]]

    return "f.req=" + quote(
        json.dumps(outer, separators=(",", ":")),
        safe="",
    )


def _post_rpc(body: str, cookies: dict[str, str] | None = None) -> str:
    response = cffi_requests.post(
        BATCHEXECUTE_URL,
        data=body,
        cookies=cookies or {},
        impersonate="chrome124",
        timeout=HTTP_TIMEOUT,
        allow_redirects=True,
        proxies=_proxy(),
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Referer": "https://news.google.com/",
            "Origin": "https://news.google.com",
            "Accept": "*/*",
        },
    )
    if response.status_code != 200:
        raise RuntimeError(f"batchexecute HTTP {response.status_code}")
    return response.text


def _extract_url_deep(obj: Any) -> str | None:
    """Recursively locate garturlres and validate its URL."""
    if isinstance(obj, dict):
        for value in obj.values():
            found = _extract_url_deep(value)
            if found:
                return found
        return None

    if isinstance(obj, list):
        # garturlres is generally [name, url, ...]
        if len(obj) >= 2 and obj[0] == "garturlres":
            target = _clean_url(obj[1])
            if target:
                return target

        for item in obj:
            found = _extract_url_deep(item)
            if found:
                return found

    return None


def _parse_rpc_response(text: str) -> str | None:
    """
    Google wraps batchexecute responses in one or more JSON-ish frames.
    Do not depend on one exact line number or '\\n\\n' position.
    """
    # Fast path: escaped garturlres.
    for pattern in (
        r'\[\\"garturlres\\",\\"(https?://[^"\\]+)',
        r'garturlres\\\\?["\']\s*,\s*\\\\?["\'](https?://[^"\\]+)',
    ):
        for m in re.finditer(pattern, text, re.I):
            target = _clean_url(m.group(1))
            if target:
                return target

    # Decode every plausible JSON fragment beginning with '['.
    starts = [m.start() for m in re.finditer(r"\[", text)]
    for start in starts[:80]:
        fragment = text[start:].strip()
        try:
            obj = json.loads(fragment)
        except Exception:
            continue
        target = _extract_url_deep(obj)
        if target:
            return target

    # Last structured fallback: extract URLs, but ONLY accept non-Google
    # publisher URLs. This prevents the lh3.googleusercontent.com bug.
    for candidate in re.findall(r"https?://[^\s\"'\\<>\]\[(),]+", text):
        target = _clean_url(candidate)
        if target:
            return target

    return None


# ---------------------------------------------------------------------------
# HTML strategy
# ---------------------------------------------------------------------------

def _html_target_candidates(html: str):
    """
    Extract only plausible publisher candidates. In particular, never use
    'first external URL' because Google's page contains lh3/gstatic URLs.
    """
    # Known Google redirect attributes.
    for attr in ("data-n-a-uc", "data-n-au"):
        for pattern in (
            rf'\b{re.escape(attr)}\s*=\s*"([^"]+)"',
            rf"\b{re.escape(attr)}\s*=\s*'([^']+)'",
        ):
            for m in re.finditer(pattern, html, re.I):
                target = _clean_url(m.group(1))
                if target:
                    yield target

    # Search for publisher-looking URLs near article metadata. This is only
    # a fallback; arbitrary first external URLs are deliberately not used.
    for pattern in (
        r'"(?:url|target|destination|articleUrl|sourceUrl)"\s*:\s*"([^"]+)"',
        r"'(?:url|target|destination|articleUrl|sourceUrl)'\s*:\s*'([^']+)'",
    ):
        for m in re.finditer(pattern, html, re.I):
            target = _clean_url(m.group(1))
            if target:
                yield target


async def _resolve_html(url: str):
    try:
        response = await asyncio.to_thread(_get_page, url)
        if not response:
            return None, "html-no-response", None

        html = response.text
        params = _extract_current_params(html, _article_id(url) or "")

        # A real HTTP redirect is useful only if it actually leaves Google.
        final = str(getattr(response, "url", "") or "")
        if _is_publisher_url(final):
            return final, "http-redirect", params

        for target in _html_target_candidates(html):
            return target, "html-target", params

        return None, "html-no-target", params
    except Exception as exc:
        log.debug("HTML resolution failed: %s", exc)
        return None, f"html-error:{type(exc).__name__}", None


# ---------------------------------------------------------------------------
# RPC strategy
# ---------------------------------------------------------------------------

async def _resolve_rpc(url: str, params: dict[str, Any] | None):
    global _last_rpc

    if not params:
        return None, "rpc-no-params"

    article_id = params.get("id") or _article_id(url)
    if not article_id:
        return None, "rpc-no-article-id"

    bodies: list[tuple[str, str]] = []

    # Most important: Google's current data-p payload.
    data_p = params.get("data_p")
    if data_p:
        body = _build_rpc_from_data_p(data_p)
        if body:
            bodies.append(("batchexecute-data-p", body))

    # Next: current data-n-a-id + signature + timestamp.
    ts = params.get("timestamp")
    sig = params.get("signature")
    if ts and sig:
        bodies.append((
            "batchexecute-signature",
            _build_rpc_legacy(article_id, str(ts), str(sig)),
        ))

    # Final unsigned shape used by older links.
    bodies.append(("batchexecute-unsigned", _build_rpc_legacy_unsigned(article_id)))

    async with _rpc_lock:
        delay = _last_rpc + MIN_RPC_INTERVAL - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        _last_rpc = time.monotonic()

        last_error = "rpc-failed"

        for method, body in bodies:
            try:
                # Re-fetch cookies immediately before RPC when possible.
                page_response = await asyncio.to_thread(_get_page, url)
                cookies = {}
                if page_response is not None:
                    try:
                        cookies = dict(page_response.cookies)
                    except Exception:
                        cookies = {}

                text = await asyncio.to_thread(_post_rpc, body, cookies)
                target = _parse_rpc_response(text)
                if target:
                    return target, method
                last_error = method + ":no-garturlres"
            except Exception as exc:
                last_error = f"{method}:{type(exc).__name__}"
                log.debug("RPC attempt failed: %s", exc)

        return None, last_error


# ---------------------------------------------------------------------------
# Browser fallback
# ---------------------------------------------------------------------------

async def _create_browser():
    if async_playwright is None:
        raise RuntimeError("Playwright is not installed")
    return await _playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-blink-features=AutomationControlled",
        ],
    )


async def _ensure_browser_pool():
    global _playwright, _browser_pool

    if _browser_pool is not None:
        return

    async with _browser_lock:
        if _browser_pool is not None:
            return

        if async_playwright is None:
            raise RuntimeError("Playwright is not installed")

        _playwright = await async_playwright().start()
        _browser_pool = asyncio.Queue(maxsize=BROWSER_POOL_SIZE)

        for _ in range(BROWSER_POOL_SIZE):
            await _browser_pool.put(await _create_browser())


@asynccontextmanager
async def _browser():
    await _ensure_browser_pool()
    browser = await _browser_pool.get()
    try:
        if not browser.is_connected():
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


async def _resolve_browser(url: str):
    try:
        async with _browser() as browser:
            context = await browser.new_context(
                locale="en-US",
                viewport={"width": 1280, "height": 800},
                user_agent=USER_AGENT,
                java_script_enabled=True,
            )
            page = await context.new_page()

            try:
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=BROWSER_TIMEOUT_MS,
                )

                started = time.monotonic()

                while (time.monotonic() - started) < (BROWSER_WAIT_MS / 1000):
                    current = page.url
                    if _is_publisher_url(current):
                        return current, "playwright"

                    # Google's redirect page can contain the destination in
                    # dynamically-created attributes.
                    try:
                        html = await page.content()
                        for target in _html_target_candidates(html):
                            return target, "playwright-html"
                    except Exception:
                        pass

                    await page.wait_for_timeout(200)

                current = page.url
                if _is_publisher_url(current):
                    return current, "playwright-final"

                return None, "playwright-no-target"

            finally:
                await context.close()

    except Exception as exc:
        log.debug("Playwright resolution failed: %s", exc)
        return None, f"playwright-error:{type(exc).__name__}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def resolve_google_news(url: str) -> tuple[str | None, str]:
    """
    Resolve a Google News URL.

    Returns:
        (publisher_url, method)

    For a non-Google URL, returns (url, "not-a-gnews-url").
    For an unresolved Google URL, returns (None, failure_method).
    """
    if not is_google_news_url(url):
        return url, "not-a-gnews-url"

    cached = _cache_get(url)
    if cached:
        return cached, "cache"

    # Explicit URL parameter.
    try:
        qs = parse_qs(urlparse(url).query)
        for value in qs.get("url", []):
            target = _clean_url(value)
            if target:
                _cache_put(url, target)
                return target, "url-param"
    except Exception:
        pass

    article_id = _article_id(url)
    if not article_id:
        return None, "no-article-id"

    # Old-format links can still be decoded locally.
    target = _decode_legacy_base64(article_id)
    if target:
        _cache_put(url, target)
        return target, "legacy-base64"

    # Current Google page + current RPC.
    _, html_method, params = await _resolve_html(url)

    target, method = await _resolve_rpc(url, params)
    if target:
        _cache_put(url, target)
        return target, method

    # Browser is deliberately after RPC. It is slower and consumes more RAM.
    target, method = await _resolve_browser(url)
    if target:
        _cache_put(url, target)
        return target, method

    return None, method or html_method or "not-resolved"


async def shutdown_resolver():
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


__all__ = [
    "is_google_news_url",
    "resolve_google_news",
    "shutdown_resolver",
]
