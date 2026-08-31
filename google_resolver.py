"""Resilient Google News URL resolver.

v1.7.1 change: added in-flight request coalescing to resolve() — the same
Google News article ID was observed being resolved multiple times
concurrently (this server + auth-bypass-scraper as fallback, and
overlapping feed-poll requests hitting this server alone). Concurrent
callers for the same URL now share one in-flight resolution instead of each
triggering their own redundant RPC/browser attempt.

Google News RSS links are not normal HTTP redirects. Current links are resolved
by reading data-n-a-id/data-n-a-sg/data-n-a-ts from the Google article page and
calling Google's DotsSplashUi/batchexecute RPC.  This module keeps that logic
isolated from article extraction and adds bounded retries, caching, validation,
rate control, batching, and a conservative legacy/base64 fallback.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urljoin

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("google_resolver")

GOOGLE_HOST = "news.google.com"
BATCH_ENDPOINT = "https://news.google.com/_/DotsSplashUi/data/batchexecute"

# Keep the concurrency deliberately small. Google can return 429 from the RPC
# even when every individual request works in isolation.
PAGE_CONCURRENCY = 3
BATCH_CONCURRENCY = 1
# Keep each upstream operation short enough that /extract still has time for
# the publisher request. The browser/client has a ~30s deadline, so Google
# resolution must never consume that whole budget.
RESOLVE_DEADLINE = 0  # no artificial resolver abort; browser navigation uses event-based waits
REQUEST_TIMEOUT = httpx.Timeout(8.0, connect=5.0, read=7.0, write=7.0, pool=5.0)
MAX_RETRIES = 1
CACHE_TTL = 6 * 60 * 60
CACHE_MAX = 2000
NEGATIVE_TTL = 8
RPC_MIN_INTERVAL = 0.45

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

_RPC_HEADERS = {
    "User-Agent": _BROWSER_HEADERS["User-Agent"],
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://news.google.com",
    "Referer": "https://news.google.com/",
    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
}


@dataclass
class ResolveResult:
    url: str
    method: str
    error: str | None = None


class GoogleNewsResolver:
    def __init__(self) -> None:
        self._cache: OrderedDict[str, tuple[float, ResolveResult]] = OrderedDict()
        self._page_sem = asyncio.Semaphore(PAGE_CONCURRENCY)
        self._rpc_sem = asyncio.Semaphore(BATCH_CONCURRENCY)
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._last_rpc_at = 0.0
        self._browser = None
        self._browser_context = None
        self._browser_lock = asyncio.Lock()
        self._browser_page_sem = asyncio.Semaphore(2)
        # In-flight coalescing (see auth-bypass-scraper's google_resolver.py
        # for the full rationale): the production logs show the same Google
        # News article ID being requested repeatedly in close succession —
        # both from this server and from auth-bypass-scraper independently,
        # and from overlapping feed-poll requests hitting this server alone.
        # Without this, N concurrent callers for the same URL each pay the
        # full RPC+browser resolution cost; with it, only the first pays it
        # and the rest await that one in-flight result.
        self._inflight: dict[str, asyncio.Future] = {}

    async def client(self) -> httpx.AsyncClient:
        if self._client and not self._client.is_closed:
            return self._client
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    headers=_BROWSER_HEADERS,
                    follow_redirects=True,
                    max_redirects=6,
                    timeout=REQUEST_TIMEOUT,
                    http2=True,
                )
        return self._client

    async def close(self) -> None:
        if self._browser_context is not None:
            try:
                await self._browser_context.close()
            except Exception:
                pass
            self._browser_context = None
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get_browser(self):
        """Independent Chromium used ONLY for Google News URL resolution."""
        if self._browser_context is not None:
            return self._browser_context
        async with self._browser_lock:
            if self._browser_context is not None:
                return self._browser_context
            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:
                raise RuntimeError("playwright-not-installed") from exc
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            self._browser_context = await self._browser.new_context(
                user_agent=_BROWSER_HEADERS["User-Agent"],
                viewport={"width": 1365, "height": 900},
                locale="en-US",
                timezone_id="America/New_York",
                ignore_https_errors=False,
            )
            await self._browser_context.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9",
            })
            return self._browser_context

    @classmethod
    def _browser_candidates(cls, html: str, current_url: str) -> list[str]:
        """Find likely publisher article URLs, never Google infrastructure/assets."""
        scored: dict[str, int] = {}
        soup = BeautifulSoup(html or "", "lxml")

        def add(value: str | None, score: int):
            if not value:
                return
            value = unquote(str(value)).strip()
            if value.startswith("//"):
                value = "https:" + value
            absolute = urljoin(current_url, value)
            if not cls._valid_destination(absolute):
                return
            p = urlparse(absolute)
            path = p.path.lower()
            bonus = 0
            if len(path.strip("/")) > 12:
                bonus += 8
            if any(x in path for x in ("/article", "/news/", "/story/", "/stories/", "/world/", "/business/", "/technology/")):
                bonus += 15
            # Prefer publisher-like URLs over generic homepages.
            if path in ("", "/"):
                bonus -= 30
            scored[absolute.split("#", 1)[0]] = max(scored.get(absolute.split("#", 1)[0], -999), score + bonus)

        for tag_name, attrs, field, score in (
            ("link", {"rel": "canonical"}, "href", 100),
            ("meta", {"property": "og:url"}, "content", 95),
            ("meta", {"name": "twitter:url"}, "content", 90),
        ):
            for tag in soup.find_all(tag_name, attrs=attrs):
                add(tag.get(field), score)

        for a in soup.find_all("a", href=True):
            add(a.get("href"), 55)

        for m in re.finditer(r"https?://[^\s\"'<>\\]+", html or ""):
            add(m.group(0).rstrip(".,)]}"), 45)

        return [u for u, _ in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)]

    async def _browser_resolve(self, url: str) -> ResolveResult:
        """Open the exact Google News URL in an independent Chromium context.

        No fixed navigation timeout is used. We wait on browser events and inspect
        the actual destination/canonical URL. If Google keeps an interstitial, the
        rendered DOM is searched for the external publisher URL.
        """
        context = await self._get_browser()
        async with self._browser_page_sem:
            page = await context.new_page()
            try:
                page.set_default_navigation_timeout(0)
                page.set_default_timeout(0)
                await page.goto(url, wait_until="domcontentloaded", timeout=6000)
                # Give Google client-side navigation a chance to complete, but do
                # not impose a wall-clock abort. Event-based checks stop as soon
                # as a publisher URL is observed.
                for _ in range(20):
                    current = page.url
                    if self._valid_destination(current):
                        return ResolveResult(current, "browser")
                    html = await page.content()
                    candidates = self._browser_candidates(html, current)
                    if candidates:
                        # Prefer the canonical/meta destination over arbitrary
                        # Google-page links; candidates are already filtered.
                        return ResolveResult(candidates[0], "browser")
                    await asyncio.sleep(0.25)
                return ResolveResult(url, "browser-failed", "publisher-url-not-observed")
            finally:
                await page.close()

    @staticmethod
    def is_google_url(url: str) -> bool:
        try:
            p = urlparse(str(url))
            return (p.hostname or "").lower() == GOOGLE_HOST and (
                p.path.startswith("/rss/articles/")
                or p.path.startswith("/articles/")
                or p.path.startswith("/read/")
            )
        except Exception:
            return False

    @staticmethod
    def article_id(url: str) -> str | None:
        try:
            p = urlparse(url)
            part = p.path.rstrip("/").split("/")[-1]
            return unquote(part) if part else None
        except Exception:
            return None

    @staticmethod
    @staticmethod
    def _host_is_google_infra(host: str) -> bool:
        h = (host or "").lower().rstrip(".")
        return (
            h == "google.com" or h.endswith(".google.com")
            or h == "gstatic.com" or h.endswith(".gstatic.com")
            or h == "googleusercontent.com" or h.endswith(".googleusercontent.com")
            or h == "googleapis.com" or h.endswith(".googleapis.com")
        )

    @classmethod
    def _valid_destination(cls, value: str | None) -> bool:
        if not value:
            return False
        try:
            p = urlparse(value)
            host = (p.hostname or "").lower()
            if p.scheme not in ("http", "https") or not host:
                return False
            if cls._host_is_google_infra(host):
                return False
            if any(x in host for x in (
                "doubleclick.net", "googlesyndication.com", "google-analytics.com",
                "googletagmanager.com", "googleadservices.com",
            )):
                return False
            path = (p.path or "").lower()
            # A Google redirect page can expose scripts/images/fonts as external
            # URLs. Those are not article destinations.
            if re.search(r"\.(?:js|css|mjs|woff2?|ttf|otf|png|jpe?g|gif|webp|svg|ico|mp4|webm)(?:$|\?)", path):
                return False
            return True
        except Exception:
            return False

    def _cache_get(self, key: str) -> ResolveResult | None:
        item = self._cache.get(key)
        if not item:
            return None
        expires, result = item
        if expires <= time.monotonic():
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return ResolveResult(result.url, "cache", result.error)

    def _cache_put(self, key: str, result: ResolveResult, ttl: float = CACHE_TTL) -> None:
        self._cache[key] = (time.monotonic() + ttl, result)
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_MAX:
            self._cache.popitem(last=False)

    @staticmethod
    def _extract_params(html: str, article_id: str) -> dict[str, str] | None:
        soup = BeautifulSoup(html, "lxml")
        # Google normally exposes all three attributes on c-wiz > div.
        candidates = soup.select("c-wiz > div[data-n-a-sg][data-n-a-ts]")
        candidates += soup.select("div[data-n-a-sg][data-n-a-ts]")
        for node in candidates:
            sig = node.get("data-n-a-sg")
            ts = node.get("data-n-a-ts")
            data_id = node.get("data-n-a-id") or article_id
            if sig and ts and data_id:
                return {"id": data_id, "sig": sig, "ts": ts}

        # Regex fallback handles malformed/partial HTML where BeautifulSoup
        # cannot build the expected tree.
        patterns = [
            r'data-n-a-id=["\']([^"\']+)["\'][^>]*data-n-a-sg=["\']([^"\']+)["\'][^>]*data-n-a-ts=["\']([^"\']+)',
            r'data-n-a-sg=["\']([^"\']+)["\'][^>]*data-n-a-ts=["\']([^"\']+)',
        ]
        for i, pat in enumerate(patterns):
            m = re.search(pat, html, re.I | re.S)
            if not m:
                continue
            if i == 0:
                return {"id": m.group(1), "sig": m.group(2), "ts": m.group(3)}
            return {"id": article_id, "sig": m.group(1), "ts": m.group(2)}
        return None

    @staticmethod
    def _legacy_extract(value: str) -> str | None:
        """Conservative fallback for older Google encodings that embed URLs."""
        seen: set[str] = set()
        queue = [value]
        for _ in range(8):
            if not queue:
                break
            current = queue.pop(0)
            if current in seen or len(current) > 200000:
                continue
            seen.add(current)
            for m in re.finditer(r"https?://[^\s\"'<>\\]+", current):
                candidate = unquote(m.group(0)).rstrip(".,)]}")
                if GoogleNewsResolver._valid_destination(candidate):
                    return candidate
            # URL query values occasionally carry the destination.
            try:
                for vals in parse_qs(urlparse(current).query).values():
                    queue.extend(vals)
            except Exception:
                pass
            # Try urlsafe base64 with/without padding. Only recurse into text
            # that decodes cleanly and contains useful URL-ish characters.
            raw = current.split("?", 1)[0].rstrip("/").split("/")[-1]
            for candidate in (raw, current):
                if len(candidate) < 20 or len(candidate) % 4 == 1:
                    continue
                try:
                    padded = candidate + "=" * (-len(candidate) % 4)
                    decoded = base64.urlsafe_b64decode(padded.encode()).decode("utf-8", "ignore")
                    if decoded and decoded != current and re.search(r"https?://|www\.", decoded):
                        queue.append(decoded)
                except Exception:
                    continue
        return None

    @staticmethod
    def _rpc_payload(params: list[dict[str, str]]) -> str:
        requests = []
        for p in params:
            art = json.dumps(
                [
                    "garturlreq",
                    [
                        [
                            "en-US", "US", ["FINANCE_TOP_INDICES", "WEB_TEST_1_0_0"],
                            None, None, 1, 1, "US:en", None, 1,
                            None, None, None, None, None, 0, 1,
                        ],
                        p["id"], int(float(p["ts"])), p["sig"],
                    ],
                ],
                separators=(",", ":"),
            )
            requests.append(["Fbv4je", art, None, "generic"])
        envelope = json.dumps([requests], separators=(",", ":"))
        return urlencode({"f.req": envelope})

    @staticmethod
    def _urls_from_rpc(text: str) -> list[str]:
        # The response is a newline-delimited wrapper containing JSON strings
        # inside JSON. Parse each line and recursively inspect strings/lists.
        candidates: list[str] = []
        fragments = [text]
        if "\n\n" in text:
            fragments.append(text.split("\n\n", 1)[1])
        for fragment in fragments:
            try:
                obj = json.loads(fragment)
            except Exception:
                continue
            stack = [obj]
            while stack:
                cur = stack.pop()
                if isinstance(cur, str):
                    if "http" in cur:
                        for m in re.finditer(r"https?://[^\s\"'<>\\]+", cur):
                            u = unquote(m.group(0)).replace("\\u003d", "=").replace("\\u0026", "&")
                            u = u.rstrip(".,)]}")
                            if GoogleNewsResolver._valid_destination(u):
                                candidates.append(u)
                    if cur.startswith("[") or cur.startswith("{"):
                        try:
                            stack.append(json.loads(cur))
                        except Exception:
                            pass
                elif isinstance(cur, list):
                    stack.extend(cur)
                elif isinstance(cur, dict):
                    stack.extend(cur.values())
        # Last-resort URL scan over raw response. This is deliberately after
        # structured parsing because the raw response contains Google URLs too.
        for m in re.finditer(r"https?://[^\s\"'<>\\]+", text):
            u = unquote(m.group(0)).replace("\\u003d", "=").replace("\\u0026", "&").rstrip(".,)]}")
            if GoogleNewsResolver._valid_destination(u):
                candidates.append(u)
        return list(dict.fromkeys(candidates))

    async def _fetch_params(self, article_id: str) -> dict[str, str]:
        client = await self.client()
        urls = [
            f"https://news.google.com/articles/{article_id}",
            f"https://news.google.com/rss/articles/{article_id}",
        ]
        last = "params-unavailable"
        async with self._page_sem:
            for target in urls:
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        r = await client.get(target, headers=_BROWSER_HEADERS)
                        if r.status_code == 429:
                            last = "google-429"
                        elif r.status_code >= 500:
                            last = f"google-http-{r.status_code}"
                        elif r.status_code == 200:
                            params = self._extract_params(r.text, article_id)
                            if params:
                                return params
                            legacy = self._legacy_extract(r.text)
                            if legacy:
                                return {"legacy_url": legacy}
                            last = "signature-not-found"
                        else:
                            last = f"google-http-{r.status_code}"
                    except Exception as exc:
                        last = f"params-{type(exc).__name__}"
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep((0.35 * (2 ** attempt)) + random.random() * 0.25)
        raise RuntimeError(last)

    async def _rpc_decode(self, params: list[dict[str, str]]) -> list[str]:
        client = await self.client()
        async with self._rpc_sem:
            # Space RPC calls even when several feed items finish parameter
            # extraction together. This is intentionally conservative: Google
            # can return 429 from batchexecute before the page endpoint does.
            wait = RPC_MIN_INTERVAL - (time.monotonic() - self._last_rpc_at)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_rpc_at = time.monotonic()
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = await client.post(
                        BATCH_ENDPOINT,
                        content=self._rpc_payload(params),
                        headers=_RPC_HEADERS,
                    )
                    if response.status_code == 429:
                        raise RuntimeError("google-rpc-429")
                    if response.status_code >= 500:
                        raise RuntimeError(f"google-rpc-http-{response.status_code}")
                    response.raise_for_status()
                    urls = self._urls_from_rpc(response.text)
                    if urls:
                        return urls
                    raise RuntimeError("google-rpc-empty")
                except Exception:
                    if attempt >= MAX_RETRIES:
                        raise
                    await asyncio.sleep((0.5 * (2 ** attempt)) + random.random() * 0.35)
        raise RuntimeError("google-rpc-failed")

    async def resolve(self, url: str) -> ResolveResult:
        url = str(url)
        if not self.is_google_url(url):
            return ResolveResult(url, "passthrough")
        cached = self._cache_get(url)
        if cached and cached.method not in {"failed", "browser-failed"}:
            return cached

        existing = self._inflight.get(url)
        if existing is not None:
            return await existing

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._inflight[url] = future
        try:
            result = await self._resolve_uncached(url)
            if not future.done():
                future.set_result(result)
            return result
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            self._inflight.pop(url, None)

    async def _resolve_uncached(self, url: str) -> ResolveResult:
        article_id = self.article_id(url)
        if not article_id:
            return ResolveResult(url, "invalid-google-url", "missing-article-id")

        # Phase 1: lightweight HTTP/RPC decoder.
        try:
            params = await self._fetch_params(article_id)
            if params.get("legacy_url") and self._valid_destination(params["legacy_url"]):
                result = ResolveResult(params["legacy_url"], "legacy-embedded")
                self._cache_put(url, result)
                return result
            decoded = await self._rpc_decode([params])
            for candidate in decoded:
                if self._valid_destination(candidate):
                    result = ResolveResult(candidate, "batchexecute")
                    self._cache_put(url, result)
                    return result
            first_error = "google-rpc-no-publisher-url"
        except Exception as exc:
            first_error = str(exc)[:160] or type(exc).__name__

        # Phase 2: ALWAYS use the independent Chromium resolver when the HTTP
        # decoder did not produce a publisher URL. This is deliberately separate
        # from the article-fetch/render browser so Google failures cannot poison
        # publisher extraction state.
        try:
            browser_result = await self._browser_resolve(url)
            if self._valid_destination(browser_result.url):
                self._cache_put(url, browser_result)
                return browser_result
            return ResolveResult(url, "failed", f"{first_error};{browser_result.error or 'browser-failed'}")
        except Exception as exc:
            return ResolveResult(url, "failed", f"{first_error};browser:{type(exc).__name__}")

    async def resolve_many(self, urls: list[str]) -> list[ResolveResult]:
        """Resolve many URLs with bounded concurrency and exact URL association.

        We deliberately do NOT assume Google's batchexecute response order.
        Some reverse-engineered implementations have observed ordering changes,
        so correctness is more important than shaving a few RPC calls.
        """
        sem = asyncio.Semaphore(PAGE_CONCURRENCY)

        async def one(raw: str) -> ResolveResult:
            async with sem:
                return await self.resolve(str(raw))

        return await asyncio.gather(*(one(str(u)) for u in urls))


resolver = GoogleNewsResolver()
