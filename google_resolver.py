"""Google News publisher URL resolver.

Resolves post-2024 Google News RSS article URLs
(news.google.com/rss/articles/...) to the real publisher article URL.

Design:
1. Validate that the input is actually a Google News article URL.
2. Fetch Google's article page and obtain data-n-a-id/data-n-a-sg/data-n-a-ts.
3. Use the documented-by-reverse-engineering garturl Fbv4je batchexecute RPC.
4. Parse garturlres specifically; never choose arbitrary URLs from the response.
5. If RPC is rate-limited/unavailable, use an independent Chromium page only
   as a fallback and extract publisher-looking canonical/OG/JSON-LD/anchor URLs.
6. Never accept Google infrastructure, XML namespaces, schema URLs, assets,
   tracking URLs, or arbitrary third-party links as the publisher URL.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as curl_requests
except Exception:  # optional; requirements.txt installs it in production
    curl_requests = None

logger = logging.getLogger("google_resolver")

GOOGLE_HOST = "news.google.com"
BATCH_ENDPOINT = "https://news.google.com/_/DotsSplashUi/data/batchexecute"

# ── Anti-block layer ────────────────────────────────────────────────────
# Every network call in this module previously went out raw from wherever
# this service is deployed. Multiple independent implementations of this
# same Google News RPC technique (Python, Node.js, Elixir ports) explicitly
# document that Google can rate-limit or block the request regardless of
# headers/TLS fingerprint -- no client-side header trick fixes an IP-level
# block. Route through the SAME residential proxy already used by the
# sibling auth-bypass-scraper service (same env var name, so existing
# credentials work here unchanged) as the primary anti-block measure, with
# ScraperAPI's residential pool as a final fallback specifically for the
# article-page fetch (the step that was observed failing).
# FlareSolverr is intentionally NOT used here: it solves Cloudflare's JS
# challenge specifically, and news.google.com is not behind Cloudflare, so
# it would add complexity without addressing the actual failure mode.
PROXY_URL = os.getenv("PROXY_URL") or None
SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY") or None
SCRAPERAPI_COUNTRY = os.getenv("SCRAPERAPI_COUNTRY") or None


def _proxies() -> dict[str, str] | None:
    if PROXY_URL:
        return {"http": PROXY_URL, "https": PROXY_URL}
    return None


def _playwright_proxy() -> dict[str, str] | None:
    """Playwright wants a structured {server, username, password} dict, not
    the http://user:pass@host:port string form used by requests/httpx."""
    if not PROXY_URL:
        return None
    try:
        p = urlparse(PROXY_URL)
        server = f"{p.scheme}://{p.hostname}:{p.port}" if p.port else f"{p.scheme}://{p.hostname}"
        proxy: dict[str, str] = {"server": server}
        if p.username:
            proxy["username"] = unquote(p.username)
        if p.password:
            proxy["password"] = unquote(p.password)
        return proxy
    except Exception:
        return None

# Keep Google requests bounded. Do not create a browser page per feed item
# unless the RPC path really fails.
PAGE_CONCURRENCY = 3
BATCH_CONCURRENCY = 1
REQUEST_TIMEOUT = 5.0  # keep each network leg below the platform request budget
MAX_RETRIES = 1
CACHE_TTL = 6 * 60 * 60
NEGATIVE_TTL = 12
CACHE_MAX = 2000
RPC_MIN_INTERVAL = 1.5
# Google News resolution is intentionally not bounded by an application-level
# wall-clock deadline. The browser is the authoritative last-resort resolver
# and must be allowed to keep navigating/polling until it obtains a publisher
# URL or the browser itself reports a genuine failure.
RESOLVE_DEADLINE = None
BROWSER_NAV_TIMEOUT_MS = 0
BROWSER_POLL_MS = 500
CHROMIUM_RESOLVE_TIMEOUT_SECONDS = 30.0

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
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
        self._playwright = None
        self._browser_context = None
        self._browser_lock = asyncio.Lock()
        self._browser_page_sem = asyncio.Semaphore(2)
        self._inflight: dict[str, asyncio.Future] = {}

    @staticmethod
    def _curl_headers() -> dict[str, str]:
        return {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    @staticmethod
    def _curl_get_sync(url: str, impersonate: str):
        if curl_requests is None:
            raise RuntimeError("curl-cffi-not-installed")
        return curl_requests.get(
            url,
            impersonate=impersonate,
            allow_redirects=True,
            timeout=5,
            headers=GoogleNewsResolver._curl_headers(),
            proxies=_proxies(),
        )

    @staticmethod
    def _curl_post_sync(url: str, body: str, impersonate: str):
        if curl_requests is None:
            raise RuntimeError("curl-cffi-not-installed")
        return curl_requests.post(
            url,
            impersonate=impersonate,
            allow_redirects=True,
            timeout=5,
            headers=_RPC_HEADERS,
            data=body,
            proxies=_proxies(),
        )

    @staticmethod
    def _scraperapi_get_sync(url: str):
        """Final-resort fetch of the Google article page via ScraperAPI's
        residential pool. Only used when both curl_cffi (proxied) and the
        proxied httpx client have already failed."""
        if not SCRAPERAPI_KEY:
            raise RuntimeError("scraperapi-not-configured")
        if curl_requests is None:
            raise RuntimeError("curl-cffi-not-installed")
        params = {"api_key": SCRAPERAPI_KEY, "url": url}
        if SCRAPERAPI_COUNTRY:
            params["country_code"] = SCRAPERAPI_COUNTRY
        return curl_requests.get(
            "https://api.scraperapi.com", params=params, timeout=25,
        )

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
                    proxy=PROXY_URL,  # httpx>=0.26 uses a single proxy=, not proxies={}
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
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get_browser(self):
        """Independent Chromium used only for Google News resolution."""
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
                proxy=_playwright_proxy(),
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
            await self._browser_context.set_extra_http_headers(
                {"Accept-Language": "en-US,en;q=0.9"}
            )
            return self._browser_context

    @staticmethod
    def is_google_url(url: str) -> bool:
        try:
            p = urlparse(str(url).strip())
            host = (p.hostname or "").lower().rstrip(".")
            return host == GOOGLE_HOST and (
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
    def _host_is_google_infra(host: str) -> bool:
        h = (host or "").lower().rstrip(".")
        return (
            h == "google.com" or h.endswith(".google.com")
            or h == "gstatic.com" or h.endswith(".gstatic.com")
            or h == "googleusercontent.com" or h.endswith(".googleusercontent.com")
            or h == "googleapis.com" or h.endswith(".googleapis.com")
            or h == "ggpht.com" or h.endswith(".ggpht.com")
            or h == "googlevideo.com" or h.endswith(".googlevideo.com")
        )

    @staticmethod
    def _host_is_non_article_infra(host: str) -> bool:
        h = (host or "").lower().rstrip(".")
        exact = {
            "w3.org", "www.w3.org",
            "schema.org", "www.schema.org",
            "xml.org", "www.xml.org",
            "example.com", "example.org", "example.net",
            "localhost",
        }
        if h in exact:
            return True
        if h.endswith(".w3.org") or h.endswith(".schema.org"):
            return True
        # Common tracking/asset/CDN infrastructure. These are not accepted
        # unless the actual publisher domain is separately identified.
        return any(x in h for x in (
            "doubleclick.net", "googlesyndication.com",
            "google-analytics.com", "googletagmanager.com",
            "googleadservices.com", "facebook.com", "facebook.net",
            "twitter.com", "x.com", "youtube.com", "youtube-nocookie.com",
            "instagram.com", "linkedin.com",
        ))

    @classmethod
    def _valid_destination(cls, value: str | None) -> bool:
        """Strict publisher URL gate.

        A URL is NOT a publisher merely because it is external to Google.
        This is the critical protection against values such as
        https://www.w3.org/XML/1998/namespace appearing in XML/HTML.
        """
        if not value:
            return False
        try:
            value = unquote(str(value)).strip()
            p = urlparse(value)
            host = (p.hostname or "").lower().rstrip(".")
            if p.scheme not in ("http", "https") or not host:
                return False
            if cls._host_is_google_infra(host) or cls._host_is_non_article_infra(host):
                return False
            if host.startswith(("cdn.", "static.", "assets.", "fonts.", "img.")):
                # Do not blindly reject real publishers using these prefixes;
                # require an article-like path below.
                path = (p.path or "").lower()
                if not any(x in path for x in (
                    "/article", "/news/", "/story/", "/stories/",
                    "/world/", "/business/", "/sports/", "/technology/",
                )):
                    return False

            path = (p.path or "").lower()
            if re.search(
                r"\.(?:js|css|mjs|woff2?|ttf|otf|eot|png|jpe?g|gif|webp|svg|ico|"
                r"mp4|webm|mp3|wav|json|xml)(?:$|\?)",
                path,
            ):
                return False

            # XML namespace / schema URLs often have these path forms.
            if host in {"w3.org", "www.w3.org"}:
                return False
            if path in {"/xml/1998/namespace", "/2001/xml.xsd"}:
                return False

            return True
        except Exception:
            return False

    @classmethod
    def _article_like_score(cls, url: str, anchor_text: str = "") -> int:
        p = urlparse(url)
        path = (p.path or "").lower()
        score = 0
        if len(path.strip("/")) >= 20:
            score += 8
        if any(x in path for x in (
            "/article", "/news/", "/story/", "/stories/", "/world/",
            "/business/", "/sports/", "/technology/", "/politics/",
            "/india/", "/entertainment/", "/science/", "/health/",
        )):
            score += 15
        if path in ("", "/"):
            score -= 35
        text = (anchor_text or "").strip().lower()
        if len(text) >= 25:
            score += 4
        if text and any(x in text for x in ("read", "article", "news", "story")):
            score += 3
        return score

    @classmethod
    def _browser_candidates(cls, html: str, current_url: str) -> list[str]:
        """Extract only plausible publisher URLs from Google-rendered HTML.

        Never regex every URL in the page and accept the first external URL.
        That old behavior is exactly how the W3 XML namespace became a fake
        publisher.
        """
        scored: dict[str, int] = {}
        soup = BeautifulSoup(html or "", "lxml")

        def add(value: str | None, base_score: int, text: str = "") -> None:
            if not value:
                return
            value = unquote(str(value)).strip()
            if value.startswith("//"):
                value = "https:" + value
            absolute = urljoin(current_url, value).split("#", 1)[0]
            if not cls._valid_destination(absolute):
                return
            score = base_score + cls._article_like_score(absolute, text)
            scored[absolute] = max(scored.get(absolute, -999), score)

        # Canonical / OG are strongest. These should beat arbitrary links.
        for tag_name, attrs, field, score in (
            ("link", {"rel": lambda v: v and "canonical" in v}, "href", 120),
            ("meta", {"property": "og:url"}, "content", 115),
            ("meta", {"name": "twitter:url"}, "content", 105),
        ):
            for tag in soup.find_all(tag_name, attrs=attrs):
                add(tag.get(field), score)

        # JSON-LD URLs, but only after strict validation.
        for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
            try:
                data = json.loads(script.string or script.get_text() or "")
            except Exception:
                continue

            def walk(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "url" and isinstance(v, str):
                            add(v, 100)
                        else:
                            walk(v)
                elif isinstance(obj, list):
                    for item in obj:
                        walk(item)

            walk(data)

        # Google sometimes renders the source as an ordinary external anchor.
        # Anchor text is retained for scoring, unlike the old raw URL regex.
        for a in soup.find_all("a", href=True):
            text = a.get_text(" ", strip=True)
            add(a.get("href"), 65, text)

        return [u for u, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))]

    @staticmethod
    def _extract_data_p(html: str, article_id: str) -> dict[str, str] | None:
        """Extract the page-provided c-wiz[data-p] decoder payload.

        This is preferred because the current Google News page can carry the
        exact signed request parameters there. We never use arbitrary URLs from
        the HTML.
        """
        soup = BeautifulSoup(html or "", "lxml")
        for node in soup.select("c-wiz[data-p]"):
            raw = node.get("data-p")
            if not raw:
                continue
            text = unquote(str(raw))
            # Keep only data that identifies this exact requested article.
            if article_id not in text:
                continue
            sig = node.get("data-n-a-sg")
            ts = node.get("data-n-a-ts")
            data_id = node.get("data-n-a-id") or article_id
            if sig and ts:
                return {"id": data_id, "sig": sig, "ts": ts, "data_p": text}
        return None

    @staticmethod
    def _extract_params(html: str, article_id: str) -> dict[str, str] | None:
        data_p = GoogleNewsResolver._extract_data_p(html, article_id)
        if data_p:
            return data_p
        soup = BeautifulSoup(html or "", "lxml")
        # Prefer the node whose data-n-a-id matches the actual article token.
        nodes = soup.select("c-wiz > div[data-n-a-sg][data-n-a-ts]")
        nodes += soup.select("div[data-n-a-sg][data-n-a-ts]")
        for node in nodes:
            sig = node.get("data-n-a-sg")
            ts = node.get("data-n-a-ts")
            data_id = node.get("data-n-a-id")
            if not sig or not ts:
                continue
            # Fail closed when Google exposes an explicit article token.
            # Never substitute the requested token for an unrelated node.
            if data_id and data_id != article_id:
                continue
            if data_id == article_id:
                return {"id": data_id, "sig": sig, "ts": ts}
            # If the page omits data-n-a-id, only accept the signature when the
            # exact requested token is present in the same element's data-p.
            data_p = node.get("data-p") or ""
            if article_id in data_p:
                return {"id": article_id, "sig": sig, "ts": ts, "data_p": data_p}

        # Conservative textual fallback: explicit data-n-a-id must match.
        m = re.search(
            r'data-n-a-id=["\']([^"\']+)["\'][^>]*'
            r'data-n-a-sg=["\']([^"\']+)["\'][^>]*'
            r'data-n-a-ts=["\']([^"\']+)',
            html or "", re.I | re.S,
        )
        if m and m.group(1) == article_id:
            return {"id": m.group(1), "sig": m.group(2), "ts": m.group(3)}
        return None

    @classmethod
    def _legacy_extract(cls, value: str) -> str | None:
        """Conservative compatibility fallback for older embedded URLs."""
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
                if cls._valid_destination(candidate):
                    return candidate
            try:
                for vals in parse_qs(urlparse(current).query).values():
                    queue.extend(vals)
            except Exception:
                pass
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
                    pass
        return None

    @staticmethod
    def _rpc_payload(params: list[dict[str, str]]) -> str:
        """Build the current Fbv4je/garturlreq payload.

        The signed id/timestamp/signature are page-provided. The surrounding
        garturl request shape follows current community decoders; never invent a
        publisher URL by scanning arbitrary Google HTML.
        """
        requests = []
        for p in params:
            art = json.dumps(
                [
                    "garturlreq",
                    [
                        [
                            "X", "X", ["X", "X"], None, None, 1, 1,
                            "US:en", None, 1, None, None, None, None,
                            None, 0, 1,
                        ],
                        "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0,
                    ],
                    p["id"], int(float(p["ts"])), p["sig"],
                ],
                separators=(",", ":"),
            )
            requests.append(["Fbv4je", art, None, "generic"])
        return urlencode({"f.req": json.dumps([requests], separators=(",", ":"))})

    @classmethod
    def _urls_from_rpc(cls, text: str) -> list[str]:
        """Parse the batchexecute response.

        First try the exact garturlres field. Only then use a strict recursive
        URL scan as a compatibility fallback.
        """
        candidates: list[str] = []

        # Exact garturlres extraction. Google wraps the response in an
        # anti-XSSI prefix/newline and nested JSON.
        marker = '\\"garturlres\\",\\"'
        pos = text.find(marker)
        if pos >= 0:
            start = pos + len(marker)
            end = text.find('\\",', start)
            if end > start:
                raw = text[start:end]
                raw = raw.replace("\\u003d", "=").replace("\\u0026", "&").replace("\\/", "/")
                raw = unquote(raw)
                if cls._valid_destination(raw):
                    candidates.append(raw)

        # JSON-aware parsing for current/variant response wrappers.
        for fragment in (text, text.split("\n\n", 1)[1] if "\n\n" in text else ""):
            if not fragment:
                continue
            try:
                obj = json.loads(fragment)
            except Exception:
                continue

            def walk(x):
                if isinstance(x, list):
                    for v in x:
                        walk(v)
                elif isinstance(x, dict):
                    for k, v in x.items():
                        if k == "garturlres" and isinstance(v, str):
                            u = unquote(v).replace("\\u003d", "=").replace("\\u0026", "&")
                            if cls._valid_destination(u):
                                candidates.append(u)
                        walk(v)
                elif isinstance(x, str):
                    # Do not inspect XML namespaces or arbitrary HTML as a
                    # source of publisher URLs. Only explicit URL strings that
                    # pass the strict destination gate are accepted.
                    if x.startswith(("http://", "https://")) and cls._valid_destination(x):
                        candidates.append(x)
                    if x.startswith(("[", "{")):
                        try:
                            walk(json.loads(x))
                        except Exception:
                            pass

            walk(obj)

        return list(dict.fromkeys(candidates))

    async def _fetch_params(self, article_id: str) -> dict[str, str]:
        """Fetch the Google article page and obtain signed decoder parameters.

        Prefer curl-cffi browser TLS impersonation because Google can treat a
        plain httpx TLS fingerprint differently from a real browser.  Chrome is
        tried first and Safari second.  A short per-leg timeout is used so one Google stall cannot kill the briefing.
        """
        # Current community implementations report fewer 429s when the
        # /articles endpoint is tried before /rss/articles. Keep RSS as the
        # fallback because both formats are seen in feeds.
        targets = (
            f"https://news.google.com/articles/{article_id}",
            f"https://news.google.com/rss/articles/{article_id}",
        )
        last = "params-unavailable"

        # Browser-TLS path. This is synchronous curl-cffi, so keep it off the
        # event loop. No timeout is passed: user explicitly requested no abort.
        if curl_requests is not None:
            for target in targets:
                for fingerprint in ("chrome", "safari"):
                    for attempt in range(MAX_RETRIES + 1):
                        try:
                            response = await asyncio.to_thread(
                                self._curl_get_sync, target, fingerprint
                            )
                            status = int(response.status_code)
                            text = response.text or ""
                            if status == 429:
                                last = f"google-429-{fingerprint}"
                            elif status == 200:
                                params = self._extract_params(text, article_id)
                                if params:
                                    params["fetch_method"] = f"curl-{fingerprint}"
                                    return params
                                legacy = self._legacy_extract(text)
                                if legacy:
                                    return {"legacy_url": legacy, "fetch_method": f"curl-{fingerprint}"}
                                last = "signature-not-found"
                            else:
                                last = f"google-http-{status}"
                        except Exception as exc:
                            last = f"curl-{fingerprint}-{type(exc).__name__}"
                        if attempt < MAX_RETRIES:
                            delay = 1.0 + attempt * 1.5 + random.random()
                            await asyncio.sleep(delay)

        # httpx fallback with the same bounded timeout.
        client = await self.client()
        async with self._page_sem:
            for target in targets:
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        response = await client.get(target)
                        if response.status_code == 429:
                            last = "google-429-httpx"
                        elif response.status_code == 200:
                            params = self._extract_params(response.text, article_id)
                            if params:
                                params["fetch_method"] = "httpx"
                                return params
                            legacy = self._legacy_extract(response.text)
                            if legacy:
                                return {"legacy_url": legacy, "fetch_method": "httpx"}
                            last = "signature-not-found"
                        else:
                            last = f"google-http-{response.status_code}"
                    except Exception as exc:
                        last = f"params-{type(exc).__name__}"
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1.0 + attempt * 1.5 + random.random())

        # Final resort: ScraperAPI's residential pool. Only reached when both
        # the proxied curl_cffi attempts and the proxied httpx attempts above
        # have already failed for every target URL.
        if SCRAPERAPI_KEY:
            for target in targets:
                try:
                    response = await asyncio.to_thread(self._scraperapi_get_sync, target)
                    status = int(response.status_code)
                    if status == 200:
                        text = response.text or ""
                        params = self._extract_params(text, article_id)
                        if params:
                            params["fetch_method"] = "scraperapi"
                            return params
                        legacy = self._legacy_extract(text)
                        if legacy:
                            return {"legacy_url": legacy, "fetch_method": "scraperapi"}
                        last = "signature-not-found-scraperapi"
                    else:
                        last = f"scraperapi-http-{status}"
                except Exception as exc:
                    last = f"scraperapi-{type(exc).__name__}"

        raise RuntimeError(last)

    async def _rpc_decode(self, params: dict[str, str]) -> str:
        """Call Google's Fbv4je decoder with browser TLS impersonation first."""
        body = self._rpc_payload([params])
        last_exc: Exception | None = None

        async with self._rpc_sem:
            if curl_requests is not None:
                for fingerprint in ("chrome", "safari"):
                    for attempt in range(MAX_RETRIES + 1):
                        try:
                            response = await asyncio.to_thread(
                                self._curl_post_sync, BATCH_ENDPOINT, body, fingerprint
                            )
                            status = int(response.status_code)
                            if status == 429:
                                raise RuntimeError("google-rpc-429")
                            if status >= 500:
                                raise RuntimeError(f"google-rpc-http-{status}")
                            response.raise_for_status()
                            urls = self._urls_from_rpc(response.text or "")
                            if urls:
                                return urls[0]
                            raise RuntimeError("google-rpc-empty")
                        except Exception as exc:
                            last_exc = exc
                            if attempt < MAX_RETRIES:
                                await asyncio.sleep(1.0 + attempt * 2.0 + random.random())

            client = await self.client()
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = await client.post(
                        BATCH_ENDPOINT, content=body, headers=_RPC_HEADERS
                    )
                    if response.status_code == 429:
                        raise RuntimeError("google-rpc-429")
                    if response.status_code >= 500:
                        raise RuntimeError(f"google-rpc-http-{response.status_code}")
                    response.raise_for_status()
                    urls = self._urls_from_rpc(response.text or "")
                    if urls:
                        return urls[0]
                    raise RuntimeError("google-rpc-empty")
                except Exception as exc:
                    last_exc = exc
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1.0 + attempt * 2.0 + random.random())

        raise last_exc or RuntimeError("google-rpc-failed")

    async def _resolve_http(self, url: str) -> ResolveResult:
        article_id = self.article_id(url)
        if not article_id:
            return ResolveResult(url, "invalid-google-url", "missing-article-id")
        try:
            params = await self._fetch_params(article_id)
            if params.get("legacy_url") and self._valid_destination(params["legacy_url"]):
                return ResolveResult(params["legacy_url"], "legacy-embedded")

            destination = await self._rpc_decode(params)
            if self._valid_destination(destination):
                return ResolveResult(destination, "batchexecute")

            return ResolveResult(url, "failed", "google-rpc-destination-rejected")
        except Exception as exc:
            return ResolveResult(url, "failed", str(exc)[:240])

    CHROMIUM_RESOLVE_TIMEOUT_SECONDS = 30.0

    async def _browser_resolve(self, url: str) -> ResolveResult:
        """Resolve with the server's installed Chromium, with a 30-second Chromium-only budget."""
        context = await self._get_browser()
        async with self._browser_page_sem:
            page = await context.new_page()
            # Google redirect resolution needs HTML/JS, but not images, fonts,
            # media, or stylesheets. Blocking those reduces latency and load
            # without interfering with the navigation/JS that performs the
            # redirect.
            async def _route(route):
                try:
                    if route.request.resource_type in {"image", "font", "media", "stylesheet"}:
                        await route.abort()
                    else:
                        await route.continue_()
                except Exception:
                    try:
                        await route.continue_()
                    except Exception:
                        pass
            await page.route("**/*", _route)
            candidates: list[str] = []
            navigation = None

            def remember(value: str | None, source: str = "") -> None:
                if not value:
                    return
                value = unquote(str(value)).strip()
                if value.startswith("//"):
                    value = "https:" + value
                absolute = urljoin(page.url or url, value).split("#", 1)[0]
                if self._valid_destination(absolute) and absolute not in candidates:
                    candidates.append(absolute)
                    logger.info("Google Chromium candidate (%s): %s", source, absolute)

            def on_response(response) -> None:
                try:
                    if getattr(response.request, "resource_type", "") == "document":
                        remember(response.url, "document-response")
                except Exception:
                    pass

            page.on("response", on_response)
            try:
                # Chromium gets the only explicit resolver timeout: 20 seconds.
                # Playwright's per-navigation timeout remains disabled so we control
                # the budget here and can also inspect redirects/DOM continuously.
                page.set_default_navigation_timeout(0)
                page.set_default_timeout(0)

                # Do not wait for DOMContentLoaded before inspecting the browser.
                # Google can navigate/redirect while the original navigation task
                # is still pending.
                navigation = asyncio.create_task(
                    page.goto(url, wait_until="commit", timeout=0)
                )

                # Chromium is allowed to follow redirects and render client-side
                # navigation freely, but the entire Chromium resolver gets a 30-second
                # budget so one Google URL cannot occupy a server worker forever.
                settled_since = None
                POST_NAV_STABILIZE_SECONDS = 20.0
                chromium_deadline = time.monotonic() + CHROMIUM_RESOLVE_TIMEOUT_SECONDS

                while True:
                    if time.monotonic() >= chromium_deadline:
                        return ResolveResult(
                            url,
                            "browser-timeout",
                            "chromium-resolve-timeout-30s",
                        )
                    current = page.url or url
                    remember(current, "page.url")

                    try:
                        html = await page.content()
                        for candidate in self._browser_candidates(html, current):
                            remember(candidate, "dom")
                    except Exception:
                        pass

                    if candidates:
                        best = max(candidates, key=self._article_like_score)
                        if self._article_like_score(best) >= 8:
                            if navigation and not navigation.done():
                                navigation.cancel()
                            return ResolveResult(best, "browser")

                    if navigation and navigation.done():
                        if settled_since is None:
                            settled_since = time.monotonic()
                        try:
                            await navigation
                        except Exception as exc:
                            if candidates:
                                best = max(candidates, key=self._article_like_score)
                                return ResolveResult(best, "browser")
                            return ResolveResult(url, "browser-failed", f"{type(exc).__name__}: {exc}"[:240])

                        # Chromium has finished its navigation. Continue watching
                        # for client-side redirects, but don't let a page that has
                        # clearly settled with no publisher URL hang the entire
                        # FastAPI worker indefinitely.
                        if time.monotonic() - settled_since >= POST_NAV_STABILIZE_SECONDS:
                            return ResolveResult(
                                url,
                                "browser-failed",
                                "chromium-navigation-complete-no-publisher",
                            )

                    await asyncio.sleep(BROWSER_POLL_MS / 1000.0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return ResolveResult(url, "browser-failed", f"{type(exc).__name__}: {exc}"[:240])
            finally:
                if navigation is not None and not navigation.done():
                    navigation.cancel()
                    try:
                        await navigation
                    except BaseException:
                        pass
                try:
                    await page.close()
                except Exception:
                    pass

    def _cache_get(self, key: str) -> ResolveResult | None:
        item = self._cache.get(key)
        if not item:
            return None
        expires, result = item
        if expires <= time.monotonic():
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return ResolveResult(result.url, "failed" if result.method == "failed" else "cache", result.error)

    def _cache_put(self, key: str, result: ResolveResult, ttl: float = CACHE_TTL) -> None:
        if ttl <= 0:
            return
        self._cache[key] = (time.monotonic() + ttl, result)
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_MAX:
            self._cache.popitem(last=False)

    async def resolve(self, url: str) -> ResolveResult:
        url = str(url).strip()
        if not self.is_google_url(url):
            return ResolveResult(url, "passthrough")

        cached = self._cache_get(url)
        if cached:
            return cached

        existing = self._inflight.get(url)
        if existing is not None:
            return await existing

        loop = asyncio.get_running_loop()
        future = loop.create_future()
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
        # No application-level timeout surrounds staged Google resolution.
        try:
            return await self._resolve_staged(url)
        except Exception as exc:
            result = ResolveResult(url, "failed", f"resolver-error:{type(exc).__name__}:{exc}"[:300])
            self._cache_put(url, result, ttl=NEGATIVE_TTL)
            return result

    async def _resolve_staged(self, url: str) -> ResolveResult:
        # Authoritative RPC first; browser is independent fallback.
        http_result = await self._resolve_http(url)
        if (
            http_result.method not in ("failed", "invalid-google-url")
            and self._valid_destination(http_result.url)
        ):
            self._cache_put(url, http_result)
            return http_result

        browser_result = await self._browser_resolve(url)
        if self._valid_destination(browser_result.url) and browser_result.method.startswith("browser"):
            self._cache_put(url, browser_result)
            return browser_result

        detail = "; ".join(x for x in [http_result.error, browser_result.error] if x)
        result = ResolveResult(url, "failed", detail[:300] or "google-url-unresolved")
        self._cache_put(url, result, ttl=NEGATIVE_TTL)
        return result

    async def resolve_many(self, urls: list[str]) -> list[ResolveResult]:
        sem = asyncio.Semaphore(PAGE_CONCURRENCY)

        async def one(u: str):
            async with sem:
                return await self.resolve(u)

        return await asyncio.gather(*(one(u) for u in urls))


resolver = GoogleNewsResolver()
