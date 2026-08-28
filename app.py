import asyncio
import ipaddress
import json
import re
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import httpx
import trafilatura
try:
    from googlenewsdecoder import gnewsdecoder
except Exception:
    gnewsdecoder = None
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

app = FastAPI(
    title="NEWS BYTE Source Extractor",
    description="Non-AI source article extraction service for NEWS BYTE.",
    version="1.4.0",
)

# NEWS BYTE is a personal extension. CORS is open so the extension can call
# the Space directly. For a public service, restrict allow_origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def _close_http_client():
    if HTTP_CLIENT is not None and not HTTP_CLIENT.is_closed:
        await HTTP_CLIENT.aclose()

# A single hard-coded UA with a custom "NEWS-BYTE/1.0" token is a trivial
# fingerprint: it's unique to this scraper, it's the exact same string on
# every request, and it doesn't match any real browser network's TLS/HTTP2
# signature. Sites that hard-block scrapers (zeenews.india.com,
# hospitalitybizindia.com in testing) do it on exactly this kind of signal.
# Use a small pool of current, ordinary desktop-browser UA strings and pick
# one per request/host so requests look like normal traffic and repeated
# hits on the same domain don't all present an identical fingerprint.
BROWSER_PROFILES = [
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        "sec-ch-ua-platform": '"Windows"',
        "sec-ch-ua-mobile": "?0",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.5 Safari/605.1.15"
        ),
        "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="127"',
        "sec-ch-ua-platform": '"macOS"',
        "sec-ch-ua-mobile": "?0",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0"
        ),
        "sec-ch-ua": '"Microsoft Edge";v="127", "Not;A=Brand";v="24", "Chromium";v="127"',
        "sec-ch-ua-platform": '"Windows"',
        "sec-ch-ua-mobile": "?0",
    },
]

# A well-behaved, clearly-labelled crawler UA. Some publishers 403 an
# anonymous "browser" UA on the very first hit from a datacenter IP but do
# allow known search-engine crawlers through (they need the SEO traffic).
# Used only as a last-resort retry, never as the first attempt.
CRAWLER_FALLBACK_UA = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"

MAX_DOWNLOAD_BYTES = 8_000_000
MIN_GOOD_WORDS = 120
MIN_GOOD_SCORE = 0.30
FETCH_RETRY_STATUSES = {403, 429, 503}
PER_HOST_CONCURRENCY = 3

# Google News RSS article links are encoded Google redirect URLs, not publisher
# article URLs. Keep a small in-process cache to avoid resolving the same link
# repeatedly during a feed refresh.
GOOGLE_RESOLVE_CACHE = {}
GOOGLE_RESOLVE_LOCK = asyncio.Lock()
GOOGLE_RESOLVE_CACHE_MAX = 500

# One shared, connection-pooled client instead of opening/closing a fresh
# TLS connection per article. This is the single biggest speed win when the
# extension crawls a domain (dozens of /extract calls back-to-back): keeps
# TCP/TLS handshakes warm to hosts that get hit repeatedly.
HTTP_CLIENT: "httpx.AsyncClient | None" = None

# Hammering one host with 10 concurrent requests (a full-domain crawl) is
# exactly the pattern that trips basic rate limiting. Cap concurrency per
# host while leaving cross-host requests fully parallel.
_HOST_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
_HOST_SEMAPHORES_LOCK = asyncio.Lock()


async def get_http_client() -> httpx.AsyncClient:
    global HTTP_CLIENT
    if HTTP_CLIENT is None or HTTP_CLIENT.is_closed:
        HTTP_CLIENT = httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=5,
            timeout=httpx.Timeout(20.0, connect=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            http2=True,
        )
    return HTTP_CLIENT


async def get_host_semaphore(host: str) -> asyncio.Semaphore:
    async with _HOST_SEMAPHORES_LOCK:
        sem = _HOST_SEMAPHORES.get(host)
        if sem is None:
            sem = asyncio.Semaphore(PER_HOST_CONCURRENCY)
            _HOST_SEMAPHORES[host] = sem
        return sem


def pick_browser_profile(url: str) -> dict:
    """Deterministic-but-varied UA choice: same host tends to get the same
    profile within a run (consistent fingerprint per session), different
    hosts get spread across the pool."""
    host = urlparse(url).hostname or ""
    return BROWSER_PROFILES[hash(host) % len(BROWSER_PROFILES)]


def browser_headers(url: str, profile: dict | None = None) -> dict:
    profile = profile or pick_browser_profile(url)
    origin = f"{urlparse(url).scheme}://{urlparse(url).hostname}"
    return {
        "User-Agent": profile["User-Agent"],
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": origin + "/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "sec-ch-ua": profile.get("sec-ch-ua", ""),
        "sec-ch-ua-platform": profile.get("sec-ch-ua-platform", ""),
        "sec-ch-ua-mobile": profile.get("sec-ch-ua-mobile", "?0"),
    }


def is_google_news_article_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
        path = urlparse(url).path
        return host == "news.google.com" and (
            path.startswith("/rss/articles/") or path.startswith("/articles/") or path.startswith("/read/")
        )
    except Exception:
        return False


async def resolve_google_news_url(url: str):
    """Resolve a Google News RSS redirect to the publisher URL."""
    if not is_google_news_article_url(url):
        return url, None

    cached = GOOGLE_RESOLVE_CACHE.get(url)
    if cached:
        return cached, "cache"

    if gnewsdecoder is None:
        return url, "decoder-unavailable"

    try:
        # The decoder performs Google's current signature/timestamp resolution.
        result = await asyncio.to_thread(gnewsdecoder, url, interval=0)
        if isinstance(result, dict) and result.get("status") and result.get("decoded_url"):
            resolved = str(result["decoded_url"])
            if is_public_url(resolved):
                if len(GOOGLE_RESOLVE_CACHE) >= GOOGLE_RESOLVE_CACHE_MAX:
                    GOOGLE_RESOLVE_CACHE.pop(next(iter(GOOGLE_RESOLVE_CACHE)))
                GOOGLE_RESOLVE_CACHE[url] = resolved
                return resolved, "googlenewsdecoder"
        return url, "decoder-failed"
    except Exception as exc:
        return url, "decoder-" + type(exc).__name__


class ExtractRequest(BaseModel):
    url: HttpUrl
    render: bool = False
    max_chars: int = 60000


def is_public_url(url: str) -> bool:
    """Basic SSRF protection: allow only public HTTP(S) destinations."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False

    host = parsed.hostname.lower()

    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return False

    try:
        addresses = socket.getaddrinfo(host, None)
        for info in addresses:
            ip = ipaddress.ip_address(info[4][0])
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
            ):
                return False
    except Exception:
        return False

    return True


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


# Lines that are common publisher chrome/boilerplate rather than article
# content: share/follow prompts, app-download nags, cookie/legal notices,
# "Also Read" cross-promo links, live-blog labels, etc. These regularly slip
# through a plain paragraph split and make extracted "news" read like a page
# full of ads and navigation instead of the story itself.
_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^(also|must)\s+(read|watch|see)\b",
        r"\bclick here\b",
        r"^read more\b",
        r"\bfollow (us|npr|ndtv)?\s*on\s+(twitter|facebook|instagram|whatsapp|telegram|x)\b",
        r"\bdownload (the|our)\s+app\b",
        r"\bsubscribe to\b.*(newsletter|channel|premium)",
        r"\bsign up for\b.*newsletter",
        r"\bwhatsapp channel\b",
        r"^advertisement$",
        r"^sponsored\b",
        r"\ball rights reserved\b",
        r"^copyright\s*(©|\(c\))",
        r"\bterms (of|and)\s*(use|service|conditions)\b",
        r"\bprivacy policy\b",
        r"\bcookie(s)?\s+policy\b",
        r"\bwe use cookies\b",
        r"^disclaimer\s*:",
        r"\bviews (expressed|are personal)\b",
        r"^catch all the\b",
        r"^stay updated with\b",
        r"this (story|article)\s+(has not been edited|is auto-generated)",
        r"^share (this|via|on)\b",
        r"^(photo gallery|view all images|in pictures)\b",
        r"^trending (news|now|stories)\b",
        r"^(watch|must watch)\s*[:\-]",
        r"^loading\.{2,3}$",
        r"\benable javascript\b",
        r"^\(?(reuters|ap|pti|ani|afp)\)?\s*[-—]\s*$",
        r"^\d+\s+(shares?|comments?|min read)$",
        r"^tags?\s*:",
        r"^published\s*:",
        r"^updated\s*:",
        r"^image\s*(credit|source)\s*:",
        r"^(related|recommended|more)\s+(stories|articles|news|posts|reads)\b",
        r"^you\s+(may|might)\s+(also\s+)?like\b",
        r"^more from\b",
        r"^editor'?s?\s+pick(s)?\b",
        r"^trending\s+now\b",
        r"^(sign in|log in|register)\s+to\b",
        r"^create (a free )?account\b",
        r"^comments?\s*\(\d+\)$",
        r"^leave a (comment|reply)\b",
        r"^\d+\s+(min(ute)?s?)\s+(read|ago)$",
        r"^for more (news|updates)\b",
    )
]


def is_boilerplate(paragraph: str) -> bool:
    """True for lines that are page chrome rather than article prose."""
    text_l = paragraph.strip()
    if not text_l:
        return True
    if len(text_l) <= 60 and text_l.isupper():
        # Short all-caps lines are almost always section/nav labels.
        return True
    return any(p.search(text_l) for p in _BOILERPLATE_PATTERNS)


def clean_title(title: str, url: str) -> str:
    """Strip a trailing ' | Publisher Name' / ' - Publisher Name' suffix.

    Only strips when the trailing segment is short and either matches the
    page's own domain or is short enough to plausibly be a site name, so a
    real subtitle (e.g. "Budget 2026: What changes for you - explained")
    is left alone.
    """
    if not title:
        return title
    try:
        domain = (urlparse(url).hostname or "").lower()
        domain_core = re.sub(r"^www\.", "", domain).split(".")[0]
    except Exception:
        domain_core = ""
    for sep in (" | ", " — ", " – ", " - "):
        if sep in title:
            head, _, tail = title.rpartition(sep)
            head, tail = head.strip(), tail.strip()
            if not head or not tail:
                continue
            tail_key = re.sub(r"[^a-z0-9]", "", tail.lower())
            looks_like_site_name = len(tail.split()) <= 5 and (
                (domain_core and len(domain_core) >= 3 and domain_core in tail_key)
                or len(tail_key) <= 24
            )
            if looks_like_site_name:
                return head
    return title


def parse_jsonld(html: str) -> dict:
    """Find NewsArticle/Article JSON-LD and return its useful fields."""
    found = {}

    soup = BeautifulSoup(html, "lxml")

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        if not raw:
            continue

        try:
            obj = json.loads(raw)
        except Exception:
            continue

        candidates = []
        if isinstance(obj, list):
            candidates.extend(obj)
        elif isinstance(obj, dict):
            candidates.append(obj)
            if isinstance(obj.get("@graph"), list):
                candidates.extend(obj["@graph"])

        for item in candidates:
            if not isinstance(item, dict):
                continue

            typ = item.get("@type", "")
            types = typ if isinstance(typ, list) else [typ]

            if (
                any(
                    str(t).lower()
                    in {"newsarticle", "article", "report", "blogposting"}
                    for t in types
                )
                or isinstance(item.get("articleBody"), str)
            ):
                for key in (
                    "headline",
                    "articleBody",
                    "datePublished",
                    "dateModified",
                    "description",
                    "image",
                    "author",
                    "publisher",
                ):
                    if key in item:
                        found[key] = item[key]
                return found

    return found


def extract_metadata(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    jsonld = parse_jsonld(html)

    def meta(*pairs):
        for attr, value in pairs:
            tag = soup.find("meta", attrs={attr: value})
            if tag and tag.get("content"):
                return clean(tag["content"])
        return ""

    title = (
        clean(jsonld.get("headline", ""))
        or meta(
            ("property", "og:title"),
            ("name", "twitter:title"),
        )
    )

    if not title:
        h1 = soup.find("h1")
        if h1:
            title = clean(h1.get_text(" ", strip=True))

    if not title and soup.title:
        title = clean(soup.title.get_text(" ", strip=True))

    description = (
        clean(jsonld.get("description", ""))
        or meta(
            ("property", "og:description"),
            ("name", "description"),
            ("name", "twitter:description"),
        )
    )

    def image_candidate(value):
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("//"):
                value = "https:" + value
            if re.match(r"^https?://", value, re.I) and "news.google.com" not in value.lower():
                return urljoin(url, value)
        elif isinstance(value, dict):
            for k in ("url", "contentUrl", "thumbnailUrl"):
                got = image_candidate(value.get(k))
                if got:
                    return got
        elif isinstance(value, list):
            for item in value:
                got = image_candidate(item)
                if got:
                    return got
        return ""

    image = image_candidate(jsonld.get("image")) or meta(
        ("property", "og:image"),
        ("property", "og:image:url"),
        ("property", "og:image:secure_url"),
        ("name", "og:image"),
        ("name", "twitter:image"),
        ("name", "twitter:image:src"),
    )
    image = image_candidate(image)

    # More publisher variants: <link rel=image_src>, lazy-loaded image attrs,
    # and srcset. Prefer a reasonably large article image over tiny icons.
    if not image:
        link_img = soup.find("link", attrs={"rel": re.compile(r"(^|\\s)image_src(\\s|$)", re.I)})
        image = image_candidate(link_img.get("href", "")) if link_img else ""
    if not image:
        imgs = []
        for tag in soup.find_all("img"):
            classes = " ".join(tag.get("class", []))
            marker = " ".join([
                str(tag.get("alt", "")), classes, str(tag.get("id", "")),
                str(tag.get("data-testid", ""))
            ]).lower()
            if any(x in marker for x in ("logo", "avatar", "icon", "author", "profile", "social")):
                continue
            candidates = [
                tag.get("src"), tag.get("data-src"), tag.get("data-original"),
                tag.get("data-lazy-src"), tag.get("data-image"), tag.get("data-url")
            ]
            srcset = tag.get("srcset") or tag.get("data-srcset")
            if srcset:
                # Usually the final/largest candidate is the best one.
                candidates.append(srcset.split(",")[-1].strip().split(" ")[0])
            for c in candidates:
                got = image_candidate(c)
                if got:
                    imgs.append(got)
                    break
        if imgs:
            image = imgs[0]

    if image and "news.google.com" in image.lower():
        image = ""

    author = ""
    author_data = jsonld.get("author", "")
    if isinstance(author_data, dict):
        author = clean(str(author_data.get("name", "")))
    elif isinstance(author_data, list):
        names = []
        for a in author_data:
            if isinstance(a, dict):
                names.append(clean(str(a.get("name", ""))))
            elif isinstance(a, str):
                names.append(clean(a))
        author = ", ".join(x for x in names if x)
    elif author_data:
        author = clean(str(author_data))

    published = clean(
        str(
            jsonld.get("datePublished", "")
            or jsonld.get("dateModified", "")
        )
    )

    return {
        "title": title,
        "description": description,
        "image": image,
        "author": author,
        "published": published,
        "jsonld": jsonld,
    }


def extract_article(html: str, url: str, method: str) -> dict:
    meta = extract_metadata(html, url)

    data = {}
    try:
        # Trafilatura 2.2 returns a Document object by default.
        # Convert it explicitly before accessing fields.
        doc = trafilatura.bare_extraction(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_links=False,
            favor_precision=True,
            favor_recall=True,
            with_metadata=True,
        )
        if doc is not None:
            if hasattr(doc, "as_dict"):
                data = doc.as_dict() or {}
            elif isinstance(doc, dict):
                data = doc
            else:
                data = {
                    "text": getattr(doc, "text", "") or "",
                    "title": getattr(doc, "title", "") or "",
                    "author": getattr(doc, "author", "") or "",
                    "date": getattr(doc, "date", "") or "",
                    "image": getattr(doc, "image", "") or "",
                }
    except Exception:
        data = {}

    # Plain-text Trafilatura fallback.
    if not data.get("text"):
        try:
            plain = trafilatura.extract(
                html, url=url, include_comments=False,
                include_tables=True, include_links=False,
                favor_precision=False, favor_recall=True,
                output_format="txt"
            )
            if plain:
                data["text"] = plain
        except Exception:
            pass

    text = clean(data.get("text", ""))
    title = clean(data.get("title", "")) or meta["title"]
    author = clean(data.get("author", "")) or meta["author"]
    published = clean(data.get("date", "")) or meta["published"]
    image = clean(data.get("image", "")) or meta["image"]

    # JSON-LD articleBody fallback.
    body = meta["jsonld"].get("articleBody")
    if isinstance(body, str) and len(body) > len(text):
        text = clean(body)
        method += "+jsonld"

    # Common publisher DOM fallback when structured extraction is short.
    if len(text.split()) < 120:
        try:
            soup = BeautifulSoup(html, "lxml")
            candidates = []
            selectors = (
                "article", "main", "[role='main']",
                "[itemprop='articleBody']", ".article-body",
                ".article-content", ".story-body", ".story-content",
                ".entry-content", ".post-content", ".article__body"
            )
            for selector in selectors:
                for node in soup.select(selector):
                    parts = []
                    for p in node.find_all(["p", "h2", "h3"]):
                        t = clean(p.get_text(" ", strip=True))
                        if 45 <= len(t) <= 3000:
                            parts.append(t)
                    if parts:
                        candidates.append("\n".join(parts))
            if candidates:
                dom_text = max(candidates, key=len)
                if len(dom_text) > len(text):
                    text = dom_text
                    method += "+dom"
        except Exception:
            pass

    paragraphs = []
    seen = set()
    junk_dropped = 0

    raw_text = data.get("text", "") or text

    for raw in re.split(r"\n+", raw_text):
        paragraph = clean(raw)

        if len(paragraph) < 40:
            continue

        if is_boilerplate(paragraph):
            junk_dropped += 1
            continue

        key = re.sub(r"[^a-z0-9]+", " ", paragraph.lower()).strip()

        if not key or key in seen:
            continue

        seen.add(key)
        paragraphs.append(paragraph)

    # Rebuild the plain-text body from the cleaned, deduplicated,
    # boilerplate-free paragraphs instead of returning the raw blob. Anything
    # that reads `text` (rather than `paragraphs`) then gets the same "proper
    # news" content, not leftover nav/ad/share-prompt lines that slipped past
    # the paragraph split.
    text = "\n\n".join(paragraphs) if paragraphs else text
    title = clean_title(title, url)

    words = len(text.split())

    # A practical quality score for deciding whether to use the fast result
    # or spend time rendering the page. A page that was mostly boilerplate
    # (lots of dropped junk lines relative to kept paragraphs) is penalized,
    # since that's a signal the real article body wasn't cleanly isolated.
    word_score = min(1.0, words / 900)
    paragraph_score = min(1.0, len(paragraphs) / 10)
    junk_ratio = junk_dropped / max(1, junk_dropped + len(paragraphs))
    quality = max(0.0, (0.65 * word_score + 0.35 * paragraph_score) - 0.4 * junk_ratio)

    description = meta["description"] or (paragraphs[0][:280] if paragraphs else "")

    return {
        "ok": bool(text),
        "url": url,
        "title": title,
        "author": author,
        "published": published,
        "image": image,
        "description": description,
        "text": text,
        "paragraphs": paragraphs,
        "word_count": words,
        "extraction_score": round(quality, 3),
        "method": method,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


async def _do_fetch(client: httpx.AsyncClient, url: str, headers: dict):
    async with client.stream("GET", url, headers=headers) as response:
        status = response.status_code
        if status in FETCH_RETRY_STATUSES:
            # Drain quickly and let the caller decide whether to retry;
            # raising here (via raise_for_status) would also work but we
            # want the status code available without re-parsing the exception.
            response.raise_for_status()

        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type and "xml" not in content_type:
            raise ValueError("Source response is not HTML")

        chunks = []
        total = 0

        async for chunk in response.aiter_bytes():
            total += len(chunk)

            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError("Source HTML is too large")

            chunks.append(chunk)

        html = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")

        return html, str(response.url)


async def fetch_html(url: str):
    """Fetch a page's HTML with a realistic header profile, bounded per-host
    concurrency, and one bounded retry for soft blocks (403/429/503).

    A 403/429/503 is very often not "this content doesn't exist" but "this
    specific request looked like a bot" — a stale/identifying UA, an
    overly-aggressive burst of concurrent requests to the same host, or a
    missing Referer. Retrying once, after backing off and swapping to a
    crawler-labelled UA, recovers a meaningful share of those without
    needing a full browser render.
    """
    client = await get_http_client()
    host = urlparse(url).hostname or ""
    semaphore = await get_host_semaphore(host)

    last_exc = None
    async with semaphore:
        for attempt in range(2):
            if attempt == 0:
                headers = browser_headers(url)
            else:
                await asyncio.sleep(0.6 + 0.4 * attempt)
                headers = browser_headers(url)
                headers["User-Agent"] = CRAWLER_FALLBACK_UA
                headers.pop("sec-ch-ua", None)
                headers.pop("sec-ch-ua-platform", None)
                headers.pop("sec-ch-ua-mobile", None)

            try:
                return await _do_fetch(client, url, headers)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code not in FETCH_RETRY_STATUSES:
                    raise
                # Loop again with the fallback profile.
                continue
            except httpx.RequestError as exc:
                last_exc = exc
                raise

        raise last_exc


async def fetch_rendered(url: str):
    """Optional browser fallback; Playwright is not installed in the free build."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is not installed in this lightweight deployment") from exc

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = await browser.new_page(
            user_agent=pick_browser_profile(url)["User-Agent"],
            viewport={"width": 1440, "height": 1800},
        )
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(1500)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.70)")
            await page.wait_for_timeout(900)
            return await page.content(), page.url
        finally:
            await browser.close()


async def extract_one(url: str, render: bool, max_chars: int):
    if not is_public_url(url):
        raise HTTPException(
            status_code=400,
            detail="Only public HTTP/HTTPS URLs are allowed.",
        )

    errors = []
    requested_url = url
    last_result = None

    # Google News RSS gives encoded /rss/articles/ URLs. They usually return a
    # Google interstitial to plain HTTP clients, so resolve them first.
    resolved_url, resolve_method = await resolve_google_news_url(url)
    url = resolved_url
    if resolve_method and resolve_method not in ("cache", "googlenewsdecoder"):
        errors.append("google-resolve:" + resolve_method)

    # FAST PATH: ordinary HTTP request.
    try:
        html, final_url = await fetch_html(url)

        result = extract_article(
            html,
            final_url,
            "http+trafilatura",
        )
        result["requested_url"] = requested_url
        result["resolved_url"] = final_url
        result["google_resolve"] = resolve_method
        last_result = result

        if (
            result["word_count"] >= MIN_GOOD_WORDS
            and result["extraction_score"] >= MIN_GOOD_SCORE
        ):
            result["text"] = result["text"][:max_chars]
            return result

    except httpx.HTTPStatusError as exc:
        # Record the actual status (e.g. "http:403") instead of the generic
        # exception class name, so blocked-vs-broken is visible in the logs
        # the extension surfaces to the user.
        errors.append(f"http:{exc.response.status_code}")
    except Exception as exc:
        errors.append("http:" + type(exc).__name__)

    # FALLBACK: render JavaScript-heavy pages.
    if render:
        try:
            html, final_url = await fetch_rendered(url)

            result = extract_article(
                html,
                final_url,
                "playwright+trafilatura",
            )

            if result["ok"]:
                result["text"] = result["text"][:max_chars]
                return result

        except Exception as exc:
            errors.append("render:" + type(exc).__name__)

    if last_result and last_result.get("image"):
        last_result["ok"] = False
        last_result["method"] = last_result.get("method", "failed") + "+low-quality"
        last_result["errors"] = errors
        last_result["text"] = last_result.get("text", "")[:max_chars]
        return last_result

    return {
        "ok": False,
        "url": requested_url,
        "requested_url": requested_url,
        "resolved_url": url,
        "google_resolve": resolve_method,
        "title": "",
        "author": "",
        "published": "",
        "image": "",
        "description": "",
        "text": "",
        "paragraphs": [],
        "word_count": 0,
        "extraction_score": 0,
        "method": "failed",
        "errors": errors,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/image")
async def proxy_image(url: str):
    """Fetch a publisher image server-side so hotlink/referrer blocking is less likely to blank cards."""
    if not is_public_url(url):
        raise HTTPException(status_code=400, detail="Only public HTTP/HTTPS image URLs are allowed.")
    headers = {
        "User-Agent": pick_browser_profile(url)["User-Agent"],
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": url,
    }
    try:
        client = await get_http_client()
        r = await client.get(url, headers=headers, timeout=httpx.Timeout(15.0, connect=8.0))
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").split(";", 1)[0].lower()
        if not ctype.startswith("image/"):
            raise HTTPException(status_code=415, detail="URL did not return an image")
        if len(r.content) > 8_000_000:
            raise HTTPException(status_code=413, detail="Image is too large")
        return Response(
            content=r.content,
            media_type=ctype,
            headers={"Cache-Control": "public, max-age=86400, stale-while-revalidate=604800"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image fetch failed: {type(exc).__name__}") from exc


@app.get("/")
async def root():
    return {
        "service": "NEWS BYTE Source Extractor",
        "version": "1.4.0",
        "ai": False,
        "usage": "POST /extract with {url, render, max_chars}",
    }


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "news-byte-source-extractor",
        "ai": False,
    }


@app.post("/extract")
async def extract_endpoint(request: ExtractRequest):
    return await extract_one(
        str(request.url),
        request.render,
        min(max(request.max_chars, 1000), 100000),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
