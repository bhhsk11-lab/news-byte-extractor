import asyncio
import ipaddress
import json
import re
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import httpx
import trafilatura
from google_resolver import resolver as google_resolver

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

app = FastAPI(
    title="NEWS BYTE Source Extractor",
    description="Non-AI source article + site-structure extraction service for NEWS BYTE.",
    version="1.11.0",
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

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Safari/537.36 NEWS-BYTE/1.0"
)

MAX_DOWNLOAD_BYTES = 8_000_000
EXTRACT_DEADLINE = 0  # retained for compatibility; /extract does not abort the request
MIN_GOOD_WORDS = 120
MIN_GOOD_SCORE = 0.30

# Google News RSS article links are encoded Google redirect URLs, not publisher
# article URLs. Keep a small in-process cache to avoid resolving the same link
# repeatedly during a feed refresh.
async def resolve_google_news_url(url: str):
    """Resolve Google News URLs through the hardened resolver."""
    result = await google_resolver.resolve(str(url))
    return result.url, result.method if result.method != "failed" else "failed", result.error


class ExtractRequest(BaseModel):
    url: HttpUrl
    render: bool = False
    max_chars: int = 60000


class ExploreRequest(BaseModel):
    url: HttpUrl
    max_pages: int = 24
    max_depth: int = 1
    concurrency: int = 8


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
            if not value:
                return ""
            if value.startswith("//"):
                value = "https:" + value
            # Resolve relative to the page URL first (govt/coaching sites often
            # publish og:image as a root-relative path like "/img/hero.jpg"),
            # then apply the same absolute-URL + Google-News filtering as before.
            resolved = urljoin(url, value)
            if re.match(r"^https?://", resolved, re.I) and "news.google.com" not in resolved.lower():
                return resolved
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

    # BUG FIX: this used to re-read `data.get("text", "")` here -- the RAW,
    # pre-enhancement trafilatura output -- which silently discarded the
    # JSON-LD articleBody swap (line ~436) and the DOM-selector fallback
    # (line ~463) whenever trafilatura's own extraction returned ANY non-empty
    # text (which is most of the time, even when that text is a thin, wrong
    # teaser). `text` at this point already holds whichever candidate was
    # actually longest/best; that's what must be paragraph-split, not the
    # stale raw trafilatura output. Confirmed via a deterministic test with
    # trafilatura's output mocked short and a DOM article-body present: before
    # this fix, method correctly reported "+dom" while word_count/text still
    # only reflected the discarded 6-word trafilatura teaser.
    raw_text = text

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


async def fetch_html(url: str):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    timeout = httpx.Timeout(10.0, connect=4.0, read=8.0, write=8.0, pool=3.0)

    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        max_redirects=5,
        timeout=timeout,
    ) as client:
        async with client.stream("GET", url) as response:
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
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 1800},
        )
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=6500)
            await page.wait_for_timeout(500)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.70)")
            await page.wait_for_timeout(300)
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
    resolved_url, resolve_method, resolve_error = await resolve_google_news_url(url)
    url = resolved_url
    if resolve_method in ("failed", "timeout"):
        errors.append("google-resolve:" + (resolve_error or resolve_method or "failed"))
    elif resolve_method and resolve_method not in ("cache", "passthrough"):
        errors.append("google-resolve:" + resolve_method)
        # If this was a Google News link and we still only have the raw
        # news.google.com redirect (not a publisher URL), fetch_html() below
        # is guaranteed to fail on it too (Google serves an interstitial to
        # plain HTTP clients) — that's the http:HTTPStatusError that always
        # rides along with google-resolve:decoder-failed in the logs. Don't
        # spend another ~10-20s httpx timeout finding that out; fail fast so
        # the extension can fail over to the second extractor sooner.
        if google_resolver.is_google_url(url):
            return {
                "ok": False,
                "url": requested_url,
                "requested_url": requested_url,
                "resolved_url": url,
                "google_resolve": resolve_method,
                "google_resolve_error": resolve_error,
                "title": "",
                "author": "",
                "published": "",
                "image": "",
                "description": "",
                "text": "",
                "paragraphs": [],
                "word_count": 0,
                "extraction_score": 0,
                "method": "google-resolve-failed",
                "fallback_required": True,
                "errors": errors,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

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
        result["google_resolve_error"] = resolve_error
        result["fallback_required"] = False
        last_result = result

        if (
            result["word_count"] >= MIN_GOOD_WORDS
            and result["extraction_score"] >= MIN_GOOD_SCORE
        ):
            result["text"] = result["text"][:max_chars]
            return result

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
                result["fallback_required"] = False
                result["text"] = result["text"][:max_chars]
                return result

        except Exception as exc:
            errors.append("render:" + type(exc).__name__)

    # Preserve any non-trivial extraction even when it misses the strict
    # quality threshold. Returning the text is much safer than converting a
    # real short article into an EMPTY/0-word result. The extension can decide
    # whether to use it for a briefing.
    if last_result and len((last_result.get("text") or "").split()) >= 25:
        last_result["ok"] = True
        last_result["method"] = last_result.get("method", "failed") + "+low-quality"
        last_result["errors"] = errors
        last_result["fallback_required"] = True
        last_result["text"] = last_result.get("text", "")[:max_chars]
        return last_result

    return {
        "ok": False,
        "url": requested_url,
        "requested_url": requested_url,
        "resolved_url": url,
        "google_resolve": resolve_method,
        "google_resolve_error": resolve_error,
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
        "fallback_required": bool(google_resolver.is_google_url(requested_url) or url != requested_url),
        "errors": errors,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Explore / Coaching: deep structured extraction + same-domain crawl.
#
# This is a server-side port of the extension's old client-side
# pageStructuredExtractor()/crawlSite() (background.js). Moving it here means
# the extension never has to open a background tab or juggle a dozen
# concurrent fetches through an offscreen document to "explore" a site (govt
# sites, coaching sites, anything) — it just POSTs a URL and gets back a
# heading -> paragraphs/bullets section tree plus classified same-domain
# links (PDFs, books, magazines, pagination, tags, categories) and media.
# ---------------------------------------------------------------------------

_BAD_RX = re.compile(
    r"(^|[-_ ])(ad|ads|advert|advertisement|banner|cookie|consent|subscribe|newsletter|"
    r"nav|navbar|menu|footer|header|sidebar|related|recommended|comments?|social|share|"
    r"promo|modal|popup|paywall|login|register|breadcrumb|utility|toolbar|app-promo|"
    r"download-app)([-_ ]|$)",
    re.I,
)


def _cls_id(tag) -> str:
    try:
        classes = " ".join(tag.get("class") or [])
    except Exception:
        classes = ""
    return f"{tag.get('id','')} {classes}"


def _is_boilerplate_tag(tag) -> bool:
    return bool(_BAD_RX.search(_cls_id(tag)))


def root_domain(host: str) -> str:
    parts = [p for p in (host or "").lower().split(".") if p]
    return ".".join(parts[-2:]) if len(parts) > 2 else ".".join(parts)


def is_same_site(href: str, base_host: str, base_root: str) -> bool:
    try:
        u = urlparse(href)
        if u.scheme not in ("http", "https"):
            return False
        host = (u.hostname or "").lower()
        return host == base_host or host.endswith("." + base_root)
    except Exception:
        return False


def build_structured_page(html: str, url: str) -> dict:
    """Turn a raw HTML page into a heading -> content section tree, same
    shape the extension's coaching.js already knows how to render."""
    soup = BeautifulSoup(html, "lxml")
    parsed = urlparse(url)
    base_host = (parsed.hostname or "").lower()
    base_root = root_domain(base_host)

    for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    meta = extract_metadata(html, url)

    sections: list = []
    stack: dict = {}

    def new_section(text: str, level: int) -> dict:
        s = {"title": text, "level": level, "paragraphs": [], "bullets": []}
        sections.append(s)
        stack[level] = s
        for lv in [lv for lv in stack if lv > level]:
            del stack[lv]
        return s

    def current_section():
        for lv in (4, 3, 2, 1):
            if lv in stack:
                return stack[lv]
        return None

    links, pdf_links, book_links, magazine_links = [], [], [], []
    tag_links, category_links, pagination_links = [], [], []
    seen_hrefs = set()
    media, seen_media = [], set()
    current_media_section = "Other"

    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "a", "img"]):
        name = el.name

        if name in ("h1", "h2", "h3", "h4"):
            text = clean(el.get_text(" ", strip=True))
            if not text:
                continue
            new_section(text, int(name[1]))
            current_media_section = text
            continue

        if name in ("p", "li", "blockquote"):
            if _is_boilerplate_tag(el):
                continue
            text = clean(el.get_text(" ", strip=True))
            if not text:
                continue
            sec = current_section()
            if not sec:
                continue
            if name == "li":
                if len(text) >= 3:
                    sec["bullets"].append(text)
            elif len(text) >= 25:
                sec["paragraphs"].append(text)
            continue

        if name == "img":
            src = el.get("src") or el.get("data-src") or el.get("data-lazy-src") or ""
            if not src:
                continue
            src = urljoin(url, src)
            if not re.match(r"^https?://", src) or src in seen_media:
                continue
            alt = clean(el.get("alt", "") or "")
            fig = el.find_parent("figure")
            caption = ""
            if fig is not None:
                cap_tag = fig.find("figcaption")
                if cap_tag:
                    caption = clean(cap_tag.get_text(" ", strip=True))
            hint = f"{alt} {caption} {' '.join(el.get('class') or [])}".lower()
            kind = "map" if re.search(r"\b(map|gis|location|route|roadmap|political map|india map)\b", hint) else "image"
            seen_media.add(src)
            media.append({"src": src, "alt": alt, "caption": caption, "kind": kind, "section": current_media_section})
            continue

        if name == "a":
            href = el.get("href") or ""
            if not href:
                continue
            href = urljoin(url, href)
            if not is_same_site(href, base_host, base_root):
                continue
            if _is_boilerplate_tag(el):
                continue
            link_title = (
                clean(el.get_text(" ", strip=True))
                or clean(el.get("aria-label", "") or "")
                or clean(el.get("title", "") or "")
            )
            if not link_title or len(link_title) > 160 or href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            sec = current_section()
            item = {"href": href, "title": link_title, "section": sec["title"] if sec else "Other useful links"}
            path = urlparse(href).path.lower()
            path_title = f"{path} {link_title}".lower()

            if re.search(r"\.pdf(?:$|[?#])", href, re.I) or re.search(r"/pdf(?:/|$)", path):
                pdf_links.append(item)
            if re.search(r"\b(book|books|ebook|e-book)\b|/books?/", path_title):
                book_links.append(item)
            if re.search(r"\b(magazine|magazines|monthly|edition)\b|/magazines?/", path_title):
                magazine_links.append(item)
            if re.search(r"/tags?/", path):
                tag_links.append(item)
            if re.search(r"/category|/categories|/subjects?|/topics?|/section", path):
                category_links.append(item)
            if (
                re.search(r"\b(next|previous|prev|older|newer)\b", path_title)
                or re.search(r"[?&](page|paged)=\d+", href, re.I)
                or re.search(r"/page/\d+", path)
            ):
                pagination_links.append(item)

            articleish = bool(re.search(
                r"/(daily-updates|current-affairs|news|editorial|article|articles|study|notes|"
                r"courses|analysis|magazine|books?|topics?|subjects?|blog|upsc|ias|exam)",
                path, re.I,
            )) or len(link_title.split()) >= 4
            if articleish and not re.search(r"/(login|signup|register|contact|privacy|terms|careers|about|search)\b", path, re.I):
                links.append(item)

    sections = [s for s in sections if s["paragraphs"] or s["bullets"]]
    if not sections:
        paras = []
        for p in soup.find_all("p"):
            if _is_boilerplate_tag(p):
                continue
            t = clean(p.get_text(" ", strip=True))
            if len(t) >= 35:
                paras.append(t)
        if paras:
            sections = [{"title": meta["title"] or "Page", "level": 1, "paragraphs": paras, "bullets": []}]

    return {
        "ok": True,
        "url": url,
        "pageTitle": meta["title"],
        "description": meta["description"],
        "author": meta["author"],
        "date": meta["published"],
        "sections": sections[:200],
        "links": links[:240],
        "pdfLinks": pdf_links[:80],
        "bookLinks": book_links[:60],
        "magazineLinks": magazine_links[:60],
        "tagLinks": tag_links[:80],
        "categoryLinks": category_links[:100],
        "paginationLinks": pagination_links[:40],
        "media": media[:40],
        "heroImage": meta["image"],
    }


async def crawl_site(start_url: str, max_pages: int, max_depth: int, concurrency: int) -> dict:
    if not is_public_url(start_url):
        raise HTTPException(status_code=400, detail="Only public HTTP/HTTPS URLs are allowed.")

    max_pages = max(1, min(100, max_pages))
    max_depth = max(0, min(5, max_depth))
    concurrency = max(1, min(12, concurrency))
    sem = asyncio.Semaphore(concurrency)

    async def fetch_one(u: str):
        async with sem:
            try:
                html, final_url = await fetch_html(u)
                return build_structured_page(html, final_url)
            except Exception:
                return None

    seen: set = set()
    all_links: list = []
    first: dict | None = None
    frontier = [start_url]
    depth = 0

    while frontier and len(seen) < max_pages and depth <= max_depth:
        batch = []
        for u in frontier:
            if u not in seen and len(seen) + len(batch) < max_pages:
                batch.append(u)
        if not batch:
            break
        seen.update(batch)

        results = await asyncio.gather(*[fetch_one(u) for u in batch])
        next_frontier: dict = {}
        for u, data in zip(batch, results):
            if not data or not data.get("ok"):
                continue
            if first is None:
                first = data
            for link in data.get("links", []):
                all_links.append({**link, "depth": depth})
            if depth < max_depth:
                base_host = (urlparse(u).hostname or "").lower()
                root = root_domain(base_host)
                for link in data.get("links", []):
                    href = link["href"]
                    host = (urlparse(href).hostname or "").lower()
                    if (host == base_host or host.endswith("." + root)) and href not in seen:
                        next_frontier[href] = True
                for link in (data.get("paginationLinks") or [])[:4]:
                    href = link["href"]
                    if href not in seen:
                        next_frontier[href] = True

        remaining = max(0, (max_pages - len(seen)) * 2)
        frontier = list(next_frontier.keys())[:remaining]
        depth += 1

    if first is None:
        return {
            "ok": False,
            "error": "No public pages could be extracted. The site may require sign-in or block automated reading.",
            "url": start_url,
        }

    dedup: dict = {}
    for link in all_links:
        dedup.setdefault(link["href"], link)

    result = dict(first)
    result["links"] = list(dedup.values())[:1000]
    result["crawlPages"] = len(seen)
    result["crawledUrls"] = list(seen)
    return result


@app.post("/explore")
async def explore_endpoint(request: ExploreRequest):
    """Structured single-page read (max_pages=1, max_depth=0) or a full
    same-domain crawl ("Explore domain" in Coaching) — same endpoint, the
    extension just varies max_pages/max_depth."""
    return await crawl_site(
        str(request.url),
        max_pages=request.max_pages,
        max_depth=request.max_depth,
        concurrency=request.concurrency,
    )


@app.get("/image")
async def proxy_image(url: str):
    """Fetch a publisher image server-side so hotlink/referrer blocking is less likely to blank cards."""
    if not is_public_url(url):
        raise HTTPException(status_code=400, detail="Only public HTTP/HTTPS image URLs are allowed.")
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": url,
    }
    try:
        timeout = httpx.Timeout(15.0, connect=8.0)
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, max_redirects=5, timeout=timeout) as client:
            r = await client.get(url)
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
        "version": "1.10.0",
        "ai": False,
        "usage": {
            "extract": "POST /extract with {url, render, max_chars} — single article, flat text.",
            "explore": "POST /explore with {url, max_pages, max_depth, concurrency} — structured "
                       "section tree + classified links for any site (news, coaching, govt). "
                       "max_pages=1,max_depth=0 reads just one page; higher values crawl the domain.",
        },
    }


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "news-byte-source-extractor",
        "version": "1.11.0",
        "ai": False,
        "google_resolver": "rpc+independent-chromium",
    }


@app.post("/resolve-google")
async def resolve_google_endpoint(request: dict):
    url = str(request.get("url", "")).strip()
    if not url or not google_resolver.is_google_url(url):
        return {"ok": True, "url": url, "resolved_url": url, "method": "passthrough"}
    result = await google_resolver.resolve(url)
    return {"ok": result.method != "failed", "url": url, "resolved_url": result.url,
            "method": result.method, "error": result.error}


@app.post("/resolve-google/batch")
async def resolve_google_batch_endpoint(request: dict):
    urls = request.get("urls") or []
    if not isinstance(urls, list) or len(urls) > 50:
        raise HTTPException(400, "urls must be a list of at most 50 URLs")
    results = await google_resolver.resolve_many([str(x) for x in urls])
    return {"results": [r.__dict__ for r in results]}


@app.post("/extract")
async def extract_endpoint(request: ExtractRequest):
    # No artificial whole-request abort. Google News resolution may need the
    # independent Chromium resolver, and the caller should receive the real
    # result rather than a synthetic timeout/empty article.
    return await extract_one(
        str(request.url),
        request.render,
        min(max(request.max_chars, 1000), 100000),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


@app.on_event("shutdown")
async def shutdown_google_resolver():
    await google_resolver.close()
