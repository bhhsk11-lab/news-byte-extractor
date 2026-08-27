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
    version="1.3.0",
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
MIN_GOOD_WORDS = 120
MIN_GOOD_SCORE = 0.30

# Google News RSS article links are encoded Google redirect URLs, not publisher
# article URLs. Keep a small in-process cache to avoid resolving the same link
# repeatedly during a feed refresh.
GOOGLE_RESOLVE_CACHE = {}
GOOGLE_RESOLVE_LOCK = asyncio.Lock()
GOOGLE_RESOLVE_CACHE_MAX = 500


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

    raw_text = data.get("text", "") or text

    for raw in re.split(r"\n+", raw_text):
        paragraph = clean(raw)

        if len(paragraph) < 40:
            continue

        key = re.sub(r"[^a-z0-9]+", " ", paragraph.lower()).strip()

        if not key or key in seen:
            continue

        seen.add(key)
        paragraphs.append(paragraph)

    words = len(text.split())

    # A practical quality score for deciding whether to use the fast result
    # or spend time rendering the page.
    word_score = min(1.0, words / 900)
    paragraph_score = min(1.0, len(paragraphs) / 10)
    quality = 0.65 * word_score + 0.35 * paragraph_score

    return {
        "ok": bool(text),
        "url": url,
        "title": title,
        "author": author,
        "published": published,
        "image": image,
        "description": meta["description"],
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

    timeout = httpx.Timeout(20.0, connect=10.0)

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
        "version": "1.3.0",
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
