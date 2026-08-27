import asyncio
import ipaddress
import json
import re
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

app = FastAPI(
    title="NEWS BYTE Source Extractor",
    description="Non-AI source article extraction service for NEWS BYTE.",
    version="1.1.0",
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

    image = meta(
        ("property", "og:image"),
        ("name", "twitter:image"),
        ("name", "twitter:image:src"),
    )

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

    # FAST PATH: ordinary HTTP request.
    try:
        html, final_url = await fetch_html(url)

        result = extract_article(
            html,
            final_url,
            "http+trafilatura",
        )

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

    return {
        "ok": False,
        "url": url,
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


@app.get("/")
async def root():
    return {
        "service": "NEWS BYTE Source Extractor",
        "version": "1.0.0",
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
