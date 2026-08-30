# NEWS BYTE Source Extractor

A lightweight, non‑AI extraction service for news and educational sites.

## Features

- **Article extraction** – returns clean title, text, metadata, and a quality score.
- **Google News resolution** – converts `/rss/articles/...` redirects to real publisher URLs using offline decoding and batchexecute RPC.
- **Structured “Explore”** – crawls a domain and returns heading‑based sections, links (PDFs, books, categories, tags), and media.
- **Proxy support** – optional `GNEWS_PROXY_URL` to avoid Google rate‑limiting.

## Endpoints

| Method | Path       | Description |
|--------|------------|-------------|
| POST   | `/extract` | Extract a single article (JSON body: `{url, render, max_chars}`). |
| POST   | `/explore` | Structured page/crawl (JSON body: `{url, max_pages, max_depth, concurrency}`). |
| GET    | `/image`   | Proxy an image with browser‑like headers. |
| GET    | `/health`  | Health check. |

## Running

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
### Google News resolver fallback chain

The resolver now combines the supplied approaches: offline ID decoding, Google HTML payload inspection, Playwright Chromium navigation (including consent handling), and the existing batchexecute RPC fallback. Playwright is installed by the Docker image and its browser pool is created lazily only when a Google News URL needs browser resolution.

Optional environment variables:
- `GNEWS_PROXY_URL` – HTTP/HTTPS proxy used for Google requests.
- `GNEWS_BROWSER_POOL_SIZE` – Playwright browser pool size (default `3`).
- `GNEWS_PLAYWRIGHT_TIMEOUT_MS` – navigation timeout (default `12000`).
- `GNEWS_REDIRECT_WAIT_MS` – browser redirect wait (default `5000`).
