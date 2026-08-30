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