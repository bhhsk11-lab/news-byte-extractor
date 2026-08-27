# NEWS BYTE Render Extractor 1.2

Fixes:
- Trafilatura 2.2 `Document` handling.
- Google News RSS `/rss/articles/...` links are resolved to the real publisher URL before extraction.
- Resolution results are cached to reduce duplicate Google requests.
- Extraction responses include `requested_url`, `resolved_url`, and `google_resolve`.

Extraction cascade:
1. Google News URL resolution when needed
2. HTTP + Trafilatura
3. JSON-LD / DOM fallbacks
4. Optional browser rendering when `render=true`

Start:
`uvicorn app:app --host 0.0.0.0 --port $PORT`
