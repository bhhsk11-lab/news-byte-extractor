# NEWS BYTE Source Extractor v1.11.0

Non-AI article extraction service with a dedicated Google News publisher URL resolver.

## Google News resolver

For `news.google.com/rss/articles/...`, the server starts two independent resolution paths concurrently:

1. Google signed `Fbv4je` / `garturlreq` HTTP/RPC decoding.
2. Dedicated Playwright Chromium.

### Chromium is unlimited

The Chromium resolver has **no application-level wall-clock deadline**, no polling-count limit, and no Playwright navigation timeout. It uses `timeout=0` and continues until:

- Chromium naturally reaches a validated publisher URL; or
- Chromium reports a real browser/navigation/runtime error; or
- the caller/process explicitly cancels the operation.

There is no `asyncio.wait_for()` / `asyncio.timeout()` around Google resolution.

The HTTP/RPC path still has finite per-network-leg timeouts so a dead HTTP connection cannot prevent the independent Chromium resolver from running. Both paths start together; the first validated publisher URL wins.

## Extraction fallback contract

When the Google URL resolves successfully, `/extract` uses the **publisher URL** for normal extraction.

If publisher extraction is empty or fails, the response preserves:

```json
"resolved_url": "https://publisher.example/news/story"
```

and sets:

```json
"fallback_required": true
```

The extension should then send **`resolved_url`** to the auth-bypass scraper. It should not send the original Google News URL in this case.

If Google resolution itself fails, `resolved_url` remains the original Google URL and `fallback_required` is true; this is the only case where the original Google URL is the last-resort fallback.

## Strict URL validation

The resolver rejects Google infrastructure, W3/XML namespaces, schema URLs, tracking/asset URLs, and other non-article destinations. Browser HTML is only used for high-confidence canonical/OG/JSON-LD/article-anchor candidates.

## Endpoints

- `POST /extract` — Google resolve + article extraction.
- `POST /resolve-google` — resolve one Google News URL.
- `POST /resolve-google/batch` — resolve up to 50 Google News URLs.
- `GET /health` — health/version information.

## Self-test

`test_google_resolver.py` verifies:

- Google News URL recognition.
- Signed parameter extraction.
- `garturlres` parsing.
- Strict publisher URL validation.
- W3/XML/schema rejection.
- Concurrent HTTP/Chromium resolution.
- Chromium navigation timeout is `0`.
- No resolver wall-clock deadline.
- No Chromium polling-count deadline.

`live_chromium_test.py` performs a real Chromium-only test and prints the final publisher URL. Run it inside the Docker image because the Dockerfile installs Playwright Chromium.
