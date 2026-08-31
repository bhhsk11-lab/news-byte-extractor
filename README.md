# NEWS BYTE Source Extractor 1.12.3

Google News is resolved before extraction. Server Chromium is the last-resort resolver with a 30-second Chromium-only resolution budget. Playwright navigation timeout remains 0; the resolver code controls the 20-second budget so redirect and DOM inspection can run continuously without a shorter Playwright navigation cutoff. If no publisher URL is resolved, `/extract` returns `resolved_url: null` and the raw Google URL must not be forwarded to the secondary/auth-bypass extractor. If a publisher URL is resolved but primary extraction is empty/low-quality, the real publisher URL is preserved so the extension can send only that URL to the secondary extractor.


## Request scheduling
`POST /extract` is strictly serialized with one in-process extraction lock. The next extraction starts only after the previous extraction has completed and its response object is ready. Google News Chromium fallback has a 30-second resolver budget; Playwright navigation timeout remains 0. Raw Google News URLs are never forwarded to the secondary/auth-bypass extractor.
