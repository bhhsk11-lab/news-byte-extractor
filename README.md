# NEWS BYTE Source Extractor 1.12.0

Google News is resolved before extraction. Server Chromium is the unlimited last-resort resolver: Playwright navigation timeout is 0, there is no application-level Google resolver deadline and no poll-count deadline. If no publisher URL is resolved, `/extract` returns `resolved_url: null` and the raw Google URL must not be forwarded to the secondary/auth-bypass extractor. If a publisher URL is resolved but primary extraction is empty/low-quality, the real publisher URL is preserved so the extension can send only that URL to the secondary extractor.
