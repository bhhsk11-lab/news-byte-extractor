---
title: NEWS BYTE Source Extractor
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# NEWS BYTE Source Extractor

Non-AI article extraction server for the NEWS BYTE browser extension.

## Endpoint

POST `/extract`

Body:

```json
{
  "url": "https://example.com/news/article",
  "render": true,
  "max_chars": 60000
}
```

The server first uses the fast HTTP + Trafilatura path.
If extraction is too short/poor, it can render the page with Chromium/Playwright
and run extraction again.

It returns the source-derived article text, paragraphs, title, image, author,
publication date and an extraction quality score.

This service does not generate or invent article facts.
