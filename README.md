# NEWS BYTE Source Extractor

Lightweight article extraction service with a hardened Google News resolver.

## Google News resolver

The resolver no longer depends on `googlenewsdecoder==0.1.7`. It uses the current
Google News `data-n-a-id` / `data-n-a-sg` / `data-n-a-ts` flow with bounded
retries, persistent HTTP/2 connection reuse, browser-like headers, rate control,
positive/negative caching, validation, a circuit breaker, and a conservative
legacy embedded-URL fallback.

Endpoints:
- `POST /extract` — extract one article.
- `POST /resolve-google` — resolve one Google News URL.
- `POST /resolve-google/batch` — resolve multiple URLs with bounded concurrency.
- `GET /health` — service health.

The resolver never returns a Google URL as a successful publisher URL. If
resolution fails, the response clearly reports the reason so a caller can choose
a different extraction path.

## v1.6.0 reliability changes
- Per-URL Google News resolution deadline (no global circuit breaker).
- Short upstream HTTP/RPC timeouts and bounded retrying.
- Negative cache is brief and does not poison unrelated Google News URLs.
- `/extract` has a hard 24-second server deadline to stay below typical 30-second clients.
- Publisher HTTP extraction gets its own bounded timeout.
- Optional rendered fallback is bounded to 7 seconds with shorter waits.


## Google News resolver v1.8.0

The resolver treats the Google News `batchexecute`/`garturlres` response as authoritative. Browser HTML is only a fallback. External XML namespaces (including `https://www.w3.org/XML/1998/namespace`), Google infrastructure, tracking hosts, and asset URLs are explicitly rejected so they can never become a publisher URL.


## Google News resolver 1.9.0

The Google News resolver uses the page-provided signed article token (`data-p` / `data-n-a-*`) and the `Fbv4je` `garturlreq` decoder. It parses `garturlres` specifically and fails closed on unrelated Google/XML/schema/asset URLs. Browser-TLS requests use `curl-cffi` Chrome/Safari impersonation when available. Resolver/network operations intentionally do not set an application-level timeout; the caller/platform remains responsible for its own request lifetime.
