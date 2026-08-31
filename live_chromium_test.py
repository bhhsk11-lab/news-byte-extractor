"""Live Google News -> publisher test. Run inside the Docker image.
Usage: python live_chromium_test.py <google-news-url> [<url> ...]
No application-level timeout is used. Stop with Ctrl-C.
"""
import asyncio
import sys
from google_resolver import GoogleNewsResolver

async def main(urls):
    r = GoogleNewsResolver()
    try:
        for url in urls:
            if not r.is_google_url(url):
                print(f"NOT_GOOGLE: {url}")
                continue
            print(f"TEST: {url}")
            result = await r._browser_resolve(url)
            print(f"METHOD: {result.method}")
            print(f"PUBLISHER_URL: {result.url}")
            print(f"ERROR: {result.error or ''}")
            print(f"VALID: {r._valid_destination(result.url)}")
            print("---")
    finally:
        await r.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Provide one or more Google News RSS article URLs")
    asyncio.run(main(sys.argv[1:]))
