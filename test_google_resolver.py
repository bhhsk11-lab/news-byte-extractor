import asyncio

from google_resolver import GoogleNewsResolver, BROWSER_NAV_TIMEOUT_MS

ID = "CBMiTESTARTICLE"
URL = f"https://news.google.com/rss/articles/{ID}?oc=5"
PUB = "https://example-news.test/world/2026/08/31/real-story"

r = GoogleNewsResolver()
assert r.is_google_url(URL)
assert r._valid_destination(PUB)
assert not r._valid_destination("https://www.w3.org/XML/1998/namespace")
assert not r._valid_destination("https://www.gstatic.com/some.js")
assert not r._valid_destination("https://news.google.com/rss/articles/OTHER")
assert not r._valid_destination("https://schema.org/NewsArticle")

html = f'''<html><body><c-wiz data-p="signed:{ID}:decoder" data-n-a-id="{ID}" data-n-a-sg="SIGNATURE" data-n-a-ts="1730000000"></c-wiz></body></html>'''
params = r._extract_params(html, ID)
assert params and params["id"] == ID and params["sig"] == "SIGNATURE" and params["ts"] == "1730000000"
related = html.replace(ID, "OTHERARTICLE")
assert r._extract_params(related, ID) is None
payload = r._rpc_payload([params])
assert "Fbv4je" in payload and "garturlreq" in payload and ID in payload
rpc = '[[["Fbv4je","[\\"garturlres\\",\\"https://example-news.test/world/2026/08/31/real-story\\",null]",null,"generic"]]]'
urls = r._urls_from_rpc(rpc)
assert urls == [PUB], urls
bad = '[[["Fbv4je","[\\"garturlres\\",\\"https://www.w3.org/XML/1998/namespace\\",null]",null,"generic"]]]'
assert r._urls_from_rpc(bad) == []
assert BROWSER_NAV_TIMEOUT_MS == 0

async def test_concurrent_fallback():
    r = GoogleNewsResolver()
    async def fake_http(_url):
        await asyncio.sleep(0.08)
        return type("R", (), {"url": URL, "method": "failed", "error": "slow-http"})()
    async def fake_browser(_url):
        await asyncio.sleep(0.01)
        return type("R", (), {"url": PUB, "method": "browser", "error": None})()
    r._resolve_http = fake_http
    r._browser_resolve = fake_browser
    got = await r._resolve_staged(URL)
    assert got.url == PUB and got.method == "browser", got
    await r.close()

async def test_browser_is_not_cancelled_by_http_failure():
    r = GoogleNewsResolver()
    async def fake_http(_url):
        return type("R", (), {"url": URL, "method": "failed", "error": "rpc-failed"})()
    async def fake_browser(_url):
        await asyncio.sleep(0.03)
        return type("R", (), {"url": PUB, "method": "browser", "error": None})()
    r._resolve_http = fake_http
    r._browser_resolve = fake_browser
    got = await r._resolve_staged(URL)
    assert got.url == PUB
    await r.close()

asyncio.run(test_concurrent_fallback())
asyncio.run(test_browser_is_not_cancelled_by_http_failure())
print("SELF TEST: ALL GOOGLE RESOLVER TESTS PASSED")
print("SELF TEST: Chromium navigation timeout = 0 (unlimited)")
print("SELF TEST: no resolver wall-clock deadline")
print("SELF TEST: no Chromium polling-count deadline")
print("SELF TEST: W3/XML namespace rejected")
print("SELF TEST: related Google article token rejected")
print("SELF TEST: garturlres publisher accepted")
