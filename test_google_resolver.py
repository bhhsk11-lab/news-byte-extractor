import asyncio
from google_resolver import GoogleNewsResolver, REQUEST_TIMEOUT, BROWSER_NAV_TIMEOUT_MS, BROWSER_POLLS

ID='CBMiTESTARTICLE'
URL=f'https://news.google.com/rss/articles/{ID}?oc=5'
PUB='https://example-news.test/world/2026/08/31/real-story'

r=GoogleNewsResolver()

assert r.is_google_url(URL)
assert r._valid_destination(PUB)
assert not r._valid_destination('https://www.w3.org/XML/1998/namespace')
assert not r._valid_destination('https://www.gstatic.com/some.js')
assert not r._valid_destination('https://news.google.com/rss/articles/OTHER')
assert not r._valid_destination('https://schema.org/NewsArticle')

html=f'''<html><body><c-wiz data-p="signed:{ID}:decoder" data-n-a-id="{ID}" data-n-a-sg="SIGNATURE" data-n-a-ts="1730000000"></c-wiz></body></html>'''
params=r._extract_params(html, ID)
assert params and params['id']==ID and params['sig']=='SIGNATURE' and params['ts']=='1730000000'

# Related-story token must not be accepted.
related=html.replace(ID, 'OTHERARTICLE')
assert r._extract_params(related, ID) is None

payload=r._rpc_payload([params])
assert 'Fbv4je' in payload and 'garturlreq' in payload and ID in payload

rpc='[[["Fbv4je","[\\"garturlres\\",\\"https://example-news.test/world/2026/08/31/real-story\\",null]",null,"generic"]]]'
urls=r._urls_from_rpc(rpc)
assert urls == [PUB], urls

bad='[[["Fbv4je","[\\"garturlres\\",\\"https://www.w3.org/XML/1998/namespace\\",null]",null,"generic"]]]'
assert r._urls_from_rpc(bad) == []

assert REQUEST_TIMEOUT == 5.0
assert BROWSER_NAV_TIMEOUT_MS == 6000
assert BROWSER_POLLS == 18

async def main():
    result=await r.resolve('https://example.com/article')
    assert result.url == 'https://example.com/article' and result.method == 'passthrough'

    # Deadline must produce a graceful result rather than raising/cancelling.
    async def slow(_url):
        await asyncio.sleep(0.05)
        raise RuntimeError('simulated-google-stall')
    r._resolve_http = slow
    r._browser_resolve = lambda _url: asyncio.sleep(0.05, result=type('R', (), {'url': URL, 'method':'browser-timeout', 'error':'simulated'})())
    timed = await r._resolve_uncached(URL)
    assert timed.method in ('failed', 'timeout')
    assert timed.url == URL
    await r.close()

asyncio.run(main())
print('SELF TEST: ALL GOOGLE RESOLVER TESTS PASSED')
print('SELF TEST: no resolver request timeout')
print('SELF TEST: no Playwright navigation timeout')
print('SELF TEST: W3/XML namespace rejected')
print('SELF TEST: related Google article token rejected')
print('SELF TEST: garturlres publisher accepted')
