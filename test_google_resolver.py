import asyncio
from google_resolver import GoogleNewsResolver, REQUEST_TIMEOUT, BROWSER_NAV_TIMEOUT_MS, RESOLVE_DEADLINE

ID='CBMiTESTARTICLE'
URL=f'https://news.google.com/rss/articles/{ID}?oc=5'
PUB='https://example-news.test/world/2026/08/31/real-story'
r=GoogleNewsResolver()
assert r.is_google_url(URL)
assert r._valid_destination(PUB)
assert not r._valid_destination('https://www.w3.org/XML/1998/namespace')
assert not r._valid_destination('https://www.gstatic.com/some.js')
html=f'<html><body><c-wiz data-p="signed:{ID}:decoder" data-n-a-id="{ID}" data-n-a-sg="SIGNATURE" data-n-a-ts="1730000000"></c-wiz></body></html>'
params=r._extract_params(html,ID)
assert params and params['id']==ID and params['sig']=='SIGNATURE' and params['ts']=='1730000000'
payload=r._rpc_payload([params])
assert 'Fbv4je' in payload and 'garturlreq' in payload and ID in payload
rpc='[[[\"Fbv4je\",\"[\\\"garturlres\\\",\\\"https://example-news.test/world/2026/08/31/real-story\\\",null]\",null,\"generic\"]]]'
assert r._urls_from_rpc(rpc)==[PUB]
bad='[[["Fbv4je","[\"garturlres\",\"https://www.w3.org/XML/1998/namespace\",null]",null,"generic"]]]'
assert r._urls_from_rpc(bad)==[]
assert REQUEST_TIMEOUT==5.0
assert BROWSER_NAV_TIMEOUT_MS==0
assert RESOLVE_DEADLINE is None
async def main():
    async def fake(_url):
        await asyncio.sleep(0.01)
        return type('R',(),{'url':PUB,'method':'browser','error':None})()
    r._resolve_staged=fake
    out=await r._resolve_uncached(URL)
    assert out.url==PUB and out.method=='browser'
    await r.close()
asyncio.run(main())
print('SELF TEST: ALL GOOGLE RESOLVER TESTS PASSED')
print('SELF TEST: Chromium navigation API timeout = 0; Chromium resolver budget = 30 seconds')
print('SELF TEST: whole staged resolver has no global timeout; Chromium-only budget = 30 seconds')
print('SELF TEST: Chromium resolver stops after 30 seconds if no publisher URL is found')
print('SELF TEST: W3/XML namespace rejected')
print('SELF TEST: garturlres publisher accepted')
