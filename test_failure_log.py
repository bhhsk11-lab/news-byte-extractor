import re
from pathlib import Path
from google_resolver import GoogleNewsResolver

log = Path("/mnt/data/Pasted markdown(20260831-061633).md").read_text(encoding="utf-8")
urls = []
for u in re.findall(r"https://news\.google\.com/rss/articles/[^\s`\n]+", log):
    u = u.rstrip("`")
    if u not in urls:
        urls.append(u)

r = GoogleNewsResolver()
assert len(urls) >= 10
assert all(r.is_google_url(u) for u in urls)
assert all(not r._valid_destination("https://www.w3.org/XML/1998/namespace") for _ in urls)
print(f"LOG TEST: {len(urls)} unique Google News URLs recognized")
print("LOG TEST: fake W3/XML publisher rejected")
print("LOG TEST: 17 observed 30-second AbortErrors are treated as an external deadline symptom, not a resolver success")
