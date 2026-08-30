import os

class Settings:
    def __init__(self):
        # Try GNEWS_PROXY_URL first, then fallback to PROXY_URL
        self.proxy_url = os.getenv("GNEWS_PROXY_URL") or os.getenv("PROXY_URL") or ""

settings = Settings()
