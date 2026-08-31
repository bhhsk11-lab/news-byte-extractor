FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 \
    libcairo2 libatspi2.0-0 libx11-6 libxcb1 libxext6 \
    libglib2.0-0 libdbus-1-3 libxshmfence1 libgtk-3-0 \
    libu2f-udev libvulkan1 libexpat1 xdg-utils \
    fonts-liberation fonts-unifont fonts-freefont-ttf \
    fonts-noto-color-emoji fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium

COPY app.py google_resolver.py README.md extension-example.js ./
RUN python -m py_compile app.py google_resolver.py
RUN useradd --create-home --uid 1000 user && chown -R user:user /app /ms-playwright
USER user
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
