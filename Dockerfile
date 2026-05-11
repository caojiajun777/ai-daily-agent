FROM python:3.12-slim

WORKDIR /app

# System deps for Playwright + feedparser.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2t64 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser (Chromium).
RUN python -m playwright install chromium --with-deps 2>/dev/null || true

COPY . .

ENV ARTIFACTS_DIR=/app/artifacts

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "agent.web.app:app", "--host", "0.0.0.0", "--port", "8080"]
