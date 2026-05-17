"""X/Twitter adapter via Playwright + proxy.

X is a JS SPA — requires browser rendering. Uses Playwright with
VPN/proxy for GFW bypass. No paid API key or login cookies needed
for reading public tweets.

Requires X_PROXY env var or HTTPS_PROXY for GFW bypass.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from agent.sources.base import RawItem

# Singleton browser (reuse across all adapter instances to avoid
# launch overhead per source).
_browser = None
_playwright = None


def _get_page(proxy_url: str, user_agent: str):
    global _browser, _playwright
    if _playwright is None:
        from playwright.sync_api import sync_playwright
        _playwright = sync_playwright().start()
    if _browser is None:
        _browser = _playwright.chromium.launch(
            channel="msedge",
            headless=True,
            proxy={"server": proxy_url} if proxy_url else None,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
    ctx = _browser.new_context(
        user_agent=user_agent,
        viewport={"width": 1280, "height": 900},
    )
    return ctx.new_page()


class XCookieAdapter:
    type_name = "x_cookie"

    def __init__(
        self,
        source_id: str,
        username: str,
        account_type: str = "official",
        max_age_hours: int = 36,
    ) -> None:
        self.source_id = source_id
        self.username = username.lstrip("@")
        self.account_type = account_type
        self.max_age_hours = max_age_hours

    def _get_proxy(self) -> str:
        return os.getenv("X_PROXY", "") or os.getenv("HTTPS_PROXY", "") or ""

    def fetch(self, *, max_items: int = 10) -> List[RawItem]:
        proxy = self._get_proxy()
        if not proxy:
            return []

        ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )

        page = None
        try:
            page = _get_page(proxy, ua)
            page.goto(
                f"https://x.com/{self.username}",
                wait_until="load",
                timeout=20000,
            )
            page.wait_for_timeout(1500)

            html = page.content()
            return _extract_tweets_from_html(
                html, self.username, self.source_id,
                self.account_type, self.max_age_hours, max_items,
            )
        except Exception:
            return []
        finally:
            if page:
                try:
                    page.close()
                except Exception:
                    pass


def _extract_tweets_from_html(
    html: str, username: str, source_id: str,
    account_type: str, max_age_hours: int, max_items: int,
) -> List[RawItem]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    items: List[RawItem] = []
    seen: set = set()

    tweet_pattern = re.compile(
        r'data-testid="tweetText"[^>]*>(.*?)</div>',
        re.DOTALL,
    )

    for m in tweet_pattern.finditer(html):
        text = _strip_html(m.group(1)).strip()
        if not text or text in seen:
            continue
        seen.add(text)

        title = text.split("\n")[0].strip()[:120]
        cleaned = _clean_tweet_text(text)

        items.append(RawItem(
            source_id=source_id,
            source_type="x_cookie",
            title=title,
            url=f"https://x.com/{username}",
            summary=cleaned[:800],
            published_at=datetime.now(timezone.utc).isoformat(),
            author=f"@{username}",
            tags=[f"x-{account_type}"],
        ))
        if len(items) >= max_items:
            break

    return items


def _strip_html(text: str) -> str:
    text = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*>', r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#x27;", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_tweet_text(text: str) -> str:
    cleaned = re.sub(r"https?://t\.co/\S+", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
