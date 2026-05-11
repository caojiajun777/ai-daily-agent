"""X / Twitter source adapter.

Monitors specified X accounts and converts recent tweets into RawItem.
Uses X API v2 with Bearer Token authentication.

Required env var: ``X_BEARER_TOKEN``
Config shape:
  - id: "x_baidu"
    type: "x"
    username: "Baidu_Inc"
    account_type: "official"   # official / kol / media
    weight: 1.1
    max_items: 10

Rate limits (X API v2 Basic, ~$100/mo):
  - 10k requests / month per app
  - 1 request per user-lookup, 1 request per timeline fetch
  Each account costs 2 requests/day → ~60/mo per account.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from agent.sources.base import RawItem


class XAdapter:
    type_name = "x"

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
        self._client: Optional[httpx.Client] = None
        self._bearer_token: Optional[str] = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        token = os.getenv("X_BEARER_TOKEN", "")
        if not token:
            raise RuntimeError(
                "X_BEARER_TOKEN env var is not set. "
                "Get one from https://developer.x.com/en/portal/dashboard"
            )
        self._bearer_token = token
        self._client = httpx.Client(
            base_url="https://api.x.com/2",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "report-agent/0.1",
            },
            timeout=30.0,
        )
        return self._client

    def fetch(self, *, max_items: int = 20) -> List[RawItem]:
        client = self._ensure_client()
        user_id = self._resolve_user_id(client)
        if not user_id:
            return []
        return self._fetch_tweets(client, user_id, max_items)

    def _resolve_user_id(self, client: httpx.Client) -> Optional[str]:
        try:
            resp = client.get(f"/users/by/username/{self.username}")
            if resp.status_code != 200:
                return None
            data = resp.json().get("data", {})
            return data.get("id")
        except Exception:
            return None

    def _fetch_tweets(
        self, client: httpx.Client, user_id: str, max_items: int
    ) -> List[RawItem]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)
        cutoff_iso = cutoff.isoformat()

        try:
            resp = client.get(
                f"/users/{user_id}/tweets",
                params={
                    "max_results": min(max_items + 5, 100),
                    "tweet.fields": "created_at,text,entities,lang",
                    "exclude": "retweets,replies",
                },
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []

        tweets = data.get("data", [])
        items: List[RawItem] = []
        for t in tweets:
            created_at = t.get("created_at", "")
            if created_at < cutoff_iso:
                continue
            text = t.get("text", "").strip()
            if not text:
                continue
            # Build a tweet URL from the tweet id.
            tweet_id = t.get("id", "")
            tweet_url = f"https://x.com/{self.username}/status/{tweet_id}"

            # Clean text: strip t.co URLs (unreadable) but keep the text.
            cleaned = _clean_tweet_text(text)

            items.append(
                RawItem(
                    source_id=self.source_id,
                    source_type=self.type_name,
                    title=_make_title(cleaned),
                    url=tweet_url,
                    summary=cleaned[:800],
                    published_at=created_at,
                    author=f"@{self.username}",
                    tags=[f"x-{self.account_type}"],
                )
            )
            if len(items) >= max_items:
                break
        return items


def _clean_tweet_text(text: str) -> str:
    # Remove t.co URLs.
    cleaned = re.sub(r"https?://t\.co/\S+", "", text)
    # Collapse whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _make_title(text: str) -> str:
    # First line or first sentence, up to 100 chars.
    first = text.split("\n")[0].strip()
    if len(first) > 120:
        # Truncate at last complete word boundary.
        first = first[:117].rsplit(" ", 1)[0] + "…"
    return first
