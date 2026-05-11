"""RSS source adapter using feedparser."""

from __future__ import annotations

from typing import List

from agent.sources.base import RawItem


class RssAdapter:
    type_name = "rss"

    def __init__(self, source_id: str, url: str) -> None:
        self.source_id = source_id
        self.url = url

    def fetch(self, *, max_items: int = 20) -> List[RawItem]:
        import feedparser

        parsed = feedparser.parse(self.url)
        items: List[RawItem] = []
        for entry in parsed.entries[:max_items]:
            items.append(
                RawItem(
                    source_id=self.source_id,
                    source_type=self.type_name,
                    title=getattr(entry, "title", "").strip(),
                    url=getattr(entry, "link", "").strip(),
                    summary=_extract_summary(entry),
                    published_at=_normalize_time(entry),
                    author=getattr(entry, "author", "") or "",
                    tags=[t.term for t in getattr(entry, "tags", []) if hasattr(t, "term")],
                )
            )
        return items


def _extract_summary(entry) -> str:
    # feedparser exposes either summary or content.
    raw = getattr(entry, "summary", "") or ""
    if not raw and hasattr(entry, "content") and entry.content:
        raw = entry.content[0].get("value", "") if isinstance(entry.content, list) else ""
    # Strip rudimentary HTML without pulling lxml just for this.
    import re as _re

    text = _re.sub(r"<[^>]+>", " ", raw)
    text = _re.sub(r"\s+", " ", text).strip()
    return text[:1000]


def _normalize_time(entry) -> str:
    import time as _time
    from datetime import datetime, timezone

    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed is None:
        return ""
    try:
        ts = _time.mktime(parsed)
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return ""
