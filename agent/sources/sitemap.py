"""Sitemap-backed source adapter.

Some official labs publish fast-moving news pages without RSS/Atom feeds.
This adapter reads a sitemap, filters URLs by path, and fetches page metadata
so those official announcements can still enter the collection stage.
"""

from __future__ import annotations

import html
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from agent.sources.base import RawItem

DEFAULT_SITEMAP_TIMEOUT_SEC = 12.0


@dataclass
class _SitemapEntry:
    loc: str
    lastmod: str


class SitemapAdapter:
    type_name = "sitemap"

    def __init__(
        self,
        source_id: str,
        url: str,
        *,
        include_path: str = "",
        timeout_sec: float = DEFAULT_SITEMAP_TIMEOUT_SEC,
        fetch_pages: bool = True,
    ) -> None:
        self.source_id = source_id
        self.url = url
        self.include_path = include_path
        self.timeout_sec = timeout_sec
        self.fetch_pages = fetch_pages

    def fetch(self, *, max_items: int = 20) -> List[RawItem]:
        entries = _parse_sitemap(
            _download(self.url, self.timeout_sec), include_path=self.include_path
        )
        items: List[RawItem] = []
        for entry in entries[:max_items]:
            title = _title_from_url(entry.loc)
            summary = ""
            if self.fetch_pages:
                title2, summary2 = _fetch_page_meta(entry.loc, self.timeout_sec)
                title = title2 or title
                summary = summary2
            items.append(
                RawItem(
                    source_id=self.source_id,
                    source_type=self.type_name,
                    title=title,
                    url=entry.loc,
                    summary=summary,
                    published_at=_normalize_time(entry.lastmod),
                )
            )
        return items


def _parse_sitemap(body: bytes, *, include_path: str = "") -> List[_SitemapEntry]:
    root = ET.fromstring(body)
    entries: List[_SitemapEntry] = []
    for node in root.findall(".//{*}url"):
        loc = _node_text(node, "loc")
        if not loc:
            continue
        if include_path and include_path not in urlparse(loc).path:
            continue
        entries.append(_SitemapEntry(loc=loc, lastmod=_node_text(node, "lastmod")))
    return sorted(entries, key=lambda e: e.lastmod or "", reverse=True)


def _node_text(node: ET.Element, tag: str) -> str:
    found = node.find(f"{{*}}{tag}")
    return (found.text or "").strip() if found is not None else ""


def _fetch_page_meta(url: str, timeout_sec: float) -> tuple[str, str]:
    try:
        text = _download(url, timeout_sec).decode("utf-8", errors="replace")
    except Exception:
        return "", ""
    return _extract_meta(text)


def _extract_meta(text: str) -> tuple[str, str]:
    title = (
        _meta_content(text, "property", "og:title")
        or _meta_content(text, "name", "twitter:title")
        or _tag_text(text, "title")
    )
    desc = (
        _meta_content(text, "name", "description")
        or _meta_content(text, "property", "og:description")
        or _meta_content(text, "name", "twitter:description")
    )
    return _clean(title), _clean(desc)[:1000]


def _meta_content(text: str, attr_name: str, attr_value: str) -> str:
    pattern = (
        r"<meta\b(?=[^>]*\b"
        + re.escape(attr_name)
        + r"\s*=\s*['\"]"
        + re.escape(attr_value)
        + r"['\"])(?=[^>]*\bcontent\s*=\s*['\"]([^'\"]+)['\"])[^>]*>"
    )
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else ""


def _tag_text(text: str, tag: str) -> str:
    m = re.search(
        rf"<{re.escape(tag)}[^>]*>(.*?)</{re.escape(tag)}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return m.group(1) if m else ""


def _clean(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_from_url(url: str) -> str:
    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    return " ".join(part.capitalize() for part in slug.split("-") if part) or url


def _normalize_time(value: str) -> str:
    if not value:
        return ""
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat()
    except Exception:
        return value


def _download(url: str, timeout_sec: float) -> bytes:
    if not url.startswith(("http://", "https://")):
        path = url.removeprefix("file://")
        return Path(path).read_bytes()

    import httpx

    timeout = httpx.Timeout(
        timeout=max(1.0, float(timeout_sec)),
        connect=min(5.0, max(1.0, float(timeout_sec))),
    )
    headers = {
        "User-Agent": "report-agent-sitemap/0.1",
        "Accept": "text/html, application/xml, text/xml, */*",
    }
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        for attempt in range(2):
            try:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.content
            except httpx.TransportError:
                if attempt:
                    raise
                time.sleep(0.2)
        raise RuntimeError("unreachable")
