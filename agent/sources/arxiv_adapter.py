"""arXiv source adapter — top AI venue paper collection.

Uses arXiv public API (free, no key). Queries can filter by:
  - AI/ML categories (cs.AI, cs.LG, etc.)
  - Top venue mentions in comments/abstract (NeurIPS, ICML, ICLR, etc.)
  - Recent submissions (1-7 day window)

Top venues tracked:
  Conferences: NeurIPS, ICML, ICLR, CVPR, ICCV, ACL, EMNLP, AAAI, IJCAI
  Journals: JMLR, TPAMI, TACL, Nature (AI/ML), Science (AI/ML)
"""

from __future__ import annotations

import time as _time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List

import httpx

from agent.sources.base import RawItem

_ARXIV_API = "https://export.arxiv.org/api/query"
_ARXIV_NS = "{http://www.w3.org/2005/Atom}"

_TOP_VENUES = [
    # Conferences
    "NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV",
    "ACL", "EMNLP", "NAACL", "AAAI", "IJCAI",
    "SIGGRAPH", "SIGKDD", "KDD", "WWW", "SIGIR",
    "ICRA", "IROS", "RSS", "CoRL",  # robotics
    # Journals
    "JMLR", "TPAMI", "TACL",
    "Nature", "Science",
    "NeurIPS", "TMLR", "JAIR",
]

_VENUE_QUERY_TEMPLATE = (
    "cat:{categories}+AND+({venue_search})"
)


class ArxivAdapter:
    type_name = "arxiv"

    def __init__(
        self,
        source_id: str,
        categories: str = "cs.AI+OR+cs.CL+OR+cs.LG+OR+cs.CV",
        max_age_days: int = 3,
        top_venue_only: bool = False,
    ) -> None:
        self.source_id = source_id
        self.categories = categories
        self.max_age_days = max_age_days
        self.top_venue_only = top_venue_only
        self._last_fetch = 0.0

    def fetch(self, *, max_items: int = 20) -> List[RawItem]:
        elapsed = _time.time() - self._last_fetch
        if elapsed < 3.0:
            _time.sleep(3.0 - elapsed)

        query = self._build_query()
        url = (
            f"{_ARXIV_API}?search_query={query}"
            f"&sortBy=submittedDate&sortOrder=descending"
            f"&max_results={max_items}"
        )

        try:
            resp = httpx.get(url, timeout=20.0,
                             headers={"User-Agent": "AI-Daily-Agent/2.0"})
            self._last_fetch = _time.time()
            if resp.status_code != 200:
                return []
            items = _parse_atom(resp.text, self.source_id, self.max_age_days)
            # If top-venue mode, additional client-side filter.
            if self.top_venue_only:
                items = [it for it in items
                         if _mentions_top_venue(it.title, it.summary)]
            return items
        except Exception:
            self._last_fetch = _time.time()
            return []

    def _build_query(self) -> str:
        # Always use plain category query. Client-side does venue filtering.
        return f"cat:{self.categories}"


def _mentions_top_venue(title: str, summary: str) -> bool:
    """Check if title or summary mentions a top venue."""
    text = (title + " " + summary).lower()
    return any(v.lower() in text for v in _TOP_VENUES)


def _parse_atom(xml_text: str, source_id: str, max_age_days: int) -> List[RawItem]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    items: List[RawItem] = []
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_days * 86400

    for entry in root.findall(f"{_ARXIV_NS}entry"):
        title_el = entry.find(f"{_ARXIV_NS}title")
        summary_el = entry.find(f"{_ARXIV_NS}summary")
        link_el = entry.find(f"{_ARXIV_NS}id")
        published_el = entry.find(f"{_ARXIV_NS}published")
        authors = entry.findall(f"{_ARXIV_NS}author/{_ARXIV_NS}name")

        title = (title_el.text or "").strip() if title_el is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        url = (link_el.text or "").strip() if link_el is not None else ""
        published = (published_el.text or "").strip() if published_el is not None else ""
        author_names = [a.text.strip() for a in authors if a.text]
        # arxiv:comment often contains venue info, e.g. "Accepted at NeurIPS 2026"
        comment_el = entry.find(f"{_ARXIV_NS}comment")
        comment = (comment_el.text or "").strip() if comment_el is not None else ""

        if not title or not url:
            continue

        try:
            pub_ts = datetime.fromisoformat(
                published.replace("Z", "+00:00")).timestamp()
            if pub_ts < cutoff:
                continue
        except Exception:
            pass

        # Append comment (venue info) to summary for filtering.
        full_text = summary
        if comment:
            full_text = summary + " [" + comment + "]"
        summary_clean = " ".join(full_text.split())[:800]
        author_str = ", ".join(author_names[:3])
        if len(author_names) > 3:
            author_str += " et al."

        # Build a news-friendly title.
        display_title = title
        if author_str:
            display_title = f"{title} — {author_str}"

        abs_url = url.replace("http://", "https://")

        items.append(RawItem(
            source_id=source_id,
            source_type="arxiv",
            title=display_title[:200],
            url=abs_url,
            summary=summary_clean,
            published_at=published,
            author=author_str,
            tags=["paper", "arxiv"] + (
                ["top-venue"] if _mentions_top_venue(title, summary) else []
            ),
        ))

    return items
