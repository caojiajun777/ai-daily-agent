"""Source adapter interface and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class RawItem:
    source_id: str
    source_type: str  # rss / arxiv / github / hn
    title: str
    url: str
    summary: str
    published_at: str  # ISO 8601 string, normalized by adapter
    author: str = ""
    tags: List[str] = None  # type: ignore[assignment]
    content_type: str = "tech_media"  # maps to content_types config key
    source_tier: str = ""
    reliability: str = ""
    evidence_type: str = ""
    confidence: str = "medium"
    section_hint: str = ""

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "url": self.url,
            "summary": self.summary,
            "published_at": self.published_at,
            "author": self.author,
            "tags": list(self.tags),
            "content_type": self.content_type,
            "source_tier": self.source_tier,
            "reliability": self.reliability,
            "evidence_type": self.evidence_type,
            "confidence": self.confidence,
            "section_hint": self.section_hint,
        }


class SourceAdapter(Protocol):
    type_name: str

    def fetch(self, *, max_items: int = 20) -> List[RawItem]: ...


def build_source(spec: Dict[str, Any]) -> SourceAdapter:
    """Construct a source adapter from a YAML config block."""
    t = spec.get("type")
    if t == "rss":
        from agent.sources.rss import RssAdapter

        return RssAdapter(
            source_id=spec["id"],
            url=spec["url"],
            timeout_sec=float(spec.get("timeout_sec", spec.get("fetch_timeout_sec", 12.0))),
        )
    if t == "sitemap":
        from agent.sources.sitemap import SitemapAdapter

        return SitemapAdapter(
            source_id=spec["id"],
            url=spec["url"],
            include_path=str(spec.get("include_path", "")),
            timeout_sec=float(spec.get("timeout_sec", spec.get("fetch_timeout_sec", 12.0))),
            fetch_pages=bool(spec.get("fetch_pages", True)),
        )
    if t == "arxiv":
        from agent.sources.arxiv_adapter import ArxivAdapter

        return ArxivAdapter(
            source_id=spec["id"],
            categories=str(spec.get("categories", "cs.AI+OR+cs.CL+OR+cs.LG+OR+cs.CV")),
            max_age_days=int(spec.get("max_age_days", 3)),
            top_venue_only=bool(spec.get("top_venue_only", False)),
        )
    if t == "github_releases":
        from agent.sources.github_stub import GithubReleasesAdapter

        return GithubReleasesAdapter(source_id=spec["id"])
    if t == "hn":
        from agent.sources.hn_stub import HackerNewsAdapter

        return HackerNewsAdapter(source_id=spec["id"])
    if t == "x":
        from agent.sources.x_adapter import XAdapter

        return XAdapter(
            source_id=spec["id"],
            username=spec["username"],
            account_type=spec.get("account_type", "official"),
            max_age_hours=int(spec.get("max_age_hours", 36)),
        )
    if t == "x_cookie":
        from agent.sources.x_cookie_adapter import XCookieAdapter

        return XCookieAdapter(
            source_id=spec["id"],
            username=spec["username"],
            account_type=spec.get("account_type", "official"),
            max_age_hours=int(spec.get("max_age_hours", 36)),
        )
    if t == "aihot":
        from agent.sources.aihot_adapter import AIHotAdapter

        return AIHotAdapter(source_id=spec["id"], url=spec["url"])
    if t == "pricing_snapshot":
        from agent.sources.pricing_snapshot import PricingSnapshotAdapter

        return PricingSnapshotAdapter(spec=spec)
    raise ValueError(f"unknown source type: {t}")
