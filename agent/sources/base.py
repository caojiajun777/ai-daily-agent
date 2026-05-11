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
        )
    if t == "arxiv":
        from agent.sources.arxiv_stub import ArxivAdapter

        return ArxivAdapter(source_id=spec["id"])
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
    if t == "aihot":
        from agent.sources.aihot_adapter import AIHotAdapter

        return AIHotAdapter(source_id=spec["id"], url=spec["url"])
    raise ValueError(f"unknown source type: {t}")
