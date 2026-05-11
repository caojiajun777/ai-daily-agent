"""Stub: GitHub Releases adapter (reserved)."""

from __future__ import annotations

from typing import List

from agent.sources.base import RawItem


class GithubReleasesAdapter:
    type_name = "github_releases"

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id

    def fetch(self, *, max_items: int = 20) -> List[RawItem]:
        raise NotImplementedError(
            "GithubReleasesAdapter is a stub in MVP. Implement in next phase."
        )
