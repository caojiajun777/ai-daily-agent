"""Stub: arXiv adapter (reserved for next phase)."""

from __future__ import annotations

from typing import List

from agent.sources.base import RawItem


class ArxivAdapter:
    type_name = "arxiv"

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id

    def fetch(self, *, max_items: int = 20) -> List[RawItem]:
        raise NotImplementedError(
            "ArxivAdapter is a stub in MVP. Implement in next phase."
        )
