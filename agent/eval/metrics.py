"""Deterministic evaluation metrics.

These are checks a regression suite can rely on: same inputs, same numbers, no
LLM. They live separately from the critic because they can also be run after
the fact on past runs to score the harness over time.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from agent.schemas import CuratedItem, CuratedItemRecord, Draft


def deterministic_metrics(
    *,
    draft: Draft,
    curated: List[CuratedItem],
    min_unique_titles_ratio: float = 0.8,
    min_section_count: int = 3,
    forbid_phrases: Optional[Iterable[str]] = None,
    curated_records: Optional[List[CuratedItemRecord]] = None,
) -> Dict[str, Any]:
    forbid_phrases = list(forbid_phrases or [])
    section_count = len(draft.sections)
    items = [it for s in draft.sections for it in s.items]
    item_count = len(items)
    titles = [it.title.strip().lower() for it in items if it.title]
    unique_titles = len(set(titles))
    unique_titles_ratio = (unique_titles / len(titles)) if titles else 0.0

    # Build allowed URL set: prefer persistent curated records when available.
    used_curated_artifact: bool
    allowed_urls: Set[str]
    if curated_records is not None:
        allowed_urls = {rec.source_url for rec in curated_records}
        used_curated_artifact = True
    else:
        allowed_urls = {c.url for c in curated}
        used_curated_artifact = False

    hallucinated_urls = sum(
        1 for it in items if it.url and allowed_urls and it.url not in allowed_urls
    )

    forbidden_hits = 0
    for it in items:
        text = f"{it.title} {it.summary}"
        for bad in forbid_phrases:
            if bad and bad in text:
                forbidden_hits += 1

    issues: List[str] = []
    if section_count < min_section_count:
        issues.append("section_count_below_threshold")
    if unique_titles_ratio < min_unique_titles_ratio:
        issues.append("low_unique_titles_ratio")
    if hallucinated_urls > 0:
        issues.append("hallucinated_urls_present")
    if forbidden_hits > 0:
        issues.append("forbidden_phrases_present")

    return {
        "section_count": section_count,
        "item_count": item_count,
        "unique_titles_ratio": round(unique_titles_ratio, 3),
        "hallucinated_urls": hallucinated_urls,
        "forbidden_hits": forbidden_hits,
        "issues": issues,
        "ok": not issues,
        "used_curated_artifact": used_curated_artifact,
    }
