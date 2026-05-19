"""Final Selector — converts ResearchEditor decisions into Writer-ready items.

Enforces section diversity, source diversity, and min/max item counts.
Falls back to rule_score ordering if LLM decisions are insufficient.
Guarantees a minimum number of arxiv papers in the final selection.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from agent.agents.event_clusterer import EventCluster
from agent.agents.research_editor import (
    EditorialDecision,
    ResearchEditorOutput,
)
from agent.schemas import CuratedItem, CuratedItemRecord

_MIN_PAPERS = 5


def select_final_items(
    *,
    editor_output: ResearchEditorOutput,
    events: List[EventCluster],
    min_items: int = 16,
    max_items: int = 22,
    min_papers: int = _MIN_PAPERS,
    section_diversity: bool = True,
    source_diversity: bool = True,
) -> Tuple[List[CuratedItem], List[CuratedItemRecord], Dict[str, Any]]:
    """Convert ResearchEditor decisions into final CuratedItems for the Writer.

    Returns (writer_items, records, meta_info).
    meta_info includes: fallback_used, editorial_review_skipped, fallback_reason.
    """
    meta: Dict[str, Any] = {
        "fallback_used": False,
        "editorial_review_skipped": False,
        "fallback_reason": "",
        "llm_selected_count": len([d for d in editor_output.selected if d.decision == "select"]),
        "final_selected_count": 0,
    }

    # Build event lookup.
    event_map: Dict[str, EventCluster] = {e.event_id: e for e in events}
    all_decisions: Dict[str, EditorialDecision] = {}
    for d in editor_output.selected:
        if d.decision == "select":
            all_decisions[d.event_id] = d

    # ── Fallback check: if LLM selected too few ──────────────────────
    if len(all_decisions) < min_items:
        meta["fallback_used"] = True
        meta["fallback_reason"] = f"LLM selected {len(all_decisions)} < {min_items} minimum"
        # Fill from rule_score ranked events not already selected.
        remaining = [e for e in events if e.event_id not in all_decisions]
        remaining.sort(key=lambda e: e.rule_score, reverse=True)
        needed = min_items - len(all_decisions)
        for evt in remaining[:needed]:
            all_decisions[evt.event_id] = EditorialDecision(
                event_id=evt.event_id,
                decision="select",
                priority="medium",
                section=_guess_section(evt),
                evidence_level="primary" if evt.source_count >= 2 else "social",
                novelty="unclear",
                reader_utility="medium",
                why_it_matters="Fallback — rule score ranked.",
                writing_angle=f"Based on {evt.source_count} source(s).",
                risk_level="low",
                sources_to_use=[],
            )
            if evt.source_urls:
                from agent.agents.research_editor import SourceUse
                all_decisions[evt.event_id].sources_to_use = [
                    SourceUse(url=evt.source_urls[0], role="primary"),
                ]

    # ── Priority ordering ────────────────────────────────────────────
    priority_order = {"must_include": 0, "high": 1, "medium": 2, "low": 3}
    sorted_ids = sorted(
        all_decisions.keys(),
        key=lambda eid: (
            priority_order.get(all_decisions[eid].priority, 2),
            -event_map[eid].rule_score,
        ),
    )

    # ── Apply section + source diversity ─────────────────────────────
    final_ids: List[str] = []
    section_counts: Counter = Counter()
    source_counts: Counter = Counter()
    section_caps: Dict[str, int] = {
        "今日头条": 3, "模型前沿": 4, "工具与开源": 3,
        "论文精选": 5, "产品落地": 3, "业界风向": 4,
    }
    section_order = ["今日头条", "模型前沿", "工具与开源", "论文精选", "产品落地", "业界风向"]

    # First pass: must_include always in.
    for eid in sorted_ids:
        d = all_decisions[eid]
        if d.priority == "must_include":
            final_ids.append(eid)
            section_counts[d.section or "业界风向"] += 1

    # Second pass: fill by priority, respecting caps.
    for eid in sorted_ids:
        if eid in final_ids or len(final_ids) >= max_items:
            break
        d = all_decisions[eid]
        if d.priority == "must_include":
            continue

        sec = d.section or "业界风向"

        # Section cap check.
        cap = section_caps.get(sec, 4)
        if section_counts.get(sec, 0) >= cap and d.priority not in ("high",):
            continue

        # Source diversity: max 3 from same source (for the whole draft).
        evt = event_map.get(eid)
        if evt and source_diversity:
            max_src = max(
                (source_counts.get(s, 0) for s in evt.source_names),
                default=0,
            )
            if max_src >= 3 and d.priority not in ("high", "must_include"):
                continue

        final_ids.append(eid)
        section_counts[sec] += 1
        for s in (evt.source_names if evt else []):
            source_counts[s] += 1

    # Ensure all 6 sections have at least 1 item.
    covered = set(section_counts.keys())
    missing = [s for s in section_order if s not in covered]
    for eid in sorted_ids:
        if not missing or len(final_ids) >= max_items:
            break
        if eid in final_ids:
            continue
        d = all_decisions[eid]
        if d.section in missing:
            final_ids.append(eid)
            missing.remove(d.section)

    # Ensure min_items.
    if len(final_ids) < min_items:
        for eid in sorted_ids:
            if len(final_ids) >= min_items:
                break
            if eid not in final_ids:
                final_ids.append(eid)

    meta["final_selected_count"] = len(final_ids)

    # ── Paper quota enforcement ──────────────────────────────────────
    # If < min_papers paper items made it through, pull the best paper
    # events and swap them in, replacing the lowest non-paper items.
    def _is_paper_event(evt: EventCluster | None) -> bool:
        if evt is None:
            return False
        return (
            any(t == "arxiv" for t in evt.source_types)
            or any(n in ("hf_daily_papers",) for n in evt.source_names)
        )

    arxiv_in_final = [
        eid for eid in final_ids
        if event_map.get(eid) and _is_paper_event(event_map[eid])
    ]
    if len(arxiv_in_final) < min_papers:
        # Candidates: paper events NOT already in final_ids, sorted by rule_score.
        all_papers = [
            e for e in events
            if _is_paper_event(e) and e.event_id not in final_ids
        ]
        all_papers.sort(key=lambda e: e.rule_score, reverse=True)
        needed = min_papers - len(arxiv_in_final)
        to_add = all_papers[:needed]

        # Remove lowest non-paper items from final_ids.
        non_paper = [
            eid for eid in final_ids
            if not _is_paper_event(event_map.get(eid))  # only safe: all eids from event_map
        ]
        non_paper.sort(
            key=lambda eid: (event_map[eid].rule_score if event_map.get(eid) else 0)
        )  # ascending, lowest first
        to_remove = set(non_paper[:len(to_add)])

        final_ids = [eid for eid in final_ids if eid not in to_remove]
        final_ids.extend(e.event_id for e in to_add)

        meta["paper_quota_enforced"] = True
        meta["paper_quota_added"] = len(to_add)

    # ── Convert to CuratedItems ──────────────────────────────────────
    writer_items: List[CuratedItem] = []
    records: List[CuratedItemRecord] = []

    for eid in final_ids:
        evt = event_map.get(eid)
        dec = all_decisions.get(eid)
        if not evt:
            continue

        # Priority to primary_url from decision if valid.
        primary_url = evt.primary_url
        if dec and dec.sources_to_use:
            for s in dec.sources_to_use:
                if s.role == "primary" and s.url in evt.source_urls:
                    primary_url = s.url
                    break

        writer_items.append(CuratedItem(
            title=evt.canonical_title,
            url=primary_url,
            summary=evt.summary[:500],
            source=evt.source_names[0] if evt.source_names else "unknown",
            source_type=evt.source_types[0] if evt.source_types else "rss",
            published_at=evt.latest_seen_at,
            score=evt.rule_score,
        ))

        records.append(CuratedItemRecord(
            raw_item_id=evt.event_id,
            title=evt.canonical_title,
            source_url=primary_url,
            source_name=evt.source_names[0] if evt.source_names else "unknown",
            published_at=evt.latest_seen_at or None,
            score=evt.rule_score,
            section=dec.section if dec else None,
            selected_reason=f"priority={dec.priority}" if dec else "rule",
            duplicate_group_id=None,
            used_in_draft=True,
        ))

    return writer_items, records, meta


def _guess_section(evt: EventCluster) -> str:
    """Guess section from event content — v2.2 six-section layout."""
    text = (evt.canonical_title + " " + evt.summary).lower()
    # Paper detection: arxiv source or paper keywords — always papers
    if any(t == "arxiv" for t in evt.source_types) or any(
        k in text for k in ["论文", "arxiv", "paper", "researchers", "研究团队",
                            "neurlps", "icml", "iclr", "cvpr", "emnlp", "aaai"]):
        return "论文精选"
    if any(k in text for k in ["融资", "funding", "ipo", "收购", "裁员", "政策", "监管",
                                 "regulation", "law", "ban", "hire", "ceo", "executive",
                                 "partnership", "合作", "投资", "估值"]):
        return "业界风向"
    if any(k in text for k in ["framework", "sdk", "tool", "library", "github", "开源",
                                 "api", "cli", "plugin", "extension", "vscode", "copilot",
                                 "降价", "定价", "免费", "open source"]):
        return "工具与开源"
    # Model before product — model keywords are more specific
    if any(k in text for k in ["model", "模型", "gpt", "claude", "gemini", "parameters",
                                 "benchmark", "eval", "llm", "diffusion", "语音",
                                 "开源模型", "architecture", "training", "推理"]):
        return "模型前沿"
    if any(k in text for k in ["app", "product", "chatbot", "assistant", "功能", "应用",
                                 "产品", "用户", "推出", "上线", "launch", "feature", "更新"]):
        return "产品落地"
    return "业界风向"
