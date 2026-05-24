"""Final Selector — converts ResearchEditor decisions into Writer-ready items.

Enforces section diversity, source diversity, and min/max item counts.
Falls back to rule_score ordering if LLM decisions are insufficient.
Uses section caps without forcing low-value filler stories.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agent.agents.event_clusterer import EventCluster
from agent.agents.research_editor import (
    EditorialDecision,
    ResearchEditorOutput,
)
from agent.agents.section_classifier import guess_section as _classify_section
from agent.schemas import CuratedItem, CuratedItemRecord

_MIN_PAPERS = 0

SECTION_ORDER = [
    "要闻", "模型发布", "开发生态", "技术与洞察",
    "产品应用", "行业动态", "前瞻与传闻",
]

SECTION_ALIASES = {
    "今日头条": "要闻",
    "模型前沿": "模型发布",
    "工具与开源": "开发生态",
    "论文精选": "技术与洞察",
    "产品落地": "产品应用",
    "资本动向": "行业动态",
    "产业风向": "行业动态",
    "业界风向": "行业动态",
}


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
                why_it_matters=_fallback_why_it_matters(evt),
                writing_angle=_fallback_writing_angle(evt, _guess_section(evt)),
                risk_level="low",
                sources_to_use=[],
            )
            if evt.source_urls:
                from agent.agents.research_editor import SourceUse
                all_decisions[evt.event_id].sources_to_use = [
                    SourceUse(url=evt.source_urls[0], role="primary"),
                ]

    normalized_sections = _normalize_decision_sections(all_decisions, event_map)
    if normalized_sections:
        meta["section_normalized_count"] = normalized_sections

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
    story_counts: Counter = Counter()
    section_caps: Dict[str, int] = {
        "要闻": 3,
        "模型发布": 4,
        "开发生态": 5,
        "技术与洞察": 3,
        "产品应用": 4,
        "行业动态": 4,
        "前瞻与传闻": 2,
    }
    section_order = SECTION_ORDER
    ranked_events = sorted(events, key=lambda e: e.rule_score, reverse=True)

    def ensure_decision(
        evt: EventCluster,
        section: str,
        *,
        priority: str = "low",
    ) -> Tuple[EditorialDecision, bool]:
        dec = all_decisions.get(evt.event_id)
        if dec is not None:
            return dec, False
        dec = EditorialDecision(
            event_id=evt.event_id,
            decision="select",
            priority=priority,
            section=section,
            evidence_level="primary" if evt.source_count >= 2 else "weak",
            novelty="unclear",
            reader_utility="medium",
            why_it_matters=_fallback_why_it_matters(evt),
            writing_angle=_fallback_writing_angle(evt, section),
            risk_level="medium",
            sources_to_use=[],
        )
        if evt.source_urls:
            from agent.agents.research_editor import SourceUse
            dec.sources_to_use = [SourceUse(url=evt.source_urls[0], role="primary")]
        all_decisions[evt.event_id] = dec
        sorted_ids.append(evt.event_id)
        return dec, True

    def try_add_event(
        evt: EventCluster,
        *,
        section: Optional[str] = None,
        respect_caps: bool = True,
        respect_story: bool = True,
        priority: str = "low",
    ) -> bool:
        if len(final_ids) >= max_items or evt.event_id in final_ids:
            return False
        if _is_stale_background_event(evt):
            return False
        sec = section or (all_decisions.get(evt.event_id).section if all_decisions.get(evt.event_id) else "")
        sec = _normalize_section(sec or _guess_section(evt))
        cap = section_caps.get(sec, 4)
        if sec == "技术与洞察" and section_counts.get(sec, 0) >= cap:
            return False
        if respect_caps and section_counts.get(sec, 0) >= cap:
            return False
        key = _story_key(evt)
        if respect_story and key and story_counts.get(key, 0) >= 1:
            return False
        if source_diversity and _source_cap_exceeded(evt, source_counts, priority=priority):
            return False

        dec, created = ensure_decision(evt, sec, priority=priority)
        if created or _normalize_section(dec.section or "") not in section_order:
            dec.section = sec
        final_ids.append(evt.event_id)
        section_counts[sec] += 1
        if key:
            story_counts[key] += 1
        for s in evt.source_names:
            source_counts[s] += 1
        return True

    # First pass: must_include always in.
    for eid in sorted_ids:
        d = all_decisions[eid]
        if d.priority == "must_include":
            evt = event_map.get(eid)
            if evt and _is_stale_background_event(evt):
                continue
            sec = _normalize_section(d.section or "行业动态")
            key = _story_key(evt)
            if section_counts.get(sec, 0) >= section_caps.get(sec, 4):
                continue
            if key and story_counts.get(key, 0) >= 1:
                continue
            if source_diversity and _source_cap_exceeded(evt, source_counts, priority=d.priority):
                continue
            final_ids.append(eid)
            section_counts[sec] += 1
            if key:
                story_counts[key] += 1
            for src in evt.source_names:
                source_counts[src] += 1

    # Second pass: fill by priority, respecting caps.
    for eid in sorted_ids:
        if eid in final_ids or len(final_ids) >= max_items:
            break
        d = all_decisions[eid]
        if d.priority == "must_include":
            continue

        sec = _normalize_section(d.section or "行业动态")

        # Section cap check.
        cap = section_caps.get(sec, 4)
        if section_counts.get(sec, 0) >= cap and (
            sec == "前瞻与传闻" or d.priority not in ("high",)
        ):
            continue

        # Source diversity: max 3 from same source (for the whole draft).
        evt = event_map.get(eid)
        if evt and _is_stale_background_event(evt):
            continue
        key = _story_key(evt)
        if key and story_counts.get(key, 0) >= 1:
            continue
        if evt and source_diversity:
            if _source_cap_exceeded(evt, source_counts, priority=d.priority):
                continue

        final_ids.append(eid)
        section_counts[sec] += 1
        if key:
            story_counts[key] += 1
        for s in (evt.source_names if evt else []):
            source_counts[s] += 1

    # Nudge coverage for the sections readers most expect, but do not force
    # low-value filler or a rumor item just to make every bucket non-empty.
    covered = set(section_counts.keys())
    preferred_sections = ["模型发布", "开发生态", "技术与洞察", "行业动态"]
    missing = [s for s in preferred_sections if s not in covered]
    for eid in sorted_ids:
        if not missing or len(final_ids) >= max_items:
            break
        if eid in final_ids:
            continue
        d = all_decisions[eid]
        d_section = _normalize_section(d.section or "")
        if d_section in missing:
            evt = event_map.get(eid)
            if evt and _is_stale_background_event(evt):
                continue
            key = _story_key(evt)
            if key and story_counts.get(key, 0) >= 1:
                continue
            if source_diversity and _source_cap_exceeded(evt, source_counts, priority=d.priority):
                continue
            final_ids.append(eid)
            section_counts[d_section] += 1
            if key:
                story_counts[key] += 1
            for src in (evt.source_names if evt else []):
                source_counts[src] += 1
            missing.remove(d_section)

    # Pull strong section fillers from the full event pool before giving up.
    if missing and len(final_ids) < max_items:
        for sec in list(missing):
            for evt in ranked_events:
                if _guess_section(evt) != sec:
                    continue
                if try_add_event(evt, section=sec, respect_caps=False, respect_story=True):
                    missing.remove(sec)
                    break

    # If diversity caps leave us short, backfill with high-scoring non-paper
    # events that the editor did not explicitly select. This keeps the daily
    # broad without letting arXiv papers flood the issue.
    if len(final_ids) < min_items:
        known_ids = set(sorted_ids)
        added_backfill = 0
        for evt in events:
            if added_backfill >= max_items:
                break
            if evt.event_id in known_ids:
                continue
            sec = _guess_section(evt)
            if sec == "技术与洞察" and section_counts.get(sec, 0) >= section_caps[sec]:
                continue
            if _is_stale_background_event(evt):
                continue
            all_decisions[evt.event_id] = EditorialDecision(
                event_id=evt.event_id,
                decision="select",
                priority="low",
                section=sec,
                evidence_level="primary" if evt.source_count >= 2 else "weak",
                novelty="unclear",
                reader_utility="medium",
                why_it_matters=_fallback_why_it_matters(evt),
                writing_angle=_fallback_writing_angle(evt, sec),
                risk_level="medium",
                sources_to_use=[],
            )
            if evt.source_urls:
                from agent.agents.research_editor import SourceUse
                all_decisions[evt.event_id].sources_to_use = [
                    SourceUse(url=evt.source_urls[0], role="primary"),
                ]
            sorted_ids.append(evt.event_id)
            known_ids.add(evt.event_id)
            added_backfill += 1

    # Ensure min_items.
    if len(final_ids) < min_items:
        for respect_caps, respect_story in ((True, True), (False, True)):
            for eid in sorted_ids:
                if len(final_ids) >= min_items:
                    break
                if eid in final_ids:
                    continue
                d = all_decisions[eid]
                sec = _normalize_section(d.section or "行业动态")
                cap = section_caps.get(sec, 4)
                if sec == "技术与洞察" and section_counts.get(sec, 0) >= cap:
                    continue
                if respect_caps and section_counts.get(sec, 0) >= cap:
                    continue
                evt = event_map.get(eid)
                if evt and _is_stale_background_event(evt):
                    continue
                key = _story_key(evt)
                if respect_story and key and story_counts.get(key, 0) >= 1:
                    continue
                if source_diversity and respect_caps and _source_cap_exceeded(evt, source_counts, priority=d.priority):
                    continue
                final_ids.append(eid)
                section_counts[sec] += 1
                if key:
                    story_counts[key] += 1
                for src in (evt.source_names if evt else []):
                    source_counts[src] += 1
            if len(final_ids) >= min_items:
                break

    if len(final_ids) < min_items:
        for evt in ranked_events:
            if len(final_ids) >= min_items:
                break
            sec = _guess_section(evt)
            if sec == "技术与洞察" and section_counts.get(sec, 0) >= section_caps["技术与洞察"]:
                continue
            if _is_stale_background_event(evt):
                continue
            try_add_event(evt, section=sec, respect_caps=False, respect_story=True)

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
    story_url_map: Dict[str, List[str]] = {}
    for evt in events:
        key = _story_key(evt)
        if not key:
            continue
        story_url_map.setdefault(key, [])
        story_url_map[key].extend(evt.source_urls)

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

        section = _normalize_section(dec.section if dec and dec.section else _guess_section(evt))
        primary_source_name = evt.primary_source_name or (
            evt.source_names[0] if evt.source_names else "unknown"
        )
        primary_source_type = evt.primary_source_type or (
            evt.source_types[0] if evt.source_types else "rss"
        )
        source_tier, reliability, evidence_type, confidence = _normalized_source_meta(
            evt=evt,
            primary_source_name=primary_source_name,
            primary_url=primary_url,
        )
        supporting_urls = _supporting_urls(evt, dec, primary_url)
        story_key = _story_key(evt)
        if story_key:
            supporting_urls = _merge_urls(
                supporting_urls,
                story_url_map.get(story_key, []),
                primary_url,
            )
        why_it_matters = dec.why_it_matters if dec else ""
        writing_angle = dec.writing_angle if dec else ""

        writer_items.append(CuratedItem(
            title=evt.canonical_title,
            url=primary_url,
            summary=evt.summary[:500],
            source=primary_source_name,
            source_type=primary_source_type,
            published_at=evt.latest_seen_at,
            score=evt.rule_score,
            content_type=evt.primary_content_type or "tech_media",
            source_tier=source_tier,
            reliability=reliability,
            evidence_type=evidence_type,
            confidence=confidence,
            section=section,
            section_hint=evt.primary_section_hint,
            why_it_matters=why_it_matters,
            writing_angle=writing_angle,
            supporting_urls=supporting_urls,
            evidence_snippets=evt.evidence_snippets[:3],
        ))

        records.append(CuratedItemRecord(
            raw_item_id=evt.event_id,
            title=evt.canonical_title,
            source_url=primary_url,
            source_name=primary_source_name,
            published_at=evt.latest_seen_at or None,
            score=evt.rule_score,
            section=section,
            selected_reason=f"priority={dec.priority}" if dec else "rule",
            duplicate_group_id=None,
            used_in_draft=True,
            content_type=evt.primary_content_type or "tech_media",
            source_tier=source_tier,
            reliability=reliability,
            confidence=confidence,
            evidence_type=evidence_type,
            section_hint=evt.primary_section_hint,
        ))

    return writer_items, records, meta


def _supporting_urls(
    evt: EventCluster,
    dec: Optional[EditorialDecision],
    primary_url: str,
    limit: int = 4,
) -> List[str]:
    urls: List[str] = []
    if dec:
        for src in dec.sources_to_use:
            if src.url != primary_url:
                urls.append(src.url)
    for url in evt.source_urls:
        if url != primary_url:
            urls.append(url)

    seen = set()
    out: List[str] = []
    for url in urls:
        if not url or url in seen or _is_social_profile_url(url):
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= limit:
            break
    return out


def _merge_urls(base: List[str], extra: List[str], primary_url: str, limit: int = 4) -> List[str]:
    merged = []
    seen = {primary_url}
    for url in [*base, *extra]:
        if not url or url in seen or _is_social_profile_url(url):
            continue
        seen.add(url)
        merged.append(url)
        if len(merged) >= limit:
            break
    return merged


def _is_social_profile_url(url: str) -> bool:
    """Skip bare social profile URLs in related links.

    Clustered social account homepages are useful during discovery, but they
    are poor evidence links and can look unrelated in the final article.
    """
    m = re.match(r"^https?://(?:www\.)?(?:x|twitter)\.com/([^/?#]+)/*(?:[?#].*)?$", url or "", re.I)
    if not m:
        return False
    account = m.group(1).lower()
    return account not in {"i", "home", "search", "explore"}


def _normalize_decision_sections(
    decisions: Dict[str, EditorialDecision],
    event_map: Dict[str, EventCluster],
) -> int:
    changed = 0
    for event_id, dec in decisions.items():
        old_section = dec.section or ""
        normalized = _normalize_section(old_section)
        if normalized != old_section:
            dec.section = normalized
            changed += 1
        if dec.decision != "select":
            continue
        evt = event_map.get(event_id)
        if evt is None:
            continue
        if dec.section == "要闻":
            if _can_stay_headline(evt):
                continue
            dec.section = _guess_section(evt)
            changed += 1
            continue
        if _is_actionable_pricing_headline(evt):
            if dec.section != "要闻":
                dec.section = "要闻"
                changed += 1
            continue
        guessed = _guess_section(evt)
        if guessed and guessed != dec.section:
            dec.section = guessed
            changed += 1
    return changed


def _normalize_section(section: str) -> str:
    if not section:
        return "行业动态"
    return SECTION_ALIASES.get(section, section if section in SECTION_ORDER else "行业动态")


def _is_actionable_pricing_headline(evt: EventCluster) -> bool:
    text = f"{evt.canonical_title} {evt.summary}".lower()
    meta = f"{evt.primary_content_type} {evt.primary_evidence_type} {evt.primary_source_tier}".lower()
    has_pricing_evidence = any(k in meta for k in (
        "pricing_page", "china_model_pricing", "official_docs",
    ))
    has_actionable_change = any(k in text for k in (
        "price", "pricing", "discount", "free", "byok",
        "降价", "定价", "价格", "优惠", "免费", "永久", "1/4", "2.5 折", "2.5折",
    ))
    return has_pricing_evidence and has_actionable_change


def _can_stay_headline(evt: EventCluster) -> bool:
    if _is_actionable_pricing_headline(evt):
        return True
    tier = (evt.primary_source_tier or "").lower()
    confidence = (evt.primary_confidence or "").lower()
    evidence = (evt.primary_evidence_type or "").lower()
    if "tier_3" in tier or confidence == "low":
        return False
    return evidence in {
        "official_release", "official_docs", "pricing_page", "paper",
        "research_paper", "financial_report", "benchmark_tracker",
        "github_release", "product_changelog",
    } or "tier_0" in tier or "tier_1" in tier


def _story_key(evt: Optional[EventCluster]) -> str:
    """Coarse duplicate key for same product/event split across posts."""
    if evt is None:
        return ""
    text = f"{evt.canonical_title} {evt.summary}".lower()
    if _is_google_io_story(text):
        return "google_io_2026"
    if _is_google_ai_edge_story(text):
        return "google_ai_edge"
    patterns = [
        r"\bqwen\d+(?:\.\d+)*(?:-[a-z0-9]+)?\b",
        r"\bgpt-\d+(?:\.\d+)*(?:-[a-z0-9]+)?\b",
        r"\bclaude\s*\d+(?:\.\d+)*(?:\s*[a-z]+)?\b",
        r"\bgemini\s*\d+(?:\.\d+)*(?:\s*[a-z]+)?\b",
        r"\bcopilot\s+for\s+eclipse\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(0).replace(" ", "")
    return ""


def _is_google_io_story(text: str) -> bool:
    return (
        "google i/o 2026" in text
        or "google io 2026" in text
        or "i/o 2026" in text
        or "io 2026" in text
        or "antigravity" in text
    )


def _is_google_ai_edge_story(text: str) -> bool:
    return (
        "google ai edge" in text
        or "ai edge gallery" in text
        or "litert-lm" in text
        or "litert lm" in text
        or ("gemma 4" in text and any(k in text for k in ("edge", "端侧", "on-device", "mobile")))
    )


def _source_cap_exceeded(
    evt: Optional[EventCluster],
    source_counts: Counter,
    *,
    priority: str,
) -> bool:
    if evt is None or not evt.source_names:
        return False
    cap = 3 if priority == "must_include" else 2
    return max((source_counts.get(s, 0) for s in evt.source_names), default=0) >= cap


def _is_material_history_update(evt: EventCluster) -> bool:
    text = f"{evt.canonical_title} {evt.summary}".lower()
    return any(k in text for k in (
        "follow-up", "now available", "general availability", "patch", "fix",
        "security patch", "new benchmark", "benchmark update", "price cut",
        "pricing change", "completed funding", "closed funding",
        "正式上线", "全面开放", "扩大开放", "新增", "补充", "修复",
        "安全补丁", "基准更新", "价格调整", "融资完成", "正式敲定",
    ))


def _fallback_why_it_matters(evt: EventCluster) -> str:
    text = " ".join((evt.summary or evt.canonical_title or "").split())
    if not text:
        return "该动态补充了今日 AI 行业的重要背景。"
    sentence = re.split(r"(?<=[。！？.!?])\s+", text)[0]
    return sentence[:90].rstrip("，,；; ") or evt.canonical_title[:90]


def _fallback_writing_angle(evt: EventCluster, section: str) -> str:
    angle_by_section = {
        "模型发布": "关注模型能力、可用性和开发者接入价值。",
        "开发生态": "关注它对开发流程、开源生态或本地部署的影响。",
        "产品应用": "关注真实用户场景和产品化路径。",
        "行业动态": "关注政策、商业、资金和行业格局信号。",
        "技术与洞察": "关注方法亮点、可信数据和工程启发。",
        "前瞻与传闻": "明确标注未确认状态，只写可核查线索。",
    }
    return angle_by_section.get(section, f"基于 {evt.source_count} 个来源提炼关键信息。")


def _normalized_source_meta(
    *,
    evt: EventCluster,
    primary_source_name: str,
    primary_url: str,
) -> Tuple[str, str, str, str]:
    tier = evt.primary_source_tier or ""
    reliability = evt.primary_reliability or ""
    evidence_type = evt.primary_evidence_type or ""
    confidence = evt.primary_confidence or "medium"

    if _is_official_primary(primary_source_name, primary_url):
        tier = "tier_0_core_evidence"
        reliability = "high"
        evidence_type = "official_release"
        confidence = "high"
    return tier, reliability, evidence_type, confidence


def _is_official_primary(source_name: str, url: str) -> bool:
    s = (source_name or "").lower()
    u = (url or "").lower()
    official_source_ids = {
        "openai_news", "anthropic_news", "google_ai_blog",
        "google_developers_blog", "google_deepmind_blog", "meta_ai_blog",
        "microsoft_ai_blog", "huggingface_blog", "ollama_releases",
        "x_openai", "x_anthropicai", "x_alibaba_qwen", "x_qwen",
        "x_tencent_hunyuan", "x_deepseek_ai", "x_googledeepmind",
        "x_stepfun",
    }
    official_url_markers = (
        "openai.com/index/",
        "anthropic.com/news/",
        "developers.googleblog.com/",
        "blog.google/technology/ai/",
        "deepmind.google/",
        "ai.meta.com/blog/",
        "github.com/ollama/ollama/releases/",
        "github.blog/changelog/",
        "x.com/openai/",
        "x.com/alibaba_qwen/",
        "x.com/tencenthunyuan/",
        "x.com/stepfun_ai/",
    )
    return s in official_source_ids or any(marker in u for marker in official_url_markers)


def _is_stale_background_event(evt: Optional[EventCluster]) -> bool:
    if evt is None:
        return False
    if getattr(evt, "already_reported", False) and not _is_material_history_update(evt):
        return True
    if _staleness_exempt(evt):
        return False
    age_h = _event_age_hours(evt)
    if age_h is None or age_h < 168:
        return False
    if age_h >= 720:
        return True
    text = f"{evt.canonical_title} {evt.summary}".lower()
    has_update = any(k in text for k in [
        "today", "now", "new", "launch", "release", "update", "announce",
        "发布", "推出", "上线", "更新", "宣布", "开源", "融资", "财报",
    ])
    return not has_update


def _event_age_hours(evt: EventCluster) -> Optional[float]:
    newest = evt.latest_seen_at or evt.published_at
    if not newest:
        return None
    try:
        dt = datetime.fromisoformat(newest.replace("Z", "+00:00"))
        return max(0.0, (time.time() - dt.timestamp()) / 3600.0)
    except Exception:
        return None


def _staleness_exempt(evt: EventCluster) -> bool:
    text = (
        f"{evt.primary_content_type} {evt.primary_evidence_type} "
        f"{' '.join(evt.source_types)} {' '.join(evt.source_names)} "
        f"{' '.join(evt.source_urls)}"
    ).lower()
    return (
        "arxiv" in text
        or "huggingface.co/papers" in text
        or "research_paper" in text
        or "pricing" in text
        or "official_docs" in text
    )


def _guess_section(evt: EventCluster) -> str:
    """Guess section from event content."""
    return _normalize_section(_classify_section(evt))
