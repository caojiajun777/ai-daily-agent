"""Trend Metrics Calculator.

Computes quantitative signals for trend detection, so the LLM
doesn't rely purely on intuition. All values in [0, 1] unless noted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

_PRIORITY_WEIGHT = {
    "must_include": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0,
    "reject": 0.2, None: 1.0,
}
_EVIDENCE_WEIGHT = {
    "official": 1.3, "primary": 1.2, "trusted_media": 1.0,
    "social": 0.7, "weak": 0.4, None: 0.8,
}


def compute_event_metrics(
    events: List[Dict],
    *,
    final_selected: Optional[set] = None,
) -> List[Dict]:
    """Annotate each event with computed metrics. Mutates and returns the list."""
    max_priority = max(
        (_PRIORITY_WEIGHT.get(e.get("priority"), 1.0) for e in events),
        default=1.0,
    )
    max_evidence = max(
        (_EVIDENCE_WEIGHT.get(e.get("evidence_level"), 0.8) for e in events),
        default=1.0,
    )
    selected_set = final_selected or set()

    for e in events:
        pw = _PRIORITY_WEIGHT.get(e.get("priority"), 1.0)
        ew = _EVIDENCE_WEIGHT.get(e.get("evidence_level"), 0.8)
        sel_boost = 1.5 if e.get("event_id") in selected_set else 0.6

        e["_priority_w"] = round(pw / max_priority, 3) if max_priority else 0.0
        e["_evidence_w"] = round(ew / max_evidence, 3) if max_evidence else 0.0
        e["_weighted_attention"] = round(pw * ew * sel_boost, 3)
        e["_is_selected"] = e.get("event_id") in selected_set

    return events


def compute_trend_signals(
    group: List[Dict],
    *,
    window_days: int,
    all_events: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Compute trend signals for a group of related events."""

    if not group:
        return _empty_signals()

    n = len(group)
    dates = sorted(set(e.get("date", "") for e in group if e.get("date")))
    active_days = len(dates)
    sources = list(set(s for e in group for s in e.get("source_names", [])))

    wa = sum(e.get("_weighted_attention", 0) for e in group)
    prev_wa = _previous_window_wa(group, all_events or [], window_days)

    # Momentum: ratio of recent to older attention.
    mid = len(dates) // 2 if len(dates) >= 2 else 1
    recent_dates = set(dates[mid:]) if len(dates) >= 2 else set(dates)
    older_dates = set(dates[:mid]) if len(dates) >= 2 else set()
    recent_wa = sum(
        e.get("_weighted_attention", 0)
        for e in group if e.get("date") in recent_dates
    )
    older_wa = sum(
        e.get("_weighted_attention", 0)
        for e in group if e.get("date") in older_dates
    )
    momentum = _safe_div(recent_wa - older_wa, max(recent_wa + older_wa, 0.001), 0)

    # Persistence: how continuously this appears across days.
    persistence = active_days / max(window_days, 1)

    # Source diversity: unique sources / total sources ratio.
    source_diversity = len(sources) / max(n, 1)

    # Evidence strength: average evidence weight.
    evidence_strength = sum(e.get("_evidence_w", 0) for e in group) / max(n, 1)

    # Novelty ratio: how many events are "new" vs previously seen.
    novelty = sum(1 for e in group if e.get("novelty") == "new_event") / max(n, 1)

    # Selected ratio.
    selected_ratio = sum(1 for e in group if e.get("_is_selected")) / max(n, 1)

    return {
        "event_count": n,
        "active_days": active_days,
        "weighted_attention": round(wa, 3),
        "previous_weighted_attention": round(prev_wa, 3),
        "momentum": round(momentum, 3),
        "persistence": round(min(persistence, 1.0), 3),
        "source_diversity": round(min(source_diversity, 1.0), 3),
        "evidence_strength": round(min(evidence_strength, 1.0), 3),
        "novelty_ratio": round(min(novelty, 1.0), 3),
        "selected_ratio": round(min(selected_ratio, 1.0), 3),
        "unique_sources": sources[:10],
        "date_range": [dates[0], dates[-1]] if dates else [],
    }


def _empty_signals() -> Dict[str, Any]:
    return {
        "event_count": 0, "active_days": 0,
        "weighted_attention": 0, "previous_weighted_attention": 0,
        "momentum": 0, "persistence": 0, "source_diversity": 0,
        "evidence_strength": 0, "novelty_ratio": 0, "selected_ratio": 0,
        "unique_sources": [], "date_range": [],
    }


def _previous_window_wa(
    group: List[Dict],
    all_events: List[Dict],
    window_days: int,
) -> float:
    """Estimate attention in the previous window for momentum calc."""
    # Simplified: use previous-week weighted attention if available,
    # else estimate as half of current.
    older = [e for e in all_events if e not in group]
    if not older:
        return sum(e.get("_weighted_attention", 0) for e in group) * 0.5
    return sum(e.get("_weighted_attention", 0) for e in older) / max(len(older), 1)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


# ── Taxonomy tagging ────────────────────────────────────────────────────────

_TAXONOMY_MAP: Dict[str, List[str]] = {
    "model_reasoning": ["reasoning", "reason", "think", "chain-of-thought", "cot"],
    "model_multimodal": ["multimodal", "vision-language", "vl", "visual"],
    "model_video": ["video", "vidgen", "sora", "kling", "runway"],
    "model_speech": ["speech", "voice", "tts", "asr", "audio", "whisper"],
    "model_long_context": ["long context", "1m token", "128k", "1m context"],
    "model_small": ["small model", "3b", "1b", "7b", "mobile", "edge", "lite"],
    "model_open_weight": ["open source", "open weight", "weights", "opensource"],
    "agent_coding": ["codex", "coding agent", "dev agent", "copilot", "claude code"],
    "agent_browser": ["browser agent", "web agent", "browser use"],
    "agent_computer_use": ["computer use", "desktop", "screen", "gui agent"],
    "agent_workflow": ["workflow", "orchestrat", "agent team", "multi-agent system"],
    "agent_enterprise": ["enterprise agent", "b2b", "deployment", "production agent"],
    "agent_multi": ["multi agent", "swarm", "agentic"],
    "product_search": ["ai search", "deep search", "perplexity", "search engine"],
    "product_office": ["ai office", "ai doc", "copilot", "notion ai"],
    "product_ide": ["ai ide", "cursor", "windsurf", "code editor"],
    "product_phone": ["ai phone", "iphone", "galaxy ai"],
    "product_wearable": ["wearable", "glasses", "watch", "air", "pin"],
    "infra_inference": ["inference", "serving", "vllm", "tensorrt"],
    "infra_compression": ["quantization", "pruning", "distillation", "gguf", "gptq"],
    "infra_gpu": ["gpu", "h100", "b100", "a100", "nvidia"],
    "infra_runtime": ["agent runtime", "sandbox", "e2b", "modal"],
    "infra_observability": ["observability", "tracing", "langfuse", "langsmith"],
    "infra_vector": ["vector db", "pinecone", "qdrant", "chroma", "faiss"],
    "business_funding": ["funding", "raised", "series", "seed", "valuation"],
    "business_acquisition": ["acquisition", "acquired", "merger", "bought"],
    "business_partnership": ["partnership", "partnered", "alliance", "collaboration"],
    "business_deployment": ["deployment", "enterprise", "production", "rollout"],
    "business_pricing": ["pricing", "price", "free tier", "subscription", "discount"],
    "safety_regulation": ["regulation", "regulatory", "eu ai act", "executive order"],
    "safety_copyright": ["copyright", "ip", "intellectual property", "lawsuit"],
    "safety_security": ["security", "jailbreak", "prompt injection", "cyber"],
    "safety_model": ["model safety", "alignment", "guardrails", "red team"],
    "safety_privacy": ["privacy", "gdpr", "data protection", "pii"],
    "research_benchmark": ["benchmark", "eval", "mmlu", "humaneval", "sota"],
    "research_paper": ["paper", "arxiv", "published", "accepted"],
    "research_dataset": ["dataset", "corpus", "data", "curation"],
    "research_architecture": ["architecture", "transformer", "diffusion", "mamba"],
    "research_evaluation": ["evaluation", "metric", "scoring", "results"],
}


def tag_event(
    title: str = "", summary: str = "", section: str = "",
    why_it_matters: str = "", writing_angle: str = "",
) -> List[str]:
    """Assign taxonomy tags to an event based on keyword matching."""
    text = f"{title} {summary} {section} {why_it_matters} {writing_angle}".lower()
    tags: List[str] = []
    for tag, keywords in _TAXONOMY_MAP.items():
        if any(kw in text for kw in keywords):
            tags.append(tag)
    return tags


def summarize_tags(events: List[Dict]) -> Dict[str, int]:
    """Count taxonomy tags across a group of events."""
    counts: Counter = Counter()
    for e in events:
        for t in e.get("_tags", []):
            counts[t] += 1
    return dict(counts.most_common(20))
