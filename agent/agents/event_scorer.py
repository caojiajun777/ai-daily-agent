"""Rule-based Event Scoring.

Scores EventClusters on multiple deterministic dimensions. Produces a
ranked list even when LLM is unavailable — this is the fallback backbone.
"""

from __future__ import annotations

import math
import re
import time as _time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from agent.agents.event_clusterer import EventCluster

# ── Source authority tiers ─────────────────────────────────────────────
_OFFICIAL_PATTERNS = [
    "openai.com", "anthropic.com", "deepmind.google", "ai.meta.com",
    "blog.google", "blogs.microsoft.com", "mistral.ai", "stability.ai",
    "huggingface.co", "paperswithcode.com", "arxiv.org", "github.com",
    "baidu", "alibaba", "qwen", "stepfun", "tencent", "deepseek",
    "zhipu", "minimax", "moonshot",
]
_TRUSTED_MEDIA = [
    "technologyreview.com", "wired.com", "venturebeat.com",
    "the-decoder.com", "arstechnica.com", "theinformation.com",
    "techcrunch.com", "theverge.com",
    "bloomberg", "cnbc", "reuters", "wsj", "ft.com", "axios",
]
_KOL_PATTERNS = ["kaboroje", "ylecun", "fchollet", "AndrewYNg",
                 "jimfan", "sama", "gdb", "dotey", "AYi_AInotes"]


def score_events(
    events: List[EventCluster],
    *,
    history_titles: Optional[List[str]] = None,
    max_items: int = 40,
) -> List[EventCluster]:
    """Score events and return top-N sorted by rule_score desc."""
    now_ts = _time.time()
    history_set = set(_norm_history(t) for t in (history_titles or []))

    for evt in events:
        evt.rule_score = _score_one(evt, now_ts, history_set, history_titles)

    events.sort(key=lambda e: e.rule_score, reverse=True)
    return events[:max_items]


def _score_one(evt: EventCluster, now_ts: float, history_set: set,
               history_titles: Optional[List[str]] = None) -> float:
    ai_rel = _ai_relevance_score(evt)          # 0-1
    novelty = _event_novelty(evt)               # 0-1
    reader_u = _reader_utility(evt)             # 0-1
    impact = _impact_scope(evt)                 # 0-1
    evidence = _evidence_quality(evt)           # 0-1
    freshness = _event_freshness(evt, now_ts)   # 0-1
    authority = _source_authority(evt)          # 0-1

    score = (
        0.22 * ai_rel
        + 0.18 * novelty
        + 0.16 * reader_u
        + 0.14 * impact
        + 0.12 * evidence
        + 0.10 * freshness
        + 0.08 * authority
    )
    score += _metadata_boost(evt)
    score = _apply_staleness_gate(evt, score, now_ts)

    # ── Penalties ──────────────────────────────────────────────────────
    # Already reported penalty.
    norm_title = _norm_for_history(evt.canonical_title)
    if any(_title_overlap(norm_title, ht) > 0.70 for ht in history_set):
        if history_titles and _is_meaningful_update(evt):
            score -= 0.04
        else:
            score -= 0.12

    # Marketing/clickbait penalty.
    marketing_words = ["独家", "重磅", "炸裂", "震惊", "突发", "颠覆",
                       "碾压", "bombshell", "game-changer", "must-read"]
    title_lower = evt.canonical_title.lower()
    if any(w in title_lower for w in marketing_words):
        score -= 0.08

    # Low information: single source, no body text.
    if evt.source_count == 1 and len(evt.summary) < 60:
        score -= 0.10

    # Rumor penalty: no official source, only KOL/secondary.
    if not any(_is_official(s) for s in evt.source_names):
        score -= 0.05

    # Over-represented: mostly same category source.
    if evt.source_count >= 3 and _same_category_ratio(evt) > 0.8:
        score -= 0.04

    # Breaking news boost: multi-source events & official sources get priority.
    if evt.source_count >= 3:
        score *= 1.12
    elif any(_is_official(s) for s in evt.source_names) and evt.source_count >= 2:
        score *= 1.06

    return max(0.0, round(score, 4))


def _metadata_boost(evt: EventCluster) -> float:
    """Use source config metadata as a small authority prior."""
    tier = (evt.primary_source_tier or "").lower()
    ctype = (evt.primary_content_type or "").lower()
    etype = (evt.primary_evidence_type or "").lower()

    boost = 0.0
    if "tier_0" in tier:
        boost += 0.08
    elif "tier_1" in tier:
        boost += 0.04
    elif "tier_3" in tier:
        boost -= 0.08

    high_signal_types = (
        "official", "pricing", "benchmark", "research_paper",
        "financial_report", "github_release", "product_changelog",
    )
    if any(k in ctype or k in etype for k in high_signal_types):
        boost += 0.03
    if "community" in ctype or "market_commentary" in ctype:
        boost -= 0.04
    return boost


# ── Dimension scoring helpers ──────────────────────────────────────────

_AI_KEYWORDS_LOWER = [
    "ai", "model", "llm", "gpt", "claude", "gemini", "deepseek",
    "openai", "anthropic", "transformer", "diffusion", "neural",
    "fine-tune", "training", "inference", "benchmark", "agent",
    "multimodal", "embedding", "rag", "copilot", "codex",
    "大模型", "模型", "智能", "推理", "训练", "智能体", "开源",
    "多模态", "语音", "机器人", "自动驾驶", "算力", "芯片",
    "earnings", "revenue", "财务", "财报", "营收", "净利润",
    "融资", "funding", "ipo", "估值", "acquisition", "收购",
]


def _ai_relevance_score(evt: EventCluster) -> float:
    text = (evt.canonical_title + " " + evt.summary).lower()
    hits = sum(1 for kw in _AI_KEYWORDS_LOWER if kw in text)
    if hits >= 5: return 1.0
    if hits >= 3: return 0.85
    if hits >= 1: return 0.65
    return 0.30


def _event_novelty(evt: EventCluster) -> float:
    # Multi-source = more likely novel event.
    if evt.source_count >= 4: return 1.0
    if evt.source_count >= 2: return 0.8
    return 0.55


def _reader_utility(evt: EventCluster) -> float:
    text = (evt.canonical_title + " " + evt.summary).lower()
    score = 0.5
    if any(k in text for k in ["release", "发布", "开源", "open source",
                                 "benchmark", "基准", "github", "pricing",
                                 "价格", "api", "docs", "文档"]):
        score += 0.20
    if any(k in text for k in ["launch", "上线", "推出", "rollout",
                                 "announce", "发布", "preview", "unveil"]):
        score += 0.15
    if any(k in text for k in ["how-to", "tutorial", "教程", "guide",
                                 "实践", "示例", "example"]):
        score += 0.15
    if any(k in text for k in ["opinion", "分析", "insight", "趋势",
                                 "预测", "观点"]):
        score += 0.05
    if any(k in text for k in ["earnings", "财报", "revenue", "营收",
                                 "funding", "融资", "ipo", "acquisition",
                                 "收购", "valuation", "估值"]):
        score += 0.10
    return min(1.0, score)


def _impact_scope(evt: EventCluster) -> float:
    text = evt.canonical_title.lower()
    if any(k in text for k in ["gpt", "claude", "gemini", "deepseek",
                                 "qwen", "ernie", "chatgpt"]):
        return 1.0
    if evt.source_count >= 3:
        return 0.8
    if any(k in evt.summary.lower() for k in ["billion", "亿", "ipo",
                                                 "融资", "acquisition",
                                                 "收购", "regulation",
                                                 "监管", "policy", "政策",
                                                 "earnings", "财报", "revenue",
                                                 "营收", "funding", "invest",
                                                 "投资", "估值", "valuation"]):
        return 0.85
    return 0.55


def _evidence_quality(evt: EventCluster) -> float:
    score = 0.40
    if any("benchmark" in (s.lower()) for s in evt.snippets): score += 0.15
    if any("github" in s.lower() for s in evt.source_urls): score += 0.15
    if any("arxiv" in s.lower() for s in evt.source_urls): score += 0.15
    if any("blog" in s.lower() or "index" in s.lower()
           for s in evt.source_urls): score += 0.10
    if evt.source_count >= 3: score += 0.10
    if evt.source_count >= 5: score += 0.05
    return min(1.0, score)


def _event_freshness(evt: EventCluster, now_ts: float) -> float:
    newest = evt.latest_seen_at or evt.published_at
    if not newest:
        return 0.5
    try:
        dt = datetime.fromisoformat(newest.replace("Z", "+00:00"))
        age_h = max(0.0, (now_ts - dt.timestamp()) / 3600.0)
        return math.exp(-age_h / 72.0)
    except Exception:
        return 0.5


def _event_age_hours(evt: EventCluster, now_ts: float) -> Optional[float]:
    newest = evt.latest_seen_at or evt.published_at
    if not newest:
        return None
    try:
        dt = datetime.fromisoformat(newest.replace("Z", "+00:00"))
        return max(0.0, (now_ts - dt.timestamp()) / 3600.0)
    except Exception:
        return None


def _apply_staleness_gate(evt: EventCluster, score: float, now_ts: float) -> float:
    """Hard-cap stale background/news items so old explainers don't resurface.

    Some high-authority feeds re-emit evergreen background stories. They can
    look important because they mention major labs/models, but a daily should
    only include them when there is a fresh update signal.
    """
    age_h = _event_age_hours(evt, now_ts)
    if age_h is None:
        return score
    if _is_staleness_exempt(evt):
        return score

    text = (evt.canonical_title + " " + evt.summary).lower()
    has_update = _is_meaningful_update(evt) and any(k in text for k in [
        "today", "now", "new", "launch", "release", "update", "announce",
        "发布", "推出", "上线", "更新", "宣布", "开源", "融资", "财报",
    ])

    if age_h >= 720:                         # 30+ days: not daily material.
        return min(score, 0.12)
    if age_h >= 168 and not has_update:      # 7+ days: stale for daily news.
        return min(score, 0.28)
    if age_h >= 96 and not has_update:       # 4+ days: strongly demote.
        return min(score, 0.45)
    return score


def _is_staleness_exempt(evt: EventCluster) -> bool:
    text = (
        f"{evt.primary_content_type} {evt.primary_evidence_type} "
        f"{' '.join(evt.source_types)} {' '.join(evt.source_names)} "
        f"{' '.join(evt.source_urls)}"
    ).lower()
    if "arxiv" in text or "huggingface.co/papers" in text:
        return True
    if "research_paper" in text or "paper" in text:
        return True
    if "pricing" in text or "official_docs" in text:
        return True
    return False


def _source_authority(evt: EventCluster) -> float:
    official = sum(1 for s in evt.source_names if _is_official(s))
    media = sum(1 for s in evt.source_names if _is_trusted_media(s))
    kol = sum(1 for s in evt.source_names if _is_kol(s))
    total = len(evt.source_names) or 1
    return min(1.0, (official * 0.40 + media * 0.25 + kol * 0.15) / total + 0.30)


# ── Helpers ────────────────────────────────────────────────────────────

def _is_official(source_name: str) -> bool:
    s = source_name.lower()
    return any(p in s for p in _OFFICIAL_PATTERNS)


def _is_trusted_media(source_name: str) -> bool:
    s = source_name.lower()
    return any(p in s for p in _TRUSTED_MEDIA)


def _is_kol(source_name: str) -> bool:
    return source_name.lower() in _KOL_PATTERNS


def _same_category_ratio(evt: EventCluster) -> float:
    if not evt.source_types:
        return 0.0
    from collections import Counter
    cnt = Counter(evt.source_types)
    return max(cnt.values()) / len(evt.source_types)


def _norm_history(title: str) -> str:
    return re.sub(r"[^\w一-鿿]", "", title.lower())


def _norm_for_history(title: str) -> str:
    return _norm_history(title)


def _is_meaningful_update(evt: EventCluster) -> bool:
    """Check if a history-overlapping event is a meaningful update rather than a repeat."""
    text = (evt.canonical_title + " " + evt.summary).lower()
    update_signals = [
        "update", "upgrade", "release", "launch", "publish",
        "benchmark", "price", "pricing", "github", "repo",
        "rollout", "deprecate", "security", "patch", "fix",
        "更新", "发布", "升级", "上线", "开源", "降价",
        "财报", "earnings", "revenue", "营收", "融资",
    ]
    return any(s in text for s in update_signals)


def _title_overlap(a: str, b: str) -> float:
    """Subsequence overlap score — avoids false positives from character-set collision.

    Uses SequenceMatcher (same algorithm as history_checker) to measure
    genuine textual overlap, preventing brand-name sharing (e.g. 'Gemini 2.5'
    vs 'Gemini 3.5') from triggering false positive history penalties.
    """
    if not a or not b:
        return 0.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()
