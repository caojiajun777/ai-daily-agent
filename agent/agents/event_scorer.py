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
        evt.rule_score = _score_one(evt, now_ts, history_set)

    events.sort(key=lambda e: e.rule_score, reverse=True)
    return events[:max_items]


def _score_one(evt: EventCluster, now_ts: float, history_set: set) -> float:
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

    # ── Penalties ──────────────────────────────────────────────────────
    # Already reported penalty.
    norm_title = _norm_for_history(evt.canonical_title)
    if any(_title_overlap(norm_title, ht) > 0.70 for ht in history_set):
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

    return max(0.0, round(score, 4))


# ── Dimension scoring helpers ──────────────────────────────────────────

_AI_KEYWORDS_LOWER = [
    "ai", "model", "llm", "gpt", "claude", "gemini", "deepseek",
    "openai", "anthropic", "transformer", "diffusion", "neural",
    "fine-tune", "training", "inference", "benchmark", "agent",
    "multimodal", "embedding", "rag", "copilot", "codex",
    "大模型", "模型", "智能", "推理", "训练", "智能体", "开源",
    "多模态", "语音", "机器人", "自动驾驶", "算力", "芯片",
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
    if any(k in text for k in ["how-to", "tutorial", "教程", "guide",
                                 "实践", "示例", "example"]):
        score += 0.15
    if any(k in text for k in ["opinion", "分析", "insight", "趋势",
                                 "预测", "观点"]):
        score += 0.05
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
                                                "监管", "policy", "政策"]):
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


def _title_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))
