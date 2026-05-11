"""Curator agent.

Deterministic filtering + ranking step that runs before the LLM writer. Deeply
boring on purpose: LLM budget should be spent on *writing*, not on
deduplicating.

Pipeline:
  1. normalize titles (lowercase, strip punctuation)
  2. drop exact-title and URL duplicates (assigning duplicate_group_id)
  3. drop items with empty title/url
  4. filter out items with zero AI-relevance (catches general-tech noise)
  5. score each item by source_weight × recency × relevance_boost × diversity
  6. sort by score desc, take top-N

Returns both the ``List[CuratedItem]`` (writer input, unchanged contract) and
a ``List[CuratedItemRecord]`` (persistence artifact).
"""

from __future__ import annotations

import hashlib
import math
import re
import time as _time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from agent.schemas import CuratedItem, CuratedItemRecord
from agent.sources.base import RawItem

_NORM_RE = re.compile(r"[^\w一-鿿]+", flags=re.UNICODE)

# ── AI relevance keywords for Chinese + English content ───────────────
# Items whose title+summary contain ZERO of these keywords are penalized
# heavily — they're likely general-tech news from broad-coverage feeds like
# IT之家 rather than actual AI stories.

_AI_KEYWORDS_CN = [
    # Model architecture
    "大模型", "语言模型", "多模态", "视觉模型", "语音模型", "推理模型",
    "开源模型", "预训练", "微调", "蒸馏", "量化", "对齐", "MoE",
    "Transformer", "扩散模型", "文生图", "文生视频", "图生视频",
    "embedding", "tokenizer", "分词",
    # Products & platforms
    "ChatGPT", "GPT-4", "GPT-5", "Claude", "Gemini", "DeepSeek", "Qwen",
    "通义千问", "文心一言", "ERNIE", "豆包", "Kimi", "ChatGLM", "智谱",
    "混元", "百川", "天工", "日日新", "盘古",
    # AI companies & labs
    "OpenAI", "Anthropic", "Google DeepMind", "Meta AI", "Mistral",
    "Stability AI", "xAI", "深度求索", "月之暗面", "阶跃星辰", "MiniMax",
    "零一万物", "商汤", "旷视", "面壁智能", "昆仑万维",
    # Core concepts
    "人工智能", "机器学习", "深度学习", "神经网络", "自然语言处理",
    "计算机视觉", "语音识别", "智能体", "agent", "RAG",
    "强化学习", "监督学习", "无监督学习", "自监督",
    "transformer", "attention", "diffusion", "generative",
    # Training & infra
    "训练", "推理", "GPU", "算力", "芯片", "MLOps", "向量数据库",
    "HuggingFace", "LangChain", "LlamaIndex", "vLLM",
    # Applications
    "自动驾驶", "机器人", "具身智能", "AI编程", "代码生成",
    "AI搜索", "AI医疗", "AI教育", "AI金融", "AI安全",
    "智能座舱", "AI手机", "AI写作", "AI绘画", "AI视频",
    "AI音乐", "AI翻译", "copilot", "codex",
    # Research & benchmarks
    "基准测试", "benchmark", "论文", "NeurIPS", "ICML", "ICLR", "CVPR",
    # Industry
    "融资", "估值", "IPO", "裁员",
]

_AI_KEYWORDS_EN = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "llm", "large language model", "gpt", "claude", "gemini", "deepseek",
    "transformer", "diffusion", "neural network", "generative",
    "nlp", "computer vision", "reinforcement learning", "rlhf",
    "fine-tune", "fine-tuned", "pre-train", "pre-trained",
    "agent", "multimodal", "embedding", "rag", "retrieval augmented",
    "open source model", "benchmark", "state-of-the-art",
    "copilot", "codex", "chatbot", "inference",
    "autonomous", "robotics", "alignment", "safety",
    "quantization", "lora", "qlora", "distillation",
]

_CN_GENERAL_SOURCES = {"ithome"}  # sources that cover broad tech, not just AI


def _is_ai_relevant(title: str, summary: str = "", source_id: str = "") -> bool:
    """Check if an item is AI-related based on keyword matching.

    For general-tech sources like IT之家, the bar is higher — both the
    Chinese AND English keyword lists must have zero matches to reject.
    For AI-focused sources, a single keyword match is sufficient to pass.
    """
    text = (title + " " + summary).lower()
    is_general_source = source_id in _CN_GENERAL_SOURCES

    cn_hits = sum(1 for kw in _AI_KEYWORDS_CN if kw.lower() in text)
    en_hits = sum(1 for kw in _AI_KEYWORDS_EN if kw in text)

    if is_general_source:
        # General sources need stronger evidence of AI relevance.
        # Require at least 1 Chinese keyword OR 2 English keywords.
        return cn_hits >= 1 or en_hits >= 2
    else:
        # AI-focused sources: one match is enough.
        return (cn_hits + en_hits) >= 1


def _relevance_boost(title: str, summary: str = "", source_id: str = "") -> float:
    """Compute an AI-relevance multiplier for the curation score.

    For general-tech sources (like IT之家) that cover broad news: items
    with zero AI keywords are hard-dropped (return 0.0).

    For AI-specialized sources: even zero keyword hits gets a pass (0.5)
    because the source itself guarantees relevance — e.g. a HuggingFace
    blog post about "datasets" is AI-relevant by definition, even if the
    title doesn't contain our keyword list.
    """
    text = (title + " " + summary).lower()
    cn_hits = sum(1 for kw in _AI_KEYWORDS_CN if kw.lower() in text)
    en_hits = sum(1 for kw in _AI_KEYWORDS_EN if kw in text)
    total_hits = cn_hits + en_hits

    is_general = source_id in _CN_GENERAL_SOURCES

    if total_hits >= 5:
        return 1.2   # strongly AI-related: slight boost
    if total_hits >= 2:
        return 1.0   # clearly AI-related: neutral
    if total_hits == 1:
        return 0.85  # marginally AI-related: mild penalty
    # Zero keyword hits.
    if is_general:
        return 0.0   # general source + no AI keywords = hard drop
    return 0.6       # AI source with no keywords: pass at reduced score


def _norm_title(title: str) -> str:
    return _NORM_RE.sub(" ", title.lower()).strip()


def _recency_score(published_at: str, now_ts: float) -> float:
    if not published_at:
        return 0.5
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        age_h = max(0.0, (now_ts - dt.timestamp()) / 3600.0)
    except Exception:
        return 0.5
    # Exponential decay with ~72h half-life.
    return math.exp(-age_h / 72.0)


def _dup_group_id(normalized_title: str) -> str:
    """Short hash of the normalized title — used as duplicate group id."""
    return hashlib.sha1(normalized_title.encode()).hexdigest()[:8]


def _raw_item_id(item: RawItem) -> str:
    return f"{item.source_id}::{item.url}"


def curate(
    items: List[RawItem],
    *,
    source_specs: List[Dict[str, Any]],
    max_items: int = 12,
) -> List[CuratedItem]:
    """Return the top-N curated items (writer input contract, unchanged)."""
    curated_items, _ = curate_with_records(
        items, source_specs=source_specs, max_items=max_items
    )
    return curated_items


def curate_with_records(
    items: List[RawItem],
    *,
    source_specs: List[Dict[str, Any]],
    max_items: int = 12,
) -> Tuple[List[CuratedItem], List[CuratedItemRecord]]:
    """Return (writer_items, persistence_records).

    ``writer_items`` is the same list ``curate()`` returns — the rest of the
    pipeline is unaware this function exists.
    ``persistence_records`` has one entry per *selected* item (used_in_draft=True).
    """
    weights: Dict[str, float] = {
        s.get("id", ""): float(s.get("weight", 1.0)) for s in source_specs
    }
    seen_titles: Dict[str, str] = {}   # norm_title → dup_group_id
    seen_urls: set = set()
    now_ts = _time.time()

    scored: List[Tuple[float, CuratedItem, CuratedItemRecord]] = []
    source_scored_count: Dict[str, int] = Counter()

    for it in items:
        if not it.title or not it.url:
            continue
        nt = _norm_title(it.title)
        if not nt:
            continue

        if nt in seen_titles:
            continue
        if it.url in seen_urls:
            continue

        # ── AI-relevance filter ──────────────────────────────────
        rel_boost = _relevance_boost(it.title, it.summary, it.source_id)
        if rel_boost == 0.0:
            continue  # hard drop: not AI-related at all

        seen_titles[nt] = _dup_group_id(nt)
        seen_urls.add(it.url)

        w = weights.get(it.source_id, 1.0)
        r = _recency_score(it.published_at, now_ts)

        # ── Source diversity: penalize items from over-represented sources ──
        source_count = source_scored_count.get(it.source_id, 0)
        diversity_penalty = 1.0
        if source_count >= 6:
            diversity_penalty = 0.6
        elif source_count >= 4:
            diversity_penalty = 0.75
        elif source_count >= 3:
            diversity_penalty = 0.85

        score = w * r * rel_boost * diversity_penalty

        source_scored_count[it.source_id] += 1

        reasons: list[str] = []
        if w != 1.0:
            reasons.append(f"source_weight={w:.2f}")
        reasons.append(f"recency={r:.3f}")
        if rel_boost != 1.0:
            reasons.append(f"relevance={rel_boost:.2f}")
        if diversity_penalty < 1.0:
            reasons.append(f"diversity={diversity_penalty:.2f}")

        curated = CuratedItem(
            title=it.title,
            url=it.url,
            summary=it.summary[:500],
            source=it.source_id,
            source_type=it.source_type,
            published_at=it.published_at,
            score=round(score, 4),
        )
        record = CuratedItemRecord(
            raw_item_id=_raw_item_id(it),
            title=it.title,
            source_url=it.url,
            source_name=it.source_id,
            published_at=it.published_at or None,
            score=round(score, 4),
            section=None,
            selected_reason="; ".join(reasons) if reasons else "recency",
            duplicate_group_id=None,
            used_in_draft=True,
        )
        scored.append((score, curated, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_items]
    writer_items = [c for _, c, _ in top]
    records = [rec for _, _, rec in top]
    return writer_items, records
