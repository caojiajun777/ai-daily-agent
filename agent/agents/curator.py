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

_CN_GENERAL_SOURCES = {"ithome"}
_AI_GUARANTEED_SOURCES = {"arxiv_top_venue", "arxiv_ai", "arxiv_cs", "hf_daily_papers"}


def _is_ai_relevant(title: str, summary: str = "", source_id: str = "") -> bool:
    """Check if an item is AI-related based on keyword matching.

    For general-tech sources like IT之家, the bar is higher — both the
    Chinese AND English keyword lists must have zero matches to reject.
    For AI-focused sources, a single keyword match is sufficient to pass.
    """
    text = (title + " " + summary).lower()
    is_general_source = source_id in _CN_GENERAL_SOURCES
    if source_id in _AI_GUARANTEED_SOURCES:
        return True  # arxiv papers are AI by definition

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
    is_guaranteed = source_id in _AI_GUARANTEED_SOURCES

    if is_guaranteed:
        return 1.0   # arxiv papers are AI by definition
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
    top = _select_with_paper_quota(scored, max_items, min_papers=5)
    writer_items = [c for _, c, _ in top]
    records = [rec for _, _, rec in top]
    return writer_items, records


_MIN_PAPERS = 5


def _select_with_paper_quota(
    scored: List[Tuple[float, CuratedItem, CuratedItemRecord]],
    max_items: int,
    min_papers: int = 5,
) -> List[Tuple[float, CuratedItem, CuratedItemRecord]]:
    """Select top-N items, ensuring at least min_papers arxiv papers.

    If the top-N doesn't include enough arxiv papers, we pull the
    highest-scoring arxiv papers from ANY position in the scored list
    and swap them in, pushing out the lowest-scoring non-arxiv items.
    """
    # Find ALL arxiv items in the full scored list and their best scores.
    all_arxiv = [(s, c, r) for s, c, r in scored if "arxiv" in r.source_name]
    all_arxiv.sort(key=lambda x: x[0], reverse=True)

    # Separate arxiv and non-arxiv in the top-N.
    top = list(scored[:max_items])
    arxiv_in_top = [(s, c, r) for s, c, r in top if "arxiv" in r.source_name]

    if len(arxiv_in_top) >= min_papers:
        return top

    # Find the best arxiv papers that are NOT in the top (or use all if no more).
    top_arxiv_ids = {r.raw_item_id for _, _, r in arxiv_in_top}
    arxiv_not_in_top = [(s, c, r) for s, c, r in all_arxiv if r.raw_item_id not in top_arxiv_ids]
    arxiv_not_in_top.sort(key=lambda x: x[0], reverse=True)

    needed = min_papers - len(arxiv_in_top)
    to_add = arxiv_not_in_top[:needed]

    # Remove lowest-scoring non-arxiv from top to make room.
    non_arxiv_top = [(s, c, r) for s, c, r in top if "arxiv" not in r.source_name]
    non_arxiv_top.sort(key=lambda x: x[0])  # ascending — lowest first
    to_remove = non_arxiv_top[:len(to_add)]

    remove_ids = {r.raw_item_id for _, _, r in to_remove}
    result = [(s, c, r) for s, c, r in top if r.raw_item_id not in remove_ids]
    result.extend(to_add)
    result.sort(key=lambda x: x[0], reverse=True)
    return result[:max_items]


# ═══════════════════════════════════════════════════════════════════════
# LLM-powered curation — replaces deterministic scoring with LLM judgment
# ═══════════════════════════════════════════════════════════════════════

_LLM_CURATION_PROMPT = """你是一个 AI 新闻编辑。你需要从一批 AI 领域候选资讯中选出今天最值得报道的条目。

评分标准（每项 1-10 分）：
- **独家性**：这条消息是否是独家/首发？是否来自一手官方渠道？（独家爆料、官方首发=高分，转述/旧闻=低分）
- **影响力**：对 AI 行业、开发者或用户的影响有多大？（模型发布、重大融资、政策变化=高分）
- **时效性**：是否是最近 24 小时内的新消息？
- **稀缺性**：这类消息在其他源中是否少见？（同质化严重的模型发布可以适当降低）

输出严格 JSON 数组，每个元素为：
{
  "index": 候选列表中的序号(从0开始),
  "title": "原标题",
  "score": 1-10 的整数,
  "reason": "一句话说明评分理由"
}

只返回得分 >= 5 的条目，按得分从高到低排列。最多返回 25 条。"""


def llm_score_items(
    *,
    items: List[RawItem],
    provider,
    tracer=None,
    budget=None,
    max_to_score: int = 30,
) -> Dict[str, float]:
    """Ask an LLM to score candidate items by news importance.

    Returns a dict mapping ``source_id::url`` → LLM importance score (0.0-1.0).
    Items not scored by the LLM default to 0.5 (neutral).
    """
    from agent.llm.base import LLMMessage

    # Truncate to manageable size — LLM scores the top candidates.
    to_score = items[:max_to_score]

    # Build candidate list for the prompt.
    item_lines: List[str] = []
    for idx, it in enumerate(to_score):
        src_label = it.source_id
        summary_short = it.summary[:150].replace("\n", " ")
        item_lines.append(f"[{idx}] [{src_label}] {it.title}")
        if summary_short:
            item_lines.append(f"    摘要: {summary_short}")

    user_msg = "请对以下 AI 新闻候选条目打分：\n\n" + "\n".join(item_lines)

    llm_scores: Dict[str, float] = {}

    try:
        if budget:
            budget.check_can_call(stage="curate")

        response = provider.complete(
            messages=[
                LLMMessage(role="system", content=_LLM_CURATION_PROMPT),
                LLMMessage(role="user", content=user_msg),
            ],
            temperature=0.1,
            max_output_tokens=2048,
        )

        if tracer:
            tracer.log_llm_call(
                provider=provider.name, model=provider.model,
                prompt=_LLM_CURATION_PROMPT + "\n" + user_msg,
                output=response.text, latency_ms=response.latency_ms,
                status="ok", stage="curate",
            )

        if budget:
            budget.record(
                stage="curate", input_tokens=response.input_tokens_est,
                output_tokens=response.output_tokens_est,
            )

        # Parse LLM output.
        import json as _json, re as _re
        raw = response.text.strip()
        # Strip think blocks and code fences.
        raw = _re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=_re.IGNORECASE).strip()
        m = _re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()
        else:
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                raw = raw[start:end + 1]

        scored_list = _json.loads(raw)
        if not isinstance(scored_list, list):
            return llm_scores

        for entry in scored_list:
            idx = int(entry.get("index", -1))
            score = float(entry.get("score", 5))
            if 0 <= idx < len(to_score):
                key = f"{to_score[idx].source_id}::{to_score[idx].url}"
                llm_scores[key] = min(1.0, max(0.1, score / 10.0))

    except Exception as e:
        if tracer:
            tracer.log("llm_curation_failed", error=str(e))

    return llm_scores


def curate_with_llm(
    *,
    items: List[RawItem],
    source_specs: List[Dict[str, Any]],
    provider,
    max_items: int = 20,
    tracer=None,
    budget=None,
) -> Tuple[List[CuratedItem], List[CuratedItemRecord]]:
    """Curate items using LLM importance scoring + deterministic filtering.

    Flow:
      1. Deterministic dedup + AI-relevance filter (same as before).
      2. LLM scores remaining items by news importance (独家性/影响力).
      3. Combine: LLM_score × 0.6 + deterministic_score × 0.4.
      4. Apply source diversity penalty to the combined score.
      5. Sort and take top-N.
    """
    weights: Dict[str, float] = {
        s.get("id", ""): float(s.get("weight", 1.0)) for s in source_specs
    }
    seen_titles: Dict[str, str] = {}
    seen_urls: set = set()
    now_ts = _time.time()

    # Phase 1: Dedup + AI-relevance filter (deterministic).
    deduped: List[RawItem] = []
    for it in items:
        if not it.title or not it.url:
            continue
        nt = _norm_title(it.title)
        if not nt or nt in seen_titles or it.url in seen_urls:
            continue
        if _relevance_boost(it.title, it.summary, it.source_id) == 0.0:
            continue
        seen_titles[nt] = _dup_group_id(nt)
        seen_urls.add(it.url)
        deduped.append(it)

    # Phase 2: LLM importance scoring.
    llm_scores = llm_score_items(
        items=deduped, provider=provider, tracer=tracer, budget=budget,
    )

    # Phase 3: Combine scores.
    scored: List[Tuple[float, CuratedItem, CuratedItemRecord]] = []
    source_scored_count: Dict[str, int] = Counter()

    for it in deduped:
        w = weights.get(it.source_id, 1.0)
        r = _recency_score(it.published_at, now_ts)
        det_score = w * r

        key = f"{it.source_id}::{it.url}"
        llm_s = llm_scores.get(key, 0.5)  # default neutral if LLM didn't score

        combined = llm_s * 0.65 + det_score * 0.35

        # Source diversity.
        source_count = source_scored_count.get(it.source_id, 0)
        div_penalty = 1.0
        if source_count >= 6:
            div_penalty = 0.6
        elif source_count >= 4:
            div_penalty = 0.75
        elif source_count >= 3:
            div_penalty = 0.85

        final_score = combined * div_penalty
        source_scored_count[it.source_id] += 1

        reasons = [f"llm={llm_s:.2f}", f"det={det_score:.2f}"]
        if div_penalty < 1.0:
            reasons.append(f"div={div_penalty:.2f}")

        curated = CuratedItem(
            title=it.title, url=it.url, summary=it.summary[:500],
            source=it.source_id, source_type=it.source_type,
            published_at=it.published_at, score=round(final_score, 4),
        )
        record = CuratedItemRecord(
            raw_item_id=_raw_item_id(it), title=it.title,
            source_url=it.url, source_name=it.source_id,
            published_at=it.published_at or None,
            score=round(final_score, 4), section=None,
            selected_reason="; ".join(reasons),
            duplicate_group_id=None, used_in_draft=True,
        )
        scored.append((final_score, curated, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = _select_with_paper_quota(scored, max_items, min_papers=_MIN_PAPERS)
    return [c for _, c, _ in top], [rec for _, _, rec in top]
