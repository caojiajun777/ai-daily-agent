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


def _recency_score(published_at: str, now_ts: float, half_life_h: float) -> float:
    """True exponential decay with configurable half-life.

    recency = exp(-ln(2) * age_h / half_life_h)

    At age = half_life_h the score is exactly 0.5.
    """
    if not published_at:
        return 0.5
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        age_h = max(0.0, (now_ts - dt.timestamp()) / 3600.0)
    except Exception:
        return 0.5
    if half_life_h <= 0:
        return 1.0
    return math.exp(-math.log(2) * age_h / half_life_h)


# ── Content-type inference from source config ───────────────────────
# Maps source characteristics to content_types config keys.

_CONTENT_TYPE_RULES: list[tuple[str, str]] = [
    # (pattern, content_type) — first match wins, patterns against source_id lowercase
    # ── Papers & Research ──
    ("arxiv", "research_paper"),
    ("hf_daily_papers", "research_paper"),
    ("paperswithcode", "research_paper"),
    ("_paper", "research_paper"),
    # ── Pricing & Docs ──
    ("_pricing", "pricing_page"),
    ("_api_docs", "official_docs"),
    ("_docs", "official_docs"),
    ("modelscope", "china_ecosystem_signal"),
    # ── Benchmark ──
    ("_benchmark", "benchmark_tracker"),
    ("_leaderboard", "benchmark_tracker"),
    ("livebench", "benchmark_tracker"),
    ("swebench", "benchmark_tracker"),
    ("aider_", "benchmark_tracker"),
    ("lmarena", "benchmark_tracker"),
    ("artificial_analysis", "benchmark_tracker"),
    # ── Infra & Financial ──
    ("_ir", "financial_report"),
    ("_ir_", "financial_report"),
    ("nvidia_", "infra_signal"),
    ("epoch_ai", "infra_signal"),
    ("semianalysis", "infra_signal"),
    ("vllm_", "infra_signal"),
    ("sglang_", "infra_signal"),
    ("ollama_", "infra_signal"),
    ("together_ai", "infra_signal"),
    ("fireworks_ai", "infra_signal"),
    ("groq_", "infra_signal"),
    ("cerebras", "infra_signal"),
    ("modal_blog", "infra_signal"),
    ("baseten_", "infra_signal"),
    # ── Insider Media ──
    ("the_information", "insider_media"),
    ("bloomberg", "insider_media"),
    ("wsj_", "insider_media"),
    ("reuters", "insider_media"),
    ("axios_", "insider_media"),
    ("semafor", "insider_media"),
    ("businessinsider", "insider_media"),
    ("nytimes", "insider_media"),
    # ── VC / Founder / Reporter signals ──
    ("vc_", "vc_signal"),
    ("reporter_", "insider_reporter_signal"),
    ("founder_", "founder_signal"),
    ("x_sama", "founder_signal"),
    ("x_gdb", "founder_signal"),
    # ── Researcher & Expert signals ──
    ("x_karpathy", "researcher_signal"),
    ("x_ilyasut", "researcher_signal"),
    ("x_ylecun", "researcher_signal"),
    ("x_fchollet", "researcher_signal"),
    ("x_jimfan", "researcher_signal"),
    ("x_akhaliq", "expert_signal"),
    ("x_andrewyng", "expert_signal"),
    # ── Builder signals ──
    ("x_simonw", "builder_signal"),
    ("x_tinygrad", "builder_signal"),
    ("builder_", "builder_signal"),
    # ── China model vendors ──
    ("deepseek_", "china_model_official"),
    ("qwen_", "china_model_official"),
    ("aliyun_", "china_model_official"),
    ("zhipu_", "china_model_official"),
    ("baidu_", "china_model_official"),
    ("moonshot_", "china_model_official"),
    ("minimax_", "china_model_official"),
    ("stepfun_", "china_model_official"),
    ("zeroone_", "china_model_official"),
    ("baichuan_", "china_model_official"),
    ("iflytek_", "china_model_official"),
    ("sensetime_", "china_model_official"),
    ("kling_", "china_model_official"),
    ("x_deepseek", "china_model_official"),
    ("x_qwen", "china_model_official"),
    ("x_zhipu", "china_model_official"),
    ("x_baidu", "china_model_official"),
    ("x_tencent_hunyuan", "china_model_official"),
    ("x_alicloud", "china_model_official"),
    ("x_moonshot", "china_model_official"),
    ("x_minimax", "china_model_official"),
    ("x_stepfun", "china_model_official"),
    ("x_01ai", "china_model_official"),
    ("x_kuaishou_kling", "china_product_changelog"),
    ("x_siliconflow", "china_ecosystem_signal"),
    ("siliconflow_", "china_ecosystem_signal"),
    ("x_dotey", "china_ecosystem_signal"),
    ("x_ayi", "china_ecosystem_signal"),
    ("x_yi_ding", "china_ecosystem_signal"),
    # ── Product changelog ──
    ("_changelog", "product_changelog"),
    ("cursor_", "product_changelog"),
    ("windsurf_", "product_changelog"),
    ("copilot_", "product_changelog"),
    # ── Safety ──
    ("safety_", "safety_eval"),
    ("metr_", "safety_eval"),
    ("apollo_research", "safety_eval"),
    ("nist_ai", "safety_eval"),
    ("mlcommons", "safety_eval"),
    # ── Newsletters ──
    ("the_batch", "expert_newsletter"),
    ("import_ai", "expert_newsletter"),
    ("latent_space", "expert_newsletter"),
    ("interconnects", "expert_newsletter"),
    ("stratechery", "expert_newsletter"),
    # ── CN aggregators ──
    ("36kr", "cn_aggregator"),
    ("ithome", "cn_aggregator"),
    ("jiqizhixin", "cn_aggregator"),
    ("qbitai", "cn_aggregator"),
    ("aihot", "cn_aggregator"),
    # ── Tech media ──
    ("techcrunch", "tech_media"),
    ("venturebeat", "tech_media"),
    ("the_verge", "tech_media"),
    ("wired", "tech_media"),
    ("ars_technica", "tech_media"),
    ("mit_tech_review", "tech_media"),
    # ── Community ──
    ("hackernews", "community_signal"),
    ("reddit_", "community_signal"),
    ("huggingface_blog", "official_release"),
]


def _infer_content_type(source_id: str, source_type: str) -> str:
    """Map a source to its content_type using pattern rules."""
    sid = source_id.lower()
    for pattern, ct in _CONTENT_TYPE_RULES:
        if pattern in sid:
            return ct
    # Official X accounts → official_release
    if source_type in ("x", "x_cookie"):
        return "official_release"
    # RSS feeds from official domains → official_release
    return "official_release"


def _build_ct_lookup(source_specs: list[dict]) -> dict[str, str]:
    """Build source_id → content_type lookup from config specs."""
    lookup: dict[str, str] = {}
    for spec in source_specs:
        sid = spec.get("id", "")
        if not sid:
            continue
        if "content_type" in spec:
            lookup[sid] = spec["content_type"]
        else:
            lookup[sid] = _infer_content_type(sid, spec.get("type", ""))
    return lookup


def _build_source_meta_lookup(source_specs: list[dict]) -> dict[str, dict]:
    """Build source_id → {tier, reliability, confidence, evidence, section}."""
    lookup: dict[str, dict] = {}
    for spec in source_specs:
        sid = spec.get("id", "")
        if not sid:
            continue
        lookup[sid] = {
            "source_tier": spec.get("source_tier", ""),
            "reliability": spec.get("reliability", ""),
            "default_confidence": spec.get("default_confidence", "medium"),
            "evidence_type": spec.get("evidence_type", ""),
            "section_hint": spec.get("section_hint", ""),
        }
    return lookup


def _source_field(lookup: dict[str, dict], sid: str, field: str, default: str = "") -> str:
    """Safely read a field from the source meta lookup."""
    return lookup.get(sid, {}).get(field, default)


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
    content_types_cfg: Dict[str, Any] | None = None,
    score_floor: float = 0.0,
    research_min: int = 0,
) -> List[CuratedItem]:
    """Return the top-N curated items (writer input contract, unchanged)."""
    curated_items, _ = curate_with_records(
        items, source_specs=source_specs, max_items=max_items,
        content_types_cfg=content_types_cfg,
        score_floor=score_floor,
        research_min=research_min,
    )
    return curated_items


def curate_with_records(
    items: List[RawItem],
    *,
    source_specs: List[Dict[str, Any]],
    max_items: int = 12,
    content_types_cfg: Dict[str, Any] | None = None,
    score_floor: float = 0.0,
    research_min: int = 0,
) -> Tuple[List[CuratedItem], List[CuratedItemRecord]]:
    """Return (writer_items, persistence_records).

    Scoring uses content-type-aware weights and half-lives when
    content_types_cfg is provided, falling back to legacy source_weight.
    """
    ct_lookup = _build_ct_lookup(source_specs)
    sid_lookup = _build_source_meta_lookup(source_specs)

    # Build content-type parameter maps.
    ct_weights: Dict[str, float] = {}
    ct_half_lives: Dict[str, float] = {}
    if content_types_cfg:
        for ct_key, ct_cfg in content_types_cfg.items():
            ct_weights[ct_key] = float(ct_cfg.get("source_weight", 0.85))
            ct_half_lives[ct_key] = float(ct_cfg.get("half_life_h", 48))
    # Also keep legacy weights for sources without content_type mapping.
    legacy_weights: Dict[str, float] = {
        s.get("id", ""): float(s.get("weight", 1.0)) for s in source_specs
    }

    seen_titles: Dict[str, str] = {}
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
            continue

        seen_titles[nt] = _dup_group_id(nt)
        seen_urls.add(it.url)

        # ── Content-type-aware scoring ───────────────────────────
        ct = ct_lookup.get(it.source_id, "tech_media")
        ct_w = ct_weights.get(ct, legacy_weights.get(it.source_id, 1.0))
        ct_hl = ct_half_lives.get(ct, 48.0)
        r = _recency_score(it.published_at, now_ts, ct_hl)

        # ── Source diversity: penalize over-represented sources ──
        source_count = source_scored_count.get(it.source_id, 0)
        diversity_penalty = 1.0
        if source_count >= 6:
            diversity_penalty = 0.6
        elif source_count >= 4:
            diversity_penalty = 0.75
        elif source_count >= 3:
            diversity_penalty = 0.85

        score = ct_w * r * rel_boost * diversity_penalty

        # ── Soft score floor ─────────────────────────────────────
        # Fast-news items below floor are discarded. Slow-moving
        # content types (research, pricing, financial) pass through.
        if score_floor > 0 and score < score_floor:
            if ct not in ("research_paper", "pricing_page", "financial_report"):
                continue

        source_scored_count[it.source_id] += 1

        reasons: list[str] = []
        reasons.append(f"ct={ct}")
        reasons.append(f"ct_w={ct_w:.2f}")
        reasons.append(f"recency={r:.3f}")
        if rel_boost != 1.0:
            reasons.append(f"relevance={rel_boost:.2f}")
        if diversity_penalty < 1.0:
            reasons.append(f"diversity={diversity_penalty:.2f}")

        # ── Tier/quality metadata from source config ──────────────
        tier = _source_field(sid_lookup, it.source_id, "source_tier", "")
        rel = _source_field(sid_lookup, it.source_id, "reliability", "")
        conf = _source_field(sid_lookup, it.source_id, "default_confidence", "medium")
        ev = _source_field(sid_lookup, it.source_id, "evidence_type", "")
        sec_hint = _source_field(sid_lookup, it.source_id, "section_hint", "")

        curated = CuratedItem(
            title=it.title,
            url=it.url,
            summary=it.summary[:500],
            source=it.source_id,
            source_type=it.source_type,
            published_at=it.published_at,
            score=round(score, 4),
            content_type=ct,
            source_tier=tier,
            evidence_type=ev,
            confidence=conf,
            section_hint=sec_hint,
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
            content_type=ct,
            source_tier=tier,
            reliability=rel,
            confidence=conf,
            evidence_type=ev,
            section_hint=sec_hint,
        )
        scored.append((score, curated, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = _select_with_paper_quota(
        scored, max_items, min_papers=5, research_min=research_min,
    )
    writer_items = [c for _, c, _ in top]
    records = [rec for _, _, rec in top]
    return writer_items, records


_MIN_PAPERS = 5
_RESEARCH_FLOOR = 0.15  # minimum score for backfilled research papers


def _is_research(item: CuratedItemRecord) -> bool:
    """Check if a curated item is a research paper."""
    return (
        "arxiv" in item.source_name.lower()
        or "hf_daily_papers" in item.source_name.lower()
    )


def _select_with_paper_quota(
    scored: List[Tuple[float, CuratedItem, CuratedItemRecord]],
    max_items: int,
    min_papers: int = 5,
    research_min: int = 0,
) -> List[Tuple[float, CuratedItem, CuratedItemRecord]]:
    """Select top-N items, ensuring min_papers arxiv/HF papers.

    If research_min > 0, also ensure at least research_min research_paper
    items with score >= _RESEARCH_FLOOR are in the final set.
    """
    # Find all research items in the full scored list.
    all_research = [(s, c, r) for s, c, r in scored if _is_research(r)]
    all_research.sort(key=lambda x: x[0], reverse=True)

    top = list(scored[:max_items])
    research_in_top = [(s, c, r) for s, c, r in top if _is_research(r)]

    effective_min = max(min_papers, research_min)
    if len(research_in_top) >= effective_min:
        return top

    # Find the best research items NOT in the top.
    top_research_ids = {r.raw_item_id for _, _, r in research_in_top}
    research_not_in_top = [
        (s, c, r) for s, c, r in all_research
        if r.raw_item_id not in top_research_ids
        and s >= _RESEARCH_FLOOR  # don't backfill below this score
    ]
    research_not_in_top.sort(key=lambda x: x[0], reverse=True)

    needed = effective_min - len(research_in_top)
    to_add = research_not_in_top[:needed]

    # Remove lowest-scoring non-research from top to make room.
    non_research_top = [(s, c, r) for s, c, r in top if not _is_research(r)]
    non_research_top.sort(key=lambda x: x[0])  # ascending — lowest first
    to_remove = non_research_top[:len(to_add)]

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
    content_types_cfg: Dict[str, Any] | None = None,
    score_floor: float = 0.0,
    research_min: int = 0,
) -> Tuple[List[CuratedItem], List[CuratedItemRecord]]:
    """Curate items using LLM importance scoring + deterministic filtering."""
    ct_lookup = _build_ct_lookup(source_specs)
    sid_lookup = _build_source_meta_lookup(source_specs)
    ct_weights: Dict[str, float] = {}
    ct_half_lives: Dict[str, float] = {}
    if content_types_cfg:
        for ct_key, ct_cfg in content_types_cfg.items():
            ct_weights[ct_key] = float(ct_cfg.get("source_weight", 0.85))
            ct_half_lives[ct_key] = float(ct_cfg.get("half_life_h", 48))
    legacy_weights: Dict[str, float] = {
        s.get("id", ""): float(s.get("weight", 1.0)) for s in source_specs
    }

    seen_titles: Dict[str, str] = {}
    seen_urls: set = set()
    now_ts = _time.time()

    # Phase 1: Dedup + AI-relevance filter.
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
        ct = ct_lookup.get(it.source_id, "tech_media")
        w = ct_weights.get(ct, legacy_weights.get(it.source_id, 1.0))
        hl = ct_half_lives.get(ct, 48.0)
        r = _recency_score(it.published_at, now_ts, hl)
        det_score = w * r

        key = f"{it.source_id}::{it.url}"
        llm_s = llm_scores.get(key, 0.5)

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

        # Score floor for fast-news types.
        if score_floor > 0 and final_score < score_floor:
            if ct not in ("research_paper", "pricing_page", "financial_report"):
                continue

        source_scored_count[it.source_id] += 1

        reasons = [f"ct={ct}", f"llm={llm_s:.2f}", f"det={det_score:.2f}"]
        if div_penalty < 1.0:
            reasons.append(f"div={div_penalty:.2f}")

        tier = _source_field(sid_lookup, it.source_id, "source_tier", "")
        rel = _source_field(sid_lookup, it.source_id, "reliability", "")
        conf = _source_field(sid_lookup, it.source_id, "default_confidence", "medium")
        ev = _source_field(sid_lookup, it.source_id, "evidence_type", "")
        sec_hint = _source_field(sid_lookup, it.source_id, "section_hint", "")

        curated = CuratedItem(
            title=it.title, url=it.url, summary=it.summary[:500],
            source=it.source_id, source_type=it.source_type,
            published_at=it.published_at, score=round(final_score, 4),
            content_type=ct, source_tier=tier, evidence_type=ev,
            confidence=conf, section_hint=sec_hint,
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
    top = _select_with_paper_quota(scored, max_items, min_papers=_MIN_PAPERS, research_min=research_min)
    return [c for _, c, _ in top], [rec for _, _, rec in top]
