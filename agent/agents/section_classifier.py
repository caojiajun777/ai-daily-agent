"""Deterministic section classifier for the Juya-style editorial layout."""

from __future__ import annotations

import re

from agent.agents.event_clusterer import EventCluster


def guess_section(evt: EventCluster) -> str:
    """Classify an event into the daily's editorial sections.

    The classifier intentionally returns topical sections. "要闻" is a
    priority bucket and is assigned by the editor/final selector.
    """
    text = _event_text(evt)
    title = (evt.canonical_title or "").lower()

    if _is_forward_or_rumor(evt, text):
        return "前瞻与传闻"

    if _is_paper_event(evt):
        return "技术与洞察"

    if _has_any(text, _CAPITAL_TERMS):
        return "行业动态"

    if _is_tier3_pulse(evt):
        return "行业动态"

    # Strong model-release signals must run before tool/product matching:
    # open-source models and flagship model launches often contain words like
    # "open source", "launch", or "product", but editorially belong here.
    if _is_model_release(text, title):
        return "模型发布"

    if _is_product_application(text):
        return "产品应用"

    if _has_any(text, _TECH_INSIGHT_TERMS):
        return "技术与洞察"

    if _has_any(text, _INDUSTRY_TERMS):
        return "行业动态"

    if _has_any(text, _TOOL_TERMS):
        return "开发生态"

    if _has_any(text, _PRODUCT_TERMS):
        return "产品应用"

    if _is_model_topic(text):
        return "模型发布"

    return "行业动态"


def _is_paper_event(evt: EventCluster) -> bool:
    url_text = " ".join(evt.source_urls).lower()
    meta_text = (
        f"{evt.primary_content_type} {evt.primary_evidence_type} "
        f"{' '.join(evt.source_types)} {' '.join(evt.source_names)}"
    ).lower()
    return (
        "arxiv" in meta_text
        or "hf_daily_papers" in meta_text
        or "research_paper" in meta_text
        or "arxiv.org" in url_text
        or "huggingface.co/papers" in url_text
    )


def _event_text(evt: EventCluster) -> str:
    return f"{evt.canonical_title} {evt.summary}".lower()


def _is_tier3_pulse(evt: EventCluster) -> bool:
    return (
        "tier_3" in (evt.primary_source_tier or "").lower()
        and not _is_official_release_source(evt)
    )


def _is_forward_or_rumor(evt: EventCluster, text: str) -> bool:
    if _has_any(text, _RUMOR_TERMS):
        return True
    ctype = (evt.primary_content_type or "").lower()
    etype = (evt.primary_evidence_type or "").lower()
    confidence = (evt.primary_confidence or "").lower()
    if confidence == "low" and _has_any(text, _LOW_CONFIDENCE_RUMOR_TERMS):
        return True
    return any(k in ctype or k in etype for k in (
        "rumor", "leak", "insider_reporter_signal", "vc_signal",
        "community_signal", "market_commentary",
    ))


def _is_official_release_source(evt: EventCluster) -> bool:
    names = {n.lower() for n in evt.source_names}
    urls = " ".join(evt.source_urls).lower()
    official_source_ids = {
        "openai_news", "anthropic_news", "google_ai_blog",
        "google_developers_blog", "google_deepmind_blog", "meta_ai_blog",
        "microsoft_ai_blog", "huggingface_blog", "ollama_releases",
        "x_openai", "x_anthropicai", "x_alibaba_qwen", "x_qwen",
        "x_tencent_hunyuan", "x_deepseek_ai", "x_googledeepmind",
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
    )
    return bool(names & official_source_ids) or any(marker in urls for marker in official_url_markers)


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def _contains_term(text: str, term: str) -> bool:
    if term.isascii() and any(ch.isalnum() for ch in term):
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        return re.search(pattern, text) is not None
    return term in text


def _is_model_release(text: str, title: str = "") -> bool:
    has_named_model = _has_any(text, _MODEL_FAMILY_TERMS) or bool(_MODEL_VERSION_RE.search(text))
    has_model_object = _has_any(text, _MODEL_OBJECT_TERMS)
    has_release = _has_any(text, _MODEL_RELEASE_TERMS)
    has_capability_claim = _has_any(text, _MODEL_CAPABILITY_TERMS)
    has_parameter_scale = bool(_PARAMETER_SCALE_RE.search(text))
    title_strong = _has_title_model_release_signal(title)
    product_or_tool_context = _has_any(text, _PRODUCT_TERMS) or _has_any(text, _TOOL_TERMS)
    explicit_release = _has_any(text, _MODEL_EXPLICIT_RELEASE_TERMS)

    if explicit_release and (title_strong or not product_or_tool_context):
        return True
    if product_or_tool_context and not title_strong:
        return False
    if has_named_model and has_release and (has_model_object or has_capability_claim or has_parameter_scale):
        return True
    if has_named_model and has_parameter_scale:
        return True
    if has_model_object and has_release and (has_capability_claim or has_parameter_scale):
        return True
    if has_model_object and has_parameter_scale:
        return True
    return False


def _has_title_model_release_signal(title: str) -> bool:
    if not title:
        return False
    title_has_named = _has_any(title, _MODEL_FAMILY_TERMS) or bool(_MODEL_VERSION_RE.search(title))
    title_has_object = _has_any(title, _MODEL_OBJECT_TERMS)
    title_has_release = _has_any(title, _MODEL_RELEASE_TERMS)
    title_has_capability = _has_any(title, _MODEL_CAPABILITY_TERMS)
    title_has_scale = bool(_PARAMETER_SCALE_RE.search(title))
    if _has_any(title, _MODEL_EXPLICIT_RELEASE_TERMS):
        return True
    if title_has_named and title_has_release and (title_has_object or title_has_capability or title_has_scale):
        return True
    if title_has_object and title_has_release and (title_has_capability or title_has_scale):
        return True
    if title_has_object and title_has_scale:
        return True
    return False


def _is_model_topic(text: str) -> bool:
    if _has_any(text, _PRODUCT_TERMS):
        return False
    return (
        _has_any(text, _MODEL_FAMILY_TERMS)
        or _has_any(text, _MODEL_OBJECT_TERMS)
        or bool(_MODEL_VERSION_RE.search(text))
    )


def _is_product_application(text: str) -> bool:
    """Detect shipped user-facing/productized AI experiences.

    Some product stories contain words like "provider", "hardware partner", or
    "stack" and would otherwise drift into industry/technical buckets. For a
    daily reader, smart-home suites, plugins, mobile/desktop features, and
    customer-facing integrations are product/application stories.
    """
    return _has_any(text, _PRODUCT_APPLICATION_TERMS)


_CAPITAL_TERMS = (
    "融资", "funding", "ipo", "收购", "投资", "估值", "财报",
    "earnings", "revenue", "营收", "净利润", "季度", "quarterly",
    "valuation", "acquisition", "merger", "并购", "billion", "亿",
    "series", "轮融资",
)

_RUMOR_TERMS = (
    "rumor", "rumour", "leak", "leaked", "testing", "a/b test",
    "ab test", "spotted", "prototype", "previewed", "unconfirmed",
    "传闻", "爆料", "泄露", "测试", "小规模测试", "内测", "灰度",
    "据称", "消息称", "网传", "尚未确认", "未获官方确认",
    "或将", "前瞻",
)

_LOW_CONFIDENCE_RUMOR_TERMS = (
    "融资", "funding", "估值", "valuation", "参投", "接近",
    "可能", "计划", "raise", "raising",
)

_TECH_INSIGHT_TERMS = (
    "research", "paper", "arxiv", "benchmark", "eval", "evaluation",
    "security", "vulnerability", "vulnerabilities", "proof", "theorem",
    "formal", "lean", "architecture", "training method", "post-training",
    "reasoning", "alignment", "safety", "whitepaper", "technical report",
    "研究", "论文", "基准", "评估", "安全", "漏洞", "证明", "定理",
    "形式化", "架构", "训练方法", "后训练", "推理", "对齐", "技术报告",
)

_INDUSTRY_TERMS = (
    "政策", "监管", "regulation", "law", "ban", "hire", "ceo",
    "executive", "partner", "partners", "partnership", "provider",
    "providers", "service provider", "hardware partner", "合作", "裁员", "layoff",
    "人事", "任命", "趋势", "trend", "预测", "outlook", "government",
    "citizen", "national", "全民", "战略", "strategy", "算力", "芯片",
)

_TOOL_TERMS = (
    "framework", "sdk", "tool", "library", "github", "api", "cli",
    "plugin", "extension", "vscode", "copilot", "release notes",
    "changelog", "ollama", "llama.cpp", "gguf", "mlx", "降价",
    "定价", "免费", "工具",
)

_PRODUCT_TERMS = (
    "app", "product", "chatbot", "assistant", "experience", "workflow",
    "workflows",
    "healthcare", "therapy", "智能家居", "功能", "应用", "产品", "用户",
    "推出", "上线", "launch", "feature", "rollout", "更新",
)

_PRODUCT_APPLICATION_TERMS = (
    "gemini for home", "smart home", "智能家居", "home ai",
    "service provider", "service providers", "hardware partner",
    "hardware partners", "reference design", "camera intelligence",
    "activity summary", "natural language query", "turnkey",
    "plugin", "powerpoint", "mobile app", "desktop app", "ios", "android",
    "用户功能", "产品功能", "应用案例", "交钥匙", "参考设计",
    "摄像头智能", "自然语言查询", "日常活动摘要",
)

_MODEL_FAMILY_TERMS = (
    "gpt", "claude", "gemini", "qwen", "deepseek", "hunyuan",
    "hy-mt", "hymt", "llama", "mistral", "kimi", "ernie", "glm",
    "doubao", "minimax", "yi-", "seed-", "opus", "sonnet",
)

_MODEL_OBJECT_TERMS = (
    "model", "models", "模型", "llm", "foundation", "基础模型",
    "开源模型", "旗舰模型", "多语言翻译模型", "translation model",
    "reasoning model", "多模态模型", "diffusion", "语音模型",
)

_MODEL_RELEASE_TERMS = (
    "release", "released", "launch", "launched", "introducing",
    "introduced", "announce", "announced", "latest", "live",
    "open source", "open-source", "开源", "发布", "推出", "上线",
    "宣布", "正式",
)

_MODEL_EXPLICIT_RELEASE_TERMS = (
    "flagship model", "latest flagship", "模型发布", "发布模型",
    "开源模型", "开源了模型", "开源hy-mt", "开源 hy-mt",
)

_MODEL_CAPABILITY_TERMS = (
    "benchmark", "benchmarks", "eval", "sota", "state of the art",
    "frontier", "agent", "coding", "reasoning", "推理", "训练",
    "参数", "性能", "基准", "能力", "多语言", "翻译", "超越",
)

_MODEL_VERSION_RE = re.compile(
    r"\b(?:gpt|claude|gemini|qwen|deepseek|hunyuan|hy-mt|llama|mistral|kimi|glm|ernie)"
    r"[-\s]?\d+(?:\.\d+)*(?:-[a-z0-9]+)?\b"
)
_PARAMETER_SCALE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:b|m|t)\b|[0-9]+(?:\.[0-9]+)?\s?亿参数")
