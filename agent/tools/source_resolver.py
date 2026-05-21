"""Source Resolver / Source Doctor v1.

Auto-discovers URLs, validates feeds, and produces resolution
recommendations for disabled sources. Never auto-enables snapshot,
VC, builder, or reporter signal sources.

Usage:
  python -m agent.cli source-resolve --dry-run
  python -m agent.cli source-resolve --apply-safe
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import yaml


# ═══════════════════════════════════════════════════════════════════════
# Known URL patterns for source discovery
# ═══════════════════════════════════════════════════════════════════════

_GITHUB_REPOS = {
    "deepseek": ("deepseek-ai", "DeepSeek-V3"),
    "qwen": ("QwenLM", "Qwen3"),
    "vllm": ("vllm-project", "vllm"),
    "ollama": ("ollama", "ollama"),
    "sglang": ("sgl-project", "sglang"),
}

_OFFICIAL_DOCS_PATTERNS = {
    "deepseek": ["https://api-docs.deepseek.com", "https://platform.deepseek.com/api-docs"],
    "qwen": ["https://help.aliyun.com/zh/model-studio", "https://bailian.console.aliyun.com"],
    "zhipu": ["https://open.bigmodel.cn/dev/api", "https://docs.bigmodel.cn"],
    "moonshot": ["https://platform.moonshot.cn/docs"],
    "minimax": ["https://api.minimax.chat/document"],
    "baidu": ["https://cloud.baidu.com/doc/WENXINWORKSHOP/s"],
    "tencent": ["https://cloud.tencent.com/document/product/1729"],
    "bytedance": ["https://www.volcengine.com/docs/82379"],
    "stepfun": ["https://platform.stepfun.com/docs"],
    "01ai": ["https://platform.01.ai/docs"],
    "baichuan": ["https://platform.baichuan-ai.com/docs"],
    "siliconflow": ["https://docs.siliconflow.cn"],
    "zhipu": ["https://open.bigmodel.cn/dev/api", "https://docs.bigmodel.cn"],
    "bigmodel": ["https://open.bigmodel.cn/dev/api", "https://docs.bigmodel.cn"],
}

_OFFICIAL_PRICING_PATTERNS = {
    "openai": "https://openai.com/api/pricing/",
    "anthropic": "https://www.anthropic.com/pricing",
    "google": "https://ai.google.dev/pricing",
    "deepseek": "https://api-docs.deepseek.com/quick_start/pricing",
    "mistral": "https://mistral.ai/technology/#pricing",
    "together": "https://www.together.ai/pricing",
    "fireworks": "https://fireworks.ai/pricing",
    "groq": "https://console.groq.com/settings/billing",
    "cerebras": "https://cerebras.ai/cloud-pricing",
    "perplexity": "https://docs.perplexity.ai/guides/pricing",
}

_BENCHMARK_URLS = {
    "artificial_analysis": "https://artificialanalysis.ai/",
    "lmarena_leaderboard": "https://lmarena.ai/",
    "livebench": "https://livebench.ai/",
    "swebench_leaderboard": "https://www.swebench.com/",
    "aider_leaderboard": "https://aider.chat/docs/leaderboards/",
    "openrouter_rankings": "https://openrouter.ai/rankings",
}

_MODELSCOPE_URLS = {
    "qwen_modelscope": "https://modelscope.cn/organization/qwen",
    "modelscope_trending": "https://modelscope.cn/models",
}

# Keywords found in page text to validate source type
_PRICING_KEYWORDS = ["pricing", "price", "tokens", "billing", "计费", "价格", "/1M", "per token"]
_DOCS_KEYWORDS = ["api", "docs", "documentation", "reference", "quickstart", "model", "endpoint"]
_BENCHMARK_KEYWORDS = ["benchmark", "leaderboard", "ranking", "score", "elo", "arena", "eval"]


@dataclass
class SourceResolutionResult:
    source_id: str = ""
    status: str = "no_candidate_found"
    old_url: Optional[str] = None
    candidate_urls: List[str] = field(default_factory=list)
    selected_url: Optional[str] = None
    url_validation_status: str = ""
    http_status: Optional[int] = None
    content_type_header: Optional[str] = None
    detected_source_kind: Optional[str] = None
    recommended_enabled: bool = False
    recommended_parser_strategy: str = ""
    required_adapter: Optional[str] = None
    confidence: str = "low"
    risk_level: str = "high"
    reason: str = ""
    notes: str = ""


@dataclass
class SourceResolutionReport:
    date: str = ""
    total_checked: int = 0
    resolved_enable_safe: int = 0
    candidate_found_needs_adapter: int = 0
    candidate_found_needs_review: int = 0
    no_candidate_found: int = 0
    invalid_existing_url: int = 0
    ambiguous_candidates: int = 0
    results: List[SourceResolutionResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# URL Discovery
# ═══════════════════════════════════════════════════════════════════════


def _discover_urls(source: dict) -> List[str]:
    """Find candidate URLs for a source based on its type and content_type."""
    sid = source.get("id", "")
    st = source.get("type", "")
    ct = source.get("content_type", "")
    existing = source.get("url", "") or source.get("source_url", "")

    candidates: List[str] = []
    if existing and existing.strip():
        candidates.append(existing.strip())

    # GitHub releases
    if ct in ("github_release",) or st in ("github_release",) or "github" in sid:
        for provider, (org, repo) in _GITHUB_REPOS.items():
            if provider in sid:
                candidates.append(f"https://github.com/{org}/{repo}/releases.atom")
                break

    # Pricing pages
    if ct in ("pricing_page", "china_model_pricing"):
        for provider, url in _OFFICIAL_PRICING_PATTERNS.items():
            if provider in sid:
                if url not in candidates:
                    candidates.append(url)
                break

    # Docs
    if ct in ("official_docs", "china_model_docs"):
        for provider, urls in _OFFICIAL_DOCS_PATTERNS.items():
            if provider in sid:
                for u in urls:
                    if u not in candidates:
                        candidates.append(u)
                break

    # Benchmark
    if ct == "benchmark_tracker":
        for bid, url in _BENCHMARK_URLS.items():
            if bid in sid:
                candidates.append(url)
                break

    # ModelScope
    if "modelscope" in sid:
        for mid, url in _MODELSCOPE_URLS.items():
            if mid in sid:
                candidates.append(url)
                break

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ═══════════════════════════════════════════════════════════════════════
# URL Validation (HTTP check — mockable via dependency injection)
# ═══════════════════════════════════════════════════════════════════════


def _validate_url(
    url: str, timeout: float = 10.0, http_client: Any = None,
) -> Tuple[int, str, str]:
    """HTTP HEAD then GET to validate URL. Returns (status, content_type, body_sample)."""
    try:
        if http_client is None:
            import httpx
            http_client = httpx.Client(timeout=timeout, follow_redirects=True,
                                       headers={"User-Agent": "AI-Frontier-Agent/3.0"})
        # HEAD first
        head = http_client.head(url)
        if head.status_code >= 400:
            # Try GET
            get = http_client.get(url)
            body = get.text[:5000] if get.status_code < 400 else ""
            return (get.status_code,
                    get.headers.get("content-type", ""),
                    body)
        return (head.status_code,
                head.headers.get("content-type", ""),
                "")
    except Exception:
        return (0, "", "")


def _is_valid_feed(status: int, content_type: str, body: str) -> bool:
    """Check if URL looks like a valid RSS/Atom feed."""
    if status != 200:
        return False
    ct_lower = content_type.lower()
    if any(x in ct_lower for x in ("xml", "rss", "atom")):
        return True
    if "<rss" in body.lower() or "<feed" in body.lower():
        return True
    return False


def _is_valid_page(status: int) -> bool:
    return status == 200


# ═══════════════════════════════════════════════════════════════════════
# Resolution logic
# ═══════════════════════════════════════════════════════════════════════


_AUTO_ENABLE_SAFE_CTS = {
    "github_release", "product_changelog", "expert_newsletter",
}

_NEVER_AUTO_ENABLE_CTS = {
    "pricing_page", "china_model_pricing", "benchmark_tracker",
    "official_docs", "china_model_docs",
    "vc_signal", "builder_signal", "insider_reporter_signal",
    "founder_signal", "researcher_signal",
}


def resolve_source(
    source: dict,
    *,
    http_client: Any = None,
    force: bool = False,
) -> SourceResolutionResult:
    """Resolve a single disabled source and produce a recommendation."""
    sid = source.get("id", "unknown")
    st = source.get("type", "")
    ct = source.get("content_type", "")
    old_url = source.get("url", "") or source.get("source_url", "") or None
    result = SourceResolutionResult(source_id=sid, old_url=old_url)

    # Discover candidates
    candidates = _discover_urls(source)
    result.candidate_urls = candidates

    if not candidates:
        result.status = "no_candidate_found"
        result.reason = f"no candidate URLs discovered for {ct}"
        result.recommended_enabled = False
        result.risk_level = "high"
        result.confidence = "low"
        if ct in _NEVER_AUTO_ENABLE_CTS:
            result.recommended_parser_strategy = _parser_for_ct(ct)
            result.required_adapter = _adapter_for_ct(ct)
        return result

    # Validate first candidate that works
    valid_url = None
    valid_status = None
    valid_ct = None
    for url in candidates:
        status, content_type, body = _validate_url(url, http_client=http_client)
        if status == 200:
            valid_url = url
            valid_status = status
            valid_ct = content_type
            result.http_status = status
            result.content_type_header = content_type
            # Detect source kind from content
            if _is_valid_feed(status, content_type, body):
                result.detected_source_kind = "rss_or_atom_feed"
            elif any(kw in body.lower() for kw in _PRICING_KEYWORDS):
                result.detected_source_kind = "pricing_page"
            elif any(kw in body.lower() for kw in _BENCHMARK_KEYWORDS):
                result.detected_source_kind = "benchmark_or_leaderboard"
            elif any(kw in body.lower() for kw in _DOCS_KEYWORDS):
                result.detected_source_kind = "docs_page"
            else:
                result.detected_source_kind = "generic_page"
            break
        elif status > 0:
            result.http_status = status

    if not valid_url:
        result.status = "invalid_existing_url"
        result.url_validation_status = f"all {len(candidates)} candidates returned non-200"
        result.reason = "no reachable URL found"
        result.recommended_enabled = False
        result.risk_level = "high"
        if ct in _NEVER_AUTO_ENABLE_CTS:
            result.recommended_parser_strategy = _parser_for_ct(ct)
            result.required_adapter = _adapter_for_ct(ct)
            result.status = "candidate_found_needs_adapter"
            result.risk_level = "medium"
            result.reason = f"candidate URLs exist but unreachable; {ct} requires {result.required_adapter}"
        return result

    result.selected_url = valid_url
    result.url_validation_status = f"HTTP {valid_status}"

    # Decide resolution
    # Always set adapter/parser for never-auto-enable types
    if ct in _NEVER_AUTO_ENABLE_CTS:
        parser = _parser_for_ct(ct)
        adapter = _adapter_for_ct(ct)
        if valid_url:
            result.status = "candidate_found_needs_adapter"
            result.recommended_parser_strategy = parser
            result.required_adapter = adapter
            result.confidence = "high"
            result.risk_level = "medium"
            result.reason = f"{ct} requires {adapter}; URL validated but source must stay disabled"
        else:
            result.status = "candidate_found_needs_adapter"
            result.recommended_parser_strategy = parser
            result.required_adapter = adapter
            result.confidence = "medium"
            result.risk_level = "medium"
            result.reason = f"{ct} requires {adapter}; URL not yet validated"
        result.notes = f"parser_strategy={parser}, implement {adapter} before enabling"
        result.recommended_enabled = False
        return result

    if ct in _AUTO_ENABLE_SAFE_CTS and result.detected_source_kind == "rss_or_atom_feed":
        result.status = "resolved_enable_safe"
        result.recommended_enabled = True
        result.recommended_parser_strategy = "rss"
        result.required_adapter = "rss"
        result.confidence = "high"
        result.risk_level = "low"
        result.reason = f"valid {result.detected_source_kind} for {ct}, auto-enable safe"
        return result

    # X/Twitter sources
    if st in ("x_cookie", "x"):
        username = source.get("username", "")
        if username and username.strip():
            result.status = "candidate_found_needs_review"
            result.recommended_enabled = False
            result.confidence = "medium"
            result.risk_level = "medium"
            result.reason = f"X handle @{username} present but {ct} requires manual review"
            result.notes = "verify handle identity before enabling"
            return result
        else:
            result.status = "no_candidate_found"
            result.reason = "X source missing username"
            return result

    # RSS sources with valid feed
    if st == "rss" and result.detected_source_kind == "rss_or_atom_feed":
        result.status = "resolved_enable_safe"
        result.recommended_enabled = True
        result.recommended_parser_strategy = "rss"
        result.required_adapter = "rss"
        result.confidence = "high"
        result.risk_level = "low"
        result.reason = f"valid RSS/Atom feed for {st}/{ct}"
        return result

    # Fallback: ambiguous
    result.status = "ambiguous_candidates"
    result.recommended_enabled = False
    result.confidence = "medium"
    result.risk_level = "medium"
    result.reason = f"ambiguous: ct={ct} st={st} kind={result.detected_source_kind}"
    result.notes = "manual review recommended"
    return result


def _parser_for_ct(ct: str) -> str:
    mapping = {
        "pricing_page": "pricing_snapshot",
        "china_model_pricing": "pricing_snapshot",
        "benchmark_tracker": "leaderboard_snapshot",
        "official_docs": "docs_snapshot",
        "china_model_docs": "docs_snapshot",
    }
    return mapping.get(ct, "rss")


def _adapter_for_ct(ct: str) -> str:
    mapping = {
        "pricing_page": "pricing_snapshot_adapter",
        "china_model_pricing": "pricing_snapshot_adapter",
        "benchmark_tracker": "benchmark_tracker_adapter",
        "official_docs": "docs_snapshot_adapter",
        "china_model_docs": "docs_snapshot_adapter",
    }
    return mapping.get(ct, "rss")


# ═══════════════════════════════════════════════════════════════════════
# Batch resolution
# ═══════════════════════════════════════════════════════════════════════


def resolve_all_disabled(
    sources: List[dict],
    *,
    http_client: Any = None,
) -> SourceResolutionReport:
    """Resolve all disabled sources and produce a report."""
    disabled = [s for s in sources if isinstance(s, dict) and not s.get("enabled", True)]
    report = SourceResolutionReport(
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        total_checked=len(disabled),
    )
    for s in disabled:
        result = resolve_source(s, http_client=http_client)
        report.results.append(result)
        # Count statuses
        if result.status == "resolved_enable_safe":
            report.resolved_enable_safe += 1
        elif result.status == "candidate_found_needs_adapter":
            report.candidate_found_needs_adapter += 1
        elif result.status == "candidate_found_needs_review":
            report.candidate_found_needs_review += 1
        elif result.status == "no_candidate_found":
            report.no_candidate_found += 1
        elif result.status == "invalid_existing_url":
            report.invalid_existing_url += 1
        elif result.status == "ambiguous_candidates":
            report.ambiguous_candidates += 1
    return report


# ═══════════════════════════════════════════════════════════════════════
# Apply safe fixes
# ═══════════════════════════════════════════════════════════════════════


def apply_safe_resolutions(
    sources: List[dict],
    report: SourceResolutionReport,
    *,
    force: bool = False,
) -> List[dict]:
    """Apply only low-risk resolutions to source config."""
    applied = 0
    for result in report.results:
        if result.status != "resolved_enable_safe":
            continue
        for s in sources:
            if not isinstance(s, dict):
                continue
            if s.get("id") == result.source_id:
                if result.selected_url and (not s.get("url") or force):
                    s["url"] = result.selected_url
                s["enabled"] = True
                s["parser_strategy"] = result.recommended_parser_strategy
                s["required_adapter"] = result.required_adapter
                s.pop("adapter_stub", None)
                s.pop("disabled_reason", None)
                s["repair_action"] = f"auto-enabled by source-resolve: {result.reason}"
                applied += 1
                break
    return sources


def apply_non_enable_fixes(
    sources: List[dict],
    report: SourceResolutionReport,
    *,
    force: bool = False,
) -> List[dict]:
    """Apply URL/parser_strategy updates for sources that stay disabled."""
    applied = 0
    for result in report.results:
        if result.status == "resolved_enable_safe":
            continue  # already handled
        for s in sources:
            if not isinstance(s, dict):
                continue
            if s.get("id") == result.source_id:
                if result.selected_url and (not s.get("url") or force):
                    s["url"] = result.selected_url
                if result.recommended_parser_strategy and not s.get("parser_strategy"):
                    s["parser_strategy"] = result.recommended_parser_strategy
                if result.required_adapter and not s.get("required_adapter"):
                    s["required_adapter"] = result.required_adapter
                applied += 1
                break
    return sources
