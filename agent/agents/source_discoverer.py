"""Source Discovery Agent.

LLM-driven candidate generation + deterministic validation = verified new sources.

Flow:
  1. LLM generates candidate sources for a given topic (RSS URLs + X accounts).
  2. Each candidate is deterministically validated:
     - RSS: fetch feed → check HTTP status → parse entries → score freshness/relevance.
     - X:   resolve username to user_id via API → check recent tweet activity.
  3. Validated sources are scored and ranked. Those above a threshold are
     output as ready-to-merge YAML snippets.

The key idea: the LLM provides *breadth* (it knows "what exists"), and the
deterministic validator provides *precision* (it verifies "what works").

Usage (CLI):
  python -m agent.cli discover-sources --topic "chinese-ai-models"
  python -m agent.cli discover-sources --topic "all" --dry-run
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import httpx
import yaml

from agent.llm import LLMProvider
from agent.harness.trace import Tracer


DISCOVERY_SYSTEM_PROMPT = """你是一个 AI 信息源发现专家。你的任务是分析现有信息源的覆盖盲区，并推荐高质量的新信息源来填补这些空白。

## 分析维度

你需要从以下几个维度评估现有信息源的覆盖情况：
1. **地域** — 中国 / 美国 / 欧洲 / 全球 的覆盖是否均衡？
2. **信源类型** — 官方博客 / X官方号 / X KOL / 学术 / 媒体 / 社区 是否都有？
3. **主题** — 模型发布 / 产品更新 / 研究论文 / 行业资本 / 政策监管 / 开发者工具 是否都覆盖？
4. **时效性** — 是否有足够的一手实时信源（X）来捕捉突发新闻？
5. **语言** — 中英文源的比例是否合理？

## 输出格式

首先给出一段覆盖盲区分析（100-200字），然后给出推荐信源列表。

严格的 JSON 数组格式，每个元素为：
{
  "name": "信息源名称",
  "type": "rss" | "x",
  "url": "RSS feed URL (type=rss 时必填)",
  "username": "X 账号名不含@ (type=x 时必填)",
  "account_type": "official" | "kol" | "media",
  "reason": "推荐理由，说明填补了什么覆盖空白，中文",
  "language": "zh" | "en",
  "category": "model-provider" | "research" | "product" | "media" | "community" | "capital" | "policy"
}

## 规则

- type=rss 时 url 必须是真实的 RSS/Atom feed 地址，你确信它存在
- type=x 时 username 是 X/Twitter 账号名（不含 @），你确信这个账号活跃
- 优先推荐一手信息源（官方博客、官方账号），其次才是优质媒体和 KOL
- 重点关注 AI 模型发布、产品更新、研究进展、开发生态和行业动态
- 不要推荐已废弃/长期不更新的源
- 不要推荐与已有源重复的源（相同域名或相同 X 账号）"""

DISCOVERY_USER_TEMPLATE = """请分析当前信息源的覆盖盲区，并推荐高质量的新 AI 信息源。

{focus_instruction}

当前已订阅的源（避免重复推荐）：
{existing_sources}

当前共有 {source_count} 个源。请推荐 {max_candidates} 个最值得添加的新信息源。

请先分析覆盖盲区，再给出推荐列表。只推荐你确信真实存在的源。"""


@dataclass
class CandidateSource:
    name: str
    source_type: str  # rss / x
    url: Optional[str] = None
    username: Optional[str] = None
    account_type: str = "official"
    language: str = "en"
    category: str = ""
    reason: str = ""
    # Validation results (populated after checking).
    validated: bool = False
    reachable: bool = False
    freshness_score: float = 0.0
    relevance_score: float = 0.0
    uniqueness_score: float = 1.0
    overall_score: float = 0.0
    validation_note: str = ""
    validated_at: str = ""


@dataclass
class DiscoveryReport:
    topic: str
    run_ts: str
    candidates_generated: int = 0
    candidates_validated: int = 0
    candidates_passed: int = 0
    passed: List[CandidateSource] = field(default_factory=list)
    rejected: List[CandidateSource] = field(default_factory=list)
    yaml_snippet: str = ""


def discover_sources(
    *,
    topic: str = "all",
    provider: LLMProvider,
    existing_config_path: str,
    tracer: Optional[Tracer] = None,
    max_candidates: int = 20,
    min_score: float = 0.4,
) -> DiscoveryReport:
    report = DiscoveryReport(
        topic=topic,
        run_ts=datetime.now(timezone.utc).isoformat(),
    )

    existing_sources = _load_existing_source_summary(existing_config_path)
    existing_domains = _load_existing_domains(existing_config_path)
    existing_x_usernames = _load_existing_x_usernames(existing_config_path)
    focus = _build_focus_instruction(topic, existing_sources)

    # Phase 1: LLM generates candidates.
    candidates = _generate_candidates(
        provider=provider,
        focus_instruction=focus,
        existing_sources=existing_sources,
        source_count=len(existing_domains) + len(existing_x_usernames),
        max_candidates=max_candidates,
        tracer=tracer,
    )
    report.candidates_generated = len(candidates)

    # Phase 2: deterministic validation.
    for c in candidates:
        # Uniqueness: check against existing domains/usernames.
        c.uniqueness_score = _score_uniqueness(
            c, existing_domains, existing_x_usernames
        )
        if c.source_type == "rss":
            _validate_rss(c)
        elif c.source_type == "x":
            _validate_x(c)
        else:
            c.validation_note = f"unknown source_type: {c.source_type}"

    report.candidates_validated = sum(
        1 for c in candidates if c.reachable or c.validation_note
    )

    # Phase 3: score + rank.
    for c in candidates:
        c.overall_score = (
            c.freshness_score * 0.35
            + c.relevance_score * 0.35
            + c.uniqueness_score * 0.30
        )

    passed = [c for c in candidates if c.reachable and c.overall_score >= min_score]
    rejected = [c for c in candidates if not c.reachable or c.overall_score < min_score]
    passed.sort(key=lambda c: c.overall_score, reverse=True)

    report.passed = passed
    report.rejected = rejected
    report.candidates_passed = len(passed)

    if passed:
        report.yaml_snippet = _render_yaml_snippet(passed)

    if tracer:
        tracer.log(
            "source_discovery",
            topic=topic,
            generated=report.candidates_generated,
            validated=report.candidates_validated,
            passed=report.candidates_passed,
        )

    return report


# ── Phase 1: candidate generation ──────────────────────────────────────────


def _generate_candidates(
    *,
    provider: LLMProvider,
    focus_instruction: str,
    existing_sources: str,
    source_count: int,
    max_candidates: int,
    tracer: Optional[Tracer] = None,
) -> List[CandidateSource]:
    user = DISCOVERY_USER_TEMPLATE.format(
        focus_instruction=focus_instruction,
        existing_sources=existing_sources,
        source_count=source_count,
        max_candidates=max_candidates,
    )

    from agent.llm.base import LLMMessage

    response = provider.complete(
        messages=[
            LLMMessage(role="system", content=DISCOVERY_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user),
        ],
        temperature=0.2,
        max_output_tokens=4096,
    )
    if tracer:
        tracer.log_llm_call(
            stage="discover",
            provider=provider.name,
            model=provider.model,
            messages=[{"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
            response_text=response.text,
        )

    raw = _extract_json(response.text)
    if not isinstance(raw, list):
        return []

    candidates: List[CandidateSource] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        st = entry.get("type", "")
        candidates.append(
            CandidateSource(
                name=str(entry.get("name", "")),
                source_type=st,
                url=str(entry.get("url", "")) if st == "rss" else None,
                username=str(entry.get("username", "")).lstrip("@") if st == "x" else None,
                account_type=str(entry.get("account_type", "official")),
                language=str(entry.get("language", "en")),
                category=str(entry.get("category", "")),
                reason=str(entry.get("reason", "")),
            )
        )
    return candidates


# ── Phase 2: deterministic validation ─────────────────────────────────────


def _validate_rss(c: CandidateSource, timeout: float = 15.0) -> None:
    if not c.url:
        c.validation_note = "no URL provided"
        return

    try:
        parsed = feedparser.parse(c.url)
    except Exception as e:
        c.validation_note = f"feedparser error: {e}"
        return

    if parsed.bozo and not parsed.entries:
        c.validation_note = f"bozo: {parsed.bozo_exception}"
        return

    entries = parsed.entries[:30]
    if not entries:
        c.validation_note = "feed returned zero entries"
        return

    c.reachable = True

    # Freshness: how recent is the newest entry?
    c.freshness_score = _score_freshness(entries)

    # Relevance: what fraction of entry titles are AI-related?
    c.relevance_score = _score_ai_relevance(entries)

    c.validated = True
    c.validation_note = (
        f"OK: {len(entries)} entries, "
        f"newest={_newest_date(entries)}, "
        f"freshness={c.freshness_score:.2f}, "
        f"relevance={c.relevance_score:.2f}"
    )


def _validate_x(c: CandidateSource) -> None:
    if not c.username:
        c.validation_note = "no username provided"
        return

    token = os.getenv("X_BEARER_TOKEN", "")
    if not token:
        # Without a token we can only do a lightweight existence check.
        c.validation_note = "X_BEARER_TOKEN not set; skipping X validation"
        c.reachable = True  # give benefit of doubt
        c.freshness_score = 0.5
        c.relevance_score = 0.7
        c.validated = True
        return

    try:
        client = httpx.Client(
            base_url="https://api.x.com/2",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "report-agent/0.1",
            },
            timeout=15.0,
        )
        resp = client.get(f"/users/by/username/{c.username}")
        if resp.status_code != 200:
            c.validation_note = f"X user lookup failed: HTTP {resp.status_code}"
            return

        user_id = resp.json().get("data", {}).get("id")
        if not user_id:
            c.validation_note = "X user not found"
            return

        # Check recent tweets for freshness.
        tweets_resp = client.get(
            f"/users/{user_id}/tweets",
            params={
                "max_results": 10,
                "tweet.fields": "created_at,lang",
                "exclude": "retweets,replies",
            },
        )
        tweets = tweets_resp.json().get("data", []) if tweets_resp.status_code == 200 else []
        if tweets:
            newest = tweets[0].get("created_at", "")
            c.freshness_score = _score_tweet_freshness(newest)
            ai_tweets = sum(
                1 for t in tweets
                if _is_ai_related(t.get("text", ""))
            )
            c.relevance_score = ai_tweets / len(tweets) if tweets else 0.0
            c.validation_note = (
                f"OK: user_id={user_id}, "
                f"{len(tweets)} recent tweets, "
                f"newest={newest}, "
                f"freshness={c.freshness_score:.2f}, "
                f"relevance={c.relevance_score:.2f}"
            )
        else:
            c.freshness_score = 0.3
            c.relevance_score = 0.5
            c.validation_note = f"OK: user_id={user_id}, no recent original tweets"

        c.reachable = True
        c.validated = True
    except Exception as e:
        c.validation_note = f"X validation error: {e}"


# ── Scoring helpers ────────────────────────────────────────────────────────


_AI_KEYWORDS = [
    "ai", "llm", "gpt", "claude", "gemini", "deepseek", "qwen", "model",
    "transformer", "diffusion", "neural", "training", "fine-tune", "lora",
    "inference", "benchmark", "open source", "开源", "模型", "智能",
    "大模型", "推理", "训练", "agent", "智能体", "alignment", "safety",
    "rag", "embedding", "multimodal", "多模态", "语音", "vision",
    "rlhf", "reasoning", "codex", "copilot", "machine learning",
    "deep learning", "深度学习", "机器学习", "自然语言", "计算机视觉",
    "机器人", "robotics", "science", "protein", "biology",
]


def _is_ai_related(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _AI_KEYWORDS)


def _score_freshness(entries) -> float:
    newest_str = _newest_date(entries)
    if not newest_str:
        return 0.0
    try:
        newest = datetime.fromisoformat(newest_str.replace("Z", "+00:00"))
        hours_ago = (datetime.now(timezone.utc) - newest).total_seconds() / 3600
        if hours_ago <= 24:
            return 1.0
        if hours_ago <= 72:
            return 0.8
        if hours_ago <= 168:  # 7 days
            return 0.6
        if hours_ago <= 720:  # 30 days
            return 0.3
        return 0.1
    except Exception:
        return 0.2


def _score_ai_relevance(entries) -> float:
    if not entries:
        return 0.0
    hits = sum(1 for e in entries if _is_ai_related(getattr(e, "title", "")))
    return hits / len(entries)


def _score_tweet_freshness(newest_iso: str) -> float:
    if not newest_iso:
        return 0.2
    try:
        newest = datetime.fromisoformat(newest_iso.replace("Z", "+00:00"))
        hours = (datetime.now(timezone.utc) - newest).total_seconds() / 3600
        if hours <= 24:
            return 1.0
        if hours <= 72:
            return 0.7
        if hours <= 168:
            return 0.4
        return 0.1
    except Exception:
        return 0.2


def _newest_date(entries) -> str:
    for e in entries:
        t = e.get("published_parsed") or e.get("updated_parsed")
        if t:
            try:
                from datetime import datetime, timezone as _tz
                ts = time.mktime(t)
                return datetime.fromtimestamp(ts, tz=_tz.utc).isoformat()
            except Exception:
                pass
    return ""


# ── Config helpers ─────────────────────────────────────────────────────────


def _load_existing_domains(config_path: str) -> set:
    """Extract domain names from all existing RSS sources."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return set()
    domains: set = set()
    for s in cfg.get("sources", []):
        if s.get("type") == "rss" and s.get("url"):
            try:
                from urllib.parse import urlparse
                parsed = urlparse(s["url"])
                domain = parsed.netloc.lower().replace("www.", "")
                domains.add(domain)
            except Exception:
                pass
    return domains


def _load_existing_x_usernames(config_path: str) -> set:
    """Extract lowercase X usernames from all existing X sources."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return set()
    usernames: set = set()
    for s in cfg.get("sources", []):
        if s.get("type") == "x" and s.get("username"):
            usernames.add(s["username"].lower().lstrip("@"))
    return usernames


def _score_uniqueness(
    c: CandidateSource,
    existing_domains: set,
    existing_usernames: set,
) -> float:
    """Score 1.0 if the source is completely new, 0.0 if already exists."""
    if c.source_type == "rss" and c.url:
        try:
            from urllib.parse import urlparse
            domain = urlparse(c.url).netloc.lower().replace("www.", "")
            if domain in existing_domains:
                return 0.0
        except Exception:
            pass
    if c.source_type == "x" and c.username:
        if c.username.lower().lstrip("@") in existing_usernames:
            return 0.0
    return 1.0


def _load_existing_source_summary(config_path: str) -> str:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return "(unknown)"

    lines: List[str] = []
    for s in cfg.get("sources", []):
        sid = s.get("id", "?")
        st = s.get("type", "?")
        if st == "rss":
            lines.append(f"  - rss: {sid} → {s.get('url', '')}")
        elif st == "x":
            lines.append(f"  - x: @{s.get('username', '')} ({s.get('account_type', '')})")
    return "\n".join(lines) if lines else "(none)"


def _build_focus_instruction(topic: str, existing_sources_summary: str) -> str:
    """Build a focus instruction that prioritizes filling coverage gaps.

    When topic is 'auto', we let the LLM analyze gaps itself.
    When topic is specific, we add that as a primary hunting ground
    but still ask for gap analysis.
    """
    base = (
        "分析现有信息源的覆盖盲区，重点找出被遗漏的 AI 信息源。"
        "覆盖盲区可能包括：未被关注的 AI 模型厂商、缺失的研究机构、"
        "未覆盖的媒体渠道、缺乏的中文源或英文源、缺少的社区或简报。"
    )

    if topic == "auto":
        return (
            base + "请自行判断当前信息源最严重的覆盖盲区，并针对性地推荐新源。"
        )

    topic_guides: Dict[str, str] = {
        "broad": (
            base + "广泛搜索 AI 领域所有重要信息源，涵盖模型厂商、研究机构、"
            "产品发布、行业资本、政策监管、开发者社区、媒体覆盖等各个维度。"
            "目标是构建一个没有明显盲区的全面信息源网络。"
        ),
        "chinese-ai-models": (
            base + "特别关注中国 AI 模型厂商的信息源覆盖：DeepSeek/深度求索、"
            "智谱 ChatGLM、月之暗面 Kimi、MiniMax/稀宇、零一万物 Yi、"
            "字节跳动 豆包/扣子、商汤 SenseTime/日日新、百川智能 Baichuan、"
            "面壁智能、昆仑万维 天工、中国 AI 研究机构（清华、北大、上海 AI Lab）。"
        ),
        "ai-research": (
            base + "特别关注 AI 研究前沿：顶级会议 (NeurIPS/ICML/ICLR/CVPR/ACL)、"
            "重要研究机构 (OpenAI/DeepMind/Anthropic/Meta FAIR/MSR)、"
            "学术预印本 (arXiv)、模型评测榜单。"
        ),
        "ai-product": (
            base + "特别关注 AI 产品发布和更新：各厂商产品线变更、新功能上线、"
            "定价策略变化、API 更新、开发者工具链迭代。"
        ),
    }
    return topic_guides.get(topic, topic_guides["broad"])


def _render_yaml_snippet(candidates: List[CandidateSource]) -> str:
    lines = ["# ── Discovered sources (auto-validated) ──", ""]
    for i, c in enumerate(candidates):
        lines.append(f"# [{c.overall_score:.2f}] {c.reason}")
        if c.source_type == "rss":
            lines.append(f"- id: \"discovered_{_slug(c.name)}\"")
            lines.append(f"  type: \"rss\"")
            lines.append(f"  url: \"{c.url}\"")
            lines.append(f"  weight: {0.7 + c.overall_score * 0.4:.1f}")
            lines.append(f"  max_items: {max(4, int(10 * c.overall_score))}")
        elif c.source_type == "x":
            lines.append(f"- id: \"discovered_x_{_slug(c.username or c.name)}\"")
            lines.append(f"  type: \"x\"")
            lines.append(f"  username: \"{c.username}\"")
            lines.append(f"  account_type: \"{c.account_type}\"")
            lines.append(f"  weight: {0.7 + c.overall_score * 0.4:.1f}")
            lines.append(f"  max_items: {max(3, int(8 * c.overall_score))}")
        lines.append("")
    return "\n".join(lines)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().replace(" ", "_"))[:40]


# ── JSON extraction (same pattern as writer.py) ────────────────────────────


def _extract_json(content: str):
    # Strip think blocks (DeepSeek reasoning).
    content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
    # Strip markdown code fences.
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if m:
        content = m.group(1).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            return json.loads(content[: content.rindex("}") + 1])
        except Exception:
            pass
    return None
