"""LLM Re-ranker — editorial-gate scoring for AI social-media daily reports.

After rule-based scoring, an LLM evaluates each event across 7 dimensions
(newsworthiness, freshness, novelty, audience, publishability, evidence, risk)
and assigns a recommended slot. This catches financial/earnings/product-launch
news undervalued by deterministic rules.

Flow:
  1. Rule-based scoring (event_scorer.py) → top-N events
  2. LLM Re-rank (this module) → multi-dimension editorial scores + slot
  3. Research Editor → final editorial selection
"""

from __future__ import annotations

import json as _json
import math as _math
import os as _os
import re as _re
from datetime import datetime as _datetime
from typing import Any, Dict, List, Optional

from agent.agents.event_clusterer import EventCluster
from agent.llm.base import LLMMessage

_RERANK_PROMPT = """你是一个 AI 新闻日报的资深编辑，任务是对候选事件进行二次排序。你的目标不是判断技术先进性，而是判断这件事是否值得进入"今日 AI 社媒日报"。

日报面向中文 AI 开发者、创业团队、产品经理、研究者和技术管理者。请特别重视会影响读者当天行动的内容：API/模型价格、模型上线、开源权重、开发工具、IDE/CLI/Agent 工作流、国产模型和国内政策。不要让泛大厂新闻、资本传闻或窄论文挤掉更有用的开发生态信息。

请基于候选事件的标题、摘要、来源、发布时间、URL 和已有上下文，对每个事件独立评分。不要因为事件来自技术媒体、论文、公司博客或社交媒体就自动高分或低分；只评估事件本身的真实新闻价值、社媒传播价值和事实可靠性。

评分维度如下：

1. 真实新闻价值 newsworthiness_score，1-10 分
评估这件事对 AI 行业、资本市场、开发者生态、企业用户、普通 AI 用户是否有实际影响。
- 9-10：全行业头条级事件。例：前沿模型重大代际发布；OpenAI/Anthropic/Google/Meta 级别公司的重大商业、产品、资本或安全事件；NVIDIA/云厂商财报中出现影响 AI 产业链的重大信号。
- 7-8：行业高度关注事件。例：重要模型发布、重大融资/收购、大厂 AI 产品策略变化、重要监管/安全事件、明显影响开发者生态的开源项目。
- 5-6：值得关注但不是头条。例：普通产品更新、一般融资、普通 benchmark 进展、有一定应用价值的研究。
- 3-4：小众或增量更新。例：窄领域论文、普通工具更新、缺乏明确影响的技术博客。
- 1-2：不值得占用日报版面。例：重复新闻、营销软文、无新增信息、影响很小的版本更新。

2. 新鲜度 freshness_score，1-10 分
评估该事件相对当前日期是否新，以及是否包含新的信息增量。
- 今天刚发生、首发、独家、或有明显新增事实：高分。
- 旧闻重发、转载、无新增信息：低分。
- 首发不是充分条件。只有事件本身重要时，首发才加分。

3. 稀缺性 novelty_score，1-10 分
评估这类事件是否罕见，以及是否有超出常规周期的信号。
- 罕见且重要：高分。
- 常规事件但数据、规模、影响异常：可以高分。
- 罕见但影响很小：不要高分。
- 常规财报、普通融资、普通模型小版本发布，不因类别本身高分；只有出现明显超预期信息时才加分。

4. 受众广度 audience_breadth_score，1-10 分
评估普通 AI 从业者、开发者、产品经理、创业者、投资人、AI 用户是否会关心。
- 产品发布、模型能力变化、价格变化、资本市场信号、开发者工具变化：通常受众更广。
- 纯学术细节、窄领域方法、单个 benchmark 小幅提升：通常受众较窄，除非影响极大。

5. 社媒发布价值 social_publishability_score，1-10 分
评估这件事是否适合做成社媒日报内容。
高分事件通常具备：
- 标题容易讲清楚；
- 有明确冲突、转折、数字、公司、人物或行业影响；
- 容易做成一张信息图；
- 用户看完会觉得"这和我有关"或"值得收藏/转发"。

低分事件通常是：
- 过于技术细节；
- 需要大量背景才能理解；
- 缺乏明确结论；
- 只有小圈子会关心。

6. 事实可靠性 evidence_strength_score，1-10 分
评估来源是否可靠，信息是否可核查。
- 官方公告、财报、论文、权威媒体、多源交叉验证：高分。
- 单一来源、匿名爆料、标题党、未证实传言：低分。
如果证据不足，即使事件看起来很热，也不要给过高分。

7. 风险 risk_penalty，0-3 分
如果存在以下情况，给出惩罚：
- 可能是未证实传闻；
- 标题容易误导；
- 来源弱或只有二手转述；
- 事件表述可能夸大；
- 与候选集中其他事件重复；
- 可能只是营销软文。

请注意：
- 不要自动低估科技媒体新闻。TechCrunch、The Verge、VentureBeat 等来源如果报道的是大公司、重要融资、产品战略、资本市场、AI 基础设施或模型公司商业化信号，可以给高分。
- 不要自动高估论文。论文只有在方法影响大、机构重要、结果明确、引发行业关注或有产品化潜力时才给高分。
- 不要自动高估开源、代码、benchmark。只有当它对开发者生态或行业采用有实际影响时才高分。
- API 降价、永久优惠、模型接入主流平台、免费/BYOK/开源等“读者能立刻用上”的事件应提高 audience_breadth_score 和 social_publishability_score。
- 未确认融资、A/B 测试、路线图、爆料可以进入"前瞻与传闻"，但 evidence_strength_score 和 confidence_score 必须反映不确定性。
- 不要被标题中的"突破""首次""重磅""颠覆"等词直接影响，必须根据具体事实判断。
- 如果信息不足以判断，不要猜测，降低 confidence_score。
- 不要输出虚构事实。只能基于输入事件内容判断。

最终只输出紧凑 JSON 数组。每个事件只输出以下字段：

{
  "event_id": "...",
  "newsworthiness_score": 1-10,
  "freshness_score": 1-10,
  "novelty_score": 1-10,
  "audience_breadth_score": 1-10,
  "social_publishability_score": 1-10,
  "evidence_strength_score": 1-10,
  "risk_penalty": 0-3,
  "confidence_score": 0.0-1.0,
  "recommended_slot": "headline | main | secondary | skip"
}

推荐规则：
- headline：通常要求 newsworthiness_score >= 8 且 social_publishability_score >= 7 且 evidence_strength_score >= 6。
- main：通常要求 newsworthiness_score >= 6 且 social_publishability_score >= 6。
- secondary：有价值但不适合作为主条。
- skip：不建议占用日报版面。

不需要输出原因和备注。只输出 JSON 数组，不要输出 markdown，不要输出额外解释。"""

_VALID_SLOTS = {"headline", "main", "secondary", "skip"}
_MAX_RERANK_EVENTS = 80
_RERANK_CHUNK_SIZE = 25


def llm_rerank_events(
    *,
    events: List[EventCluster],
    provider=None,
    tracer=None,
    budget=None,
    timeout_sec: int = 60,
    artifacts_root: str = "artifacts",
) -> List[EventCluster]:
    """Re-rank events using multi-dimensional LLM editorial scoring.

    - Saves original rule_score before modification
    - Computes fusion score from 7 LLM dimensions
    - Outputs debug JSON to artifacts/rerank/{date}/
    - On any failure, returns events with original rule_score unchanged
    """
    if not events or not provider:
        return events

    # Preserve original rule_score on each event.
    for evt in events:
        if not hasattr(evt, '_original_rule_score'):
            evt._original_rule_score = evt.rule_score  # type: ignore[attr-defined]

    # ── LLM call ─────────────────────────────────────────────────
    debug_rows: List[Dict[str, Any]] = []
    try:
        result_map: Dict[str, Dict[str, Any]] = {}
        scored_events = events[:_MAX_RERANK_EVENTS]
        for chunk_idx, chunk in enumerate(_chunks(scored_events, _RERANK_CHUNK_SIZE), start=1):
            user_msg = "候选事件列表：\n\n" + "\n".join(_candidate_lines(chunk))
            if budget:
                budget.check_can_call(stage="llm_rerank")

            response = provider.complete(
                messages=[
                    LLMMessage(role="system", content=_RERANK_PROMPT),
                    LLMMessage(role="user", content=user_msg),
                ],
                temperature=0.0,
                    max_output_tokens=6144,
            )

            if tracer:
                tracer.log_llm_call(
                    provider=provider.name, model=provider.model,
                    prompt=_RERANK_PROMPT + "\n" + user_msg,
                    output=response.text, latency_ms=response.latency_ms,
                    status="ok", stage="llm_rerank",
                )
            if budget:
                budget.record(
                    stage="llm_rerank",
                    input_tokens=response.input_tokens_est,
                    output_tokens=response.output_tokens_est,
                )

            raw = _extract_json_array(response.text)
            try:
                llm_results = _loads_array_with_repairs(raw)
            except Exception as e:
                if tracer:
                    tracer.log(
                        "llm_rerank_parse_failed",
                        chunk=chunk_idx,
                        error=str(e),
                        raw_preview=raw[:200],
                    )
                continue
            if not isinstance(llm_results, list):
                if tracer:
                    tracer.log(
                        "llm_rerank_parse_failed",
                        chunk=chunk_idx,
                        error="not a list",
                        raw_preview=raw[:200],
                    )
                continue

            for entry in llm_results:
                parsed = _parse_score_entry(entry)
                if parsed:
                    result_map[str(entry.get("event_id", ""))] = parsed

        if not result_map:
            _write_debug(events, debug_rows, artifacts_root)
            return events

        # ── Fusion formula ──────────────────────────────────────
        for evt in events:
            rm = result_map.get(evt.event_id)
            if rm is None:
                # Partial LLM outputs should not let unscored candidates float
                # above explicitly-rated main/headline items.
                evt.rule_score = round(min(evt.rule_score, 0.45), 4)
                debug_rows.append(_build_debug_row(evt, None, "llm_missed_penalized"))
                continue

            # Compute llm_core.
            llm_core = (
                rm["newsworthiness_score"] * 0.45
                + rm["social_publishability_score"] * 0.25
                + rm["freshness_score"] * 0.15
                + rm["audience_breadth_score"] * 0.10
                + rm["novelty_score"] * 0.05
                - rm["risk_penalty"]
            )

            # Normalize rule_score to 0-10 scale.
            # rule_score currently ranges 0.0-1.0 (from event_scorer).
            norm_rule = _clamp(evt.rule_score * 10.0, 0.0, 10.0)

            # Dynamic LLM weight based on confidence.
            llm_weight = 0.35 + 0.30 * rm["confidence_score"]
            llm_weight = _clamp(llm_weight, 0.35, 0.70)

            final_score = norm_rule * (1.0 - llm_weight) + llm_core * llm_weight
            final_score += _editorial_prior_boost(evt)

            # Evidence gate: if evidence_strength <= 4, cap at 7.0.
            if rm["evidence_strength_score"] <= 4:
                final_score = min(final_score, 7.0)
            final_score = min(final_score, _confidence_cap(evt, rm))

            # Skip penalty: downgrade skip items.
            if rm["recommended_slot"] == "skip":
                final_score = min(final_score, 4.0)

            # Store back: keep rule_score as the final ordering key,
            # but normalize to 0-1 scale for compatibility with downstream.
            evt.rule_score = round(final_score / 10.0, 4)

            debug_rows.append(_build_debug_row(evt, rm, ""))

        events.sort(key=lambda e: e.rule_score, reverse=True)

        if tracer:
            top5 = [(e.event_id, e.rule_score, e.canonical_title[:60])
                    for e in events[:5]]
            tracer.log("llm_rerank_top5", top5=top5)

    except Exception as e:
        if tracer:
            tracer.log("llm_rerank_failed", error=str(e))
        # On failure, restore original rule_scores.
        for evt in events:
            if hasattr(evt, '_original_rule_score'):
                evt.rule_score = evt._original_rule_score  # type: ignore[attr-defined]

    _write_debug(events, debug_rows, artifacts_root)
    return events


# ── Helpers ──────────────────────────────────────────────────────────

def _chunks(values: List[EventCluster], size: int) -> List[List[EventCluster]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def _candidate_lines(events: List[EventCluster]) -> List[str]:
    lines: List[str] = []
    for evt in events:
        has_arxiv = any("arxiv" in s.lower() for s in evt.source_names)
        has_fin = any(k in (evt.canonical_title + " " + evt.summary).lower()
                      for k in ("earning", "revenue", "财报", "融资", "funding", "ipo"))
        ct_hint = ""
        if has_arxiv:
            ct_hint = " [学术论文]"
        elif has_fin:
            ct_hint = " [财经/资本]"

        lines.append(f"[{evt.event_id}]{ct_hint} | {evt.canonical_title[:120]}")
        if evt.summary:
            lines.append(f"     {evt.summary[:200]}")
        lines.append(f"     sources: {evt.source_count} ({', '.join(evt.source_names[:4])})")
        lines.append("")
    return lines


def _editorial_prior_boost(evt: EventCluster) -> float:
    """Small 0-10 scale boost for reader-actionable Chinese AI daily stories."""
    text = f"{evt.canonical_title} {evt.summary}".lower()
    names = " ".join(evt.source_names).lower()
    urls = " ".join(evt.source_urls).lower()
    boost = 0.0
    if any(k in text for k in (
        "price", "pricing", "discount", "free", "byok", "api",
        "降价", "定价", "优惠", "免费", "价格", "接入",
    )):
        boost += 0.45
    if any(k in text or k in names or k in urls for k in (
        "deepseek", "qwen", "通义", "千问", "zhipu", "智谱",
        "glm", "kimi", "moonshot", "豆包", "doubao", "hunyuan", "混元",
    )):
        boost += 0.35
    if any(k in text for k in (
        "cli", "ide", "sdk", "github", "open source", "开源",
        "agent", "copilot", "warp", "trae", "ollama",
    )):
        boost += 0.25
    if any(k in text for k in ("rumor", "爆料", "传闻", "网传", "尚未确认", "未获官方确认")):
        boost -= 0.25
    if any("arxiv" in url for url in evt.source_urls) and not any(
        k in text for k in ("open source", "github", "benchmark", "漏洞", "安全", "代码")
    ):
        boost -= 0.20
    return boost


def _confidence_cap(evt: EventCluster, rm: Dict[str, Any]) -> float:
    text = f"{evt.canonical_title} {evt.summary}".lower()
    weak_signal = (
        rm["risk_penalty"] >= 2
        or rm["confidence_score"] < 0.55
        or any(k in text for k in ("rumor", "爆料", "传闻", "网传", "尚未确认", "未获官方确认"))
    )
    if weak_signal:
        return 6.8
    return 10.0


def _extract_json_array(text: str) -> str:
    raw = text.strip()
    raw = _re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=_re.IGNORECASE).strip()
    m = _re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        return m.group(1).strip()
    raw = _re.sub(r"^```(?:json)?\s*", "", raw).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        return raw[start:end + 1]
    return raw


def _parse_score_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict) or not entry.get("event_id"):
        return None
    parsed = {
        "newsworthiness_score": _clamp(_to_int(entry.get("newsworthiness_score"), 5), 1, 10),
        "freshness_score": _clamp(_to_int(entry.get("freshness_score"), 5), 1, 10),
        "novelty_score": _clamp(_to_int(entry.get("novelty_score"), 5), 1, 10),
        "audience_breadth_score": _clamp(_to_int(entry.get("audience_breadth_score"), 5), 1, 10),
        "social_publishability_score": _clamp(_to_int(entry.get("social_publishability_score"), 5), 1, 10),
        "evidence_strength_score": _clamp(_to_int(entry.get("evidence_strength_score"), 5), 1, 10),
        "risk_penalty": _clamp(_to_int(entry.get("risk_penalty"), 0), 0, 3),
        "confidence_score": _clamp(_to_float(entry.get("confidence_score"), 0.5), 0.0, 1.0),
        "recommended_slot": entry.get("recommended_slot", "main"),
        "one_sentence_reason": str(entry.get("one_sentence_reason", ""))[:120],
        "editor_note": str(entry.get("editor_note", ""))[:200],
    }
    if parsed["recommended_slot"] not in _VALID_SLOTS:
        parsed["recommended_slot"] = "main"
    return parsed


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _loads_array_with_repairs(raw: str) -> List[Dict[str, Any]]:
    candidates = [
        raw,
        _re.sub(r",(\s*[}\]])", r"\1", raw),
    ]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = _json.loads(candidate)
            if isinstance(payload, list):
                return payload
            raise ValueError("rerank JSON root is not a list")
        except Exception as e:
            last_error = e
    raise last_error or ValueError("rerank JSON parse failed")


def _today_str() -> str:
    return _datetime.now().strftime("%Y-%m-%d")


def _build_debug_row(evt: EventCluster, rm: Optional[Dict[str, Any]],
                     fallback: str) -> Dict[str, Any]:
    orig = getattr(evt, '_original_rule_score', evt.rule_score)
    norm_rule = round(_clamp(orig * 10.0, 0.0, 10.0), 2)
    row: Dict[str, Any] = {
        "event_id": evt.event_id,
        "title": evt.canonical_title[:120],
        "sources": evt.source_names[:5],
        "source_count": evt.source_count,
        "published_at": evt.published_at or evt.latest_seen_at or "",
        "rule_score": round(orig, 4),
        "normalized_rule_score": norm_rule,
        "fallback_reason": fallback,
    }
    if rm:
        row.update({
            "newsworthiness_score": rm.get("newsworthiness_score"),
            "freshness_score": rm.get("freshness_score"),
            "novelty_score": rm.get("novelty_score"),
            "audience_breadth_score": rm.get("audience_breadth_score"),
            "social_publishability_score": rm.get("social_publishability_score"),
            "evidence_strength_score": rm.get("evidence_strength_score"),
            "risk_penalty": rm.get("risk_penalty"),
            "confidence_score": rm.get("confidence_score"),
            "recommended_slot": rm.get("recommended_slot"),
            "one_sentence_reason": rm.get("one_sentence_reason"),
            "editor_note": rm.get("editor_note"),
            "llm_core": round(
                rm["newsworthiness_score"] * 0.45
                + rm["social_publishability_score"] * 0.25
                + rm["freshness_score"] * 0.15
                + rm["audience_breadth_score"] * 0.10
                + rm["novelty_score"] * 0.05
                - rm["risk_penalty"], 2,
            ),
            "final_score": round(evt.rule_score * 10.0, 2),
        })
    else:
        row["final_score"] = round(evt.rule_score * 10.0, 2)
    return row


def _write_debug(events: List[EventCluster], rows: List[Dict[str, Any]],
                 artifacts_root: str) -> None:
    """Write rerank debug JSON to artifacts/rerank/{date}/rerank_debug.json."""
    if not rows:
        return
    date_str = _today_str()
    out_dir = _os.path.join(artifacts_root, "rerank", date_str)
    try:
        _os.makedirs(out_dir, exist_ok=True)
        path = _os.path.join(out_dir, "rerank_debug.json")
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(rows, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # Don't let debug output break the pipeline.
