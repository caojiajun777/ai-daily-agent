"""Research Editor Agent — evidence-grounded editorial decision maker.

The LLM's role is NOT to score items 1-10. It is to make editorial judgments:
  select / reject, priority, section, evidence_level, novelty,
  reader_utility, writing_angle, risk_level.

All decisions are constrained: event_id must exist, sources_to_use URLs
must come from the event's source_urls, no URL fabrication allowed.
"""

from __future__ import annotations

import json as _json
import re as _re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent.agents.event_clusterer import EventCluster
from agent.agents.section_classifier import guess_section

# ═══════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════


class SourceUse(BaseModel):
    url: str
    role: str = "primary"  # primary | supporting


class EditorialDecision(BaseModel):
    event_id: str
    decision: str  # select | reject
    priority: str = "medium"  # must_include | high | medium | low
    section: str | None = None
    evidence_level: str = "primary"  # official | primary | trusted_media | social | weak
    novelty: str = "new_event"  # new_event | meaningful_update | repeated_without_update | unclear
    reader_utility: str = "medium"  # high | medium | low
    why_it_matters: str = ""
    writing_angle: str = ""
    risk_level: str = "low"  # low | medium | high
    sources_to_use: List[SourceUse] = []
    reject_reason: str | None = None


class ResearchEditorOutput(BaseModel):
    selected: List[EditorialDecision] = []
    rejected: List[EditorialDecision] = []
    notes: str | None = None


# ═══════════════════════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════════════════════

_RESEARCH_EDITOR_PROMPT = """你是一个面向 AI 开发者、研究者和技术管理者的中文 AI 日报资深编辑。

你的任务不是给新闻打 1-10 分或按热度排序。你的任务是基于候选事件、证据摘要和历史上下文，选择今天最值得读者看到的事件。

候选中的 evidence 字段来自对原始 URL 的抓取结果，是事实核验和细节提取的优先依据。若 evidence 与摘要不一致，以 evidence 为准；若 evidence 抓取失败或只有社交/媒体二手转述，应降低 confidence、使用 reported language，必要时标记 risk_level=high 或 reject。

## 日报 7 板块
1. 要闻 — 当日最重要、最有影响力、读者不该错过的 1-3 条
2. 模型发布 — 新模型、模型能力、模型开源、推理/语音/多模态模型
3. 开发生态 — SDK/API/框架/CLI/IDE/开源项目/定价变动/开发者工具
4. 技术与洞察 — 论文、研究进展、安全漏洞、Benchmark、技术报告、重要工程洞察
5. 产品应用 — 产品发布、功能更新、真实应用案例、面向用户的能力上线
6. 行业动态 — 政策/监管/财报/融资/IPO/估值/并购/人事/公司战略/产业合作
7. 前瞻与传闻 — A/B 测试、爆料、未确认融资、路线图、社区发现、弱信号

## 优先选择
1. 官方发布、论文、代码、文档、API、价格、benchmark、一手证据充分的事件
2. API 降价、模型上线、开发者工具变化、开源发布等能影响读者当天决策的事件
3. 重要财报、融资、IPO、收购等资本事件（earnings/revenue/funding/IPO/acquisition）
4. 对中文 AI 开发者、创业团队、产品团队有直接价值的中国模型/工具/政策动态
5. 对 AI 模型、开发工具链、产品生态、企业采用、政策环境、研究社区有实际影响的事件
6. 相比历史日报有新增信息的事件
7. 能改变读者判断或行动的事件
8. 能补充日报结构的事件，但不要为了补齐板块选择低价值内容

## 降低优先级
1. 旧闻换标题
2. 媒体二次转述且无新增信息
3. 只有营销措辞没有技术/产品细节
4. 泛科技新闻中 AI 关联较弱
5. 无法验证的爆料，除非它适合放入“前瞻与传闻”且明确标注不确定性
6. 同一事件重复来源
7. 窄领域论文，除非有明确方法突破、重要机构、代码结果或与当天大事件互相印证

## 约束
1. 只能选择给定候选中的 event_id
2. sources_to_use 中的 URL 必须来自该 event 的 source_urls
3. 不允许生成新 URL
4. 不允许编造事实
5. 如果证据不足应标记 risk_level=high 或 reject
6. 只输出 JSON，不要 markdown，不要解释
7. rejected 数组最多 5 条；不要为了覆盖所有未选候选而输出长 rejected 列表

## 输出格式
严格按照以下 JSON 结构输出：
{
  "selected": [
    {
      "event_id": "evt_xxx",
      "decision": "select",
      "priority": "must_include | high | medium | low",
       "section": "要闻 | 模型发布 | 开发生态 | 技术与洞察 | 产品应用 | 行业动态 | 前瞻与传闻",
      "evidence_level": "official | primary | trusted_media | social | weak",
      "novelty": "new_event | meaningful_update | repeated_without_update | unclear",
      "reader_utility": "high | medium | low",
      "why_it_matters": "一句话说清为什么这条对 AI 从业者重要",
      "writing_angle": "给 Writer 的写作角度建议，30 字以内",
      "risk_level": "low | medium | high",
      "sources_to_use": [
        {"url": "https://...", "role": "primary"},
        {"url": "https://...", "role": "supporting"}
      ]
    }
  ],
  "rejected": [
    {
      "event_id": "evt_xxx",
      "decision": "reject",
      "reject_reason": "拒绝理由"
    }
  ],
  "notes": "可选的整体编辑备注"
}

## 优先级定义
- must_include: 今天不写会明显漏掉重大事件（最多 3 条）
- high: 应进入主日报
- medium: 可用于补充板块
- low: 仅候选不足时使用

selected 数量控制在 16-25 条。不要重复。"""


# ═══════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════


def run_research_editor(
    *,
    events: List[EventCluster],
    evidence: Optional[List[List[Any]]] = None,
    history_titles: Optional[List[str]] = None,
    provider=None,
    tracer=None,
    budget=None,
    timeout_sec: int = 60,
) -> ResearchEditorOutput:
    """Run the ResearchEditor LLM agent. Falls back to empty output on any error."""

    # Build the candidate listing for the LLM.
    candidate_lines: List[str] = []
    for i, evt in enumerate(events[:50]):
        evlist = evidence[i] if evidence and i < len(evidence) else []
        candidate_lines.append(
            f"[{evt.event_id}] rule_score={evt.rule_score:.3f} "
            f"sources={evt.source_count} "
            f"title={evt.canonical_title[:120]}"
        )
        candidate_lines.append(
            f"     source_meta: content_type={evt.primary_content_type}; "
            f"tier={evt.primary_source_tier}; reliability={evt.primary_reliability}; "
            f"evidence_type={evt.primary_evidence_type}; "
            f"confidence={evt.primary_confidence}; "
            f"section_hint={evt.primary_section_hint}"
        )
        if evt.summary:
            candidate_lines.append(f"     summary: {evt.summary[:200]}")
        candidate_lines.append(
            f"     source_urls: {', '.join(evt.source_urls[:4])}"
        )
        evidence_lines = _candidate_evidence_lines(evt, evlist)
        if evidence_lines:
            candidate_lines.append("     evidence:")
            candidate_lines.extend(f"       - {line}" for line in evidence_lines)
        candidate_lines.append("")

    # History context.
    hist_text = ""
    if history_titles:
        hist_text = "最近 7 天已报道：\n" + "\n".join(
            f"- {t[:100]}" for t in history_titles[:30]
        ) + "\n\n"

    user_msg = (
        f"{hist_text}"
        f"候选事件列表：\n\n"
        + "\n".join(candidate_lines)
    )

    from agent.llm.base import LLMMessage

    # ── Attempt LLM call ───────────────────────────────────────────
    try:
        if budget:
            budget.check_can_call(stage="curate_editor")

        response = provider.complete(
            messages=[
                LLMMessage(role="system", content=_RESEARCH_EDITOR_PROMPT),
                LLMMessage(role="user", content=user_msg),
            ],
            temperature=0.1,
            max_output_tokens=6144,
        )

        if tracer:
            tracer.log_llm_call(
                provider=provider.name, model=provider.model,
                prompt=_RESEARCH_EDITOR_PROMPT + "\n" + user_msg,
                output=response.text, latency_ms=response.latency_ms,
                status="ok", stage="curate_editor",
            )
        if budget:
            budget.record(
                stage="curate_editor",
                input_tokens=response.input_tokens_est,
                output_tokens=response.output_tokens_est,
            )

        # Parse + validate.
        output = _parse_and_validate(response.text, events)
        if tracer:
            tracer.log(
                "editor_parse_result",
                raw_len=len(response.text),
                raw_preview=response.text[:300],
                raw_suffix=response.text[-300:],
                selected_count=len([d for d in output.selected if d.decision == "select"]),
                rejected_count=len(output.rejected),
                notes=(output.notes or "")[:300],
            )

    except Exception as e:
        if tracer:
            tracer.log("research_editor_failed", error=str(e))
        output = ResearchEditorOutput(notes=f"LLM failed: {e}")

    return output


def _candidate_evidence_lines(evt: EventCluster, evlist: List[Any]) -> List[str]:
    lines: List[str] = []
    if evlist:
        for snippet in evlist[:2]:
            status = getattr(snippet, "fetch_status", "")
            etype = getattr(snippet, "evidence_type", "")
            title = " ".join(str(getattr(snippet, "title", "")).split())[:120]
            text = " ".join(str(getattr(snippet, "text_snippet", "")).split())[:280]
            url = getattr(snippet, "url", "")
            parts = [f"status={status}", f"type={etype}"]
            if title:
                parts.append(f"title={title}")
            if text:
                parts.append(f"text={text}")
            if url:
                parts.append(f"url={url}")
            lines.append(" | ".join(parts))
    elif evt.evidence_snippets:
        lines.extend(evt.evidence_snippets[:2])
    return lines


def _parse_and_validate(
    raw_text: str,
    events: List[EventCluster],
) -> ResearchEditorOutput:
    """Parse LLM JSON output and validate against candidate constraints."""

    # Build lookup maps for validation.
    valid_event_ids = {e.event_id for e in events}
    event_map = {e.event_id: e for e in events}
    event_urls = {e.event_id: set(e.source_urls) for e in events}

    # Strip think blocks / fences / conversational prefixes.
    raw = raw_text.strip()
    raw = _re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=_re.IGNORECASE).strip()
    m = _re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()

    # Try to parse JSON. If it fails, try extracting from { ... } brackets
    # (deepseek-chat often adds conversational text around the JSON).
    payload = None
    parse_error = ""
    try:
        payload = _json.loads(raw)
    except _json.JSONDecodeError as e:
        parse_error = f"direct_parse_at_{e.pos}: {raw[e.pos-20:e.pos+20] if e.pos < len(raw) else 'EOF'}"
        # Extract JSON object from between outermost { and }.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                payload = _json.loads(raw[start:end + 1])
            except _json.JSONDecodeError as e2:
                parse_error += f" | bracket_parse_at_{e2.pos}: {raw[start:end+1][e2.pos-20:e2.pos+20] if e2.pos < len(raw[start:end+1]) else 'EOF'}"
                offset = start + e2.pos
                parse_error += f" | around_offset_{offset}: {raw[offset-30:offset+30]}"
        if payload is None:
            payload = _salvage_selected_payload(raw)
            if payload is not None:
                parse_error += " | salvaged_selected_only"

    if payload is None:
        return ResearchEditorOutput(notes=f"JSON parse failed: {parse_error}")

    if not isinstance(payload, dict):
        return ResearchEditorOutput(notes="Not a dict")

    # Validate with Pydantic, catching schema issues gracefully.
    try:
        output = ResearchEditorOutput.model_validate(payload)
    except Exception as first_error:
        coerced = _coerce_editor_payload(payload)
        try:
            output = ResearchEditorOutput.model_validate(coerced)
        except Exception:
            return ResearchEditorOutput(notes=f"Schema validation failed: {first_error}")

    # ── Post-validation constraints ─────────────────────────────────
    valid_selected: List[EditorialDecision] = []
    url_warnings: List[str] = []
    for d in output.selected:
        if d.event_id not in valid_event_ids:
            url_warnings.append(f"invalid_event_id={d.event_id}")
            continue
        allowed_urls: set = event_urls.get(d.event_id, set())
        if not allowed_urls:
            url_warnings.append(f"no_source_urls_for={d.event_id}")
            continue

        # Check each source URL — drop fabricated ones.
        clean_sources: List[SourceUse] = []
        for s in d.sources_to_use:
            if s.url in allowed_urls:
                clean_sources.append(s)
            else:
                url_warnings.append(f"invalid_llm_url_removed={s.url}")

        # If no valid sources remain, fall back to event's primary_url.
        if not clean_sources:
            urls = list(allowed_urls)
            clean_sources = [SourceUse(url=urls[0], role="primary")]
            if len(urls) > 1:
                clean_sources.append(SourceUse(url=urls[1], role="supporting"))
            url_warnings.append(
                f"all_sources_dropped_for={d.event_id},"
                f" fallback_to={urls[0][:80]}"
            )

        d.sources_to_use = clean_sources
        if d.decision == "select":
            d.reject_reason = None
        valid_selected.append(d)

    output.selected = valid_selected

    # Store warnings in notes field.
    if url_warnings:
        existing = output.notes or ""
        output.notes = (existing + " | warnings: " + "; ".join(url_warnings[:5])).strip()

    for d in output.rejected:
        if d.decision == "reject" and not d.reject_reason:
            d.reject_reason = "no reason given"

    valid_rejected: List[EditorialDecision] = []
    for d in output.rejected:
        if d.event_id not in valid_event_ids:
            continue
        if d.decision == "reject" and not d.reject_reason:
            d.reject_reason = "no reason given"
        valid_rejected.append(d)

    output.selected = valid_selected
    output.rejected = valid_rejected

    # Ensure selected have normalized sections.
    for d in output.selected:
        if d.decision == "select":
            d.section = _normalize_editor_section(d.section or "行业动态")

    # arXiv/HF papers placed in non-technical sections should move to
    # 技术与洞察. Exception: 要闻 can keep a paper if it is genuinely a top
    # story for the day.
    paper_sec_warnings: List[str] = []
    for d in output.selected:
        if d.decision != "select" or d.section == "技术与洞察":
            continue
        if d.section == "要闻":
            continue
        has_paper_url = any(
            "arxiv.org" in s.url or "huggingface.co/papers" in s.url
            for s in d.sources_to_use
        )
        if has_paper_url:
            old_section = d.section
            d.section = "技术与洞察"
            paper_sec_warnings.append(
                f"moved_{d.event_id}_to_tech_insight_from_{old_section}: has arxiv/HF source URL"
            )
    # Deterministic section sanity pass. Keep 要闻 stable; it is an editorial
    # priority bucket rather than a topic.
    section_warnings: List[str] = []
    for d in output.selected:
        if d.decision != "select" or d.section == "要闻":
            continue
        evt = event_map.get(d.event_id)
        if evt is None:
            continue
        guessed = _guess_editor_section(evt)
        if guessed and guessed != d.section:
            old_section = d.section
            d.section = guessed
            section_warnings.append(
                f"moved_{d.event_id}_from_{old_section}_to_{guessed}: deterministic_section_check"
            )
    if paper_sec_warnings:
        existing = output.notes or ""
        output.notes = (existing + " | paper_sec_fix: " + "; ".join(paper_sec_warnings[:5])).strip()
    if section_warnings:
        existing = output.notes or ""
        output.notes = (existing + " | section_fix: " + "; ".join(section_warnings[:8])).strip()

    return output


def _coerce_editor_payload(payload: Any) -> Any:
    """Tolerate minor LLM schema drift without dropping valid selections."""
    if not isinstance(payload, dict):
        return payload

    data = dict(payload)
    for list_key, default_decision in (("selected", "select"), ("rejected", "reject")):
        raw_items = data.get(list_key, [])
        if not isinstance(raw_items, list):
            data[list_key] = []
            continue

        clean_items: List[Dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict) or not item.get("event_id"):
                continue
            obj = dict(item)
            decision = str(obj.get("decision") or default_decision).strip().lower()
            if list_key == "selected":
                decision = "select" if decision not in {"reject"} else decision
            else:
                decision = "reject"
            obj["decision"] = decision

            sources = obj.get("sources_to_use", [])
            clean_sources: List[Dict[str, Any]] = []
            if isinstance(sources, list):
                for src in sources:
                    if isinstance(src, str) and src:
                        clean_sources.append({"url": src, "role": "primary"})
                    elif isinstance(src, dict) and src.get("url"):
                        clean_sources.append(dict(src))
            obj["sources_to_use"] = clean_sources
            clean_items.append(obj)

        data[list_key] = clean_items
    return data


def _guess_editor_section(evt: EventCluster) -> str:
    """Lightweight non-headline section validator for ResearchEditor output."""
    return _normalize_editor_section(guess_section(evt))


def _normalize_editor_section(section: str) -> str:
    aliases = {
        "今日头条": "要闻",
        "模型前沿": "模型发布",
        "工具与开源": "开发生态",
        "论文精选": "技术与洞察",
        "产品落地": "产品应用",
        "资本动向": "行业动态",
        "产业风向": "行业动态",
        "业界风向": "行业动态",
    }
    valid = {
        "要闻", "模型发布", "开发生态", "技术与洞察",
        "产品应用", "行业动态", "前瞻与传闻",
    }
    mapped = aliases.get(section, section)
    return mapped if mapped in valid else "行业动态"


def _salvage_selected_payload(raw: str) -> Optional[Dict[str, Any]]:
    """Recover a complete selected array when the rejected tail is truncated."""
    key = '"selected"'
    key_pos = raw.find(key)
    if key_pos == -1:
        return None
    array_start = raw.find("[", key_pos)
    if array_start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(array_start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                selected_text = raw[array_start:idx + 1]
                try:
                    selected = _json.loads(selected_text)
                except Exception:
                    return None
                if isinstance(selected, list):
                    return {"selected": selected, "rejected": [], "notes": "salvaged selected; rejected omitted"}
                return None
    return None
