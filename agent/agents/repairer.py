"""Repair agent — fixes semantic duplicates in a draft.

Strategy: single LLM call.
- Input: full draft + duplicate report + unused curated URLs as candidates.
- LLM decides which items to drop/replace and with what candidate.
- We validate every proposed replacement URL against the allowed set.
- Any URL not in the curated artifact is rejected (fabrication prevention).
- After repair, #N numbers are re-assigned globally.
- Sections are kept even if their last item was removed (replacement is
  mandatory when possible); a section can be left with one placeholder item
  drawn from unused curated candidates.

Raises RepairerFailed if:
- LLM output is invalid JSON
- Schema validation fails
- No valid repair action could be applied (all proposed URLs invented)
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from agent.harness.budget import BudgetTracker
from agent.harness.trace import Tracer
from agent.llm import LLMMessage, LLMProvider, LLMResponse
from agent.schemas import (
    CuratedItemRecord,
    CuratedOutput,
    Draft,
    DraftItem,
    DraftSection,
    RepairAction,
    RepairReport,
    SemanticDuplicate,
    SemanticDuplicateReport,
)

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_SYSTEM = """\
你是一名 AI 技术情报编辑，负责修复日报草稿中的语义重复问题。

规则：
1. 只处理 severity=high 或 severity=medium 的重复对。
2. 优先保留"今日头条"section 中的条目；删除或替换重复对中排名靠后 section 的条目。
3. 被删除的条目，必须从下面提供的"候选替换条目"中选一个替换，以维持 section 不空。
   如果候选为空，则可以省略 replacement_url/replacement_title（表示直接删除）。
4. 绝对禁止编造 URL —— replacement_url 必须来自"候选替换条目"列表，不得修改。
5. 输出严格 JSON，格式如下：

{
  "actions": [
    {
      "section": "<要删除/替换条目所在 section heading>",
      "removed_title": "<要删除条目的完整 title>",
      "removed_url": "<要删除条目的 url>",
      "replacement_url": "<候选替换条目的 url，或 null>",
      "replacement_title": "<候选替换条目的 title，或 null>",
      "reason": "<一句话说明>"
    }
  ]
}

如果没有需要修复的重复（全部为 low），返回 {"actions": []}。
不要输出任何 JSON 之外的文字。
"""

_USER_TEMPLATE = """\
当前草稿条目（共 {item_count} 条）：
{items_json}

需要修复的重复对（仅 high/medium）：
{duplicates_json}

候选替换条目（来自 curated artifact，尚未出现在草稿中）：
{candidates_json}

请输出修复 actions JSON。
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class RepairerFailed(RuntimeError):
    pass


def _item_list(draft: Draft) -> List[Dict[str, str]]:
    result = []
    for sec in draft.sections:
        for item in sec.items:
            result.append(
                {
                    "section": sec.heading,
                    "title": item.title,
                    "summary": item.summary[:200],
                    "url": item.url,
                }
            )
    return result


def _blocking_dups(sem_report: SemanticDuplicateReport) -> List[SemanticDuplicate]:
    return [d for d in sem_report.duplicates if d.severity in ("high", "medium")]


def _draft_urls(draft: Draft) -> Set[str]:
    return {item.url for sec in draft.sections for item in sec.items}


def _unused_candidates(
    curated_records: List[CuratedItemRecord], draft: Draft
) -> List[Dict[str, str]]:
    used = _draft_urls(draft)
    return [
        {"url": r.source_url, "title": r.title, "source": r.source_name}
        for r in curated_records
        if r.source_url not in used
    ]


def _renumber_draft(draft: Draft) -> Draft:
    """Re-assign global #N prefixes to all item titles."""
    counter = 0
    new_sections = []
    for sec in draft.sections:
        new_items = []
        for item in sec.items:
            counter += 1
            # Replace or prepend the #N prefix.
            title = re.sub(r"^#\d+\s*", "", item.title).strip()
            new_items.append(
                DraftItem(
                    title=f"#{counter} {title}",
                    summary=item.summary,
                    url=item.url,
                    source=item.source,
                )
            )
        new_sections.append(DraftSection(heading=sec.heading, items=new_items))
    return Draft(
        date=draft.date,
        title=draft.title,
        overview=draft.overview,
        sections=new_sections,
    )


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_actions_json(raw: str) -> List[Dict[str, Any]]:
    text = _strip_think(raw)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise RepairerFailed(f"no JSON object in repairer output: {text[:200]}")
    payload = json.loads(text[start : end + 1])
    return payload.get("actions", [])


# --------------------------------------------------------------------------- #
# Core repair logic (pure, no LLM)
# --------------------------------------------------------------------------- #


def apply_repair_actions(
    draft: Draft,
    actions: List[RepairAction],
    allowed_urls: Set[str],
) -> Tuple[Draft, List[RepairAction]]:
    """Apply validated repair actions to the draft.

    Returns (repaired_draft, applied_actions).
    Skips actions whose replacement_url is not in allowed_urls (fabrication guard).
    """
    # Build lookup: url → (section_idx, item_idx)
    applied: List[RepairAction] = []

    new_sections = [
        DraftSection(heading=sec.heading, items=list(sec.items))
        for sec in draft.sections
    ]
    section_by_heading = {sec.heading: sec for sec in new_sections}

    for action in actions:
        # Fabrication guard: replacement must be from curated artifact.
        if action.replacement_url and action.replacement_url not in allowed_urls:
            continue  # skip invented URL silently; caller sees it's not applied

        target_sec = section_by_heading.get(action.section)
        if target_sec is None:
            continue

        # Find the item to remove (match by url, fall back to title substring).
        remove_idx = None
        for i, item in enumerate(target_sec.items):
            if item.url == action.removed_url:
                remove_idx = i
                break
        if remove_idx is None:
            for i, item in enumerate(target_sec.items):
                if action.removed_title.lower() in item.title.lower():
                    remove_idx = i
                    break
        if remove_idx is None:
            continue  # item already gone or title mismatch

        if action.replacement_url:
            # Replace in-place.
            target_sec.items[remove_idx] = DraftItem(
                title=action.replacement_title or action.removed_title,
                summary="（替补条目）",
                url=action.replacement_url,
                source="curated",
            )
        else:
            # Drop the item; keep section alive only if ≥1 item remains.
            if len(target_sec.items) > 1:
                target_sec.items.pop(remove_idx)
            # If this would empty the section and no replacement provided,
            # leave the item (don't create an empty section).

        applied.append(action)

    repaired = Draft(
        date=draft.date,
        title=draft.title,
        overview=draft.overview,
        sections=new_sections,
    )
    return _renumber_draft(repaired), applied


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def repair_draft(
    *,
    draft: Draft,
    sem_report: SemanticDuplicateReport,
    curated_records: List[CuratedItemRecord],
    provider: LLMProvider,
    date: str,
    run_id: str,
    tracer: Tracer,
    budget: BudgetTracker,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
) -> Tuple[Draft, RepairReport]:
    """Attempt one repair pass on the draft.

    Returns (repaired_draft, repair_report).
    Raises RepairerFailed only on LLM/parse errors — the caller decides
    whether to treat that as needs_human_review.
    """
    blocking = _blocking_dups(sem_report)
    if not blocking:
        report = RepairReport(
            date=date,
            run_id=run_id,
            attempted=False,
            succeeded=False,
            reason="no high/medium duplicates — repair skipped",
            pre_duplicate_count=len(sem_report.duplicates),
        )
        tracer.log("repair_skipped", reason=report.reason)
        return draft, report

    candidates = _unused_candidates(curated_records, draft)
    allowed_urls: Set[str] = {r.source_url for r in curated_records}

    items_json = json.dumps(_item_list(draft), ensure_ascii=False, indent=2)
    dups_json = json.dumps(
        [
            {
                "item_a_id": d.item_a_id,
                "item_b_id": d.item_b_id,
                "item_a_title": d.item_a_title,
                "item_b_title": d.item_b_title,
                "reason": d.reason,
                "severity": d.severity,
            }
            for d in blocking
        ],
        ensure_ascii=False,
        indent=2,
    )
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)

    user_content = _USER_TEMPLATE.format(
        item_count=sum(len(s.items) for s in draft.sections),
        items_json=items_json,
        duplicates_json=dups_json,
        candidates_json=candidates_json,
    )
    messages = [
        LLMMessage(role="system", content=_SYSTEM),
        LLMMessage(role="user", content=user_content),
    ]

    tracer.log("repair_started", duplicate_count=len(blocking), candidates=len(candidates))
    budget.check_can_call(stage="repair")

    t0 = time.time()
    try:
        resp: LLMResponse = provider.complete(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    except Exception as e:
        tracer.log_llm_call(
            provider=provider.name,
            model=provider.model,
            prompt=_SYSTEM + "\n" + user_content,
            output="",
            latency_ms=int((time.time() - t0) * 1000),
            status="error",
            error=str(e),
            stage="repair",
        )
        raise RepairerFailed(f"LLM call failed during repair: {e}") from e

    tracer.log_llm_call(
        provider=provider.name,
        model=provider.model,
        prompt=_SYSTEM + "\n" + user_content,
        output=resp.text,
        latency_ms=resp.latency_ms,
        status="ok",
        stage="repair",
    )
    budget.record(
        stage="repair",
        input_tokens=resp.input_tokens_est,
        output_tokens=resp.output_tokens_est,
    )
    tracer.log("repair_llm_call", latency_ms=resp.latency_ms, output_head=resp.text[:200])

    # Parse LLM response.
    try:
        raw_actions = _parse_actions_json(resp.text)
    except (json.JSONDecodeError, RepairerFailed) as e:
        tracer.log("repair_failed", reason=str(e))
        raise RepairerFailed(str(e)) from e

    # Validate each action against schema.
    actions: List[RepairAction] = []
    for raw in raw_actions:
        try:
            actions.append(RepairAction.model_validate(raw))
        except ValidationError:
            continue  # skip malformed action entries

    if not actions:
        # LLM returned no valid actions (empty or all malformed).
        report = RepairReport(
            date=date,
            run_id=run_id,
            attempted=True,
            succeeded=False,
            reason="LLM returned no valid repair actions",
            pre_duplicate_count=len(blocking),
        )
        tracer.log("repair_failed", reason=report.reason)
        return draft, report

    repaired, applied = apply_repair_actions(draft, actions, allowed_urls)

    if not applied:
        report = RepairReport(
            date=date,
            run_id=run_id,
            attempted=True,
            succeeded=False,
            reason="all proposed replacement URLs were outside the curated artifact (fabrication rejected)",
            actions=actions,
            pre_duplicate_count=len(blocking),
        )
        tracer.log("repair_failed", reason=report.reason)
        return draft, report

    # Validate repaired draft.
    try:
        repaired = Draft.model_validate(repaired.model_dump())
    except ValidationError as e:
        report = RepairReport(
            date=date,
            run_id=run_id,
            attempted=True,
            succeeded=False,
            reason=f"repaired draft failed schema validation: {e}",
            actions=applied,
            pre_duplicate_count=len(blocking),
        )
        tracer.log("repair_failed", reason=report.reason)
        return draft, report

    report = RepairReport(
        date=date,
        run_id=run_id,
        attempted=True,
        succeeded=True,
        reason=f"repaired {len(applied)} item(s)",
        actions=applied,
        pre_duplicate_count=len(blocking),
        draft_version="v2",
    )
    tracer.log("repair_succeeded", applied_count=len(applied))
    return repaired, report
