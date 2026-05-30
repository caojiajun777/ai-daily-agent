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
2. 优先保留"要闻"section 中的条目；删除或替换重复对中排名靠后 section 的条目。
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


def _item_id(item: DraftItem) -> str:
    m = re.match(r"^(#\d+)\b", item.title or "")
    return m.group(1) if m else ""


def _draft_refs(draft: Draft) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for sec in draft.sections:
        for item in sec.items:
            refs.append({
                "id": _item_id(item),
                "section": sec.heading,
                "item": item,
            })
    return refs


def _find_dup_ref(
    refs: List[Dict[str, Any]],
    item_id: str,
    item_title: str,
) -> Optional[Dict[str, Any]]:
    if item_id:
        for ref in refs:
            if ref["id"] == item_id:
                return ref
    needle = _strip_item_number(item_title).lower()
    for ref in refs:
        title = _strip_item_number(ref["item"].title).lower()
        if needle and (needle == title or needle in title or title in needle):
            return ref
    return None


def _strip_item_number(title: str) -> str:
    return re.sub(r"^#\d+\s*", "", title or "").strip()


def _protected_model_release_actions(
    blocking: List[SemanticDuplicate],
    draft: Draft,
) -> Tuple[List[Tuple[RepairAction, str]], List[SemanticDuplicate]]:
    """Build deterministic repairs that preserve official model releases.

    Returns (actions_with_keep_url, remaining_duplicates). The action removes
    the less authoritative duplicate; keep_url receives the removed URL as a
    related link before removal.
    """
    refs = _draft_refs(draft)
    protected: List[Tuple[RepairAction, str]] = []
    remaining: List[SemanticDuplicate] = []

    for dup in blocking:
        a = _find_dup_ref(refs, dup.item_a_id, dup.item_a_title)
        b = _find_dup_ref(refs, dup.item_b_id, dup.item_b_title)
        if not a or not b:
            remaining.append(dup)
            continue
        a_item: DraftItem = a["item"]
        b_item: DraftItem = b["item"]
        a_protected = _is_protected_official_model_release_item(a_item, a["section"])
        b_protected = _is_protected_official_model_release_item(b_item, b["section"])
        if a_protected == b_protected:
            remaining.append(dup)
            continue

        keep = a if a_protected else b
        drop = b if a_protected else a
        keep_item: DraftItem = keep["item"]
        drop_item: DraftItem = drop["item"]
        action = RepairAction(
            section=drop["section"],
            removed_title=drop_item.title,
            removed_url=drop_item.url,
            replacement_url=None,
            replacement_title=None,
            reason=(
                "保留官方模型发布，将同模型的平台接入或二次报道合并为相关链接。"
            ),
        )
        protected.append((action, keep_item.url))

    return protected, remaining


def _merge_related_links_for_kept_items(
    draft: Draft,
    protected_actions: List[Tuple[RepairAction, str]],
) -> Draft:
    if not protected_actions:
        return draft
    links_by_keep_url: Dict[str, List[str]] = {}
    for action, keep_url in protected_actions:
        if not keep_url or not action.removed_url:
            continue
        links_by_keep_url.setdefault(keep_url, []).append(action.removed_url)

    new_sections: List[DraftSection] = []
    for sec in draft.sections:
        new_items: List[DraftItem] = []
        for item in sec.items:
            extra = links_by_keep_url.get(item.url)
            if not extra:
                new_items.append(item)
                continue
            links = _dedupe_urls([*item.related_links, *extra])
            new_items.append(item.model_copy(update={"related_links": links[:4]}))
        new_sections.append(DraftSection(heading=sec.heading, items=new_items))
    return draft.model_copy(update={"sections": new_sections})


def _dedupe_urls(urls: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _is_protected_official_model_release_item(item: DraftItem, section: str) -> bool:
    text = f"{item.title} {item.summary} {' '.join(item.body_paragraphs)}".lower()
    if not _model_story_key(text):
        return False
    if not _has_model_release_signal(text, section, item):
        return False
    if not _is_model_provider_source(item):
        return False
    if _is_platform_access_story(text):
        return False
    return True


def _model_story_key(text: str) -> str:
    patterns = [
        r"\bclaude\s+(?:opus|sonnet|haiku)\s+\d+(?:\.\d+)*\b",
        r"\bclaude\s*\d+(?:\.\d+)*(?:\s*(?:opus|sonnet|haiku))?\b",
        r"\bgpt[-\s]?\d+(?:\.\d+)*(?:-[a-z0-9]+)?\b",
        r"\bgemini\s*\d+(?:\.\d+)*(?:\s*(?:pro|flash|ultra|nano))?\b",
        r"\bqwen\d+(?:\.\d+)*(?:-[a-z0-9]+)?\b",
        r"\bdeepseek[-\s]?[vr]?\d+(?:\.\d+)*(?:-[a-z0-9]+)?\b",
        r"\bminimax\s*m\d+(?:\.\d+)*(?:-[a-z0-9]+)?\b",
        r"\bstep\s*\d+(?:\.\d+)*(?:\s*flash)?\b",
        r"\bstepaudio\s*\d+(?:\.\d+)*(?:\s*realtime)?\b",
        r"\bglm[-\s]?\d+(?:\.\d+)*(?:-[a-z0-9]+)?\b",
        r"\bhunyuan[-\s]?\d+(?:\.\d+)*(?:-[a-z0-9]+)?\b",
        r"\bkimi\s*k?\d+(?:\.\d+)*(?:-[a-z0-9]+)?\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return re.sub(r"\s+", "", m.group(0).lower())
    return ""


def _has_model_release_signal(text: str, section: str, item: DraftItem) -> bool:
    if section == "模型发布" or item.item_type in {"model", "release"}:
        return True
    release_terms = (
        "release", "released", "launch", "launched", "introducing",
        "introduced", "announce", "announced", "latest", "new", "live",
        "flagship", "发布", "推出", "上线", "宣布", "开源", "旗舰", "最新",
    )
    capability_terms = (
        "model", "models", "模型", "benchmark", "benchmarks", "coding",
        "reasoning", "agent", "推理", "基准", "能力", "智能体", "编码",
    )
    return _contains_any(text, release_terms) and _contains_any(text, capability_terms)


def _is_platform_access_story(text: str) -> bool:
    access_terms = (
        "github copilot", "copilot", "openrouter", "vertex ai", "bedrock",
        "azure ai", "available for", "integrated into", "integration",
        "登陆", "接入", "集成至", "集成到",
    )
    return _contains_any(text, access_terms)


def _contains_any(text: str, terms: Tuple[str, ...]) -> bool:
    for term in terms:
        if term in text:
            return True
    return False


def _is_model_provider_source(item: DraftItem) -> bool:
    source = (item.source or "").lower()
    url = (item.url or "").lower()
    provider_sources = {
        "openai", "openai_news", "anthropic", "anthropic_news",
        "google_ai_blog", "google_deepmind_blog", "meta_ai_blog",
        "mistral", "mistral_ai", "huggingface_blog", "qwen",
        "x_qwen", "deepseek", "x_deepseek", "minimax", "stepfun",
        "x_stepfun", "zhipu", "moonshot",
    }
    provider_markers = (
        "openai.com/index/", "openai.com/news/", "anthropic.com/news/",
        "claude.com/blog/", "deepmind.google/", "blog.google/technology/ai/",
        "ai.meta.com/blog/", "mistral.ai/news/", "huggingface.co/blog/",
        "github.com/qwenlm/", "github.com/deepseek-ai/",
        "x.com/openai/", "x.com/anthropicai/", "x.com/googledeepmind/",
        "x.com/aiatmeta/", "x.com/mistralai/", "x.com/alibaba_qwen/",
        "x.com/deepseek_ai/", "x.com/tencenthunyuan/", "x.com/stepfun_ai/",
        "x.com/minimax_ai/", "x.com/chatglm/", "x.com/moonshot",
    )
    return source in provider_sources or any(marker in url for marker in provider_markers)


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
            new_items.append(item.model_copy(update={"title": f"#{counter} {title}"}))
        new_sections.append(DraftSection(heading=sec.heading, items=new_items))
    return draft.model_copy(update={"sections": new_sections, "overview_groups": []})


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
    replacement_records: Optional[Dict[str, CuratedItemRecord]] = None,
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
            rec = (replacement_records or {}).get(action.replacement_url)
            if rec:
                replacement = DraftItem(
                    title=action.replacement_title or rec.title,
                    summary=rec.title,
                    body_paragraphs=[rec.title],
                    url=rec.source_url,
                    source=rec.source_name,
                    related_links=[],
                    content_type=rec.content_type,
                    source_tier=rec.source_tier,
                    evidence_type=rec.evidence_type,
                    confidence=rec.confidence,
                    evidence_note="replacement from curated artifact",
                )
            else:
                replacement = DraftItem(
                    title=action.replacement_title or action.removed_title,
                    summary=action.replacement_title or "（替补条目）",
                    body_paragraphs=[action.replacement_title or "（替补条目）"],
                    url=action.replacement_url,
                    source="curated",
                )
            target_sec.items[remove_idx] = replacement
        else:
            # Drop the duplicate even if it leaves a section empty. A sparse
            # section is better than knowingly publishing duplicate stories.
            target_sec.items.pop(remove_idx)

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
    replacement_records = {r.source_url: r for r in curated_records}

    deterministic_actions, remaining_blocking = _protected_model_release_actions(
        blocking, draft
    )
    applied_deterministic: List[RepairAction] = []
    if deterministic_actions:
        draft_with_links = _merge_related_links_for_kept_items(draft, deterministic_actions)
        repaired, applied_deterministic = apply_repair_actions(
            draft_with_links,
            [action for action, _keep_url in deterministic_actions],
            allowed_urls,
            replacement_records,
        )
        if applied_deterministic:
            draft = repaired
            blocking = remaining_blocking
            candidates = _unused_candidates(curated_records, draft)
            tracer.log(
                "repair_protected_official_model_release",
                applied_count=len(applied_deterministic),
                remaining_duplicates=len(blocking),
            )

    if applied_deterministic and not blocking:
        report = RepairReport(
            date=date,
            run_id=run_id,
            attempted=True,
            succeeded=True,
            reason=(
                "protected official model release and merged duplicate access "
                "story as related link"
            ),
            actions=applied_deterministic,
            pre_duplicate_count=len(_blocking_dups(sem_report)),
            draft_version="v2",
        )
        tracer.log("repair_succeeded", applied_count=len(applied_deterministic))
        return draft, report

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

    repaired, applied = apply_repair_actions(
        draft, actions, allowed_urls, replacement_records
    )

    if not applied:
        report = RepairReport(
            date=date,
            run_id=run_id,
            attempted=True,
            succeeded=False,
            reason="all proposed replacement URLs were outside the curated artifact (fabrication rejected)",
            actions=[*applied_deterministic, *actions],
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
            actions=[*applied_deterministic, *applied],
            pre_duplicate_count=len(blocking),
        )
        tracer.log("repair_failed", reason=report.reason)
        return draft, report

    report = RepairReport(
        date=date,
        run_id=run_id,
        attempted=True,
        succeeded=True,
        reason=f"repaired {len(applied_deterministic) + len(applied)} item(s)",
        actions=[*applied_deterministic, *applied],
        pre_duplicate_count=len(_blocking_dups(sem_report)),
        draft_version="v2",
    )
    tracer.log("repair_succeeded", applied_count=len(applied_deterministic) + len(applied))
    return repaired, report
