"""Semantic duplicate critic.

Sends all draft items to the LLM in a single call and asks it to identify
pairs that cover the same real-world event (even if they differ in title,
section or URL). One call, no pairwise loops, no embeddings.

Severity:
  high   — unmistakably the same event (e.g. same product launch, same paper)
  medium — highly overlapping event / likely redundant from reader perspective
  low    — thematically related but not a genuine duplicate

Gate policy (enforced in the publish gate, not here):
  high   → block
  medium → block
  low    → warning only
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from agent.harness.budget import BudgetTracker
from agent.harness.trace import Tracer, estimate_tokens
from agent.llm import LLMMessage, LLMProvider, LLMResponse
from agent.schemas import Draft, SemanticDuplicate, SemanticDuplicateReport

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_SYSTEM = """\
你是一名严谨的 AI 技术情报编辑，负责检测日报草稿中的语义重复问题。

任务：阅读下方所有条目，找出报道同一真实事件的条目对（即使标题、章节或 URL 不同）。

输出严格 JSON，格式如下：
{
  "duplicates": [
    {
      "item_a_id": "<条目 id，格式为 #N>",
      "item_b_id": "<条目 id，格式为 #N>",
      "item_a_title": "<原始标题>",
      "item_b_title": "<原始标题>",
      "reason": "<一句话说明为何认为是同一事件>",
      "severity": "high|medium|low"
    }
  ]
}

判断标准：
- high：明确同一事件（同一产品/论文/发布/事故）
- medium：高度重叠，读者会感到重复
- low：主题相关，但角度不同，仅作提示

如果没有发现重复，返回 {"duplicates": []}。
不要输出任何 JSON 之外的文字。
"""

_USER_TEMPLATE = """\
以下是今日早报的所有条目（共 {count} 条），请检测语义重复：

{items_json}
"""


# --------------------------------------------------------------------------- #
# Item descriptor sent to LLM
# --------------------------------------------------------------------------- #


def _item_descriptors(draft: Draft) -> List[Dict[str, str]]:
    """Flatten draft items into a compact list for the LLM."""
    result = []
    for section in draft.sections:
        for item in section.items:
            result.append(
                {
                    "id": item.title.split()[0] if item.title.startswith("#") else item.title[:6],
                    "section": section.heading,
                    "title": item.title,
                    "summary": item.summary[:300],
                    "url": item.url,
                }
            )
    return result


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class SemanticDuplicateCriticFailed(RuntimeError):
    """Raised when the LLM returns unparseable output."""


def run_semantic_duplicate_critic(
    *,
    draft: Draft,
    provider: LLMProvider,
    date: str,
    run_id: str,
    tracer: Tracer,
    budget: BudgetTracker,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
) -> SemanticDuplicateReport:
    """Call LLM once to detect semantic duplicates across all draft items.

    Raises SemanticDuplicateCriticFailed if the LLM output cannot be parsed.
    Never silently returns ok=True when the response is invalid.
    """
    descriptors = _item_descriptors(draft)
    checked = len(descriptors)

    user_content = _USER_TEMPLATE.format(
        count=checked,
        items_json=json.dumps(descriptors, ensure_ascii=False, indent=2),
    )
    messages = [
        LLMMessage(role="system", content=_SYSTEM),
        LLMMessage(role="user", content=user_content),
    ]

    budget.check_can_call(stage="semantic_dup")

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
            stage="semantic_dup",
        )
        raise

    tracer.log_llm_call(
        provider=provider.name,
        model=provider.model,
        prompt=_SYSTEM + "\n" + user_content,
        output=resp.text,
        latency_ms=resp.latency_ms,
        status="ok",
        stage="semantic_dup",
    )
    budget.record(
        stage="semantic_dup",
        input_tokens=resp.input_tokens_est,
        output_tokens=resp.output_tokens_est,
    )

    raw = resp.text.strip()
    # Strip optional markdown fence.
    if raw.startswith("```"):
        import re as _re
        raw = _re.sub(r"^```(?:json)?\s*", "", raw)
        raw = _re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    # Strip <think>...</think> blocks (reasoning models).
    import re as _re2
    raw = _re2.sub(r"<think>.*?</think>", "", raw, flags=_re2.DOTALL).strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        tracer.log(
            "semantic_dup_parse_failed",
            error=str(e),
            output_head=resp.text[:300],
        )
        raise SemanticDuplicateCriticFailed(
            f"LLM returned invalid JSON for semantic duplicate check: {e}\n"
            f"Output head: {resp.text[:200]}"
        ) from e

    try:
        raw_dups = payload.get("duplicates", [])
        duplicates = [SemanticDuplicate.model_validate(d) for d in raw_dups]
    except Exception as e:
        tracer.log(
            "semantic_dup_schema_failed",
            error=str(e),
            payload_head=str(payload)[:300],
        )
        raise SemanticDuplicateCriticFailed(
            f"LLM output failed schema validation: {e}"
        ) from e

    has_blocking = any(d.severity in ("high", "medium") for d in duplicates)

    report = SemanticDuplicateReport(
        date=date,
        run_id=run_id,
        duplicates=duplicates,
        ok=not has_blocking,
        checked_item_count=checked,
        provider=provider.name,
    )
    tracer.log(
        "semantic_dup_checked",
        date=date,
        ok=report.ok,
        duplicate_count=len(duplicates),
        checked=checked,
    )
    return report
