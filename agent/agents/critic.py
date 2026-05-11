"""Critic agent.

Two-layer review:

  1. Deterministic checks (fast, cheap, trustworthy): URL consistency between
     draft and curated inputs, min section/item counts, forbidden phrases. We
     never rely on the LLM for checks a plain function can do.

  2. LLM critique (optional, off by default in MVP to keep the run cheap). The
     hook is implemented and tested so the next phase can flip it on.

When the deterministic layer finds any violation the critic returns
``verdict="reject"`` with reasons. The orchestrator then marks the stage
``needs_human_review`` instead of ``failed``: the draft exists on disk, a
human can edit it.
"""

from __future__ import annotations

import json
import time
from typing import Iterable, List, Optional

from agent.harness.budget import BudgetTracker
from agent.harness.trace import Tracer
from agent.llm import LLMMessage, LLMProvider
from agent.schemas import CritiqueResult, CuratedItem, Draft


def deterministic_critique(
    draft: Draft,
    curated: List[CuratedItem],
    *,
    min_section_count: int = 3,
    forbid_phrases: Optional[Iterable[str]] = None,
) -> CritiqueResult:
    reasons: List[str] = []
    forbid_phrases = list(forbid_phrases or [])
    allowed_urls = {c.url for c in curated}

    if len(draft.sections) < min_section_count:
        reasons.append(
            f"sections fewer than {min_section_count} (got {len(draft.sections)})"
        )

    total_items = sum(len(s.items) for s in draft.sections)
    if total_items < min_section_count:
        reasons.append(f"items fewer than {min_section_count} (got {total_items})")

    seen_titles: set = set()
    for section in draft.sections:
        for item in section.items:
            if item.url and allowed_urls and item.url not in allowed_urls:
                reasons.append(f"hallucinated url: {item.url}")
            norm = item.title.strip().lower()
            if norm in seen_titles:
                reasons.append(f"duplicate item title: {item.title}")
            seen_titles.add(norm)
            combined = f"{item.title} {item.summary}"
            for bad in forbid_phrases:
                if bad and bad in combined:
                    reasons.append(f"forbidden phrase present: {bad}")

    if reasons:
        return CritiqueResult(verdict="reject", reasons=reasons, score=0)
    score = max(0, 100 - len(reasons) * 10)
    return CritiqueResult(verdict="pass", reasons=[], score=score)


def llm_critique(
    *,
    provider: LLMProvider,
    draft: Draft,
    curated: List[CuratedItem],
    system_prompt: str,
    user_template: str,
    tracer: Tracer,
    budget: BudgetTracker,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
) -> CritiqueResult:
    items_json = json.dumps([i.model_dump() for i in curated], ensure_ascii=False)
    draft_json = draft.model_dump_json()
    user = user_template.format(items_json=items_json, draft_json=draft_json)
    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user),
    ]
    budget.check_can_call(stage="critique")

    t0 = time.time()
    try:
        resp = provider.complete(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        tracer.log_llm_call(
            provider=provider.name,
            model=provider.model,
            prompt=system_prompt + "\n" + user,
            output="",
            latency_ms=int((time.time() - t0) * 1000),
            status="error",
            error=str(e),
            stage="critique",
        )
        raise

    tracer.log_llm_call(
        provider=provider.name,
        model=provider.model,
        prompt=system_prompt + "\n" + user,
        output=resp.text,
        latency_ms=resp.latency_ms,
        status="ok",
        stage="critique",
    )
    budget.record(
        stage="critique",
        input_tokens=resp.input_tokens_est,
        output_tokens=resp.output_tokens_est,
    )

    try:
        payload = json.loads(resp.text)
        return CritiqueResult.model_validate(payload)
    except Exception:
        # Don't let a noisy critic break the run; treat as soft pass but log.
        tracer.log(
            "critic_parse_failed",
            stage="critique",
            output_head=resp.text[:200],
        )
        return CritiqueResult(verdict="pass", reasons=["critic_parse_failed"], score=50)
