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

    nonempty_section_count = sum(1 for section in draft.sections if section.items)
    if nonempty_section_count < min_section_count:
        reasons.append(
            f"non-empty sections fewer than {min_section_count} (got {nonempty_section_count})"
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

    # ── Tier-aware quality checks ──────────────────────────────────
    tier_pulse_sections = {"硅谷脉搏", "社区脉搏", "Silicon Valley / Community Pulse",
                           "业界风向", "产业风向", "资本动向",
                           "Industry Watch", "Capital Moves",
                           "Industry Strategy & Company Moves",
                           "行业动态", "前瞻与传闻"}
    investment_keywords = ["买入", "卖出", "必涨", "必跌", "稳赚", "抄底", "逃顶",
                           "买入推荐", "卖出推荐", "目标价", "评级上调", "评级下调",
                           "buy", "sell", "long", "short", "overweight", "underweight"]

    for section in draft.sections:
        heading = section.heading or ""
        for item in section.items:
            ct = getattr(item, "content_type", "")
            tier = getattr(item, "source_tier", "")
            ev = getattr(item, "evidence_type", "")
            conf = getattr(item, "confidence", "")

            # Tier 3 items should not appear in non-Pulse sections (only if tier set).
            if tier and "tier_3" in tier and heading not in tier_pulse_sections:
                if heading not in ("产业风向", "资本动向", "行业动态", "前瞻与传闻",
                                   "Industry Watch", "Capital Moves"):
                    reasons.append(
                        f"tier3_major_claim: {item.title[:60]} has tier={tier} "
                        f"but section={heading} is not a Pulse section"
                    )

            # Missing source_tier (only flag if content_type is explicitly set).
            if not tier and ct and ct != "tech_media":
                reasons.append(f"missing_source_tier: {item.title[:60]} ct={ct}")

            # Insider media must use reported language — check summary for
            # official-sounding claims.
            if ct == "insider_media" and conf == "high":
                reasons.append(
                    f"confidence_too_high_for_source_tier: {item.title[:60]} "
                    f"ct={ct} tier={tier} conf={conf}"
                )

            # Tier 2/Tier 3 with high confidence (only if tier is set).
            if tier and ("tier_2" in tier or "tier_3" in tier) and conf == "high":
                reasons.append(
                    f"confidence_too_high_for_source_tier: {item.title[:60]} "
                    f"tier={tier} conf={conf}"
                )

            # Market items must not contain investment advice.
            if ct in ("market_commentary", "vc_signal") or ev in ("market_commentary",):
                for kw in investment_keywords:
                    if kw in item.summary or kw in item.title:
                        reasons.append(
                            f"market_investment_advice: {item.title[:60]} "
                            f"contains '{kw}'"
                        )
                        break

            # Pricing items require official pricing/docs source.
            price_ev_types = {"pricing_page", "china_model_pricing", "china_model_docs",
                              "official_docs", "official_release"}
            if heading and "price" in heading.lower() and heading and "cost" in heading.lower():
                if ev not in price_ev_types and ct not in ("pricing_page", "china_model_pricing"):
                    reasons.append(
                        f"pricing_without_official_source: {item.title[:60]} "
                        f"ev={ev} ct={ct}"
                    )

    quality_flags = [r for r in reasons if any(
        kw in r for kw in ["tier3_major_claim", "missing_source_tier",
                           "confidence_too_high", "market_investment_advice",
                           "pricing_without_official_source"])
    ]

    if reasons:
        return CritiqueResult(verdict="reject", reasons=reasons, score=0,
                             quality_flags=quality_flags)
    score = max(0, 100 - len(reasons) * 10)
    return CritiqueResult(verdict="pass", reasons=[], score=score,
                         quality_flags=[])


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
