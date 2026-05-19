"""Writer agent — calls an LLM to produce a structured AI daily draft."""

from __future__ import annotations

import json
import re as _re_mod
import time
from typing import Dict, List

from agent.harness.budget import BudgetTracker
from agent.harness.trace import Tracer
from agent.llm.base import LLMMessage, LLMProvider
from agent.schemas import CuratedItem, Draft, DraftItem
from pydantic import ValidationError


class WriterFailed(Exception):
    """Raised when the LLM output cannot be parsed into a valid Draft."""


def _extract_json(text: str) -> str:
    """Pull the first top-level JSON object out of a noisy completion."""
    t = text.strip()
    t = _re_mod.sub(r"<think>.*?</think>", "", t, flags=_re_mod.DOTALL).strip()
    fence = _re_mod.match(r"^```(?:json)?\s*(.*?)\s*```$", t, _re_mod.DOTALL)
    if fence:
        return fence.group(1)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise WriterFailed("no JSON object found in model output")
    return t[start : end + 1]


def _repair_json(text: str) -> str:
    """Attempt to repair common LLM JSON mistakes."""
    t = text.strip()
    t = _re_mod.sub(r",(\s*[}\]])", r"\1", t)
    t = _re_mod.sub(r"\}(\s*)\n(\s*)\{", r"},\n\2{", t)
    t = _re_mod.sub(r'"\s*\n\s*"', r'",\n"', t)
    return t


def write_draft(
    *,
    provider: LLMProvider,
    items: List[CuratedItem],
    date: str,
    system_prompt: str,
    user_template: str,
    max_items: int,
    tracer: Tracer,
    budget: BudgetTracker,
    temperature: float = 0.3,
    max_output_tokens: int = 2048,
) -> Draft:
    items_json = json.dumps([i.model_dump() for i in items], ensure_ascii=False)
    user = user_template.format(date=date, max_items=max_items, items_json=items_json)
    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user),
    ]
    budget.check_can_call(stage="write")

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
            stage="write",
        )
        raise

    tracer.log_llm_call(
        provider=provider.name,
        model=provider.model,
        prompt=system_prompt + "\n" + user,
        output=resp.text,
        latency_ms=resp.latency_ms,
        status="ok",
        stage="write",
    )
    budget.record(
        stage="write",
        input_tokens=resp.input_tokens_est,
        output_tokens=resp.output_tokens_est,
    )

    raw = _extract_json(resp.text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e1:
        # Retry once with temperature 0.0 for deterministic JSON.
        try:
            retry_resp = provider.complete(
                messages, temperature=0.0, max_output_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            )
            raw2 = _extract_json(retry_resp.text)
            payload = json.loads(raw2)
            budget.record(stage="write",
                          input_tokens=retry_resp.input_tokens_est,
                          output_tokens=retry_resp.output_tokens_est)
        except (json.JSONDecodeError, Exception) as e2:
            raise WriterFailed(
                f"writer output is not valid JSON after retry: {e2}"
            ) from e1
    try:
        draft = Draft.model_validate(payload)
    except ValidationError as e:
        raise WriterFailed(f"writer output violates schema: {e}") from e
    return draft


# ── Single-tier Markdown rendering ─────────────────────────────────────


def render_markdown(draft: Draft) -> str:
    """Render a Draft into single-tier Markdown.

    Each item appears once with title, image, summary, highlights, and links.
    No duplication — one item, one block.
    """
    lines: List[str] = []

    if draft.cover_image:
        lines.append(f"![]({draft.cover_image})")
        lines.append("")

    lines.append(f"# {draft.title}")
    lines.append("")
    if draft.overview:
        lines.append(f"> {draft.overview}")
        lines.append("")

    for section in draft.sections:
        heading_en = _section_subtitle(section.heading)
        lines.append(f"## {section.heading}")
        if heading_en:
            lines.append(f"*{heading_en}*")
        lines.append("")

        for item in section.items:
            lines.append(f"### {item.title}")
            lines.append("")

            if item.image_url:
                lines.append(f"![]({item.image_url})")
                lines.append("")

            src = _source_label(item.source)
            if item.url:
                lines.append(f"> 来源：{src} — [原文链接]({item.url})")
            else:
                lines.append(f"> 来源：{src}")
            lines.append("")

            lines.append(item.summary)
            lines.append("")

            if item.highlights:
                for h in item.highlights:
                    lines.append(f"- {h}")
                lines.append("")

            lines.append("---")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


_SECTION_SUBTITLES: Dict[str, str] = {
    "今日头条": "Headlines",
    "模型前沿": "Model Frontier",
    "工具与开源": "Tools & Open Source",
    "论文精选": "Paper Picks",
    "产品落地": "Launchpad",
    "业界风向": "Industry Watch",
}


def _section_subtitle(heading: str) -> str:
    return _SECTION_SUBTITLES.get(heading, "")


_SOURCE_LABELS: Dict[str, str] = {}


def _source_label(source_id: str) -> str:
    """Human-readable source label."""
    if source_id in _SOURCE_LABELS:
        return _SOURCE_LABELS[source_id]
    label = source_id
    for prefix in ("x_", "aihot:", "diffused_", "scout_"):
        if source_id.startswith(prefix):
            label = source_id[len(prefix):]
            break
    name_map = {
        "openai_news": "OpenAI", "anthropic_news": "Anthropic",
        "huggingface_blog": "Hugging Face", "google_ai_blog": "Google AI Blog",
        "google_deepmind_blog": "Google DeepMind", "meta_ai_blog": "Meta AI",
        "microsoft_ai_blog": "Microsoft AI",
        "mit_tech_review_ai": "MIT Technology Review",
        "venturebeat_ai": "VentureBeat", "ars_technica_ai": "Ars Technica",
        "wired_ai": "WIRED", "the_decoder": "The Decoder",
        "ithome": "IT之家", "qbitai": "量子位", "jiqizhixin": "机器之心",
        "the_batch": "The Batch", "import_ai": "Import AI",
        "paperswithcode_blog": "Papers With Code",
        "aihot_daily": "AI HOT 日报", "curated": "GitHub Releases",
    }
    label = name_map.get(source_id, label)
    _SOURCE_LABELS[source_id] = label
    return label
