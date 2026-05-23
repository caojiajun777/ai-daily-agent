"""Writer agent — calls an LLM to produce a structured AI daily draft."""

from __future__ import annotations

import json
import re as _re_mod
import time
from typing import Dict, List

from agent.harness.budget import BudgetTracker
from agent.harness.trace import Tracer
from agent.llm.base import LLMMessage, LLMProvider
from agent.schemas import (
    CuratedItem,
    Draft,
    DraftItem,
    DraftSection,
    OverviewEntry,
    OverviewGroup,
)
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
    t = _re_mod.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)
    return t


def _loads_with_repairs(raw: str) -> dict:
    errors: List[Exception] = []
    candidates = [raw, _repair_json(raw)]
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
            raise ValueError("writer JSON root is not an object")
        except Exception as e:
            errors.append(e)
    raise errors[-1]


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
    allow_fallback: bool = False,
    complete_with_items: bool = False,
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
    json_error: Exception | None = None
    try:
        payload = _loads_with_repairs(raw)
    except Exception as e1:
        json_error = e1
        # Retry once with a stricter compact instruction. Long rich JSON is
        # the common failure mode here; the compact retry usually avoids
        # comma drift or truncation without changing the selected sources.
        retry_user = (
            user
            + "\n\n上一次输出不是合法 JSON。请重新输出更紧凑的合法 JSON："
            + "每条 item 只保留 title、one_liner、summary、body_paragraphs(2段)、"
            + "url、source、highlights(2条)、related_links、content_type、"
            + "source_tier、evidence_type、confidence、item_type、rumor_level、"
            + "evidence_note。不要输出任何解释。"
        )
        retry_messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=retry_user),
        ]
        try:
            budget.check_can_call(stage="write")
            t_retry = time.time()
            retry_resp = provider.complete(
                retry_messages, temperature=0.0,
                max_output_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            )
            tracer.log_llm_call(
                provider=provider.name,
                model=provider.model,
                prompt=system_prompt + "\n" + retry_user,
                output=retry_resp.text,
                latency_ms=retry_resp.latency_ms,
                status="ok",
                stage="write",
            )
            raw2 = _extract_json(retry_resp.text)
            payload = _loads_with_repairs(raw2)
            budget.record(stage="write",
                          input_tokens=retry_resp.input_tokens_est,
                          output_tokens=retry_resp.output_tokens_est)
        except Exception as e2:
            json_error = e2
            if allow_fallback and raw.lstrip().startswith("{"):
                tracer.log(
                    "writer_deterministic_fallback",
                    reason=f"invalid_json_after_retry: {e2}",
                )
                return _fallback_draft_from_items(
                    items=items, date=date, max_items=max_items,
                )
            raise WriterFailed(
                f"writer output is not valid JSON after retry: {e2}"
            ) from e1
    try:
        draft = Draft.model_validate(payload)
    except ValidationError as e:
        if allow_fallback:
            tracer.log(
                "writer_deterministic_fallback",
                reason=f"schema_validation_failed: {e}",
            )
            return _fallback_draft_from_items(
                items=items, date=date, max_items=max_items,
            )
        raise WriterFailed(f"writer output violates schema: {e}") from e
    if complete_with_items:
        draft = _complete_draft_with_items(
            draft=draft,
            items=items,
            date=date,
            max_items=max_items,
            tracer=tracer,
        )
    return draft


_SECTION_ORDER = [
    "要闻", "模型发布", "开发生态", "技术与洞察",
    "产品应用", "行业动态", "前瞻与传闻",
]

_SECTION_ALIASES = {
    "今日头条": "要闻",
    "模型前沿": "模型发布",
    "工具与开源": "开发生态",
    "论文精选": "技术与洞察",
    "产品落地": "产品应用",
    "资本动向": "行业动态",
    "产业风向": "行业动态",
    "业界风向": "行业动态",
}


def _fallback_draft_from_items(
    *,
    items: List[CuratedItem],
    date: str,
    max_items: int,
) -> Draft:
    buckets: Dict[str, List[DraftItem]] = {name: [] for name in _SECTION_ORDER}
    selected = items[:max_items]

    for seq, item in enumerate(selected, start=1):
        section = _section_for_curated(item)
        buckets[section].append(_draft_item_from_curated(item, seq=seq))

    sections = [
        DraftSection(heading=heading, items=buckets[heading])
        for heading in _SECTION_ORDER
    ]
    overview_groups = [
        OverviewGroup(
            heading=section.heading,
            items=[
                OverviewEntry(
                    title=_strip_item_number(item.title),
                    url=item.url,
                    item_id=_item_id(item.title, idx + 1),
                    source=item.source,
                )
                for idx, item in enumerate(section.items)
            ],
        )
        for section in sections
        if section.items
    ]
    headline = selected[0].title if selected else "今日 AI 重要动态"
    overview = (
        f"本期精选 {len(selected)} 条 AI 动态，头条关注"
        f"{_strip_item_number(headline)}。"
    )
    return Draft(
        date=date,
        title=f"AI 日报 | {date}",
        overview=overview,
        overview_groups=overview_groups,
        sections=sections,
    )


def _complete_draft_with_items(
    *,
    draft: Draft,
    items: List[CuratedItem],
    date: str,
    max_items: int,
    tracer: Tracer,
) -> Draft:
    """Normalize a valid LLM draft and append any omitted curated items.

    LLMs sometimes return a schema-valid draft with fewer items than the
    curated set, or with one of the fixed sections left empty. The curated
    items are the source of truth for what should be published; this pass keeps
    the model's prose where present and fills gaps deterministically.
    """
    buckets: Dict[str, List[DraftItem]] = {name: [] for name in _SECTION_ORDER}
    used_urls: set[str] = set()
    used_titles: set[str] = set()
    kept = 0

    for section in draft.sections:
        heading = _normalize_section_heading(section.heading)
        for item in section.items:
            if kept >= max_items:
                break
            url_key = (item.url or "").strip()
            title_key = _title_key(item.title)
            if (url_key and url_key in used_urls) or (title_key and title_key in used_titles):
                continue
            buckets[heading].append(item)
            if url_key:
                used_urls.add(url_key)
            if title_key:
                used_titles.add(title_key)
            kept += 1

    def append_curated(curated: CuratedItem) -> bool:
        nonlocal kept
        if kept >= max_items:
            return False
        url_key = (curated.url or "").strip()
        title_key = _title_key(curated.title)
        if (url_key and url_key in used_urls) or (title_key and title_key in used_titles):
            return False
        section = _section_for_curated(curated)
        buckets[section].append(_draft_item_from_curated(curated, seq=kept + 1))
        if url_key:
            used_urls.add(url_key)
        if title_key:
            used_titles.add(title_key)
        kept += 1
        return True

    added = 0
    # First make any section fillable from curated candidates non-empty.
    for heading in _SECTION_ORDER:
        if buckets[heading]:
            continue
        for curated in items:
            if _section_for_curated(curated) == heading and append_curated(curated):
                added += 1
                break

    # Then include the rest of the curated list up to the configured ceiling.
    for curated in items:
        if kept >= max_items:
            break
        if append_curated(curated):
            added += 1

    sections = _renumber_sections(buckets)
    overview_groups = _overview_groups_from_sections(sections)
    item_count = sum(len(s.items) for s in sections)
    if added:
        tracer.log("writer_completed_from_curated", added=added, item_count=item_count)

    return draft.model_copy(update={
        "date": date,
        "sections": sections,
        "overview_groups": overview_groups,
    })


def _draft_item_from_curated(item: CuratedItem, *, seq: int) -> DraftItem:
    title = _strip_item_number(item.title)
    one_liner = item.why_it_matters or item.writing_angle or _first_sentence(item.summary)
    paragraphs = _fallback_paragraphs(item)
    highlights = _fallback_highlights(item)
    return DraftItem(
        title=f"#{seq} {title}",
        one_liner=one_liner,
        summary=item.summary or one_liner or title,
        body_paragraphs=paragraphs,
        url=item.url,
        source=item.source,
        highlights=highlights,
        related_links=item.supporting_urls[:4],
        content_type=item.content_type,
        source_tier=item.source_tier,
        evidence_type=item.evidence_type,
        confidence=item.confidence,
        item_type=_item_type_for_curated(item),
        rumor_level=_rumor_level_for_curated(item),
        evidence_note=_evidence_note_for_curated(item),
    )


def _renumber_sections(buckets: Dict[str, List[DraftItem]]) -> List[DraftSection]:
    seq = 1
    sections: List[DraftSection] = []
    for heading in _SECTION_ORDER:
        renumbered: List[DraftItem] = []
        for item in buckets.get(heading, []):
            title = _strip_item_number(item.title)
            renumbered.append(item.model_copy(update={"title": f"#{seq} {title}"}))
            seq += 1
        sections.append(DraftSection(heading=heading, items=renumbered))
    return sections


def _overview_groups_from_sections(sections: List[DraftSection]) -> List[OverviewGroup]:
    groups: Dict[str, List[OverviewEntry]] = {}
    for section in sections:
        if not section.items:
            continue
        group_name = _normalize_section_heading(section.heading)
        if group_name not in groups:
            groups[group_name] = []
        for idx, item in enumerate(section.items):
            groups[group_name].append(OverviewEntry(
                title=_strip_item_number(item.title),
                url=item.url,
                item_id=_item_id(item.title, idx + 1),
                source=item.source,
            ))
    result = []
    for heading in _SECTION_ORDER:
        if heading in groups and groups[heading]:
            result.append(OverviewGroup(heading=heading, items=groups[heading]))
    for heading, items in groups.items():
        if heading not in _SECTION_ORDER:
            result.append(OverviewGroup(heading=heading, items=items))
    return result


def _title_key(title: str) -> str:
    return _re_mod.sub(r"\W+", "", _strip_item_number(title or "").lower())


def _normalize_section_heading(section: str) -> str:
    mapped = _SECTION_ALIASES.get(section or "", section or "")
    return mapped if mapped in _SECTION_ORDER else "行业动态"


def _section_for_curated(item: CuratedItem) -> str:
    if item.section:
        normalized = _normalize_section_heading(item.section)
        if normalized in _SECTION_ORDER:
            return normalized
    text = f"{item.section_hint} {item.content_type} {item.evidence_type} {item.title}".lower()
    if "paper" in text or "arxiv" in item.url or "huggingface.co/papers" in item.url:
        return "技术与洞察"
    if any(k in text for k in ("rumor", "leak", "testing", "爆料", "传闻", "测试", "尚未确认")):
        return "前瞻与传闻"
    if any(k in text for k in ("pricing", "github", "sdk", "api", "tool", "开源", "changelog")):
        return "开发生态"
    if any(k in text for k in ("funding", "earnings", "revenue", "ipo", "acquisition", "融资", "财报", "收购")):
        return "行业动态"
    if any(k in text for k in ("product", "launch", "feature", "产品", "上线")):
        return "产品应用"
    if any(k in text for k in ("model", "benchmark", "模型", "推理")):
        return "模型发布"
    return "行业动态"


def _fallback_paragraphs(item: CuratedItem) -> List[str]:
    summary = item.summary.strip() or item.title
    paragraphs = _summary_to_paragraphs(summary)
    # If summary is too thin (title repeated), generate from metadata.
    if len(paragraphs) <= 1 and not item.why_it_matters:
        parts = []
        title = _strip_item_number(item.title)
        if item.source_tier and "tier_0" in item.source_tier:
            parts.append(f"官方发布：{title}。")
        elif item.source_tier and "tier_1" in item.source_tier:
            parts.append(f"据{_source_label(item.source)}报道，{title}。")
        else:
            parts.append(f"报道称{title}。")
        if item.evidence_type == "paper" or "arxiv.org" in item.url:
            parts.append("这条更适合当作技术趋势信号看，关键是方法、数据和可复现线索。")
        elif item.content_type == "insider_media":
            parts.append(f"该消息来自{_source_label(item.source)}，建议查阅原文获取完整信息。")
        if parts:
            paragraphs.extend(parts)
    if item.why_it_matters:
        paragraphs.append(item.why_it_matters)
    if item.writing_angle and item.writing_angle not in paragraphs[-1:]:
        paragraphs.append(item.writing_angle)
    # Deduplicate: avoid exact repeats of title in paragraphs.
    title_clean = _strip_item_number(item.title).strip()
    paragraphs = [p for p in paragraphs if p.strip() != title_clean]
    return paragraphs[:3]


def _fallback_highlights(item: CuratedItem) -> List[str]:
    out = []
    if item.why_it_matters:
        out.append(item.why_it_matters[:40])
    # Use concrete metadata instead of generic labels.
    if item.evidence_type and item.evidence_type not in ("media_report", "media_aggregator"):
        et_label = {"official_release": "官方发布", "paper": "学术论文", "insider_report": "内部报道",
                     "pricing_page": "定价信息", "github_release": "GitHub 发布",
                     "benchmark_tracker": "基准测试", "newsletter": "行业通讯"}
        out.append(et_label.get(item.evidence_type, item.evidence_type))
    if item.content_type == "paper" or (item.url and "arxiv" in item.url):
        out.append("arXiv 论文")
    if len(out) < 2:
        out.append(_first_sentence(item.summary or item.title)[:40])
    return out[:4]


def _item_type_for_curated(item: CuratedItem) -> str:
    text = f"{item.content_type} {item.evidence_type} {item.title}".lower()
    if "paper" in text or "arxiv" in item.url:
        return "paper"
    if "pricing" in text:
        return "pricing"
    if "github" in text or "open source" in text or "开源" in text:
        return "tool"
    if any(k in text for k in ("funding", "earnings", "revenue", "ipo", "融资", "财报")):
        return "capital"
    if "model" in text or "模型" in text:
        return "model"
    return "news"


def _rumor_level_for_curated(item: CuratedItem) -> str:
    if _section_for_curated(item) == "前瞻与传闻":
        return "rumor" if item.confidence == "low" else "reported"
    if item.confidence == "high" or "tier_0" in item.source_tier:
        return "confirmed"
    if "insider" in item.content_type or "report" in item.evidence_type:
        return "reported"
    if item.confidence == "low":
        return "rumor"
    return "reported"


def _evidence_note_for_curated(item: CuratedItem) -> str:
    if item.evidence_type in ("official_release", "official_docs"):
        return "官方发布或文档可核验"
    if item.evidence_type == "pricing_page":
        return "官方定价页或价格快照"
    if item.evidence_type in ("paper", "research_paper") or "arxiv.org" in item.url:
        return "论文或技术报告"
    if item.confidence == "low" or _section_for_curated(item) == "前瞻与传闻":
        return "未完全确认，按传闻/测试线索处理"
    if "tier_1" in item.source_tier:
        return "可信媒体或高信号来源报道"
    if item.reliability:
        return f"来源可靠性：{item.reliability}"
    return ""


# ── Juya-style Markdown rendering ─────────────────────────────────────


def render_markdown(draft: Draft) -> str:
    """Render a Draft into a readable daily product.

    Layout:
      1. title + short editor overview
      2. grouped overview index
      3. one flat, linkable detail block per item
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

    flat_items = _flatten_items(draft)
    overview_groups = _normalize_overview_groups(
        draft.overview_groups or _auto_overview_groups(draft)
    )
    if overview_groups:
        lines.append("## 概览")
        lines.append("")
        for group in overview_groups:
            if not group.items:
                continue
            lines.append(f"### {group.heading}")
            lines.append("")
            for entry in group.items:
                title = _strip_item_number(entry.title)
                item_id = entry.item_id or ""
                source = f"（{_source_label(entry.source)}）" if entry.source else ""
                if entry.url:
                    lines.append(f"- [{item_id} {title}]({entry.url}){source}".strip())
                else:
                    lines.append(f"- {item_id} {title}{source}".strip())
            lines.append("")
        lines.append("---")
        lines.append("")

    for seq, (_section_heading, item) in enumerate(flat_items, start=1):
        item_id = _item_id(item.title, seq)
        title = _strip_item_number(item.title)
        if item.url:
            lines.append(f"## [{title}]({item.url}) {item_id}")
        else:
            lines.append(f"## {title} {item_id}")
        lines.append("")

        callout = item.one_liner or _first_sentence(item.summary)
        if callout:
            lines.append(f"> {callout}")
            lines.append("")

        for img in _item_images(item):
            lines.append(f"![]({img})")
            lines.append("")

        paragraphs = item.body_paragraphs or _summary_to_paragraphs(item.summary)
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                lines.append(paragraph)
                lines.append("")

        if item.highlights:
            lines.append("要点：")
            for h in item.highlights:
                lines.append(f"- {h}")
            lines.append("")

        if item.evidence_note:
            lines.append(f"> 证据说明：{item.evidence_note}")
            lines.append("")

        src = _source_label(item.source)
        lines.append(f"来源：{src}")
        links = _related_links(item)
        if links:
            lines.append("")
            lines.append("相关链接：")
            for idx, url in enumerate(links, start=1):
                label = "原文" if idx == 1 else f"参考 {idx - 1}"
                lines.append(f"- [{label}]({url})")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _flatten_items(draft: Draft) -> List[tuple[str, DraftItem]]:
    out: List[tuple[str, DraftItem]] = []
    for section in draft.sections:
        for item in section.items:
            out.append((section.heading, item))
    return out


def _normalize_overview_groups(groups: List[OverviewGroup]) -> List[OverviewGroup]:
    """Map section-heading groups to juya-style news-type groups, merging duplicates."""
    _GROUP_MAP = {
        **_SECTION_ALIASES,
        "Headlines": "要闻",
        "Model Frontier": "模型发布",
        "Tools & Open Source": "开发生态",
        "Paper Picks": "技术与洞察",
        "Launchpad": "产品应用",
        "Capital Moves": "行业动态",
        "Industry Watch": "行业动态",
    }
    merged: Dict[str, List[OverviewEntry]] = {}
    for group in groups:
        mapped = _GROUP_MAP.get(group.heading, group.heading)
        if mapped not in merged:
            merged[mapped] = []
        merged[mapped].extend(group.items)
    display_order = _SECTION_ORDER
    result = []
    for heading in display_order:
        if heading in merged and merged[heading]:
            result.append(OverviewGroup(heading=heading, items=merged[heading]))
    for heading, items in merged.items():
        if heading not in display_order:
            result.append(OverviewGroup(heading=heading, items=items))
    return result


def _auto_overview_groups(draft: Draft):
    """Build juya-style overview groups by news type, not section."""
    groups: Dict[str, List[OverviewEntry]] = {}
    seq = 1
    for section in draft.sections:
        group_name = _normalize_section_heading(section.heading)
        if group_name not in groups:
            groups[group_name] = []
        for item in section.items:
            item_id = _item_id(item.title, seq)
            groups[group_name].append(OverviewEntry(
                title=_strip_item_number(item.title),
                url=item.url,
                item_id=item_id,
                source=item.source,
            ))
            seq += 1
    result = []
    # Known group headings in display order; unknown ones appended after.
    display_order = _SECTION_ORDER
    for heading in display_order:
        if heading in groups and groups[heading]:
            result.append(OverviewGroup(heading=heading, items=groups[heading]))
    for heading, items in groups.items():
        if heading not in display_order:
            result.append(OverviewGroup(heading=heading, items=items))
    return result


def _item_id(title: str, fallback: int) -> str:
    m = _re_mod.match(r"^\s*#?(\d+)\b", title or "")
    if m:
        return f"#{m.group(1)}"
    return f"#{fallback}"


def _strip_item_number(title: str) -> str:
    return _re_mod.sub(r"^\s*#?\d+[\s.、:-]*", "", title or "").strip()


def _first_sentence(text: str) -> str:
    t = " ".join((text or "").split())
    if not t:
        return ""
    m = _re_mod.search(r"[。！？.!?]", t)
    if m:
        return t[:m.end()]
    return t[:120]


def _summary_to_paragraphs(summary: str) -> List[str]:
    text = (summary or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in _re_mod.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    sentences = _re_mod.split(r"(?<=[。！？.!?])\s+", text)
    if len(sentences) <= 3:
        return [text]
    mid = max(2, len(sentences) // 2)
    return [
        " ".join(sentences[:mid]).strip(),
        " ".join(sentences[mid:]).strip(),
    ]


def _item_images(item: DraftItem) -> List[str]:
    images = []
    if item.image_url:
        images.append(item.image_url)
    images.extend(item.images)
    return _dedupe(images)[:3]


def _related_links(item: DraftItem) -> List[str]:
    links = []
    if item.url:
        links.append(item.url)
    links.extend(item.related_links)
    return _dedupe([u for u in links if u])


def _dedupe(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


_SECTION_SUBTITLES: Dict[str, str] = {
    "要闻": "Headlines",
    "模型发布": "Model Releases",
    "开发生态": "Developer Ecosystem",
    "技术与洞察": "Technical Insight",
    "产品应用": "Product Applications",
    "行业动态": "Industry Watch",
    "前瞻与传闻": "Forward Signals",
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
