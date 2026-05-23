import json

import pytest

from agent.agents.writer import WriterFailed, write_draft, render_markdown
from agent.harness.budget import BudgetTracker
from agent.harness.trace import Tracer
from agent.llm import LLMMessage
from agent.llm.mock_provider import MockLLMProvider
from agent.schemas import CuratedItem, Draft, DraftItem, DraftSection, OverviewEntry, OverviewGroup


def _make_curated():
    return [
        CuratedItem(
            title=f"item {i}",
            url=f"https://x.com/{i}",
            summary=f"sum {i}",
            source="s",
            source_type="rss",
            published_at="",
            score=1.0,
        )
        for i in range(3)
    ]


def _provider_emitting(text: str) -> MockLLMProvider:
    return MockLLMProvider(model="m", responder=lambda msgs: text)


def test_writer_accepts_valid_json(tmp_path):
    valid = json.dumps(
        {
            "date": "2026-05-09",
            "title": "T",
            "sections": [
                {
                    "heading": "h",
                    "items": [
                        {"title": "t1", "summary": "s1", "url": "u1", "source": "src"}
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )
    provider = _provider_emitting(valid)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)
    draft = write_draft(
        provider=provider,
        items=_make_curated(),
        date="2026-05-09",
        system_prompt="s",
        user_template="d={date} m={max_items} i={items_json}",
        max_items=5,
        tracer=tracer,
        budget=budget,
    )
    assert draft.title == "T"
    assert draft.sections[0].items[0].url == "u1"
    md = render_markdown(draft)
    assert "## 概览" in md       # juya-style overview index
    assert "#1" in md            # item number in overview
    assert "https://x.com" not in md  # we only used u1 in the draft
    assert "u1" in md


def test_writer_can_complete_omitted_curated_items(tmp_path):
    items = [
        CuratedItem(
            title="模型更新",
            url="https://example.com/model",
            summary="模型更新摘要。",
            source="official",
            source_type="rss",
            section="模型发布",
        ),
        CuratedItem(
            title="开源工具",
            url="https://example.com/tool",
            summary="开源工具摘要。",
            source="github",
            source_type="rss",
            section="开发生态",
        ),
        CuratedItem(
            title="融资新闻",
            url="https://example.com/funding",
            summary="融资新闻摘要。",
            source="media",
            source_type="rss",
            section="行业动态",
        ),
    ]
    valid = json.dumps(
        {
            "date": "2026-05-09",
            "title": "T",
            "sections": [
                {
                    "heading": "模型发布",
                    "items": [
                        {
                            "title": "#1 模型更新",
                            "summary": "模型更新摘要。",
                            "url": "https://example.com/model",
                            "source": "official",
                        }
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )
    provider = _provider_emitting(valid)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)
    draft = write_draft(
        provider=provider,
        items=items,
        date="2026-05-09",
        system_prompt="s",
        user_template="d={date} m={max_items} i={items_json}",
        max_items=5,
        tracer=tracer,
        budget=budget,
        complete_with_items=True,
    )
    urls = [item.url for section in draft.sections for item in section.items]
    assert urls == [
        "https://example.com/model",
        "https://example.com/tool",
        "https://example.com/funding",
    ]
    assert [section.heading for section in draft.sections] == [
        "要闻", "模型发布", "开发生态", "技术与洞察",
        "产品应用", "行业动态", "前瞻与传闻",
    ]


def test_overview_groups_render_in_stable_juya_order():
    draft = Draft(
        date="2026-05-09",
        title="T",
        overview_groups=[
            OverviewGroup(heading="技术与洞察", items=[
                OverviewEntry(title="Paper", url="https://p.com", item_id="#3", source="arxiv")
            ]),
            OverviewGroup(heading="要闻", items=[
                OverviewEntry(title="Headline", url="https://h.com", item_id="#1", source="src")
            ]),
            OverviewGroup(heading="行业动态", items=[
                OverviewEntry(title="Capital", url="https://c.com", item_id="#2", source="src")
            ]),
        ],
        sections=[
            DraftSection(heading="要闻", items=[
                DraftItem(title="#1 Headline", summary="s", url="https://h.com", source="src")
            ]),
            DraftSection(heading="行业动态", items=[
                DraftItem(title="#2 Capital", summary="s", url="https://c.com", source="src")
            ]),
            DraftSection(heading="技术与洞察", items=[
                DraftItem(title="#3 Paper", summary="s", url="https://p.com", source="arxiv")
            ]),
        ],
    )
    md = render_markdown(draft)
    assert md.index("### 要闻") < md.index("### 技术与洞察") < md.index("### 行业动态")


def test_writer_rejects_non_json(tmp_path):
    provider = _provider_emitting("this is not json at all")
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)
    with pytest.raises(WriterFailed):
        write_draft(
            provider=provider,
            items=_make_curated(),
            date="2026-05-09",
            system_prompt="s",
            user_template="d={date} m={max_items} i={items_json}",
            max_items=5,
            tracer=tracer,
            budget=budget,
        )


def test_writer_rejects_schema_mismatch(tmp_path):
    bad = json.dumps({"date": "2026-05-09", "title": "T"})  # sections missing
    provider = _provider_emitting(bad)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)
    with pytest.raises(WriterFailed):
        write_draft(
            provider=provider,
            items=_make_curated(),
            date="2026-05-09",
            system_prompt="s",
            user_template="d={date} m={max_items} i={items_json}",
            max_items=5,
            tracer=tracer,
            budget=budget,
        )


def test_writer_strips_code_fence(tmp_path):
    fenced = "```json\n" + json.dumps(
        {
            "date": "2026-05-09",
            "title": "T",
            "sections": [
                {
                    "heading": "h",
                    "items": [
                        {"title": "t1", "summary": "s1", "url": "u1", "source": "src"}
                    ],
                }
            ],
        }
    ) + "\n```"
    provider = _provider_emitting(fenced)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)
    draft = write_draft(
        provider=provider,
        items=_make_curated(),
        date="2026-05-09",
        system_prompt="s",
        user_template="d={date} m={max_items} i={items_json}",
        max_items=5,
        tracer=tracer,
        budget=budget,
    )
    assert draft.title == "T"
