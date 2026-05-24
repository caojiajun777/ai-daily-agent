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
        "模型发布", "开发生态", "行业动态",
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


def test_render_markdown_strips_internal_image_descriptions():
    draft = Draft(
        date="2026-05-09",
        title="T",
        sections=[
            DraftSection(heading="要闻", items=[
                DraftItem(
                    title="#1 Headline",
                    one_liner="事实判断（配图显示：一张品牌图。）",
                    summary="正文摘要（配图显示：一个 logo 和背景光效。）",
                    body_paragraphs=["第一段（配图显示：截图说明。）", "第二段。"],
                    url="https://h.com",
                    source="src",
                )
            ])
        ],
    )

    md = render_markdown(draft)

    assert "配图显示" not in md
    assert "第一段" in md
    assert "第二段" in md


def test_writer_localizes_english_titles_from_chinese_callout(tmp_path):
    valid = json.dumps(
        {
            "date": "2026-05-09",
            "title": "T",
            "overview": "DeepSeek 推进融资。",
            "sections": [
                {"heading": "前瞻与传闻", "items": [
                    {
                        "title": "#1 All the news from the Google I/O Developer keynote",
                        "one_liner": "Google I/O 发布 Gemini 与 Antigravity 更新。",
                        "summary": "Google I/O 发布 Gemini 与 Antigravity 更新。",
                        "url": "https://example.com/io",
                        "source": "src",
                    }
                ]},
            ],
        },
        ensure_ascii=False,
    )
    provider = _provider_emitting(valid)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)

    draft = write_draft(
        provider=provider,
        items=[
            CuratedItem(
                title="Google I/O",
                url="https://example.com/io",
                summary="s",
                source="src",
                source_type="rss",
                section="前瞻与传闻",
            )
        ],
        date="2026-05-09",
        system_prompt="s",
        user_template="d={date} m={max_items} i={items_json}",
        max_items=5,
        tracer=tracer,
        budget=budget,
        complete_with_items=True,
    )

    assert draft.sections[0].items[0].title == "#1 Google I/O 发布 Gemini 与 Antigravity 更新"


def test_writer_localized_titles_preserve_decimal_versions(tmp_path):
    valid = json.dumps(
        {
            "date": "2026-05-09",
            "title": "T",
            "sections": [
                {"heading": "要闻", "items": [
                    {
                        "title": "#1 All the news from the Google I/O Developer keynote",
                        "one_liner": "Google I/O 发布 Gemini 3.5 系列与 Antigravity 2.0 平台。",
                        "summary": "Google I/O 发布 Gemini 3.5 系列与 Antigravity 2.0 平台。",
                        "url": "https://example.com/io",
                        "source": "src",
                    },
                    {
                        "title": "#2 Alibaba's latest model optimizes chip code",
                        "one_liner": "阿里 Qwen3.7-Max 自主运行 35 小时优化芯片代码。",
                        "summary": "阿里 Qwen3.7-Max 自主运行 35 小时优化芯片代码。",
                        "url": "https://example.com/qwen",
                        "source": "src",
                    },
                    {
                        "title": "#3 Databricks integrates GPT into enterprise agents",
                        "one_liner": "Databricks 集成 GPT-5.5 用于企业 Agent 工作流。",
                        "summary": "Databricks 集成 GPT-5.5 用于企业 Agent 工作流。",
                        "url": "https://example.com/databricks",
                        "source": "src",
                    },
                ]},
            ],
        },
        ensure_ascii=False,
    )
    provider = _provider_emitting(valid)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)

    draft = write_draft(
        provider=provider,
        items=[],
        date="2026-05-09",
        system_prompt="s",
        user_template="d={date} m={max_items} i={items_json}",
        max_items=5,
        tracer=tracer,
        budget=budget,
        complete_with_items=True,
    )

    titles = [item.title for item in draft.sections[0].items]
    assert "Gemini 3.5" in titles[0]
    assert "Antigravity 2.0" in titles[0]
    assert "Qwen3.7-Max" in titles[1]
    assert "GPT-5.5" in titles[2]


def test_writer_repairs_hard_truncated_product_titles(tmp_path):
    valid = json.dumps(
        {
            "date": "2026-05-09",
            "title": "T",
            "sections": [
                {"heading": "要闻", "items": [
                    {
                        "title": "#1 Google I/O 2026 发布 Gemini 3.5 系列和 Antigravit",
                        "one_liner": "Google I/O 2026 发布 Gemini 3.5 系列和 Antigravity 2.0 智能体平台。",
                        "summary": "Google I/O 2026 发布 Gemini 3.5 系列和 Antigravity 2.0 智能体平台。",
                        "url": "https://example.com/io",
                        "source": "src",
                    },
                    {
                        "title": "#2 阿里 Qwen3.7-Max 自主运行 35 小时优化芯片代码，匹配 Claude Op",
                        "one_liner": "阿里 Qwen3.7-Max 自主运行 35 小时优化芯片代码，匹配 Claude Opus 4.6。",
                        "summary": "阿里 Qwen3.7-Max 自主运行 35 小时优化芯片代码，匹配 Claude Opus 4.6。",
                        "url": "https://example.com/qwen",
                        "source": "src",
                    },
                ]},
            ],
        },
        ensure_ascii=False,
    )
    provider = _provider_emitting(valid)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)

    draft = write_draft(
        provider=provider,
        items=[],
        date="2026-05-09",
        system_prompt="s",
        user_template="d={date} m={max_items} i={items_json}",
        max_items=5,
        tracer=tracer,
        budget=budget,
        complete_with_items=True,
    )

    titles = [item.title for item in draft.sections[0].items]
    assert titles[0] == "#1 Google I/O 2026 发布 Gemini 3.5 与 Antigravity 2.0"
    assert titles[1] == "#2 阿里 Qwen3.7-Max 自主运行 35 小时优化芯片代码"
    assert "Antigravity 2.0" in titles[0]
    assert "Claude Op" not in titles[1]


def test_overview_marks_forward_signals_as_reported(tmp_path):
    valid = json.dumps(
        {
            "date": "2026-05-09",
            "title": "T",
            "overview": "DeepSeek 推进 700 亿元融资，英伟达财报更新。",
            "sections": [
                {"heading": "前瞻与传闻", "items": [
                    {
                        "title": "#1 DeepSeek 推进 700 亿元融资",
                        "summary": "据报道 DeepSeek 推进融资。",
                        "url": "https://example.com/deepseek",
                        "source": "src",
                    }
                ]},
            ],
        },
        ensure_ascii=False,
    )
    provider = _provider_emitting(valid)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)

    draft = write_draft(
        provider=provider,
        items=[
            CuratedItem(
                title="DeepSeek 融资",
                url="https://example.com/deepseek",
                summary="s",
                source="src",
                source_type="rss",
                section="前瞻与传闻",
            )
        ],
        date="2026-05-09",
        system_prompt="s",
        user_template="d={date} m={max_items} i={items_json}",
        max_items=5,
        tracer=tracer,
        budget=budget,
        complete_with_items=True,
    )

    assert draft.overview.startswith("据报道，")


def test_overview_does_not_mark_confirmed_story_for_shared_company_name(tmp_path):
    valid = json.dumps(
        {
            "date": "2026-05-09",
            "title": "T",
            "overview": "DeepSeek-V4-Pro API 永久降价。",
            "sections": [
                {"heading": "要闻", "items": [
                    {
                        "title": "#1 DeepSeek-V4-Pro API 永久降价",
                        "summary": "DeepSeek 官方定价页更新。",
                        "url": "https://example.com/pricing",
                        "source": "src",
                    }
                ]},
                {"heading": "前瞻与传闻", "items": [
                    {
                        "title": "#2 DeepSeek 推进 700 亿元融资",
                        "summary": "据报道 DeepSeek 推进融资。",
                        "url": "https://example.com/funding",
                        "source": "src",
                    }
                ]},
            ],
        },
        ensure_ascii=False,
    )
    provider = _provider_emitting(valid)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)

    draft = write_draft(
        provider=provider,
        items=[
            CuratedItem(
                title="DeepSeek 定价",
                url="https://example.com/pricing",
                summary="s",
                source="src",
                source_type="rss",
                section="要闻",
            ),
            CuratedItem(
                title="DeepSeek 融资",
                url="https://example.com/funding",
                summary="s",
                source="src",
                source_type="rss",
                section="前瞻与传闻",
            ),
        ],
        date="2026-05-09",
        system_prompt="s",
        user_template="d={date} m={max_items} i={items_json}",
        max_items=5,
        tracer=tracer,
        budget=budget,
        complete_with_items=True,
    )

    assert draft.overview == "DeepSeek-V4-Pro API 永久降价。"


def test_overview_scopes_reported_qualifier_to_weak_sentence(tmp_path):
    valid = json.dumps(
        {
            "date": "2026-05-09",
            "title": "T",
            "overview": (
                "据报道，Google I/O 发布 Gemini 3.5 与 Antigravity 2.0。"
                "DeepSeek-V4-Pro API 永久降价。"
                "Anthropic 被曝最快下周完成超 300 亿美元融资。"
            ),
            "sections": [
                {"heading": "要闻", "items": [
                    {
                        "title": "#1 Google I/O 发布 Gemini 3.5 与 Antigravity 2.0",
                        "summary": "Google 官方发布。",
                        "url": "https://example.com/io",
                        "source": "src",
                        "confidence": "high",
                        "rumor_level": "confirmed",
                    },
                    {
                        "title": "#2 DeepSeek-V4-Pro API 永久降价",
                        "summary": "DeepSeek 官方定价页更新。",
                        "url": "https://example.com/pricing",
                        "source": "src",
                        "confidence": "high",
                        "rumor_level": "confirmed",
                    },
                ]},
                {"heading": "前瞻与传闻", "items": [
                    {
                        "title": "#3 消息称 Anthropic 完成超 300 亿美元融资",
                        "summary": "消息尚未获官方确认。",
                        "url": "https://example.com/anthropic",
                        "source": "src",
                        "confidence": "low",
                        "rumor_level": "rumor",
                        "evidence_note": "媒体报道，尚未获官方确认",
                    }
                ]},
            ],
        },
        ensure_ascii=False,
    )
    provider = _provider_emitting(valid)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)

    draft = write_draft(
        provider=provider,
        items=[],
        date="2026-05-09",
        system_prompt="s",
        user_template="d={date} m={max_items} i={items_json}",
        max_items=5,
        tracer=tracer,
        budget=budget,
        complete_with_items=True,
    )

    assert draft.overview.startswith("今天主线是")
    assert "DeepSeek-V4-Pro API 永久降价" in draft.overview
    assert "据报道，Google" not in draft.overview
    assert "Anthropic" in draft.overview
    assert "仍需等待进一步确认" in draft.overview


def test_writer_rebuilds_overview_as_editorial_mainlines(tmp_path):
    valid = json.dumps(
        {
            "date": "2026-05-09",
            "title": "T",
            "overview": "Google 发布更新。DeepSeek 降价。Qwen 发布模型。",
            "sections": [
                {"heading": "要闻", "items": [
                    {
                        "title": "#1 DeepSeek-V4-Pro API 永久降价",
                        "summary": "DeepSeek-V4-Pro API 永久降价。",
                        "url": "https://example.com/pricing",
                        "source": "src",
                        "confidence": "high",
                    }
                ]},
                {"heading": "模型发布", "items": [
                    {
                        "title": "#2 Qwen3.7-Max 发布",
                        "summary": "阿里发布 Qwen3.7-Max 模型。",
                        "url": "https://example.com/qwen",
                        "source": "src",
                        "confidence": "high",
                    }
                ]},
                {"heading": "开发生态", "items": [
                    {
                        "title": "#3 Genkit Middleware 加固 Agent 应用",
                        "summary": "Genkit Middleware 支持 Agent 应用拦截和扩展。",
                        "url": "https://example.com/genkit",
                        "source": "src",
                        "confidence": "high",
                    }
                ]},
            ],
        },
        ensure_ascii=False,
    )
    provider = _provider_emitting(valid)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)

    draft = write_draft(
        provider=provider,
        items=[],
        date="2026-05-09",
        system_prompt="s",
        user_template="d={date} m={max_items} i={items_json}",
        max_items=5,
        tracer=tracer,
        budget=budget,
        complete_with_items=True,
    )

    assert draft.overview.startswith("今天主线是")
    assert "模型成本战" in draft.overview
    assert "Agent 工程化" in draft.overview
    assert "确认消息里" in draft.overview


def test_render_markdown_softens_unsupported_price_comparisons():
    draft = Draft(
        date="2026-05-09",
        title="T",
        overview="DeepSeek-V4-Pro API 永久降价，输出价格仅为 GPT-5.5 的 1/34。",
        sections=[
            DraftSection(heading="要闻", items=[
                DraftItem(
                    title="#1 DeepSeek-V4-Pro API 永久降价",
                    one_liner="DeepSeek-V4-Pro API 永久降价，输出价格仅为 GPT-5.5 的 1/34。",
                    summary="DeepSeek 官方定价页更新。",
                    body_paragraphs=[
                        "这一永久降价策略使 DeepSeek-V4-Pro 的输出价格比 GPT-5.5 低 34 倍以上，对成本敏感应用有吸引力。"
                    ],
                    highlights=["输出价格比 GPT-5.5 低 34 倍"],
                    url="https://example.com/pricing",
                    source="deepseek_pricing",
                    evidence_note="官方定价页明确显示价格；竞品对比需按公开价格口径估算",
                )
            ])
        ],
    )

    md = render_markdown(draft)

    assert "输出价格仅为 GPT-5.5" not in md
    assert "输出价格比 GPT-5.5 低 34 倍" not in md
    assert "按公开价格口径估算" in md


def test_empty_sections_are_pruned_when_completing_with_curated(tmp_path):
    valid = json.dumps(
        {
            "date": "2026-05-09",
            "title": "T",
            "sections": [
                {"heading": "要闻", "items": []},
                {"heading": "模型发布", "items": [
                    {"title": "#1 模型更新", "summary": "模型更新摘要。",
                     "url": "https://example.com/model", "source": "official"}
                ]},
                {"heading": "前瞻与传闻", "items": []},
            ],
        },
        ensure_ascii=False,
    )
    provider = _provider_emitting(valid)
    tracer = Tracer(str(tmp_path / "t.jsonl"), run_id="r")
    budget = BudgetTracker(100_000, 10_000, 10)

    draft = write_draft(
        provider=provider,
        items=[
            CuratedItem(
                title="模型更新",
                url="https://example.com/model",
                summary="模型更新摘要。",
                source="official",
                source_type="rss",
                section="模型发布",
            )
        ],
        date="2026-05-09",
        system_prompt="s",
        user_template="d={date} m={max_items} i={items_json}",
        max_items=5,
        tracer=tracer,
        budget=budget,
        complete_with_items=True,
    )

    assert [section.heading for section in draft.sections] == ["模型发布"]


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
