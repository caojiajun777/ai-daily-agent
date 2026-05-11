import json

import pytest

from agent.agents.writer import WriterFailed, write_draft, render_markdown
from agent.harness.budget import BudgetTracker
from agent.harness.trace import Tracer
from agent.llm import LLMMessage
from agent.llm.mock_provider import MockLLMProvider
from agent.schemas import CuratedItem


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
    assert "## h" in md
    assert "https://x.com" not in md  # we only used u1 in the draft
    assert "u1" in md


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
