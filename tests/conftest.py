"""Shared pytest fixtures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List

import pytest

from agent.llm import LLMMessage
from agent.llm.mock_provider import MockLLMProvider
from agent.sources.base import RawItem


@pytest.fixture
def fake_raw_items() -> List[RawItem]:
    now = datetime.now(timezone.utc).isoformat()
    return [
        RawItem(
            source_id="hf_blog",
            source_type="rss",
            title="HuggingFace launches new dataset",
            url="https://hf.co/blog/dataset-x",
            summary="Today HF launched dataset X with 10M rows.",
            published_at=now,
        ),
        RawItem(
            source_id="oai_news",
            source_type="rss",
            title="OpenAI releases new model",
            url="https://openai.com/news/model-y",
            summary="Model Y is now available.",
            published_at=now,
        ),
        RawItem(
            source_id="ant_news",
            source_type="rss",
            title="Anthropic publishes safety paper",
            url="https://anthropic.com/news/safety-z",
            summary="Paper Z covers safety evaluation.",
            published_at=now,
        ),
        RawItem(
            source_id="hf_blog",
            source_type="rss",
            title="HuggingFace launches new dataset",  # exact dup of #1
            url="https://hf.co/blog/dataset-x",
            summary="duplicate",
            published_at=now,
        ),
    ]


@pytest.fixture
def fake_source_specs() -> List[Dict[str, object]]:
    return [
        {"id": "hf_blog", "type": "rss", "url": "https://hf.co/blog/feed.xml", "weight": 1.0},
        {"id": "oai_news", "type": "rss", "url": "https://openai.com/news/rss.xml", "weight": 1.2},
        {"id": "ant_news", "type": "rss", "url": "https://anthropic.com/news/rss.xml", "weight": 1.1},
    ]


@pytest.fixture
def cfg():
    return {
        "run": {"timezone": "Asia/Shanghai", "max_items_curate": 5},
        "llm": {"temperature": 0.0, "max_output_tokens": 1024},
        "budget": {
            "max_total_input_tokens": 50_000,
            "max_total_output_tokens": 10_000,
            "max_total_calls": 10,
            "hard_fail_on_exceed": True,
        },
        "context": {"max_messages_keep": 10, "per_message_max_chars": 4000},
        "eval": {
            "min_section_count": 3,
            "min_unique_titles_ratio": 0.8,
            "forbid_phrases": ["作为AI", "I cannot"],
        },
        "sources": [],
    }


@pytest.fixture
def prompts():
    return {
        "writer_system": "system",
        "writer_user_template": "date={date} max={max_items} items={items_json}",
        "critic_system": "critic",
        "critic_user_template": "items={items_json} draft={draft_json}",
    }


@pytest.fixture
def scripted_writer_provider(fake_raw_items):
    """Mock provider that handles both writer and semantic-dup-critic calls."""

    def responder(messages: List[LLMMessage]) -> str:
        system = messages[0].content if messages else ""
        # Repairer prompt contains "修复" AND "语义重复" — check "修复" first.
        if "修复" in system:
            return json.dumps({"actions": []})
        # Semantic dup critic detection.
        if "语义重复" in system or "duplicates" in system:
            return json.dumps({"duplicates": []})

        # Writer call: echo URLs from the user prompt so hallucination check passes.
        urls = [it.url for it in fake_raw_items[:3]]
        titles = [it.title for it in fake_raw_items[:3]]
        sources = [it.source_id for it in fake_raw_items[:3]]
        payload = {
            "date": "2026-05-09",
            "title": "AI 日报 2026-05-09",
            "sections": [
                {
                    "heading": "模型与产品",
                    "items": [
                        {
                            "title": titles[0],
                            "summary": "数据集发布，含1000万行数据。",
                            "url": urls[0],
                            "source": sources[0],
                        }
                    ],
                },
                {
                    "heading": "研究",
                    "items": [
                        {
                            "title": titles[1],
                            "summary": "新模型公开发布。",
                            "url": urls[1],
                            "source": sources[1],
                        }
                    ],
                },
                {
                    "heading": "安全",
                    "items": [
                        {
                            "title": titles[2],
                            "summary": "安全评估论文发布。",
                            "url": urls[2],
                            "source": sources[2],
                        }
                    ],
                },
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    return MockLLMProvider(model="mock-writer", responder=responder)
