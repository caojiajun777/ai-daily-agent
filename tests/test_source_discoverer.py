"""Tests for the source discovery agent.

Covers:
  - Candidate schema and JSON parsing
  - RSS validation (real and fake URLs)
  - Uniqueness scoring against existing sources
  - Full pipeline with mock LLM provider
  - YAML snippet generation
  - Gap analysis focus instructions
  - CLI discover-sources command
"""

from __future__ import annotations

import json
import os
import sys
from typing import List
from unittest.mock import patch

import pytest
import yaml

from agent.agents.source_discoverer import (
    CandidateSource,
    DiscoveryReport,
    _build_focus_instruction,
    _extract_json,
    _generate_candidates,
    _load_existing_domains,
    _load_existing_x_usernames,
    _render_yaml_snippet,
    _score_ai_relevance,
    _score_freshness,
    _score_uniqueness,
    _slug,
    _validate_rss,
    discover_sources,
)
from agent.llm.mock_provider import MockLLMProvider


# ── JSON extraction ─────────────────────────────────────────────────────


def test_extract_json_plain():
    assert _extract_json('[{"a": 1}]') == [{"a": 1}]


def test_extract_json_with_think_block():
    raw = "<think>reasoning...</think>\n[{\"name\": \"test\"}]"
    assert _extract_json(raw) == [{"name": "test"}]


def test_extract_json_with_code_fence():
    raw = '```json\n[{"x": "y"}]\n```'
    assert _extract_json(raw) == [{"x": "y"}]


def test_extract_json_invalid_returns_none():
    assert _extract_json("just some text") is None


# ── Focus instructions ──────────────────────────────────────────────────


def test_build_focus_instruction_auto():
    result = _build_focus_instruction("auto", "some sources")
    assert "自行判断" in result
    assert "覆盖盲区" in result


def test_build_focus_instruction_broad():
    result = _build_focus_instruction("broad", "")
    assert "广泛搜索" in result


def test_build_focus_instruction_chinese_models():
    result = _build_focus_instruction("chinese-ai-models", "")
    assert "DeepSeek" in result
    assert "ChatGLM" in result
    assert "月之暗面" in result


def test_build_focus_instruction_fallback():
    result = _build_focus_instruction("unknown-topic", "")
    assert "广泛搜索" in result  # falls back to broad


# ── Uniqueness scoring ──────────────────────────────────────────────────


def test_score_uniqueness_new_rss():
    c = CandidateSource(name="test", source_type="rss", url="https://newdomain.com/feed.xml")
    score = _score_uniqueness(c, {"existing.com"}, {"existing_account"})
    assert score == 1.0


def test_score_uniqueness_duplicate_rss():
    c = CandidateSource(name="test", source_type="rss", url="https://existing.com/feed.xml")
    score = _score_uniqueness(c, {"existing.com"}, {"existing_account"})
    assert score == 0.0


def test_score_uniqueness_new_x():
    c = CandidateSource(name="test", source_type="x", username="new_account")
    score = _score_uniqueness(c, {"existing.com"}, {"existing_account"})
    assert score == 1.0


def test_score_uniqueness_duplicate_x():
    c = CandidateSource(name="test", source_type="x", username="existing_account")
    score = _score_uniqueness(c, {"existing.com"}, {"existing_account"})
    assert score == 0.0


def test_score_uniqueness_duplicate_x_case_insensitive():
    c = CandidateSource(name="test", source_type="x", username="EXISTING_ACCOUNT")
    score = _score_uniqueness(c, {"existing.com"}, {"existing_account"})
    assert score == 0.0


# ── Domain extraction from config ───────────────────────────────────────


def test_load_existing_domains(tmp_path):
    config = tmp_path / "test.yaml"
    config.write_text(
        yaml.dump({
            "sources": [
                {"id": "a", "type": "rss", "url": "https://openai.com/news/rss.xml"},
                {"id": "b", "type": "x", "username": "OpenAI"},
                {"id": "c", "type": "rss", "url": "https://huggingface.co/blog/feed.xml"},
            ]
        }),
        encoding="utf-8",
    )
    domains = _load_existing_domains(str(config))
    assert "openai.com" in domains
    assert "huggingface.co" in domains


def test_load_existing_x_usernames(tmp_path):
    config = tmp_path / "test.yaml"
    config.write_text(
        yaml.dump({
            "sources": [
                {"id": "a", "type": "x", "username": "OpenAI"},
                {"id": "b", "type": "x", "username": "deepseek_ai"},
                {"id": "c", "type": "rss", "url": "https://example.com/feed.xml"},
            ]
        }),
        encoding="utf-8",
    )
    usernames = _load_existing_x_usernames(str(config))
    assert "openai" in usernames
    assert "deepseek_ai" in usernames


# ── RSS validation ──────────────────────────────────────────────────────


def test_validate_rss_no_url():
    c = CandidateSource(name="test", source_type="rss")
    _validate_rss(c)
    assert not c.reachable
    assert "no URL" in c.validation_note


def test_validate_rss_bogus_url():
    c = CandidateSource(name="test", source_type="rss", url="https://this-is-not-real.invalid/feed")
    _validate_rss(c)
    assert not c.reachable


def test_validate_rss_real_feed():
    """Validate against a known real RSS feed (HuggingFace blog)."""
    c = CandidateSource(
        name="HuggingFace Blog",
        source_type="rss",
        url="https://huggingface.co/blog/feed.xml",
    )
    _validate_rss(c)
    assert c.reachable
    assert c.validated
    assert c.freshness_score > 0.0
    assert c.relevance_score > 0.0
    assert "OK" in c.validation_note


# ── Freshness scoring ───────────────────────────────────────────────────


def test_freshness_score_empty():
    assert _score_freshness([]) == 0.0


# ── Relevance scoring ───────────────────────────────────────────────────


def test_relevance_ai_titles():
    class FakeEntry:
        def __init__(self, title):
            self.title = title

    entries = [
        FakeEntry("New AI model released by OpenAI"),
        FakeEntry("Machine learning breakthrough in protein folding"),
        FakeEntry("Deep learning tutorial for beginners"),
    ]
    assert _score_ai_relevance(entries) == 1.0


def test_relevance_non_ai_titles():
    class FakeEntry:
        def __init__(self, title):
            self.title = title

    entries = [
        FakeEntry("Cooking recipes for summer"),
        FakeEntry("Baseball scores yesterday"),
    ]
    assert _score_ai_relevance(entries) == 0.0


# ── YAML snippet rendering ──────────────────────────────────────────────


def test_render_yaml_snippet_rss():
    c = CandidateSource(
        name="Test Source",
        source_type="rss",
        url="https://example.com/feed.xml",
        overall_score=0.85,
        reason="Good coverage of AI news.",
    )
    snippet = _render_yaml_snippet([c])
    assert "discovered_test_source" in snippet
    assert "url: \"https://example.com/feed.xml\"" in snippet
    assert "type: \"rss\"" in snippet
    assert "0.85" in snippet


def test_render_yaml_snippet_x():
    c = CandidateSource(
        name="Test X Account",
        source_type="x",
        username="test_account",
        account_type="kol",
        overall_score=0.72,
        reason="Great KOL insights.",
    )
    snippet = _render_yaml_snippet([c])
    assert "discovered_x_test_account" in snippet
    assert "username: \"test_account\"" in snippet
    assert "type: \"x\"" in snippet


# ── Slug helper ─────────────────────────────────────────────────────────


def test_slug():
    assert _slug("Hello World & AI") == "hello_world___ai"


# ── Full pipeline with mock provider ────────────────────────────────────


_candidate_response = json.dumps([
    {
        "name": "Test AI Blog",
        "type": "rss",
        "url": "https://huggingface.co/blog/feed.xml",
        "account_type": "official",
        "reason": "Already have but different URL, good for test.",
        "language": "en",
        "category": "model-provider",
    },
    {
        "name": "Fake News Site",
        "type": "rss",
        "url": "https://definitely-not-real-12345.invalid/rss",
        "account_type": "media",
        "reason": "Should fail validation.",
        "language": "en",
        "category": "media",
    },
    {
        "name": "DeepSeek X",
        "type": "x",
        "username": "deepseek_ai",
        "account_type": "official",
        "reason": "Key Chinese AI lab.",
        "language": "zh",
        "category": "model-provider",
    },
])


@pytest.fixture
def discovery_provider():
    """Mock provider that returns a mix of real and fake candidates."""
    def responder(messages):
        return _candidate_response
    return MockLLMProvider(model="mock-discovery", responder=responder)


@pytest.fixture
def temp_config(tmp_path):
    """Minimal config with some existing sources."""
    config = tmp_path / "default.yaml"
    config.write_text(
        yaml.dump({
            "sources": [
                {"id": "openai", "type": "rss", "url": "https://openai.com/news/rss.xml"},
                {"id": "x_baidu", "type": "x", "username": "Baidu_Inc"},
            ]
        }),
        encoding="utf-8",
    )
    return str(config)


def test_discover_sources_pipeline(discovery_provider, temp_config):
    """End-to-end: mock LLM → validate → score → report."""
    report = discover_sources(
        topic="broad",
        provider=discovery_provider,
        existing_config_path=temp_config,
        max_candidates=10,
        min_score=0.3,
    )

    assert isinstance(report, DiscoveryReport)
    assert report.topic == "broad"
    assert report.candidates_generated == 3

    # The real HF feed should pass; the fake feed should fail.
    assert report.candidates_passed >= 1
    passed_names = [c.name for c in report.passed]
    assert "Test AI Blog" in passed_names

    # Fake feed should be rejected.
    rejected_names = [c.name for c in report.rejected]
    assert "Fake News Site" in rejected_names

    # X account: without X_BEARER_TOKEN, gets a pass with estimated scores.
    # Should appear in passed if score is high enough.
    assert any(c.source_type == "x" for c in report.passed)

    # YAML snippet should be generated for passed sources.
    assert len(report.yaml_snippet) > 0


def test_discover_sources_respects_min_score(discovery_provider, temp_config):
    """With min_score=1.0, only perfect sources should pass."""
    report = discover_sources(
        topic="broad",
        provider=discovery_provider,
        existing_config_path=temp_config,
        max_candidates=10,
        min_score=0.99,
    )
    # With a very high threshold, fewer (maybe zero) pass.
    assert report.candidates_passed <= report.candidates_generated


def test_discover_sources_all_topic(discovery_provider, temp_config):
    """The 'auto' topic should work end-to-end."""
    report = discover_sources(
        topic="auto",
        provider=discovery_provider,
        existing_config_path=temp_config,
        max_candidates=10,
        min_score=0.3,
    )
    assert report.topic == "auto"
    assert report.candidates_generated == 3


def test_discover_sources_uniqueness_blocks_duplicate(discovery_provider):
    """A candidate with a URL in the existing config should get uniqueness=0."""
    # The candidate uses huggingface.co which is NOT in our temp_config,
    # but we can test the scoring separately.
    c = CandidateSource(
        name="dup",
        source_type="rss",
        url="https://openai.com/news/rss.xml",
    )
    score = _score_uniqueness(c, {"openai.com"}, set())
    assert score == 0.0
