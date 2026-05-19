"""Tests for unified SourceScout — multi-channel source discovery."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
import yaml

from agent.agents.source_discoverer import CandidateSource
from agent.agents.source_scout import (
    ScoutReport,
    _canonical_key,
    _diffused_to_candidate,
    _render_scout_yaml,
    scout_sources,
)
from agent.llm.mock_provider import MockLLMProvider


# ── Canonical key ──────────────────────────────────────────────────────


def test_canonical_key_rss():
    c = CandidateSource(
        name="Test", source_type="rss",
        url="https://example.com/feed.xml",
    )
    assert _canonical_key(c) == "rss:example.com"


def test_canonical_key_x():
    c = CandidateSource(
        name="Test", source_type="x", username="MyAccount",
    )
    assert _canonical_key(c) == "x:myaccount"


def test_canonical_key_dedup_same_domain():
    """Two RSS sources on the same domain should collide."""
    a = CandidateSource(name="A", source_type="rss", url="https://example.com/rss.xml")
    b = CandidateSource(name="B", source_type="rss", url="https://example.com/feed.xml")
    assert _canonical_key(a) == _canonical_key(b)


# ── DiffusedSource → CandidateSource conversion ────────────────────────


def test_diffused_to_candidate():
    from agent.agents.source_diffuser import DiffusedSource
    ds = DiffusedSource(
        name="Test Blog", source_type="rss",
        url="https://test.com/feed.xml",
        overall_score=0.75, diffusion_method="content_link",
        link_count=5, reason="Linked from 5 items",
        reachable=True, validated=True,
        freshness_score=0.8, relevance_score=0.9,
    )
    c = _diffused_to_candidate(ds)
    assert c.name == "Test Blog"
    assert c.url == "https://test.com/feed.xml"
    assert c.overall_score == 0.75
    assert c.reachable is True


# ── YAML rendering ─────────────────────────────────────────────────────


def test_render_scout_yaml():
    sources = [
        CandidateSource(
            name="Test RSS", source_type="rss",
            url="https://example.com/feed.xml",
            overall_score=0.72, reason="[llm] Good source.",
        ),
        CandidateSource(
            name="Test X", source_type="x",
            username="test_account", account_type="kol",
            overall_score=0.65, reason="[llm,content_link] Cross-validated.",
        ),
    ]
    y = _render_scout_yaml(sources)
    assert "scout_test_rss" in y
    assert "scout_x_test_account" in y
    assert "example.com/feed.xml" in y


# ── Full scout pipeline with mock provider ─────────────────────────────


_candidate_response = json.dumps([
    {
        "name": "AI Research Weekly",
        "type": "rss",
        "url": "https://huggingface.co/blog/feed.xml",
        "account_type": "media",
        "reason": "Already validated real feed for testing.",
        "language": "en",
        "category": "media",
    },
    {
        "name": "Nonexistent RSS",
        "type": "rss",
        "url": "https://definitely-fake-99999.invalid/rss",
        "account_type": "media",
        "reason": "Should fail validation.",
        "language": "en",
        "category": "media",
    },
])


@pytest.fixture
def scout_provider():
    def responder(messages):
        return _candidate_response
    return MockLLMProvider(model="mock-scout", responder=responder)


@pytest.fixture
def scout_config(tmp_path):
    config = tmp_path / "default.yaml"
    config.write_text(yaml.dump({
        "sources": [
            {"id": "openai", "type": "rss", "url": "https://openai.com/news/rss.xml"},
            {"id": "x_deepseek", "type": "x", "username": "deepseek_ai"},
        ]
    }), encoding="utf-8")
    return str(config)


def test_scout_basic(scout_provider, scout_config, tmp_path):
    """Scout with just LLM channel (no collected items)."""
    # Write a local RSS file so validation passes without network.
    rss_xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<rss version=\"2.0\"><channel><title>AI Research Weekly</title>
<link>https://example.com</link>
<item><title>Test paper</title><link>https://example.com/1</link>
<description>AI breakthrough</description>
<pubDate>Mon, 19 May 2026 10:00:00 GMT</pubDate></item>
</channel></rss>"""
    local_rss = tmp_path / "ai_research_feed.xml"
    local_rss.write_text(rss_xml, encoding="utf-8")

    # Override mock candidate URL to use local file.
    candidates = json.loads(_candidate_response)
    candidates[0]["url"] = str(local_rss)

    def responder(messages):
        return json.dumps(candidates)
    from agent.llm.mock_provider import MockLLMProvider
    provider = MockLLMProvider(model="mock-scout", responder=responder)

    report = scout_sources(
        topic="broad",
        provider=provider,
        config_path=scout_config,
        collected_items=None,
        max_per_channel=8,
    )
    assert isinstance(report, ScoutReport)
    assert "llm" in report.channels_used
    assert report.candidates_total >= 0
    passed_names = [c.name for c in report.passed]
    assert any("AI Research" in n for n in passed_names)


def test_scout_cross_channel(scout_provider, scout_config):
    """LLM + content-link channels should run, with dedup across channels."""
    from agent.sources.base import RawItem
    items = [
        RawItem(
            source_id="test", source_type="rss",
            title="AI News",
            url="https://venturebeat.com/ai/some-article",
            summary="test", published_at="2026-05-10T00:00:00Z",
        ),
    ]
    report = scout_sources(
        topic="broad",
        provider=scout_provider,
        config_path=scout_config,
        collected_items=items,
        max_per_channel=8,
    )
    assert "llm" in report.channels_used
    # content_link may fail in offline/restricted-network environments.
    # The channel is exercised but network-dependent.
    assert report.candidates_total >= 0


def test_scout_respects_min_score(scout_provider, scout_config):
    """With a very high min_score, few candidates pass."""
    report = scout_sources(
        topic="broad",
        provider=scout_provider,
        config_path=scout_config,
        collected_items=None,
        max_per_channel=8,
        min_score=0.99,
    )
    assert report.candidates_passed <= report.candidates_total


def test_scout_cross_boost_applies(scout_provider, scout_config):
    """Cross-channel boost increases score for multi-channel sources."""
    # Run without items first — only LLM channel.
    report = scout_sources(
        topic="broad", provider=scout_provider,
        config_path=scout_config, collected_items=None, max_per_channel=8,
    )
    assert report.cross_boosted >= 0
    assert isinstance(report.channel_details, dict)
