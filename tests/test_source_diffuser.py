"""Tests for source diffuser — social graph and content-link diffusion."""

from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from agent.agents.source_diffuser import (
    DiffusedSource,
    DiffusionReport,
    _domain,
    _feed_ai_relevance,
    _feed_freshness,
    _guess_language,
    _infer_account_type,
    _is_ai_text,
    _load_existing_domains,
    _load_existing_x_usernames,
    _load_seed_x_accounts,
    _probe_rss,
    _render_diffused_yaml,
    _RSS_PROBE_PATHS,
    _tweet_freshness,
    diffuse_sources,
)
from agent.sources.base import RawItem


# ── Helpers ────────────────────────────────────────────────────────────


def test_domain_extraction():
    assert _domain("https://example.com/blog/post") == "example.com"
    assert _domain("https://www.openai.com/news/") == "openai.com"
    assert _domain("") == ""


def test_guess_language():
    assert _guess_language("example.cn") == "zh"
    assert _guess_language("example.com") == "en"


def test_infer_account_type():
    assert _infer_account_type(1_000_000) == "official"
    assert _infer_account_type(50_000) == "kol"
    assert _infer_account_type(100) == "media"


def test_is_ai_text():
    assert _is_ai_text("New AI model released by OpenAI")
    assert _is_ai_text("大模型推理能力提升")
    assert not _is_ai_text("Baseball scores for yesterday")


def test_tweet_freshness_recent():
    from datetime import datetime, timezone, timedelta
    recent = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    assert _tweet_freshness(recent) == 1.0


def test_tweet_freshness_old():
    from datetime import datetime, timezone, timedelta
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    assert _tweet_freshness(old) < 0.4


# ── RSS probe paths ────────────────────────────────────────────────────


def test_probe_paths_exist():
    assert len(_RSS_PROBE_PATHS) >= 8


def test_probe_rss_fake_domain():
    result = _probe_rss("definitely-not-real-99999.invalid", timeout=3.0)
    assert result is None


# ── Config loading ─────────────────────────────────────────────────────


def test_load_existing_domains(tmp_path):
    import yaml
    config = tmp_path / "test.yaml"
    config.write_text(yaml.dump({
        "sources": [
            {"id": "a", "type": "rss", "url": "https://openai.com/news/rss.xml"},
            {"id": "b", "type": "rss", "url": "https://example.org/feed.xml"},
            {"id": "c", "type": "x", "username": "test"},
        ]
    }), encoding="utf-8")
    domains = _load_existing_domains(str(config))
    assert "openai.com" in domains
    assert "example.org" in domains


def test_load_existing_x_usernames(tmp_path):
    import yaml
    config = tmp_path / "test.yaml"
    config.write_text(yaml.dump({
        "sources": [
            {"id": "a", "type": "x", "username": "OpenAI"},
            {"id": "b", "type": "rss", "url": "https://x.com"},
        ]
    }), encoding="utf-8")
    usernames = _load_existing_x_usernames(str(config))
    assert "openai" in usernames


def test_load_seed_x_accounts(tmp_path):
    import yaml
    config = tmp_path / "test.yaml"
    config.write_text(yaml.dump({
        "sources": [
            {"id": "a", "type": "x", "username": "deepseek_ai", "weight": 1.4},
            {"id": "b", "type": "x", "username": "OpenAI", "weight": 1.3},
            {"id": "c", "type": "rss", "url": "https://x.com"},
        ]
    }), encoding="utf-8")
    seeds = _load_seed_x_accounts(str(config))
    assert len(seeds) == 2
    assert seeds[0] == ("deepseek_ai", 1.4)  # sorted by weight desc


# ── YAML rendering ─────────────────────────────────────────────────────


def test_render_diffused_yaml():
    sources = [
        DiffusedSource(
            name="Test AI Blog", source_type="rss",
            url="https://example.com/feed.xml",
            overall_score=0.75, diffusion_method="content_link",
            link_count=5, reason="Linked from 5 items",
        ),
        DiffusedSource(
            name="Test KOL", source_type="x", username="test_kol",
            account_type="kol", overall_score=0.82,
            diffusion_method="social_graph", seed_overlap_count=3,
            reason="Followed by 3 seeds",
        ),
    ]
    yaml_str = _render_diffused_yaml(sources)
    assert "diffused_" in yaml_str
    assert "example.com/feed.xml" in yaml_str
    assert "test_kol" in yaml_str
    assert "[graph]" in yaml_str
    assert "[link]" in yaml_str


# ── Content link diffusion (offline) ───────────────────────────────────


@pytest.fixture
def tmp_config(tmp_path):
    import yaml
    config = tmp_path / "default.yaml"
    config.write_text(yaml.dump({
        "sources": [
            {"id": "openai", "type": "rss", "url": "https://openai.com/news/rss.xml"},
            {"id": "hf", "type": "rss", "url": "https://huggingface.co/blog/feed.xml"},
            {"id": "x_deepseek", "type": "x", "username": "deepseek_ai"},
        ]
    }), encoding="utf-8")
    return str(config)


def test_diffuse_content_links_real_data(tmp_config):
    """Use real collected URLs to verify link diffusion works end-to-end."""
    items = [
        RawItem(
            source_id="test", source_type="rss",
            title="AI breakthrough in robotics",
            url="https://venturebeat.com/ai/robot-breakthrough-2026",
            summary="Robots are getting smarter.", published_at="2026-05-10T00:00:00Z",
        ),
        RawItem(
            source_id="test", source_type="rss",
            title="New GPU for AI training",
            url="https://venturebeat.com/ai/nvidia-new-gpu-2026",
            summary="NVIDIA releases new GPU.", published_at="2026-05-10T00:00:00Z",
        ),
        RawItem(
            source_id="test", source_type="rss",
            title="Quantum computing meets AI",
            url="https://spectrum.ieee.org/quantum-ai-2026",
            summary="Quantum meets AI.", published_at="2026-05-10T00:00:00Z",
        ),
    ]
    result = diffuse_sources(
        config_path=tmp_config,
        collected_items=items,
    )
    assert "content_links" in result
    link_report = result["content_links"]
    # VentureBeat might or might not have a valid RSS feed at probed paths.
    # The key assertion: the system runs without errors and produces a report.
    assert link_report is not None
    assert link_report.method == "content_links"
    assert link_report.candidates_discovered >= 1  # at least venturebeat.com
    assert result["summary"]


def test_diffuse_content_links_skips_existing(tmp_config):
    """Domains already in config should be skipped."""
    items = [
        RawItem(
            source_id="test", source_type="rss",
            title="OpenAI releases GPT-5",
            url="https://openai.com/index/gpt-5",
            summary="GPT-5 is here.", published_at="2026-05-10T00:00:00Z",
        ),
    ]
    result = diffuse_sources(
        config_path=tmp_config,
        collected_items=items,
    )
    # openai.com is already in config, so no new domains to discover.
    link_report = result["content_links"]
    assert link_report.candidates_discovered == 0


def test_diffuse_without_collected_items(tmp_config):
    """Without collected items, link diffusion is skipped."""
    result = diffuse_sources(config_path=tmp_config, collected_items=None)
    assert result["content_links"] is None
    assert result["social_graph"] is None  # no X token set


# ── Feed scoring ────────────────────────────────────────────────────────


class _FakeEntry:
    def __init__(self, title, published_parsed=None):
        self.title = title
        self.published_parsed = published_parsed

    def get(self, key, default=None):
        return getattr(self, key, default)


def test_feed_ai_relevance():
    entries = [
        _FakeEntry("New AI model"),
        _FakeEntry("Deep learning tutorial"),
        _FakeEntry("Sports news"),
    ]
    assert _feed_ai_relevance(entries) == 2/3


def test_feed_ai_relevance_empty():
    assert _feed_ai_relevance([]) == 0.0


def test_feed_freshness():
    import time as _time
    from datetime import datetime, timezone
    recent = datetime.now(timezone.utc).timetuple()
    entries = [_FakeEntry("test", published_parsed=recent)]
    assert _feed_freshness(entries) == 1.0


def test_feed_freshness_empty():
    assert _feed_freshness([]) == 0.2
