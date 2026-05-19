"""Tests for Trend Intelligence Layer."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta

import pytest

from agent.schemas import (
    TrendEvidence, TrendFinding, TrendReport, HeatChange,
)
from agent.tools.trend_metrics import (
    compute_event_metrics, compute_trend_signals, tag_event, summarize_tags,
)
from agent.agents.trend_validator import validate_report, metrics_only_report


# ── Schema tests ────────────────────────────────────────────────────────────

def test_trend_finding_validation():
    f = TrendFinding(
        trend_id="test_001", editorial_title="Test Trend",
        trend_type="topic", direction="rising", confidence="high",
        evidence_event_ids=["evt_a", "evt_b", "evt_c"],
    )
    assert f.trend_id == "test_001"
    assert f.evidence_event_ids == ["evt_a", "evt_b", "evt_c"]


def test_trend_report_validation():
    r = TrendReport(
        report_id="rpt_test", generated_at="2026-05-15T00:00:00Z",
        days=7, start_date="2026-05-08", end_date="2026-05-15",
        headline_summary="Test summary.",
        findings=[TrendFinding(trend_id="f1", evidence_event_ids=["e1", "e2"])],
    )
    assert r.total_findings == 0  # not auto-computed
    assert len(r.findings) == 1


# ── Metrics tests ───────────────────────────────────────────────────────────

def make_events() -> list:
    return [
        {"event_id": "e1", "date": "2026-05-10", "title": "GPT-5 release",
         "source_names": ["openai"], "urls": ["https://openai.com"], "section": "要闻",
         "priority": "must_include", "evidence_level": "official", "novelty": "new_event"},
        {"event_id": "e2", "date": "2026-05-11", "title": "Claude model update",
         "source_names": ["anthropic"], "urls": ["https://anthropic.com"], "section": "模型发布",
         "priority": "high", "evidence_level": "official", "novelty": "new_event"},
        {"event_id": "e3", "date": "2026-05-12", "title": "AI safety paper",
         "source_names": ["arxiv"], "urls": ["https://arxiv.org"], "section": "技术与洞察",
         "priority": "medium", "evidence_level": "primary", "novelty": "new_event"},
        {"event_id": "e4", "date": "2026-05-13", "title": "Repost of GPT-5 news",
         "source_names": ["venturebeat"], "urls": ["https://vb.com"], "section": "行业动态",
         "priority": "low", "evidence_level": "social", "novelty": "repeated_without_update"},
    ]


def test_compute_event_metrics():
    events = make_events()
    result = compute_event_metrics(events, final_selected={"e1", "e2"})
    assert len(result) == 4
    # Selected events get 1.5x boost.
    assert result[0]["_is_selected"] is True
    assert result[0]["_weighted_attention"] > 0
    assert result[3]["_is_selected"] is False


def test_compute_trend_signals():
    events = compute_event_metrics(make_events(), final_selected={"e1", "e2", "e3"})
    signals = compute_trend_signals(events, window_days=7)
    assert signals["event_count"] == 4
    assert signals["active_days"] == 4
    assert -1.0 <= signals["momentum"] <= 1.0
    assert 0.0 <= signals["persistence"] <= 1.0
    assert 0.0 <= signals["source_diversity"] <= 1.0


def test_momentum_rising():
    """Events concentrated in later days should have positive momentum."""
    events = compute_event_metrics([
        {"event_id": "e1", "date": "2026-05-12", "priority": "high",
         "source_names": ["a"], "urls": ["http://a.com"],
         "evidence_level": "official", "novelty": "new_event"},
        {"event_id": "e2", "date": "2026-05-14", "priority": "high",
         "source_names": ["b"], "urls": ["http://b.com"],
         "evidence_level": "official", "novelty": "new_event"},
        {"event_id": "e3", "date": "2026-05-14", "priority": "high",
         "source_names": ["c"], "urls": ["http://c.com"],
         "evidence_level": "official", "novelty": "new_event"},
    ])
    signals = compute_trend_signals(events, window_days=7)
    # More events in recent half → positive momentum.
    assert signals["momentum"] >= 0


def test_source_diversity_low():
    """Single-source groups have low diversity."""
    events = compute_event_metrics([
        {"event_id": f"e{i}", "date": "2026-05-1{i}", "priority": "high",
         "source_names": ["samesource"], "urls": [f"http://samesource.com/{i}"],
         "evidence_level": "social", "novelty": "repeated_without_update"}
        for i in range(5)
    ])
    signals = compute_trend_signals(events, window_days=7)
    assert signals["source_diversity"] < 0.3


# ── Taxonomy tests ──────────────────────────────────────────────────────────

def test_tag_event_model():
    tags = tag_event(title="GPT-5 reasoning model released", summary="new reasoning capabilities")
    assert "model_reasoning" in tags


def test_tag_event_agent():
    tags = tag_event(title="New coding agent with Claude Code integration")
    assert "agent_coding" in tags


def test_tag_event_funding():
    tags = tag_event(title="AI startup raises $500M Series C funding")
    assert "business_funding" in tags


def test_tag_event_paper():
    tags = tag_event(title="New benchmark SOTA on MMLU and HumanEval", summary="arxiv paper accepted at NeurIPS")
    assert "research_benchmark" in tags


def test_tag_event_empty():
    tags = tag_event(title="some random text with no keywords")
    assert tags == []


def test_summarize_tags():
    events = [
        {"_tags": ["model_reasoning", "agent_coding"]},
        {"_tags": ["model_reasoning", "business_funding"]},
        {"_tags": ["agent_coding"]},
    ]
    counts = summarize_tags(events)
    assert counts["model_reasoning"] == 2
    assert counts["agent_coding"] == 2
    assert counts["business_funding"] == 1


# ── Validator tests ─────────────────────────────────────────────────────────

def test_high_confidence_downgraded_with_few_events():
    f = TrendFinding(
        trend_id="t1", confidence="high", evidence_event_ids=["e1", "e2"],
    )
    f, warnings = _validate_wrapper(f, {"e1", "e2"}, 7, {"event_count": 2, "active_days": 2})
    assert f.confidence == "medium"
    assert len(warnings) >= 1


def test_single_day_spike_downgraded():
    f = TrendFinding(
        trend_id="t2", confidence="high", evidence_event_ids=["e1"],
    )
    f, warnings = _validate_wrapper(f, {"e1"}, 7, {"event_count": 1, "active_days": 1})
    # Downgraded due to insufficient events OR single-day spike.
    assert f.confidence == "medium"


def test_short_window_no_structural():
    f = TrendFinding(
        trend_id="t3", confidence="medium", evidence_event_ids=["e1"],
        window_type="confirmed_trend",
    )
    f, warnings = _validate_wrapper(f, {"e1"}, 4, {"event_count": 1, "active_days": 1})
    assert f.window_type == "short_signal"


def test_low_source_diversity_downgraded():
    f = TrendFinding(
        trend_id="t4", confidence="high", evidence_event_ids=["e1", "e2", "e3"],
    )
    f, warnings = _validate_wrapper(f, {"e1", "e2", "e3"}, 7, {
        "event_count": 3, "active_days": 3, "source_diversity": 0.2,
    })
    assert f.confidence == "medium"


def test_invalid_event_ids_removed():
    f = TrendFinding(
        trend_id="t5", confidence="medium",
        evidence_event_ids=["e1", "e_nonexistent", "e3"],
    )
    f, warnings = _validate_wrapper(f, {"e1", "e3"}, 7, {"event_count": 3, "active_days": 2})
    assert len(f.evidence_event_ids) == 2
    assert "e_nonexistent" not in f.evidence_event_ids


def test_novelty_low_rising_trend_warned():
    f = TrendFinding(
        trend_id="t6", direction="rising", confidence="high",
        evidence_event_ids=["e1", "e2", "e3"],
    )
    f, warnings = _validate_wrapper(f, {"e1", "e2", "e3"}, 7, {
        "event_count": 3, "active_days": 3, "novelty_ratio": 0.1,
    })
    assert "Low novelty" in (f.risk_of_overinterpretation or "")
    assert f.confidence == "medium"


def _validate_wrapper(f, valid_ids, window_days, metrics):
    from agent.agents.trend_validator import _validate_finding
    return _validate_finding(f, valid_ids, window_days, metrics)


# ── Metrics-only fallback ───────────────────────────────────────────────────

def test_metrics_only_report():
    r = metrics_only_report(days=7, start_date="2026-05-08", end_date="2026-05-15", total_events=50)
    assert r.metrics_fallback_used is True
    assert "LLM skipped" in r.headline_summary
    assert r.findings == []


# ── CLI smoke test (mock) ───────────────────────────────────────────────────

@pytest.fixture
def trend_seed(tmp_path):
    """Create minimal curated artifacts for trend testing."""
    curated_dir = tmp_path / "artifacts" / "curated"
    curated_dir.mkdir(parents=True)
    for i in range(3):
        date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
        item = {
            "date": date, "items": [{
                "title": f"AI event on {date}",
                "source_name": f"source_{i}", "source_url": f"https://example.com/{i}",
                "section": "要闻" if i == 0 else "模型发布",
                "score": 0.9 - i * 0.1,
            } for _ in range(3)],
        }
        with open(curated_dir / f"{date}.json", "w", encoding="utf-8") as f:
            json.dump(item, f)
    return str(tmp_path / "artifacts")


def test_trend_cli_mock(trend_seed, monkeypatch):
    """Trend analysis runs without real LLM."""
    from agent.agents.trend_analyzer import analyze_trends
    from agent.llm.mock_provider import MockLLMProvider

    def responder(msgs):
        return json.dumps({
            "headline_summary": "Mock trend summary.",
            "findings": [{
                "trend_id": "trend_test_001",
                "editorial_title": "Mock Trend",
                "trend_type": "topic",
                "direction": "rising",
                "confidence": "high",
                "window_type": "weekly_trend",
                "summary": "A test trend.",
                "evidence_event_ids": [],
                "companies_to_watch": ["TestCorp"],
                "why_it_matters": "Testing.",
                "implications": "",
                "counter_signals": "",
                "risk_of_overinterpretation": "",
                "what_to_watch_next": "",
            }],
            "heat_changes": [],
            "weak_signals": [],
            "noise_or_hype": [],
            "next_week_watchlist": ["TestCorp"],
        }, ensure_ascii=False)

    provider = MockLLMProvider(model="mock-trends", responder=responder)
    r = analyze_trends(provider=provider, artifacts_dir=trend_seed, days=7)
    assert r["ok"] is True
    assert r["findings"] >= 0
    assert "paths" in r
