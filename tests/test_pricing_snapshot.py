"""Tests for Pricing Snapshot Adapter v1."""

import json
import os
import tempfile

from agent.schemas import (
    PricingChange,
    PricingDiff,
    PricingModelRecord,
    PricingProviderSnapshot,
    PricingSnapshot,
)
from agent.sources.pricing_snapshot import (
    PricingSnapshotAdapter,
    _compute_content_hash,
    _compute_diff,
    _from_static_config,
    pricing_diff_to_candidates,
    snapshot_pricing,
)


# ── Schema tests ──────────────────────────────────────────────────────


def test_pricing_model_record_schema():
    rec = PricingModelRecord(
        provider="deepseek", model="deepseek-chat",
        input_price_per_m=0.27, output_price_per_m=1.10,
        currency="USD", source_url="https://example.com", observed_at="2026-01-01",
    )
    assert rec.provider == "deepseek"
    assert rec.input_price_per_m == 0.27
    assert rec.output_price_per_m == 1.10


def test_pricing_snapshot_schema():
    snap = PricingSnapshot(
        date="2026-05-09", run_id="test",
        providers=[PricingProviderSnapshot(
            provider="deepseek", source_id="d1", source_url="https://x.com",
            observed_at="2026-01-01", models=[], content_hash="abc",
        )],
    )
    assert len(snap.providers) == 1


def test_pricing_diff_schema():
    diff = PricingDiff(
        date="2026-05-09", run_id="test", previous_date="2026-05-08",
        has_changes=True,
        changes=[PricingChange(
            provider="oai", model="gpt-5", field="input_price_per_m",
            old=1.0, new=0.5, change_type="price_decrease",
            source_url="https://x.com",
        )],
    )
    assert diff.has_changes
    assert len(diff.changes) == 1


# ── Adapter tests ──────────────────────────────────────────────────────


def test_static_config_pricing_snapshot():
    """Static config mode produces valid snapshot."""
    spec = {
        "id": "deepseek_pricing",
        "parser_strategy": "static_config",
        "enabled": True,
        "provider": "deepseek",
        "source_url": "https://api-docs.deepseek.com/pricing",
        "pricing_records": [
            {"model": "deepseek-chat", "input_price_per_m": 0.27, "currency": "USD"},
            {"model": "deepseek-v4-pro", "input_price_per_m": 0.14, "output_price_per_m": 0.28, "currency": "USD"},
        ],
    }
    specs = [spec]
    with tempfile.TemporaryDirectory() as d:
        snap, diff = snapshot_pricing(
            source_specs=specs, artifacts_root=d,
            date="2026-05-09", run_id="test",
        )
    assert len(snap.providers) == 1
    assert snap.providers[0].provider == "deepseek"
    assert len(snap.providers[0].models) == 2
    assert snap.providers[0].content_hash
    # No previous snapshot => diff with no changes
    assert diff is not None
    assert diff.previous_date is None
    assert not diff.has_changes


def test_featured_pricing_candidate_emits_without_static_candidates():
    spec = {
        "id": "deepseek_pricing",
        "type": "pricing_snapshot",
        "provider": "deepseek",
        "source_url": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
        "content_type": "china_model_pricing",
        "source_tier": "tier_0_core_evidence",
        "reliability": "high",
        "evidence_type": "pricing_page",
        "default_confidence": "high",
        "featured_candidates": [{
            "title": "DeepSeek-V4-Pro API 2.5 折优惠转为永久正式定价",
            "summary": "官方定价页显示优惠转为永久正式定价。",
            "candidate_until": "2099-01-01",
        }],
        "pricing_records": [{
            "model": "deepseek-v4-pro",
            "input_price_per_m": 3,
            "currency": "CNY",
        }],
    }
    adapter = PricingSnapshotAdapter(spec)
    items = adapter.fetch(max_items=5)
    assert len(items) == 1
    assert items[0].title.startswith("DeepSeek-V4-Pro")
    assert items[0].url.endswith("/pricing")
    assert items[0].evidence_type == "pricing_page"


def test_featured_pricing_candidate_expires():
    spec = {
        "id": "old_pricing",
        "type": "pricing_snapshot",
        "source_url": "https://example.com/pricing",
        "featured_candidates": [{
            "title": "Old pricing news",
            "candidate_until": "2000-01-01",
        }],
    }
    adapter = PricingSnapshotAdapter(spec)
    assert adapter.fetch(max_items=5) == []


def test_pricing_snapshot_content_hash_stable():
    """Same records produce same hash."""
    records = [{"model": "gpt-5", "input_price_per_m": 1.0}]
    raw1 = json.dumps(records, sort_keys=True)
    raw2 = json.dumps(records, sort_keys=True)
    assert _compute_content_hash(raw1) == _compute_content_hash(raw2)


def test_pricing_snapshot_handles_no_sources():
    """No pricing sources should not crash."""
    with tempfile.TemporaryDirectory() as d:
        snap, diff = snapshot_pricing(
            source_specs=[], artifacts_root=d,
            date="2026-05-09", run_id="test",
        )
    assert len(snap.providers) == 0
    assert diff is not None
    assert not diff.has_changes


# ── Diff tests ─────────────────────────────────────────────────────────


def _make_snap(provider: str, model: str, input_p: float, date: str):
    return PricingSnapshot(
        date=date, run_id="test",
        providers=[PricingProviderSnapshot(
            provider=provider, source_id="s1", source_url="https://x.com",
            observed_at=date, content_hash="abc",
            models=[PricingModelRecord(
                provider=provider, model=model,
                input_price_per_m=input_p, output_price_per_m=None,
                source_url="https://x.com", observed_at=date,
            )],
        )],
    )


def test_pricing_diff_detects_price_decrease():
    with tempfile.TemporaryDirectory() as d:
        # Write previous snapshot
        prev = _make_snap("oai", "gpt-5", 1.0, "2026-05-08")
        os.makedirs(os.path.join(d, "pricing"), exist_ok=True)
        with open(os.path.join(d, "pricing", "2026-05-08.json"), "w") as f:
            json.dump(prev.model_dump(), f)

        curr = _make_snap("oai", "gpt-5", 0.5, "2026-05-09")
        diff = _compute_diff(curr, d, "2026-05-09", "test")

    assert diff is not None
    assert diff.has_changes
    assert len(diff.changes) == 1
    assert diff.changes[0].change_type == "price_decrease"
    assert diff.changes[0].old == 1.0
    assert diff.changes[0].new == 0.5


def test_pricing_diff_detects_price_increase():
    with tempfile.TemporaryDirectory() as d:
        prev = _make_snap("oai", "gpt-5", 1.0, "2026-05-08")
        os.makedirs(os.path.join(d, "pricing"), exist_ok=True)
        with open(os.path.join(d, "pricing", "2026-05-08.json"), "w") as f:
            json.dump(prev.model_dump(), f)

        curr = _make_snap("oai", "gpt-5", 2.0, "2026-05-09")
        diff = _compute_diff(curr, d, "2026-05-09", "test")

    assert diff.has_changes
    assert diff.changes[0].change_type == "price_increase"


def test_pricing_diff_detects_new_model():
    with tempfile.TemporaryDirectory() as d:
        prev = _make_snap("oai", "gpt-4", 1.0, "2026-05-08")
        os.makedirs(os.path.join(d, "pricing"), exist_ok=True)
        with open(os.path.join(d, "pricing", "2026-05-08.json"), "w") as f:
            json.dump(prev.model_dump(), f)

        curr = _make_snap("oai", "gpt-5", 1.0, "2026-05-09")
        diff = _compute_diff(curr, d, "2026-05-09", "test")

    assert diff.has_changes
    assert diff.changes[0].change_type == "new_model"


def test_pricing_diff_no_previous_snapshot_no_changes():
    with tempfile.TemporaryDirectory() as d:
        curr = _make_snap("oai", "gpt-5", 1.0, "2026-05-09")
        diff = _compute_diff(curr, d, "2026-05-09", "test")
    assert not diff.has_changes
    assert diff.previous_date is None


def test_pricing_diff_no_change():
    with tempfile.TemporaryDirectory() as d:
        prev = _make_snap("oai", "gpt-5", 1.0, "2026-05-08")
        os.makedirs(os.path.join(d, "pricing"), exist_ok=True)
        with open(os.path.join(d, "pricing", "2026-05-08.json"), "w") as f:
            json.dump(prev.model_dump(), f)

        curr = _make_snap("oai", "gpt-5", 1.0, "2026-05-09")
        diff = _compute_diff(curr, d, "2026-05-09", "test")

    assert not diff.has_changes
    assert len(diff.changes) == 0


# ── Candidate generation ───────────────────────────────────────────────


def test_pricing_change_generates_candidate():
    diff = PricingDiff(
        date="2026-05-09", run_id="test", previous_date="2026-05-08",
        has_changes=True,
        changes=[PricingChange(
            provider="deepseek", model="v4-pro", field="output_price_per_m",
            old=0.50, new=0.28, change_type="price_decrease",
            source_url="https://example.com",
        )],
    )
    candidates = pricing_diff_to_candidates(diff)
    assert len(candidates) == 1
    c = candidates[0]
    assert c["content_type"] == "pricing_page"
    assert c["source_tier"] == "tier_0_core_evidence"
    assert c["confidence"] == "high"
    assert "decreased" in c["summary"]


def test_pricing_unchanged_does_not_generate_candidate():
    diff = PricingDiff(
        date="2026-05-09", run_id="test", previous_date="2026-05-08",
        has_changes=False, changes=[],
    )
    candidates = pricing_diff_to_candidates(diff)
    assert len(candidates) == 0


# ── Token budget ───────────────────────────────────────────────────────


def test_pricing_full_page_not_sent_to_llm():
    """Pricing candidates are compact summaries, not full pages."""
    diff = PricingDiff(
        date="2026-05-09", run_id="test", has_changes=True,
        changes=[PricingChange(
            provider="deepseek", model="v4-pro", field="input_price_per_m",
            old=0.14, new=0.07, change_type="price_decrease",
            source_url="https://example.com",
        )],
    )
    candidates = pricing_diff_to_candidates(diff)
    for c in candidates:
        # Summary must be under 500 chars (compact, not full page)
        assert len(c["summary"]) < 500
        # Must not contain raw HTML or long pricing tables
        assert "<table" not in c["summary"].lower()
        assert "pricing_records" not in c["summary"]


def test_pricing_diff_only_sent_to_writer():
    """Only diff changes generate candidates, not the full snapshot."""
    # Full snapshot with 5 models
    snap = PricingSnapshot(
        date="2026-05-09", run_id="test",
        providers=[PricingProviderSnapshot(
            provider="deepseek", source_id="d1", source_url="https://x.com",
            observed_at="2026-05-09", content_hash="abc",
            models=[PricingModelRecord(
                provider="deepseek", model=f"m{i}",
                input_price_per_m=0.1 * i, source_url="https://x.com",
                observed_at="2026-05-09",
            ) for i in range(1, 6)],
        )],
    )
    # But diff has only 1 change
    diff = PricingDiff(
        date="2026-05-09", run_id="test", previous_date="2026-05-08",
        has_changes=True,
        changes=[PricingChange(
            provider="deepseek", model="m1", field="input_price_per_m",
            old=0.10, new=0.05, change_type="price_decrease",
            source_url="https://x.com",
        )],
    )
    candidates = pricing_diff_to_candidates(diff)
    # Only 1 candidate from diff, not 5 from full snapshot
    assert len(candidates) == 1


# ── CLI ────────────────────────────────────────────────────────────────


def test_cli_pricing_snapshot_json_output(tmp_path):
    """CLI --json mode produces valid JSON output."""
    import subprocess, sys
    # Write a minimal config with pricing source
    config = tmp_path / "config.yaml"
    import yaml
    config.write_text(yaml.dump({
        "sources": [{
            "id": "deepseek_pricing",
            "type": "pricing_snapshot",
            "enabled": True,
            "parser_strategy": "static_config",
            "provider": "deepseek",
            "source_url": "https://example.com",
            "pricing_records": [
                {"model": "deepseek-chat", "input_price_per_m": 0.27, "currency": "USD"},
            ],
        }],
    }), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "agent.cli", "pricing-snapshot",
         "--date", "2026-05-09", "--config", str(config),
         "--artifacts", str(tmp_path), "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["providers_count"] == 1
    assert data["models_count"] == 1
    assert data["has_changes"] is False


def test_cli_pricing_snapshot_missing_sources_does_not_crash(tmp_path):
    """CLI with no pricing sources should exit cleanly."""
    import subprocess, sys
    config = tmp_path / "config.yaml"
    import yaml
    config.write_text(yaml.dump({"sources": []}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "agent.cli", "pricing-snapshot",
         "--date", "2026-05-09", "--config", str(config),
         "--artifacts", str(tmp_path), "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
