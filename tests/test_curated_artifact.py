"""Tests for curated artifact persistence."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from agent.agents.curator import curate, curate_with_records
from agent.eval.metrics import deterministic_metrics
from agent.schemas import (
    CuratedItem,
    CuratedItemRecord,
    CuratedOutput,
    Draft,
    DraftItem,
    DraftSection,
)
from agent.sources.base import RawItem


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _raw(
    *,
    source_id: str = "src",
    title: str = "Test Title",
    url: str = "https://example.com/a",
    published_at: str = "2026-05-09T00:00:00Z",
) -> RawItem:
    return RawItem(
        source_id=source_id,
        source_type="rss",
        title=title,
        url=url,
        summary="summary",
        published_at=published_at,
    )


def _source_specs() -> List[Dict[str, Any]]:
    return [{"id": "src", "type": "rss", "url": "https://x.com/rss", "weight": 1.0}]


def _draft_with_urls(urls: List[str]) -> Draft:
    sections = []
    for i, url in enumerate(urls):
        sections.append(
            DraftSection(
                heading=f"Section {i+1}",
                items=[
                    DraftItem(
                        title=f"#{ i+1 } Title {i+1}",
                        summary="summary",
                        url=url,
                        source="src",
                    )
                ],
            )
        )
    return Draft(date="2026-05-09", title="AI 日报 2026-05-09", sections=sections)


# --------------------------------------------------------------------------- #
# 1. curate_with_records — schema validity
# --------------------------------------------------------------------------- #


def test_curated_artifact_schema_valid():
    items = [_raw(title=f"Title {i}", url=f"https://example.com/{i}") for i in range(5)]
    writer_items, records = curate_with_records(
        items, source_specs=_source_specs(), max_items=5
    )
    assert len(records) == 5
    for rec in records:
        # Validates against Pydantic schema
        assert isinstance(rec, CuratedItemRecord)
        assert rec.source_url.startswith("https://")
        assert rec.score >= 0.0
        assert rec.used_in_draft is True
        assert rec.section is None  # not yet back-filled
        assert rec.selected_reason  # non-empty

    # Round-trip through CuratedOutput
    output = CuratedOutput(date="2026-05-09", run_id="test-run", items=records)
    payload = output.model_dump(mode="json")
    revived = CuratedOutput.model_validate(payload)
    assert len(revived.items) == 5
    assert revived.date == "2026-05-09"


def test_curated_artifact_written(tmp_path, monkeypatch, cfg, prompts, fake_raw_items, scripted_writer_provider):
    """Pipeline writes artifacts/curated/<date>.json after a successful run."""
    monkeypatch.setattr(
        "agent.pipelines.daily_report.collect", lambda specs, **k: fake_raw_items
    )
    from agent.pipelines.daily_report import run_pipeline

    report = run_pipeline(
        cfg=cfg,
        prompts=prompts,
        provider=scripted_writer_provider,
        artifacts_root=str(tmp_path / "artifacts"),
        date="2026-05-09",
    )

    curated_path = report.get("curated_path")
    assert curated_path is not None
    assert (tmp_path / "artifacts" / "curated" / "2026-05-09.json").exists()

    with open(curated_path, encoding="utf-8") as f:
        output = CuratedOutput.model_validate(json.load(f))

    assert output.date == "2026-05-09"
    assert output.run_id == report["run_id"]
    assert len(output.items) > 0
    for rec in output.items:
        assert rec.source_url
        assert rec.raw_item_id
        assert rec.score >= 0.0
        assert rec.used_in_draft is True


# --------------------------------------------------------------------------- #
# 2. Section backfill
# --------------------------------------------------------------------------- #


def test_section_backfill_from_draft():
    """curate_with_records records get section back-filled by matching source_url."""
    urls = [f"https://example.com/{i}" for i in range(3)]
    items = [_raw(title=f"T{i}", url=urls[i]) for i in range(3)]

    writer_items, records = curate_with_records(
        items, source_specs=_source_specs(), max_items=3
    )
    # Simulate what daily_report.py does after write:
    section_names = ["要闻", "模型发布", "开发生态"]
    url_to_section = {urls[i]: section_names[i] for i in range(3)}
    for rec in records:
        rec.section = url_to_section.get(rec.source_url)

    by_url = {rec.source_url: rec for rec in records}
    for i, url in enumerate(urls):
        assert by_url[url].section == section_names[i]


def test_section_backfill_in_pipeline(tmp_path, monkeypatch, cfg, prompts, fake_raw_items, scripted_writer_provider):
    """Pipeline back-fills section on curated records for URLs used in the draft."""
    monkeypatch.setattr(
        "agent.pipelines.daily_report.collect", lambda specs, **k: fake_raw_items
    )
    from agent.pipelines.daily_report import run_pipeline

    report = run_pipeline(
        cfg=cfg,
        prompts=prompts,
        provider=scripted_writer_provider,
        artifacts_root=str(tmp_path / "artifacts"),
        date="2026-05-09",
    )
    curated_path = report["curated_path"]
    with open(curated_path, encoding="utf-8") as f:
        output = CuratedOutput.model_validate(json.load(f))

    # At least one record should have section back-filled (URLs overlap with draft).
    sections = [rec.section for rec in output.items if rec.section is not None]
    assert len(sections) > 0


# --------------------------------------------------------------------------- #
# 3. Eval uses curated records when provided
# --------------------------------------------------------------------------- #


def test_eval_uses_curated_urls():
    """When curated_records is passed, allowed_urls comes from source_url."""
    real_url = "https://real-source.com/article"
    records = [
        CuratedItemRecord(
            raw_item_id="src::https://real-source.com/article",
            title="Real Article",
            source_url=real_url,
            source_name="src",
            published_at=None,
            score=1.0,
            section=None,
            selected_reason="recency",
            duplicate_group_id=None,
            used_in_draft=True,
        )
    ]
    draft = _draft_with_urls([real_url])
    # curated list is empty on purpose — if records weren't used, hallucinated_urls > 0
    metrics = deterministic_metrics(
        draft=draft,
        curated=[],
        min_section_count=1,
        curated_records=records,
    )
    assert metrics["hallucinated_urls"] == 0
    assert metrics["used_curated_artifact"] is True


def test_hallucinated_url_detected_against_curated():
    """A URL in the draft that is NOT in curated_records is flagged."""
    allowed_url = "https://real.com/ok"
    hallucinated_url = "https://invented.com/bad"

    records = [
        CuratedItemRecord(
            raw_item_id="src::https://real.com/ok",
            title="OK Article",
            source_url=allowed_url,
            source_name="src",
            published_at=None,
            score=1.0,
            section=None,
            selected_reason="recency",
            duplicate_group_id=None,
            used_in_draft=True,
        )
    ]
    draft = _draft_with_urls([allowed_url, hallucinated_url])
    metrics = deterministic_metrics(
        draft=draft,
        curated=[],
        min_section_count=1,
        curated_records=records,
    )
    assert metrics["hallucinated_urls"] == 1
    assert "hallucinated_urls_present" in metrics["issues"]
    assert metrics["used_curated_artifact"] is True


def test_eval_fallback_without_curated_records():
    """Without curated_records, eval falls back to curated list, used_curated_artifact=False."""
    url = "https://example.com/x"
    curated = [
        CuratedItem(
            title="X", url=url, summary="s", source="src",
            source_type="rss", published_at="", score=1.0,
        )
    ]
    draft = _draft_with_urls([url])
    metrics = deterministic_metrics(
        draft=draft, curated=curated, min_section_count=1
    )
    assert metrics["hallucinated_urls"] == 0
    assert metrics["used_curated_artifact"] is False


def test_eval_flags_hard_truncated_titles():
    draft = Draft(
        date="2026-05-09",
        title="T",
        sections=[
            DraftSection(heading="要闻", items=[
                DraftItem(
                    title="#1 Google I/O 发布 Gemini 3.5 和 Antigravit",
                    summary="summary",
                    url="https://example.com/io",
                    source="src",
                ),
                DraftItem(
                    title="#2 阿里 Qwen3.7-Max 匹配 Claude Op",
                    summary="summary",
                    url="https://example.com/qwen",
                    source="src",
                ),
            ])
        ],
    )

    metrics = deterministic_metrics(draft=draft, curated=[], min_section_count=1)

    assert metrics["truncated_title_count"] == 2
    assert "truncated_titles_present" in metrics["issues"]
    assert metrics["ok"] is False


# --------------------------------------------------------------------------- #
# 4. Dedup and raw_item_id
# --------------------------------------------------------------------------- #


def test_curator_dedup_assigns_raw_item_id():
    """Each record carries raw_item_id = '<source_id>::<url>'."""
    items = [
        _raw(source_id="hf", title="T1", url="https://hf.co/1"),
        _raw(source_id="hf", title="T1", url="https://hf.co/1"),  # exact dup
        _raw(source_id="oai", title="T2", url="https://oai.com/2"),
    ]
    _, records = curate_with_records(items, source_specs=[
        {"id": "hf", "type": "rss", "url": "x", "weight": 1.0},
        {"id": "oai", "type": "rss", "url": "x", "weight": 1.0},
    ], max_items=10)
    assert len(records) == 2
    ids = {rec.raw_item_id for rec in records}
    assert "hf::https://hf.co/1" in ids
    assert "oai::https://oai.com/2" in ids


def test_curate_backward_compatible():
    """curate() (original API) still returns only List[CuratedItem]."""
    items = [_raw(title=f"T{i}", url=f"https://x.com/{i}") for i in range(3)]
    result = curate(items, source_specs=_source_specs(), max_items=3)
    assert all(isinstance(r, CuratedItem) for r in result)
    assert len(result) == 3


# --------------------------------------------------------------------------- #
# 5. pipeline_mock_run: curated_path present in report
# --------------------------------------------------------------------------- #


def test_pipeline_mock_run_curated_path(
    tmp_path, monkeypatch, cfg, prompts, fake_raw_items, scripted_writer_provider
):
    monkeypatch.setattr(
        "agent.pipelines.daily_report.collect", lambda specs, **k: fake_raw_items
    )
    from agent.pipelines.daily_report import run_pipeline

    report = run_pipeline(
        cfg=cfg,
        prompts=prompts,
        provider=scripted_writer_provider,
        artifacts_root=str(tmp_path / "artifacts"),
        date="2026-05-09",
    )
    assert report.get("curated_path") is not None
    assert (tmp_path / "artifacts" / "curated" / "2026-05-09.json").exists()

    # Confirm report JSON on disk also has curated_path
    saved = json.load(open(report["report_path"], encoding="utf-8"))
    assert saved.get("curated_path") is not None
