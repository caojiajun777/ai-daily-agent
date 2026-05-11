import json
import os

import pytest
import yaml

from agent.pipelines.daily_report import run_pipeline



def test_pipeline_mock_run(
    tmp_path, monkeypatch, cfg, prompts, fake_raw_items, scripted_writer_provider
):
    # Bypass network: collector returns our fake items.
    monkeypatch.setattr(
        "agent.pipelines.daily_report.collect", lambda specs, **k: fake_raw_items
    )
    artifacts = tmp_path / "artifacts"
    report = run_pipeline(
        cfg=cfg,
        prompts=prompts,
        provider=scripted_writer_provider,
        artifacts_root=str(artifacts),
        date="2026-05-09",
    )
    assert report["date"] == "2026-05-09"
    assert report["is_failed"] is False
    # critique should pass deterministic checks for the scripted draft
    assert report["stages"]["critique"]["status"] == "ok"
    assert report["stages"]["publish"]["status"] == "ok"

    md_path = report["draft_path"]
    assert md_path and os.path.exists(md_path)
    md = open(md_path, encoding="utf-8").read()
    assert "AI 日报 2026-05-09" in md
    assert "https://hf.co/blog/dataset-x" in md

    trace_path = report["trace_path"]
    assert os.path.exists(trace_path)
    lines = open(trace_path, encoding="utf-8").read().strip().splitlines()
    events = [json.loads(l) for l in lines]
    kinds = {e["event"] for e in events}
    assert "run_start" in kinds
    assert "stage" in kinds
    assert "llm_call" in kinds
    assert "run_end" in kinds

    report_path = report["report_path"]
    assert os.path.exists(report_path)
    rep = json.load(open(report_path, encoding="utf-8"))
    assert rep["budget"]["calls_used"] >= 1
