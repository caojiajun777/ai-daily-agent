"""Tests for CLI eval command — sem dup + repair summary."""

from __future__ import annotations

import json
import sys
from io import StringIO
from typing import Any, Dict

import pytest

from agent.cli import cmd_eval
from agent.schemas import RepairReport, SemanticDuplicateReport, SemanticDuplicate


DATE = "2026-05-09"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _draft_payload() -> Dict[str, Any]:
    return {
        "date": DATE,
        "title": f"AI 日报 {DATE}",
        "sections": [
            {
                "heading": "要闻",
                "items": [{"title": "#1 A", "summary": "s", "url": "https://a.com/1", "source": "src"}],
            },
            {
                "heading": "研究",
                "items": [{"title": "#2 B", "summary": "s", "url": "https://a.com/2", "source": "src"}],
            },
            {
                "heading": "安全",
                "items": [{"title": "#3 C", "summary": "s", "url": "https://a.com/3", "source": "src"}],
            },
        ],
    }


def _seed_eval_artifacts(
    tmp_path,
    *,
    sem_dup_report: SemanticDuplicateReport | None = None,
    repair_report: RepairReport | None = None,
) -> str:
    root = tmp_path / "artifacts"
    (root / "drafts").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "curated").mkdir(parents=True, exist_ok=True)
    (root / "drafts" / f"{DATE}.json").write_text(
        json.dumps(_draft_payload(), ensure_ascii=False), encoding="utf-8"
    )
    if sem_dup_report is not None:
        (root / "reports" / f"semantic_duplicates_{DATE}.json").write_text(
            json.dumps(sem_dup_report.model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )
    if repair_report is not None:
        (root / "reports" / f"repair_{DATE}.json").write_text(
            json.dumps(repair_report.model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )
    return str(root)


class _Args:
    """Minimal argparse.Namespace stand-in."""

    def __init__(self, run_id: str, artifacts: str, config: str | None = None):
        self.run_id = run_id
        self.artifacts = artifacts
        self.config = config


# --------------------------------------------------------------------------- #
# 1. CLI eval includes semantic duplicate summary
# --------------------------------------------------------------------------- #


def test_cli_eval_includes_semantic_duplicate_summary(tmp_path, capsys):
    sem = SemanticDuplicateReport(
        date=DATE,
        run_id="r-1",
        duplicates=[
            SemanticDuplicate(
                item_a_id="#1",
                item_b_id="#3",
                item_a_title="#1 A",
                item_b_title="#3 C",
                reason="same topic",
                severity="low",
            )
        ],
        ok=True,
        checked_item_count=3,
        provider="mock",
    )
    artifacts = _seed_eval_artifacts(tmp_path, sem_dup_report=sem)
    args = _Args(run_id=DATE, artifacts=artifacts)
    rc = cmd_eval(args)
    captured = json.loads(capsys.readouterr().out)
    assert "semantic_duplicate_count" in captured
    assert captured["semantic_duplicate_count"] == 1
    assert "semantic_duplicate_ok" in captured
    assert captured["semantic_duplicate_ok"] is True


# --------------------------------------------------------------------------- #
# 2. CLI eval includes repair summary
# --------------------------------------------------------------------------- #


def test_cli_eval_includes_repair_summary(tmp_path, capsys):
    rep = RepairReport(
        date=DATE,
        run_id="r-1",
        attempted=True,
        succeeded=True,
        reason="repaired 1 item(s)",
        pre_duplicate_count=1,
        post_duplicate_count=0,
        draft_version="v2",
    )
    artifacts = _seed_eval_artifacts(tmp_path, repair_report=rep)
    args = _Args(run_id=DATE, artifacts=artifacts)
    cmd_eval(args)
    captured = json.loads(capsys.readouterr().out)
    assert captured.get("repair_attempted") is True
    assert captured.get("repair_succeeded") is True
    assert captured.get("repair_reason") == "repaired 1 item(s)"
    assert captured.get("draft_version") == "v2"


# --------------------------------------------------------------------------- #
# 3. Missing optional reports do not crash
# --------------------------------------------------------------------------- #


def test_cli_eval_missing_optional_reports_does_not_crash(tmp_path, capsys):
    # No sem dup file, no repair file — eval should complete normally.
    artifacts = _seed_eval_artifacts(tmp_path)
    args = _Args(run_id=DATE, artifacts=artifacts)
    rc = cmd_eval(args)
    captured = json.loads(capsys.readouterr().out)
    # Core metrics must be present; optional keys absent is fine.
    assert "section_count" in captured
    assert "hallucinated_urls" in captured
    # Must not have crashed (rc is 0 or 1 based on draft quality, not 2+).
    assert rc in (0, 1)
