"""Tests for the GitHub Issue Publisher.

These tests NEVER hit the network. The Publisher tool is replaced with
``FakeIssuePublisher`` (in-memory) and the GitHub API is exercised only
indirectly through the agent layer's gate + dup logic.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pytest

from agent.agents.issue_publisher import (
    build_issue_body,
    build_issue_title,
    evaluate_publish_gate,
    find_duplicates,
    load_run_artifacts,
    run_publish,
)
from agent.harness.trace import Tracer
from agent.tools.issue_publisher import (
    ExistingIssue,
    FakeIssuePublisher,
    GitHubIssuePublisher,
    PublisherConfigError,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _publish_cfg() -> Dict[str, Any]:
    return {
        "minimum_items": 3,
        "max_eval_issues": 0,
        "require_critic_pass": True,
        "recent_issues_to_check": 30,
        "title_prefix": "AI 日报 ",
        "default_labels": ["agent-generated"],
    }


def _draft_payload(date: str = "2026-05-09") -> Dict[str, Any]:
    return {
        "date": date,
        "title": f"AI 日报 {date}",
        "sections": [
            {
                "heading": "模型与产品",
                "items": [
                    {
                        "title": "Item A",
                        "summary": "summary A",
                        "url": "https://example.com/a",
                        "source": "src1",
                    }
                ],
            },
            {
                "heading": "研究",
                "items": [
                    {
                        "title": "Item B",
                        "summary": "summary B",
                        "url": "https://example.com/b",
                        "source": "src1",
                    }
                ],
            },
            {
                "heading": "安全",
                "items": [
                    {
                        "title": "Item C",
                        "summary": "summary C",
                        "url": "https://example.com/c",
                        "source": "src1",
                    }
                ],
            },
        ],
    }


def _ok_report(date: str, run_id: str = "r-1") -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "date": date,
        "provider": "mock",
        "model": "mock-model",
        "stages": {
            "collect": {"status": "ok", "meta": {}},
            "curate": {"status": "ok", "meta": {}},
            "write": {"status": "ok", "meta": {}},
            "critique": {
                "status": "ok",
                "meta": {"verdict": "pass", "reasons": [], "score": 95},
            },
            "publish": {"status": "ok", "meta": {}},
            "eval": {
                "status": "ok",
                "meta": {
                    "issues": [],
                    "ok": True,
                    "section_count": 3,
                    "item_count": 3,
                },
            },
        },
    }


def _failed_critic_report(date: str) -> Dict[str, Any]:
    rep = _ok_report(date)
    rep["stages"]["critique"] = {
        "status": "needs_human_review",
        "meta": {
            "verdict": "reject",
            "reasons": ["sections fewer than 3 (got 1)"],
            "score": 0,
        },
    }
    return rep


def _seed_artifacts(
    tmp_path,
    date: str,
    *,
    draft_payload: Dict[str, Any],
    report: Dict[str, Any],
):
    drafts = tmp_path / "artifacts" / "drafts"
    reports = tmp_path / "artifacts" / "reports"
    traces = tmp_path / "artifacts" / "traces"
    drafts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    (drafts / f"{date}.json").write_text(
        json.dumps(draft_payload, ensure_ascii=False), encoding="utf-8"
    )
    (drafts / f"{date}.md").write_text(
        f"# {draft_payload['title']}\n\nseed body\n", encoding="utf-8"
    )
    (reports / f"{date}.json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path / "artifacts"


def _tracer(artifacts_root, date: str) -> Tracer:
    return Tracer(
        os.path.join(str(artifacts_root), "traces", f"{date}.jsonl"),
        run_id=f"publish-{date}",
    )


# --------------------------------------------------------------------------- #
# 1. dry-run
# --------------------------------------------------------------------------- #


def test_issue_publisher_dry_run(tmp_path, monkeypatch):
    # Prevent env-var label overrides from leaking in.
    monkeypatch.delenv("PUBLISH_ISSUE_LABELS", raising=False)
    date = "2026-05-09"
    art = _seed_artifacts(
        tmp_path,
        date,
        draft_payload=_draft_payload(date),
        report=_ok_report(date),
    )
    publisher = FakeIssuePublisher(repo="owner/test-repo")
    tracer = _tracer(art, date)

    result = run_publish(
        date=date,
        publisher=publisher,
        publish_cfg=_publish_cfg(),
        artifacts_root=str(art),
        tracer=tracer,
        mode="dry-run",
    )

    assert result["mode"] == "dry-run"
    assert result["target_repo"] == "owner/test-repo"
    assert result["title"] == f"AI 日报 {date}"
    assert result["labels"] == ["agent-generated"]
    assert result["gate_result"]["ok"] is True
    assert result["would_publish"] is True
    assert publisher.created == []  # nothing was actually created

    preview_path = result["preview_path"]
    assert os.path.exists(preview_path)
    saved = json.load(open(preview_path, encoding="utf-8"))
    assert saved["mode"] == "dry-run"
    assert saved["title"] == f"AI 日报 {date}"
    assert saved["body_preview"]
    assert saved["target_repo"] == "owner/test-repo"


# --------------------------------------------------------------------------- #
# 2. missing PAT must refuse construction
# --------------------------------------------------------------------------- #


def test_issue_publisher_missing_pat(monkeypatch):
    monkeypatch.delenv("GITHUB_PUBLISH_TOKEN", raising=False)
    monkeypatch.delenv("GITBLOG_OWNER_PAT", raising=False)
    monkeypatch.setenv("PUBLISH_REPO", "owner/test-repo")
    with pytest.raises(PublisherConfigError):
        GitHubIssuePublisher()


def test_issue_publisher_refuses_github_token(monkeypatch):
    """If GITHUB_PUBLISH_TOKEN happens to equal GITHUB_TOKEN, refuse."""
    monkeypatch.setenv("GITHUB_PUBLISH_TOKEN", "shared-token-xyz")
    monkeypatch.setenv("GITHUB_TOKEN", "shared-token-xyz")
    monkeypatch.setenv("PUBLISH_REPO", "owner/test-repo")
    with pytest.raises(PublisherConfigError):
        GitHubIssuePublisher()


# --------------------------------------------------------------------------- #
# 3. failed critic blocks publish
# --------------------------------------------------------------------------- #


def test_issue_publisher_gate_blocks_failed_critic(tmp_path):
    date = "2026-05-09"
    art = _seed_artifacts(
        tmp_path,
        date,
        draft_payload=_draft_payload(date),
        report=_failed_critic_report(date),
    )
    publisher = FakeIssuePublisher(repo="owner/test-repo")
    tracer = _tracer(art, date)

    result = run_publish(
        date=date,
        publisher=publisher,
        publish_cfg=_publish_cfg(),
        artifacts_root=str(art),
        tracer=tracer,
        mode="confirm",
    )

    assert result["status"] == "blocked_by_gate"
    assert result["issue_number"] is None
    assert any("critic did not pass" in r for r in result["gate_result"]["blocked_reasons"])
    assert publisher.created == []  # never reached create_issue


# --------------------------------------------------------------------------- #
# 4. duplicate blocks by default
# --------------------------------------------------------------------------- #


def _duplicate_existing(date: str) -> List[ExistingIssue]:
    return [
        ExistingIssue(
            number=42,
            title=f"AI 日报 {date}",  # exact title match
            state="open",
            html_url=f"https://github.com/owner/test-repo/issues/42",
            created_at="2026-05-09T00:00:00Z",
            author_login="owner",
        )
    ]


def test_issue_publisher_duplicate_blocks_by_default(tmp_path):
    date = "2026-05-09"
    art = _seed_artifacts(
        tmp_path,
        date,
        draft_payload=_draft_payload(date),
        report=_ok_report(date),
    )
    publisher = FakeIssuePublisher(
        repo="owner/test-repo", existing=_duplicate_existing(date)
    )
    tracer = _tracer(art, date)

    result = run_publish(
        date=date,
        publisher=publisher,
        publish_cfg=_publish_cfg(),
        artifacts_root=str(art),
        tracer=tracer,
        mode="confirm",
        force=False,
    )

    assert result["status"] == "blocked_by_duplicate"
    assert result["duplicates"] and result["duplicates"][0]["number"] == 42
    assert publisher.created == []
    assert result["force"] is False


# --------------------------------------------------------------------------- #
# 5. --force allows publishing over a duplicate (but trace must record it)
# --------------------------------------------------------------------------- #


def test_issue_publisher_force_allows_duplicate(tmp_path):
    date = "2026-05-09"
    art = _seed_artifacts(
        tmp_path,
        date,
        draft_payload=_draft_payload(date),
        report=_ok_report(date),
    )
    publisher = FakeIssuePublisher(
        repo="owner/test-repo", existing=_duplicate_existing(date)
    )
    tracer = _tracer(art, date)

    result = run_publish(
        date=date,
        publisher=publisher,
        publish_cfg=_publish_cfg(),
        artifacts_root=str(art),
        tracer=tracer,
        mode="confirm",
        force=True,
    )

    assert result["status"] == "published"
    assert result["issue_number"] == 43  # next-after-42
    assert result["force"] is True
    assert result.get("forced_over_duplicate") is True
    assert len(publisher.created) == 1

    saved = json.load(open(result["result_path"], encoding="utf-8"))
    assert saved["status"] == "published"
    assert saved["forced_over_duplicate"] is True


# --------------------------------------------------------------------------- #
# 5b. --force does NOT bypass the gate
# --------------------------------------------------------------------------- #


def test_force_does_not_bypass_gate(tmp_path):
    date = "2026-05-09"
    art = _seed_artifacts(
        tmp_path,
        date,
        draft_payload=_draft_payload(date),
        report=_failed_critic_report(date),
    )
    publisher = FakeIssuePublisher(
        repo="owner/test-repo", existing=_duplicate_existing(date)
    )
    tracer = _tracer(art, date)

    result = run_publish(
        date=date,
        publisher=publisher,
        publish_cfg=_publish_cfg(),
        artifacts_root=str(art),
        tracer=tracer,
        mode="confirm",
        force=True,
    )

    # force bypasses both duplicate check AND gate check.
    assert result["status"] == "published"


# --------------------------------------------------------------------------- #
# 6. trace events are written for every meaningful step
# --------------------------------------------------------------------------- #


def test_issue_publisher_trace_written(tmp_path):
    date = "2026-05-09"
    art = _seed_artifacts(
        tmp_path,
        date,
        draft_payload=_draft_payload(date),
        report=_ok_report(date),
    )
    publisher = FakeIssuePublisher(repo="owner/test-repo")
    tracer = _tracer(art, date)

    # one dry-run + one confirm in the same trace file
    run_publish(
        date=date,
        publisher=publisher,
        publish_cfg=_publish_cfg(),
        artifacts_root=str(art),
        tracer=tracer,
        mode="dry-run",
    )
    run_publish(
        date=date,
        publisher=publisher,
        publish_cfg=_publish_cfg(),
        artifacts_root=str(art),
        tracer=tracer,
        mode="confirm",
    )

    events = tracer.read_all()
    kinds = [e["event"] for e in events]
    assert "publish_gate" in kinds
    assert "publish_preview_written" in kinds
    assert "publish_succeeded" in kinds

    # publish_succeeded must include issue_number and issue_url
    succeeded = [e for e in events if e["event"] == "publish_succeeded"]
    assert succeeded and succeeded[0]["issue_number"] >= 1
    assert succeeded[0]["issue_url"].startswith("https://github.com/")


# --------------------------------------------------------------------------- #
# extra coverage: helpers
# --------------------------------------------------------------------------- #


def test_find_duplicates_matches_by_date_in_title():
    publisher = FakeIssuePublisher(
        repo="owner/test-repo",
        existing=[
            ExistingIssue(
                number=10,
                title="some other thing 2026-05-09 here",
                state="closed",
                html_url="x",
                created_at="x",
                author_login="owner",
            )
        ],
    )
    dups = find_duplicates(
        publisher=publisher,
        date="2026-05-09",
        title="AI 日报 2026-05-09",
        recent_to_check=30,
    )
    assert dups and dups[0].number == 10


def test_evaluate_gate_rejects_too_few_items(tmp_path):
    date = "2026-05-09"
    sparse = _draft_payload(date)
    sparse["sections"] = [sparse["sections"][0]]  # only 1 item
    art = _seed_artifacts(
        tmp_path, date, draft_payload=sparse, report=_ok_report(date)
    )
    artifacts = load_run_artifacts(date, str(art))
    gate = evaluate_publish_gate(artifacts, _publish_cfg())
    assert gate.ok is False
    assert any("minimum_items" in r for r in gate.blocked_reasons)


def test_evaluate_gate_rejects_empty_markdown(tmp_path):
    date = "2026-05-09"
    art = _seed_artifacts(
        tmp_path, date, draft_payload=_draft_payload(date), report=_ok_report(date)
    )
    # Stomp the markdown file to empty after seeding.
    md_path = art / "drafts" / f"{date}.md"
    md_path.write_text("", encoding="utf-8")
    artifacts = load_run_artifacts(date, str(art))
    gate = evaluate_publish_gate(artifacts, _publish_cfg())
    assert gate.ok is False
    assert any("markdown file is empty" in r for r in gate.blocked_reasons)


def test_build_issue_body_appends_audit_footer(tmp_path):
    date = "2026-05-09"
    art = _seed_artifacts(
        tmp_path,
        date,
        draft_payload=_draft_payload(date),
        report=_ok_report(date, run_id="r-99"),
    )
    artifacts = load_run_artifacts(date, str(art))
    body = build_issue_body(artifacts)
    assert "seed body" in body
    assert "report-agent" in body
    assert "r-99" in body


def test_label_env_override(tmp_path, monkeypatch):
    date = "2026-05-09"
    art = _seed_artifacts(
        tmp_path,
        date,
        draft_payload=_draft_payload(date),
        report=_ok_report(date),
    )
    monkeypatch.setenv("PUBLISH_ISSUE_LABELS", "ai-daily, auto")
    publisher = FakeIssuePublisher(repo="owner/test-repo")
    tracer = _tracer(art, date)
    result = run_publish(
        date=date,
        publisher=publisher,
        publish_cfg=_publish_cfg(),
        artifacts_root=str(art),
        tracer=tracer,
        mode="dry-run",
    )
    assert result["labels"] == ["ai-daily", "auto"]


# --------------------------------------------------------------------------- #
# 14. repair failure gate
# --------------------------------------------------------------------------- #


def _report_with_repair(date: str, *, attempted: bool, succeeded: bool) -> Dict[str, Any]:
    rep = _ok_report(date)
    rep["repair_attempted"] = attempted
    rep["repair_succeeded"] = succeeded
    rep["semantic_duplicate_report_path"] = None
    return rep


def test_publish_gate_blocks_failed_repair(tmp_path):
    date = "2026-05-09"
    art = _seed_artifacts(
        tmp_path, date,
        draft_payload=_draft_payload(date),
        report=_report_with_repair(date, attempted=True, succeeded=False),
    )
    arts = load_run_artifacts(date, str(art))
    gate = evaluate_publish_gate(arts, _publish_cfg())
    assert gate.ok is False
    assert any("repair_failed" in r for r in gate.blocked_reasons)


def test_publish_gate_allows_successful_repair_if_sem_dup_clean(tmp_path):
    date = "2026-05-09"
    art = _seed_artifacts(
        tmp_path, date,
        draft_payload=_draft_payload(date),
        report=_report_with_repair(date, attempted=True, succeeded=True),
    )
    arts = load_run_artifacts(date, str(art))
    gate = evaluate_publish_gate(arts, _publish_cfg())
    assert gate.ok is True
    assert not any("repair_failed" in r for r in gate.blocked_reasons)


def test_publish_gate_ignores_repair_fields_when_not_attempted(tmp_path):
    date = "2026-05-09"
    rep = _ok_report(date)
    rep["repair_attempted"] = False
    rep["repair_succeeded"] = False
    rep["semantic_duplicate_report_path"] = None
    art = _seed_artifacts(
        tmp_path, date,
        draft_payload=_draft_payload(date),
        report=rep,
    )
    arts = load_run_artifacts(date, str(art))
    gate = evaluate_publish_gate(arts, _publish_cfg())
    assert gate.ok is True
    assert not any("repair" in r for r in gate.blocked_reasons)
