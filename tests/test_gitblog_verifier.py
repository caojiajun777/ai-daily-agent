"""verify-gitblog tests. All GitHub API access is mocked via FakeGitHubReadAPI."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from typing import Dict, List

import pytest

from agent.agents.gitblog_verifier import verify_gitblog
from agent.tools.gitblog_verifier import (
    FakeGitHubReadAPI,
    IssueView,
    WorkflowRunView,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _issue(
    *,
    number: int = 84,
    author: str = "test-owner",
    title: str = "AI 日报 2026-05-09",
    body: str = "# AI 日报 2026-05-09\n\nbody body",
    created_at: str = "2026-05-09T01:23:45Z",
) -> IssueView:
    return IssueView(
        number=number,
        title=title,
        body=body,
        state="open",
        html_url=f"https://github.com/test-owner/test-repo/issues/{number}",
        created_at=created_at,
        author_login=author,
    )


def _good_workflow_run(after: str = "2026-05-09T01:24:30Z") -> WorkflowRunView:
    return WorkflowRunView(
        id=987654321,
        name="Generate GitBlog README",
        event="issues",
        status="completed",
        conclusion="success",
        created_at=after,
        head_branch="master",
        html_url="https://github.com/test-owner/test-repo/actions/runs/987654321",
    )


def _api_with_full_pipeline(
    *,
    issue: IssueView = None,
    title_in_readme: bool = True,
    title_in_rss: bool = True,
    backup_present: bool = True,
    workflow_runs: List[WorkflowRunView] = None,
) -> FakeGitHubReadAPI:
    issue = issue or _issue()
    files: Dict[str, str] = {}
    files["README.md"] = (
        f"# 橘鸦AI日报\n\n## 最近更新\n- [{issue.title}](https://x)\n"
        if title_in_readme
        else "# 橘鸦AI日报\n\n## 最近更新\n- (no entry)\n"
    )
    files["rss.xml"] = (
        f"<?xml version='1.0'?><rss><channel><item><title>{issue.title}</title></item></channel></rss>"
        if title_in_rss
        else "<?xml version='1.0'?><rss><channel></channel></rss>"
    )
    backup_files = (
        [f"{issue.number}_AI.日报.{issue.title.split()[-1]}.md"]
        if backup_present
        else []
    )
    return FakeGitHubReadAPI(
        repo="test-owner/test-repo",
        owner_login="test-owner",
        issues={issue.number: issue},
        workflow_runs={
            "generate_readme.yml": workflow_runs
            if workflow_runs is not None
            else [_good_workflow_run()]
        },
        files=files,
        dirs={"BACKUP": backup_files},
    )


# --------------------------------------------------------------------------- #
# Required tests
# --------------------------------------------------------------------------- #


def test_verify_gitblog_issue_author_owner():
    """Happy path: every check passes when the issue was created by owner and
    the downstream pipeline has run."""
    api = _api_with_full_pipeline()
    report = verify_gitblog(api=api, issue_number=84, date="2026-05-09")

    assert report.ok is True
    assert report.fatal_error is None
    names = {c.name for c in report.checks}
    expected = {
        "issue_exists",
        "author_is_owner",
        "title_or_body_contains_date",
        "generate_readme_workflow_triggered",
        "readme_contains_issue_title",
        "backup_has_issue_file",
        "rss_contains_issue_title",
    }
    assert expected <= names

    by_name = {c.name: c for c in report.checks}
    assert by_name["author_is_owner"].ok is True
    assert by_name["readme_contains_issue_title"].ok is True
    assert by_name["backup_has_issue_file"].ok is True
    assert by_name["rss_contains_issue_title"].ok is True


def test_verify_gitblog_detects_bot_author():
    """When the issue was created by github-actions[bot], the author check must
    fail with a clear explanation that mentions main.py's is_me() filter."""
    api = _api_with_full_pipeline(
        issue=_issue(author="github-actions[bot]"),
        # Bot author means main.py would skip it: README/RSS/BACKUP empty.
        title_in_readme=False,
        title_in_rss=False,
        backup_present=False,
    )
    report = verify_gitblog(api=api, issue_number=84, date="2026-05-09")

    assert report.ok is False
    by_name = {c.name: c for c in report.checks}
    author_check = by_name["author_is_owner"]
    assert author_check.ok is False
    assert "is_me" in author_check.detail
    assert author_check.data == {
        "author": "github-actions[bot]",
        "owner": "test-owner",
    }
    # And the cascade is reported, not hidden:
    assert by_name["readme_contains_issue_title"].ok is False
    assert by_name["backup_has_issue_file"].ok is False
    assert by_name["rss_contains_issue_title"].ok is False


def test_verify_gitblog_detects_missing_readme_entry():
    """Issue was created by owner, workflow ran, but README has no entry — the
    verifier must surface that single failure without claiming the whole
    pipeline broke."""
    api = _api_with_full_pipeline(title_in_readme=False)
    report = verify_gitblog(api=api, issue_number=84, date="2026-05-09")

    assert report.ok is False
    by_name = {c.name: c for c in report.checks}
    assert by_name["author_is_owner"].ok is True
    assert by_name["generate_readme_workflow_triggered"].ok is True
    assert by_name["readme_contains_issue_title"].ok is False
    assert "NOT found" in by_name["readme_contains_issue_title"].detail
    # The other passing checks should still report ok=True so the user can
    # tell what *did* work.
    assert by_name["backup_has_issue_file"].ok is True
    assert by_name["rss_contains_issue_title"].ok is True


def test_verify_gitblog_json_report():
    """--json mode must produce a stable, parsable structure."""
    from agent.cli import build_parser, cmd_verify_gitblog

    api = _api_with_full_pipeline()
    report = verify_gitblog(api=api, issue_number=84, date="2026-05-09")

    payload = report.to_dict()
    # Top-level shape
    assert payload["repo"] == "test-owner/test-repo"
    assert payload["issue_number"] == 84
    assert payload["date"] == "2026-05-09"
    assert payload["ok"] is True
    assert payload["fatal_error"] is None
    assert isinstance(payload["checks"], list)
    # Each check has the documented schema
    for c in payload["checks"]:
        assert set(c.keys()) >= {"name", "ok", "detail", "data"}
        assert isinstance(c["name"], str)
        assert isinstance(c["ok"], bool)

    # The dict round-trips through JSON without losing fields.
    serialized = json.dumps(payload, ensure_ascii=False)
    revived = json.loads(serialized)
    assert revived["ok"] is True
    assert len(revived["checks"]) == len(payload["checks"])


# --------------------------------------------------------------------------- #
# Extra coverage: 404 / workflow / partial states / no real network
# --------------------------------------------------------------------------- #


def test_verify_gitblog_issue_not_found_is_fatal():
    api = _api_with_full_pipeline()  # contains issue #84 only
    report = verify_gitblog(api=api, issue_number=99999)
    assert report.fatal_error is not None
    assert "issue 99999 not found" in report.fatal_error
    assert report.ok is False


def test_verify_gitblog_workflow_not_yet_triggered():
    """Workflow exists but its newest run predates the issue → check fails
    with a hint to retry."""
    api = _api_with_full_pipeline(
        workflow_runs=[
            WorkflowRunView(
                id=1,
                name="Generate GitBlog README",
                event="push",
                status="completed",
                conclusion="success",
                created_at="2026-05-08T10:00:00Z",  # before issue
                head_branch="master",
                html_url="https://x",
            )
        ],
        title_in_readme=False,
        title_in_rss=False,
        backup_present=False,
    )
    report = verify_gitblog(api=api, issue_number=84, date="2026-05-09")
    by_name = {c.name: c for c in report.checks}
    wf = by_name["generate_readme_workflow_triggered"]
    assert wf.ok is False
    assert "predates issue" in wf.detail


def test_verify_gitblog_no_workflow_runs_at_all():
    api = _api_with_full_pipeline(workflow_runs=[])
    report = verify_gitblog(api=api, issue_number=84, date="2026-05-09")
    by_name = {c.name: c for c in report.checks}
    assert by_name["generate_readme_workflow_triggered"].ok is False
    assert "no runs found" in by_name["generate_readme_workflow_triggered"].detail


def test_verify_gitblog_date_inferred_from_issue():
    """If --date is omitted, we use issue.created_at[:10]."""
    api = _api_with_full_pipeline(
        issue=_issue(created_at="2026-04-01T05:06:07Z", title="AI 日报 2026-04-01"),
    )
    api._files["README.md"] = "## ... AI 日报 2026-04-01 ..."
    api._files["rss.xml"] = "<rss><item><title>AI 日报 2026-04-01</title></item></rss>"
    api._dirs["BACKUP"] = ["84_AI.日报.2026-04-01.md"]
    report = verify_gitblog(api=api, issue_number=84)
    assert report.date == "2026-04-01"
    by_name = {c.name: c for c in report.checks}
    assert by_name["title_or_body_contains_date"].ok is True


def test_verify_gitblog_does_not_perform_real_network():
    """The Fake adapter records calls; assert we only ever hit it (no httpx)."""
    api = _api_with_full_pipeline()
    verify_gitblog(api=api, issue_number=84, date="2026-05-09")
    # Calls are recorded — we used the fake, nothing else was hit.
    assert api.calls
    assert any(c.startswith("get_issue:") for c in api.calls)
    assert any(c.startswith("list_workflow_runs:") for c in api.calls)
    assert "get_file_text:README.md" in api.calls
    assert "get_file_text:rss.xml" in api.calls
    assert "list_dir:BACKUP" in api.calls


def test_verify_gitblog_human_print_does_not_crash():
    """The CLI's human-readable formatter should run without exceptions on a
    realistic mixed-result report."""
    from agent.cli import _print_verify_report_human

    api = _api_with_full_pipeline(title_in_readme=False)
    report = verify_gitblog(api=api, issue_number=84, date="2026-05-09")
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_verify_report_human(report)
    text = buf.getvalue()
    assert "verify-gitblog" in text
    assert "FAIL" in text  # at least one failed check
    assert "OK " in text  # and at least one passed check
