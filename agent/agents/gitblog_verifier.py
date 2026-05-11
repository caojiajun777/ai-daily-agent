"""GitBlog verifier — read-only end-to-end check after a smoke-test publish.

**Legacy note**: this module was originally designed to verify that an
agent-published issue propagated through the juya-ai-daily downstream
pipeline (generate_readme.yml → README / RSS / BACKUP / Pages). It can
equally be used to verify any repo that follows the same conventions:
pass ``workflow_filename`` to the API client to point at your own
workflow file, and set the ``BACKUP/`` path expectations accordingly.

Given a GitHub issue number that we believe was created by the agent, this
agent walks the target repo and reports whether each expected artifact was
updated:

  1. issue_exists                        — issue is present in target repo
  2. author_is_owner                     — issue.user.login == repo.owner.login
  3. title_or_body_contains_date         — heuristic: ``<date>`` somewhere in
                                           title or body, so we know which
                                           issue we're really looking at
  4. generate_readme_workflow_triggered  — most recent run of
                                           ``generate_readme.yml`` is newer
                                           than ``issue.created_at``
  5. readme_contains_issue_title         — README.md mentions the issue title
  6. backup_has_issue_file               — BACKUP/ contains a file whose
                                           name starts with ``<issue.number>_``
  7. rss_contains_issue_title            — rss.xml contains the issue title

All checks are independent. The agent never raises on a check failure;
failures land in the report so a human (or CI) can read the full picture.

The only failure modes that *do* raise are:

  - the publisher adapter cannot reach GitHub at all
  - the issue endpoint returns 404 (we cannot proceed without an issue)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.tools.gitblog_verifier import (
    GitHubReadAPI,
    IssueView,
    VerifierError,
    WorkflowRunView,
)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "data": self.data,
        }


@dataclass
class VerifyReport:
    repo: str
    issue_number: int
    date: str
    checks: List[CheckResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    fatal_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.fatal_error is None and all(c.ok for c in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo": self.repo,
            "issue_number": self.issue_number,
            "date": self.date,
            "ok": self.ok,
            "fatal_error": self.fatal_error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "checks": [c.to_dict() for c in self.checks],
        }


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

GENERATE_README_WORKFLOW = "generate_readme.yml"
README_PATH = "README.md"
RSS_PATH = "rss.xml"
BACKUP_DIR = "BACKUP"

_ISO_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _parse_iso(ts: str) -> float:
    """Best-effort ISO-8601 → epoch. Returns 0.0 on failure."""
    if not ts or not _ISO_TS.match(ts):
        return 0.0
    try:
        from datetime import datetime, timezone

        # GitHub returns either Z-suffixed UTC or offset-suffixed timestamps.
        s = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(s).astimezone(timezone.utc).timestamp()
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def _check_author_is_owner(issue: IssueView, owner: str) -> CheckResult:
    if issue.author_login == owner:
        return CheckResult(
            name="author_is_owner",
            ok=True,
            detail=f"author={issue.author_login!r} == owner",
            data={"author": issue.author_login, "owner": owner},
        )
    msg = (
        f"author={issue.author_login!r} != owner={owner!r}. "
        "The downstream main.py uses is_me(issue, me) where me = repo.owner.login; "
        "issues created by github-actions[bot] would be silently skipped."
    )
    return CheckResult(
        name="author_is_owner",
        ok=False,
        detail=msg,
        data={"author": issue.author_login, "owner": owner},
    )


def _check_title_or_body_contains_date(
    issue: IssueView, date: str
) -> CheckResult:
    in_title = date in (issue.title or "")
    in_body = date in (issue.body or "")
    ok = in_title or in_body
    return CheckResult(
        name="title_or_body_contains_date",
        ok=ok,
        detail=(
            f"date={date!r} found in "
            f"{'title' if in_title else ''}"
            f"{'+body' if in_title and in_body else ('body' if in_body else '')}"
            if ok
            else f"date={date!r} not found in title/body"
        ),
        data={"in_title": in_title, "in_body": in_body},
    )


def _check_workflow_recently_triggered(
    runs: List[WorkflowRunView], issue_created_at: str
) -> CheckResult:
    if not runs:
        return CheckResult(
            name="generate_readme_workflow_triggered",
            ok=False,
            detail=(
                f"no runs found for {GENERATE_README_WORKFLOW}; either the "
                "workflow file is missing or it has never run."
            ),
            data={"run_count": 0},
        )
    issue_ts = _parse_iso(issue_created_at)
    newest = max(runs, key=lambda r: _parse_iso(r.created_at))
    newest_ts = _parse_iso(newest.created_at)
    if newest_ts >= issue_ts and issue_ts > 0:
        return CheckResult(
            name="generate_readme_workflow_triggered",
            ok=True,
            detail=(
                f"workflow run #{newest.id} ({newest.event}/"
                f"{newest.status}/{newest.conclusion or 'pending'}) at "
                f"{newest.created_at} is at-or-after issue created_at"
            ),
            data={
                "run_id": newest.id,
                "event": newest.event,
                "status": newest.status,
                "conclusion": newest.conclusion,
                "created_at": newest.created_at,
                "html_url": newest.html_url,
            },
        )
    return CheckResult(
        name="generate_readme_workflow_triggered",
        ok=False,
        detail=(
            f"newest workflow run #{newest.id} at {newest.created_at} "
            f"predates issue created_at {issue_created_at}; the workflow "
            "may not have been triggered yet, give it ~1 minute and re-run."
        ),
        data={
            "newest_run_id": newest.id,
            "newest_created_at": newest.created_at,
            "issue_created_at": issue_created_at,
        },
    )


def _check_readme_mentions_title(
    api: GitHubReadAPI, issue: IssueView
) -> CheckResult:
    try:
        readme = api.get_file_text(README_PATH)
    except VerifierError as e:
        return CheckResult(
            name="readme_contains_issue_title",
            ok=False,
            detail=f"could not read {README_PATH}: {e}",
        )
    title = (issue.title or "").strip()
    if not title:
        return CheckResult(
            name="readme_contains_issue_title",
            ok=False,
            detail="issue has empty title",
        )
    if title in readme:
        return CheckResult(
            name="readme_contains_issue_title",
            ok=True,
            detail=f"title {title!r} present in {README_PATH}",
        )
    return CheckResult(
        name="readme_contains_issue_title",
        ok=False,
        detail=(
            f"title {title!r} NOT found in {README_PATH}. The downstream "
            "generate_readme workflow may not have completed yet, or "
            "main.py skipped this issue (often because the author isn't "
            "the repo owner)."
        ),
    )


def _check_backup_has_issue_file(
    api: GitHubReadAPI, issue: IssueView
) -> CheckResult:
    entries = api.list_dir(BACKUP_DIR)
    prefix = f"{issue.number}_"
    matches = [name for name in entries if name.startswith(prefix)]
    if matches:
        return CheckResult(
            name="backup_has_issue_file",
            ok=True,
            detail=f"found {len(matches)} matching file(s) in {BACKUP_DIR}/",
            data={"matches": matches},
        )
    return CheckResult(
        name="backup_has_issue_file",
        ok=False,
        detail=(
            f"no file in {BACKUP_DIR}/ starts with {prefix!r}. main.py's "
            "save_issue() writes <number>_<title>.md; absence usually means "
            "main.py never processed this issue."
        ),
        data={"backup_dir_entry_count": len(entries)},
    )


def _check_rss_contains_title(
    api: GitHubReadAPI, issue: IssueView
) -> CheckResult:
    try:
        rss = api.get_file_text(RSS_PATH)
    except VerifierError as e:
        return CheckResult(
            name="rss_contains_issue_title",
            ok=False,
            detail=f"could not read {RSS_PATH}: {e}",
        )
    title = (issue.title or "").strip()
    if title and title in rss:
        return CheckResult(
            name="rss_contains_issue_title",
            ok=True,
            detail=f"title {title!r} present in {RSS_PATH}",
        )
    return CheckResult(
        name="rss_contains_issue_title",
        ok=False,
        detail=(
            f"title {title!r} NOT found in {RSS_PATH}. The RSS regenerator "
            "runs inside the same workflow as the README writer, so the "
            "same root cause (author filter, workflow not yet run) usually "
            "explains both."
        ),
    )


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def verify_gitblog(
    *,
    api: GitHubReadAPI,
    issue_number: int,
    date: Optional[str] = None,
) -> VerifyReport:
    """Run all 7 checks against an existing GitHub issue.

    ``date`` defaults to the issue's ``created_at`` date if not provided.
    """
    report = VerifyReport(repo=api.repo, issue_number=issue_number, date=date or "")

    # --- preconditions: get issue + owner ---
    try:
        owner = api.get_repo_owner()
    except VerifierError as e:
        report.fatal_error = f"could not resolve repo owner: {e}"
        report.ended_at = time.time()
        return report

    try:
        issue = api.get_issue(issue_number)
    except VerifierError as e:
        report.fatal_error = f"issue {issue_number} not found: {e}"
        report.ended_at = time.time()
        return report

    # 1. issue_exists is implied by surviving the get_issue call.
    report.checks.append(
        CheckResult(
            name="issue_exists",
            ok=True,
            detail=f"#{issue.number} {issue.state!r} by {issue.author_login!r}",
            data={
                "title": issue.title,
                "state": issue.state,
                "html_url": issue.html_url,
                "created_at": issue.created_at,
                "author_login": issue.author_login,
            },
        )
    )

    # 2. author == owner
    report.checks.append(_check_author_is_owner(issue, owner))

    # 3. date present in title/body
    effective_date = date or (issue.created_at[:10] if issue.created_at else "")
    report.date = effective_date
    report.checks.append(_check_title_or_body_contains_date(issue, effective_date))

    # 4. workflow recently triggered
    try:
        runs = api.list_workflow_runs(
            workflow_filename=GENERATE_README_WORKFLOW, limit=10
        )
    except VerifierError as e:
        runs = []
        report.checks.append(
            CheckResult(
                name="generate_readme_workflow_triggered",
                ok=False,
                detail=f"workflow runs API error: {e}",
            )
        )
    else:
        report.checks.append(
            _check_workflow_recently_triggered(runs, issue.created_at)
        )

    # 5. README contains title
    report.checks.append(_check_readme_mentions_title(api, issue))

    # 6. BACKUP/<number>_*.md exists
    report.checks.append(_check_backup_has_issue_file(api, issue))

    # 7. rss.xml contains title
    report.checks.append(_check_rss_contains_title(api, issue))

    report.ended_at = time.time()
    return report
