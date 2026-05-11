"""GitHub Issue publisher.

This module owns the *effect* of creating a GitHub issue. All gate checks and
gating policy live one layer up in ``agent/agents/issue_publisher.py``. The
split is deliberate: agents are pure Python, tools have side effects. Testing
the agent layer only needs a fake ``IssuePublisher``, never a network.

Authentication policy
---------------------
Issues MUST be created with a Personal Access Token belonging to the repo
owner account, not the default ``GITHUB_TOKEN`` (github-actions[bot]):

  - Bot-created issues lack audit attribution and make the publish history
    hard to inspect on the Issues tab.
  - The token must correspond to a real GitHub user for proper authorship
    tracking in the project's own resume/demo context.

Therefore:

  - we REFUSE to construct ``GitHubIssuePublisher`` without
    ``GITHUB_PUBLISH_TOKEN`` (or its deprecated alias ``GITBLOG_OWNER_PAT``)
  - we REFUSE to accept a token equal to ``GITHUB_TOKEN`` (bot identity)

API surface
-----------
``IssuePublisher`` is a Protocol. Two implementations:

  - ``GitHubIssuePublisher`` — real, talks to api.github.com over httpx.
  - ``FakeIssuePublisher`` — in-memory, used by tests.

Both implementations never swallow errors: on HTTP failure they raise
``PublisherError``. The agent layer is responsible for translating that into
trace events and a publish_result file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class PublisherError(RuntimeError):
    pass


class PublisherConfigError(RuntimeError):
    """Configuration is invalid or missing (no network attempted yet)."""


@dataclass
class ExistingIssue:
    number: int
    title: str
    state: str
    html_url: str
    created_at: str
    author_login: str


@dataclass
class CreatedIssue:
    number: int
    html_url: str
    created_at: str
    # Raw API response retained for debugging / audit, not serialized.
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


class IssuePublisher(Protocol):
    repo: str

    def list_recent_issues(self, limit: int) -> List[ExistingIssue]: ...

    def create_issue(
        self, *, title: str, body: str, labels: List[str]
    ) -> CreatedIssue: ...


# --------------------------------------------------------------------------- #
# Real implementation (httpx against api.github.com)
# --------------------------------------------------------------------------- #


class GitHubIssuePublisher:
    """Real GitHub publisher.

    Reads config from env at construction time. Does not talk to the network
    until one of its methods is called.
    """

    API_ROOT = "https://api.github.com"

    def __init__(
        self,
        *,
        repo: Optional[str] = None,
        token: Optional[str] = None,
        token_env_var: str = "GITHUB_PUBLISH_TOKEN",
        repo_env_var: str = "PUBLISH_REPO",
        timeout_s: float = 20.0,
    ) -> None:
        # Token: prefer GITHUB_PUBLISH_TOKEN; fall back to deprecated GITBLOG_OWNER_PAT.
        resolved_token = token or os.environ.get(token_env_var)
        if not resolved_token and token_env_var == "GITHUB_PUBLISH_TOKEN":
            resolved_token = os.environ.get("GITBLOG_OWNER_PAT")
        if not resolved_token:
            raise PublisherConfigError(
                f"{token_env_var} is not set. Set GITHUB_PUBLISH_TOKEN to a "
                "Personal Access Token owned by the target repo's owner account. "
                "Issues created with a bot token lack proper audit attribution."
            )
        # Sanity: disallow the default workflow bot token if plumbed by mistake.
        if resolved_token == os.environ.get("GITHUB_TOKEN"):
            raise PublisherConfigError(
                "Refusing to use GITHUB_TOKEN (github-actions[bot]) as the "
                "publisher token. Use GITHUB_PUBLISH_TOKEN (repo owner's PAT) "
                "to ensure proper authorship attribution."
            )
        self._token = resolved_token

        # Repo: prefer PUBLISH_REPO; fall back to deprecated GITBLOG_REPO.
        resolved_repo = repo or os.environ.get(repo_env_var)
        if not resolved_repo and repo_env_var == "PUBLISH_REPO":
            resolved_repo = os.environ.get("GITBLOG_REPO")
        if not resolved_repo or "/" not in resolved_repo:
            raise PublisherConfigError(
                f"repo must be 'owner/name' (got {resolved_repo!r}); set "
                f"{repo_env_var} or pass repo=... explicitly."
            )
        self.repo = resolved_repo
        self._timeout_s = timeout_s
        try:
            import httpx  # noqa: F401 — import here to surface clearer errors
        except ImportError as e:  # pragma: no cover - dep is in requirements
            raise PublisherConfigError("httpx is required") from e

    # ---------- network ----------

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "report-agent/0.1",
        }

    def list_recent_issues(self, limit: int) -> List[ExistingIssue]:
        import httpx

        url = f"{self.API_ROOT}/repos/{self.repo}/issues"
        params = {
            "state": "all",
            "per_page": str(min(100, max(1, limit))),
            "sort": "created",
            "direction": "desc",
        }
        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.get(url, headers=self._headers(), params=params)
        except httpx.HTTPError as e:
            raise PublisherError(f"list_recent_issues network error: {e}") from e
        if resp.status_code >= 400:
            raise PublisherError(
                f"list_recent_issues failed: {resp.status_code} {resp.text[:200]}"
            )
        data = resp.json()
        out: List[ExistingIssue] = []
        for entry in data:
            # GitHub's /issues endpoint returns PRs too; skip them.
            if "pull_request" in entry:
                continue
            out.append(
                ExistingIssue(
                    number=int(entry["number"]),
                    title=entry.get("title", ""),
                    state=entry.get("state", ""),
                    html_url=entry.get("html_url", ""),
                    created_at=entry.get("created_at", ""),
                    author_login=(entry.get("user") or {}).get("login", ""),
                )
            )
            if len(out) >= limit:
                break
        return out

    def create_issue(
        self, *, title: str, body: str, labels: List[str]
    ) -> CreatedIssue:
        import httpx

        url = f"{self.API_ROOT}/repos/{self.repo}/issues"
        payload = {"title": title, "body": body, "labels": list(labels)}
        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.post(url, headers=self._headers(), json=payload)
        except httpx.HTTPError as e:
            raise PublisherError(f"create_issue network error: {e}") from e
        if resp.status_code >= 400:
            raise PublisherError(
                f"create_issue failed: {resp.status_code} {resp.text[:300]}"
            )
        data = resp.json()
        return CreatedIssue(
            number=int(data["number"]),
            html_url=data.get("html_url", ""),
            created_at=data.get("created_at", ""),
            raw=data,
        )


# --------------------------------------------------------------------------- #
# Fake implementation (tests)
# --------------------------------------------------------------------------- #


class FakeIssuePublisher:
    """In-memory publisher for tests and local dry-runs with synthetic data."""

    def __init__(
        self,
        repo: str = "test-owner/test-repo",
        existing: Optional[List[ExistingIssue]] = None,
        fail_on_create: bool = False,
    ) -> None:
        self.repo = repo
        self._existing: List[ExistingIssue] = list(existing or [])
        self._created: List[CreatedIssue] = []
        self._fail_on_create = fail_on_create
        self._next_number = (
            max([i.number for i in self._existing], default=0) + 1
        )

    def list_recent_issues(self, limit: int) -> List[ExistingIssue]:
        return list(self._existing)[:limit]

    def create_issue(
        self, *, title: str, body: str, labels: List[str]
    ) -> CreatedIssue:
        if self._fail_on_create:
            raise PublisherError("fake: create failed")
        number = self._next_number
        self._next_number += 1
        created = CreatedIssue(
            number=number,
            html_url=f"https://github.com/{self.repo}/issues/{number}",
            created_at="2026-05-09T00:00:00Z",
            raw={"fake": True, "title": title, "body": body, "labels": labels},
        )
        self._created.append(created)
        self._existing.append(
            ExistingIssue(
                number=number,
                title=title,
                state="open",
                html_url=created.html_url,
                created_at=created.created_at,
                author_login="test-owner",
            )
        )
        return created

    # Test helpers — not part of the Protocol.
    @property
    def created(self) -> List[CreatedIssue]:
        return list(self._created)
