"""Read-only GitHub adapter for the gitblog verifier.

This module mirrors the layered design used by ``issue_publisher``: a
``GitHubReadAPI`` Protocol, one real implementation that talks to
``api.github.com`` over httpx, and one fake implementation for tests. The
verifier *never* writes — there is no ``create_*`` method anywhere here.

Why a separate adapter
----------------------
We could reuse pieces of ``GitHubIssuePublisher``, but separating read from
write makes the access surface obvious in code review: the verifier cannot
accidentally publish or modify anything because the methods don't exist on
its adapter type.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


class VerifierError(RuntimeError):
    pass


class VerifierConfigError(RuntimeError):
    """Configuration is invalid (no network attempted)."""


@dataclass
class IssueView:
    number: int
    title: str
    body: str
    state: str
    html_url: str
    created_at: str
    author_login: str


@dataclass
class WorkflowRunView:
    id: int
    name: str
    event: str
    status: str  # queued / in_progress / completed
    conclusion: str  # success / failure / cancelled / "" while running
    created_at: str
    head_branch: str
    html_url: str


class GitHubReadAPI(Protocol):
    repo: str
    owner_login: str  # cached after the first get_repo() call

    def get_repo_owner(self) -> str: ...

    def get_issue(self, number: int) -> IssueView: ...

    def list_workflow_runs(
        self, *, workflow_filename: str, limit: int
    ) -> List[WorkflowRunView]: ...

    def get_file_text(self, path: str, *, ref: Optional[str] = None) -> str: ...

    def list_dir(self, path: str, *, ref: Optional[str] = None) -> List[str]: ...


# --------------------------------------------------------------------------- #
# Real implementation
# --------------------------------------------------------------------------- #


class GitHubReadAPIClient:
    """Real read-only GitHub client.

    The token is optional for public repos but recommended (rate limits +
    private repo support). The verifier defaults to reading the same env var
    as the publisher (``GITBLOG_OWNER_PAT``); callers can override the env
    var name from the CLI.
    """

    API_ROOT = "https://api.github.com"
    RAW_ROOT = "https://raw.githubusercontent.com"

    def __init__(
        self,
        *,
        repo: Optional[str] = None,
        token: Optional[str] = None,
        token_env_var: str = "GITBLOG_OWNER_PAT",
        repo_env_var: str = "GITBLOG_REPO",
        timeout_s: float = 20.0,
    ) -> None:
        resolved_repo = repo or os.environ.get(repo_env_var)
        if not resolved_repo or "/" not in resolved_repo:
            raise VerifierConfigError(
                f"repo must be 'owner/name' (got {resolved_repo!r}); set "
                f"{repo_env_var} or pass --repo."
            )
        self.repo = resolved_repo
        self._token = token if token is not None else os.environ.get(token_env_var)
        self._timeout_s = timeout_s
        self.owner_login = ""  # populated lazily by get_repo_owner()
        try:
            import httpx  # noqa: F401
        except ImportError as e:  # pragma: no cover - dep is in requirements
            raise VerifierConfigError("httpx is required") from e

    # ---- helpers ----

    def _headers(self) -> Dict[str, str]:
        h = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "report-agent-verifier/0.1",
        }
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _get(self, url: str, params: Optional[Dict[str, str]] = None) -> Any:
        import httpx

        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.get(url, headers=self._headers(), params=params)
        except httpx.HTTPError as e:
            raise VerifierError(f"GET {url} failed: {e}") from e
        if resp.status_code == 404:
            raise VerifierError(f"GET {url} -> 404 not found")
        if resp.status_code >= 400:
            raise VerifierError(
                f"GET {url} -> {resp.status_code} {resp.text[:200]}"
            )
        return resp

    # ---- API surface ----

    def get_repo_owner(self) -> str:
        if self.owner_login:
            return self.owner_login
        resp = self._get(f"{self.API_ROOT}/repos/{self.repo}")
        data = resp.json()
        owner = ((data.get("owner") or {}).get("login")) or self.repo.split("/", 1)[0]
        self.owner_login = owner
        return owner

    def get_issue(self, number: int) -> IssueView:
        resp = self._get(f"{self.API_ROOT}/repos/{self.repo}/issues/{number}")
        data = resp.json()
        return IssueView(
            number=int(data["number"]),
            title=data.get("title", "") or "",
            body=data.get("body", "") or "",
            state=data.get("state", "") or "",
            html_url=data.get("html_url", "") or "",
            created_at=data.get("created_at", "") or "",
            author_login=(data.get("user") or {}).get("login", "") or "",
        )

    def list_workflow_runs(
        self, *, workflow_filename: str, limit: int
    ) -> List[WorkflowRunView]:
        url = (
            f"{self.API_ROOT}/repos/{self.repo}/actions/workflows/"
            f"{workflow_filename}/runs"
        )
        try:
            resp = self._get(url, params={"per_page": str(min(100, max(1, limit)))})
        except VerifierError as e:
            # If the workflow file isn't present yet (404) we treat it as zero
            # runs rather than a hard failure — the check will report missing.
            if "404" in str(e):
                return []
            raise
        data = resp.json()
        runs = data.get("workflow_runs") or []
        out: List[WorkflowRunView] = []
        for r in runs[:limit]:
            out.append(
                WorkflowRunView(
                    id=int(r.get("id", 0)),
                    name=r.get("name", "") or "",
                    event=r.get("event", "") or "",
                    status=r.get("status", "") or "",
                    conclusion=r.get("conclusion") or "",
                    created_at=r.get("created_at", "") or "",
                    head_branch=r.get("head_branch", "") or "",
                    html_url=r.get("html_url", "") or "",
                )
            )
        return out

    def get_file_text(self, path: str, *, ref: Optional[str] = None) -> str:
        url = f"{self.API_ROOT}/repos/{self.repo}/contents/{path}"
        params = {"ref": ref} if ref else None
        resp = self._get(url, params=params)
        data = resp.json()
        if isinstance(data, list):
            raise VerifierError(f"{path} is a directory, not a file")
        encoding = data.get("encoding")
        if encoding == "base64":
            return base64.b64decode(data.get("content") or "").decode(
                "utf-8", errors="replace"
            )
        # Some endpoints return raw content for small files.
        return data.get("content") or ""

    def list_dir(self, path: str, *, ref: Optional[str] = None) -> List[str]:
        url = f"{self.API_ROOT}/repos/{self.repo}/contents/{path}"
        params = {"ref": ref} if ref else None
        try:
            resp = self._get(url, params=params)
        except VerifierError as e:
            if "404" in str(e):
                return []
            raise
        data = resp.json()
        if not isinstance(data, list):
            return []
        return [
            entry.get("name", "")
            for entry in data
            if entry.get("type") in ("file", "dir") and entry.get("name")
        ]


# --------------------------------------------------------------------------- #
# Fake implementation (tests)
# --------------------------------------------------------------------------- #


class FakeGitHubReadAPI:
    """In-memory read-only GitHub API for tests.

    Pre-seed the fake with whatever state you want the verifier to observe.
    Every method records it was called for assertions.
    """

    def __init__(
        self,
        *,
        repo: str = "test-owner/test-repo",
        owner_login: str = "test-owner",
        issues: Optional[Dict[int, IssueView]] = None,
        workflow_runs: Optional[Dict[str, List[WorkflowRunView]]] = None,
        files: Optional[Dict[str, str]] = None,
        dirs: Optional[Dict[str, List[str]]] = None,
        missing_files: Optional[List[str]] = None,
    ) -> None:
        self.repo = repo
        self.owner_login = owner_login
        self._issues = dict(issues or {})
        self._runs = dict(workflow_runs or {})
        self._files = dict(files or {})
        self._dirs = dict(dirs or {})
        self._missing = set(missing_files or [])
        self.calls: List[str] = []

    def get_repo_owner(self) -> str:
        self.calls.append("get_repo_owner")
        return self.owner_login

    def get_issue(self, number: int) -> IssueView:
        self.calls.append(f"get_issue:{number}")
        if number not in self._issues:
            raise VerifierError(f"GET issue {number} -> 404 not found")
        return self._issues[number]

    def list_workflow_runs(
        self, *, workflow_filename: str, limit: int
    ) -> List[WorkflowRunView]:
        self.calls.append(f"list_workflow_runs:{workflow_filename}")
        return list(self._runs.get(workflow_filename, []))[:limit]

    def get_file_text(self, path: str, *, ref: Optional[str] = None) -> str:
        self.calls.append(f"get_file_text:{path}")
        if path in self._missing:
            raise VerifierError(f"GET {path} -> 404 not found")
        return self._files.get(path, "")

    def list_dir(self, path: str, *, ref: Optional[str] = None) -> List[str]:
        self.calls.append(f"list_dir:{path}")
        return list(self._dirs.get(path, []))
