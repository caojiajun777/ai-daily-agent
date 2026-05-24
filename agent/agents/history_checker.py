"""History Checker — detect events already reported in recent daily reports.

Best-effort: reads local artifacts, GitHub Issues, or falls back gracefully.
"""

from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from typing import List, Optional


def load_recent_titles(
    artifacts_dir: str = "artifacts",
    window_days: int = 7,
    repo: str = "",
    token: str = "",
    exclude_date: str = "",
) -> List[str]:
    """Load recent published item titles/URLs. Best-effort, never raises.

    GitHub Actions runs do not retain yesterday's local artifacts by default,
    so the GitHub Issues fallback must parse the issue body, not just the issue
    title ("AI 日报 2026-05-24"). The returned list intentionally contains
    both item titles and source URLs; downstream scorers can use either signal.
    """
    titles: List[str] = []

    # 1. Try local artifacts/drafts.
    drafts_dir = os.path.join(artifacts_dir, "drafts")
    if os.path.isdir(drafts_dir):
        for fname in sorted(os.listdir(drafts_dir), reverse=True):
            if not fname.endswith(".json"):
                continue
            if exclude_date and fname.startswith(exclude_date):
                continue
            try:
                path = os.path.join(drafts_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    draft = json.load(f)
                title = draft.get("title", "")
                if title:
                    titles.append(title)
                for sec in draft.get("sections", []):
                    for item in sec.get("items", []):
                        t = item.get("title", "")
                        if t:
                            titles.append(t)
                        u = item.get("url", "")
                        if u:
                            titles.append(u)
            except Exception:
                pass
        if titles:
            return titles[:200]

    # 2. Fallback: try GitHub Issues API.
    if repo and token:
        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "report-agent-history",
            }
            issues = _fetch_recent_issues(repo, headers, window_days, exclude_date=exclude_date)
            if not issues:
                issues = _fetch_recent_issues(repo, headers, window_days, labels="", exclude_date=exclude_date)
            for issue in issues:
                titles.extend(_extract_issue_history_entries(issue))
        except Exception:
            pass

    return _dedupe_keep_order(titles)[:300]


def check_already_reported(
    canonical_title: str,
    history_titles: List[str],
    threshold: float = 0.65,
) -> bool:
    """Check if a canonical title overlaps with any historical title."""
    if not history_titles:
        return False
    norm = _norm(canonical_title)
    for ht in history_titles:
        sim = SequenceMatcher(None, norm, _norm(ht)).ratio()
        if sim >= threshold:
            return True
    return False


def is_meaningful_update(
    canonical_title: str,
    summary: str,
    history_titles: List[str],
) -> bool:
    """If partially overlapping, check if this is a meaningful update vs repeat."""
    text = (canonical_title + " " + summary).lower()
    update_signals = [
        "update", "upgrade", "release", "launch", "publish",
        "benchmark", "price", "pricing", "github", "repo",
        "rollout", "deprecate", "security", "patch", "fix",
        "更新", "发布", "升级", "上线", "开源", "降价",
    ]
    return any(s in text for s in update_signals)


def _norm(text: str) -> str:
    return re.sub(r"[^\w一-鿿]", "", text.lower())


def _fetch_recent_issues(
    repo: str,
    headers: dict,
    window_days: int,
    *,
    labels: str = "agent-generated",
    exclude_date: str = "",
) -> List[dict]:
    import httpx
    from datetime import datetime, timedelta, timezone

    params = {
        "state": "all",
        "per_page": 20,
        "sort": "created",
        "direction": "desc",
    }
    if labels:
        params["labels"] = labels
    r = httpx.get(
        f"https://api.github.com/repos/{repo}/issues",
        params=params,
        headers=headers,
        timeout=15.0,
    )
    if r.status_code != 200:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, window_days))
    out = []
    for issue in r.json():
        if "pull_request" in issue:
            continue
        created = issue.get("created_at") or ""
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if created_dt < cutoff:
                continue
        except Exception:
            pass
        title = issue.get("title", "")
        if exclude_date and exclude_date in title:
            continue
        if title.startswith("AI 日报") or "AI 日报" in title:
            out.append(issue)
    return out


_MD_LINK_RE = re.compile(r"\[(?:#\d+\s*)?([^\]]+?)\]\((https?://[^)]+)\)")


def _extract_issue_history_entries(issue: dict) -> List[str]:
    entries = []
    title = issue.get("title", "")
    if title:
        entries.append(title)
    body = issue.get("body") or ""
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        # Overview bullets and detail headings both use Markdown links.
        for m in _MD_LINK_RE.finditer(line):
            linked_title = _strip_markdown_title(m.group(1))
            url = m.group(2).strip()
            if linked_title and not linked_title.startswith("原文"):
                entries.append(linked_title)
            if url:
                entries.append(url)
                if linked_title:
                    entries.append(f"{linked_title} {url}")
    return entries


def _strip_markdown_title(title: str) -> str:
    title = re.sub(r"^\s*#?\d+[\s.、:-]*", "", title or "").strip()
    return re.sub(r"\s+", " ", title)


def _dedupe_keep_order(values: List[str]) -> List[str]:
    out = []
    seen = set()
    for value in values:
        value = (value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
