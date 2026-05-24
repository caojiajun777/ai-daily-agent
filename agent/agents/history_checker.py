"""History Checker — detect events already reported in recent daily reports.

Best-effort: reads local artifacts, GitHub Issues, or falls back gracefully.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


def load_recent_titles(
    artifacts_dir: str = "artifacts",
    window_days: int = 7,
    repo: str = "",
    token: str = "",
    exclude_date: str = "",
) -> List[str]:
    entries, _meta = load_recent_history(
        artifacts_dir=artifacts_dir,
        window_days=window_days,
        repo=repo,
        token=token,
        exclude_date=exclude_date,
    )
    return entries


def load_recent_history(
    artifacts_dir: str = "artifacts",
    window_days: int = 7,
    repo: str = "",
    token: str = "",
    exclude_date: str = "",
) -> Tuple[List[str], Dict[str, Any]]:
    """Load recent published item titles/URLs. Best-effort, never raises.

    GitHub Actions runs do not retain yesterday's local artifacts by default,
    so the GitHub Issues fallback must parse the issue body, not just the issue
    title ("AI 日报 2026-05-24"). The returned list intentionally contains
    both item titles and source URLs; downstream scorers can use either signal.
    """
    titles: List[str] = []
    meta: Dict[str, Any] = {
        "history_source": "none",
        "history_entry_count": 0,
        "history_local_entry_count": 0,
        "history_github_entry_count": 0,
    }
    errors: List[str] = []
    ref_date = _reference_date(exclude_date)

    # 1. Try local artifacts/drafts.
    try:
        local_entries = _load_local_draft_entries(
            artifacts_dir=artifacts_dir,
            window_days=window_days,
            reference_date=ref_date,
            exclude_date=exclude_date,
        )
        titles.extend(local_entries)
        meta["history_local_entry_count"] = len(local_entries)
    except Exception as e:
        errors.append(f"local:{type(e).__name__}")

    # 2. Also try GitHub Issues API. Do not skip this when local artifacts
    # exist: the repo contains a few old sample artifacts, while the latest
    # published history lives in issue bodies during CI runs.
    if repo and token:
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "report-agent-history",
            }
            issues = _fetch_recent_issues(
                repo, headers, window_days,
                exclude_date=exclude_date,
                reference_date=ref_date,
            )
            if not issues:
                issues = _fetch_recent_issues(
                    repo, headers, window_days,
                    labels="",
                    exclude_date=exclude_date,
                    reference_date=ref_date,
                )
            github_entries: List[str] = []
            for issue in issues:
                github_entries.extend(_extract_issue_history_entries(issue))
            titles.extend(github_entries)
            meta["history_github_entry_count"] = len(github_entries)
        except Exception as e:
            errors.append(f"github:{type(e).__name__}")

    entries = _dedupe_keep_order(titles)[:300]
    sources = []
    if meta["history_local_entry_count"]:
        sources.append("local")
    if meta["history_github_entry_count"]:
        sources.append("github")
    meta["history_source"] = "+".join(sources) if sources else "none"
    meta["history_entry_count"] = len(entries)
    if errors:
        meta["history_errors"] = errors
    return entries, meta


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
    reference_date: Optional[date] = None,
) -> List[dict]:
    import httpx

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
    ref_date = reference_date or _reference_date(exclude_date)
    cutoff = ref_date - timedelta(days=max(1, window_days))
    out = []
    for issue in r.json():
        if "pull_request" in issue:
            continue
        title = issue.get("title", "")
        issue_date = _date_from_text(title)
        if issue_date:
            if not _within_history_window(issue_date, ref_date, window_days, exclude_date):
                continue
        else:
            created = issue.get("created_at") or ""
            try:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                created_date = created_dt.astimezone(timezone.utc).date()
                if created_date < cutoff or created_date >= ref_date:
                    continue
            except Exception:
                pass
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


def _load_local_draft_entries(
    *,
    artifacts_dir: str,
    window_days: int,
    reference_date: date,
    exclude_date: str,
) -> List[str]:
    drafts_dir = os.path.join(artifacts_dir, "drafts")
    if not os.path.isdir(drafts_dir):
        return []
    entries: List[str] = []
    for fname in sorted(os.listdir(drafts_dir), reverse=True):
        if not fname.endswith(".json"):
            continue
        if exclude_date and fname.startswith(exclude_date):
            continue
        path = os.path.join(drafts_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                draft = json.load(f)
        except Exception:
            continue
        draft_date = _date_from_text(fname) or _date_from_text(str(draft.get("date", "")))
        if not _within_history_window(draft_date, reference_date, window_days, exclude_date):
            continue
        title = draft.get("title", "")
        if title:
            entries.append(title)
        for sec in draft.get("sections", []):
            for item in sec.get("items", []):
                t = item.get("title", "")
                if t:
                    entries.append(t)
                u = item.get("url", "")
                if u:
                    entries.append(u)
    return entries


def _reference_date(exclude_date: str = "") -> date:
    return _date_from_text(exclude_date) or datetime.now(timezone.utc).date()


def _date_from_text(text: str) -> Optional[date]:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


def _within_history_window(
    item_date: Optional[date],
    reference_date: date,
    window_days: int,
    exclude_date: str,
) -> bool:
    if item_date is None:
        return False
    excluded = _date_from_text(exclude_date)
    if excluded and item_date == excluded:
        return False
    cutoff = reference_date - timedelta(days=max(1, window_days))
    return cutoff <= item_date < reference_date
