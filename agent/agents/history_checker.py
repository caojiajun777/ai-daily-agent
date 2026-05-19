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
) -> List[str]:
    """Load titles from recent daily reports. Best-effort, never raises."""
    titles: List[str] = []

    # 1. Try local artifacts/drafts.
    drafts_dir = os.path.join(artifacts_dir, "drafts")
    if os.path.isdir(drafts_dir):
        for fname in sorted(os.listdir(drafts_dir), reverse=True):
            if not fname.endswith(".json"):
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
            except Exception:
                pass
        if titles:
            return titles[:200]

    # 2. Fallback: try GitHub Issues API.
    if repo and token:
        try:
            import httpx
            r = httpx.get(
                f"https://api.github.com/repos/{repo}/issues",
                params={"state": "all", "labels": "agent-generated",
                        "per_page": 10, "sort": "created", "direction": "desc"},
                headers={"Authorization": f"Bearer {token}",
                         "Accept": "application/vnd.github.v3+json",
                         "User-Agent": "report-agent-history"},
                timeout=15.0,
            )
            for issue in r.json() if r.status_code == 200 else []:
                titles.append(issue.get("title", ""))
        except Exception:
            pass

    return titles[:200]


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
