"""Auto-admit discovered sources into the YAML config.

Reads a scout report JSON, checks for duplicates in default.yaml,
and appends new sources with low initial weight. Idempotent — running
it multiple times won't produce duplicates.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Set
from urllib.parse import urlparse

import yaml


def _existing_domains(sources: List[dict]) -> Set[str]:
    domains: Set[str] = set()
    for s in sources:
        if s.get("type") == "rss" and s.get("url"):
            try:
                d = urlparse(s["url"]).netloc.lower().replace("www.", "")
                domains.add(d)
            except Exception:
                pass
    return domains


def _existing_x_usernames(sources: List[dict]) -> Set[str]:
    usernames: Set[str] = set()
    for s in sources:
        if s.get("type") in ("x", "x_cookie") and s.get("username"):
            usernames.add(s["username"].lower().lstrip("@"))
    return usernames


def _is_duplicate(candidate: dict, domains: Set[str], usernames: Set[str]) -> bool:
    if candidate.get("type") == "rss":
        try:
            d = urlparse(candidate["url"]).netloc.lower().replace("www.", "")
            return d in domains
        except Exception:
            return True  # skip invalid URLs
    if candidate.get("type") == "x":
        u = (candidate.get("username") or "").lower().lstrip("@")
        return u in usernames or not u
    return True


def _render_source_yaml(candidate: dict) -> str:
    """Render a single candidate as a YAML config block."""
    name = candidate.get("name", "unknown")
    slug = re.sub(r"[^a-z0-9_]", "_", name.lower().replace(" ", "_"))[:30]
    score = float(candidate.get("overall_score", candidate.get("score", 0.5)))
    reason = candidate.get("reason", "auto-discovered")

    lines = [f"  # auto-admitted [{score:.2f}] {reason}"]
    if candidate.get("type") == "rss":
        lines.append(f'  - id: "auto_{slug}"')
        lines.append(f'    type: "rss"')
        lines.append(f'    url: "{candidate["url"]}"')
    elif candidate.get("type") == "x":
        uname = candidate.get("username", "").lstrip("@")
        lines.append(f'  - id: "auto_x_{slug}"')
        lines.append(f'    type: "x"')
        lines.append(f'    username: "{uname}"')
        lines.append(f'    account_type: "kol"')
    else:
        return ""

    lines.append(f"    weight: 0.55")
    lines.append(f"    max_items: 5")
    lines.append("")
    return "\n".join(lines)


def auto_admit_from_scout(
    scout_report_path: str,
    config_path: str,
    *,
    dry_run: bool = False,
) -> Dict:
    """Read a scout report and append new sources to the YAML config.

    Returns a dict with stats about what was done.
    """
    if not os.path.exists(scout_report_path):
        return {"status": "skipped", "reason": "scout report not found"}

    with open(scout_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    passed = report.get("passed", [])
    if not passed:
        return {"status": "skipped", "reason": "no candidates passed", "candidates": 0}

    with open(config_path, "r", encoding="utf-8") as f:
        config_text = f.read()

    sources = yaml.safe_load(config_text).get("sources", [])
    domains = _existing_domains(sources)
    usernames = _existing_x_usernames(sources)

    new_sources: List[str] = []
    admitted = 0
    for c in passed:
        if _is_duplicate(c, domains, usernames):
            continue
        block = _render_source_yaml(c)
        if block:
            new_sources.append(block)
            admitted += 1
            # Track domain/username to avoid intra-batch duplicates.
            if c.get("type") == "rss":
                try:
                    domains.add(urlparse(c["url"]).netloc.lower().replace("www.", ""))
                except Exception:
                    pass
            elif c.get("type") == "x":
                usernames.add((c.get("username") or "").lower().lstrip("@"))

    if not new_sources:
        return {"status": "skipped", "reason": "all candidates are duplicates",
                "candidates": len(passed), "admitted": 0}

    if dry_run:
        return {"status": "dry_run", "admitted": admitted, "preview": "\n".join(new_sources)}

    # Append to config.
    separator = "\n  # ═══════════════════════════════════════════════════════════════\n"
    header = "  # Auto-discovered sources (scout pipeline)\n"
    block_text = separator + header + separator + "\n" + "\n".join(new_sources)

    with open(config_path, "a", encoding="utf-8") as f:
        f.write(block_text)

    return {"status": "ok", "admitted": admitted, "candidates": len(passed)}
