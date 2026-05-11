"""Source health & pipeline monitoring.

Checks RSS feed freshness, tracks pipeline success/failure patterns,
and generates alerts for actionable issues.

Integration points:
  - Web dashboard:   /monitor page shows live health status
  - Daily workflow:   post-run health-check step
  - Weekly workflow:  scheduled deep-check of all sources
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ── Config ─────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "default.yaml"
)
ARTIFACTS = os.environ.get(
    "ARTIFACTS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "artifacts"),
)

# DeepSeek v4-pro pricing (per 1K tokens).
COST_PER_1K_IN = 0.00028   # $0.28 / 1M tokens
COST_PER_1K_OUT = 0.00110  # $1.10 / 1M tokens


@dataclass
class SourceHealth:
    source_id: str
    source_type: str
    url: str = ""
    username: str = ""
    status: str = "unknown"       # ok | stale | dead | skipped
    last_seen: str = ""           # ISO date of newest entry we fetched
    days_since_update: int = 99
    current_weight: float = 0.0
    note: str = ""


@dataclass
class PipelineHealth:
    date: str
    status: str                   # ok | failed | needs_review
    collect_items: int = 0
    curated_items: int = 0
    draft_items: int = 0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_est: float = 0.0
    duration_s: float = 0.0


@dataclass
class MonitoringReport:
    checked_at: str
    sources: List[SourceHealth] = field(default_factory=list)
    pipelines: List[PipelineHealth] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# Source health
# ═══════════════════════════════════════════════════════════════════════


def check_source_health() -> List[SourceHealth]:
    """Check all configured RSS and X sources for freshness."""
    sources: List[SourceHealth] = []
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        source_specs = cfg.get("sources", [])
    except Exception:
        return sources

    for spec in source_specs:
        sid = spec.get("id", "?")
        stype = spec.get("type", "?")

        if stype == "rss":
            health = _check_rss_health(sid, spec.get("url", ""))
        elif stype == "x":
            health = _check_x_health(sid, spec.get("username", ""))
        elif stype == "aihot":
            health = SourceHealth(
                source_id=sid, source_type="aihot",
                url=spec.get("url", ""), status="skipped",
                note="AI HOT scraper — health checked at runtime",
            )
        else:
            continue
        health.current_weight = float(spec.get("weight", 0))
        sources.append(health)

    return sources


def _check_rss_health(source_id: str, url: str) -> SourceHealth:
    h = SourceHealth(source_id=source_id, source_type="rss", url=url)
    if not url:
        h.status = "dead"; h.note = "no URL configured"
        return h
    try:
        import feedparser
        parsed = feedparser.parse(url)
        if parsed.bozo and not parsed.entries:
            h.status = "dead"
            h.note = f"parse error: {str(parsed.bozo_exception)[:100]}"
            return h
        entries = parsed.entries[:10]
        if not entries:
            h.status = "stale"; h.note = "feed returned 0 entries"
            h.days_since_update = 99
            return h
        newest = _newest_entry_date(entries)
        h.last_seen = newest
        if newest:
            try:
                newest_dt = datetime.fromisoformat(newest.replace("Z", "+00:00"))
                days = (datetime.now(timezone.utc) - newest_dt).days
                h.days_since_update = days
                if days <= 2:
                    h.status = "ok"
                    h.note = f"active, {len(entries)} recent entries"
                elif days <= 5:
                    h.status = "stale"
                    h.note = f"{days}d since last update"
                else:
                    h.status = "dead"
                    h.note = f"{days}d since last update — likely abandoned"
            except Exception:
                h.status = "ok"
        else:
            h.status = "ok"; h.note = f"{len(entries)} entries, date unknown"
    except Exception as e:
        h.status = "dead"; h.note = str(e)[:100]
    return h


def _check_x_health(source_id: str, username: str) -> SourceHealth:
    h = SourceHealth(source_id=source_id, source_type="x", username=username)
    token = os.getenv("X_BEARER_TOKEN", "") or os.getenv("X_AUTH_TOKEN", "")
    if not token:
        h.status = "skipped"; h.note = "no X token configured"
        return h
    # Quick check via API if token available.
    try:
        import httpx
        client = httpx.Client(
            base_url="https://api.x.com/2",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "report-agent-monitor"},
            timeout=10.0,
        )
        resp = client.get(f"/users/by/username/{username}")
        if resp.status_code == 200:
            uid = resp.json().get("data", {}).get("id")
            h.note = f"user_id={uid}, account exists"
            h.status = "ok"
        elif resp.status_code == 429:
            h.status = "skipped"; h.note = "rate limited"
        else:
            h.status = "stale"; h.note = f"HTTP {resp.status_code}"
        client.close()
    except Exception as e:
        h.status = "skipped"; h.note = f"API error: {str(e)[:80]}"
    return h


# ═══════════════════════════════════════════════════════════════════════
# Pipeline health
# ═══════════════════════════════════════════════════════════════════════


def check_pipeline_health(days: int = 14) -> List[PipelineHealth]:
    """Read recent pipeline reports and compute health metrics."""
    results: List[PipelineHealth] = []
    reports_dir = os.path.join(ARTIFACTS, "reports")
    if not os.path.isdir(reports_dir):
        return results

    for fname in sorted(os.listdir(reports_dir), reverse=True):
        if not fname.endswith(".json"):
            continue
        if any(x in fname for x in ("publish", "semantic", "repair", "scout")):
            continue
        date = fname.replace(".json", "")
        try:
            with open(os.path.join(reports_dir, fname), "r", encoding="utf-8") as f:
                r = json.load(f)
        except Exception:
            continue

        stages = r.get("stages", {})
        collect_items = stages.get("collect", {}).get("meta", {}).get("raw_item_count", 0)
        curated_items = stages.get("curate", {}).get("meta", {}).get("curated_item_count", 0)
        eval_meta = stages.get("eval", {}).get("meta", {})
        draft_items = eval_meta.get("item_count", 0)

        budget = r.get("budget", {})
        in_tok = budget.get("input_tokens_used", 0)
        out_tok = budget.get("output_tokens_used", 0)
        calls = budget.get("calls_used", 0)

        cost = (in_tok / 1000) * COST_PER_1K_IN + (out_tok / 1000) * COST_PER_1K_OUT

        duration = r.get("ended_at", 0) - r.get("started_at", 0)

        status = "ok"
        if r.get("is_failed"):
            status = "failed"
        elif r.get("needs_human_review"):
            status = "needs_review"

        results.append(PipelineHealth(
            date=date, status=status,
            collect_items=collect_items, curated_items=curated_items,
            draft_items=draft_items, llm_calls=calls,
            tokens_in=in_tok, tokens_out=out_tok,
            cost_est=round(cost, 4), duration_s=round(duration, 1),
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════
# Alerting
# ═══════════════════════════════════════════════════════════════════════


def generate_alerts(
    sources: List[SourceHealth],
    pipelines: List[PipelineHealth],
) -> List[str]:
    """Generate human-readable alerts from health data."""
    alerts: List[str] = []

    # Source alerts.
    stale = [s for s in sources if s.status == "stale"]
    dead = [s for s in sources if s.status == "dead"]
    skipped = [s for s in sources if s.status == "skipped"]

    if dead:
        names = ", ".join(s.source_id for s in dead[:5])
        alerts.append(f"DEAD SOURCES ({len(dead)}): {names}")
    if stale:
        names = ", ".join(f"{s.source_id}({s.days_since_update}d)" for s in stale[:5])
        alerts.append(f"STALE SOURCES ({len(stale)}): {names}")
    if skipped:
        names = ", ".join(s.source_id for s in skipped[:3])
        alerts.append(f"SKIPPED ({len(skipped)}): {names}")

    # Pipeline alerts.
    if pipelines:
        recent = pipelines[:7]
        failures = [p for p in recent if p.status == "failed"]
        reviews = [p for p in recent if p.status == "needs_review"]
        if failures:
            alerts.append(f"RECENT FAILURES ({len(failures)}): {', '.join(p.date for p in failures)}")
        if reviews:
            alerts.append(f"NEEDS REVIEW ({len(reviews)}): {', '.join(p.date for p in reviews)}")
        # Consecutive failures.
        streak = 0
        for p in pipelines:
            if p.status == "failed":
                streak += 1
            else:
                break
        if streak >= 2:
            alerts.append(f"CRITICAL: {streak} consecutive pipeline failures!")

    return alerts


def full_monitoring_report() -> MonitoringReport:
    """Run all checks and produce a comprehensive monitoring report."""
    sources = check_source_health()
    pipelines = check_pipeline_health(days=30)
    alerts = generate_alerts(sources, pipelines)

    # Summary.
    ok_sources = sum(1 for s in sources if s.status == "ok")
    stale_sources = sum(1 for s in sources if s.status == "stale")
    dead_sources = sum(1 for s in sources if s.status == "dead")
    recent_pipelines = pipelines[:14]
    recent_ok = sum(1 for p in recent_pipelines if p.status == "ok")
    total_cost = sum(p.cost_est for p in pipelines)
    total_tokens = sum(p.tokens_in + p.tokens_out for p in pipelines)

    return MonitoringReport(
        checked_at=datetime.now(timezone.utc).isoformat(),
        sources=sources,
        pipelines=pipelines,
        alerts=alerts,
        summary={
            "total_sources": len(sources),
            "ok_sources": ok_sources,
            "stale_sources": stale_sources,
            "dead_sources": dead_sources,
            "skipped_sources": len([s for s in sources if s.status == "skipped"]),
            "recent_pipeline_ok": recent_ok,
            "recent_pipeline_total": len(recent_pipelines),
            "total_cost_est": round(total_cost, 4),
            "total_tokens": total_tokens,
            "alert_count": len(alerts),
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _newest_entry_date(entries) -> str:
    for e in entries:
        t = e.get("published_parsed") or e.get("updated_parsed")
        if t:
            try:
                ts = time.mktime(t)
                return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            except Exception:
                pass
    return ""
