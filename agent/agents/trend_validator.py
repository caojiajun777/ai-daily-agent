"""Trend Validator — evidence and rules-based validation of LLM findings.

Prevents over-interpretation: single-day spikes ≠ trends,
big-company frequency ≠ structural shift, low-evidence claims ≠ high confidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from agent.schemas import TrendFinding, TrendReport


def validate_report(
    report: TrendReport,
    *,
    valid_event_ids: Set[str],
    window_days: int,
    metrics_by_group: Optional[Dict[str, Dict[str, Any]]] = None,
) -> TrendReport:
    """Validate and potentially downgrade findings in a TrendReport.

    Returns the (possibly modified) report.
    """
    warnings: List[str] = []

    if window_days < 7:
        report.data_quality_notes = (
            (report.data_quality_notes or "") +
            f" Short window ({window_days}d): structural claims disabled. "
        )

    valid_findings: List[TrendFinding] = []
    for f in report.findings:
        m = (metrics_by_group or {}).get(f.trend_id, {})
        f, fw = _validate_finding(f, valid_event_ids, window_days, m)
        warnings.extend(fw)
        if f:
            valid_findings.append(f)

    report.findings = valid_findings

    # Validate weak_signals.
    valid_weak: List[TrendFinding] = []
    for f in report.weak_signals:
        f, fw = _validate_finding(f, valid_event_ids, window_days, {}, allow_weak=True)
        warnings.extend(fw)
        if f:
            valid_weak.append(f)
    report.weak_signals = valid_weak

    report.validation_warnings = list(set(warnings))
    return report


def _validate_finding(
    f: TrendFinding,
    valid_event_ids: Set[str],
    window_days: int,
    metrics: Dict[str, Any],
    allow_weak: bool = False,
):
    warnings: List[str] = []

    # 1. evidence_event_ids must exist.
    valid_ev = [eid for eid in f.evidence_event_ids if eid in valid_event_ids]
    removed = len(f.evidence_event_ids) - len(valid_ev)
    if removed > 0:
        warnings.append(f"{f.trend_id}: removed {removed} invalid event_ids")
    f.evidence_event_ids = valid_ev

    # 2. High confidence needs >= 3 evidence events.
    ev_count = metrics.get("event_count", len(valid_ev))
    if f.confidence == "high" and ev_count < 3:
        f.confidence = "medium"
        f.risk_of_overinterpretation = (
            (f.risk_of_overinterpretation or "") +
            " Downgraded: fewer than 3 evidence events. "
        )
        warnings.append(f"{f.trend_id}: high→medium (insufficient events)")

    # 3. active_days < 2 cannot be high confidence.
    active_days = metrics.get("active_days", 1)
    if f.confidence == "high" and active_days < 2:
        f.confidence = "medium"
        f.risk_of_overinterpretation += " Downgraded: single-day spike. "
        warnings.append(f"{f.trend_id}: downgraded (single-day spike)")

    # 4. days < 7 → no confirmed_trend or structural_movement.
    if window_days < 7 and f.window_type in ("confirmed_trend", "structural_movement"):
        f.window_type = "short_signal"
        warnings.append(
            f"{f.trend_id}: window_type→short_signal (window < 7 days)"
        )

    # 5. source_diversity < 0.3 cannot be high confidence.
    src_div = metrics.get("source_diversity", 0.5)
    if f.confidence == "high" and src_div < 0.3:
        f.confidence = "medium"
        f.risk_of_overinterpretation += " Downgraded: low source diversity. "
        warnings.append(f"{f.trend_id}: downgraded (low source diversity)")

    # 6. novelty_ratio < 0.3 for rising trend → downgrade or warn.
    novelty = metrics.get("novelty_ratio", 0.5)
    if f.direction == "rising" and novelty < 0.3:
        f.risk_of_overinterpretation = (
            (f.risk_of_overinterpretation or "") +
            " Low novelty ratio for rising trend — may be repeated reporting, not new signal. "
        )
        if f.confidence == "high":
            f.confidence = "medium"
        warnings.append(f"{f.trend_id}: low novelty for rising trend")

    # 7. No evidence events at all → move to noise or drop.
    if not valid_ev and not allow_weak:
        f.confidence = "low"
        f.trend_type = "noise"
        warnings.append(f"{f.trend_id}: no valid evidence → marked as noise")

    return f, warnings


def metrics_only_report(
    *,
    days: int,
    start_date: str,
    end_date: str,
    total_events: int,
) -> TrendReport:
    """Generate a minimal report when LLM is unavailable."""
    from datetime import datetime, timezone

    return TrendReport(
        report_id=f"fallback-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
        generated_at=datetime.now(timezone.utc).isoformat(),
        days=days, start_date=start_date, end_date=end_date,
        headline_summary="LLM skipped — metrics-only report.",
        findings=[],
        total_events=total_events,
        metrics_fallback_used=True,
        data_quality_notes="LLM call failed; report generated from rule-based metrics only.",
    )
