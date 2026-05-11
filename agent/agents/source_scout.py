"""Unified Source Scout — combines all discovery channels into one pipeline.

Three discovery channels, one output:

  Channel 1 — LLM semantic discovery  (``discover_sources``)
    LLM knows "what exists" — generates candidate RSS + X sources
    by topic and coverage-gap analysis.

  Channel 2 — Content-link diffusion   (``diffuse_sources``)
    Extracts outbound domains from collected items, probes for RSS.
    "Sites you read link to these other sites."

  Channel 3 — Social-graph diffusion   (``diffuse_sources``)
    Traverses X follow graph from seed accounts.
    "People you trust also follow X."

Cross-channel boost:
  A source discovered by 2+ independent channels gets a +0.15
  confidence bonus — it's been "corroborated" by different signals.

Usage:
  python -m agent.cli scout --topic broad --run-id 2026-05-10
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.agents.source_discoverer import (
    CandidateSource,
    _extract_json,
    _validate_rss,
    _validate_x,
)
from agent.llm import LLMProvider
from agent.harness.trace import Tracer


@dataclass
class ScoutReport:
    topic: str
    run_ts: str
    channels_used: List[str] = field(default_factory=list)
    candidates_total: int = 0
    candidates_validated: int = 0
    candidates_passed: int = 0
    cross_boosted: int = 0
    passed: List[CandidateSource] = field(default_factory=list)
    rejected: List[CandidateSource] = field(default_factory=list)
    channel_details: Dict[str, Dict[str, int]] = field(default_factory=dict)
    yaml_snippet: str = ""


def scout_sources(
    *,
    topic: str = "broad",
    provider: LLMProvider,
    config_path: str,
    collected_items: Optional[List[Any]] = None,
    tracer: Optional[Tracer] = None,
    max_per_channel: int = 12,
    min_score: float = 0.35,
    cross_boost: float = 0.12,
) -> ScoutReport:
    """Run all discovery channels and merge results with cross-validation."""

    report = ScoutReport(
        topic=topic,
        run_ts=datetime.now(timezone.utc).isoformat(),
    )

    # Dict for dedup: canonical_key → list of (channel, CandidateSource)
    unified: Dict[str, List[Tuple[str, CandidateSource]]] = defaultdict(list)

    # ── Channel 1: LLM semantic discovery ────────────────────────────
    try:
        from agent.agents.source_discoverer import discover_sources

        disc_report = discover_sources(
            topic=topic,
            provider=provider,
            existing_config_path=config_path,
            tracer=tracer,
            max_candidates=max_per_channel,
            min_score=min_score,
        )
        report.channels_used.append("llm")
        report.channel_details["llm"] = {
            "generated": disc_report.candidates_generated,
            "passed": disc_report.candidates_passed,
        }
        for c in disc_report.passed:
            key = _canonical_key(c)
            unified[key].append(("llm", c))
    except Exception as e:
        if tracer:
            tracer.log("scout_channel_error", channel="llm", error=str(e))

    # ── Channel 2: Content-link diffusion ────────────────────────────
    if collected_items:
        try:
            from agent.agents.source_diffuser import diffuse_sources
            from agent.sources.base import RawItem

            # Convert items to RawItem if needed.
            raw_items: List[Any] = []
            for it in collected_items:
                if isinstance(it, RawItem):
                    raw_items.append(it)
                elif isinstance(it, dict):
                    raw_items.append(RawItem(
                        source_id=it.get("source_name", it.get("source_id", "")),
                        source_type="rss",
                        title=it.get("title", ""),
                        url=it.get("source_url", it.get("url", "")),
                        summary=it.get("summary", ""),
                        published_at=it.get("published_at") or "",
                    ))

            diff_result = diffuse_sources(
                config_path=config_path,
                collected_items=raw_items if raw_items else None,
            )
            link_report = diff_result.get("content_links")
            if link_report and link_report.passed:
                report.channels_used.append("content_link")
                report.channel_details["content_link"] = {
                    "generated": link_report.candidates_discovered,
                    "passed": link_report.candidates_passed,
                }
                for ds in link_report.passed:
                    c = _diffused_to_candidate(ds)
                    key = _canonical_key(c)
                    unified[key].append(("content_link", c))
        except Exception as e:
            if tracer:
                tracer.log("scout_channel_error", channel="content_link", error=str(e))

    # ── Channel 3: Social-graph diffusion ────────────────────────────
    if os.getenv("X_BEARTER_TOKEN", ""):
        try:
            from agent.agents.source_diffuser import diffuse_sources as diffuse

            # Only run graph mode.
            diff_result = diffuse(
                config_path=config_path,
                collected_items=None,
            )
            graph_report = diff_result.get("social_graph")
            if graph_report and graph_report.passed:
                report.channels_used.append("social_graph")
                report.channel_details["social_graph"] = {
                    "generated": graph_report.candidates_discovered,
                    "passed": graph_report.candidates_passed,
                }
                for ds in graph_report.passed:
                    c = _diffused_to_candidate(ds)
                    key = _canonical_key(c)
                    unified[key].append(("social_graph", c))
        except Exception as e:
            if tracer:
                tracer.log("scout_channel_error", channel="social_graph", error=str(e))

    # ── Merge + cross-validate ───────────────────────────────────────
    report.candidates_total = len(unified)

    for key, entries in unified.items():
        channels = [ch for ch, _ in entries]
        # Use the first candidate as the base; take best validation from all.
        base = entries[0][1]

        # Cross-channel boost.
        boost = cross_boost * (len(channels) - 1)  # 0 for 1 channel, +0.12 for 2, +0.24 for 3
        if boost > 0:
            report.cross_boosted += 1

        # Inherit the best validation result across channels.
        for ch, c in entries:
            if c.reachable and not base.reachable:
                base = c
            if c.freshness_score > base.freshness_score:
                base.freshness_score = c.freshness_score
            if c.relevance_score > base.relevance_score:
                base.relevance_score = c.relevance_score

        # Tag with discovery method.
        channel_tags = ",".join(sorted(set(channels)))
        base.reason = f"[{channel_tags}] {base.reason}"

        # Recalculate overall score with cross-channel boost.
        base.overall_score = min(1.0,
            base.freshness_score * 0.35
            + base.relevance_score * 0.35
            + base.uniqueness_score * 0.30
            + boost
        )

        if base.reachable and base.overall_score >= min_score:
            report.passed.append(base)
        else:
            report.rejected.append(base)

    report.passed.sort(key=lambda c: c.overall_score, reverse=True)
    report.candidates_validated = len(report.passed) + len(report.rejected)
    report.candidates_passed = len(report.passed)

    if report.passed:
        report.yaml_snippet = _render_scout_yaml(report.passed)

    if tracer:
        tracer.log(
            "scout_complete",
            topic=topic,
            channels=report.channels_used,
            total=report.candidates_total,
            passed=report.candidates_passed,
            cross_boosted=report.cross_boosted,
        )

    return report


# ── Helpers ────────────────────────────────────────────────────────────────


def _canonical_key(c: CandidateSource) -> str:
    """Unique key for dedup across channels."""
    if c.source_type == "rss" and c.url:
        from agent.agents.source_diffuser import _domain
        return f"rss:{_domain(c.url)}"
    if c.source_type == "x" and c.username:
        return f"x:{c.username.lower().lstrip('@')}"
    return f"unknown:{c.name}"


def _diffused_to_candidate(ds) -> CandidateSource:
    """Convert a DiffusedSource to a unified CandidateSource."""
    return CandidateSource(
        name=ds.name,
        source_type=ds.source_type,
        url=ds.url,
        username=ds.username,
        account_type=ds.account_type,
        language=ds.language,
        category=ds.category,
        reason=ds.reason,
        validated=ds.validated,
        reachable=ds.reachable,
        freshness_score=ds.freshness_score,
        relevance_score=ds.relevance_score,
        uniqueness_score=1.0,
        overall_score=ds.overall_score,
        validation_note=ds.validation_note,
    )


def _render_scout_yaml(candidates: List[CandidateSource]) -> str:
    lines = ["# ── Unified Source Scout results (multi-channel) ──", ""]
    for c in candidates:
        lines.append(f"# [{c.overall_score:.2f}] {c.reason}")
        if c.source_type == "rss":
            slug = re.sub(r"[^a-z0-9_]", "_", c.name.lower().replace(" ", "_"))[:30]
            lines.append(f"- id: \"scout_{slug}\"")
            lines.append(f"  type: \"rss\"")
            lines.append(f"  url: \"{c.url}\"")
        elif c.source_type == "x":
            slug = re.sub(r"[^a-z0-9_]", "_", (c.username or c.name).lower())[:30]
            lines.append(f"- id: \"scout_x_{slug}\"")
            lines.append(f"  type: \"x\"")
            lines.append(f"  username: \"{c.username}\"")
            lines.append(f"  account_type: \"{c.account_type}\"")
        lines.append(f"  weight: {0.6 + c.overall_score * 0.4:.1f}")
        lines.append(f"  max_items: {max(3, int(8 * c.overall_score))}")
        lines.append("")
    return "\n".join(lines)
