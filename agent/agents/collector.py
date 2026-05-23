"""Collector agent.

Walks the configured source list, invokes each adapter, normalizes the output,
and returns a flat list of ``RawItem``. No LLM calls.

Failures per-source are isolated: one broken feed shouldn't sink the run. We
log the failure to the tracer and continue.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from agent.harness.trace import Tracer
from agent.sources.base import RawItem, build_source


def collect(
    source_specs: List[Dict[str, Any]],
    *,
    tracer: Tracer,
    default_max_per_source: int = 15,
) -> List[RawItem]:
    out: List[RawItem] = []
    for spec in source_specs:
        if spec.get("enabled", True) is False:
            tracer.log_tool_call(
                name=f"source.{spec.get('type')}",
                args_summary=f"id={spec.get('id', '<unknown>')}",
                status="skipped",
                latency_ms=0,
                error="disabled",
                stage="collect",
            )
            continue
        sid = spec.get("id", "<unknown>")
        max_items = int(spec.get("max_items", default_max_per_source))
        t0 = time.time()
        try:
            adapter = build_source(spec)
            items = adapter.fetch(max_items=max_items)
            for item in items:
                _enrich_source_metadata(item, spec)
            out.extend(items)
            tracer.log_tool_call(
                name=f"source.{spec.get('type')}",
                args_summary=f"id={sid} url={spec.get('url','')}",
                status="ok",
                latency_ms=int((time.time() - t0) * 1000),
                stage="collect",
            )
        except NotImplementedError as e:
            tracer.log_tool_call(
                name=f"source.{spec.get('type')}",
                args_summary=f"id={sid}",
                status="skipped",
                latency_ms=int((time.time() - t0) * 1000),
                error=str(e),
                stage="collect",
            )
        except Exception as e:
            tracer.log_tool_call(
                name=f"source.{spec.get('type')}",
                args_summary=f"id={sid}",
                status="error",
                latency_ms=int((time.time() - t0) * 1000),
                error=str(e),
                stage="collect",
            )
    # Post-process: transfer metadata for AIHot-mapped sources to their real config tiers.
    _transfer_aihot_metadata(out, source_specs)
    return out


def _enrich_source_metadata(item: RawItem, spec: Dict[str, Any]) -> None:
    """Attach config-level source quality metadata to each raw item."""
    item.content_type = spec.get("content_type", item.content_type)
    item.source_tier = spec.get("source_tier", item.source_tier)
    item.reliability = spec.get("reliability", item.reliability)
    item.evidence_type = spec.get("evidence_type", item.evidence_type)
    item.confidence = spec.get("default_confidence", item.confidence)
    item.section_hint = spec.get("section_hint", item.section_hint)


def _transfer_aihot_metadata(items: List[RawItem], source_specs: List[Dict[str, Any]]) -> None:
    """For items mapped through AIHot, look up the target source's real metadata.

    AIHot adapter maps external source names to our own source_ids (e.g.
    'The Decoder' → 'the_decoder'). The collector enriches these items with
    the AIHot source spec's own metadata (tier_3_pulse_noise), which is
    wrong — the target source may have a higher tier.
    """
    spec_by_id: Dict[str, Dict[str, Any]] = {
        s["id"]: s for s in source_specs if "id" in s
    }
    for item in items:
        target_spec = spec_by_id.get(item.source_id)
        if target_spec is None:
            continue
        # Only override if the item has the AIHot aggregator's low tier
        # and the target source has a higher tier.
        if item.source_tier not in ("", "tier_3_pulse_noise"):
            continue
        target_tier = target_spec.get("source_tier", "")
        if not target_tier:
            continue
        item.source_tier = target_tier
        item.reliability = target_spec.get("reliability", item.reliability)
        item.confidence = target_spec.get("default_confidence", item.confidence)
        item.evidence_type = target_spec.get("evidence_type", item.evidence_type)
        item.section_hint = target_spec.get("section_hint", item.section_hint)
        item.content_type = target_spec.get("content_type", item.content_type)
