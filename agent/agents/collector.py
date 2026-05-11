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
        sid = spec.get("id", "<unknown>")
        max_items = int(spec.get("max_items", default_max_per_source))
        t0 = time.time()
        try:
            adapter = build_source(spec)
            items = adapter.fetch(max_items=max_items)
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
    return out
