"""Replay stub.

Reads a trace JSONL and reconstructs a coarse view of what happened. The MVP
implementation reports per-stage status, LLM call counts, and total latency.
Future work: deterministic re-execution from a chosen step, using cached LLM
outputs keyed by ``prompt_hash``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def replay(trace_path: str) -> Dict[str, Any]:
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"trace not found: {trace_path}")
    events: List[Dict[str, Any]] = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    stages: Dict[str, Dict[str, Any]] = {}
    llm_calls: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    for e in events:
        et = e.get("event")
        if et == "stage":
            stages.setdefault(e["stage"], []).append(e)
        elif et == "llm_call":
            llm_calls.append(e)
        elif et == "tool_call":
            tool_calls.append(e)

    summary = {
        "trace_path": trace_path,
        "event_count": len(events),
        "llm_call_count": len(llm_calls),
        "tool_call_count": len(tool_calls),
        "stages": {
            name: hist[-1].get("status") if hist else "unknown"
            for name, hist in stages.items()
        },
        "total_input_tokens_est": sum(c.get("input_tokens_est", 0) for c in llm_calls),
        "total_output_tokens_est": sum(
            c.get("output_tokens_est", 0) for c in llm_calls
        ),
        "total_latency_ms": sum(c.get("latency_ms", 0) for c in llm_calls),
    }
    return summary
