"""Append-only JSONL trace.

Every meaningful event in a run gets one JSON line: stage transitions, LLM
calls, tool calls, validation errors. The trace is the single source of truth
for replay and post-hoc analysis. Writes are append-only so a crashed run
still leaves a usable partial trace on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    # Cheap heuristic: ~4 chars/token for English, ~1.5 chars/token for CJK.
    # Good enough for budget tracking without pulling tokenizer deps.
    if not text:
        return 0
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


class Tracer:
    def __init__(self, path: str, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Touch file so consumers can rely on it existing.
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                pass

    def log(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "ts": time.time(),
            "run_id": self.run_id,
            "event": event_type,
        }
        rec.update(fields)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def log_stage(self, stage: str, status: str, **fields: Any) -> None:
        self.log("stage", stage=stage, status=status, **fields)

    def log_llm_call(
        self,
        *,
        provider: str,
        model: str,
        prompt: str,
        output: str,
        latency_ms: int,
        status: str,
        error: Optional[str] = None,
        attempt: int = 1,
        stage: Optional[str] = None,
    ) -> None:
        self.log(
            "llm_call",
            provider=provider,
            model=model,
            prompt_hash=prompt_hash(prompt),
            input_tokens_est=estimate_tokens(prompt),
            output_tokens_est=estimate_tokens(output),
            latency_ms=latency_ms,
            status=status,
            error=error,
            attempt=attempt,
            stage=stage,
        )

    def log_tool_call(
        self,
        *,
        name: str,
        args_summary: str,
        status: str,
        latency_ms: int,
        error: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> None:
        self.log(
            "tool_call",
            name=name,
            args_summary=args_summary,
            status=status,
            latency_ms=latency_ms,
            error=error,
            stage=stage,
        )

    def read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out
