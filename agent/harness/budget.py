"""Token / call budget tracker.

Aggregates LLM consumption across a run and refuses further calls once
configured caps are exceeded. The tracker is provider-agnostic: providers feed
it estimated tokens after each call. When ``hard_fail_on_exceed`` is true a
``BudgetExceeded`` is raised so the orchestrator can mark the stage failed
rather than silently truncating output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetTracker:
    max_total_input_tokens: int
    max_total_output_tokens: int
    max_total_calls: int
    hard_fail_on_exceed: bool = True

    input_tokens_used: int = 0
    output_tokens_used: int = 0
    calls_used: int = 0
    by_stage: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def check_can_call(self, stage: str = "") -> None:
        if self.calls_used >= self.max_total_calls:
            self._violate(stage, "max_total_calls reached")

    def record(
        self,
        *,
        stage: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.calls_used += 1
        self.input_tokens_used += input_tokens
        self.output_tokens_used += output_tokens
        slot = self.by_stage.setdefault(
            stage or "unknown", {"calls": 0, "in": 0, "out": 0}
        )
        slot["calls"] += 1
        slot["in"] += input_tokens
        slot["out"] += output_tokens
        if self.input_tokens_used > self.max_total_input_tokens:
            self._violate(stage, "max_total_input_tokens exceeded")
        if self.output_tokens_used > self.max_total_output_tokens:
            self._violate(stage, "max_total_output_tokens exceeded")

    def _violate(self, stage: str, reason: str) -> None:
        if self.hard_fail_on_exceed:
            raise BudgetExceeded(f"[{stage}] {reason}")

    def snapshot(self) -> Dict[str, object]:
        return {
            "input_tokens_used": self.input_tokens_used,
            "output_tokens_used": self.output_tokens_used,
            "calls_used": self.calls_used,
            "by_stage": self.by_stage,
            "limits": {
                "max_total_input_tokens": self.max_total_input_tokens,
                "max_total_output_tokens": self.max_total_output_tokens,
                "max_total_calls": self.max_total_calls,
            },
        }
