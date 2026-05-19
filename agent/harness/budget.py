"""Token / call budget tracker with cost estimation.

Aggregates LLM consumption across a run and refuses further calls once
configured caps are exceeded. Also tracks provider, model, latency and
estimated cost for internal observability (dashboard, not the public report).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


class BudgetExceeded(RuntimeError):
    pass


# ── Cost model ($ per 1K tokens) ──────────────────────────────────────────

_COST_PER_1K = {
    "deepseek": {"in": 0.00014, "out": 0.00028},   # DeepSeek V4: $0.14/$0.28 per 1M
    "qwen":     {"in": 0.0005,  "out": 0.002},     # Qwen: ~$0.50/$2.00 per 1M
    "anthropic":{"in": 0.003,   "out": 0.015},      # Claude: ~$3/$15 per 1M
    "mock":     {"in": 0.0,     "out": 0.0},
    "unknown":  {"in": 0.0005,  "out": 0.002},
}


def _estimate_cost(provider: str, tokens_in: int, tokens_out: int) -> float:
    rates = _COST_PER_1K.get(provider, _COST_PER_1K["unknown"])
    return round((tokens_in / 1000) * rates["in"] + (tokens_out / 1000) * rates["out"], 6)


@dataclass
class BudgetTracker:
    max_total_input_tokens: int
    max_total_output_tokens: int
    max_total_calls: int
    hard_fail_on_exceed: bool = True

    input_tokens_used: int = 0
    output_tokens_used: int = 0
    calls_used: int = 0
    by_stage: Dict[str, Dict[str, object]] = field(default_factory=dict)
    _call_log: List[Dict[str, object]] = field(default_factory=list)
    default_provider: str = "unknown"
    default_model: str = ""

    def check_can_call(self, stage: str = "") -> None:
        if self.calls_used >= self.max_total_calls:
            self._violate(stage, "max_total_calls reached")

    def record(
        self,
        *,
        stage: str,
        input_tokens: int,
        output_tokens: int,
        provider: str = "",
        model: str = "",
        latency_ms: int = 0,
    ) -> None:
        self.calls_used += 1
        self.input_tokens_used += input_tokens
        self.output_tokens_used += output_tokens

        prov = provider or self.default_provider
        mod = model or self.default_model
        slot = self.by_stage.setdefault(
            stage or "unknown", {"calls": 0, "in": 0, "out": 0, "cost": 0.0,
                                 "provider": prov, "model": mod, "latency_ms": 0}
        )
        slot["calls"] = int(slot["calls"]) + 1
        slot["in"] = int(slot["in"]) + input_tokens
        slot["out"] = int(slot["out"]) + output_tokens
        slot["latency_ms"] = int(slot["latency_ms"]) + latency_ms
        if prov and prov != "unknown":
            slot["provider"] = prov
        if mod:
            slot["model"] = mod if mod else slot.get("model", "")

        cost = _estimate_cost(prov, input_tokens, output_tokens)
        slot["cost"] = round(float(slot["cost"]) + cost, 6)

        self._call_log.append({
            "stage": stage, "provider": provider, "model": model,
            "tokens_in": input_tokens, "tokens_out": output_tokens,
            "latency_ms": latency_ms, "cost": cost,
        })

        if self.input_tokens_used > self.max_total_input_tokens:
            self._violate(stage, "max_total_input_tokens exceeded")
        if self.output_tokens_used > self.max_total_output_tokens:
            self._violate(stage, "max_total_output_tokens exceeded")

    def _violate(self, stage: str, reason: str) -> None:
        if self.hard_fail_on_exceed:
            raise BudgetExceeded(f"[{stage}] {reason}")

    def total_cost(self) -> float:
        return round(sum(float(s.get("cost", 0)) for s in self.by_stage.values()), 6)

    def snapshot(self) -> Dict[str, object]:
        return {
            "input_tokens_used": self.input_tokens_used,
            "output_tokens_used": self.output_tokens_used,
            "calls_used": self.calls_used,
            "total_cost_est": self.total_cost(),
            "by_stage": self.by_stage,
            "call_log": self._call_log,
            "limits": {
                "max_total_input_tokens": self.max_total_input_tokens,
                "max_total_output_tokens": self.max_total_output_tokens,
                "max_total_calls": self.max_total_calls,
            },
        }
