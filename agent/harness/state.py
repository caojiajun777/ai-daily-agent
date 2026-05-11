"""Run-level state machine.

A pipeline run goes through a fixed sequence of named stages. Each stage records
its status (pending / running / ok / failed / needs_human_review) plus
input/output artifacts and an optional error. The state itself is a plain
dataclass that can be serialized to JSON for replay and reporting.
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    SKIPPED = "skipped"


# Canonical stage order used by the MVP pipeline. The orchestrator walks this
# list; stages can be skipped (e.g. during replay) but never reordered without
# an explicit code change.
STAGES: List[str] = [
    "collect",
    "curate",
    "write",
    "critique",
    "publish",
    "eval",
]


@dataclass
class StageState:
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None
    artifact_path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def mark_running(self) -> None:
        self.status = StageStatus.RUNNING
        self.started_at = time.time()

    def mark_ok(self, artifact_path: Optional[str] = None) -> None:
        self.status = StageStatus.OK
        self.ended_at = time.time()
        if artifact_path is not None:
            self.artifact_path = artifact_path

    def mark_failed(self, error: str) -> None:
        self.status = StageStatus.FAILED
        self.ended_at = time.time()
        self.error = error

    def mark_needs_review(self, error: str) -> None:
        self.status = StageStatus.NEEDS_HUMAN_REVIEW
        self.ended_at = time.time()
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class RunState:
    run_id: str
    date: str  # YYYY-MM-DD, the logical report date
    provider: str
    model: str
    stages: Dict[str, StageState] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    @classmethod
    def new(cls, date: str, provider: str, model: str) -> "RunState":
        run_id = f"{date}-{uuid.uuid4().hex[:8]}"
        stages = {name: StageState(name=name) for name in STAGES}
        return cls(run_id=run_id, date=date, provider=provider, model=model, stages=stages)

    def stage(self, name: str) -> StageState:
        if name not in self.stages:
            raise KeyError(f"unknown stage: {name}")
        return self.stages[name]

    def is_failed(self) -> bool:
        return any(s.status == StageStatus.FAILED for s in self.stages.values())

    def needs_review(self) -> bool:
        return any(
            s.status == StageStatus.NEEDS_HUMAN_REVIEW for s in self.stages.values()
        )

    def finish(self) -> None:
        self.ended_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "date": self.date,
            "provider": self.provider,
            "model": self.model,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "is_failed": self.is_failed(),
            "needs_human_review": self.needs_review(),
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
