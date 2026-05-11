"""Tests for Semantic Duplicate Critic."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pytest

from agent.agents.semantic_duplicate_critic import (
    SemanticDuplicateCriticFailed,
    run_semantic_duplicate_critic,
)
from agent.agents.issue_publisher import evaluate_publish_gate, load_run_artifacts
from agent.harness.budget import BudgetTracker
from agent.harness.trace import Tracer
from agent.llm import LLMMessage
from agent.llm.mock_provider import MockLLMProvider
from agent.schemas import (
    Draft,
    DraftItem,
    DraftSection,
    SemanticDuplicate,
    SemanticDuplicateReport,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _draft_with_items(items: List[Dict[str, str]]) -> Draft:
    """Build a single-section draft from a list of dicts."""
    return Draft(
        date="2026-05-09",
        title="AI 日报 2026-05-09",
        sections=[
            DraftSection(
                heading="测试",
                items=[
                    DraftItem(
                        title=it["title"],
                        summary=it.get("summary", "summary"),
                        url=it.get("url", f"https://example.com/{i}"),
                        source="src",
                    )
                    for i, it in enumerate(items)
                ],
            )
        ],
    )


def _budget() -> BudgetTracker:
    return BudgetTracker(
        max_total_input_tokens=100_000,
        max_total_output_tokens=10_000,
        max_total_calls=20,
        hard_fail_on_exceed=True,
    )


def _tracer(tmp_path) -> Tracer:
    return Tracer(str(tmp_path / "trace.jsonl"), run_id="test-run")


def _no_dup_provider() -> MockLLMProvider:
    def responder(messages: List[LLMMessage]) -> str:
        return json.dumps({"duplicates": []})

    return MockLLMProvider(model="mock-sem", responder=responder)


def _dup_provider(duplicates: List[Dict[str, Any]]) -> MockLLMProvider:
    def responder(messages: List[LLMMessage]) -> str:
        return json.dumps({"duplicates": duplicates})

    return MockLLMProvider(model="mock-sem", responder=responder)


def _invalid_provider() -> MockLLMProvider:
    def responder(messages: List[LLMMessage]) -> str:
        return "这不是 JSON，只是随意文字。"

    return MockLLMProvider(model="mock-sem", responder=responder)


# --------------------------------------------------------------------------- #
# 1. schema valid
# --------------------------------------------------------------------------- #


def test_semantic_duplicate_schema_valid():
    dup = SemanticDuplicate(
        item_a_id="#2",
        item_b_id="#10",
        item_a_title="可信联系人功能上线",
        item_b_title="WhatsApp 推出可信联系人",
        reason="两条均报道同一产品发布事件",
        severity="high",
    )
    report = SemanticDuplicateReport(
        date="2026-05-09",
        run_id="r-1",
        duplicates=[dup],
        ok=False,
        checked_item_count=10,
        provider="deepseek",
    )
    payload = report.model_dump(mode="json")
    revived = SemanticDuplicateReport.model_validate(payload)
    assert len(revived.duplicates) == 1
    assert revived.duplicates[0].severity == "high"
    assert revived.ok is False


# --------------------------------------------------------------------------- #
# 2. no duplicates → ok=True
# --------------------------------------------------------------------------- #


def test_no_duplicates_pass(tmp_path):
    draft = _draft_with_items([
        {"title": "#1 GPT-5 发布", "summary": "OpenAI 发布 GPT-5"},
        {"title": "#2 Gemini 更新", "summary": "Google 更新 Gemini"},
    ])
    report = run_semantic_duplicate_critic(
        draft=draft,
        provider=_no_dup_provider(),
        date="2026-05-09",
        run_id="r-1",
        tracer=_tracer(tmp_path),
        budget=_budget(),
    )
    assert report.ok is True
    assert report.duplicates == []
    assert report.checked_item_count == 2


# --------------------------------------------------------------------------- #
# 3. same event, different titles → detected, ok=False
# --------------------------------------------------------------------------- #


def test_same_event_different_titles_detected(tmp_path):
    draft = _draft_with_items([
        {"title": "#2 Meta AI 可信联系人功能上线", "summary": "要闻报道"},
        {"title": "#10 WhatsApp 推出可信联系人", "summary": "产品应用报道"},
    ])
    dup_data = [
        {
            "item_a_id": "#2",
            "item_b_id": "#10",
            "item_a_title": "#2 Meta AI 可信联系人功能上线",
            "item_b_title": "#10 WhatsApp 推出可信联系人",
            "reason": "两条均报道同一产品功能发布",
            "severity": "high",
        }
    ]
    report = run_semantic_duplicate_critic(
        draft=draft,
        provider=_dup_provider(dup_data),
        date="2026-05-09",
        run_id="r-1",
        tracer=_tracer(tmp_path),
        budget=_budget(),
    )
    assert report.ok is False
    assert len(report.duplicates) == 1
    assert report.duplicates[0].severity == "high"
    assert report.duplicates[0].item_a_id == "#2"


# --------------------------------------------------------------------------- #
# 4. invalid LLM JSON does not pass silently
# --------------------------------------------------------------------------- #


def test_invalid_llm_json_raises(tmp_path):
    draft = _draft_with_items([{"title": "#1 Test", "summary": "s"}])
    with pytest.raises(SemanticDuplicateCriticFailed):
        run_semantic_duplicate_critic(
            draft=draft,
            provider=_invalid_provider(),
            date="2026-05-09",
            run_id="r-1",
            tracer=_tracer(tmp_path),
            budget=_budget(),
        )


# --------------------------------------------------------------------------- #
# 5. publish gate blocks high semantic duplicate
# --------------------------------------------------------------------------- #


def _seed_artifacts_with_sem_dup(
    tmp_path,
    date: str,
    sem_dup_report: SemanticDuplicateReport,
) -> str:
    """Write minimal pipeline artifacts + a semantic duplicate report."""
    drafts = tmp_path / "artifacts" / "drafts"
    reports = tmp_path / "artifacts" / "reports"
    traces = tmp_path / "artifacts" / "traces"
    drafts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)

    draft_payload = {
        "date": date,
        "title": f"AI 日报 {date}",
        "sections": [
            {
                "heading": "要闻",
                "items": [{"title": "#1 A", "summary": "s", "url": "https://a.com/1", "source": "src"}],
            },
            {
                "heading": "模型发布",
                "items": [{"title": "#2 B", "summary": "s", "url": "https://a.com/2", "source": "src"}],
            },
            {
                "heading": "产品应用",
                "items": [{"title": "#3 C", "summary": "s", "url": "https://a.com/3", "source": "src"}],
            },
        ],
    }
    sem_path = str(reports / f"semantic_duplicates_{date}.json")
    (reports / f"semantic_duplicates_{date}.json").write_text(
        json.dumps(sem_dup_report.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    run_report = {
        "run_id": "r-1",
        "date": date,
        "provider": "mock",
        "model": "mock-model",
        "semantic_duplicate_report_path": sem_path,
        "stages": {
            "collect": {"status": "ok", "meta": {}},
            "curate": {"status": "ok", "meta": {}},
            "write": {"status": "ok", "meta": {}},
            "critique": {
                "status": "ok",
                "meta": {"verdict": "pass", "reasons": [], "score": 100},
            },
            "publish": {"status": "ok", "meta": {}},
            "eval": {
                "status": "ok",
                "meta": {"issues": [], "ok": True, "section_count": 3, "item_count": 3},
            },
        },
    }
    (drafts / f"{date}.json").write_text(
        json.dumps(draft_payload, ensure_ascii=False), encoding="utf-8"
    )
    (drafts / f"{date}.md").write_text(f"# AI 日报 {date}\n\nbody\n", encoding="utf-8")
    (reports / f"{date}.json").write_text(
        json.dumps(run_report, ensure_ascii=False), encoding="utf-8"
    )
    return str(tmp_path / "artifacts")


def test_publish_gate_blocks_high_semantic_duplicate(tmp_path):
    sem_report = SemanticDuplicateReport(
        date="2026-05-09",
        run_id="r-1",
        duplicates=[
            SemanticDuplicate(
                item_a_id="#2",
                item_b_id="#10",
                item_a_title="可信联系人上线",
                item_b_title="WhatsApp 可信联系人",
                reason="同一事件",
                severity="high",
            )
        ],
        ok=False,
        checked_item_count=10,
        provider="mock",
    )
    artifacts_root = _seed_artifacts_with_sem_dup(tmp_path, "2026-05-09", sem_report)
    arts = load_run_artifacts("2026-05-09", artifacts_root)
    gate = evaluate_publish_gate(
        arts,
        {"minimum_items": 3, "max_eval_issues": 0, "require_critic_pass": True},
    )
    assert gate.ok is False
    assert any("semantic_duplicate" in r and "high" in r for r in gate.blocked_reasons)


def test_publish_gate_blocks_medium_semantic_duplicate(tmp_path):
    sem_report = SemanticDuplicateReport(
        date="2026-05-09",
        run_id="r-1",
        duplicates=[
            SemanticDuplicate(
                item_a_id="#3",
                item_b_id="#7",
                item_a_title="Llama 4 发布",
                item_b_title="Meta 开源 Llama 4",
                reason="高度重叠",
                severity="medium",
            )
        ],
        ok=False,
        checked_item_count=10,
        provider="mock",
    )
    artifacts_root = _seed_artifacts_with_sem_dup(tmp_path, "2026-05-09", sem_report)
    arts = load_run_artifacts("2026-05-09", artifacts_root)
    gate = evaluate_publish_gate(
        arts,
        {"minimum_items": 3, "max_eval_issues": 0, "require_critic_pass": True},
    )
    assert gate.ok is False
    assert any("semantic_duplicate" in r and "medium" in r for r in gate.blocked_reasons)


def test_publish_gate_allows_low_semantic_duplicate(tmp_path):
    sem_report = SemanticDuplicateReport(
        date="2026-05-09",
        run_id="r-1",
        duplicates=[
            SemanticDuplicate(
                item_a_id="#1",
                item_b_id="#5",
                item_a_title="AI 安全研究",
                item_b_title="对齐技术综述",
                reason="主题相关",
                severity="low",
            )
        ],
        ok=True,
        checked_item_count=10,
        provider="mock",
    )
    artifacts_root = _seed_artifacts_with_sem_dup(tmp_path, "2026-05-09", sem_report)
    arts = load_run_artifacts("2026-05-09", artifacts_root)
    gate = evaluate_publish_gate(
        arts,
        {"minimum_items": 3, "max_eval_issues": 0, "require_critic_pass": True},
    )
    assert gate.ok is True
    # Warning is still surfaced in blocked_reasons for visibility.
    assert any("low" in r for r in gate.blocked_reasons)


# --------------------------------------------------------------------------- #
# 6. pipeline mock run: semantic duplicate report written to disk
# --------------------------------------------------------------------------- #


def test_semantic_duplicate_report_written_in_pipeline(
    tmp_path, monkeypatch, cfg, prompts, fake_raw_items, scripted_writer_provider
):
    monkeypatch.setattr(
        "agent.pipelines.daily_report.collect", lambda specs, **k: fake_raw_items
    )
    from agent.pipelines.daily_report import run_pipeline

    report = run_pipeline(
        cfg=cfg,
        prompts=prompts,
        provider=scripted_writer_provider,
        artifacts_root=str(tmp_path / "artifacts"),
        date="2026-05-09",
    )
    sem_path = report.get("semantic_duplicate_report_path")
    assert sem_path is not None
    assert os.path.exists(sem_path)

    with open(sem_path, encoding="utf-8") as f:
        loaded = SemanticDuplicateReport.model_validate(json.load(f))

    assert loaded.date == "2026-05-09"
    assert loaded.run_id == report["run_id"]
    assert loaded.checked_item_count >= 1
    assert isinstance(report.get("semantic_duplicate_count"), int)
    assert isinstance(report.get("semantic_duplicate_ok"), bool)
