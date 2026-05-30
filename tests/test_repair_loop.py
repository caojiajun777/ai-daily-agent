"""Tests for the Semantic Duplicate Repair Loop."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pytest

from agent.agents.repairer import RepairerFailed, apply_repair_actions, repair_draft
from agent.harness.budget import BudgetTracker
from agent.harness.trace import Tracer
from agent.llm import LLMMessage
from agent.llm.mock_provider import MockLLMProvider
from agent.schemas import (
    CuratedItemRecord,
    Draft,
    DraftItem,
    DraftSection,
    RepairAction,
    RepairReport,
    SemanticDuplicate,
    SemanticDuplicateReport,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

DATE = "2026-05-09"
RUN_ID = "test-run-repair"


def _budget() -> BudgetTracker:
    return BudgetTracker(
        max_total_input_tokens=100_000,
        max_total_output_tokens=20_000,
        max_total_calls=40,
        hard_fail_on_exceed=True,
    )


def _tracer(tmp_path) -> Tracer:
    return Tracer(str(tmp_path / "trace.jsonl"), run_id=RUN_ID)


def _section(heading: str, items: List[Dict]) -> DraftSection:
    return DraftSection(
        heading=heading,
        items=[
            DraftItem(
                title=it["title"],
                summary=it.get("summary", "summary"),
                url=it["url"],
                source=it.get("source", "src"),
            )
            for it in items
        ],
    )


def _draft(*sections: DraftSection) -> Draft:
    return Draft(date=DATE, title=f"AI 日报 {DATE}", sections=list(sections))


def _sem_report(dups: List[SemanticDuplicate]) -> SemanticDuplicateReport:
    blocking = any(d.severity in ("high", "medium") for d in dups)
    return SemanticDuplicateReport(
        date=DATE,
        run_id=RUN_ID,
        duplicates=dups,
        ok=not blocking,
        checked_item_count=5,
        provider="mock",
    )


def _curated_records(urls: List[str]) -> List[CuratedItemRecord]:
    return [
        CuratedItemRecord(
            raw_item_id=f"src::{u}",
            title=f"Title for {u}",
            source_url=u,
            source_name="src",
            score=1.0,
            selected_reason="recency",
        )
        for u in urls
    ]


def _no_repair_provider() -> MockLLMProvider:
    def r(messages: List[LLMMessage]) -> str:
        return json.dumps({"actions": []})
    return MockLLMProvider(model="mock-repair", responder=r)


def _repair_provider(actions: List[Dict]) -> MockLLMProvider:
    def r(messages: List[LLMMessage]) -> str:
        return json.dumps({"actions": actions})
    return MockLLMProvider(model="mock-repair", responder=r)


def _invalid_provider() -> MockLLMProvider:
    def r(messages: List[LLMMessage]) -> str:
        return "not json at all"
    return MockLLMProvider(model="mock-repair", responder=r)


# --------------------------------------------------------------------------- #
# 1. no duplicates → repair skipped
# --------------------------------------------------------------------------- #


def test_no_duplicate_repair_skipped(tmp_path):
    draft = _draft(_section("要闻", [{"title": "#1 A", "url": "https://a.com/1"}]))
    sem = _sem_report([])  # no duplicates
    records = _curated_records(["https://a.com/1", "https://a.com/2"])

    result_draft, report = repair_draft(
        draft=draft,
        sem_report=sem,
        curated_records=records,
        provider=_no_repair_provider(),
        date=DATE,
        run_id=RUN_ID,
        tracer=_tracer(tmp_path),
        budget=_budget(),
    )
    assert report.attempted is False
    assert report.succeeded is False
    assert "skipped" in report.reason
    assert result_draft is draft  # same object, no copy


# --------------------------------------------------------------------------- #
# 2. low-only duplicate → repair skipped
# --------------------------------------------------------------------------- #


def test_low_only_repair_skipped(tmp_path):
    draft = _draft(
        _section("要闻", [{"title": "#1 A", "url": "https://a.com/1"}]),
        _section("行业动态", [{"title": "#2 B", "url": "https://a.com/2"}]),
    )
    sem = _sem_report([
        SemanticDuplicate(
            item_a_id="#1", item_b_id="#2",
            item_a_title="#1 A", item_b_title="#2 B",
            reason="thematically related", severity="low",
        )
    ])
    records = _curated_records(["https://a.com/1", "https://a.com/2"])
    _, report = repair_draft(
        draft=draft, sem_report=sem, curated_records=records,
        provider=_no_repair_provider(),
        date=DATE, run_id=RUN_ID, tracer=_tracer(tmp_path), budget=_budget(),
    )
    assert report.attempted is False
    assert "skipped" in report.reason


# --------------------------------------------------------------------------- #
# 3. high duplicate → repair attempted
# --------------------------------------------------------------------------- #


def test_high_duplicate_repair_attempted(tmp_path):
    draft = _draft(
        _section("要闻", [{"title": "#2 可信联系人上线", "url": "https://a.com/2"}]),
        _section("产品应用", [{"title": "#10 WhatsApp 可信联系人", "url": "https://a.com/10"}]),
    )
    sem = _sem_report([
        SemanticDuplicate(
            item_a_id="#2", item_b_id="#10",
            item_a_title="#2 可信联系人上线", item_b_title="#10 WhatsApp 可信联系人",
            reason="同一事件", severity="high",
        )
    ])
    replacement_url = "https://a.com/99"
    records = _curated_records(["https://a.com/2", "https://a.com/10", replacement_url])
    actions = [{
        "section": "产品应用",
        "removed_title": "#10 WhatsApp 可信联系人",
        "removed_url": "https://a.com/10",
        "replacement_url": replacement_url,
        "replacement_title": "替补条目",
        "reason": "重复，删除低优先级 section 中的条目",
    }]

    result_draft, report = repair_draft(
        draft=draft, sem_report=sem, curated_records=records,
        provider=_repair_provider(actions),
        date=DATE, run_id=RUN_ID, tracer=_tracer(tmp_path), budget=_budget(),
    )
    assert report.attempted is True
    assert report.succeeded is True
    assert len(report.actions) == 1
    # Replaced URL should be in the repaired draft.
    all_urls = {item.url for sec in result_draft.sections for item in sec.items}
    assert replacement_url in all_urls
    assert "https://a.com/10" not in all_urls


def test_repair_protects_official_model_release_and_merges_access_link(tmp_path):
    official_url = "https://www.anthropic.com/news/claude-opus-4-8"
    platform_url = (
        "https://github.blog/changelog/2026-05-28-claude-opus-4-8-is-generally-available-for-github-copilot"
    )
    draft = Draft(
        date=DATE,
        title=f"AI 日报 {DATE}",
        sections=[
            DraftSection(heading="模型发布", items=[
                DraftItem(
                    title="#1 Claude Opus 4.8 发布：编码与 Agent 能力全面升级",
                    summary="Anthropic 发布 Claude Opus 4.8 模型，编码、智能体和推理能力提升。",
                    url=official_url,
                    source="Anthropic",
                    source_tier="tier_0_core_evidence",
                    evidence_type="official_release",
                    item_type="model",
                )
            ]),
            DraftSection(heading="开发生态", items=[
                DraftItem(
                    title="#2 Claude Opus 4.8 正式登陆 GitHub Copilot",
                    summary="GitHub Copilot 集成 Claude Opus 4.8。",
                    url=platform_url,
                    source="github_copilot_changelog",
                    source_tier="tier_0_core_evidence",
                    evidence_type="official_release",
                )
            ]),
        ],
    )
    sem = _sem_report([
        SemanticDuplicate(
            item_a_id="#1", item_b_id="#2",
            item_a_title="#1 Claude Opus 4.8 发布：编码与 Agent 能力全面升级",
            item_b_title="#2 Claude Opus 4.8 正式登陆 GitHub Copilot",
            reason="同一模型发布与平台接入高度重叠",
            severity="high",
        )
    ])
    records = _curated_records([official_url, platform_url])
    call_count = [0]

    def responder(messages: List[LLMMessage]) -> str:
        call_count[0] += 1
        return json.dumps({"actions": [{
            "section": "模型发布",
            "removed_title": "#1 Claude Opus 4.8 发布：编码与 Agent 能力全面升级",
            "removed_url": official_url,
            "replacement_url": None,
            "replacement_title": None,
            "reason": "bad suggestion",
        }]})

    repaired, report = repair_draft(
        draft=draft,
        sem_report=sem,
        curated_records=records,
        provider=MockLLMProvider(model="mock-repair", responder=responder),
        date=DATE,
        run_id=RUN_ID,
        tracer=_tracer(tmp_path),
        budget=_budget(),
    )

    assert call_count[0] == 0
    assert report.succeeded is True
    assert report.actions[0].removed_url == platform_url
    all_items = [item for sec in repaired.sections for item in sec.items]
    assert [item.url for item in all_items] == [official_url]
    assert platform_url in all_items[0].related_links


# --------------------------------------------------------------------------- #
# 4. repaired draft passes critic
# --------------------------------------------------------------------------- #


def test_repaired_draft_passes_critic(tmp_path):
    from agent.agents.critic import deterministic_critique
    from agent.schemas import CuratedItem

    curated_items = [
        CuratedItem(title="A", url="https://a.com/1", summary="s", source="src",
                    source_type="rss", published_at="", score=1.0),
        CuratedItem(title="B", url="https://a.com/2", summary="s", source="src",
                    source_type="rss", published_at="", score=1.0),
        CuratedItem(title="C", url="https://a.com/99", summary="s", source="src",
                    source_type="rss", published_at="", score=1.0),
    ]
    draft = _draft(
        _section("要闻", [{"title": "#1 A", "url": "https://a.com/1"}]),
        _section("产品应用", [{"title": "#2 Dup of A", "url": "https://a.com/2"}]),
        _section("行业动态", [{"title": "#3 C", "url": "https://a.com/99"}]),
    )
    sem = _sem_report([
        SemanticDuplicate(
            item_a_id="#1", item_b_id="#2",
            item_a_title="#1 A", item_b_title="#2 Dup of A",
            reason="same event", severity="high",
        )
    ])
    records = _curated_records(["https://a.com/1", "https://a.com/2", "https://a.com/99"])
    actions = [{
        "section": "产品应用",
        "removed_title": "#2 Dup of A",
        "removed_url": "https://a.com/2",
        "replacement_url": "https://a.com/99",
        "replacement_title": "#2 C replacement",
        "reason": "duplicate",
    }]
    repaired, _ = repair_draft(
        draft=draft, sem_report=sem, curated_records=records,
        provider=_repair_provider(actions),
        date=DATE, run_id=RUN_ID, tracer=_tracer(tmp_path), budget=_budget(),
    )
    critique = deterministic_critique(repaired, curated_items, min_section_count=2)
    assert critique.verdict == "pass"


# --------------------------------------------------------------------------- #
# 5. repair cannot invent URL outside curated artifact
# --------------------------------------------------------------------------- #


def test_repair_cannot_invent_url(tmp_path):
    draft = _draft(
        _section("要闻", [{"title": "#1 A", "url": "https://real.com/1"}]),
        _section("产品应用", [{"title": "#2 B", "url": "https://real.com/2"}]),
    )
    sem = _sem_report([
        SemanticDuplicate(
            item_a_id="#1", item_b_id="#2",
            item_a_title="#1 A", item_b_title="#2 B",
            reason="same event", severity="high",
        )
    ])
    # Records only contain real.com URLs — NOT the invented URL.
    records = _curated_records(["https://real.com/1", "https://real.com/2"])
    invented_url = "https://invented.example.com/fabricated"
    actions = [{
        "section": "产品应用",
        "removed_title": "#2 B",
        "removed_url": "https://real.com/2",
        "replacement_url": invented_url,   # not in curated!
        "replacement_title": "fabricated",
        "reason": "attempted fabrication",
    }]
    result_draft, report = repair_draft(
        draft=draft, sem_report=sem, curated_records=records,
        provider=_repair_provider(actions),
        date=DATE, run_id=RUN_ID, tracer=_tracer(tmp_path), budget=_budget(),
    )
    # Action should have been rejected → succeeded=False.
    assert report.succeeded is False
    assert invented_url not in {
        item.url for sec in result_draft.sections for item in sec.items
    }


# --------------------------------------------------------------------------- #
# 6. repair failure sets needs_human_review (pipeline level)
# --------------------------------------------------------------------------- #


def test_repair_failure_sets_needs_human_review(
    tmp_path, monkeypatch, cfg, prompts, fake_raw_items
):
    """When repairer raises, pipeline marks write stage needs_human_review."""
    monkeypatch.setattr(
        "agent.pipelines.daily_report.collect", lambda specs, **k: fake_raw_items
    )

    # Provider: first call = writer (returns valid draft),
    #           second call = semantic dup (returns high dup),
    #           third call = repairer (raises / returns invalid JSON).
    call_count = [0]

    def multi_responder(messages: List[LLMMessage]) -> str:
        call_count[0] += 1
        system = messages[0].content if messages else ""
        if "语义重复" in system or "duplicates" in system:
            # Semantic dup critic: return a high dup.
            return json.dumps({
                "duplicates": [{
                    "item_a_id": "#1",
                    "item_b_id": "#2",
                    "item_a_title": "A",
                    "item_b_title": "B",
                    "reason": "same",
                    "severity": "high",
                }]
            })
        if "修复" in system or "actions" in system or "repair" in system.lower():
            return "totally invalid response !!!"
        # Writer call: return valid 3-section draft using fake_raw_items URLs.
        urls = [it.url for it in fake_raw_items[:3]]
        titles = [it.title for it in fake_raw_items[:3]]
        payload = {
            "date": DATE,
            "title": f"AI 日报 {DATE}",
            "sections": [
                {"heading": s, "items": [{"title": titles[i], "summary": "s",
                                           "url": urls[i], "source": "src"}]}
                for i, s in enumerate(["要闻", "研究", "安全"])
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    provider = MockLLMProvider(model="mock", responder=multi_responder)
    from agent.pipelines.daily_report import run_pipeline

    report = run_pipeline(
        cfg=cfg,
        prompts=prompts,
        provider=provider,
        artifacts_root=str(tmp_path / "artifacts"),
        date=DATE,
    )
    # Repair was attempted but failed → repair_attempted=True, repair_succeeded=False.
    assert report.get("repair_attempted") is True
    assert report.get("repair_succeeded") is False
    # needs_human_review should be set on the run overall.
    assert report.get("needs_human_review") is True


# --------------------------------------------------------------------------- #
# 7. repair only attempts once
# --------------------------------------------------------------------------- #


def test_repair_only_attempts_once(tmp_path):
    """apply_repair_actions is deterministic; repair loop runs exactly once."""
    draft = _draft(
        _section("要闻", [{"title": "#1 X", "url": "https://x.com/1"}]),
        _section("产品应用", [{"title": "#2 X dup", "url": "https://x.com/2"}]),
    )
    sem = _sem_report([
        SemanticDuplicate(
            item_a_id="#1", item_b_id="#2",
            item_a_title="#1 X", item_b_title="#2 X dup",
            reason="same", severity="high",
        )
    ])
    replacement = "https://x.com/99"
    records = _curated_records(["https://x.com/1", "https://x.com/2", replacement])
    call_count = [0]

    def counting_responder(messages: List[LLMMessage]) -> str:
        call_count[0] += 1
        return json.dumps({"actions": [{
            "section": "产品应用",
            "removed_title": "#2 X dup",
            "removed_url": "https://x.com/2",
            "replacement_url": replacement,
            "replacement_title": "Replacement",
            "reason": "dup",
        }]})

    provider = MockLLMProvider(model="mock-repair", responder=counting_responder)
    _, report = repair_draft(
        draft=draft, sem_report=sem, curated_records=records,
        provider=provider,
        date=DATE, run_id=RUN_ID, tracer=_tracer(tmp_path), budget=_budget(),
    )
    # repair_draft makes exactly one LLM call.
    assert call_count[0] == 1
    assert report.succeeded is True


# --------------------------------------------------------------------------- #
# 8. report contains all repair fields (pipeline integration)
# --------------------------------------------------------------------------- #


def test_report_contains_repair_fields(
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
        date=DATE,
    )
    # Fields must always be present regardless of whether repair ran.
    assert "repair_attempted" in report
    assert "repair_succeeded" in report
    assert "repair_reason" in report
    assert "draft_version" in report
    assert "pre_repair_semantic_duplicate_count" in report
    assert "post_repair_semantic_duplicate_count" in report
    assert report["draft_version"] in ("v1", "v2")

    # v1 file must always exist after a successful write.
    v1_path = (tmp_path / "artifacts" / "drafts" / f"{DATE}_v1.md")
    assert v1_path.exists()

    # repair_report_path is None when no repair was needed.
    if report.get("repair_attempted"):
        assert report.get("repair_report_path") is not None
        assert os.path.exists(report["repair_report_path"])


# --------------------------------------------------------------------------- #
# 9. End-to-end: pipeline with high semantic duplicate gets repaired
# --------------------------------------------------------------------------- #


def test_e2e_high_dup_repair_pipeline(
    tmp_path, monkeypatch, fake_raw_items
):
    """Full pipeline run: writer produces a high dup, sem critic detects it,
    repairer fixes it. Asserts all artifact files and report fields."""
    monkeypatch.setattr(
        "agent.pipelines.daily_report.collect", lambda specs, **k: fake_raw_items
    )

    # URLs from fake_raw_items (first 3 are unique, index 3 is a dup).
    url_a = fake_raw_items[0].url   # https://hf.co/blog/dataset-x  (要闻)
    url_b = fake_raw_items[1].url   # https://openai.com/news/model-y (产品应用 — dup of url_a event)
    url_c = fake_raw_items[2].url   # https://anthropic.com/news/safety-z (unused → candidate)

    call_seq: list = []

    def three_stage_responder(messages):
        system = messages[0].content if messages else ""
        # Check "修复" BEFORE "语义重复": the repairer prompt contains both.
        if "修复" in system:
            call_seq.append("repair")
            return json.dumps({"actions": [{
                "section": "研究",
                "removed_title": "#2 OpenAI releases new model",
                "removed_url": url_b,
                "replacement_url": url_c,
                "replacement_title": "#2 Anthropic publishes safety paper",
                "reason": "duplicate of item in 要闻; replace with unused candidate",
            }]})
        if "语义重复" in system:
            call_seq.append("sem_dup")
            # First sem dup call: report high dup. Second call (post-repair): clean.
            if call_seq.count("sem_dup") == 1:
                return json.dumps({
                    "duplicates": [{
                        "item_a_id": "#1",
                        "item_b_id": "#2",
                        "item_a_title": "#1 HuggingFace launches new dataset",
                        "item_b_title": "#2 OpenAI releases new model",
                        "reason": "both describe the same product launch event",
                        "severity": "high",
                    }]
                })
            else:
                return json.dumps({"duplicates": []})
        # Writer call.
        call_seq.append("writer")
        payload = {
            "date": DATE,
            "title": f"AI 日报 {DATE}",
            "sections": [
                {"heading": "要闻", "items": [
                    {"title": "#1 HuggingFace launches new dataset",
                     "summary": "HF dataset released.", "url": url_a, "source": "hf_blog"}
                ]},
                {"heading": "研究", "items": [
                    {"title": "#2 OpenAI releases new model",
                     "summary": "Model Y released.", "url": url_b, "source": "oai_news"}
                ]},
                {"heading": "安全", "items": [
                    {"title": "#3 Anthropic publishes safety paper",
                     "summary": "Safety paper Z.", "url": url_c, "source": "ant_news"}
                ]},
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    from agent.llm.mock_provider import MockLLMProvider
    from agent.pipelines.daily_report import run_pipeline

    cfg = {
        "run": {"timezone": "Asia/Shanghai", "max_items_curate": 5},
        "llm": {
            "temperature": 0.0,
            "max_output_tokens": 1024,
            "sem_dup_max_tokens": 512,
            "repair_max_tokens": 512,
        },
        "budget": {
            "max_total_input_tokens": 50_000,
            "max_total_output_tokens": 10_000,
            "max_total_calls": 20,
            "hard_fail_on_exceed": True,
        },
        "context": {"max_messages_keep": 10, "per_message_max_chars": 4000},
        "eval": {"min_section_count": 2, "min_unique_titles_ratio": 0.8, "forbid_phrases": []},
        "sources": [],
    }
    prompts = {
        "writer_system": "system",
        "writer_user_template": "date={date} max={max_items} items={items_json}",
        "critic_system": "critic",
        "critic_user_template": "items={items_json} draft={draft_json}",
    }

    provider = MockLLMProvider(model="mock", responder=three_stage_responder)
    report = run_pipeline(
        cfg=cfg,
        prompts=prompts,
        provider=provider,
        artifacts_root=str(tmp_path / "artifacts"),
        date=DATE,
    )

    # Core repair assertions.
    assert report["repair_attempted"] is True
    assert report["repair_succeeded"] is True
    assert report["draft_version"] == "v2"
    assert report["pre_repair_semantic_duplicate_count"] == 1
    assert report["post_repair_semantic_duplicate_count"] == 0

    # Artifact files.
    arts = tmp_path / "artifacts"
    assert (arts / "drafts" / f"{DATE}_v1.md").exists()
    assert (arts / "drafts" / f"{DATE}_v2.md").exists()
    assert (arts / "drafts" / f"{DATE}.md").exists()
    assert (arts / "reports" / f"repair_{DATE}.json").exists()

    # Final draft must not contain removed URL, must contain replacement.
    final_md = (arts / "drafts" / f"{DATE}.md").read_text(encoding="utf-8")
    assert url_b not in final_md
    assert url_c in final_md

    # Publish gate: should pass since sem dup is now clean and repair succeeded.
    from agent.agents.issue_publisher import evaluate_publish_gate, load_run_artifacts
    # We need a minimal report on disk for the gate; run_pipeline already wrote it.
    arts_root = str(arts)
    loaded = load_run_artifacts(DATE, arts_root)
    gate = evaluate_publish_gate(
        loaded,
        {"minimum_items": 2, "max_eval_issues": 0, "require_critic_pass": True},
    )
    blocking = [r for r in gate.blocked_reasons
                if not r.startswith("semantic_duplicate_warning")]
    assert not blocking, f"unexpected blocking reasons: {blocking}"


# --------------------------------------------------------------------------- #
# 10. post-repair duplicate residue sets needs_human_review
# --------------------------------------------------------------------------- #


def test_post_repair_duplicate_residue_sets_needs_human_review(
    tmp_path, monkeypatch, fake_raw_items
):
    monkeypatch.setattr(
        "agent.pipelines.daily_report.collect", lambda specs, **k: fake_raw_items
    )
    url_a = fake_raw_items[0].url
    url_b = fake_raw_items[1].url
    url_c = fake_raw_items[2].url

    def responder(messages):
        system = messages[0].content if messages else ""
        if "修复" in system:
            return json.dumps({"actions": [{
                "section": "研究",
                "removed_title": "#2 B",
                "removed_url": url_b,
                "replacement_url": url_c,
                "replacement_title": "#2 C",
                "reason": "replace duplicate",
            }]})
        if "语义重复" in system:
            return json.dumps({
                "duplicates": [{
                    "item_a_id": "#1",
                    "item_b_id": "#2",
                    "item_a_title": "#1 A",
                    "item_b_title": "#2 C",
                    "reason": "still overlaps",
                    "severity": "medium",
                }]
            })
        return json.dumps({
            "date": DATE,
            "title": f"AI 日报 {DATE}",
            "sections": [
                {"heading": "要闻", "items": [
                    {"title": "#1 A", "summary": "s", "url": url_a, "source": "src"}
                ]},
                {"heading": "研究", "items": [
                    {"title": "#2 B", "summary": "s", "url": url_b, "source": "src"}
                ]},
            ],
        }, ensure_ascii=False)

    from agent.llm.mock_provider import MockLLMProvider
    from agent.pipelines.daily_report import run_pipeline

    cfg = {
        "run": {"timezone": "Asia/Shanghai", "max_items_curate": 5},
        "llm": {
            "temperature": 0.0,
            "max_output_tokens": 1024,
            "sem_dup_max_tokens": 512,
            "repair_max_tokens": 512,
        },
        "budget": {
            "max_total_input_tokens": 50_000,
            "max_total_output_tokens": 10_000,
            "max_total_calls": 20,
            "hard_fail_on_exceed": True,
        },
        "context": {"max_messages_keep": 10, "per_message_max_chars": 4000},
        "eval": {"min_section_count": 1, "min_unique_titles_ratio": 0.8, "forbid_phrases": []},
        "sources": [],
    }
    prompts = {
        "writer_system": "system",
        "writer_user_template": "date={date} max={max_items} items={items_json}",
        "critic_system": "critic",
        "critic_user_template": "items={items_json} draft={draft_json}",
    }

    report = run_pipeline(
        cfg=cfg,
        prompts=prompts,
        provider=MockLLMProvider(model="mock", responder=responder),
        artifacts_root=str(tmp_path / "artifacts"),
        date=DATE,
    )

    assert report["repair_attempted"] is True
    assert report["repair_succeeded"] is True
    assert report["post_repair_semantic_duplicate_count"] == 1
    assert report["needs_human_review"] is True


# --------------------------------------------------------------------------- #
# 11. max_tokens: sem dup critic respects configured value
# --------------------------------------------------------------------------- #


def test_semantic_duplicate_critic_uses_configured_max_tokens(tmp_path):
    """run_semantic_duplicate_critic receives max_output_tokens from caller."""
    from agent.agents.semantic_duplicate_critic import run_semantic_duplicate_critic

    received_tokens: list = []

    class CapturingProvider(MockLLMProvider):
        def complete(self, messages, *, temperature=0.3, max_output_tokens=1024,
                     response_format=None):
            received_tokens.append(max_output_tokens)
            return super().complete(
                messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
            )

    provider = CapturingProvider(
        model="mock",
        responder=lambda msgs: json.dumps({"duplicates": []}),
    )
    draft = _draft(_section("要闻", [{"title": "#1 A", "url": "https://x.com/1"}]))
    run_semantic_duplicate_critic(
        draft=draft,
        provider=provider,
        date=DATE,
        run_id=RUN_ID,
        tracer=_tracer(tmp_path),
        budget=_budget(),
        max_output_tokens=2048,
    )
    assert received_tokens == [2048]


def test_repairer_uses_configured_max_tokens(tmp_path):
    """repair_draft receives max_output_tokens from caller."""
    received_tokens: list = []

    class CapturingProvider(MockLLMProvider):
        def complete(self, messages, *, temperature=0.3, max_output_tokens=1024,
                     response_format=None):
            received_tokens.append(max_output_tokens)
            return super().complete(
                messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
            )

    replacement = "https://x.com/99"
    provider = CapturingProvider(
        model="mock",
        responder=lambda msgs: json.dumps({"actions": [{
            "section": "产品应用",
            "removed_title": "#2 dup",
            "removed_url": "https://x.com/2",
            "replacement_url": replacement,
            "replacement_title": "New",
            "reason": "dup",
        }]}),
    )
    draft = _draft(
        _section("要闻", [{"title": "#1 A", "url": "https://x.com/1"}]),
        _section("产品应用", [{"title": "#2 dup", "url": "https://x.com/2"}]),
    )
    sem = _sem_report([SemanticDuplicate(
        item_a_id="#1", item_b_id="#2",
        item_a_title="#1 A", item_b_title="#2 dup",
        reason="same", severity="high",
    )])
    records = _curated_records(["https://x.com/1", "https://x.com/2", replacement])
    repair_draft(
        draft=draft, sem_report=sem, curated_records=records,
        provider=provider,
        date=DATE, run_id=RUN_ID, tracer=_tracer(tmp_path), budget=_budget(),
        max_output_tokens=2048,
    )
    assert received_tokens == [2048]
