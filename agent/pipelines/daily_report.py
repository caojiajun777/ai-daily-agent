"""Daily report pipeline.

Orchestrates one full run end-to-end. Each stage:

  1. Marks the corresponding ``StageState`` as RUNNING and logs a trace event
  2. Calls the role agent
  3. On success: marks OK and logs the artifact path
  4. On failure: marks FAILED or NEEDS_HUMAN_REVIEW (per stage policy) and
     re-raises if it's a hard failure

The function is intentionally readable top-to-bottom so a reviewer can audit
the harness without chasing layers of indirection.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agent.agents.collector import collect
from agent.agents.critic import deterministic_critique
from agent.agents.curator import curate_with_records
from agent.agents.publisher import publish_local
from agent.agents.repairer import repair_draft, RepairerFailed
from agent.agents.semantic_duplicate_critic import (
    run_semantic_duplicate_critic,
    SemanticDuplicateCriticFailed,
)
from agent.agents.writer import write_draft, WriterFailed
from agent.harness.budget import BudgetTracker
from agent.harness.context import ContextManager
from agent.harness.state import RunState, StageStatus
from agent.harness.tools import ToolRegistry
from agent.harness.trace import Tracer
from agent.llm import LLMProvider
from agent.schemas import (
    CritiqueResult,
    CuratedOutput,
    Draft,
    RepairReport,
    SemanticDuplicateReport,
)


def _enrich_images_for_draft(draft, tracer, timeout: float = 4.0) -> int:
    """Post-writer enrichment: fetch og:image for article-page URLs only.

    Skips URLs that are unlikely to have extractable og:image:
      - X/Twitter, GitHub, YouTube, etc. (platform pages, not articles)
      - Non-HTTP URLs

    Runs after the draft is validated. Failures are per-item and silent.
    """
    from agent.tools.image_extractor import extract_image
    from urllib.parse import urlparse

    # IT之家 image extraction is too unreliable — skip.
    _SKIP = {"ithome.com", "x.com", "twitter.com", "github.com",
             "youtube.com", "youtu.be", "arxiv.org"}

    count = 0
    for section in draft.sections:
        for item in section.items:
            url = item.url
            if not url or not url.startswith(("http://", "https://")):
                continue
            try:
                domain = urlparse(url).netloc.lower().replace("www.", "")
                if domain in _SKIP:
                    continue
            except Exception:
                continue
            try:
                img = extract_image(url, timeout=timeout)
                if img:
                    item.image_url = img
                    count += 1
            except Exception:
                pass
    if count > 0 and tracer:
        tracer.log("image_enrichment", enriched=count)
    return count


def _today(tz_name: str) -> str:
    # Cheap TZ resolution: fall back to UTC if zoneinfo not available.
    try:
        from zoneinfo import ZoneInfo  # type: ignore

        return datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def run_pipeline(
    *,
    cfg: Dict[str, Any],
    prompts: Dict[str, str],
    provider: LLMProvider,
    artifacts_root: str = "artifacts",
    date: Optional[str] = None,
) -> Dict[str, Any]:
    run_cfg = cfg.get("run", {})
    llm_cfg = cfg.get("llm", {})
    budget_cfg = cfg.get("budget", {})
    ctx_cfg = cfg.get("context", {})
    eval_cfg = cfg.get("eval", {})
    sources = cfg.get("sources", [])

    date = date or _today(run_cfg.get("timezone", "Asia/Shanghai"))
    state = RunState.new(date=date, provider=provider.name, model=provider.model)

    traces_dir = os.path.join(artifacts_root, "traces")
    drafts_dir = os.path.join(artifacts_root, "drafts")
    reports_dir = os.path.join(artifacts_root, "reports")
    curated_dir = os.path.join(artifacts_root, "curated")
    os.makedirs(traces_dir, exist_ok=True)
    os.makedirs(drafts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(curated_dir, exist_ok=True)

    trace_path = os.path.join(traces_dir, f"{date}.jsonl")
    tracer = Tracer(trace_path, run_id=state.run_id)
    tracer.log(
        "run_start",
        date=date,
        provider=provider.name,
        model=provider.model,
        run_id=state.run_id,
    )

    budget = BudgetTracker(
        max_total_input_tokens=int(budget_cfg.get("max_total_input_tokens", 200_000)),
        max_total_output_tokens=int(budget_cfg.get("max_total_output_tokens", 30_000)),
        max_total_calls=int(budget_cfg.get("max_total_calls", 40)),
        hard_fail_on_exceed=bool(budget_cfg.get("hard_fail_on_exceed", True)),
    )
    context = ContextManager(
        max_messages_keep=int(ctx_cfg.get("max_messages_keep", 20)),
        per_message_max_chars=int(ctx_cfg.get("per_message_max_chars", 6000)),
    )
    tools = ToolRegistry()  # reserved; not yet exposed to LLM in MVP

    # ---------- collect ----------
    s = state.stage("collect")
    s.mark_running()
    tracer.log_stage("collect", "running")
    try:
        raw = collect(sources, tracer=tracer)
        s.meta["raw_item_count"] = len(raw)
        s.mark_ok()
        tracer.log_stage("collect", "ok", count=len(raw))
    except Exception as e:
        s.mark_failed(str(e))
        tracer.log_stage("collect", "failed", error=str(e))
        return _finalize(state, tracer, budget, reports_dir, draft_path=None, curated_path=None)

    # ---------- curate ----------
    s = state.stage("curate")
    s.mark_running()
    tracer.log_stage("curate", "running")
    try:
        curated, curated_records = curate_with_records(
            raw,
            source_specs=sources,
            max_items=int(run_cfg.get("max_items_curate", 12)),
        )
        s.meta["curated_item_count"] = len(curated)
        s.mark_ok()
        tracer.log_stage("curate", "ok", count=len(curated))
    except Exception as e:
        s.mark_failed(str(e))
        tracer.log_stage("curate", "failed", error=str(e))
        return _finalize(state, tracer, budget, reports_dir, draft_path=None, curated_path=None)

    if not curated:
        # No content is a soft failure; mark downstream stages skipped.
        for stage in ("write", "critique", "publish", "eval"):
            state.stage(stage).status = StageStatus.SKIPPED
        tracer.log_stage("curate", "no_content")
        return _finalize(state, tracer, budget, reports_dir, draft_path=None, curated_path=None)

    # ---------- write ----------
    s = state.stage("write")
    s.mark_running()
    tracer.log_stage("write", "running")
    draft: Optional[Draft] = None
    curated_path: Optional[str] = None
    try:
        draft = write_draft(
            provider=provider,
            items=curated,
            date=date,
            system_prompt=prompts["writer_system"],
            user_template=prompts["writer_user_template"],
            max_items=int(run_cfg.get("max_items_curate", 12)),
            tracer=tracer,
            budget=budget,
            temperature=float(llm_cfg.get("temperature", 0.3)),
            max_output_tokens=int(llm_cfg.get("max_output_tokens", 2048)),
        )
        s.mark_ok()
        tracer.log_stage("write", "ok")
        # Authoritative date is the run's date — never trust the LLM with it.
        if draft.date != date:
            tracer.log("draft_date_overridden", llm_value=draft.date, run_value=date)
            draft = draft.model_copy(update={"date": date})

        # Back-fill section names onto curated records by matching URLs.
        url_to_section: Dict[str, str] = {}
        for sec in draft.sections:
            for item in sec.items:
                url_to_section[item.url] = sec.heading
        for rec in curated_records:
            rec.section = url_to_section.get(rec.source_url)

        # Persist curated artifact.
        curated_output = CuratedOutput(
            date=date,
            run_id=state.run_id,
            items=curated_records,
        )
        curated_path = os.path.join(curated_dir, f"{date}.json")
        with open(curated_path, "w", encoding="utf-8") as f:
            json.dump(
                curated_output.model_dump(mode="json"),
                f,
                ensure_ascii=False,
                indent=2,
            )
        tracer.log(
            "curated_artifact_written",
            path=curated_path,
            item_count=len(curated_records),
        )

        # ── Enrich items with og:image from source URLs ───────────────
        _enrich_images_for_draft(draft, tracer, timeout=6.0)

    except WriterFailed as e:
        s.mark_needs_review(str(e))
        tracer.log_stage("write", "needs_human_review", error=str(e))
        return _finalize(state, tracer, budget, reports_dir, draft_path=None, curated_path=None)
    except Exception as e:
        s.mark_failed(str(e))
        tracer.log_stage("write", "failed", error=str(e))
        return _finalize(state, tracer, budget, reports_dir, draft_path=None, curated_path=None)

    # ---------- critique ----------
    s = state.stage("critique")
    s.mark_running()
    tracer.log_stage("critique", "running")
    try:
        critique: CritiqueResult = deterministic_critique(
            draft,
            curated,
            min_section_count=int(eval_cfg.get("min_section_count", 3)),
            forbid_phrases=eval_cfg.get("forbid_phrases", []),
        )
        s.meta["verdict"] = critique.verdict
        s.meta["reasons"] = critique.reasons
        s.meta["score"] = critique.score
        if critique.verdict == "pass":
            s.mark_ok()
            tracer.log_stage("critique", "ok", score=critique.score)
        else:
            s.mark_needs_review("; ".join(critique.reasons))
            tracer.log_stage(
                "critique", "needs_human_review", reasons=critique.reasons
            )
    except Exception as e:
        s.mark_failed(str(e))
        tracer.log_stage("critique", "failed", error=str(e))
        return _finalize(state, tracer, budget, reports_dir, draft_path=None, curated_path=curated_path)

    # ---------- publish ----------
    # Publish even when the critic flagged the draft: a human reviewer needs
    # the file on disk to review it.
    s = state.stage("publish")
    s.mark_running()
    tracer.log_stage("publish", "running")
    try:
        md_path, json_path = publish_local(draft, out_dir=drafts_dir)
        s.artifact_path = md_path
        s.meta["json_path"] = json_path
        s.mark_ok()
        tracer.log_stage("publish", "ok", path=md_path)
    except Exception as e:
        s.mark_failed(str(e))
        tracer.log_stage("publish", "failed", error=str(e))
        return _finalize(state, tracer, budget, reports_dir, draft_path=None, curated_path=curated_path)

    # ---------- eval (deterministic only in MVP) ----------
    s = state.stage("eval")
    s.mark_running()
    tracer.log_stage("eval", "running")
    try:
        from agent.eval.metrics import deterministic_metrics

        metrics = deterministic_metrics(
            draft=draft,
            curated=curated,
            min_unique_titles_ratio=float(
                eval_cfg.get("min_unique_titles_ratio", 0.8)
            ),
            min_section_count=int(eval_cfg.get("min_section_count", 3)),
            forbid_phrases=eval_cfg.get("forbid_phrases", []),
            curated_records=curated_records,
        )
        s.meta.update(metrics)
        if metrics.get("ok", False):
            s.mark_ok()
        else:
            s.mark_needs_review("eval metrics flagged issues")
        tracer.log_stage("eval", s.status.value, **metrics)
    except Exception as e:
        s.mark_failed(str(e))
        tracer.log_stage("eval", "failed", error=str(e))

    # ---------- semantic duplicate critic (v1) ----------
    sem_dup_report: Optional[SemanticDuplicateReport] = None
    sem_dup_report_path: Optional[str] = None
    repair_report: Optional[RepairReport] = None
    repair_report_path: Optional[str] = None
    draft_version: str = "v1"

    def _run_sem_dup(current_draft: Draft, suffix: str = "") -> Optional[SemanticDuplicateReport]:
        nonlocal sem_dup_report_path
        try:
            report = run_semantic_duplicate_critic(
                draft=current_draft,
                provider=provider,
                date=date,
                run_id=state.run_id,
                tracer=tracer,
                budget=budget,
                temperature=float(llm_cfg.get("temperature", 0.0)),
                max_output_tokens=int(llm_cfg.get("sem_dup_max_tokens", 1024)),
            )
            path = os.path.join(
                reports_dir, f"semantic_duplicates_{date}{suffix}.json"
            )
            with open(path, "w", encoding="utf-8") as _f:
                json.dump(report.model_dump(mode="json"), _f, ensure_ascii=False, indent=2)
            sem_dup_report_path = path
            tracer.log(
                "semantic_dup_report_written",
                path=path,
                ok=report.ok,
                duplicate_count=len(report.duplicates),
            )
            return report
        except SemanticDuplicateCriticFailed as e:
            tracer.log("semantic_dup_failed", error=str(e))
            return None
        except Exception as e:
            tracer.log("semantic_dup_error", error=str(e))
            return None

    sem_dup_report = _run_sem_dup(draft, suffix="")

    # ---------- repair loop (at most 1 attempt) ----------
    # Save v1 before any repair attempt.
    from agent.agents.writer import render_markdown as _render
    v1_md_path = os.path.join(drafts_dir, f"{date}_v1.md")
    with open(v1_md_path, "w", encoding="utf-8") as f:
        f.write(_render(draft))

    needs_repair = (
        sem_dup_report is not None
        and not sem_dup_report.ok
        and any(d.severity in ("high", "medium") for d in sem_dup_report.duplicates)
    )
    pre_repair_sem_dup_count: Optional[int] = (
        len([d for d in sem_dup_report.duplicates if d.severity in ("high", "medium")])
        if needs_repair and sem_dup_report is not None else None
    )

    if needs_repair:
        try:
            repaired_draft, repair_report = repair_draft(
                draft=draft,
                sem_report=sem_dup_report,
                curated_records=curated_records,
                provider=provider,
                date=date,
                run_id=state.run_id,
                tracer=tracer,
                budget=budget,
                temperature=float(llm_cfg.get("temperature", 0.0)),
                max_output_tokens=int(llm_cfg.get("repair_max_tokens", 1024)),
            )
        except RepairerFailed as e:
            repair_report = RepairReport(
                date=date,
                run_id=state.run_id,
                attempted=True,
                succeeded=False,
                reason=str(e),
                pre_duplicate_count=len(
                    [d for d in sem_dup_report.duplicates if d.severity in ("high", "medium")]
                ),
            )
            tracer.log("repair_failed", reason=str(e))
            repaired_draft = draft  # keep original on repair failure

        if repair_report and repair_report.succeeded:
            draft_version = "v2"
            draft = repaired_draft

            # Write v2 files.
            v2_md_path = os.path.join(drafts_dir, f"{date}_v2.md")
            v2_json_path = os.path.join(drafts_dir, f"{date}_v2.json")
            with open(v2_md_path, "w", encoding="utf-8") as f:
                f.write(_render(draft))
            with open(v2_json_path, "w", encoding="utf-8") as f:
                f.write(draft.model_dump_json(indent=2))

            # Overwrite canonical draft files with v2 content.
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(_render(draft))
            with open(os.path.join(drafts_dir, f"{date}.json"), "w", encoding="utf-8") as f:
                f.write(draft.model_dump_json(indent=2))

            # Re-run critic + eval on repaired draft.
            _re_critique = deterministic_critique(
                draft,
                curated,
                min_section_count=int(eval_cfg.get("min_section_count", 3)),
                forbid_phrases=eval_cfg.get("forbid_phrases", []),
            )
            tracer.log(
                "post_repair_critique",
                verdict=_re_critique.verdict,
                score=_re_critique.score,
            )

            # Re-run semantic dup on v2 (final authoritative report).
            post_sem = _run_sem_dup(draft, suffix="")
            if post_sem is not None:
                repair_report = repair_report.model_copy(
                    update={"post_duplicate_count": len(post_sem.duplicates)}
                )
                sem_dup_report = post_sem
        else:
            # Repair attempted but failed → needs_human_review.
            state.stage("write").mark_needs_review(
                repair_report.reason if repair_report else "repair failed"
            )

        # Persist repair report.
        repair_report_path = os.path.join(reports_dir, f"repair_{date}.json")
        with open(repair_report_path, "w", encoding="utf-8") as f:
            json.dump(repair_report.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        tracer.log("repair_report_written", path=repair_report_path)

    return _finalize(
        state,
        tracer,
        budget,
        reports_dir,
        draft_path=md_path,
        curated_path=curated_path,
        sem_dup_report=sem_dup_report,
        sem_dup_report_path=sem_dup_report_path,
        repair_report=repair_report,
        repair_report_path=repair_report_path,
        draft_version=draft_version,
        pre_repair_sem_dup_count=pre_repair_sem_dup_count,
    )


def _finalize(
    state: RunState,
    tracer: Tracer,
    budget: BudgetTracker,
    reports_dir: str,
    *,
    draft_path: Optional[str],
    curated_path: Optional[str],
    sem_dup_report: Optional[SemanticDuplicateReport] = None,
    sem_dup_report_path: Optional[str] = None,
    repair_report: Optional[RepairReport] = None,
    repair_report_path: Optional[str] = None,
    draft_version: str = "v1",
    pre_repair_sem_dup_count: Optional[int] = None,
) -> Dict[str, Any]:
    state.finish()
    report = state.to_dict()
    report["budget"] = budget.snapshot()
    report["draft_path"] = draft_path
    report["curated_path"] = curated_path
    report["semantic_duplicate_report_path"] = sem_dup_report_path
    if sem_dup_report is not None:
        report["semantic_duplicate_count"] = len(sem_dup_report.duplicates)
        report["semantic_duplicate_ok"] = sem_dup_report.ok
    else:
        report["semantic_duplicate_count"] = None
        report["semantic_duplicate_ok"] = None
    # Repair fields.
    report["repair_attempted"] = repair_report.attempted if repair_report else False
    report["repair_succeeded"] = repair_report.succeeded if repair_report else False
    report["repair_reason"] = repair_report.reason if repair_report else None
    report["repair_report_path"] = repair_report_path
    report["draft_version"] = draft_version
    report["pre_repair_semantic_duplicate_count"] = pre_repair_sem_dup_count
    post = repair_report.post_duplicate_count if repair_report else None
    report["post_repair_semantic_duplicate_count"] = post
    report_path = os.path.join(reports_dir, f"{state.date}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    tracer.log("run_end", report_path=report_path)
    report["report_path"] = report_path
    report["trace_path"] = tracer.path
    return report
