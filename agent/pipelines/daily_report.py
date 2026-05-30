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
from typing import Any, Dict, List, Optional

from agent.agents.collector import collect
from agent.agents.critic import deterministic_critique
from agent.agents.curator import curate_with_records, curate_with_llm
from agent.agents.event_clusterer import cluster_items as cluster_events
from agent.agents.event_scorer import score_events as score_events_rules
from agent.agents.research_editor import run_research_editor
from agent.agents.final_selector import select_final_items, _guess_section, _story_key
from agent.agents.history_checker import load_recent_history
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


def _run_research_editor_flow(
    *,
    raw_items,
    sources,
    provider,
    tracer,
    budget,
    candidate_top_k: int,
    final_min: int,
    final_max: int,
    history_days: int,
    enable_evidence: bool,
    evidence_timeout: float,
    evidence_max_events: int,
    evidence_urls_per_event: int,
    llm_timeout: int,
    artifacts_root: str,
    date: str = "",
):
    """Run the full Research Editor curation pipeline.

    Flow: Cluster → Score → HistoryCheck → [Evidence] → ResearchEditor → FinalSelect
    Returns (curated_items, curated_records, editorial_meta).
    """
    editorial_meta: Dict[str, Any] = {"fallback_used": False, "fallback_reason": ""}

    # 1. Event Clustering.
    events = cluster_events(raw_items)
    tracer.log("event_clustering", raw=len(raw_items), events=len(events))
    editorial_meta["raw_item_count"] = len(raw_items)
    editorial_meta["clustered_event_count"] = len(events)

    # 2. History check.
    import os as _os
    history_titles: List[str] = []
    try:
        repo = _os.getenv("PUBLISH_REPO", "")
        token = _os.getenv("GITHUB_PUBLISH_TOKEN", "") or _os.getenv("GITHUB_TOKEN", "")
        history_titles, history_meta = load_recent_history(
            artifacts_dir=artifacts_root,
            window_days=history_days,
            repo=repo, token=token,
            exclude_date=date,
        )
        editorial_meta.update(history_meta)
        tracer.log("history_loaded", **history_meta)
    except Exception as e:
        editorial_meta.update({
            "history_source": "error",
            "history_entry_count": 0,
            "history_error": str(e)[:200],
        })
        tracer.log("history_load_error", error=str(e))

    # 3. Rule-based scoring.
    events = score_events_rules(events, history_titles=history_titles,
                                max_items=candidate_top_k)
    editorial_meta["candidate_top_k"] = len(events)
    events = _balance_candidate_events(events, limit=min(max(candidate_top_k, 60), 80))
    editorial_meta["balanced_candidate_count"] = len(events)

    # 3b. LLM Re-rank — double-gate: ask LLM to rate real-world importance.
    # Catches financial/earnings/product-launch news undervalued by rules.
    try:
        from agent.agents.llm_reranker import llm_rerank_events
        # Use a non-reasoning model for structured scoring.
        if provider.name == "deepseek":
            from agent.llm.factory import build_provider
            rerank_p = build_provider(
                "deepseek", model="deepseek-chat",
                skip_model_check=True,
                request_timeout_s=llm_timeout,
            )
        else:
            rerank_p = provider
        events = llm_rerank_events(
            events=events,
            provider=rerank_p,
            tracer=tracer,
            budget=budget,
            timeout_sec=llm_timeout,
        )
        tracer.log("llm_rerank_done", event_count=len(events),
                    top_score=events[0].rule_score if events else 0)
    except ImportError:
        pass
    except Exception as e:
        tracer.log("llm_rerank_skip", error=str(e))

    events = _balance_candidate_events(events, limit=min(max(candidate_top_k, 60), 80))
    editorial_meta["post_rerank_balanced_candidate_count"] = len(events)

    # 4. Evidence fetch (optional).
    if enable_evidence:
        try:
            from agent.tools.evidence_fetcher import fetch_evidence_for_events
            top_events = events[:max(0, evidence_max_events)]
            url_lists = [
                _evidence_urls_for_event(e, max(1, evidence_urls_per_event))
                for e in top_events
            ]
            evidence_t0 = time.time()
            tracer.log(
                "evidence_fetch_start",
                event_count=len(top_events),
                url_count=sum(len(urls) for urls in url_lists),
                timeout_sec=evidence_timeout,
            )
            fetched_evidence = fetch_evidence_for_events(
                url_lists, timeout=evidence_timeout, max_workers=8,
            )
            evidence_list = fetched_evidence + [
                [] for _ in range(max(0, len(events) - len(fetched_evidence)))
            ]
            for evt, evlist in zip(events, evidence_list):
                evt.evidence_snippets = [
                    _format_evidence_snippet(e) for e in evlist[:3]
                ]
            editorial_meta["evidence_fetch_success"] = sum(
                1 for evlist in evidence_list for e in evlist if e.fetch_status == "ok"
            )
            editorial_meta["evidence_fetch_failed"] = sum(
                1 for evlist in evidence_list for e in evlist if e.fetch_status != "ok"
            )
            tracer.log(
                "evidence_fetch_done",
                latency_ms=int((time.time() - evidence_t0) * 1000),
                ok=editorial_meta["evidence_fetch_success"],
                failed=editorial_meta["evidence_fetch_failed"],
            )
        except Exception as e:
            evidence_list = []
            editorial_meta["evidence_fetch_error"] = "exception"
            tracer.log("evidence_fetch_error", error=str(e))
    else:
        evidence_list = []

    # 5. Research Editor (LLM).
    # Use a non-reasoning model for structured editorial decisions.
    # deepseek-v4-pro (reasoning) consistently selects 0 items because it
    # "overthinks" and produces minimal output. deepseek-chat handles
    # structured JSON generation reliably.
    editor_provider = provider
    if provider.name == "deepseek":
        try:
            from agent.llm.factory import build_provider
            editor_provider = build_provider(
                "deepseek", model="deepseek-chat",
                skip_model_check=True,
                request_timeout_s=llm_timeout,
            )
            tracer.log("editor_model_switch", from_model=provider.model, to_model="deepseek-chat")
        except Exception as e:
            tracer.log("editor_model_switch_failed", error=str(e))

    editor_output = run_research_editor(
        events=events,
        evidence=evidence_list if evidence_list else None,
        history_titles=history_titles,
        provider=editor_provider,
        tracer=tracer,
        budget=budget,
        timeout_sec=llm_timeout,
    )

    llm_selected = len([d for d in editor_output.selected if d.decision == "select"])
    editorial_meta["llm_selected_count"] = llm_selected

    # 6. Final selection with fallback.
    curated, curated_records, sel_meta = select_final_items(
        editor_output=editor_output,
        events=events,
        min_items=final_min,
        max_items=final_max,
    )
    editorial_meta.update(sel_meta)

    return curated, curated_records, editorial_meta


def _balance_candidate_events(events: List[Any], *, limit: int = 70) -> List[Any]:
    """Keep the editor candidate pool broad enough for a real daily.

    Rule scores tend to over-rank arXiv because papers have excellent metadata
    and fresh timestamps. A daily issue needs enough model/product/tool/market
    candidates too, so we preserve top global items while capping any one
    section before the LLM editor sees the list.
    """
    if len(events) <= limit:
        return events

    section_minimums = {
        "模型发布": 10,
        "开发生态": 10,
        "技术与洞察": 8,
        "产品应用": 7,
        "行业动态": 7,
        "前瞻与传闻": 4,
    }
    section_caps = {
        "模型发布": 14,
        "开发生态": 14,
        "技术与洞察": 10,
        "产品应用": 10,
        "行业动态": 10,
        "前瞻与传闻": 5,
    }
    buckets: Dict[str, List[Any]] = {}
    for evt in events:
        buckets.setdefault(_guess_section(evt), []).append(evt)

    selected: List[Any] = []
    seen_ids: set[str] = set()
    section_counts: Dict[str, int] = {}
    story_counts: Dict[str, int] = {}

    def add(evt: Any, *, enforce_cap: bool = True, story_limit: int = 2) -> bool:
        if len(selected) >= limit or evt.event_id in seen_ids:
            return False
        sec = _guess_section(evt)
        cap = section_caps.get(sec, 10)
        if enforce_cap and section_counts.get(sec, 0) >= cap:
            return False
        key = _story_key(evt)
        if key and story_counts.get(key, 0) >= story_limit:
            return False
        selected.append(evt)
        seen_ids.add(evt.event_id)
        section_counts[sec] = section_counts.get(sec, 0) + 1
        if key:
            story_counts[key] = story_counts.get(key, 0) + 1
        return True

    # Preserve the strongest global candidates first.
    for evt in events[:10]:
        add(evt, enforce_cap=False, story_limit=3)

    # Then force enough non-paper variety for the editor to choose from.
    for sec, target in section_minimums.items():
        for evt in buckets.get(sec, []):
            if section_counts.get(sec, 0) >= target:
                break
            add(evt, enforce_cap=False)

    # Fill the remaining space by score, with section caps.
    for evt in events:
        if len(selected) >= limit:
            break
        add(evt)

    # If the source day is unusually concentrated, fill any leftover slots.
    for evt in events:
        if len(selected) >= limit:
            break
        add(evt, enforce_cap=False, story_limit=4)

    return selected


def _format_evidence_snippet(snippet: Any) -> str:
    """Compact fetched evidence into prompt-safe text."""
    status = getattr(snippet, "fetch_status", "")
    etype = getattr(snippet, "evidence_type", "")
    title = getattr(snippet, "title", "")
    text = getattr(snippet, "text_snippet", "")
    url = getattr(snippet, "url", "")
    text = " ".join(str(text).split())[:450]
    title = " ".join(str(title).split())[:160]
    parts = [f"status={status}", f"type={etype}"]
    if title:
        parts.append(f"title={title}")
    if text:
        parts.append(f"text={text}")
    if url:
        parts.append(f"url={url}")
    return " | ".join(parts)


def _evidence_urls_for_event(evt: Any, limit: int) -> List[str]:
    urls: List[str] = []
    for url in getattr(evt, "source_urls", []) or []:
        if len(urls) >= limit:
            break
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        if _is_low_value_evidence_url(url):
            continue
        if url not in urls:
            urls.append(url)
    return urls


def _is_low_value_evidence_url(url: str) -> bool:
    """Skip pages that rarely yield useful article text in evidence fetch."""
    from urllib.parse import urlparse

    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return False
    return host in {"x.com", "twitter.com", "t.co"}


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
                    if img not in item.images:
                        item.images.append(img)
                    count += 1
            except Exception:
                pass
    if count > 0 and tracer:
        tracer.log("image_enrichment", enriched=count)
    return count


def _enrich_vision_for_draft(draft, tracer, max_items: int = 6) -> int:
    """Post-image enrichment: describe images with Qwen VL for richer context.

    Runs after image extraction. Stores a Chinese image description as
    structured metadata. It deliberately does not mutate public article text.
    """
    from agent.tools.vision_enricher import describe_image

    count = 0
    for section in draft.sections:
        for item in section.items:
            img_url = item.image_url or (item.images[0] if item.images else "")
            if not img_url:
                continue
            if any(w in img_url.lower() for w in ("logo", "icon", "avatar", "t.png", "qrcode")):
                continue
            try:
                desc = describe_image(
                    img_url,
                    title=item.title,
                    article_text=item.summary,
                )
                if desc:
                    item.image_caption = desc
                    count += 1
            except Exception:
                pass
            if count >= max_items:
                break
        if count >= max_items:
            break
    if count > 0 and tracer:
        tracer.log("vision_enrichment", enriched=count)
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
        default_provider=provider.name,
        default_model=provider.model,
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

        # Save raw items for downstream scout/content-link diffusion.
        _collected_dir = os.path.join(artifacts_root, "collected")
        os.makedirs(_collected_dir, exist_ok=True)
        _raw_path = os.path.join(_collected_dir, f"{date}.json")
        try:
            with open(_raw_path, "w", encoding="utf-8") as _f:
                json.dump([it.to_dict() for it in raw], _f, ensure_ascii=False)
        except Exception:
            pass  # non-critical: scout can fall back to curated items
    except Exception as e:
        s.mark_failed(str(e))
        tracer.log_stage("collect", "failed", error=str(e))
        return _finalize(state, tracer, budget, reports_dir, draft_path=None, curated_path=None)

    # ---------- curate ----------
    s = state.stage("curate")
    s.mark_running()
    tracer.log_stage("curate", "running")

    # Read curation config.
    curation_cfg = cfg.get("curation", {})
    curation_mode = curation_cfg.get("mode", "research_editor")
    candidate_top_k = int(curation_cfg.get("candidate_top_k", 40))
    final_min = int(curation_cfg.get("final_min_items", 16))
    final_max = int(curation_cfg.get("final_max_items", 22))
    history_days = int(curation_cfg.get("history_window_days", 7))
    fallback_enabled = bool(curation_cfg.get("fallback_to_rules", True))
    enable_evidence = bool(curation_cfg.get("enable_evidence_fetch", False))
    evidence_timeout = float(curation_cfg.get("evidence_fetch_timeout_sec", 8))
    evidence_max_events = int(curation_cfg.get("evidence_max_events", 24))
    evidence_urls_per_event = int(curation_cfg.get("evidence_urls_per_event", 3))
    llm_timeout = int(curation_cfg.get("llm_rerank_timeout_sec", 60))

    # Content-type-aware scoring config.
    content_types_cfg = cfg.get("content_types", {})
    score_floor = float(curation_cfg.get("score_floor", 0))
    section_quotas = cfg.get("section_quotas", {})
    research_min = int(
        section_quotas.get("research_frontier", {}).get("min", 0)
    )

    try:
        # ── Fallback: rules_only mode ────────────────────────────
        if curation_mode == "rules_only":
            curated, curated_records = curate_with_records(
                raw, source_specs=sources,
                max_items=int(run_cfg.get("max_items_curate", 20)),
                content_types_cfg=content_types_cfg,
                score_floor=score_floor,
                research_min=research_min,
            )

        # ── Legacy LLM scoring (disabled by default) ─────────────
        elif curation_mode == "legacy_llm_scoring":
            if not curation_cfg.get("legacy_llm_scoring_enabled", False):
                raise ValueError("legacy_llm_scoring is disabled. Set curation.legacy_llm_scoring_enabled=true")
            curated, curated_records = curate_with_llm(
                items=raw, source_specs=sources, provider=provider,
                max_items=int(run_cfg.get("max_items_curate", 20)),
                tracer=tracer, budget=budget,
                content_types_cfg=content_types_cfg,
                score_floor=score_floor,
                research_min=research_min,
            )

        # ── Research Editor mode (default) ───────────────────────
        else:  # research_editor
            curated, curated_records, editorial_meta = _run_research_editor_flow(
                raw_items=raw, sources=sources, provider=provider,
                tracer=tracer, budget=budget,
                candidate_top_k=candidate_top_k,
                final_min=final_min, final_max=final_max,
                history_days=history_days,
                enable_evidence=enable_evidence,
                evidence_timeout=evidence_timeout,
                evidence_max_events=evidence_max_events,
                evidence_urls_per_event=evidence_urls_per_event,
                llm_timeout=llm_timeout,
                artifacts_root=artifacts_root,
                date=date,
            )
            # Write editorial meta to trace.
            for key in ("fallback_used", "fallback_reason", "llm_selected_count",
                         "final_selected_count"):
                if key in editorial_meta:
                    s.meta[key] = editorial_meta[key]
            if editorial_meta.get("fallback_used"):
                tracer.log("curate_fallback", reason=editorial_meta.get("fallback_reason", ""))

        s.meta["curated_item_count"] = len(curated)
        s.mark_ok()
        tracer.log_stage("curate", "ok", count=len(curated),
                         mode=curation_mode)

    except Exception as e:
        s.mark_failed(str(e))
        tracer.log_stage("curate", "failed", error=str(e))
        # If fallback enabled, try rules_only as last resort.
        if fallback_enabled and curation_mode == "research_editor":
            try:
                curated, curated_records = curate_with_records(
                    raw, source_specs=sources,
                    max_items=int(run_cfg.get("max_items_curate", 20)),
                )
                s.meta["curated_item_count"] = len(curated)
                s.mark_ok()
                s.meta["fallback_used"] = True
                s.meta["fallback_reason"] = str(e)
                tracer.log_stage("curate", "ok", count=len(curated),
                                 fallback_used=True, fallback_reason=str(e))
            except Exception as e2:
                return _finalize(state, tracer, budget, reports_dir,
                                 draft_path=None, curated_path=None)
        else:
            return _finalize(state, tracer, budget, reports_dir,
                             draft_path=None, curated_path=None)

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
    # Use a non-reasoning model for structured JSON generation.
    write_provider = provider
    if provider.name == "deepseek":
        try:
            from agent.llm.factory import build_provider
            write_provider = build_provider(
                "deepseek", model="deepseek-chat",
                skip_model_check=True,
                request_timeout_s=60,
            )
            tracer.log("writer_model_switch", from_model=provider.model, to_model="deepseek-chat")
        except Exception as e:
            tracer.log("writer_model_switch_failed", error=str(e))
    try:
        draft = write_draft(
            provider=write_provider,
            items=curated,
            date=date,
            system_prompt=prompts["writer_system"],
            user_template=prompts["writer_user_template"],
            max_items=int(run_cfg.get("max_items_curate", 12)),
            tracer=tracer,
            budget=budget,
            temperature=float(llm_cfg.get("temperature", 0.3)),
            max_output_tokens=int(llm_cfg.get("max_output_tokens", 2048)),
            allow_fallback=True,
            complete_with_items=bool(
                run_cfg.get(
                    "complete_writer_with_curated",
                    curation_cfg.get("mode") == "research_editor",
                )
            ),
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
            rec.section = url_to_section.get(rec.source_url) or rec.section

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

        # ── Vision enrichment: describe images with Qwen VL ──────────
        if run_cfg.get("enable_vision_enrichment", False):
            _enrich_vision_for_draft(draft, tracer)

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
                remaining_severe = [
                    d for d in post_sem.duplicates
                    if d.severity in ("high", "medium")
                ]
                if remaining_severe:
                    state.stage("write").mark_needs_review(
                        "semantic duplicates remain after repair"
                    )
                    tracer.log(
                        "post_repair_semantic_dup_needs_review",
                        duplicate_count=len(remaining_severe),
                    )
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
