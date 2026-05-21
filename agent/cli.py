"""CLI entrypoint:

    python -m agent.cli run --provider deepseek
    python -m agent.cli run --provider mock
    python -m agent.cli replay --run-id <date>
    python -m agent.cli eval --run-id <date>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import yaml

from agent.harness.replay import replay
from agent.llm import build_provider

# Auto-load .env so users don't need to set env vars manually each time.
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), os.pardir, ".env")
    _env_path = os.path.normpath(_env_path)
    if os.path.exists(_env_path):
        _load_dotenv(_env_path)
except ImportError:
    pass

CFG_PATH = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "configs", "prompts.yaml")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config or CFG_PATH)
    prompts = load_yaml(args.prompts or PROMPTS_PATH)
    provider_name = args.provider or cfg.get("llm", {}).get("default_provider", "mock")
    model = args.model or cfg.get("llm", {}).get("default_model")
    provider_kwargs: Dict[str, Any] = {}
    if provider_name == "deepseek" and args.skip_model_check:
        provider_kwargs["skip_model_check"] = True
    provider = build_provider(provider_name, model=model, **provider_kwargs)

    from agent.pipelines.daily_report import run_pipeline

    report = run_pipeline(
        cfg=cfg,
        prompts=prompts,
        provider=provider,
        artifacts_root=args.artifacts or "artifacts",
        date=args.date,
    )
    print(json.dumps(
        {
            "run_id": report["run_id"],
            "date": report["date"],
            "is_failed": report["is_failed"],
            "needs_human_review": report["needs_human_review"],
            "draft_path": report.get("draft_path"),
            "report_path": report.get("report_path"),
            "trace_path": report.get("trace_path"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if not report["is_failed"] else 1


def cmd_replay(args: argparse.Namespace) -> int:
    trace_path = args.trace or os.path.join(
        args.artifacts or "artifacts", "traces", f"{args.run_id}.jsonl"
    )
    summary = replay(trace_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    artifacts = args.artifacts or "artifacts"
    draft_json_path = os.path.join(artifacts, "drafts", f"{args.run_id}.json")
    if not os.path.exists(draft_json_path):
        print(f"draft not found: {draft_json_path}", file=sys.stderr)
        return 2
    with open(draft_json_path, "r", encoding="utf-8") as f:
        draft_payload = json.load(f)
    from agent.eval.metrics import deterministic_metrics
    from agent.schemas import CuratedItem, CuratedItemRecord, CuratedOutput, Draft

    draft = Draft.model_validate(draft_payload)

    # Prefer the persisted curated artifact; fall back to reconstructing from draft.
    curated_records = None
    curated_json_path = os.path.join(artifacts, "curated", f"{args.run_id}.json")
    if os.path.exists(curated_json_path):
        with open(curated_json_path, "r", encoding="utf-8") as f:
            curated_output = CuratedOutput.model_validate(json.load(f))
        curated_records = curated_output.items
        curated: list[CuratedItem] = []
    else:
        # Legacy fallback: reconstruct curated set from draft URLs.
        # hallucinated_urls will be 0 in this path (no ground-truth to compare).
        curated = [
            CuratedItem(
                title=it.title,
                url=it.url,
                summary=it.summary,
                source=it.source,
                source_type="rss",
                published_at="",
                score=0.0,
            )
            for s in draft.sections
            for it in s.items
        ]

    cfg = load_yaml(args.config or CFG_PATH)
    eval_cfg = cfg.get("eval", {})
    metrics = deterministic_metrics(
        draft=draft,
        curated=curated,
        min_unique_titles_ratio=float(eval_cfg.get("min_unique_titles_ratio", 0.8)),
        min_section_count=int(eval_cfg.get("min_section_count", 3)),
        forbid_phrases=eval_cfg.get("forbid_phrases", []),
        curated_records=curated_records,
    )

    # Augment with semantic duplicate summary (optional).
    sem_dup_path = os.path.join(artifacts, "reports", f"semantic_duplicates_{args.run_id}.json")
    if os.path.exists(sem_dup_path):
        try:
            from agent.schemas import SemanticDuplicateReport
            with open(sem_dup_path, "r", encoding="utf-8") as f:
                sem = SemanticDuplicateReport.model_validate(json.load(f))
            metrics["semantic_duplicate_count"] = len(sem.duplicates)
            metrics["semantic_duplicate_ok"] = sem.ok
        except Exception:
            pass

    # Augment with repair summary (optional).
    repair_path = os.path.join(artifacts, "reports", f"repair_{args.run_id}.json")
    if os.path.exists(repair_path):
        try:
            from agent.schemas import RepairReport
            with open(repair_path, "r", encoding="utf-8") as f:
                rep = RepairReport.model_validate(json.load(f))
            metrics["repair_attempted"] = rep.attempted
            metrics["repair_succeeded"] = rep.succeeded
            metrics["repair_reason"] = rep.reason
            metrics["draft_version"] = rep.draft_version
        except Exception:
            pass

    # Source quality distribution (if curated records available).
    if curated_records:
        from collections import Counter
        tiers = Counter(getattr(r, "source_tier", "?") or "?" for r in curated_records)
        cts = Counter(getattr(r, "content_type", "?") or "?" for r in curated_records)
        total = len(curated_records)
        tier0 = tiers.get("tier_0_core_evidence", 0)
        tier1 = tiers.get("tier_1_high_signal", 0)
        high_non_tier0 = sum(
            1 for r in curated_records
            if getattr(r, "confidence", "medium") == "high"
            and "tier_0" not in (getattr(r, "source_tier", "") or "")
        )
        metrics["source_quality"] = {
            "item_count_by_source_tier": dict(tiers),
            "item_count_by_content_type": dict(cts),
            "tier0_item_ratio": round(tier0 / total, 3) if total else 0,
            "tier0_tier1_item_ratio": round((tier0 + tier1) / total, 3) if total else 0,
            "tier3_item_count": sum(
                1 for k, v in tiers.items() if "tier_3" in k
            ),
            "high_confidence_non_tier0_count": high_non_tier0,
        }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["ok"] else 1


def cmd_publish_issue(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.confirm:
        print(
            "publish-issue requires --dry-run or --confirm (refusing to "
            "guess the user's intent)",
            file=sys.stderr,
        )
        return 2
    if args.dry_run and args.confirm:
        print("--dry-run and --confirm are mutually exclusive", file=sys.stderr)
        return 2

    cfg = load_yaml(args.config or CFG_PATH)
    publish_cfg = cfg.get("publish", {})
    artifacts = args.artifacts or "artifacts"

    # Build publisher: real one for confirm, real one for dry-run by default
    # (so we can also detect duplicates), but allow callers to opt out.
    from agent.harness.trace import Tracer
    from agent.tools.issue_publisher import (
        FakeIssuePublisher,
        GitHubIssuePublisher,
        PublisherConfigError,
    )

    if args.fake_publisher:
        publisher = FakeIssuePublisher(repo=args.repo or "test-owner/test-repo")
    else:
        try:
            publisher = GitHubIssuePublisher(
                repo=args.repo,
            )
        except PublisherConfigError as e:
            print(f"publisher config error: {e}", file=sys.stderr)
            return 3

    traces_dir = os.path.join(artifacts, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    trace_path = os.path.join(traces_dir, f"{args.run_id}.jsonl")
    tracer = Tracer(trace_path, run_id=f"publish-{args.run_id}")

    from agent.agents.issue_publisher import run_publish

    try:
        result = run_publish(
            date=args.run_id,
            publisher=publisher,
            publish_cfg=publish_cfg,
            artifacts_root=artifacts,
            tracer=tracer,
            mode="dry-run" if args.dry_run else "confirm",
            force=bool(args.force),
            force_dup=bool(args.force_dup),
        )
    except FileNotFoundError as e:
        print(f"missing artifact: {e}", file=sys.stderr)
        return 4

    summary = {
        "mode": result["mode"],
        "date": result["date"],
        "target_repo": result["target_repo"],
        "title": result["title"],
        "labels": result["labels"],
        "gate_ok": result["gate_result"]["ok"],
        "blocked_reasons": result["gate_result"]["blocked_reasons"],
        "duplicates": [d["number"] for d in result["duplicates"]],
        "duplicate_blocked": result["duplicate_blocked"],
        "force": result["force"],
    }
    if result["mode"] == "dry-run":
        summary["would_publish"] = result["would_publish"]
        summary["preview_path"] = result["preview_path"]
    else:
        summary["status"] = result["status"]
        summary["issue_number"] = result.get("issue_number")
        summary["issue_url"] = result.get("issue_url")
        summary["result_path"] = result["result_path"]

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Exit codes: 0 = success path, 5 = blocked, 6 = publisher error.
    if result["mode"] == "dry-run":
        return 0
    if result["status"] == "published":
        return 0
    if result["status"] in ("blocked_by_gate", "blocked_by_duplicate"):
        return 5
    return 6


def cmd_discover_sources(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config or CFG_PATH)
    provider_name = args.provider or cfg.get("llm", {}).get("default_provider", "mock")
    model = args.model or cfg.get("llm", {}).get("default_model")

    try:
        provider = build_provider(provider_name, model=model)
    except Exception as e:
        print(f"failed to build provider: {e}", file=sys.stderr)
        return 3

    from agent.agents.source_discoverer import discover_sources

    report = discover_sources(
        topic=args.topic,
        provider=provider,
        existing_config_path=args.config or CFG_PATH,
        max_candidates=args.max_candidates,
        min_score=args.min_score,
    )

    if args.json:
        output = {
            "topic": report.topic,
            "run_ts": report.run_ts,
            "candidates_generated": report.candidates_generated,
            "candidates_validated": report.candidates_validated,
            "candidates_passed": report.candidates_passed,
            "passed": [
                {
                    "name": c.name,
                    "type": c.source_type,
                    "url": c.url,
                    "username": c.username,
                    "score": round(c.overall_score, 3),
                    "freshness": round(c.freshness_score, 3),
                    "relevance": round(c.relevance_score, 3),
                    "reason": c.reason,
                    "note": c.validation_note,
                }
                for c in report.passed
            ],
            "rejected": [
                {"name": c.name, "type": c.source_type, "note": c.validation_note}
                for c in report.rejected
            ],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"\nSource Discovery Report — {report.topic}")
        print(f"  generated: {report.candidates_generated}")
        print(f"  validated: {report.candidates_validated}")
        print(f"  passed:    {report.candidates_passed}\n")
        if report.passed:
            print("─" * 60)
            print("PASSED (ready to merge):\n")
            for c in report.passed:
                tag = f"@{c.username}" if c.source_type == "x" else c.url
                print(f"  [{c.overall_score:.2f}] {c.name}")
                print(f"        {tag}")
                print(f"        {c.validation_note}")
                print()
        if report.rejected:
            print("─" * 60)
            print("REJECTED:\n")
            for c in report.rejected:
                print(f"  ✗ {c.name}: {c.validation_note}")
            print()
        if report.yaml_snippet:
            print("─" * 60)
            print("YAML snippet (copy into default.yaml sources):\n")
            print(report.yaml_snippet)

    return 0 if report.candidates_passed > 0 else 1


def cmd_scout(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config or CFG_PATH)
    provider_name = args.provider or cfg.get("llm", {}).get("default_provider", "mock")
    model = args.model or cfg.get("llm", {}).get("default_model")

    try:
        kwargs = {}
        if args.skip_model_check:
            kwargs["skip_model_check"] = True
        provider = build_provider(provider_name, model=model, **kwargs)
    except Exception as e:
        print(f"failed to build provider: {e}", file=sys.stderr)
        return 3

    # Load collected items from a prior run if requested.
    # Prefer raw collected items (all sources, full URLs) over curated (16-22 items).
    collected = None
    if args.run_id:
        import json as _json
        raw_path = os.path.join(args.artifacts or "artifacts", "collected", f"{args.run_id}.json")
        curated_path = os.path.join(args.artifacts or "artifacts", "curated", f"{args.run_id}.json")
        for path in [raw_path, curated_path]:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = _json.load(f)
                    if isinstance(data, list):
                        collected = data  # raw items: list of dicts
                    else:
                        collected = data.get("items", [])  # curated: dict with items key
                    if collected:
                        break
                except Exception as e:
                    print(f"warning: failed to load items from {path}: {e}", file=sys.stderr)

    from agent.agents.source_scout import scout_sources

    report = scout_sources(
        topic=args.topic,
        provider=provider,
        config_path=args.config or CFG_PATH,
        collected_items=collected,
        max_per_channel=args.max_per_channel,
        min_score=args.min_score,
    )

    if args.json:
        output = {
            "topic": report.topic,
            "run_ts": report.run_ts,
            "channels_used": report.channels_used,
            "candidates_total": report.candidates_total,
            "candidates_passed": report.candidates_passed,
            "cross_boosted": report.cross_boosted,
            "channel_details": report.channel_details,
            "passed": [
                {
                    "name": c.name,
                    "type": c.source_type,
                    "url": c.url,
                    "username": c.username,
                    "score": round(c.overall_score, 3),
                    "reason": c.reason,
                    "note": c.validation_note,
                }
                for c in report.passed
            ],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"\nUnified Source Scout — {report.topic}")
        print(f"  channels:  {', '.join(report.channels_used)}")
        print(f"  found:     {report.candidates_total} unique across all channels")
        print(f"  passed:    {report.candidates_passed}")
        print(f"  boosted:   {report.cross_boosted} (found by 2+ channels)")
        for ch, det in report.channel_details.items():
            print(f"  [{ch}] generated={det['generated']} passed={det['passed']}")
        print()
        if report.passed:
            print("─" * 60)
            print("PASSED:\n")
            for c in report.passed:
                tag = f"@{c.username}" if c.source_type == "x" else (c.url or "")
                print(f"  [{c.overall_score:.2f}] {c.name} ({c.source_type})")
                print(f"        {tag}")
                print(f"        {c.reason}")
                print(f"        {c.validation_note}")
                print()
        if report.yaml_snippet:
            print("─" * 60)
            print("YAML snippet:\n")
            print(report.yaml_snippet)

    return 0 if report.candidates_passed > 0 else 1


def cmd_diffuse(args: argparse.Namespace) -> int:
    from agent.agents.source_diffuser import diffuse_sources
    from agent.sources.base import RawItem

    artifacts = args.artifacts
    config_path = args.config or CFG_PATH

    # Optionally load collected items from a prior run.
    collected: list = []
    if args.run_id:
        curated_path = os.path.join(artifacts, "curated", f"{args.run_id}.json")
        if os.path.exists(curated_path):
            try:
                import json as _json
                with open(curated_path, "r", encoding="utf-8") as f:
                    curated_data = _json.load(f)
                for rec in curated_data.get("items", []):
                    collected.append(RawItem(
                        source_id=rec.get("source_name", ""),
                        source_type="rss",
                        title=rec.get("title", ""),
                        url=rec.get("source_url", ""),
                        summary="",
                        published_at=rec.get("published_at") or "",
                    ))
            except Exception as e:
                print(f"warning: failed to load curated items: {e}", file=sys.stderr)

    if args.mode in ("graph", "both") and not os.getenv("X_BEARTER_TOKEN"):
        print("note: X_BEARTER_TOKEN not set — social graph diffusion will be skipped", file=sys.stderr)

    result = diffuse_sources(
        config_path=config_path,
        collected_items=collected if collected else None,
    )

    if args.json:
        output = {
            "social_graph": (
                {
                    "candidates": result["social_graph"].candidates_discovered,
                    "passed": result["social_graph"].candidates_passed,
                    "sources": [
                        {"name": c.name, "username": c.username, "score": round(c.overall_score, 3),
                         "overlap": c.seed_overlap_count, "reason": c.reason}
                        for c in result["social_graph"].passed
                    ] if result["social_graph"] else [],
                }
                if result["social_graph"] else None
            ),
            "content_links": (
                {
                    "candidates": result["content_links"].candidates_discovered,
                    "passed": result["content_links"].candidates_passed,
                    "sources": [
                        {"name": c.name, "url": c.url, "score": round(c.overall_score, 3),
                         "link_count": c.link_count, "reason": c.reason}
                        for c in result["content_links"].passed
                    ] if result["content_links"] else [],
                }
                if result["content_links"] else None
            ),
            "summary": result["summary"],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"\nSource Diffusion Report")
        print(f"  {result['summary']}\n")
        if result.get("merged_yaml"):
            print("─" * 60)
            print("New Sources (copy into default.yaml sources section):\n")
            print(result["merged_yaml"])

    total = 0
    if result["social_graph"]:
        total += result["social_graph"].candidates_passed
    if result["content_links"]:
        total += result["content_links"].candidates_passed
    return 0 if total > 0 else 1


def cmd_admit_sources(args: argparse.Namespace) -> int:
    """Auto-admit discovered sources from a scout report into the YAML config."""
    from agent.tools.auto_admit import auto_admit_from_scout

    config_path = args.config or CFG_PATH
    result = auto_admit_from_scout(
        scout_report_path=args.scout_report,
        config_path=config_path,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        status = result.get("status", "error")
        admitted = result.get("admitted", 0)
        candidates = result.get("candidates", 0)
        print(f"admit-sources: {status} | admitted={admitted}/{candidates}")
        if status == "ok":
            print(f"  config updated: {config_path}")
    return 0 if result.get("status") in ("ok", "dry_run", "skipped") else 1


def cmd_send(args: argparse.Namespace) -> int:
    """Generate PDF and email the daily report."""
    from agent.tools.pdf_emailer import send_daily_report

    run_id = args.run_id
    md_path = os.path.join(args.artifacts or "artifacts", "drafts", f"{run_id}.md")

    if not os.path.exists(md_path):
        print(f"error: draft not found: {md_path}", file=sys.stderr)
        return 2

    result = send_daily_report(
        date=run_id,
        md_path=md_path,
        pdf_dir=os.path.join(args.artifacts or "artifacts", "pdf"),
    )

    if not result.get("ok"):
        error = result.get("error", "unknown")
        print(f"send failed: {error}", file=sys.stderr)
        return 1

    print(f"Sent to: {', '.join(result.get('recipients', []))}")
    if result.get("pdf_path"):
        print(f"PDF saved: {result['pdf_path']}")
    return 0


def cmd_trends(args: argparse.Namespace) -> int:
    cfg = load_yaml(CFG_PATH)
    provider_name = args.provider or cfg.get("llm", {}).get("default_provider", "deepseek")
    model = args.model or cfg.get("llm", {}).get("default_model")
    try:
        provider = build_provider(provider_name, model=model)
    except Exception as e:
        print(f"failed to build provider: {e}", file=sys.stderr)
        return 3

    from agent.agents.trend_analyzer import analyze_trends, analyze_multi_window

    if args.multi_window:
        windows = [int(w.strip()) for w in args.multi_window.split(",") if w.strip().isdigit()]
        result = analyze_multi_window(
            provider=provider, artifacts_dir=args.artifacts or "artifacts",
            windows=windows,
        )
        for label, r in result.items():
            _print_trend_result(label, r)
        return 0

    r = analyze_trends(
        provider=provider, artifacts_dir=args.artifacts or "artifacts",
        days=args.days,
    )
    _print_trend_result(f"{args.days}d", r)
    return 0 if r.get("ok") else 1


def _print_trend_result(label: str, r: dict) -> None:
    print(f"\n=== Trends [{label}] ===")
    if r.get("error"):
        print(f"  Error: {r['error']}")
        return
    fb = " (metrics-only fallback)" if r.get("fallback_used") else ""
    print(f"  findings: {r.get('findings', 0)}{fb}")
    print(f"  weak_signals: {r.get('weak_signals', 0)}")
    print(f"  noise/hype: {r.get('noise', 0)}")
    print(f"  validation warnings: {r.get('warnings', 0)}")
    for path_type, path in r.get("paths", {}).items():
        print(f"  saved: {path}")


def cmd_verify_gitblog(args: argparse.Namespace) -> int:
    from agent.agents.gitblog_verifier import verify_gitblog
    from agent.tools.gitblog_verifier import (
        GitHubReadAPIClient,
        VerifierConfigError,
    )

    try:
        api = GitHubReadAPIClient(
            repo=args.repo,
            token_env_var=args.token_env or "GITBLOG_OWNER_PAT",
        )
    except VerifierConfigError as e:
        print(f"verifier config error: {e}", file=sys.stderr)
        return 3

    report = verify_gitblog(
        api=api, issue_number=args.issue_number, date=args.date
    )

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_verify_report_human(report)

    if report.fatal_error:
        return 4
    return 0 if report.ok else 5


def _print_verify_report_human(report) -> None:
    print(f"verify-gitblog: {report.repo} issue #{report.issue_number} (date={report.date})")
    if report.fatal_error:
        print(f"  FATAL: {report.fatal_error}")
        return
    overall = "PASS" if report.ok else "FAIL"
    print(f"  overall: {overall}")
    for c in report.checks:
        marker = "OK " if c.ok else "FAIL"
        print(f"  [{marker}] {c.name}: {c.detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent", description="AI Tech Intelligence Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="execute one pipeline run")
    p_run.add_argument("--provider", default=None, help="mock | deepseek | anthropic | openai_compatible")
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--config", default=None)
    p_run.add_argument("--prompts", default=None)
    p_run.add_argument("--artifacts", default=None)
    p_run.add_argument("--date", default=None, help="logical report date YYYY-MM-DD")
    p_run.add_argument(
        "--skip-model-check",
        action="store_true",
        help="DeepSeek only; skip models.list validation (use only when API gateway blocks list)",
    )
    p_run.set_defaults(func=cmd_run)

    p_replay = sub.add_parser("replay", help="summarize a previous run from its trace")
    p_replay.add_argument("--run-id", required=True, help="report date or full run id")
    p_replay.add_argument("--trace", default=None)
    p_replay.add_argument("--artifacts", default=None)
    p_replay.set_defaults(func=cmd_replay)

    p_eval = sub.add_parser("eval", help="run deterministic eval over a stored draft")
    p_eval.add_argument("--run-id", required=True, help="report date YYYY-MM-DD")
    p_eval.add_argument("--config", default=None)
    p_eval.add_argument("--artifacts", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_pub = sub.add_parser(
        "publish-issue",
        help="publish a draft as a GitHub issue (gated by critic + eval)",
    )
    p_pub.add_argument("--run-id", required=True, help="report date YYYY-MM-DD")
    p_pub.add_argument(
        "--dry-run",
        action="store_true",
        help="don't create an issue; write artifacts/reports/publish_preview_<date>.json",
    )
    p_pub.add_argument(
        "--confirm",
        action="store_true",
        help="actually create the issue (gate + dup checks still apply)",
    )
    p_pub.add_argument(
        "--force",
        action="store_true",
        help="bypass both publish gate and duplicate check",
    )
    p_pub.add_argument(
        "--force-dup",
        action="store_true",
        help="bypass only the duplicate check (gate still applies)",
    )
    p_pub.add_argument(
        "--repo",
        default=None,
        help="override target repo; defaults to GITBLOG_REPO env var",
    )
    p_pub.add_argument(
        "--fake-publisher",
        action="store_true",
        help="(testing/local-only) use an in-memory publisher instead of GitHub",
    )
    p_pub.add_argument("--config", default=None)
    p_pub.add_argument("--artifacts", default=None)
    p_pub.set_defaults(func=cmd_publish_issue)

    p_discover = sub.add_parser(
        "discover-sources",
        help="LLM-driven source discovery: generate candidates + validate them",
    )
    p_discover.add_argument(
        "--topic",
        default="auto",
        help="auto (gap analysis) | broad | chinese-ai-models | ai-research | ai-product",
    )
    p_discover.add_argument(
        "--provider", default=None, help="LLM provider (default: from config or mock)"
    )
    p_discover.add_argument("--model", default=None)
    p_discover.add_argument("--config", default=None)
    p_discover.add_argument(
        "--max-candidates", type=int, default=15, help="max candidates to generate"
    )
    p_discover.add_argument(
        "--min-score", type=float, default=0.4, help="minimum overall score to pass"
    )
    p_discover.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    p_discover.set_defaults(func=cmd_discover_sources)

    p_scout = sub.add_parser(
        "scout",
        help="unified source discovery: LLM + content diffusion + social graph",
    )
    p_scout.add_argument(
        "--topic", default="broad",
        help="broad | chinese-ai-models | ai-research | ai-product",
    )
    p_scout.add_argument(
        "--provider", default=None,
        help="LLM provider for the semantic discovery channel (default: config or mock)",
    )
    p_scout.add_argument("--model", default="deepseek-chat",
                         help="LLM model (default: deepseek-chat, non-reasoning)")
    p_scout.add_argument("--config", default=None)
    p_scout.add_argument("--artifacts", default="artifacts")
    p_scout.add_argument(
        "--run-id", default=None,
        help="use items from a prior run for content-link diffusion (YYYY-MM-DD)",
    )
    p_scout.add_argument(
        "--max-per-channel", type=int, default=12,
        help="max candidates per discovery channel",
    )
    p_scout.add_argument(
        "--min-score", type=float, default=0.35,
        help="minimum overall score to pass",
    )
    p_scout.add_argument(
        "--json", action="store_true", help="machine-readable output"
    )
    p_scout.add_argument(
        "--skip-model-check", action="store_true",
        help="skip model list validation (for custom API gateways)"
    )
    p_scout.set_defaults(func=cmd_scout)

    p_admit = sub.add_parser(
        "admit-sources",
        help="auto-admit discovered sources from scout report into config",
    )
    p_admit.add_argument(
        "--scout-report", required=True,
        help="path to scout report JSON (e.g. artifacts/reports/scout_2026-05-19.json)",
    )
    p_admit.add_argument("--config", default=None)
    p_admit.add_argument(
        "--dry-run", action="store_true",
        help="preview only, do not modify config",
    )
    p_admit.set_defaults(func=cmd_admit_sources)

    p_diffuse = sub.add_parser(
        "diffuse",
        help="discover new sources via social graph and content-link diffusion",
    )
    p_diffuse.add_argument(
        "--mode", default="both",
        choices=["graph", "links", "both"],
        help="graph=social graph (needs X token), links=content links, both=all",
    )
    p_diffuse.add_argument(
        "--run-id", default=None,
        help="use collected items from a previous run for link diffusion (YYYY-MM-DD)",
    )
    p_diffuse.add_argument("--config", default=None)
    p_diffuse.add_argument("--artifacts", default="artifacts")
    p_diffuse.add_argument(
        "--json", action="store_true", help="machine-readable output"
    )
    p_diffuse.set_defaults(func=cmd_diffuse)

    p_send = sub.add_parser(
        "send",
        help="generate PDF and email the daily report to subscribers",
    )
    p_send.add_argument("--run-id", required=True, help="YYYY-MM-DD")
    p_send.add_argument("--no-pdf", action="store_true", help="skip PDF, send HTML only")
    p_send.add_argument("--artifacts", default="artifacts")
    p_send.set_defaults(func=cmd_send)

    p_trends = sub.add_parser(
        "trends",
        help="analyze industry trends from past daily reports",
    )
    p_trends.add_argument("--days", type=int, default=7, help="days of history to analyze")
    p_trends.add_argument("--multi-window", default=None, help="comma-separated windows, e.g. 4,7,14,30")
    p_trends.add_argument("--provider", default=None)
    p_trends.add_argument("--model", default=None)
    p_trends.add_argument("--artifacts", default="artifacts")
    p_trends.set_defaults(func=cmd_trends)

    p_verify = sub.add_parser(
        "verify-gitblog",
        help="read-only end-to-end check that a published issue propagated through the gitblog pipeline",
    )
    p_verify.add_argument(
        "--repo",
        default=None,
        help="target repo (owner/name); defaults to GITBLOG_REPO env var",
    )
    p_verify.add_argument(
        "--issue-number", required=True, type=int, help="GitHub issue number"
    )
    p_verify.add_argument(
        "--date",
        default=None,
        help="logical report date YYYY-MM-DD; defaults to issue.created_at[:10]",
    )
    p_verify.add_argument(
        "--token-env",
        default="GITBLOG_OWNER_PAT",
        help="env var name to read the GitHub token from (default: GITBLOG_OWNER_PAT)",
    )
    p_verify.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON report"
    )
    p_verify.set_defaults(func=cmd_verify_gitblog)

    p_pricing = sub.add_parser(
        "pricing-snapshot",
        help="snapshot model pricing and compute diff vs previous",
    )
    p_pricing.add_argument("--date", default=None)
    p_pricing.add_argument("--config", default=None)
    p_pricing.add_argument("--artifacts", default="artifacts")
    p_pricing.add_argument(
        "--json", action="store_true", help="machine-readable output"
    )
    p_pricing.set_defaults(func=cmd_pricing_snapshot)

    p_src_resolve = sub.add_parser(
        "source-resolve",
        help="auto-diagnose and fix disabled sources: find URLs, validate, apply-safe",
    )
    p_src_resolve.add_argument("--source-id", default=None)
    p_src_resolve.add_argument("--config", default=None)
    p_src_resolve.add_argument("--artifacts", default="artifacts")
    p_src_resolve.add_argument(
        "--dry-run", action="store_true",
        help="preview only, write resolution report without modifying config",
    )
    p_src_resolve.add_argument(
        "--apply-safe", action="store_true",
        help="apply only low-risk fixes (valid RSS/GitHub releases)",
    )
    p_src_resolve.add_argument(
        "--force", action="store_true",
        help="overwrite existing URLs even if already set",
    )
    p_src_resolve.add_argument(
        "--json", action="store_true", help="machine-readable output"
    )
    p_src_resolve.set_defaults(func=cmd_source_resolve)

    return parser


def cmd_source_resolve(args: argparse.Namespace) -> int:
    """Auto-diagnose and repair disabled sources."""
    from agent.tools.source_resolver import (
        resolve_all_disabled, resolve_source,
        apply_safe_resolutions, apply_non_enable_fixes,
        SourceResolutionReport,
    )
    import json as _json, os as _os

    cfg = load_yaml(args.config or CFG_PATH)
    sources = cfg.get("sources", [])

    if args.source_id:
        target = next((s for s in sources if isinstance(s, dict) and s["id"] == args.source_id), None)
        if not target:
            print(f"Source not found: {args.source_id}", file=sys.stderr)
            return 2
        result = resolve_source(target)
        if args.json:
            print(_json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2))
        else:
            print(f"  Source: {result.source_id}")
            print(f"  Status: {result.status}")
            print(f"  Selected URL: {result.selected_url}")
            print(f"  Recommend enabled: {result.recommended_enabled}")
            print(f"  Risk: {result.risk_level}")
            print(f"  Reason: {result.reason}")
        return 0

    report = resolve_all_disabled(sources)

    # Write report
    artifacts = args.artifacts or "artifacts"
    reports_dir = _os.path.join(artifacts, "reports")
    _os.makedirs(reports_dir, exist_ok=True)
    date = report.date
    report_path = _os.path.join(reports_dir, f"source_resolution_{date}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        _json.dump({
            "date": date,
            "total_checked": report.total_checked,
            "resolved_enable_safe": report.resolved_enable_safe,
            "candidate_found_needs_adapter": report.candidate_found_needs_adapter,
            "candidate_found_needs_review": report.candidate_found_needs_review,
            "no_candidate_found": report.no_candidate_found,
            "invalid_existing_url": report.invalid_existing_url,
            "ambiguous_candidates": report.ambiguous_candidates,
            "results": [_result_to_dict(r) for r in report.results],
        }, f, ensure_ascii=False, indent=2)

    if args.apply_safe:
        sources = apply_safe_resolutions(sources, report, force=args.force)
        sources = apply_non_enable_fixes(sources, report, force=args.force)
        cfg["sources"] = sources
        with open(args.config or CFG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"Applied safe fixes to config")

    if args.json:
        print(_json.dumps({
            "date": date,
            "report_path": report_path,
            "total_checked": report.total_checked,
            "resolved_enable_safe": report.resolved_enable_safe,
            "applied": args.apply_safe,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"Source Resolution Report: {date}")
        print(f"  Checked: {report.total_checked}")
        print(f"  Safe enable: {report.resolved_enable_safe}")
        print(f"  Needs adapter: {report.candidate_found_needs_adapter}")
        print(f"  Needs review: {report.candidate_found_needs_review}")
        print(f"  No candidate: {report.no_candidate_found}")
        print(f"  Invalid URL: {report.invalid_existing_url}")
        print(f"  Ambiguous: {report.ambiguous_candidates}")
        print(f"  Report: {report_path}")

    return 0


def _result_to_dict(r) -> dict:
    return {
        "source_id": r.source_id,
        "status": r.status,
        "old_url": r.old_url,
        "candidate_urls": r.candidate_urls,
        "selected_url": r.selected_url,
        "url_validation_status": r.url_validation_status,
        "http_status": r.http_status,
        "detected_source_kind": r.detected_source_kind,
        "recommended_enabled": r.recommended_enabled,
        "recommended_parser_strategy": r.recommended_parser_strategy,
        "required_adapter": r.required_adapter,
        "confidence": r.confidence,
        "risk_level": r.risk_level,
        "reason": r.reason,
        "notes": r.notes,
    }


def cmd_pricing_snapshot(args: argparse.Namespace) -> int:
    """Snapshot model pricing and compute diff."""
    from agent.sources.pricing_snapshot import snapshot_pricing
    import json as _json, os as _os

    cfg = load_yaml(args.config or CFG_PATH)
    sources = cfg.get("sources", [])
    pricing_sources = [
        s for s in sources
        if isinstance(s, dict)
        and s.get("parser_strategy") in ("static_config", "http_snapshot_stub")
        and s.get("enabled", True)
    ]

    if not pricing_sources:
        print("No enabled pricing sources found")
        return 0

    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    artifacts = args.artifacts or "artifacts"

    snapshot, diff = snapshot_pricing(
        source_specs=pricing_sources,
        artifacts_root=artifacts,
        date=date,
        run_id=f"pricing-{date}",
    )

    # Write snapshot.
    pricing_dir = _os.path.join(artifacts, "pricing")
    _os.makedirs(pricing_dir, exist_ok=True)
    snap_path = _os.path.join(pricing_dir, f"{date}.json")
    with open(snap_path, "w", encoding="utf-8") as f:
        _json.dump(snapshot.model_dump(), f, ensure_ascii=False, indent=2)

    diff_path = None
    change_count = 0
    has_changes = False
    if diff:
        diff_path = _os.path.join(pricing_dir, f"diff_{date}.json")
        with open(diff_path, "w", encoding="utf-8") as f:
            _json.dump(diff.model_dump(), f, ensure_ascii=False, indent=2)
        change_count = len(diff.changes)
        has_changes = diff.has_changes

    models_count = sum(len(p.models) for p in snapshot.providers)

    if args.json:
        print(_json.dumps({
            "date": date,
            "snapshot_path": snap_path,
            "diff_path": diff_path,
            "providers_count": len(snapshot.providers),
            "models_count": models_count,
            "changes_count": change_count,
            "has_changes": has_changes,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"Pricing snapshot: {len(snapshot.providers)} providers, {models_count} models")
        if diff:
            print(f"Diff: {change_count} changes, has_changes={has_changes}")

    return 0 if not has_changes else 0


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
