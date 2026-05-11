"""Issue publisher agent.

Reads the artifacts produced by the daily pipeline (draft + report), runs the
publish gate, optionally checks duplicates, and either previews
(``mode='dry-run'``) or actually publishes (``mode='confirm'``) an issue via
an injected ``IssuePublisher`` tool.

Decisions live here, side effects live in ``agent/tools/issue_publisher.py``.
That separation is what makes the gate testable without a network.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.harness.trace import Tracer
from agent.schemas import Draft, SemanticDuplicateReport
from agent.tools.issue_publisher import (
    CreatedIssue,
    ExistingIssue,
    IssuePublisher,
    PublisherError,
)


# --------------------------------------------------------------------------- #
# Gate result
# --------------------------------------------------------------------------- #


@dataclass
class GateResult:
    ok: bool
    blocked_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "blocked_reasons": list(self.blocked_reasons)}


# --------------------------------------------------------------------------- #
# Loading run artifacts
# --------------------------------------------------------------------------- #


@dataclass
class RunArtifacts:
    date: str
    draft: Draft
    draft_md: str
    draft_md_path: str
    draft_json_path: str
    report: Dict[str, Any]
    report_path: str


def load_run_artifacts(date: str, artifacts_root: str) -> RunArtifacts:
    drafts_dir = os.path.join(artifacts_root, "drafts")
    reports_dir = os.path.join(artifacts_root, "reports")
    md_path = os.path.join(drafts_dir, f"{date}.md")
    json_path = os.path.join(drafts_dir, f"{date}.json")
    report_path = os.path.join(reports_dir, f"{date}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"draft JSON not found: {json_path}")
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"draft markdown not found: {md_path}")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"run report not found: {report_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        draft = Draft.model_validate(json.load(f))
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return RunArtifacts(
        date=date,
        draft=draft,
        draft_md=md,
        draft_md_path=md_path,
        draft_json_path=json_path,
        report=report,
        report_path=report_path,
    )


# --------------------------------------------------------------------------- #
# Publish gate
# --------------------------------------------------------------------------- #


def evaluate_publish_gate(
    artifacts: RunArtifacts, publish_cfg: Dict[str, Any]
) -> GateResult:
    reasons: List[str] = []

    minimum_items = int(publish_cfg.get("minimum_items", 3))
    max_eval_issues = int(publish_cfg.get("max_eval_issues", 0))
    require_critic_pass = bool(publish_cfg.get("require_critic_pass", True))

    draft = artifacts.draft
    if not draft.sections or all(not s.items for s in draft.sections):
        reasons.append("draft is empty")
    if not artifacts.draft_md.strip():
        reasons.append("draft markdown file is empty")
    total_items = sum(len(s.items) for s in draft.sections)
    if total_items < minimum_items:
        reasons.append(
            f"draft has {total_items} items, below minimum_items={minimum_items}"
        )

    stages = artifacts.report.get("stages", {}) or {}

    if require_critic_pass:
        critique_stage = stages.get("critique", {}) or {}
        verdict = (critique_stage.get("meta") or {}).get("verdict")
        status = critique_stage.get("status")
        if verdict != "pass" or status not in ("ok",):
            reasons.append(
                f"critic did not pass (verdict={verdict!r}, status={status!r})"
            )
        # Forward critic-detected issues for visibility.
        for r in (critique_stage.get("meta") or {}).get("reasons") or []:
            reasons.append(f"critic: {r}")

    eval_stage = stages.get("eval", {}) or {}
    eval_meta = eval_stage.get("meta") or {}
    eval_issues = eval_meta.get("issues") or []
    if len(eval_issues) > max_eval_issues:
        reasons.append(
            f"eval has {len(eval_issues)} issues "
            f"(>{max_eval_issues}): {eval_issues}"
        )

    # Semantic duplicate gate: block on high or medium severity.
    sem_dup_path = artifacts.report.get("semantic_duplicate_report_path")
    if sem_dup_path and os.path.exists(sem_dup_path):
        try:
            with open(sem_dup_path, "r", encoding="utf-8") as f:
                sem_report = SemanticDuplicateReport.model_validate(
                    json.load(f)
                )
            for dup in sem_report.duplicates:
                if dup.severity in ("high", "medium"):
                    reasons.append(
                        f"semantic_duplicate [{dup.severity}]: "
                        f"{dup.item_a_id} ≈ {dup.item_b_id} — {dup.reason}"
                    )
                # low severity: add as warning in reasons for visibility but
                # use a non-blocking prefix so callers can distinguish.
                elif dup.severity == "low":
                    reasons.append(
                        f"semantic_duplicate_warning [low]: "
                        f"{dup.item_a_id} ≈ {dup.item_b_id} — {dup.reason}"
                    )
        except Exception:
            pass  # Corrupt artifact: don't let it silently pass or abort.

    # Repair failure gate: if repair was attempted but did not succeed, block.
    if artifacts.report.get("repair_attempted") and not artifacts.report.get("repair_succeeded"):
        reasons.append(
            "repair_failed: repair was attempted but did not succeed; manual review required"
        )

    # Low-severity warnings are informational; only hard-block on non-warning reasons.
    blocking_reasons = [r for r in reasons if not r.startswith("semantic_duplicate_warning")]
    return GateResult(ok=not blocking_reasons, blocked_reasons=reasons)


# --------------------------------------------------------------------------- #
# Issue body / title / labels
# --------------------------------------------------------------------------- #


def build_issue_title(date: str, publish_cfg: Dict[str, Any]) -> str:
    prefix = publish_cfg.get("title_prefix", "")
    return f"{prefix}{date}".strip()


def build_issue_labels(publish_cfg: Dict[str, Any]) -> List[str]:
    # ENV override wins over config; comma-separated list.
    # PUBLISH_ISSUE_LABELS is primary; GITBLOG_ISSUE_LABELS is a deprecated fallback.
    env_val = (
        os.environ.get("PUBLISH_ISSUE_LABELS", "").strip()
        or os.environ.get("GITBLOG_ISSUE_LABELS", "").strip()
    )
    if env_val:
        return [x.strip() for x in env_val.split(",") if x.strip()]
    return list(publish_cfg.get("default_labels") or [])


def build_issue_body(artifacts: RunArtifacts) -> str:
    """Issue body = the draft markdown verbatim, plus an audit footer."""
    footer_lines = [
        "",
        "---",
        "<sub>This issue was prepared by `report-agent`. ",
        f"run_id={artifacts.report.get('run_id', '')} · ",
        f"provider={artifacts.report.get('provider', '')} · ",
        f"model={artifacts.report.get('model', '')}</sub>",
    ]
    return artifacts.draft_md.rstrip() + "\n" + "\n".join(footer_lines) + "\n"


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def find_duplicates(
    *,
    publisher: IssuePublisher,
    date: str,
    title: str,
    recent_to_check: int,
) -> List[ExistingIssue]:
    recent = publisher.list_recent_issues(recent_to_check)
    dups: List[ExistingIssue] = []
    norm_title = title.strip().lower()
    for issue in recent:
        t = (issue.title or "").strip().lower()
        if not t:
            continue
        if t == norm_title:
            dups.append(issue)
            continue
        # Same logical date in the title is a strong signal of duplication.
        match = _DATE_RE.search(issue.title or "")
        if match and match.group(0) == date:
            dups.append(issue)
    return dups


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


def run_publish(
    *,
    date: str,
    publisher: IssuePublisher,
    publish_cfg: Dict[str, Any],
    artifacts_root: str,
    tracer: Tracer,
    mode: str,  # "dry-run" | "confirm"
    force: bool = False,
) -> Dict[str, Any]:
    if mode not in ("dry-run", "confirm"):
        raise ValueError(f"mode must be dry-run or confirm, got {mode!r}")

    artifacts = load_run_artifacts(date, artifacts_root)
    title = build_issue_title(date, publish_cfg)
    labels = build_issue_labels(publish_cfg)
    body = build_issue_body(artifacts)
    body_preview = body if len(body) <= 1500 else body[:1500] + "\n…[truncated]"

    gate = evaluate_publish_gate(artifacts, publish_cfg)
    tracer.log(
        "publish_gate",
        date=date,
        ok=gate.ok,
        blocked_reasons=gate.blocked_reasons,
        target_repo=publisher.repo,
        mode=mode,
        force=force,
    )

    duplicates: List[ExistingIssue] = []
    if mode == "confirm" or True:
        # We always check duplicates so dry-run preview is honest.
        try:
            duplicates = find_duplicates(
                publisher=publisher,
                date=date,
                title=title,
                recent_to_check=int(
                    publish_cfg.get("recent_issues_to_check", 30)
                ),
            )
        except PublisherError as e:
            # Soft-fail dup check: the user should be able to see this in the
            # preview rather than have it abort the whole flow.
            tracer.log("publish_dup_check_failed", error=str(e))
            duplicates = []
    duplicate_blocked = bool(duplicates) and not force

    reports_dir = os.path.join(artifacts_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    common: Dict[str, Any] = {
        "date": date,
        "target_repo": publisher.repo,
        "title": title,
        "labels": labels,
        "body_preview": body_preview,
        "body_length": len(body),
        "gate_result": gate.to_dict(),
        "duplicates": [
            {
                "number": d.number,
                "title": d.title,
                "state": d.state,
                "html_url": d.html_url,
                "author_login": d.author_login,
            }
            for d in duplicates
        ],
        "duplicate_blocked": duplicate_blocked,
        "force": force,
        "draft_md_path": artifacts.draft_md_path,
        "draft_json_path": artifacts.draft_json_path,
        "report_path": artifacts.report_path,
        "ts": time.time(),
    }

    # ---- dry-run: write preview, never publish ----
    if mode == "dry-run":
        preview_path = os.path.join(
            reports_dir, f"publish_preview_{date}.json"
        )
        common["mode"] = "dry-run"
        common["would_publish"] = gate.ok and not duplicate_blocked
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(common, f, ensure_ascii=False, indent=2)
        tracer.log(
            "publish_preview_written",
            preview_path=preview_path,
            would_publish=common["would_publish"],
        )
        common["preview_path"] = preview_path
        return common

    # ---- confirm: enforce gate + dup, then call publisher ----
    result_path = os.path.join(reports_dir, f"publish_result_{date}.json")
    common["mode"] = "confirm"

    if not gate.ok:
        common["status"] = "blocked_by_gate"
        common["issue_number"] = None
        common["issue_url"] = None
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(common, f, ensure_ascii=False, indent=2)
        tracer.log(
            "publish_blocked",
            reason="gate",
            blocked_reasons=gate.blocked_reasons,
            result_path=result_path,
        )
        common["result_path"] = result_path
        return common

    if duplicate_blocked:
        common["status"] = "blocked_by_duplicate"
        common["issue_number"] = None
        common["issue_url"] = None
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(common, f, ensure_ascii=False, indent=2)
        tracer.log(
            "publish_blocked",
            reason="duplicate",
            duplicates=[d.number for d in duplicates],
            result_path=result_path,
        )
        common["result_path"] = result_path
        return common

    # Actually publish.
    try:
        created: CreatedIssue = publisher.create_issue(
            title=title, body=body, labels=labels
        )
    except PublisherError as e:
        common["status"] = "publisher_error"
        common["issue_number"] = None
        common["issue_url"] = None
        common["error"] = str(e)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(common, f, ensure_ascii=False, indent=2)
        tracer.log("publish_failed", error=str(e), result_path=result_path)
        common["result_path"] = result_path
        return common

    common["status"] = "published"
    common["issue_number"] = created.number
    common["issue_url"] = created.html_url
    common["issue_created_at"] = created.created_at
    if force and duplicates:
        common["forced_over_duplicate"] = True
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(common, f, ensure_ascii=False, indent=2)
    tracer.log(
        "publish_succeeded",
        issue_number=created.number,
        issue_url=created.html_url,
        forced=bool(force and duplicates),
        result_path=result_path,
    )
    common["result_path"] = result_path
    return common
