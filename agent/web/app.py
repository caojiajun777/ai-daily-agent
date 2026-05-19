"""FastAPI web dashboard and REST API for AI Daily Report.

Provides:
  - Web dashboard: latest report, archive, stats (Jinja2 templates)
  - REST API:    reports, stats, subscribe, unsubscribe (JSON)
  - Run:         uvicorn agent.web.app:app --reload
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

app = FastAPI(title="AI Daily Report", version="2.0.0")

# ── Static files & templates ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

_jinja = Environment(loader=FileSystemLoader(str(BASE_DIR / "templates")))


def _render(name: str, ctx: dict = None) -> HTMLResponse:
    """Render a Jinja2 template, bypassing Starlette's broken TemplateResponse."""
    template = _jinja.get_template(name)
    html = template.render(**(ctx or {}))
    return HTMLResponse(content=html)

ARTIFACTS = os.environ.get("ARTIFACTS_DIR", str(BASE_DIR.parent.parent / "artifacts"))


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _list_reports() -> List[Dict[str, Any]]:
    """List all available report dates with metadata."""
    drafts_dir = os.path.join(ARTIFACTS, "drafts")
    reports_dir = os.path.join(ARTIFACTS, "reports")
    results: List[Dict[str, Any]] = []

    if not os.path.isdir(drafts_dir):
        return results

    for fname in sorted(os.listdir(drafts_dir), reverse=True):
        if not fname.endswith(".json") or fname.startswith("."):
            continue
        date = fname.replace(".json", "")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            continue

        info: Dict[str, Any] = {"date": date}

        # Load draft for metadata.
        draft_path = os.path.join(drafts_dir, fname)
        try:
            with open(draft_path, "r", encoding="utf-8") as f:
                draft = json.load(f)
            info["title"] = draft.get("title", f"AI 日报 {date}")
            item_count = sum(len(s.get("items", [])) for s in draft.get("sections", []))
            info["item_count"] = item_count
            info["section_count"] = len(draft.get("sections", []))
        except Exception:
            info["title"] = f"AI 日报 {date}"
            info["item_count"] = 0
            info["section_count"] = 0

        # Load report for pipeline status.
        report_path = os.path.join(reports_dir, fname)
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                info["is_failed"] = report.get("is_failed", False)
                info["needs_review"] = report.get("needs_human_review", False)
                info["provider"] = report.get("provider", "?")
                info["model"] = report.get("model", "?")
                info["budget"] = report.get("budget", {})
            except Exception:
                info["is_failed"] = False
                info["needs_review"] = False

        md_path = os.path.join(drafts_dir, f"{date}.md")
        info["has_markdown"] = os.path.exists(md_path)
        results.append(info)

    return results


def _load_draft(date: str) -> Optional[Dict[str, Any]]:
    """Load a draft JSON for a specific date."""
    path = os.path.join(ARTIFACTS, "drafts", f"{date}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_markdown(date: str) -> Optional[str]:
    """Load rendered markdown for a specific date."""
    path = os.path.join(ARTIFACTS, "drafts", f"{date}.md")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_subscribers() -> List[str]:
    """Load subscriber emails."""
    from agent.tools.pdf_emailer import _load_subscribers
    return _load_subscribers()


def _load_cost_history() -> List[Dict[str, Any]]:
    """Load token/cost data from all historical pipeline reports."""
    results: List[Dict[str, Any]] = []
    reports_dir = os.path.join(ARTIFACTS, "reports")
    if not os.path.isdir(reports_dir):
        return results
    for fname in sorted(os.listdir(reports_dir), reverse=True):
        if not fname.endswith(".json") or any(x in fname for x in ("publish", "semantic", "repair", "scout", "health")):
            continue
        date = fname.replace(".json", "")
        try:
            with open(os.path.join(reports_dir, fname), "r", encoding="utf-8") as f:
                r = json.load(f)
            budget = r.get("budget", {})
            by_stage = budget.get("by_stage", {})
            results.append({
                "date": date,
                "total_in": budget.get("input_tokens_used", 0),
                "total_out": budget.get("output_tokens_used", 0),
                "total_calls": budget.get("calls_used", 0),
                "total_cost": round(budget.get("total_cost_est", sum(
                    float(s.get("cost", 0)) for s in by_stage.values()
                )), 6),
                "by_stage": by_stage,
                "provider": r.get("provider", "?"),
                "model": r.get("model", "?"),
            })
        except Exception:
            pass
    return results


def _load_papers() -> List[Dict[str, Any]]:
    """Load all paper items from curated artifacts."""
    papers: List[Dict[str, Any]] = []
    curated_dir = os.path.join(ARTIFACTS, "curated")
    if not os.path.isdir(curated_dir):
        return papers
    paper_sources = {"arxiv", "hf_daily_papers"}
    for fname in sorted(os.listdir(curated_dir), reverse=True):
        if not fname.endswith(".json"):
            continue
        date = fname.replace(".json", "")
        try:
            with open(os.path.join(curated_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("items", []):
                src = item.get("source_name", "")
                # Check if this item is from a paper source.
                is_paper = any(p in src for p in paper_sources)
                if is_paper:
                    item["_date"] = date
                    papers.append(item)
        except Exception:
            pass
    return papers


def _source_distribution() -> List[Dict]:
    """Aggregate source distribution across all reports."""
    src_counts: Counter = Counter()
    drafts_dir = os.path.join(ARTIFACTS, "drafts")
    if os.path.isdir(drafts_dir):
        for fname in os.listdir(drafts_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(drafts_dir, fname), "r", encoding="utf-8") as f:
                    draft = json.load(f)
                for sec in draft.get("sections", []):
                    for item in sec.get("items", []):
                        src_counts[item.get("source", "unknown")] += 1
            except Exception:
                pass
    return [{"source": k, "count": v} for k, v in src_counts.most_common(20)]


def _aggregate_stats() -> Dict[str, Any]:
    """Compute aggregate statistics across all reports."""
    reports_dir = os.path.join(ARTIFACTS, "reports")
    total_items = 0
    total_calls = 0
    total_in = 0
    total_out = 0
    report_count = 0
    failures = 0

    if os.path.isdir(reports_dir):
        for fname in os.listdir(reports_dir):
            if not fname.endswith(".json") or "publish" in fname or "semantic" in fname or "repair" in fname:
                continue
            try:
                with open(os.path.join(reports_dir, fname), "r", encoding="utf-8") as f:
                    r = json.load(f)
                report_count += 1
                if r.get("is_failed"):
                    failures += 1
                b = r.get("budget", {})
                total_calls += b.get("calls_used", 0)
                total_in += b.get("input_tokens_used", 0)
                total_out += b.get("output_tokens_used", 0)
                for stage_name, stage in r.get("stages", {}).items():
                    if stage_name == "eval":
                        total_items += stage.get("meta", {}).get("item_count", 0)
            except Exception:
                pass

    return {
        "report_count": report_count,
        "failures": failures,
        "total_items": total_items,
        "total_llm_calls": total_calls,
        "total_tokens_in": total_in,
        "total_tokens_out": total_out,
        "subscriber_count": len(_load_subscribers()),
    }


# ═══════════════════════════════════════════════════════════════════════
# Web dashboard routes
# ═══════════════════════════════════════════════════════════════════════


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page — latest report."""
    reports = _list_reports()
    latest = reports[0] if reports else None
    md_html = ""
    if latest:
        md_text = _load_markdown(latest["date"])
        if md_text:
            import markdown
            # Disable codehilite to avoid complex HTML conflicts.
            md_html = markdown.markdown(
                md_text, extensions=["extra", "tables"],
            )
    stats = _aggregate_stats()
    # Ensure stats values are plain ints.
    stats = {k: int(v) if isinstance(v, (int, float)) else v for k, v in stats.items()}
    return _render("index.html", {
        "request": request,
        "reports": reports[:30],
        "latest": latest,
        "md_html": md_html,
        "stats": stats,
        "now": datetime.now(),
    })


@app.get("/archive", response_class=HTMLResponse)
async def archive(request: Request):
    """Archive — all past reports."""
    reports = _list_reports()
    stats = _aggregate_stats()
    return _render("archive.html", {
        "request": request,
        "reports": reports,
        "stats": stats,
    })


@app.get("/report/{date}", response_class=HTMLResponse)
async def view_report(request: Request, date: str):
    """View a specific report."""
    draft = _load_draft(date)
    if not draft:
        raise HTTPException(status_code=404, detail="Report not found")
    md_text = _load_markdown(date)
    md_html = ""
    if md_text:
        import markdown
        md_html = markdown.markdown(md_text, extensions=["extra", "codehilite", "tables"])
    return _render("report.html", {
        "request": request,
        "date": date,
        "draft": draft,
        "md_html": md_html,
    })


@app.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    """Statistics dashboard."""
    stats = _aggregate_stats()
    sources = _source_distribution()
    reports = _list_reports()[:30]
    return _render("stats.html", {
        "request": request,
        "stats": stats,
        "sources": sources,
        "reports": reports,
    })


@app.get("/papers", response_class=HTMLResponse)
async def papers_page(request: Request):
    """Papers archive — all arxiv/HF papers from curated artifacts."""
    papers = _load_papers()
    return _render("papers.html", {
        "request": request,
        "papers": papers,
        "total": len(papers),
    })


@app.get("/cost", response_class=HTMLResponse)
async def cost_page(request: Request):
    """Token usage & cost dashboard — internal observability."""
    cost_data = _load_cost_history()
    return _render("cost.html", {
        "request": request,
        "cost_data": cost_data,
        "reports": cost_data,
    })


@app.get("/monitor", response_class=HTMLResponse)
async def monitor_page(request: Request):
    """Monitoring dashboard — source health, pipeline health, alerts."""
    from agent.tools.monitor import full_monitoring_report
    report = full_monitoring_report()
    return _render("monitor.html", {
        "request": request,
        "report": report,
        "alerts": report.alerts,
        "summary": report.summary,
        "sources": report.sources,
        "pipelines": report.pipelines,
    })


# ═══════════════════════════════════════════════════════════════════════
# REST API
# ═══════════════════════════════════════════════════════════════════════


@app.get("/api/reports")
async def api_list_reports(limit: int = Query(30, le=100)):
    """List all reports."""
    reports = _list_reports()
    return {"count": len(reports), "reports": reports[:limit]}


@app.get("/api/reports/{date}")
async def api_get_report(date: str):
    """Get a specific report as structured JSON."""
    draft = _load_draft(date)
    if not draft:
        raise HTTPException(status_code=404, detail="Report not found")
    return draft


@app.get("/api/stats")
async def api_stats():
    """Aggregate statistics."""
    return _aggregate_stats()


@app.get("/api/sources")
async def api_sources():
    """Source distribution."""
    return {"sources": _source_distribution()}


@app.post("/api/subscribe")
async def api_subscribe(email: str = Query(...)):
    """Subscribe an email address."""
    import re as _re
    if not _re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    disposable = {"mailinator.com", "guerrillamail.com", "10minutemail.com",
                  "tempmail.com", "yopmail.com", "trashmail.com"}
    domain = email.split("@")[1]
    if domain in disposable:
        raise HTTPException(status_code=400, detail="Disposable email not allowed")

    subs = _load_subscribers()
    if email.lower() in [s.lower() for s in subs]:
        return {"ok": True, "message": "Already subscribed", "email": email}

    # Add to subscribers.txt
    txt_path = os.path.join(str(BASE_DIR.parent.parent), "subscribers.txt")
    try:
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"\n{email}")
    except Exception:
        pass

    return {"ok": True, "message": "Subscribed successfully", "email": email}


@app.get("/api/monitor")
async def api_monitor():
    """Full monitoring report as JSON."""
    from agent.tools.monitor import full_monitoring_report
    report = full_monitoring_report()
    return {
        "checked_at": report.checked_at,
        "alerts": report.alerts,
        "summary": report.summary,
        "sources": [
            {"id": s.source_id, "type": s.source_type, "status": s.status,
             "last_seen": s.last_seen, "days_stale": s.days_since_update, "note": s.note}
            for s in report.sources
        ],
        "pipelines": [
            {"date": p.date, "status": p.status, "items": p.draft_items,
             "llm_calls": p.llm_calls, "cost": p.cost_est}
            for p in report.pipelines[:14]
        ],
    }


@app.post("/api/unsubscribe")
async def api_unsubscribe(email: str = Query(...)):
    """Unsubscribe an email address."""
    txt_path = os.path.join(str(BASE_DIR.parent.parent), "subscribers.txt")
    if not os.path.exists(txt_path):
        raise HTTPException(status_code=404, detail="No subscribers file")

    subs = _load_subscribers()
    email_lower = email.lower()
    new_subs = [s for s in subs if s.lower() != email_lower]

    if len(new_subs) == len(subs):
        return {"ok": True, "message": "Email not found in subscriber list"}

    with open(txt_path, "w", encoding="utf-8") as f:
        for s in new_subs:
            f.write(s + "\n")

    return {"ok": True, "message": "Unsubscribed successfully", "email": email}
