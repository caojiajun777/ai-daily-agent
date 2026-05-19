"""Trend Intelligence Layer — structured, evidence-grounded industry analysis.

Reads curated artifacts, computes trend metrics, calls an LLM for editorial
findings (with forced structured JSON), validates, and saves TrendReports.

Usage:
  python -m agent.cli trends --days 7 --provider deepseek
  python -m agent.cli trends --days 30 --multi-window 4,7,14,30
"""

from __future__ import annotations

import json as _json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.llm.base import LLMMessage
from agent.schemas import TrendFinding, TrendReport, TrendEvidence, HeatChange

_TREND_SYSTEM_PROMPT = """你是一个 AI 行业趋势分析师，基于定量指标和事件时间线识别和评估趋势。

## 输出格式
必须输出严格 JSON，格式如下：
{
  "headline_summary": "1-2句本周核心总结",
  "findings": [
    {
      "trend_id": "trend_001",
      "editorial_title": "趋势：xxx",
      "analytical_title": "详细分析标题",
      "trend_type": "topic | entity | capability | market | weak_signal | noise",
      "direction": "rising | stable | declining | mixed",
      "confidence": "high | medium | low",
      "window_type": "short_signal | weekly_trend",
      "summary": "趋势描述，2-3句",
      "evidence_event_ids": ["evt_xxx", "evt_yyy"],
      "companies_to_watch": ["公司A", "公司B"],
      "why_it_matters": "为什么重要",
      "implications": "影响分析",
      "counter_signals": "反面信号或风险",
      "risk_of_overinterpretation": "过度解读风险",
      "what_to_watch_next": "接下来关注什么"
    }
  ],
  "heat_changes": [
    {"category": "大模型发布与升级 | AI Agent与工具链 | 融资与资本市场 | ...",
     "direction": "heating | cooling | stable",
     "evidence": "一句话证据",
     "evidence_event_ids": ["evt_xxx"]}
  ],
  "weak_signals": [...],
  "noise_or_hype": [...],
  "next_week_watchlist": ["关注项1", "关注项2"]
}

## 分析规则
1. trend_id: 自拟，如 trend_model_cost_war
2. evidence_event_ids: 必须来自输入的候选趋势，不可编造
3. confidence: 有3+独立来源事件 + 跨多天 = high；单日2事件 = medium；单来源 = low
4. trend_type: 置信度低 → weak_signal；重复报道无新信息 → noise
5. companies_to_watch: 从事件中提取实际出现的公司名
6. 不要编造公司、URL、不在输入中的事件
7. 如果证据不足以支撑某趋势，放入 weak_signals 或 noise_or_hype"""


def load_timeline(
    artifacts_dir: str = "artifacts", days: int = 7,
) -> List[Dict[str, Any]]:
    curated_dir = os.path.join(artifacts_dir, "curated")
    if not os.path.isdir(curated_dir):
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    timeline: List[Dict[str, Any]] = []
    for fname in sorted(os.listdir(curated_dir), reverse=True):
        if not fname.endswith(".json"):
            continue
        date = fname.replace(".json", "")
        if date < cutoff:
            continue
        try:
            with open(os.path.join(curated_dir, fname), "r", encoding="utf-8") as f:
                data = _json.load(f)
            items = data.get("items", [])
            if items:
                timeline.append({"date": date, "count": len(items), "items": items})
        except Exception:
            pass
    return timeline


def organize_by_theme(
    timeline: List[Dict[str, Any]],
) -> Dict[str, List[Dict]]:
    themes: Dict[str, List[Dict]] = defaultdict(list)
    theme_keywords = {
        "大模型发布与升级": ["模型", "model", "release", "发布", "参数", "benchmark"],
        "AI Agent与工具链": ["agent", "智能体", "codex", "copilot", "mcp", "tool", "sdk"],
        "AI产品与商业化": ["product", "产品", "feature", "pricing", "定价", "用户"],
        "融资与资本市场": ["融资", "funding", "估值", "ipo", "收购", "billion", "亿"],
        "AI安全与治理": ["安全", "safety", "alignment", "监管", "regulation", "风险"],
        "研究前沿": ["论文", "paper", "arxiv", "research", "benchmark", "sota"],
        "AI基础设施": ["gpu", "算力", "芯片", "云", "推理", "部署", "量化"],
    }
    for day in timeline:
        date = day["date"]
        for item in day.get("items", []):
            title = item.get("title", "")
            text = (title + " " + (item.get("summary", "") or "")).lower()
            best_theme, best_score = "其他", 0
            for theme, keywords in theme_keywords.items():
                score = sum(1 for kw in keywords if kw.lower() in text)
                if score > best_score:
                    best_score, best_theme = score, theme
            themes[best_theme].append({
                "date": date, "title": title,
                "source_name": item.get("source_name", ""),
                "source_url": item.get("source_url", ""),
                "section": item.get("section", ""),
                "theme": best_theme,
            })
    return dict(themes)


def build_prompt(
    timeline: List[Dict[str, Any]],
    themes: Dict[str, List[Dict]],
) -> str:
    lines: List[str] = []
    total_items = sum(d["count"] for d in timeline)
    dr = f"{timeline[-1]['date']} ~ {timeline[0]['date']}" if timeline else "N/A"
    lines.append(f"## 数据概览: {dr}, {len(timeline)}期, {total_items}条\n")

    lines.append("## 每日关键事件")
    for day in timeline:
        lines.append(f"\n### {day['date']}（{day['count']}条）")
        for item in day["items"][:12]:
            t = item.get("title", "")[:100]
            s = item.get("section", "")
            src = item.get("source_name", "")
            lines.append(f"- [{s}] {t}（{src}）")
    lines.append("")

    lines.append("## 按主题聚类")
    for theme, items in sorted(themes.items(), key=lambda x: -len(x[1])):
        if len(items) < 2:
            continue
        dates = sorted(set(i["date"] for i in items))
        lines.append(f"\n### {theme}（{len(items)}条, {dates[0]}~{dates[-1]}）")
        for item in items[:8]:
            lines.append(f"- [{item['date']}] {item['title'][:100]}")
    lines.append("")

    return "\n".join(lines)


def analyze_trends(
    *, provider, artifacts_dir: str = "artifacts", days: int = 7,
    tracer=None, budget=None, output_dir: str = "",
) -> Dict[str, Any]:
    """Run the full trend analysis pipeline. Returns a result dict."""
    from agent.tools.trend_metrics import (
        compute_event_metrics, compute_trend_signals, summarize_tags, tag_event,
    )
    from agent.agents.trend_validator import validate_report, metrics_only_report

    result: Dict[str, Any] = {"ok": False, "findings": 0, "warnings": 0}

    timeline = load_timeline(artifacts_dir, days)
    if not timeline:
        result["error"] = f"No curated data in past {days} days"
        return result

    start_date = timeline[-1]["date"]
    end_date = timeline[0]["date"]
    total_events = sum(d["count"] for d in timeline)
    themes = organize_by_theme(timeline)

    # Build a flat event list with IDs.
    all_events: List[Dict] = []
    for day in timeline:
        for item in day.get("items", []):
            eid = f"evt_{day['date']}_{item.get('source_name','')[:20]}_{hash(item.get('title','')) % 10000:04d}"
            eid = re.sub(r"[^a-zA-Z0-9_]", "_", eid)[:50]
            all_events.append({
                "event_id": eid, "date": day["date"],
                "title": item.get("title", ""),
                "source_names": [item.get("source_name", "")],
                "urls": [item.get("source_url", "")],
                "section": item.get("section", ""),
                "priority": item.get("priority", ""),
                "evidence_level": item.get("evidence_level", ""),
                "novelty": item.get("novelty", ""),
            })

    compute_event_metrics(all_events)
    valid_event_ids = {e["event_id"] for e in all_events}

    # Tag events with taxonomy.
    for e in all_events:
        e["_tags"] = tag_event(
            title=e.get("title", ""), section=e.get("section", ""),
        )
    taxonomy_counts = summarize_tags(all_events)

    # Build prompt and call LLM.
    report: Optional[TrendReport] = None
    prompt = build_prompt(timeline, themes)

    try:
        if budget:
            budget.check_can_call(stage="trend")
        response = provider.complete(
            messages=[
                LLMMessage(role="system", content=_TREND_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.2, max_output_tokens=4096,
        )
        if tracer:
            tracer.log_llm_call(
                provider=provider.name, model=provider.model,
                prompt=_TREND_SYSTEM_PROMPT + "\n" + prompt,
                output=response.text, latency_ms=response.latency_ms,
                status="ok", stage="trend",
            )
        if budget:
            budget.record(
                stage="trend", input_tokens=response.input_tokens_est,
                output_tokens=response.output_tokens_est,
            )
        raw = _extract_json(response.text)
        try:
            payload = _json.loads(raw)
            report = TrendReport.model_validate(_clean_payload(payload))
        except Exception:
            report = None
    except Exception as e:
        if tracer:
            tracer.log("trend_llm_failed", error=str(e))

    if report is None:
        report = metrics_only_report(
            days=days, start_date=start_date, end_date=end_date,
            total_events=total_events,
        )
        result["fallback_used"] = True
    else:
        result["fallback_used"] = False

    # Compute metrics per finding group.
    metrics_by_group: Dict[str, Dict] = {}
    for f in report.findings:
        group = [e for e in all_events if e["event_id"] in f.evidence_event_ids]
        signals = compute_trend_signals(group, window_days=days, all_events=all_events)
        f.supporting_metrics = signals
        metrics_by_group[f.trend_id] = signals
        # Build timeline_evidence.
        for g in group[:8]:
            f.timeline_evidence.append(TrendEvidence(
                date=g.get("date", ""), event_id=g.get("event_id", ""),
                title=g.get("title", ""), source_names=g.get("source_names", []),
                urls=g.get("urls", []), section=g.get("section", ""),
            ))

    # Validate.
    report = validate_report(
        report, valid_event_ids=valid_event_ids, window_days=days,
        metrics_by_group=metrics_by_group,
    )

    report.report_id = f"trends-{end_date}-{days}d"
    report.generated_at = datetime.now(timezone.utc).isoformat()
    report.days = days
    report.start_date = start_date
    report.end_date = end_date
    report.total_events = total_events
    report.total_findings = len(report.findings)
    report.taxonomy_counts = taxonomy_counts

    # Save.
    saved_paths = _save_report(report, output_dir or os.path.join(artifacts_dir, "trends"))

    result["ok"] = True
    result["findings"] = len(report.findings)
    result["weak_signals"] = len(report.weak_signals)
    result["noise"] = len(report.noise_or_hype)
    result["warnings"] = len(report.validation_warnings)
    result["paths"] = saved_paths
    result["report"] = report

    return result


def _extract_json(text: str) -> str:
    raw = text.strip()
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


def _clean_payload(payload: dict) -> dict:
    """Ensure the payload has all required fields for TrendReport."""
    payload.setdefault("findings", [])
    payload.setdefault("heat_changes", [])
    payload.setdefault("weak_signals", [])
    payload.setdefault("noise_or_hype", [])
    payload.setdefault("next_week_watchlist", [])
    payload.setdefault("headline_summary", "")
    # Ensure each finding has trend_id.
    for i, f in enumerate(payload.get("findings", [])):
        if not f.get("trend_id"):
            f["trend_id"] = f"trend_{i:03d}"
    return payload


def _save_report(report: TrendReport, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    base = f"{report.end_date}_{report.days}d"
    paths: Dict[str, str] = {}

    json_path = os.path.join(output_dir, f"{base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(report.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    paths["json"] = json_path

    md_path = os.path.join(output_dir, f"{base}.md")
    md = _render_markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    paths["md"] = md_path

    return paths


def analyze_multi_window(
    *, provider, artifacts_dir: str = "artifacts",
    windows: List[int] = None, tracer=None, budget=None,
) -> Dict[str, Any]:
    """Run trend analysis across multiple time windows."""
    if windows is None:
        windows = [4, 7, 14, 30]
    results: Dict[str, Any] = {}
    for w in windows:
        r = analyze_trends(
            provider=provider, artifacts_dir=artifacts_dir, days=w,
            tracer=tracer, budget=budget,
        )
        results[f"{w}d"] = r
    return results


# ── Markdown rendering ──────────────────────────────────────────────────────

def _render_markdown(report: TrendReport) -> str:
    lines: List[str] = []
    lines.append(f"# AI 行业趋势报告（{report.start_date} ~ {report.end_date}）")
    lines.append(f"> 生成时间：{report.generated_at[:19]} | 窗口：{report.days}天 | 事件：{report.total_events}条")
    if report.metrics_fallback_used:
        lines.append("> ⚠ LLM skipped — metrics-only report")
    lines.append("")

    if report.headline_summary:
        lines.append(f"**核心摘要：** {report.headline_summary}")
        lines.append("")

    lines.append(f"## 趋势发现（{len(report.findings)}条）")
    for f in report.findings:
        m = f.supporting_metrics
        lines.append(f"\n### {f.editorial_title or f.trend_id}")
        lines.append(f"- 类型：{f.trend_type} | 方向：{f.direction} | 置信度：{f.confidence} | 窗口：{f.window_type}")
        lines.append(f"- 事件数：{m.get('event_count','?')} | 活跃天数：{m.get('active_days','?')} | 动量：{m.get('momentum','?')}")
        if f.summary:
            lines.append(f"- {f.summary}")
        if f.why_it_matters:
            lines.append(f"- 重要性：{f.why_it_matters}")
        if f.companies_to_watch:
            lines.append(f"- 关注公司：{', '.join(f.companies_to_watch)}")
        if f.evidence_event_ids:
            lines.append(f"- 证据ID：{', '.join(f.evidence_event_ids[:5])}")

    if report.heat_changes:
        lines.append(f"\n## 领域热度变化（{len(report.heat_changes)}项）")
        for h in report.heat_changes:
            lines.append(f"- [{h.direction}] {h.category}: {h.evidence}")

    if report.weak_signals:
        lines.append(f"\n## 微弱信号（{len(report.weak_signals)}条）")
        for f in report.weak_signals:
            lines.append(f"- {f.editorial_title or f.trend_id}: {f.summary[:100]}")

    if report.noise_or_hype:
        lines.append(f"\n## 噪音/炒作（{len(report.noise_or_hype)}条）")
        for f in report.noise_or_hype:
            lines.append(f"- {f.editorial_title or f.trend_id}: {f.summary[:100]}")

    if report.next_week_watchlist:
        lines.append(f"\n## 下周关注")
        for w in report.next_week_watchlist:
            lines.append(f"- {w}")

    if report.validation_warnings:
        lines.append(f"\n## 校验警告（{len(report.validation_warnings)}条）")
        for w in report.validation_warnings[:10]:
            lines.append(f"- {w}")

    lines.append(f"\n---\n*自动生成于 {report.generated_at[:19]}*")
    return "\n".join(lines)
