"""Pricing Snapshot Adapter — structured model pricing snapshots.

v1: static_config mode reads pricing records from source config.
http_snapshot_stub mode fetches a URL and computes a content hash
without parsing actual prices.

Only pricing changes (diffs) enter the Writer, never full pages.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from agent.schemas import (
    PricingChange,
    PricingDiff,
    PricingModelRecord,
    PricingProviderSnapshot,
    PricingSnapshot,
)
from agent.sources.base import RawItem


class PricingSnapshotAdapter:
    """Collector adapter for pricing snapshot sources.

    Static pricing configs are primarily used by the dedicated pricing diff
    workflow. During normal collection we stay quiet unless a source explicitly
    opts in with ``emit_static_candidates: true``; otherwise unchanged pricing
    pages would create repetitive daily stories.
    """

    type_name = "pricing_snapshot"

    def __init__(self, spec: Dict[str, Any]) -> None:
        self.spec = spec
        self.source_id = spec.get("id", "pricing_snapshot")

    def fetch(self, *, max_items: int = 20) -> List[RawItem]:
        if not self.spec.get("emit_static_candidates", False):
            return []

        now_ts = datetime.now(timezone.utc).isoformat()
        source_url = self.spec.get("source_url") or self.spec.get("url", "")
        provider = self.spec.get("provider", self.source_id)
        items: List[RawItem] = []
        for rec in (self.spec.get("pricing_records") or [])[:max_items]:
            if not isinstance(rec, dict):
                continue
            model = str(rec.get("model", "")).strip()
            if not model:
                continue
            price_bits = []
            if rec.get("input_price_per_m") is not None:
                price_bits.append(f"input {rec.get('input_price_per_m')}/{rec.get('currency', 'USD')} per 1M")
            if rec.get("output_price_per_m") is not None:
                price_bits.append(f"output {rec.get('output_price_per_m')}/{rec.get('currency', 'USD')} per 1M")
            if rec.get("cache_hit_price_per_m") is not None:
                price_bits.append(f"cache hit {rec.get('cache_hit_price_per_m')}/{rec.get('currency', 'USD')} per 1M")
            summary = "; ".join(price_bits) or str(rec.get("notes", "pricing snapshot"))
            items.append(RawItem(
                source_id=self.source_id,
                source_type=self.type_name,
                title=f"{provider} {model} pricing snapshot",
                url=source_url,
                summary=summary,
                published_at=now_ts,
                content_type=self.spec.get("content_type", "pricing_page"),
            ))
        return items


def _compute_content_hash(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode()).hexdigest()[:16]


def snapshot_pricing(
    *,
    source_specs: List[Dict[str, Any]],
    artifacts_root: str,
    date: str,
    run_id: str = "",
) -> Tuple[PricingSnapshot, Optional[PricingDiff]]:
    """Build a pricing snapshot from all enabled pricing sources.

    Returns (snapshot, diff) where diff may be None if no previous
    snapshot exists for comparison.
    """
    providers: List[PricingProviderSnapshot] = []
    now_ts = datetime.now(timezone.utc).isoformat()

    for spec in source_specs:
        if not isinstance(spec, dict):
            continue
        parser = spec.get("parser_strategy", "")
        if parser not in ("static_config", "http_snapshot_stub"):
            continue
        if not spec.get("enabled", True):
            continue

        provider_name = spec.get("provider", spec.get("id", "unknown"))
        source_url = spec.get("source_url", "")
        source_id = spec.get("id", "")

        if parser == "static_config":
            records = _from_static_config(spec, source_url, now_ts)
            raw = json.dumps(spec.get("pricing_records", []), sort_keys=True)
            content_hash = _compute_content_hash(raw)
        elif parser == "http_snapshot_stub":
            records, content_hash = _from_http_stub(source_url, now_ts)
        else:
            continue

        providers.append(PricingProviderSnapshot(
            provider=provider_name,
            source_id=source_id,
            source_url=source_url,
            observed_at=now_ts,
            models=records,
            content_hash=content_hash,
        ))

    snapshot = PricingSnapshot(
        date=date,
        run_id=run_id,
        providers=providers,
    )

    # Diff against previous snapshot.
    diff = _compute_diff(snapshot, artifacts_root, date, run_id)

    return snapshot, diff


def _from_static_config(
    spec: dict, source_url: str, observed_at: str,
) -> List[PricingModelRecord]:
    records: List[PricingModelRecord] = []
    for rec in spec.get("pricing_records", []) or []:
        if not isinstance(rec, dict):
            continue
        records.append(PricingModelRecord(
            provider=spec.get("provider", spec.get("id", "")),
            model=str(rec.get("model", "")),
            input_price_per_m=rec.get("input_price_per_m"),
            output_price_per_m=rec.get("output_price_per_m"),
            cache_hit_price_per_m=rec.get("cache_hit_price_per_m"),
            cache_write_price_per_m=rec.get("cache_write_price_per_m"),
            context_window=rec.get("context_window"),
            currency=str(rec.get("currency", "USD")),
            billing_unit=rec.get("billing_unit"),
            source_url=source_url,
            observed_at=observed_at,
            notes=str(rec.get("notes", "")),
        ))
    return records


def _from_http_stub(
    source_url: str, observed_at: str,
) -> Tuple[List[PricingModelRecord], str]:
    """Fetch URL, compute hash, return stub record."""
    try:
        import httpx
        resp = httpx.get(source_url, timeout=15.0,
                        headers={"User-Agent": "AI-Frontier-Agent/3.0"})
        raw = resp.text[:50000]
    except Exception:
        raw = ""
    content_hash = _compute_content_hash(raw) if raw else "fetch_failed"
    rec = PricingModelRecord(
        model="unknown",
        source_url=source_url,
        observed_at=observed_at,
        notes=f"http_snapshot_stub: hash={content_hash}",
    )
    return [rec], content_hash


# ── Diff logic ──────────────────────────────────────────────────────────


def _compute_diff(
    snapshot: PricingSnapshot,
    artifacts_root: str,
    date: str,
    run_id: str,
) -> Optional[PricingDiff]:
    """Compare current snapshot with the most recent previous one."""
    pricing_dir = os.path.join(artifacts_root, "pricing")
    os.makedirs(pricing_dir, exist_ok=True)

    # Find previous snapshot.
    previous_date = None
    previous: Optional[PricingSnapshot] = None
    if os.path.isdir(pricing_dir):
        files = sorted(
            [f for f in os.listdir(pricing_dir)
             if f.endswith(".json") and not f.startswith("diff_")],
            reverse=True,
        )
        for fn in files:
            if fn.replace(".json", "") < date:
                try:
                    with open(os.path.join(pricing_dir, fn), "r", encoding="utf-8") as f:
                        prev_data = json.load(f)
                    previous = PricingSnapshot.model_validate(prev_data)
                    previous_date = previous.date
                    break
                except Exception:
                    continue

    changes: List[PricingChange] = []
    if previous is None:
        return PricingDiff(
            date=date, run_id=run_id, previous_date=None,
            has_changes=False, changes=[],
        )

    # Compare by provider + model.
    prev_models: Dict[Tuple[str, str], PricingModelRecord] = {}
    for p in previous.providers:
        for m in p.models:
            prev_models[(p.provider, m.model)] = m

    curr_models: Dict[Tuple[str, str], PricingModelRecord] = {}
    for p in snapshot.providers:
        for m in p.models:
            curr_models[(p.provider, m.model)] = m

    for (provider, model), curr in curr_models.items():
        if (provider, model) not in prev_models:
            changes.append(PricingChange(
                provider=provider, model=model, field="model",
                change_type="new_model", source_url=curr.source_url,
            ))
            continue
        prev = prev_models[(provider, model)]
        for field in ["input_price_per_m", "output_price_per_m",
                       "cache_hit_price_per_m", "cache_write_price_per_m",
                       "context_window"]:
            old_val = getattr(prev, field, None)
            new_val = getattr(curr, field, None)
            if old_val != new_val and new_val is not None:
                change_type = "price_decrease" if (
                    isinstance(old_val, (int, float)) and isinstance(new_val, (int, float))
                    and new_val < old_val
                ) else "price_increase" if (
                    isinstance(old_val, (int, float)) and isinstance(new_val, (int, float))
                ) else "metadata_change"
                changes.append(PricingChange(
                    provider=provider, model=model, field=field,
                    old=old_val, new=new_val, change_type=change_type,
                    source_url=curr.source_url,
                ))

    for (provider, model) in prev_models:
        if (provider, model) not in curr_models:
            prev_m = prev_models[(provider, model)]
            changes.append(PricingChange(
                provider=provider, model=model, field="model",
                change_type="removed_model", source_url=prev_m.source_url,
            ))

    return PricingDiff(
        date=date, run_id=run_id, previous_date=previous_date,
        has_changes=len(changes) > 0, changes=changes,
    )


# ── Candidate generation ────────────────────────────────────────────────


def pricing_diff_to_candidates(diff: PricingDiff) -> List[Dict[str, Any]]:
    """Convert pricing diff changes into compact Writer candidates."""
    if not diff.has_changes:
        return []
    candidates: List[Dict[str, Any]] = []
    for ch in diff.changes:
        if ch.change_type == "new_model":
            snippet = f"{ch.provider} new model: {ch.model}"
        elif ch.change_type == "removed_model":
            snippet = f"{ch.provider} removed model: {ch.model}"
        elif ch.change_type == "price_decrease":
            snippet = f"{ch.provider} {ch.model} {ch.field} decreased from {ch.old} to {ch.new}"
        elif ch.change_type == "price_increase":
            snippet = f"{ch.provider} {ch.model} {ch.field} increased from {ch.old} to {ch.new}"
        else:
            snippet = f"{ch.provider} {ch.model} {ch.field} changed: {ch.old} -> {ch.new}"
        candidates.append({
            "title": f"Pricing: {snippet[:120]}",
            "url": ch.source_url,
            "summary": snippet[:500],
            "source": "pricing_diff",
            "source_type": "pricing_snapshot",
            "content_type": "pricing_page",
            "source_tier": "tier_0_core_evidence",
            "evidence_type": "pricing_page",
            "confidence": "high",
            "section_hint": "Cost, Pricing & Access",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "score": 1.30,
        })
    return candidates
