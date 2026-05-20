"""Pydantic schemas for inter-stage artifacts."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class CuratedItem(BaseModel):
    title: str
    url: str
    summary: str
    source: str
    source_type: str
    published_at: str = ""
    score: float = 0.0
    content_type: str = "tech_media"
    source_tier: str = ""
    evidence_type: str = ""
    confidence: str = "medium"
    section_hint: str = ""


class DraftItem(BaseModel):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)          # deep analysis, 100-300 chars
    url: str                                     # primary source
    source: str
    image_url: str = ""                          # extracted og:image from source
    highlights: List[str] = []                   # 2-4 key takeaway bullets
    related_links: List[str] = []                # additional reference URLs
    content_type: str = "tech_media"
    source_tier: str = ""
    evidence_type: str = ""
    confidence: str = "medium"


class DraftSection(BaseModel):
    heading: str = Field(min_length=1)
    items: List[DraftItem]


class Draft(BaseModel):
    date: str
    title: str
    overview: str = ""
    cover_image: str = ""                        # optional cover image URL
    sections: List[DraftSection]


class CritiqueResult(BaseModel):
    verdict: str  # "pass" | "reject"
    reasons: List[str] = []
    score: int = 0
    quality_flags: List[str] = []


class CuratedItemRecord(BaseModel):
    """Persistent record of one item selected by the curator."""

    raw_item_id: str
    title: str
    source_url: str
    source_name: str
    published_at: Optional[str] = None
    score: float
    section: Optional[str] = None       # back-filled after write stage
    selected_reason: str
    duplicate_group_id: Optional[str] = None
    used_in_draft: bool = True
    content_type: str = "tech_media"
    source_tier: str = ""
    reliability: str = ""
    confidence: str = "medium"
    evidence_type: str = ""
    section_hint: str = ""


class CuratedOutput(BaseModel):
    """Top-level envelope written to artifacts/curated/<date>.json."""

    date: str
    run_id: str
    items: List[CuratedItemRecord]


class SemanticDuplicate(BaseModel):
    item_a_id: str
    item_b_id: str
    item_a_title: str
    item_b_title: str
    reason: str
    severity: Literal["low", "medium", "high"]


class SemanticDuplicateReport(BaseModel):
    """Written to artifacts/reports/semantic_duplicates_<date>.json."""

    date: str
    run_id: str
    duplicates: List[SemanticDuplicate] = []
    ok: bool
    checked_item_count: int
    provider: Optional[str] = None


class RepairAction(BaseModel):
    """One item-level change made during repair."""

    section: str
    removed_title: str
    removed_url: str
    replacement_url: Optional[str] = None   # None → item simply dropped
    replacement_title: Optional[str] = None
    reason: str


class RepairReport(BaseModel):
    """Written to artifacts/reports/repair_<date>.json."""

    date: str
    run_id: str
    attempted: bool
    succeeded: bool
    reason: str                     # why repair was triggered / why it failed
    actions: List[RepairAction] = []
    pre_duplicate_count: int = 0
    post_duplicate_count: Optional[int] = None
    draft_version: str = "v1"       # "v1" | "v2"


# ═══════════════════════════════════════════════════════════════════════
# Trend Intelligence Layer schemas
# ═══════════════════════════════════════════════════════════════════════


class TrendEvidence(BaseModel):
    date: str = ""
    event_id: str = ""
    title: str = ""
    source_names: List[str] = []
    urls: List[str] = []
    section: str = ""
    priority: str = ""
    evidence_level: str = ""


class TrendFinding(BaseModel):
    trend_id: str = ""
    editorial_title: str = ""
    analytical_title: str = ""
    trend_type: Literal["topic", "entity", "capability", "market", "weak_signal", "noise"] = "topic"
    direction: Literal["rising", "stable", "declining", "mixed"] = "stable"
    confidence: Literal["high", "medium", "low"] = "medium"
    window_type: Literal["short_signal", "weekly_trend", "confirmed_trend", "structural_movement"] = "short_signal"
    summary: str = ""
    evidence_event_ids: List[str] = []
    timeline_evidence: List[TrendEvidence] = []
    supporting_metrics: dict = Field(default_factory=dict)
    companies_to_watch: List[str] = []
    why_it_matters: str = ""
    implications: str = ""
    counter_signals: str = ""
    risk_of_overinterpretation: str = ""
    what_to_watch_next: str = ""


class HeatChange(BaseModel):
    category: str = ""
    direction: Literal["heating", "cooling", "stable"] = "stable"
    evidence: str = ""
    evidence_event_ids: List[str] = []


# ═══════════════════════════════════════════════════════════════════════
# Pricing Snapshot Layer schemas
# ═══════════════════════════════════════════════════════════════════════


class PricingModelRecord(BaseModel):
    provider: str = ""
    model: str = ""
    input_price_per_m: Optional[float] = None
    output_price_per_m: Optional[float] = None
    cache_hit_price_per_m: Optional[float] = None
    cache_write_price_per_m: Optional[float] = None
    context_window: Optional[int] = None
    currency: str = "USD"
    billing_unit: Optional[str] = None
    source_url: str = ""
    observed_at: str = ""
    notes: str = ""


class PricingProviderSnapshot(BaseModel):
    provider: str = ""
    source_id: str = ""
    source_url: str = ""
    observed_at: str = ""
    models: list[PricingModelRecord] = []
    content_hash: str = ""


class PricingSnapshot(BaseModel):
    date: str = ""
    run_id: str = ""
    providers: list[PricingProviderSnapshot] = []


class PricingChange(BaseModel):
    provider: str = ""
    model: str = ""
    field: str = ""
    old: Optional[float] = None
    new: Optional[float] = None
    change_type: str = "price_decrease"  # price_increase | price_decrease | new_model | removed_model | context_change | metadata_change
    source_url: str = ""


class PricingDiff(BaseModel):
    date: str = ""
    run_id: str = ""
    previous_date: Optional[str] = None
    has_changes: bool = False
    changes: list[PricingChange] = []


class TrendReport(BaseModel):
    report_id: str = ""
    generated_at: str = ""
    days: int = 7
    start_date: str = ""
    end_date: str = ""
    headline_summary: str = ""
    findings: List[TrendFinding] = []
    heat_changes: List[HeatChange] = []
    weak_signals: List[TrendFinding] = []
    noise_or_hype: List[TrendFinding] = []
    next_week_watchlist: List[str] = []
    data_quality_notes: str = ""
    total_events: int = 0
    total_findings: int = 0
    metrics_fallback_used: bool = False
    validation_warnings: List[str] = []
    taxonomy_counts: dict = Field(default_factory=dict)
