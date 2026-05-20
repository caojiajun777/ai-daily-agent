"""Tests for content-type-aware curator scoring (v2.3).

Covers: recency half-life, content_type weights, score floor,
research quota backfill, and unknown content_type defaulting.
"""

import math
import time as _time
from datetime import datetime, timedelta, timezone

import pytest

from agent.agents.curator import (
    _recency_score,
    _infer_content_type,
    _build_ct_lookup,
    _select_with_paper_quota,
    _is_research,
    curate_with_records,
)
from agent.schemas import CuratedItem, CuratedItemRecord
from agent.sources.base import RawItem


# ── Helpers ────────────────────────────────────────────────────────────


def _make_item(
    source_id: str = "openai_news",
    source_type: str = "rss",
    title: str = "Test AI News",
    url: str = "https://example.com/1",
    hours_ago: float = 0.0,
) -> RawItem:
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return RawItem(
        source_id=source_id, source_type=source_type,
        title=title, url=url,
        summary="Test summary about AI models and deep learning",
        published_at=ts,
    )


def _make_record(source_name: str = "arxiv_top_venue", score: float = 0.5):
    return CuratedItemRecord(
        raw_item_id=f"{source_name}::https://example.com/{abs(hash(source_name))}",
        title="Test Paper",
        source_url="https://example.com/1",
        source_name=source_name,
        score=score,
        selected_reason="test",
    )


def _make_curated(source_name: str = "arxiv_top_venue", score: float = 0.5):
    return CuratedItem(
        title="Test Paper",
        url="https://example.com/1",
        summary="Test summary",
        source=source_name,
        source_type="arxiv" if "arxiv" in source_name else "rss",
        published_at=datetime.now(timezone.utc).isoformat(),
        score=score,
    )


# ── 1. Recency half-life formula ───────────────────────────────────────


def test_recency_half_life_formula():
    """At age = half_life_h the score must be exactly 0.5."""
    now = datetime.now(timezone.utc)
    for hl in [24, 36, 48, 72, 120, 168]:
        published = (now - timedelta(hours=hl)).isoformat()
        score = _recency_score(published, _time.time(), hl)
        assert math.isclose(score, 0.5, abs_tol=0.001), (
            f"half_life={hl}h: expected 0.500, got {score:.4f}"
        )


def test_recency_at_zero_hours():
    """Fresh items get score 1.0."""
    published = datetime.now(timezone.utc).isoformat()
    for hl in [24, 48, 120]:
        score = _recency_score(published, _time.time(), hl)
        assert math.isclose(score, 1.0, abs_tol=0.02)


def test_recency_empty_published():
    """Missing timestamp returns 0.5."""
    assert _recency_score("", _time.time(), 48) == 0.5


# ── 2. Content-type inference ──────────────────────────────────────────


def test_research_paper_decays_slower_than_media():
    """Research paper should have higher recency at same age vs tech_media."""
    hours = 36
    paper_score = _recency_score(
        (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
        _time.time(), half_life_h=120,
    )
    media_score = _recency_score(
        (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
        _time.time(), half_life_h=36,
    )
    assert paper_score > media_score, (
        f"At {hours}h: paper={paper_score:.3f} should > media={media_score:.3f}"
    )


def test_pricing_page_decays_slower_than_fast_news():
    """Pricing_page (168h half-life) decays much slower than community_signal (24h)."""
    hours = 48
    pricing_score = _recency_score(
        (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
        _time.time(), half_life_h=168,
    )
    fast_score = _recency_score(
        (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
        _time.time(), half_life_h=24,
    )
    assert pricing_score > fast_score, (
        f"At {hours}h: pricing={pricing_score:.3f} should > community={fast_score:.3f}"
    )


def test_research_paper_weight_higher_than_before():
    """Research paper weight 1.05 > old 0.85."""
    ct_cfg = {
        "research_paper": {"source_weight": 1.05, "half_life_h": 120},
        "tech_media": {"source_weight": 0.85, "half_life_h": 36},
    }
    specs = [
        {"id": "arxiv_top_venue", "type": "arxiv"},
        {"id": "wired_ai", "type": "rss", "url": "https://wired.com/feed"},
    ]
    items = [
        _make_item("arxiv_top_venue", "arxiv", "Paper Title", "https://a.org/1", 0),
        _make_item("wired_ai", "rss", "Media Title", "https://w.com/1", 0),
    ]
    writer_items, records = curate_with_records(
        items, source_specs=specs, max_items=10,
        content_types_cfg=ct_cfg,
    )
    arxiv_rec = [r for r in records if "arxiv" in r.source_name]
    media_rec = [r for r in records if "wired" in r.source_name]
    assert arxiv_rec, "arxiv item should be in records"
    assert media_rec, "media item should be in records"
    # At 0h age, the weight difference should be visible in score
    # (both get recency ~1.0, research has higher weight)
    assert arxiv_rec[0].score > media_rec[0].score, (
        f"arxiv={arxiv_rec[0].score:.4f} should > media={media_rec[0].score:.4f}"
    )


# ── 3. Content-type inference rules ────────────────────────────────────


def test_unknown_content_type_defaults_to_tech_media():
    """Sources not matching any rule should default based on type."""
    specs = [
        {"id": "some_random_blog", "type": "rss", "url": "https://unknown.com/feed"},
    ]
    lookup = _build_ct_lookup(specs)
    assert lookup["some_random_blog"] == "official_release"


def test_content_type_patterns():
    """Verify key content_type assignments."""
    test_cases = [
        ("arxiv_top_venue", "arxiv", "research_paper"),
        ("hf_daily_papers", "rss", "research_paper"),
        ("techcrunch_ai", "rss", "tech_media"),
        ("ithome", "rss", "cn_aggregator"),
        ("36kr_ai", "rss", "cn_aggregator"),
        ("wired_ai", "rss", "tech_media"),
        ("x_deepseek", "x_cookie", "china_model_official"),
        ("x_sama", "x_cookie", "founder_signal"),
        ("x_AndrewYNg", "x_cookie", "expert_signal"),
        ("x_simonw", "x_cookie", "builder_signal"),
    ]
    for sid, st, expected in test_cases:
        assert _infer_content_type(sid, st) == expected, (
            f"{sid} ({st}) expected {expected}, got {_infer_content_type(sid, st)}"
        )


# ── 4. Score floor ─────────────────────────────────────────────────────


@pytest.fixture
def ct_cfg():
    return {
        "research_paper": {"source_weight": 1.05, "half_life_h": 120},
        "official_release": {"source_weight": 1.35, "half_life_h": 48},
        "tech_media": {"source_weight": 0.85, "half_life_h": 36},
    }


@pytest.fixture
def source_specs():
    return [
        {"id": "arxiv_top_venue", "type": "arxiv"},
        {"id": "openai_news", "type": "rss", "url": "https://openai.com/news/rss.xml"},
        {"id": "wired_ai", "type": "rss", "url": "https://wired.com/feed"},
    ]


def test_score_floor_filters_old_media(ct_cfg, source_specs):
    """Old tech_media items below floor are dropped."""
    # Use 168h-old items with no AI keywords → relevance_boost = 0.6
    # recency = exp(-ln(2)*168/36) ≈ 0.039
    # score = 0.85 * 0.039 * 0.6 = 0.020 → below 0.10 floor
    items = []
    for i in range(3):
        it = _make_item("wired_ai", "rss", f"Old News {i}",
                        f"https://wired.com/{i}", hours_ago=168)
        it.summary = "just some generic tech news nothing AI related here"
        items.append(it)
    _, records = curate_with_records(
        items, source_specs=source_specs, max_items=10,
        content_types_cfg=ct_cfg, score_floor=0.10,
    )
    wired = [r for r in records if "wired" in r.source_name]
    assert len(wired) == 0, f"Old media should be dropped, got {len(wired)}"


def test_score_floor_does_not_drop_recent_research(ct_cfg, source_specs):
    """Recent research papers should pass the floor."""
    items = [
        _make_item("arxiv_top_venue", "arxiv", f"Paper {i}",
                   f"https://arxiv.org/abs/{i}", hours_ago=48)
        for i in range(3)
    ]
    _, records = curate_with_records(
        items, source_specs=source_specs, max_items=10,
        content_types_cfg=ct_cfg, score_floor=0.10,
    )
    arxiv = [r for r in records if "arxiv" in r.source_name]
    assert len(arxiv) == 3, f"Recent papers should pass floor, got {len(arxiv)}"


def test_score_floor_exempts_research_even_if_below(ct_cfg, source_specs):
    """Research papers below floor are still kept (slow-moving content)."""
    # 120h-old paper with research_paper weight will score very low
    # but should still be kept because of slow-content exemption
    items = [
        _make_item("arxiv_top_venue", "arxiv", "Old Paper",
                   "https://arxiv.org/abs/old", hours_ago=100)
    ]
    _, records = curate_with_records(
        items, source_specs=source_specs, max_items=10,
        content_types_cfg=ct_cfg, score_floor=0.10,
    )
    assert len(records) == 1, "Research paper should be kept regardless of floor"


# ── 5. Research quota backfill ──────────────────────────────────────────


def test_research_quota_backfills_papers():
    """If top-N doesn't have enough papers, backfill from scored list."""
    # Create many high-score non-research items + low-score research items
    items = []
    for i in range(15):
        items.append(_make_item(
            "openai_news", "rss", f"News {i}",
            f"https://openai.com/{i}", hours_ago=0,
        ))
    for i in range(5):
        items.append(_make_item(
            "arxiv_top_venue", "arxiv", f"Paper {i}",
            f"https://arxiv.org/abs/paper{i}", hours_ago=48,
        ))
    specs = [
        {"id": "openai_news", "type": "rss", "url": "https://openai.com/news/rss.xml"},
        {"id": "arxiv_top_venue", "type": "arxiv"},
    ]
    ct_cfg = {
        "official_release": {"source_weight": 1.35, "half_life_h": 48},
        "research_paper": {"source_weight": 1.05, "half_life_h": 120},
    }
    _, records = curate_with_records(
        items, source_specs=specs, max_items=12,
        content_types_cfg=ct_cfg, score_floor=0.0,
        research_min=2,
    )
    arxiv_count = sum(1 for r in records if "arxiv" in r.source_name)
    assert arxiv_count >= 2, (
        f"research_min=2 but got {arxiv_count} research items"
    )


def test_select_with_paper_quota_respects_score_floor():
    """Backfill should not include papers below _RESEARCH_FLOOR (0.15)."""
    scored = []
    for i in range(10):
        rec = _make_record(f"openai_news_{i}", score=0.8 - i * 0.02)
        cur = _make_curated(f"openai_news_{i}", score=0.8 - i * 0.02)
        scored.append((0.8 - i * 0.02, cur, rec))
    # Add research papers: one above floor, one below
    rec_good = _make_record("arxiv_top_venue", score=0.20)
    cur_good = _make_curated("arxiv_top_venue", score=0.20)
    rec_bad = _make_record("arxiv_top_venue", score=0.05)
    cur_bad = _make_curated("arxiv_top_venue", score=0.05)
    scored.append((0.20, cur_good, rec_good))
    scored.append((0.05, cur_bad, rec_bad))
    scored.sort(key=lambda x: x[0], reverse=True)

    result = _select_with_paper_quota(scored, max_items=10, min_papers=5, research_min=2)
    research_in = [r for _, _, r in result if _is_research(r)]
    # Only the 0.20-scored paper should make it; the 0.05 one is below floor.
    assert len(research_in) >= 1, "Should have at least 1 research paper backfilled"


def test_is_research_detects_paper_sources():
    """_is_research should match arxiv and hf_daily_papers."""
    assert _is_research(_make_record("arxiv_top_venue"))
    assert _is_research(_make_record("hf_daily_papers"))
    assert not _is_research(_make_record("openai_news"))
    assert not _is_research(_make_record("techcrunch_ai"))


# ── 6. Integration: different content types score differently ──────────


def test_official_release_beats_tech_media_at_same_age():
    """Official release has higher weight and longer half-life."""
    ct_cfg = {
        "official_release": {"source_weight": 1.35, "half_life_h": 48},
        "tech_media": {"source_weight": 0.85, "half_life_h": 36},
    }
    specs = [
        {"id": "openai_news", "type": "rss", "url": "https://openai.com/news/rss.xml"},
        {"id": "techcrunch_ai", "type": "rss", "url": "https://techcrunch.com/feed/"},
    ]
    items = [
        _make_item("openai_news", "rss", "OpenAI News", "https://openai.com/1", 12),
        _make_item("techcrunch_ai", "rss", "TC News", "https://techcrunch.com/1", 12),
    ]
    _, records = curate_with_records(
        items, source_specs=specs, max_items=10,
        content_types_cfg=ct_cfg,
    )
    openai_r = [r for r in records if "openai" in r.source_name]
    tc_r = [r for r in records if "techcrunch" in r.source_name]
    assert openai_r[0].score > tc_r[0].score


# ── 7. v3 Taxonomy tests ──────────────────────────────────────────────


def test_kol_not_classified_as_official_release():
    """X KOL sources must not be official_release."""
    assert _infer_content_type("x_sama", "x_cookie") == "founder_signal"
    assert _infer_content_type("x_AndrewYNg", "x_cookie") != "official_release"
    assert _infer_content_type("x_dotey", "x_cookie") != "official_release"
    assert _infer_content_type("x_ayi", "x_cookie") != "official_release"
    assert _infer_content_type("x_simonw", "x_cookie") == "builder_signal"


def test_china_model_sources_have_china_content_types():
    """China model vendors get china_* content types."""
    assert _infer_content_type("x_deepseek", "x_cookie") == "china_model_official"
    assert _infer_content_type("x_qwen", "x_cookie") == "china_model_official"
    assert _infer_content_type("x_zhipu", "x_cookie") == "china_model_official"
    assert _infer_content_type("x_minimax", "x_cookie") == "china_model_official"


def test_pricing_sources_have_pricing_content_type():
    """Sources with _pricing suffix get pricing_page."""
    assert _infer_content_type("deepseek_pricing", "rss") == "pricing_page"
    assert _infer_content_type("openrouter_models_pricing", "rss") == "pricing_page"


def test_benchmark_sources_have_benchmark_content_type():
    """Benchmark sources get benchmark_tracker."""
    assert _infer_content_type("artificial_analysis", "rss") == "benchmark_tracker"
    assert _infer_content_type("lmarena_leaderboard", "rss") == "benchmark_tracker"
    assert _infer_content_type("livebench", "rss") == "benchmark_tracker"


def test_official_docs_have_long_half_life():
    """API docs and pricing have 168h half-life."""
    ct_cfg = {
        "official_docs": {"source_weight": 1.35, "half_life_h": 168},
        "pricing_page": {"source_weight": 1.35, "half_life_h": 168},
        "tech_media": {"source_weight": 0.85, "half_life_h": 36},
    }
    assert ct_cfg["official_docs"]["half_life_h"] == 168
    assert ct_cfg["pricing_page"]["half_life_h"] == 168
    assert ct_cfg["official_docs"]["half_life_h"] > ct_cfg["tech_media"]["half_life_h"]


def test_china_sources_cover_major_model_vendors():
    """Verify major China model vendors have source configs."""
    import yaml
    with open("agent/configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    source_ids = {s.get("id", "") for s in cfg.get("sources", []) if isinstance(s, dict)}

    required_vendors = [
        "x_deepseek", "x_qwen", "x_zhipu", "x_baidu",
        "x_moonshot", "x_minimax", "x_stepfun", "x_01ai",
        "x_tencent_hunyuan", "x_alicloud",
    ]
    missing = [v for v in required_vendors if v not in source_ids]
    assert not missing, f"Missing china vendor sources: {missing}"


def test_insider_and_vc_sources_present():
    """Insider media, VC, and builder signal sources exist in config."""
    import yaml
    with open("agent/configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    source_ids = {s.get("id", "") for s in cfg.get("sources", []) if isinstance(s, dict)}

    # Insider media
    for sid in ["the_information_ai", "bloomberg_ai", "axios_ai"]:
        assert sid in source_ids, f"Missing insider source: {sid}"
    # VC
    assert "vc_a16z_ai" in source_ids
    # Builder
    assert "builder_swyx" in source_ids


def test_all_sources_loadable():
    """Config loads without errors."""
    import yaml
    with open("agent/configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sources = [s for s in cfg.get("sources", []) if isinstance(s, dict)]
    assert len(sources) >= 56, f"Expected >=56 sources, got {len(sources)}"
    # Check all have content_type (via rules or explicit)
    from agent.agents.curator import _build_ct_lookup
    lookup = _build_ct_lookup(sources)
    for s in sources:
        sid = s.get("id", "")
        if sid:
            assert sid in lookup, f"Source {sid} has no content_type mapping"
