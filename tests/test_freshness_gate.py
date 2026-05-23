import json
from datetime import datetime, timedelta, timezone

from agent.agents.event_clusterer import EventCluster
from agent.agents.event_scorer import score_events
from agent.agents.final_selector import select_final_items
from agent.agents.research_editor import EditorialDecision, ResearchEditorOutput


def _iso(hours_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def test_old_background_story_is_capped_below_recent_news():
    old_background = EventCluster(
        event_id="old",
        canonical_title="What to Know About DeepSeek AI",
        primary_url="https://wsj.com/old-deepseek",
        source_urls=["https://wsj.com/old-deepseek"],
        source_names=["wsj_ai"],
        source_types=["rss"],
        source_count=1,
        summary="A background explainer about DeepSeek and low-cost AI training.",
        published_at=_iso(24 * 120),
        latest_seen_at=_iso(24 * 120),
    )
    recent_news = EventCluster(
        event_id="recent",
        canonical_title="GitHub Copilot for Eclipse is open source",
        primary_url="https://github.blog/changelog/copilot-eclipse",
        source_urls=["https://github.blog/changelog/copilot-eclipse"],
        source_names=["github_copilot_changelog"],
        source_types=["rss"],
        source_count=1,
        summary="GitHub released Copilot for Eclipse as open source today.",
        published_at=_iso(6),
        latest_seen_at=_iso(6),
    )

    scored = score_events([old_background, recent_news], max_items=2)
    by_id = {evt.event_id: evt.rule_score for evt in scored}
    assert by_id["old"] <= 0.12
    assert scored[0].event_id == "recent"


def test_final_selector_drops_stale_background_even_if_editor_selected_it():
    old_background = EventCluster(
        event_id="old",
        canonical_title="What to Know About DeepSeek AI",
        primary_url="https://wsj.com/old-deepseek",
        source_urls=["https://wsj.com/old-deepseek"],
        source_names=["wsj_ai"],
        source_types=["rss"],
        source_count=1,
        summary="A background explainer about DeepSeek and low-cost AI training.",
        published_at=_iso(24 * 120),
        latest_seen_at=_iso(24 * 120),
        rule_score=0.99,
    )
    fresh = [
        EventCluster(
            event_id=f"fresh_{i}",
            canonical_title=f"Fresh AI product update {i}",
            primary_url=f"https://example.com/fresh-{i}",
            source_urls=[f"https://example.com/fresh-{i}"],
            source_names=[f"src_{i}"],
            source_types=["rss"],
            source_count=1,
            summary="A new AI product feature launched today.",
            published_at=_iso(4),
            latest_seen_at=_iso(4),
            rule_score=0.7 - i * 0.01,
        )
        for i in range(4)
    ]
    output = ResearchEditorOutput(selected=[
        EditorialDecision(
            event_id="old",
            decision="select",
            priority="high",
            section="行业动态",
            evidence_level="primary",
            novelty="old_background",
            reader_utility="medium",
            why_it_matters="old",
            writing_angle="old",
            risk_level="low",
            sources_to_use=[],
        ),
        *[
            EditorialDecision(
                event_id=evt.event_id,
                decision="select",
                priority="medium",
                section="产品应用",
                evidence_level="primary",
                novelty="new_event",
                reader_utility="medium",
                why_it_matters="fresh",
                writing_angle="fresh",
                risk_level="low",
                sources_to_use=[],
            )
            for evt in fresh
        ],
    ], rejected=[])

    items, _records, _meta = select_final_items(
        editor_output=output,
        events=[old_background, *fresh],
        min_items=3,
        max_items=5,
        min_papers=0,
    )
    assert all(item.url != "https://wsj.com/old-deepseek" for item in items)
