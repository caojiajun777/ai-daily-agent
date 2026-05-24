import json
import re

from agent.agents.event_clusterer import EventCluster
from agent.agents.llm_reranker import _extract_json_array, llm_rerank_events
from agent.llm.mock_provider import MockLLMProvider


def test_llm_reranker_scores_second_chunk():
    calls = []

    def responder(messages):
        user = messages[-1].content
        ids = re.findall(r"\[(evt_\d+)\]", user)
        calls.append(ids)
        payload = []
        for eid in ids:
            high = eid == "evt_44"
            payload.append({
                "event_id": eid,
                "newsworthiness_score": 10 if high else 2,
                "freshness_score": 10 if high else 2,
                "novelty_score": 10 if high else 2,
                "audience_breadth_score": 10 if high else 2,
                "social_publishability_score": 10 if high else 2,
                "evidence_strength_score": 10,
                "risk_penalty": 0,
                "confidence_score": 1.0,
                "recommended_slot": "headline" if high else "secondary",
            })
        return json.dumps(payload)

    events = [
        EventCluster(
            event_id=f"evt_{i}",
            canonical_title=f"Event {i}",
            primary_url=f"https://example.com/{i}",
            source_urls=[f"https://example.com/{i}"],
            source_names=["src"],
            source_types=["rss"],
            source_count=1,
            summary="AI event.",
            rule_score=0.1,
        )
        for i in range(45)
    ]
    ranked = llm_rerank_events(
        events=events,
        provider=MockLLMProvider(model="mock", responder=responder),
    )
    assert len(calls) == 2
    assert ranked[0].event_id == "evt_44"


def test_extract_json_array_handles_unclosed_code_fence():
    raw = "```json\n[{\"event_id\":\"evt_1\"}]"
    assert _extract_json_array(raw) == '[{"event_id":"evt_1"}]'


def test_llm_reranker_keeps_already_reported_events_capped():
    seen_prompt = {}

    def responder(messages):
        user = messages[-1].content
        seen_prompt["text"] = user
        return json.dumps([{
            "event_id": "evt_old",
            "newsworthiness_score": 10,
            "freshness_score": 10,
            "novelty_score": 10,
            "audience_breadth_score": 10,
            "social_publishability_score": 10,
            "evidence_strength_score": 10,
            "risk_penalty": 0,
            "confidence_score": 1.0,
            "recommended_slot": "headline",
        }])

    event = EventCluster(
        event_id="evt_old",
        canonical_title="Google I/O 2026 发布 Gemini 3.5 与 Antigravity 2.0",
        primary_url="https://developers.googleblog.com/io",
        source_urls=["https://developers.googleblog.com/io"],
        source_names=["google_developers_blog"],
        source_types=["rss"],
        source_count=1,
        published_at="2026-05-23T02:00:00Z",
        summary="Google I/O 2026 发布 Gemini 3.5。",
        rule_score=0.9,
        already_reported=True,
        primary_evidence_type="official_release",
        primary_confidence="high",
    )

    ranked = llm_rerank_events(
        events=[event],
        provider=MockLLMProvider(model="mock", responder=responder),
    )

    assert "already_reported=true" in seen_prompt["text"]
    assert "published_at: 2026-05-23T02:00:00Z" in seen_prompt["text"]
    assert "source_urls: https://developers.googleblog.com/io" in seen_prompt["text"]
    assert ranked[0].rule_score <= 0.24
