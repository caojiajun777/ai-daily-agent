from __future__ import annotations

from agent.tools import evidence_fetcher as ef


def test_fetch_evidence_for_events_preserves_event_order(monkeypatch):
    def fake_fetch_one(url: str, timeout: float) -> ef.EvidenceSnippet:
        return ef.EvidenceSnippet(
            url=url,
            title=url.rsplit("/", 1)[-1],
            fetch_status="ok",
            evidence_type="unknown",
        )

    monkeypatch.setattr(ef, "_fetch_one", fake_fetch_one)

    results = ef.fetch_evidence_for_events(
        [
            ["https://example.com/first"],
            ["https://example.com/second"],
        ],
        timeout=0.1,
        max_workers=2,
    )

    assert [group[0].title for group in results] == ["first", "second"]
