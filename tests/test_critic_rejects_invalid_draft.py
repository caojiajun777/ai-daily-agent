from agent.agents.critic import deterministic_critique
from agent.schemas import CuratedItem, Draft, DraftItem, DraftSection


def _draft_from(items):
    sections = [DraftSection(heading=f"h{i}", items=[it]) for i, it in enumerate(items)]
    return Draft(date="2026-05-09", title="T", sections=sections)


def _curated(urls):
    return [
        CuratedItem(
            title=f"t{i}",
            url=u,
            summary="s",
            source="src",
            source_type="rss",
            published_at="",
            score=1.0,
        )
        for i, u in enumerate(urls)
    ]


def test_critic_passes_clean_draft():
    items = [
        DraftItem(title="A", summary="sa", url="u1", source="src"),
        DraftItem(title="B", summary="sb", url="u2", source="src"),
        DraftItem(title="C", summary="sc", url="u3", source="src"),
    ]
    draft = _draft_from(items)
    curated = _curated(["u1", "u2", "u3"])
    res = deterministic_critique(draft, curated, min_section_count=3)
    assert res.verdict == "pass"
    assert res.reasons == []


def test_critic_rejects_hallucinated_url():
    items = [
        DraftItem(title="A", summary="sa", url="u1", source="src"),
        DraftItem(title="B", summary="sb", url="u2", source="src"),
        DraftItem(title="C", summary="sc", url="u-fake", source="src"),
    ]
    draft = _draft_from(items)
    curated = _curated(["u1", "u2", "u3"])
    res = deterministic_critique(draft, curated, min_section_count=3)
    assert res.verdict == "reject"
    assert any("hallucinated" in r for r in res.reasons)


def test_critic_rejects_too_few_sections():
    items = [DraftItem(title="A", summary="sa", url="u1", source="src")]
    draft = _draft_from(items)
    curated = _curated(["u1"])
    res = deterministic_critique(draft, curated, min_section_count=3)
    assert res.verdict == "reject"
    assert any("section" in r for r in res.reasons)


def test_critic_rejects_duplicate_titles():
    items = [
        DraftItem(title="Dup", summary="sa", url="u1", source="src"),
        DraftItem(title="Dup", summary="sb", url="u2", source="src"),
        DraftItem(title="C", summary="sc", url="u3", source="src"),
    ]
    draft = _draft_from(items)
    curated = _curated(["u1", "u2", "u3"])
    res = deterministic_critique(draft, curated, min_section_count=3)
    assert res.verdict == "reject"
    assert any("duplicate" in r for r in res.reasons)


def test_critic_rejects_forbidden_phrase():
    items = [
        DraftItem(title="A", summary="作为AI助手不能回答", url="u1", source="src"),
        DraftItem(title="B", summary="sb", url="u2", source="src"),
        DraftItem(title="C", summary="sc", url="u3", source="src"),
    ]
    draft = _draft_from(items)
    curated = _curated(["u1", "u2", "u3"])
    res = deterministic_critique(
        draft, curated, min_section_count=3, forbid_phrases=["作为AI", "I cannot"]
    )
    assert res.verdict == "reject"
    assert any("forbidden phrase" in r for r in res.reasons)
