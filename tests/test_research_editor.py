"""Tests for Research Editor curation pipeline."""

import json
import os
import pytest
from agent.agents.event_clusterer import cluster_items, EventCluster, _extract_model_anchors
from agent.agents.event_scorer import score_events
from agent.agents.research_editor import (
    ResearchEditorOutput, EditorialDecision, SourceUse,
    _parse_and_validate,
)
from agent.agents.final_selector import select_final_items
from agent.agents.final_selector import _story_key
from agent.agents import history_checker
from agent.agents.history_checker import _extract_issue_history_entries, load_recent_history
from agent.agents.section_classifier import guess_section
from agent.sources.base import RawItem


def make_item(sid, url, title, summary="", stype="rss"):
    return RawItem(source_id=sid, source_type=stype, title=title,
                   url=url, summary=summary, published_at="2026-05-11T00:00:00Z")


# Event Clustering

def test_cluster_same_title():
    items = [make_item("a", "https://a.com/1", "GPT-5.5 Released"),
             make_item("b", "https://b.com/1", "GPT-5.5 Released")]
    clusters = cluster_items(items)
    assert len(clusters) == 1
    assert clusters[0].source_count == 2


def test_cluster_different_events():
    items = [make_item("a", "https://a.com/1", "GPT-5.5 Released"),
             make_item("b", "https://b.com/2", "Claude 4 Released")]
    clusters = cluster_items(items)
    assert len(clusters) == 2


def test_cluster_similar_titles():
    items = [make_item("a", "https://a.com/1", "OpenAI releases GPT-5.5"),
             make_item("b", "https://b.com/2", "OpenAI Releases GPT-5.5!")]
    clusters = cluster_items(items)
    assert len(clusters) == 1


def test_cluster_strips_tracking():
    items = [make_item("a", "https://x.com/page?utm_source=tw", "News"),
             make_item("b", "https://x.com/page", "News")]
    clusters = cluster_items(items)
    assert len(clusters) == 1


def test_cluster_official_wins_primary():
    items = [make_item("x_kol", "https://x.com/post", "GPT-5 Released by OpenAI", stype="x"),
             make_item("openai_news", "https://openai.com/index/gpt-5", "GPT-5 Released by OpenAI", stype="rss")]
    clusters = cluster_items(items)
    assert len(clusters) == 1
    assert "openai_news" in clusters[0].source_names


def test_cluster_merges_same_model_anchor_despite_different_context():
    """Model announcement + platform availability link should merge."""
    items = [
        make_item("anthropic_news", "https://www.anthropic.com/news/claude-opus-4-8",
                  "Introducing Claude Opus 4.8", stype="sitemap"),
        make_item("github_copilot_changelog",
                  "https://github.blog/changelog/2026-05-28-claude-opus-4-8",
                  "Claude Opus 4.8 is generally available for GitHub Copilot",
                  stype="rss"),
    ]
    clusters = cluster_items(items)
    assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"
    assert clusters[0].source_count == 2
    # Anthropic official source should be canonical
    assert "anthropic_news" in clusters[0].source_names
    assert "github_copilot_changelog" in clusters[0].source_names


def test_cluster_keeps_different_model_versions_separate():
    """Gemini 2.5 Flash vs Gemini 3.5 Flash should NOT merge."""
    items = [
        make_item("google_blog", "https://blog.google/gemini-2-5-flash",
                  "Introducing Gemini 2.5 Flash", stype="rss"),
        make_item("google_blog", "https://blog.google/gemini-3-5-flash",
                  "Introducing Gemini 3.5 Flash", stype="rss"),
    ]
    clusters = cluster_items(items)
    assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"


# Rule Scoring

def test_scoring_official_higher():
    official = EventCluster(
        event_id="evt_1", canonical_title="GPT-5.5 Release",
        primary_url="https://openai.com/gpt5", source_urls=["https://openai.com/gpt5"],
        source_names=["openai_news"], source_types=["rss"], source_count=1,
        summary="OpenAI releases GPT-5.5.",
    )
    kol = EventCluster(
        event_id="evt_2", canonical_title="some speculation",
        primary_url="https://x.com/r/post", source_urls=["https://x.com/r/post"],
        source_names=["random_kol"], source_types=["x"], source_count=1,
        summary="speculation.",
    )
    scored = score_events([official, kol])
    assert scored[0].event_id == "evt_1"


def test_scoring_ai_irrelevant_lower():
    good = EventCluster(
        event_id="evt_a", canonical_title="New AI model released",
        primary_url="https://x.com/y", source_urls=["https://x.com/y"],
        source_names=["src"], source_types=["x"], source_count=1,
        summary="A new large language model.",
    )
    bad = EventCluster(
        event_id="evt_b", canonical_title="New phone charger",
        primary_url="https://x.com/z", source_urls=["https://x.com/z"],
        source_names=["src2"], source_types=["x"], source_count=1,
        summary="USB-C charger.",
    )
    scored = score_events([bad, good])
    assert scored[0].event_id == "evt_a"


# Research Editor Schema

def test_valid_editor_output():
    output = ResearchEditorOutput.model_validate({
        "selected": [{"event_id": "evt_x", "decision": "select",
                      "priority": "high", "section": "模型发布",
                      "evidence_level": "official", "novelty": "new_event",
                      "reader_utility": "high", "why_it_matters": "x",
                      "writing_angle": "x", "risk_level": "low",
                      "sources_to_use": [{"url": "https://o.com", "role": "primary"}]}],
        "rejected": [],
    })
    assert len(output.selected) == 1


def test_editor_parse_tolerates_rejected_items_without_decision():
    event = EventCluster(
        event_id="evt_1", canonical_title="DeepSeek pricing",
        primary_url="https://api-docs.deepseek.com/pricing",
        source_urls=["https://api-docs.deepseek.com/pricing"],
        source_names=["deepseek_pricing"], source_types=["pricing_snapshot"],
        source_count=1, summary="DeepSeek pricing changed.",
    )
    raw = json.dumps({
        "selected": [{
            "event_id": "evt_1", "decision": "select",
            "priority": "must_include", "section": "要闻",
            "evidence_level": "official", "novelty": "new_event",
            "reader_utility": "high", "why_it_matters": "pricing matters",
            "writing_angle": "explain API cost impact", "risk_level": "low",
            "sources_to_use": ["https://api-docs.deepseek.com/pricing"],
        }],
        "rejected": [{
            "event_id": "evt_1",
            "reject_reason": "duplicate candidate",
        }],
    }, ensure_ascii=False)

    output = _parse_and_validate(raw, [event])

    assert len(output.selected) == 1
    assert output.selected[0].sources_to_use[0].url == event.primary_url
    assert len(output.rejected) == 1
    assert output.rejected[0].decision == "reject"


def test_fake_event_id_filtered():
    event = EventCluster(
        event_id="evt_real", canonical_title="Real",
        primary_url="https://real.com", source_urls=["https://real.com"],
        source_names=["src"], source_types=["rss"], source_count=1,
        summary="Real.",
    )
    raw = json.dumps({"selected": [{"event_id": "evt_fake", "decision": "select",
                       "priority": "high", "section": "要闻",
                       "evidence_level": "primary", "novelty": "new_event",
                       "reader_utility": "high", "why_it_matters": "x",
                       "writing_angle": "x", "risk_level": "low",
                       "sources_to_use": []}], "rejected": []})
    output = _parse_and_validate(raw, [event])
    assert len(output.selected) == 0


def test_fake_url_auto_fixed():
    event = EventCluster(
        event_id="evt_1", canonical_title="News",
        primary_url="https://official.com", source_urls=["https://official.com"],
        source_names=["src"], source_types=["rss"], source_count=1,
        summary="News.",
    )
    raw = json.dumps({"selected": [{"event_id": "evt_1", "decision": "select",
                       "priority": "high", "section": "要闻",
                       "evidence_level": "primary", "novelty": "new_event",
                       "reader_utility": "high", "why_it_matters": "x",
                       "writing_angle": "x", "risk_level": "low",
                       "sources_to_use": [{"url": "https://fake.com", "role": "primary"}]}],
                       "rejected": []})
    output = _parse_and_validate(raw, [event])
    assert len(output.selected) == 1
    assert output.selected[0].sources_to_use[0].url == "https://official.com"


def test_section_validator_moves_obvious_product_out_of_model_frontier():
    event = EventCluster(
        event_id="evt_product",
        canonical_title="AdventHealth advances whole-person care with OpenAI",
        primary_url="https://openai.com/index/adventhealth",
        source_urls=["https://openai.com/index/adventhealth"],
        source_names=["openai_news"], source_types=["rss"], source_count=1,
        summary="AdventHealth is using ChatGPT for Healthcare to streamline workflows.",
    )
    raw = json.dumps({"selected": [{
        "event_id": "evt_product", "decision": "select",
        "priority": "high", "section": "模型发布",
        "evidence_level": "primary", "novelty": "new_event",
        "reader_utility": "high", "why_it_matters": "x",
        "writing_angle": "x", "risk_level": "low",
        "sources_to_use": [{"url": "https://openai.com/index/adventhealth", "role": "primary"}],
    }], "rejected": []})
    output = _parse_and_validate(raw, [event])
    assert output.selected[0].section == "产品应用"


def test_section_validator_moves_model_release_out_of_product():
    event = EventCluster(
        event_id="evt_qwen",
        canonical_title="Meet Qwen3.7-Max — our latest flagship, made for the Agent Era",
        primary_url="https://x.com/Alibaba_Qwen/status/2057450220708147250",
        source_urls=["https://x.com/Alibaba_Qwen/status/2057450220708147250"],
        source_names=["x_qwen"], source_types=["x_cookie"], source_count=1,
        summary="The new Qwen3.7-Max is live, with big jumps in coding and agent benchmarks over Qwen3.6.",
    )
    raw = json.dumps({"selected": [{
        "event_id": "evt_qwen", "decision": "select",
        "priority": "high", "section": "产品应用",
        "evidence_level": "official", "novelty": "new_event",
        "reader_utility": "high", "why_it_matters": "x",
        "writing_angle": "x", "risk_level": "low",
        "sources_to_use": [{"url": event.primary_url, "role": "primary"}],
    }], "rejected": []})
    output = _parse_and_validate(raw, [event])
    assert output.selected[0].section == "模型发布"


def test_section_validator_moves_open_source_model_out_of_tools():
    event = EventCluster(
        event_id="evt_hymt",
        canonical_title="腾讯开源 Hy-MT2 多语言翻译模型，1.8B 版本超越商业 API",
        primary_url="https://x.com/TencentHunyuan/status/2057384034544804136",
        source_urls=["https://x.com/TencentHunyuan/status/2057384034544804136"],
        source_names=["x_tencent_hunyuan"], source_types=["x_cookie"], source_count=1,
        summary="7B 与 30B-A3B 版本达到开源模型先进翻译性能，1.8B 版本可在手机端运行。",
    )
    raw = json.dumps({"selected": [{
        "event_id": "evt_hymt", "decision": "select",
        "priority": "high", "section": "开发生态",
        "evidence_level": "official", "novelty": "new_event",
        "reader_utility": "high", "why_it_matters": "x",
        "writing_angle": "x", "risk_level": "low",
        "sources_to_use": [{"url": event.primary_url, "role": "primary"}],
    }], "rejected": []}, ensure_ascii=False)
    output = _parse_and_validate(raw, [event])
    assert output.selected[0].section == "模型发布"


def test_section_validator_keeps_model_powered_agent_tool_out_of_model_frontier():
    event = EventCluster(
        event_id="evt_datasette",
        canonical_title="Datasette Agent",
        primary_url="https://simonwillison.net/2026/May/21/datasette-agent",
        source_urls=["https://simonwillison.net/2026/May/21/datasette-agent"],
        source_names=["aihot:Simon Willison 博客"], source_types=["aihot"], source_count=1,
        primary_source_tier="tier_3_pulse_noise",
        summary=(
            "Datasette Agent 是 Datasette 推出的首个可扩展 AI 助手，"
            "支持通过插件生成图表，也可运行于 Gemini 3.1 Flash-Lite 等云端模型。"
        ),
    )
    raw = json.dumps({"selected": [{
        "event_id": "evt_datasette", "decision": "select",
        "priority": "high", "section": "模型发布",
        "evidence_level": "primary", "novelty": "new_event",
        "reader_utility": "high", "why_it_matters": "x",
        "writing_angle": "x", "risk_level": "low",
        "sources_to_use": [{"url": event.primary_url, "role": "primary"}],
    }], "rejected": []}, ensure_ascii=False)
    output = _parse_and_validate(raw, [event])
    assert output.selected[0].section == "行业动态"


def test_smart_home_ai_suite_is_product_application():
    event = EventCluster(
        event_id="evt_home",
        canonical_title="Empowering Service Providers and Hardware Partners with Gemini for Home",
        primary_url="https://developers.googleblog.com/gemini-for-home",
        source_urls=["https://developers.googleblog.com/gemini-for-home"],
        source_names=["google_developers_blog"],
        source_types=["rss"],
        source_count=1,
        summary=(
            "Google 推出 Gemini for Home 全栈 AI 方案，面向服务商和硬件伙伴，"
            "提供摄像头智能、自然语言查询和交钥匙参考设计。"
        ),
    )

    assert guess_section(event) == "产品应用"


def test_google_io_rollup_and_antigravity_share_story_key():
    rollup = EventCluster(
        event_id="evt_io",
        canonical_title="All the news from the Google I/O 2026 Developer keynote",
        primary_url="https://developers.googleblog.com/io",
        source_urls=["https://developers.googleblog.com/io"],
        source_names=["google_developers_blog"],
        source_types=["rss"],
        source_count=1,
        summary="Google I/O 2026 announced Gemini and Antigravity.",
    )
    antigravity = EventCluster(
        event_id="evt_cli",
        canonical_title="An important update: Transitioning Gemini CLI to Antigravity CLI",
        primary_url="https://developers.googleblog.com/antigravity",
        source_urls=["https://developers.googleblog.com/antigravity"],
        source_names=["google_developers_blog"],
        source_types=["rss"],
        source_count=1,
        summary="Gemini CLI is transitioning to Antigravity CLI.",
    )

    assert _story_key(rollup) == _story_key(antigravity) == "google_io_2026"


def test_google_ai_edge_updates_share_story_key():
    litert = EventCluster(
        event_id="evt_litert",
        canonical_title="LiteRT-LM brings Gemma 4 to fast on-device GenAI",
        primary_url="https://developers.googleblog.com/litert-lm",
        source_urls=["https://developers.googleblog.com/litert-lm"],
        source_names=["google_developers_blog"],
        source_types=["rss"],
        source_count=1,
        summary="Google AI Edge发布 LiteRT-LM，为 Gemma 4 提供端侧推理。",
    )
    gallery = EventCluster(
        event_id="evt_gallery",
        canonical_title="A smarter Google AI Edge Gallery with MCP integration",
        primary_url="https://developers.googleblog.com/ai-edge-gallery",
        source_urls=["https://developers.googleblog.com/ai-edge-gallery"],
        source_names=["google_developers_blog"],
        source_types=["rss"],
        source_count=1,
        summary="Google AI Edge Gallery 引入 MCP 集成。",
    )

    assert _story_key(litert) == _story_key(gallery) == "google_ai_edge"


def test_history_checker_extracts_item_titles_and_urls_from_issue_body():
    issue = {
        "title": "AI 日报 2026-05-24",
        "body": """
### 要闻
- [#1 DeepSeek-V4-Pro API 永久降价](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)（deepseek_pricing）

## [Google I/O 2026 发布 Gemini 3.5 与 Antigravity 2.0](https://developers.googleblog.com/io) #2

相关链接：
- [原文](https://developers.googleblog.com/io)
""",
    }

    entries = _extract_issue_history_entries(issue)

    assert "DeepSeek-V4-Pro API 永久降价" in entries
    assert "https://api-docs.deepseek.com/zh-cn/quick_start/pricing" in entries
    assert "Google I/O 2026 发布 Gemini 3.5 与 Antigravity 2.0" in entries


def test_history_loader_filters_stale_local_artifacts_and_merges_github(tmp_path, monkeypatch):
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "2026-05-10.json").write_text(
        json.dumps({
            "date": "2026-05-10",
            "title": "AI 日报 | 2026-05-10",
            "sections": [{"items": [
                {"title": "旧样例条目", "url": "https://example.com/old"}
            ]}],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (drafts / "2026-05-23.json").write_text(
        json.dumps({
            "date": "2026-05-23",
            "title": "AI 日报 | 2026-05-23",
            "sections": [{"items": [
                {"title": "昨日新条目", "url": "https://example.com/local"}
            ]}],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    def fake_fetch(repo, headers, window_days, *, labels="agent-generated",
                   exclude_date="", reference_date=None):
        assert reference_date.isoformat() == "2026-05-24"
        return [{
            "title": "AI 日报 2026-05-22",
            "body": "- [#1 GitHub 正文条目](https://example.com/github)",
        }]

    monkeypatch.setattr(history_checker, "_fetch_recent_issues", fake_fetch)

    entries, meta = load_recent_history(
        artifacts_dir=str(tmp_path),
        window_days=7,
        repo="owner/repo",
        token="token",
        exclude_date="2026-05-24",
    )

    assert "昨日新条目" in entries
    assert "https://example.com/local" in entries
    assert "GitHub 正文条目" in entries
    assert "旧样例条目" not in entries
    assert meta["history_source"] == "local+github"
    assert meta["history_entry_count"] == len(entries)


def test_history_overlap_marks_repeated_event_and_demotes_score():
    event = EventCluster(
        event_id="evt_deepseek",
        canonical_title="DeepSeek-V4-Pro API 2.5 折优惠转为永久正式定价",
        primary_url="https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
        source_urls=["https://api-docs.deepseek.com/zh-cn/quick_start/pricing"],
        source_names=["deepseek_pricing"],
        source_types=["pricing_snapshot"],
        source_count=1,
        primary_source_tier="tier_0_core_evidence",
        primary_evidence_type="pricing_page",
        summary="官方定价页显示优惠转为永久正式定价。",
    )

    scored = score_events(
        [event],
        history_titles=[
            "DeepSeek-V4-Pro API 永久降价",
            "https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
        ],
        max_items=1,
    )

    assert scored[0].already_reported is True
    assert scored[0].rule_score <= 0.24


def test_selector_skips_already_reported_without_material_update():
    old = EventCluster(
        event_id="evt_old",
        canonical_title="Google I/O 2026 发布 Gemini 3.5 与 Antigravity 2.0",
        primary_url="https://developers.googleblog.com/io",
        source_urls=["https://developers.googleblog.com/io"],
        source_names=["google_developers_blog"],
        source_types=["rss"],
        source_count=1,
        summary="Google I/O 2026 发布 Gemini 3.5。",
        rule_score=0.95,
        already_reported=True,
    )
    fresh = EventCluster(
        event_id="evt_fresh",
        canonical_title="GitHub Copilot for Eclipse 以 MIT 许可证开源",
        primary_url="https://github.blog/changelog/eclipse",
        source_urls=["https://github.blog/changelog/eclipse"],
        source_names=["github_copilot_changelog"],
        source_types=["rss"],
        source_count=1,
        summary="GitHub 发布 Copilot for Eclipse 开源消息。",
        rule_score=0.7,
    )
    output = ResearchEditorOutput(selected=[
        EditorialDecision(event_id="evt_old", decision="select", priority="high",
                          section="要闻", sources_to_use=[SourceUse(url=old.primary_url)]),
        EditorialDecision(event_id="evt_fresh", decision="select", priority="high",
                          section="开发生态", sources_to_use=[SourceUse(url=fresh.primary_url)]),
    ])

    items, _recs, _meta = select_final_items(
        editor_output=output, events=[old, fresh], min_items=1, max_items=3, min_papers=0)

    assert [item.url for item in items] == [fresh.primary_url]


def test_selector_caps_same_source_to_reduce_vendor_bundles():
    events = []
    decisions = []
    for i in range(3):
        event = EventCluster(
            event_id=f"evt_google_{i}",
            canonical_title=f"Google Developer update {i}",
            primary_url=f"https://developers.googleblog.com/update-{i}",
            source_urls=[f"https://developers.googleblog.com/update-{i}"],
            source_names=["google_developers_blog"],
            source_types=["rss"],
            source_count=1,
            summary="Google 发布开发者更新。",
            rule_score=0.9 - i * 0.01,
        )
        events.append(event)
        decisions.append(EditorialDecision(
            event_id=event.event_id, decision="select", priority="high",
            section="开发生态", sources_to_use=[SourceUse(url=event.primary_url)],
        ))
    other = EventCluster(
        event_id="evt_other",
        canonical_title="OpenAI 发布企业案例",
        primary_url="https://openai.com/index/case",
        source_urls=["https://openai.com/index/case"],
        source_names=["openai_news"],
        source_types=["rss"],
        source_count=1,
        summary="OpenAI 发布企业案例。",
        rule_score=0.7,
    )
    events.append(other)
    decisions.append(EditorialDecision(
        event_id="evt_other", decision="select", priority="high",
        section="产品应用", sources_to_use=[SourceUse(url=other.primary_url)],
    ))

    items, _recs, _meta = select_final_items(
        editor_output=ResearchEditorOutput(selected=decisions),
        events=events,
        min_items=1,
        max_items=4,
        min_papers=0,
    )

    google_items = [item for item in items if item.source == "google_developers_blog"]
    assert len(google_items) == 2


def test_official_x_model_launch_is_model_release_not_rumor():
    event = EventCluster(
        event_id="evt_stepaudio",
        canonical_title="StepAudio 2.5 Realtime 发布：副语言感知与人格化交互",
        primary_url="https://x.com/StepFun_ai/status/2058303294544425197",
        source_urls=["https://x.com/StepFun_ai/status/2058303294544425197"],
        source_names=["x_stepfun"],
        source_types=["x_cookie"],
        source_count=1,
        primary_source_tier="tier_0_core_evidence",
        primary_content_type="china_model_official",
        primary_evidence_type="official_social",
        primary_confidence="high",
        summary="阶跃星辰官方 X 发布 StepAudio 2.5 Realtime 实时语音模型。",
    )

    assert guess_section(event) == "模型发布"


def test_final_selector_normalizes_model_release_sections():
    event = EventCluster(
        event_id="evt_qwen",
        canonical_title="Meet Qwen3.7-Max — our latest flagship, made for the Agent Era",
        primary_url="https://x.com/Alibaba_Qwen/status/2057450220708147250",
        source_urls=["https://x.com/Alibaba_Qwen/status/2057450220708147250"],
        source_names=["x_qwen"], source_types=["x_cookie"], source_count=1,
        summary="The new Qwen3.7-Max is live with coding and agent benchmark gains.",
        rule_score=0.9,
    )
    output = ResearchEditorOutput(selected=[
        EditorialDecision(
            event_id="evt_qwen", decision="select", priority="high",
            section="产品应用", evidence_level="official", novelty="new_event",
            reader_utility="high", why_it_matters="x", writing_angle="x",
            risk_level="low", sources_to_use=[SourceUse(url=event.primary_url)],
        )
    ], rejected=[])
    items, recs, meta = select_final_items(
        editor_output=output, events=[event], min_items=1, max_items=3, min_papers=0)
    assert items[0].section == "模型发布"
    assert meta["section_normalized_count"] == 1


def test_official_model_release_beats_platform_access_and_merges_link():
    official = EventCluster(
        event_id="evt_claude_release",
        canonical_title="Claude Opus 4.8 发布：编码与 Agent 能力全面升级",
        primary_url="https://www.anthropic.com/news/claude-opus-4-8",
        source_urls=["https://www.anthropic.com/news/claude-opus-4-8"],
        source_names=["anthropic_news"],
        source_types=["rss"],
        source_count=1,
        primary_source_tier="tier_0_core_evidence",
        primary_evidence_type="official_release",
        primary_confidence="high",
        summary="Anthropic 发布 Claude Opus 4.8 模型，编码、智能体和推理能力提升。",
        rule_score=0.8,
    )
    platform = EventCluster(
        event_id="evt_claude_copilot",
        canonical_title="Claude Opus 4.8 is generally available for GitHub Copilot",
        primary_url="https://github.blog/changelog/2026-05-28-claude-opus-4-8-is-generally-available-for-github-copilot",
        source_urls=[
            "https://github.blog/changelog/2026-05-28-claude-opus-4-8-is-generally-available-for-github-copilot"
        ],
        source_names=["github_copilot_changelog"],
        source_types=["rss"],
        source_count=1,
        primary_source_tier="tier_0_core_evidence",
        primary_evidence_type="official_release",
        primary_confidence="high",
        summary="GitHub Copilot 集成 Claude Opus 4.8。",
        rule_score=0.9,
    )
    output = ResearchEditorOutput(selected=[
        EditorialDecision(
            event_id="evt_claude_copilot", decision="select",
            priority="must_include", section="开发生态",
            evidence_level="official", novelty="new_event",
            reader_utility="high", why_it_matters="platform",
            writing_angle="platform", risk_level="low",
            sources_to_use=[SourceUse(url=platform.primary_url)],
        )
    ], rejected=[])

    items, _recs, meta = select_final_items(
        editor_output=output,
        events=[platform, official],
        min_items=1,
        max_items=3,
        min_papers=0,
    )

    assert len(items) == 1
    assert items[0].url == official.primary_url
    assert items[0].section == "模型发布"
    assert platform.primary_url in items[0].supporting_urls
    assert meta["official_model_release_promoted_count"] >= 1


def test_final_selector_normalizes_agent_tool_sections():
    event = EventCluster(
        event_id="evt_datasette",
        canonical_title="Datasette Agent",
        primary_url="https://simonwillison.net/2026/May/21/datasette-agent",
        source_urls=["https://simonwillison.net/2026/May/21/datasette-agent"],
        source_names=["aihot:Simon Willison 博客"], source_types=["aihot"], source_count=1,
        primary_source_tier="tier_3_pulse_noise",
        summary="A conversational AI assistant for querying data, generating charts through plugins, and using Gemini models.",
        rule_score=0.6,
    )
    output = ResearchEditorOutput(selected=[
        EditorialDecision(
            event_id="evt_datasette", decision="select", priority="high",
            section="模型发布", evidence_level="primary", novelty="new_event",
            reader_utility="high", why_it_matters="x", writing_angle="x",
            risk_level="low", sources_to_use=[SourceUse(url=event.primary_url)],
        )
    ], rejected=[])
    items, recs, meta = select_final_items(
        editor_output=output, events=[event], min_items=1, max_items=3, min_papers=0)
    assert items[0].section == "行业动态"
    assert meta["section_normalized_count"] == 1


def test_tier3_financial_story_is_not_automatically_rumor():
    event = EventCluster(
        event_id="evt_nvidia",
        canonical_title="黄仁勋：AI 基建年度开支要冲到 4 万亿美元",
        primary_url="https://www.ithome.com/0/954/223.htm",
        source_urls=["https://www.ithome.com/0/954/223.htm"],
        source_names=["ithome"],
        source_types=["rss"],
        source_count=1,
        primary_source_tier="tier_3_pulse_noise",
        summary="报道称英伟达 Q1 营收 816 亿美元，黄仁勋预测 AI 基建年度开支将达 4 万亿美元。",
        rule_score=0.8,
    )
    output = ResearchEditorOutput(selected=[
        EditorialDecision(
            event_id="evt_nvidia", decision="select", priority="high",
            section="前瞻与传闻", evidence_level="trusted_media",
            novelty="new_event", reader_utility="high",
            why_it_matters="x", writing_angle="x", risk_level="medium",
            sources_to_use=[SourceUse(url=event.primary_url)],
        )
    ], rejected=[])
    items, _recs, _meta = select_final_items(
        editor_output=output, events=[event], min_items=1, max_items=3, min_papers=0)
    assert items[0].section == "行业动态"


def test_supporting_links_drop_bare_social_profiles():
    event = EventCluster(
        event_id="evt_databricks",
        canonical_title="Databricks brings GPT-5.5 to enterprise agent workflows",
        primary_url="https://openai.com/index/databricks",
        source_urls=[
            "https://openai.com/index/databricks",
            "https://x.com/Alibaba_Qwen",
        ],
        source_names=["openai_news", "x_qwen"],
        source_types=["rss", "x_cookie"],
        source_count=2,
        summary="Databricks integrates GPT-5.5 for enterprise agent workflows.",
        rule_score=0.8,
    )
    output = ResearchEditorOutput(selected=[
        EditorialDecision(
            event_id="evt_databricks", decision="select", priority="high",
            section="技术与洞察", evidence_level="official",
            novelty="new_event", reader_utility="high",
            why_it_matters="x", writing_angle="x", risk_level="low",
            sources_to_use=[SourceUse(url=event.primary_url)],
        )
    ], rejected=[])
    items, _recs, _meta = select_final_items(
        editor_output=output, events=[event], min_items=1, max_items=3, min_papers=0)
    assert "https://x.com/Alibaba_Qwen" not in items[0].supporting_urls


def test_official_pricing_change_promotes_to_headline():
    event = EventCluster(
        event_id="evt_deepseek_pricing",
        canonical_title="DeepSeek-V4-Pro API 2.5 折优惠转为永久正式定价",
        primary_url="https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
        source_urls=["https://api-docs.deepseek.com/zh-cn/quick_start/pricing"],
        source_names=["deepseek_pricing"],
        source_types=["pricing_snapshot"],
        source_count=1,
        primary_content_type="china_model_pricing",
        primary_evidence_type="pricing_page",
        primary_source_tier="tier_0_core_evidence",
        summary="官方定价页显示优惠将在 5 月 31 日结束后转为永久正式定价。",
        rule_score=0.8,
    )
    output = ResearchEditorOutput(selected=[
        EditorialDecision(
            event_id="evt_deepseek_pricing", decision="select", priority="high",
            section="开发生态", evidence_level="official",
            novelty="new_event", reader_utility="high",
            why_it_matters="x", writing_angle="x", risk_level="low",
            sources_to_use=[SourceUse(url=event.primary_url)],
        )
    ], rejected=[])
    items, _recs, meta = select_final_items(
        editor_output=output, events=[event], min_items=1, max_items=3, min_papers=0)
    assert items[0].section == "要闻"
    assert meta["section_normalized_count"] == 1


def test_tier3_low_confidence_item_cannot_stay_headline():
    event = EventCluster(
        event_id="evt_nvidia",
        canonical_title="黄仁勋：AI 基建年度开支要冲到 4 万亿美元",
        primary_url="https://www.ithome.com/0/954/223.htm",
        source_urls=["https://www.ithome.com/0/954/223.htm"],
        source_names=["ithome"],
        source_types=["rss"],
        source_count=1,
        primary_source_tier="tier_3_pulse_noise",
        primary_confidence="low",
        summary="报道称英伟达财报和 AI 基建开支预测。",
        rule_score=0.8,
    )
    output = ResearchEditorOutput(selected=[
        EditorialDecision(
            event_id="evt_nvidia", decision="select", priority="must_include",
            section="要闻", evidence_level="trusted_media",
            novelty="new_event", reader_utility="high",
            why_it_matters="x", writing_angle="x", risk_level="medium",
            sources_to_use=[SourceUse(url=event.primary_url)],
        )
    ], rejected=[])
    items, _recs, _meta = select_final_items(
        editor_output=output, events=[event], min_items=1, max_items=3, min_papers=0)
    assert items[0].section == "行业动态"


# Final Selector

def test_fallback_when_llm_empty():
    events = [EventCluster(event_id=f"evt_{i}", canonical_title=f"E{i}",
                primary_url=f"https://x.com/{i}", source_urls=[f"https://x.com/{i}"],
                source_names=[f"s{i}"], source_types=["rss"], source_count=1,
                summary=f"S{i}", rule_score=0.8) for i in range(30)]
    output = ResearchEditorOutput(selected=[], rejected=[])
    items, recs, meta = select_final_items(
        editor_output=output, events=events, min_items=16, max_items=22)
    assert meta["fallback_used"]
    assert len(items) >= 16


def test_selector_backfills_missing_sections_from_all_events():
    events = [
        EventCluster(event_id="evt_head", canonical_title="OpenAI launches a new assistant feature",
                     primary_url="https://example.com/head", source_urls=["https://example.com/head"],
                     source_names=["openai"], source_types=["rss"], source_count=1,
                     summary="A product feature launch.", rule_score=0.95),
        EventCluster(event_id="evt_model", canonical_title="Google releases Gemini 4 model for coding agents",
                     primary_url="https://example.com/model", source_urls=["https://example.com/model"],
                     source_names=["google"], source_types=["rss"], source_count=1,
                     summary="Google released a new Gemini model with coding and agent benchmark gains.", rule_score=0.9),
        EventCluster(event_id="evt_tool", canonical_title="New GitHub SDK open source tool",
                     primary_url="https://example.com/tool", source_urls=["https://example.com/tool"],
                     source_names=["github"], source_types=["rss"], source_count=1,
                     summary="A developer tool is open source.", rule_score=0.88),
        EventCluster(event_id="evt_capital", canonical_title="AI startup raises Series B funding",
                     primary_url="https://example.com/capital", source_urls=["https://example.com/capital"],
                     source_names=["media"], source_types=["rss"], source_count=1,
                     summary="Funding round for an AI startup.", rule_score=0.86),
        EventCluster(event_id="evt_policy", canonical_title="AI regulation law gains support",
                     primary_url="https://example.com/policy", source_urls=["https://example.com/policy"],
                     source_names=["media2"], source_types=["rss"], source_count=1,
                     summary="A policy and regulation update.", rule_score=0.84),
    ]
    output = ResearchEditorOutput(selected=[
        EditorialDecision(
            event_id="evt_head", decision="select", priority="high",
            section="要闻", evidence_level="primary", novelty="new_event",
            reader_utility="high", why_it_matters="x", writing_angle="x",
            risk_level="low", sources_to_use=[],
        )
    ], rejected=[])

    items, recs, meta = select_final_items(
        editor_output=output,
        events=events,
        min_items=5,
        max_items=7,
        min_papers=0,
    )

    sections = {item.section for item in items}
    assert len(items) >= 5
    assert {"模型发布", "开发生态", "行业动态"} <= sections


# URL validation — fabricated URLs are dropped with warnings

def test_llm_fake_url_dropped_with_warning():
    event = EventCluster(
        event_id="evt_1", canonical_title="News",
        primary_url="https://official.com/article",
        source_urls=["https://official.com/article"],
        source_names=["src"], source_types=["rss"], source_count=1,
        summary="News.",
    )
    # LLM outputs a fabricated URL not in source_urls.
    raw = json.dumps({
        "selected": [{
            "event_id": "evt_1", "decision": "select",
            "priority": "high", "section": "要闻",
            "evidence_level": "primary", "novelty": "new_event",
            "reader_utility": "high", "why_it_matters": "x",
            "writing_angle": "x", "risk_level": "low",
            "sources_to_use": [
                {"url": "https://fabricated-by-llm.com/fake", "role": "primary"},
            ],
        }],
        "rejected": [],
    })
    output = _parse_and_validate(raw, [event])
    # Should still have the event selected.
    assert len(output.selected) == 1
    # Fake URL should be dropped, fallback to event's primary_url.
    assert output.selected[0].sources_to_use[0].url == "https://official.com/article"
    # Warning should be recorded.
    assert output.notes and "invalid_llm_url_removed" in output.notes


def test_fake_event_id_dropped():
    event = EventCluster(
        event_id="evt_real", canonical_title="Real",
        primary_url="https://real.com", source_urls=["https://real.com"],
        source_names=["src"], source_types=["rss"], source_count=1,
        summary="Real.",
    )
    raw = json.dumps({
        "selected": [{"event_id": "evt_nonexistent", "decision": "select",
                      "priority": "high", "section": "要闻",
                      "evidence_level": "primary", "novelty": "new_event",
                      "reader_utility": "high", "why_it_matters": "x",
                      "writing_angle": "x", "risk_level": "low",
                      "sources_to_use": [{"url": "https://real.com", "role": "primary"}]}],
        "rejected": [],
    })
    output = _parse_and_validate(raw, [event])
    assert len(output.selected) == 0
    assert "invalid_event_id" in (output.notes or "")


# Full pipeline smoke test (mock LLM, no network)



def test_full_pipeline_smoke(cfg, prompts, tmp_path):
    """End-to-end pipeline smoke test with mock LLM, zero network."""
    from agent.pipelines.daily_report import run_pipeline
    from agent.llm.mock_provider import MockLLMProvider
    import re as _re

    # Mock provider: dynamically match event IDs from the user prompt.
    section_cycle = ["要闻", "模型发布", "开发生态", "技术与洞察", "产品应用", "行业动态"]
    call_count = [0]

    def responder(msgs):
        call_count[0] += 1
        system = msgs[0].content if msgs else ""
        user = msgs[-1].content if len(msgs) > 1 else ""

        # Research Editor call.
        if "面向 AI 开发者" in system:
            evt_ids = _re.findall(r"\[(evt_[a-f0-9]+)\]", user)
            selected = []
            for idx, eid in enumerate(evt_ids[:8]):
                selected.append({
                    "event_id": eid, "decision": "select",
                    "priority": "must_include" if idx < 2 else "high",
                    "section": section_cycle[idx % 6],
                    "evidence_level": "primary", "novelty": "new_event",
                    "reader_utility": "high" if idx < 3 else "medium",
                    "why_it_matters": f"Reason for {eid}",
                    "writing_angle": f"Angle for {eid}",
                    "risk_level": "low", "sources_to_use": [],
                })
            return json.dumps({"selected": selected, "rejected": [], "notes": ""}, ensure_ascii=False)

        # Semantic duplicate critic / repairer: return no duplicates.
        if "语义重复" in system or "duplicates" in system or "修复" in system:
            return json.dumps({"duplicates": []})

        # Writer response — use real source URLs to avoid critic hallucinations.
        return json.dumps({
            "date": "2026-05-15", "title": "AI Daily Test", "overview": "Test smoke.",
            "sections": [{"heading": h, "items": [
                {"title": f"#{n+1} test item {h}", "summary": "summary",
                 "url": f"https://example.com/{(n % 5) + 1}", "source": "mock",
                 "highlights": ["a"], "related_links": []}
            ]} for n, h in enumerate(section_cycle)],
        }, ensure_ascii=False)

    provider = MockLLMProvider(model="mock-smoke", responder=responder)

    # Write a local RSS file so the test works without network.
    rss_feed = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Test Feed</title>
  <link>https://example.com</link>
  <item><title>Test AI News 1</title><link>https://example.com/1</link>
    <description>An AI model announcement</description>
    <pubDate>Mon, 19 May 2026 10:00:00 GMT</pubDate></item>
  <item><title>Test AI News 2</title><link>https://example.com/2</link>
    <description>A new framework release</description>
    <pubDate>Mon, 19 May 2026 09:00:00 GMT</pubDate></item>
  <item><title>Test AI News 3</title><link>https://example.com/3</link>
    <description>AI product launch today</description>
    <pubDate>Mon, 19 May 2026 08:00:00 GMT</pubDate></item>
  <item><title>Test AI News 4</title><link>https://example.com/4</link>
    <description>Funding round for AI startup</description>
    <pubDate>Mon, 19 May 2026 07:00:00 GMT</pubDate></item>
  <item><title>Test AI News 5</title><link>https://example.com/5</link>
    <description>Research paper on transformers</description>
    <pubDate>Mon, 19 May 2026 06:00:00 GMT</pubDate></item>
</channel></rss>"""
    rss_path = tmp_path / "test_feed.xml"
    rss_path.write_text(rss_feed, encoding="utf-8")

    cfg2 = dict(cfg)
    cfg2["sources"] = [
        {"id": "test_feed", "type": "rss", "url": str(rss_path), "weight": 1.0, "max_items": 5},
    ]
    cfg2["curation"] = {
        "mode": "research_editor",
        "candidate_top_k": 30,
        "final_min_items": 3,
        "final_max_items": 10,
        "history_window_days": 7,
        "fallback_to_rules": True,
        "legacy_llm_scoring_enabled": False,
        "enable_evidence_fetch": False,
    }

    result = run_pipeline(
        cfg=cfg2, prompts=prompts, provider=provider,
        artifacts_root=str(tmp_path / "artifacts"),
        date="2026-05-15",
    )

    assert not result.get("is_failed"), f"Pipeline failed: {result}"
    # Check if pipeline failed gracefully via needs_human_review.
    if result.get("needs_human_review"):
        # Check stages for more detail.
        stages = result.get("stages", {})
        for name, s in stages.items():
            if s.get("status") not in ("ok", "pending"):
                print(f"  Stage {name}: {s.get('status')} error={s.get('error','')}")
    draft_path = result.get("draft_path")
    assert draft_path, f"No draft_path in result. Stages: {result.get('stages', {})}"
    assert os.path.exists(draft_path), f"Draft not found: {draft_path}"

    with open(draft_path, "r", encoding="utf-8") as f:
        md = f.read()
    assert "AI Daily Test" in md
    assert "要闻" in md

    curated_path = result.get("curated_path")
    assert curated_path and os.path.exists(curated_path), "No curated_path"

    trace_path = result.get("trace_path")
    assert trace_path and os.path.exists(trace_path)

    assert call_count[0] >= 2  # editor + writer


# Pipeline smoke test

def test_pipeline_rules_only(cfg, prompts):
    from agent.pipelines.daily_report import run_pipeline
    from agent.llm.mock_provider import MockLLMProvider

    def responder(msgs):
        return json.dumps({
            "date": "2026-05-11", "title": "AI Daily", "overview": "Test.",
            "sections": [{"heading": h, "items": [
                {"title": f"#{i} t", "summary": "t", "url": "https://x.com",
                 "source": "m", "highlights": ["a","b"], "related_links": []}
            ]} for h in ["要闻","模型发布","开发生态","产品应用","技术与洞察","行业动态"]],
        })
    provider = MockLLMProvider(model="mock", responder=responder)
    cfg2 = dict(cfg)
    cfg2["curation"] = {"mode": "rules_only", "candidate_top_k": 20,
                        "final_min_items": 6, "final_max_items": 12,
                        "fallback_to_rules": True}
    result = run_pipeline(cfg=cfg2, prompts=prompts, provider=provider,
                          artifacts_root="artifacts", date="2026-05-11")
    assert not result.get("is_failed")
