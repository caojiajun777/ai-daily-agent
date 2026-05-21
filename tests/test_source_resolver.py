"""Tests for Source Resolver / Source Doctor v1.

All HTTP is mocked — no real network calls.
"""

import json
import os
import tempfile

import pytest

from agent.tools.source_resolver import (
    SourceResolutionReport,
    SourceResolutionResult,
    _discover_urls,
    _is_valid_feed,
    _is_valid_page,
    apply_safe_resolutions,
    apply_non_enable_fixes,
    resolve_all_disabled,
    resolve_source,
)


# ── Mock HTTP client ────────────────────────────────────────────────────


class MockHTTPResponse:
    def __init__(self, status_code=200, text="", headers=None):
        self.status_code = status_code
        self._text = text
        self.headers = headers or {}

    @property
    def text(self):
        return self._text


class MockHTTPClient:
    """Configurable mock: url -> (status, body, content_type)."""
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def head(self, url):
        self.calls.append(("head", url))
        r = self.responses.get(url, {"status": 404, "body": "", "ct": ""})
        return MockHTTPResponse(r["status"], r.get("body", ""),
                               {"content-type": r.get("ct", "")})

    def get(self, url):
        self.calls.append(("get", url))
        r = self.responses.get(url, {"status": 404, "body": "", "ct": ""})
        return MockHTTPResponse(r["status"], r.get("body", ""),
                               {"content-type": r.get("ct", "")})

    def close(self):
        pass


_RSS_BODY = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test RSS</title>
<item><title>Release v1.0</title><link>https://example.com/1</link></item>
</channel></rss>"""

_ATOM_BODY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Test Feed</title><entry><title>v1.0</title></entry></feed>"""


# ── Helpers ─────────────────────────────────────────────────────────────


def _source(**kw):
    defaults = {"id": "test_source", "type": "rss", "enabled": False,
                "content_type": "tech_media"}
    defaults.update(kw)
    return defaults


# ── URL discovery tests ─────────────────────────────────────────────────


def test_github_release_url_discovery():
    """GitHub release source should discover releases.atom URL."""
    src = _source(id="deepseek_github_releases", content_type="github_release")
    urls = _discover_urls(src)
    assert any("github.com" in u and "releases.atom" in u for u in urls)


def test_pricing_url_discovery():
    """Pricing source should discover known pricing URL."""
    src = _source(id="openai_pricing", content_type="pricing_page")
    urls = _discover_urls(src)
    assert any("openai.com" in u and "pricing" in u for u in urls)


def test_docs_url_discovery():
    """Docs source should discover known docs URLs."""
    src = _source(id="baidu_qianfan_docs", content_type="china_model_docs")
    urls = _discover_urls(src)
    assert len(urls) >= 1


def test_benchmark_url_discovery():
    """Benchmark tracker discovers leaderboard URL."""
    src = _source(id="livebench", content_type="benchmark_tracker")
    urls = _discover_urls(src)
    assert any("livebench.ai" in u for u in urls)


# ── Validation tests ────────────────────────────────────────────────────


def test_is_valid_feed_detects_rss():
    assert _is_valid_feed(200, "application/rss+xml", _RSS_BODY)


def test_is_valid_feed_detects_atom():
    assert _is_valid_feed(200, "application/atom+xml", _ATOM_BODY)


def test_is_valid_feed_rejects_404():
    assert not _is_valid_feed(404, "text/html", "")


def test_is_valid_page_accepts_200():
    assert _is_valid_page(200)


def test_is_valid_page_rejects_500():
    assert not _is_valid_page(500)


# ── Resolution tests ────────────────────────────────────────────────────


def test_resolve_enable_safe_for_valid_github_release():
    http = MockHTTPClient({
        "https://github.com/deepseek-ai/DeepSeek-V3/releases.atom": {
            "status": 200, "body": _ATOM_BODY, "ct": "application/atom+xml",
        },
    })
    src = _source(id="deepseek_github_releases", content_type="github_release",
                  url="https://github.com/deepseek-ai/DeepSeek-V3/releases.atom")
    result = resolve_source(src, http_client=http)
    assert result.status == "resolved_enable_safe"
    assert result.recommended_enabled
    assert result.risk_level == "low"


def test_resolve_enable_safe_for_valid_rss():
    http = MockHTTPClient({
        "https://www.latent.space/feed": {
            "status": 200, "body": _RSS_BODY, "ct": "application/rss+xml",
        },
    })
    src = _source(id="latent_space", content_type="expert_newsletter",
                  url="https://www.latent.space/feed")
    result = resolve_source(src, http_client=http)
    assert result.status == "resolved_enable_safe"


def test_pricing_requires_adapter_not_auto_enabled():
    http = MockHTTPClient({
        "https://openai.com/api/pricing/": {
            "status": 200,
            "body": "<html><body>GPT-5 pricing $1 per 1M tokens</body></html>",
            "ct": "text/html",
        },
    })
    src = _source(id="openai_pricing", content_type="pricing_page")
    result = resolve_source(src, http_client=http)
    assert result.status == "candidate_found_needs_adapter"
    assert not result.recommended_enabled
    assert result.required_adapter == "pricing_snapshot_adapter"


def test_benchmark_requires_adapter_not_auto_enabled():
    http = MockHTTPClient({
        "https://livebench.ai/": {
            "status": 200,
            "body": "<html><body>benchmark ranking elo score</body></html>",
            "ct": "text/html",
        },
    })
    src = _source(id="livebench", content_type="benchmark_tracker")
    result = resolve_source(src, http_client=http)
    assert result.status == "candidate_found_needs_adapter"
    assert not result.recommended_enabled
    assert result.required_adapter == "benchmark_tracker_adapter"


def test_vc_signal_never_auto_enabled():
    src = _source(id="vc_a16z_ai", type="x_cookie", content_type="vc_signal",
                  username="a16z")
    result = resolve_source(src)
    assert not result.recommended_enabled
    assert result.status != "resolved_enable_safe"


def test_builder_signal_never_auto_enabled():
    src = _source(id="builder_swyx", type="x_cookie", content_type="builder_signal",
                  username="swyx")
    result = resolve_source(src)
    assert not result.recommended_enabled
    assert result.status != "resolved_enable_safe"


def test_no_candidate_url_reported():
    src = _source(id="unknown_thing", content_type="tech_media",
                  url="")
    result = resolve_source(src)
    assert result.status == "no_candidate_found"


def test_ambiguous_not_auto_enabled():
    http = MockHTTPClient({
        "https://example.com/feed": {
            "status": 200, "body": "<html>just a page</html>", "ct": "text/html",
        },
    })
    src = _source(id="weird_source", content_type="tech_media",
                  url="https://example.com/feed")
    result = resolve_source(src, http_client=http)
    assert not result.recommended_enabled
    assert result.status in ("ambiguous_candidates", "no_candidate_found")


# ── Batch resolution tests ──────────────────────────────────────────────


def test_resolve_all_disabled_produces_report():
    sources = [
        _source(id="github_src", content_type="github_release",
                url="https://github.com/deepseek-ai/DeepSeek-V3/releases.atom", enabled=False),
        _source(id="openai_pricing", content_type="pricing_page", enabled=False),
        _source(id="vc_src", type="x_cookie", content_type="vc_signal",
                username="a16z", enabled=False),
    ]
    http = MockHTTPClient({
        "https://github.com/deepseek-ai/DeepSeek-V3/releases.atom": {
            "status": 200, "body": _ATOM_BODY, "ct": "application/atom+xml",
        },
        "https://openai.com/api/pricing/": {
            "status": 200, "body": "<html>pricing</html>", "ct": "text/html",
        },
    })
    report = resolve_all_disabled(sources, http_client=http)
    assert report.total_checked == 3
    assert report.resolved_enable_safe >= 1
    assert report.candidate_found_needs_adapter >= 1


# ── Apply safe tests ────────────────────────────────────────────────────


def test_apply_safe_only_enables_low_risk_sources():
    http = MockHTTPClient({
        "https://github.com/deepseek-ai/DeepSeek-V3/releases.atom": {
            "status": 200, "body": _ATOM_BODY, "ct": "application/atom+xml",
        },
    })
    sources = [
        _source(id="github_src", content_type="github_release",
                url="https://github.com/deepseek-ai/DeepSeek-V3/releases.atom", enabled=False),
        _source(id="openai_pricing", content_type="pricing_page", enabled=False),
    ]
    report = resolve_all_disabled(sources, http_client=http)
    sources = apply_safe_resolutions(sources, report)
    github = next(s for s in sources if s["id"] == "github_src")
    pricing = next(s for s in sources if s["id"] == "openai_pricing")
    assert github["enabled"]
    assert not pricing.get("enabled", True)  # was False, stays False


def test_apply_safe_does_not_enable_snapshot_sources():
    http = MockHTTPClient({
        "https://openai.com/api/pricing/": {
            "status": 200, "body": "<html>pricing tokens</html>", "ct": "text/html",
        },
    })
    sources = [
        _source(id="openai_pricing", content_type="pricing_page", enabled=False),
        _source(id="livebench", content_type="benchmark_tracker", enabled=False),
    ]
    report = resolve_all_disabled(sources, http_client=http)
    sources = apply_safe_resolutions(sources, report)
    for s in sources:
        assert not s.get("enabled", True), f"{s['id']} should stay disabled"


def test_apply_safe_preserves_manual_urls_without_force():
    sources = [
        _source(id="github_src", content_type="github_release",
                url="https://manual-url.com/feed", enabled=False),
    ]
    report = SourceResolutionReport(total_checked=1)
    report.results.append(SourceResolutionResult(
        source_id="github_src", status="resolved_enable_safe",
        selected_url="https://auto-url.com/feed",
        recommended_enabled=True, recommended_parser_strategy="rss",
        reason="test",
    ))
    sources = apply_safe_resolutions(sources, report, force=False)
    assert sources[0]["url"] == "https://manual-url.com/feed"  # preserved


# ── Dry run report test ────────────────────────────────────────────────


def test_source_resolve_dry_run_writes_report():
    sources = [
        _source(id="test_src", content_type="github_release",
                url="https://example.com/rss", enabled=False),
    ]
    http = MockHTTPClient({
        "https://example.com/rss": {
            "status": 200, "body": _ATOM_BODY, "ct": "application/atom+xml",
        },
    })
    with tempfile.TemporaryDirectory() as d:
        report = resolve_all_disabled(sources, http_client=http)
        reports_dir = os.path.join(d, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f"source_resolution_{report.date}.json")
        with open(report_path, "w") as f:
            json.dump({"results": []}, f)
        assert os.path.exists(report_path)


def test_source_resolve_json_output():
    http = MockHTTPClient({
        "https://github.com/deepseek-ai/DeepSeek-V3/releases.atom": {
            "status": 200, "body": _ATOM_BODY, "ct": "application/atom+xml",
        },
    })
    src = _source(id="deepseek_github_releases", content_type="github_release",
                  url="https://github.com/deepseek-ai/DeepSeek-V3/releases.atom")
    result = resolve_source(src, http_client=http)
    d = {
        "source_id": result.source_id,
        "status": result.status,
        "recommended_enabled": result.recommended_enabled,
        "risk_level": result.risk_level,
    }
    assert d["status"] == "resolved_enable_safe"
    assert d["recommended_enabled"]
