from __future__ import annotations

from agent.sources.rss import RssAdapter


def test_rss_adapter_fetches_feed_with_bounded_http_client(monkeypatch):
    captured = {}
    feed = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Model launch</title>
      <link>https://example.com/model</link>
      <description>New model is available.</description>
      <pubDate>Fri, 22 May 2026 10:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""

    class FakeResponse:
        content = feed

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *, timeout, follow_redirects, headers):
            captured["timeout"] = timeout
            captured["follow_redirects"] = follow_redirects
            captured["headers"] = headers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            captured["url"] = url
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

    items = RssAdapter(
        source_id="example_feed",
        url="https://example.com/feed.xml",
        timeout_sec=3,
    ).fetch(max_items=5)

    assert captured["url"] == "https://example.com/feed.xml"
    assert captured["follow_redirects"] is True
    assert "report-agent-rss" in captured["headers"]["User-Agent"]
    assert len(items) == 1
    assert items[0].title == "Model launch"
    assert items[0].url == "https://example.com/model"
    assert items[0].summary == "New model is available."
