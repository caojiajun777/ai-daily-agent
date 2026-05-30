from agent.sources.sitemap import SitemapAdapter


def test_sitemap_adapter_filters_and_fetches_page_meta(tmp_path):
    page = tmp_path / "claude-opus-4-8.html"
    page.write_text(
        """
        <html>
          <head>
            <meta property="og:title" content="Introducing Claude Opus 4.8">
            <meta name="description" content="Our latest model improves coding and agentic work.">
          </head>
        </html>
        """,
        encoding="utf-8",
    )
    sitemap = tmp_path / "sitemap.xml"
    sitemap.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url>
            <loc>{page}</loc>
            <lastmod>2026-05-28T16:53:42.000Z</lastmod>
          </url>
          <url>
            <loc>{tmp_path / 'about.html'}</loc>
            <lastmod>2026-05-30T10:00:00.000Z</lastmod>
          </url>
        </urlset>
        """,
        encoding="utf-8",
    )

    adapter = SitemapAdapter(
        source_id="anthropic_news",
        url=str(sitemap),
        include_path="claude-opus",
    )

    items = adapter.fetch(max_items=5)

    assert len(items) == 1
    assert items[0].source_id == "anthropic_news"
    assert items[0].source_type == "sitemap"
    assert items[0].title == "Introducing Claude Opus 4.8"
    assert items[0].summary == "Our latest model improves coding and agentic work."
    assert items[0].published_at == "2026-05-28T16:53:42+00:00"
