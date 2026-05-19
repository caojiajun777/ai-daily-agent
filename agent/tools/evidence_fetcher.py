"""Evidence Fetcher — lightweight URL evidence extraction for ResearchEditor.

Fetches title, meta description, and first ~1500 chars of body text from
each URL. Provides source_type hints. Never blocks — all failures are silent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_DESC_RE = re.compile(
    r'<meta\s[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_OG_DESC_RE = re.compile(
    r'<meta\s[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_BODY_START = re.compile(r"<body[^>]*>", re.IGNORECASE)
_BODY_TEXT_RE = re.compile(r">([^<]{20,})<")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


_EVIDENCE_TYPE_HINTS: Dict[str, str] = {
    "github.com": "github_repo",
    "arxiv.org": "arxiv_or_paper",
    "openai.com/index": "official_blog",
    "openai.com/news": "official_blog",
    "anthropic.com": "official_blog",
    "deepmind.google": "official_blog",
    "huggingface.co/blog": "official_blog",
    "huggingface.co/papers": "arxiv_or_paper",
    "paperswithcode.com": "arxiv_or_paper",
    "technologyreview.com": "trusted_media",
    "wired.com": "trusted_media",
    "venturebeat.com": "trusted_media",
    "the-decoder.com": "trusted_media",
    "docs.": "docs",
    "/docs/": "docs",
    "pricing": "pricing_page",
    "benchmark": "benchmark",
    "regulation": "regulatory_or_filing",
    ".gov": "regulatory_or_filing",
    "x.com": "social_post",
    "twitter.com": "social_post",
}


@dataclass
class EvidenceSnippet:
    url: str
    title: str = ""
    text_snippet: str = ""
    source_type: str = ""
    evidence_type: str = "unknown"
    fetch_status: str = "skipped"
    confidence: str = "low"


def fetch_evidence(
    urls: List[str],
    timeout: float = 8.0,
) -> List[EvidenceSnippet]:
    """Fetch evidence from a list of URLs. Non-blocking, best-effort."""
    results: List[EvidenceSnippet] = []
    for url in urls[:10]:  # Cap at 10 URLs per event.
        try:
            snippet = _fetch_one(url, timeout)
            results.append(snippet)
        except Exception:
            results.append(EvidenceSnippet(
                url=url, fetch_status="failed",
                evidence_type=_guess_evidence_type(url),
            ))
    return results


def fetch_evidence_for_events(
    event_urls_list: List[List[str]],
    timeout: float = 8.0,
) -> List[List[EvidenceSnippet]]:
    """Fetch evidence for multiple events. Returns one list per event."""
    return [fetch_evidence(urls, timeout) for urls in event_urls_list]


def _fetch_one(url: str, timeout: float) -> EvidenceSnippet:
    if not url.startswith(("http://", "https://")):
        return EvidenceSnippet(url=url, fetch_status="invalid_url",
                               evidence_type="unknown")

    evidence_type = _guess_evidence_type(url)

    try:
        with httpx.Client(timeout=min(timeout, 10.0), follow_redirects=True) as client:
            resp = client.get(
                url,
                headers={
                    "User-Agent": "report-agent-evidence/0.1",
                    "Accept": "text/html",
                },
            )
            if resp.status_code != 200:
                return EvidenceSnippet(
                    url=url, fetch_status=f"http_{resp.status_code}",
                    evidence_type=evidence_type,
                )
            html = resp.text
    except Exception as e:
        return EvidenceSnippet(
            url=url, fetch_status=f"error: {str(e)[:80]}",
            evidence_type=evidence_type,
        )

    # Extract metadata.
    title = ""
    m = _TITLE_RE.search(html[:10000])
    if m:
        title = _strip_html(m.group(1)).strip()[:200]

    desc = ""
    m = _META_DESC_RE.search(html[:20000])
    if not m:
        m = _OG_DESC_RE.search(html[:20000])
    if m:
        desc = m.group(1).strip()[:300]

    # Extract body text (first ~1500 chars of meaningful text).
    body_text = ""
    body_start = _BODY_START.search(html[:50000])
    if body_start:
        body_html = html[body_start.start():body_start.start() + 80000]
        # Remove script/style blocks.
        body_html = re.sub(r"<(script|style)[^>]*>[\s\S]*?</\1>", "", body_html, flags=re.IGNORECASE)
        clean = _HTML_TAG_RE.sub(" ", body_html)
        clean = re.sub(r"\s+", " ", clean).strip()
        body_text = clean[:1500]
    else:
        body_text = desc

    text_snippet = f"{title}. {desc}. {body_text}"[:2000]

    return EvidenceSnippet(
        url=url,
        title=title,
        text_snippet=text_snippet,
        source_type=_guess_source_type(url),
        evidence_type=evidence_type,
        fetch_status="ok",
        confidence="medium" if title and desc else "low",
    )


def _guess_evidence_type(url: str) -> str:
    lower = url.lower()
    for hint, etype in _EVIDENCE_TYPE_HINTS.items():
        if hint in lower:
            return etype
    return "unknown"


def _guess_source_type(url: str) -> str:
    lower = url.lower()
    if any(d in lower for d in ("openai.com", "anthropic.com", "deepmind.google",
                                  "ai.meta.com", "blog.google", "mistral.ai")):
        return "official"
    if "github.com" in lower: return "github"
    if "arxiv.org" in lower: return "paper"
    if "x.com" in lower or "twitter.com" in lower: return "social"
    if any(d in lower for d in ("technologyreview.com", "wired.com",
                                  "venturebeat.com", "the-decoder.com")):
        return "trusted_media"
    return "unknown"


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text)
