"""Event Clusterer — groups RawItems into EventClusters by semantic identity.

Groups articles about "the same thing" — e.g. 3 sources reporting GPT-5.5
release → one EventCluster. Rules are deterministic, fast, offline.

Key clustering signals (in priority order):
  1. Exact title hash match (strongest — same article scraped/re-shared)
  2. Canonical URL match (tracking params stripped)
  3. Normalized title match (stemmed, de-branded)
  4. difflib title similarity ≥ 0.75
"""

from __future__ import annotations

import hashlib
import re
import time as _time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs, urlunparse

from agent.sources.base import RawItem

_NORM_RE = re.compile(r"[^\w一-鿿]+", flags=re.UNICODE)
_VERSION_DOT_RE = re.compile(r"(\d+)\.(\d+)")
_VERSION_PATTERN = re.compile(r"(\d+\.\d+(?:\.\d+)?)")
_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term",
                     "utm_content", "ref", "spm", "from", "fbclid", "gclid",
                     "mc_cid", "mc_eid", "_ga", "utm_id"}
_SIMILARITY_THRESHOLD = 0.68
_MARKETING_WORDS = {"exclusive", "独家", "重磅", "炸裂", "震惊", "突发",
                    "must-read", "breaking", "bombshell", "game-changer",
                    "颠覆", "碾压", "完爆", "吊打"}


@dataclass
class EventCluster:
    event_id: str
    canonical_title: str
    primary_url: str
    primary_source_name: str = ""
    primary_source_type: str = ""
    primary_content_type: str = "tech_media"
    primary_source_tier: str = ""
    primary_reliability: str = ""
    primary_evidence_type: str = ""
    primary_confidence: str = "medium"
    primary_section_hint: str = ""
    source_urls: List[str] = field(default_factory=list)
    source_names: List[str] = field(default_factory=list)
    source_types: List[str] = field(default_factory=list)
    source_count: int = 0
    published_at: str = ""
    first_seen_at: str = ""
    latest_seen_at: str = ""
    summary: str = ""
    snippets: List[str] = field(default_factory=list)
    section_hint: str = ""
    rule_score: float = 0.0
    already_reported: bool = False
    duplicate_candidates: List[str] = field(default_factory=list)
    evidence_snippets: List[str] = field(default_factory=list)


def cluster_items(items: List[RawItem]) -> List[EventCluster]:
    """Group RawItems into EventClusters. Returns clusters sorted by source_count desc."""
    if not items:
        return []

    clusters: List[EventCluster] = []
    # Maps for fast lookup: canonical_url → cluster_idx, norm_title → cluster_idx
    url_map: Dict[str, int] = {}
    title_map: Dict[str, int] = {}

    for it in items:
        if not it.title or not it.url:
            continue

        norm_title = _norm(it.title)
        canon_url = _canonical_url(it.url)

        # Check exact URL match first.
        idx = url_map.get(canon_url)
        if idx is None:
            # Check normalized title match.
            idx = title_map.get(norm_title)
        if idx is None:
            # Check fuzzy title match against existing clusters.
            for i, cluster in enumerate(clusters):
                cluster_norm = _norm(cluster.canonical_title)
                sim = SequenceMatcher(None, norm_title, cluster_norm).ratio()
                if sim >= _SIMILARITY_THRESHOLD:
                    # Prevent merging distinct product versions (e.g. Gemini 2.5 vs 3.5).
                    if _same_product_version(norm_title, cluster_norm):
                        idx = i
                        break

        if idx is not None:
            # Merge into existing cluster.
            c = clusters[idx]
            if it.url not in c.source_urls:
                c.source_urls.append(it.url)
            if it.source_id not in c.source_names:
                c.source_names.append(it.source_id)
                c.source_types.append(it.source_type)
            c.source_count = len(c.source_names)
            c.snippets.append(it.summary[:200])
            if it.published_at > (c.latest_seen_at or ""):
                c.latest_seen_at = it.published_at
            if it.published_at < (c.first_seen_at or "z"):
                c.first_seen_at = it.published_at
            # Primary URL: prefer official sources.
            if _source_rank(it.source_id, it.source_type) > _best_source_rank(c):
                c.primary_url = it.url
                c.canonical_title = it.title
                c.summary = it.summary[:500]
                _set_primary_source(c, it)
            if canon_url not in url_map:
                url_map[canon_url] = idx
            if norm_title not in title_map:
                title_map[norm_title] = idx
        else:
            # New cluster.
            cluster = EventCluster(
                event_id=_gen_event_id(it.title, it.url),
                canonical_title=it.title,
                primary_url=it.url,
                primary_source_name=it.source_id,
                primary_source_type=it.source_type,
                primary_content_type=getattr(it, "content_type", "tech_media"),
                primary_source_tier=getattr(it, "source_tier", ""),
                primary_reliability=getattr(it, "reliability", ""),
                primary_evidence_type=getattr(it, "evidence_type", ""),
                primary_confidence=getattr(it, "confidence", "medium"),
                primary_section_hint=getattr(it, "section_hint", ""),
                source_urls=[it.url],
                source_names=[it.source_id],
                source_types=[it.source_type],
                source_count=1,
                published_at=it.published_at,
                first_seen_at=it.published_at,
                latest_seen_at=it.published_at,
                summary=it.summary[:500],
                snippets=[it.summary[:200]],
            )
            idx = len(clusters)
            clusters.append(cluster)
            url_map[canon_url] = idx
            title_map[norm_title] = idx

    clusters.sort(key=lambda c: c.source_count, reverse=True)
    return clusters


def _set_primary_source(cluster: EventCluster, item: RawItem) -> None:
    cluster.primary_source_name = item.source_id
    cluster.primary_source_type = item.source_type
    cluster.primary_content_type = getattr(item, "content_type", "tech_media")
    cluster.primary_source_tier = getattr(item, "source_tier", "")
    cluster.primary_reliability = getattr(item, "reliability", "")
    cluster.primary_evidence_type = getattr(item, "evidence_type", "")
    cluster.primary_confidence = getattr(item, "confidence", "medium")
    cluster.primary_section_hint = getattr(item, "section_hint", "")


def _norm(text: str) -> str:
    t = text.lower()
    for w in _MARKETING_WORDS:
        t = t.replace(w.lower(), "")
    t = re.sub(r"&[a-z]+;", " ", t)
    # Protect version number dots so "Gemini 2.5" and "Gemini 3.5" stay distinct
    t = _VERSION_DOT_RE.sub(r"\1_VDOT_\2", t)
    t = _NORM_RE.sub(" ", t)
    t = t.replace("_VDOT_", ".")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _same_product_version(norm_a: str, norm_b: str) -> bool:
    """Returns True if both titles reference the same product version, or
    if at least one has no version number (can't determine divergence).
    Prevents merging 'Gemini 2.5 Flash' with 'Gemini 3.5 Flash'."""
    versions_a = set(_VERSION_PATTERN.findall(norm_a))
    versions_b = set(_VERSION_PATTERN.findall(norm_b))
    if not versions_a or not versions_b:
        return True
    return versions_a == versions_b


def _canonical_url(url: str) -> str:
    """Strip tracking parameters and normalize the URL."""
    try:
        p = urlparse(url)
        # Remove tracking params.
        qs = parse_qs(p.query, keep_blank_values=False)
        clean_qs = {k: v for k, v in qs.items()
                    if k.lower() not in _TRACKING_PARAMS}
        # Rebuild query string.
        new_query = "&".join(f"{k}={vs[0]}" for k, vs in clean_qs.items())
        # Lowercase netloc, strip www.
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return urlunparse((p.scheme or "https", netloc, p.path.rstrip("/") or "/",
                          p.params, new_query, ""))
    except Exception:
        return url


def _source_rank(source_id: str, source_type: str) -> int:
    """Higher = more authoritative. Official blogs/docs/releases > media > KOL > unknown."""
    s = (source_id + source_type).lower()
    if any(k in s for k in ("official", "openai", "anthropic", "deepmind",
                              "google", "github", "arxiv", "huggingface",
                              "release", "docs", "blog")):
        return 10
    if "media" in s or "news" in s or "press" in s:
        return 6
    if any(k in s for k in ("x_", "twitter", "kol")):
        return 4
    return 2


def _best_source_rank(cluster: EventCluster) -> int:
    return max((_source_rank(n, t) for n, t in zip(cluster.source_names, cluster.source_types)), default=0)


def _gen_event_id(title: str, url: str) -> str:
    h = hashlib.sha1(f"{_norm(title)}::{_canonical_url(url)}".encode()).hexdigest()[:12]
    return f"evt_{h}"
