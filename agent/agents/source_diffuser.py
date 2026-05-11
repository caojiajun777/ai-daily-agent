"""Source Diffuser — social-graph and content-based source expansion.

Inspired by social-media recommendation algorithms (collaborative filtering,
graph traversal, PageRank-like authority scoring), this module discovers
new information sources by "diffusing" outward from a trusted seed set.

Two diffusion modes:

  1. **Social graph diffusion** (X API required):
     From seed X accounts, traverse the "following" graph 1-2 hops.
     Accounts followed by multiple trusted seeds are high-signal candidates.
     This is collaborative filtering: "people you trust also trust X".

  2. **Content link diffusion** (no API key needed):
     From collected RawItems, extract outbound URL domains that aren't
     already in the config. Try to find RSS feeds on those domains.
     This is content-based recommendation: "sources you read link to X".

Both modes produce validated CandidateSource objects ready for merge.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import feedparser
import httpx
import yaml

from agent.harness.trace import Tracer
from agent.sources.base import RawItem


# ── Common RSS paths to probe on newly discovered domains ──────────────
_RSS_PROBE_PATHS = [
    "/feed.xml", "/feed/", "/feed", "/rss.xml", "/rss/", "/rss",
    "/index.xml", "/atom.xml", "/blog/feed.xml", "/blog/feed/",
    "/news/rss.xml", "/news/feed", "/blog/rss.xml",
]


@dataclass
class DiffusedSource:
    """A candidate source discovered via diffusion."""
    name: str
    source_type: str          # rss / x
    url: Optional[str] = None
    username: Optional[str] = None
    account_type: str = "media"
    language: str = "en"
    category: str = ""
    reason: str = ""
    # Diffusion metadata.
    diffusion_method: str = ""      # "social_graph" | "content_link"
    seed_overlap_count: int = 0     # how many seeds follow/endorse this source
    link_count: int = 0             # how many collected items link to this domain
    # Validation results.
    validated: bool = False
    reachable: bool = False
    freshness_score: float = 0.0
    relevance_score: float = 0.0
    overall_score: float = 0.0
    validation_note: str = ""


@dataclass
class DiffusionReport:
    method: str
    seeds_used: int
    candidates_discovered: int
    candidates_validated: int
    candidates_passed: int
    passed: List[DiffusedSource] = field(default_factory=list)
    rejected: List[DiffusedSource] = field(default_factory=list)
    yaml_snippet: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════


def diffuse_sources(
    *,
    config_path: str,
    collected_items: Optional[List[RawItem]] = None,
    tracer: Optional[Tracer] = None,
    max_graph_candidates: int = 30,
    max_link_candidates: int = 20,
    min_overlap: int = 2,
    max_graph_hops: int = 1,
) -> Dict[str, Any]:
    """Run both diffusion modes and merge results.

    Returns a dict with keys:
      - social_graph: DiffusionReport or None (None if no X_BEARER_TOKEN)
      - content_links: DiffusionReport or None
      - merged_yaml: combined YAML snippet
      - summary: human-readable summary
    """
    existing_domains = _load_existing_domains(config_path)
    existing_x = _load_existing_x_usernames(config_path)

    result: Dict[str, Any] = {}

    # ── Mode 1: Social graph diffusion ──────────────────────────────
    graph_report = None
    x_token = os.getenv("X_BEARER_TOKEN", "")
    if x_token:
        graph_report = _diffuse_social_graph(
            config_path=config_path,
            existing_x=existing_x,
            max_candidates=max_graph_candidates,
            min_overlap=min_overlap,
            max_hops=max_graph_hops,
        )
    result["social_graph"] = graph_report

    # ── Mode 2: Content link diffusion ──────────────────────────────
    link_report = None
    if collected_items:
        link_report = _diffuse_content_links(
            items=collected_items,
            existing_domains=existing_domains,
            existing_x=existing_x,
            max_candidates=max_link_candidates,
            config_path=config_path,
        )
    result["content_links"] = link_report

    # ── Merge ───────────────────────────────────────────────────────
    all_passed: List[DiffusedSource] = []
    if graph_report:
        all_passed.extend(graph_report.passed)
    if link_report:
        # Avoid dupes: skip link candidates that already appeared in graph.
        graph_usernames = {c.username.lower() for c in (graph_report.passed if graph_report else []) if c.username}
        graph_domains = {_domain(c.url) for c in (graph_report.passed if graph_report else []) if c.url}
        for c in link_report.passed:
            if c.username and c.username.lower() in graph_usernames:
                continue
            if c.url and _domain(c.url) in graph_domains:
                continue
            all_passed.append(c)

    all_passed.sort(key=lambda c: c.overall_score, reverse=True)
    result["merged_yaml"] = _render_diffused_yaml(all_passed[:20])
    result["summary"] = (
        f"Graph diffusion: {graph_report.candidates_passed if graph_report else 'skipped (no X token)'} passed. "
        f"Link diffusion: {link_report.candidates_passed if link_report else 'skipped (no items)'} passed. "
        f"Merged: {len(all_passed)} unique candidates."
    )

    if tracer:
        tracer.log("source_diffusion", summary=result["summary"])

    return result


# ═══════════════════════════════════════════════════════════════════════
# Mode 1: Social Graph Diffusion
# ═══════════════════════════════════════════════════════════════════════


def _diffuse_social_graph(
    *,
    config_path: str,
    existing_x: Set[str],
    max_candidates: int,
    min_overlap: int,
    max_hops: int,
) -> DiffusionReport:
    """Traverse the X follow graph outward from seed accounts.

    Algorithm (collaborative filtering):
      1. For each seed, GET /users/:id/following (up to 200 per seed).
      2. Count how many seeds follow each candidate.
      3. Filter: candidates with >= min_overlap seed followers.
      4. For each candidate, fetch profile + recent tweets to score.
      5. Rank by: overlap_count × follower_ratio × AI_relevance.
    """
    token = os.getenv("X_BEARTER_TOKEN", "")
    if not token:
        return DiffusionReport(
            method="social_graph", seeds_used=0,
            candidates_discovered=0, candidates_validated=0,
            candidates_passed=0,
        )

    seeds = _load_seed_x_accounts(config_path)
    if not seeds:
        return DiffusionReport(
            method="social_graph", seeds_used=0,
            candidates_discovered=0, candidates_validated=0,
            candidates_passed=0,
        )

    client = httpx.Client(
        base_url="https://api.x.com/2",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "report-agent-diffuser/0.1",
        },
        timeout=20.0,
    )

    # Phase 1: collect follows from each seed.
    follow_counts: Counter = Counter()
    seed_followers: Dict[str, int] = {}  # username → follower_count
    seen_ids: Set[str] = set()

    for seed_username, seed_weight in seeds[:15]:  # top 15 seeds by weight
        uid = _resolve_user_id(client, seed_username)
        if not uid:
            continue
        follows = _get_following(client, uid, max_results=200)
        for f_username, f_name, f_followers in follows:
            f_username_lower = f_username.lower()
            if f_username_lower in existing_x:
                continue  # already subscribed
            if f_username_lower in {s[0].lower() for s in seeds}:
                continue  # is a seed itself
            follow_counts[f_username_lower] += 1
            if f_username_lower not in seed_followers:
                seed_followers[f_username_lower] = f_followers

    # Phase 2: filter candidates by overlap threshold.
    candidates: List[Tuple[str, int, str, int]] = []  # (username, overlap, name, followers)
    for username, count in follow_counts.most_common(200):
        if count >= min_overlap:
            name = username  # will be refined in Phase 3
            followers = seed_followers.get(username, 0)
            candidates.append((username, count, name, followers))

    if not candidates:
        return DiffusionReport(
            method="social_graph", seeds_used=len(seeds),
            candidates_discovered=0, candidates_validated=0,
            candidates_passed=0,
        )

    report = DiffusionReport(
        method="social_graph",
        seeds_used=len(seeds),
        candidates_discovered=len(candidates),
        candidates_validated=0,
        candidates_passed=0,
    )

    # Phase 3: validate each candidate (fetch profile + recent tweets).
    # Sort by overlap * log(followers) to prioritize high-signal first.
    candidates.sort(key=lambda x: x[1] * (1 + (x[3] or 0) ** 0.1), reverse=True)

    for username, overlap, name, followers in candidates[:max_candidates]:
        ds = DiffusedSource(
            name=name or f"@{username}",
            source_type="x",
            username=username,
            account_type=_infer_account_type(followers),
            language="en",
            category="",
            reason=f"Followed by {overlap} trusted AI sources (collaborative filtering)",
            diffusion_method="social_graph",
            seed_overlap_count=overlap,
        )

        # Validate with X API.
        uid = _resolve_user_id(client, username)
        if not uid:
            ds.validation_note = f"X user '{username}' not found"
            report.rejected.append(ds)
            continue

        # Get profile info.
        profile = _get_user_profile(client, uid)
        ds.name = profile.get("name", ds.name) or ds.name
        ds.relevance_score = 0.8  # social signal is relevance

        # Get recent tweets for freshness.
        tweets = _get_recent_tweets(client, uid, count=10)
        if tweets:
            newest = tweets[0].get("created_at", "")
            ds.freshness_score = _tweet_freshness(newest)
            ai_count = sum(1 for t in tweets if _is_ai_text(t.get("text", "")))
            ds.relevance_score = max(0.5, ai_count / len(tweets) if tweets else 0.5)
            ds.validation_note = (
                f"OK: {len(tweets)} recent tweets, "
                f"overlap={overlap}, followers={followers:,}, "
                f"AI relevance={ds.relevance_score:.2f}"
            )
        else:
            ds.freshness_score = 0.3
            ds.validation_note = f"OK: profile found, followers={followers:,}, overlap={overlap}"

        ds.reachable = True
        ds.validated = True
        ds.overall_score = min(1.0,
            (overlap / max(1, len(seeds))) * 0.5
            + ds.freshness_score * 0.25
            + ds.relevance_score * 0.25
        )

        if ds.overall_score >= 0.3:
            report.passed.append(ds)
        else:
            report.rejected.append(ds)

    report.candidates_validated = len(report.passed) + len(report.rejected)
    report.candidates_passed = len(report.passed)
    report.yaml_snippet = _render_diffused_yaml(report.passed)

    return report


# ═══════════════════════════════════════════════════════════════════════
# Mode 2: Content Link Diffusion
# ═══════════════════════════════════════════════════════════════════════


def _diffuse_content_links(
    *,
    items: List[RawItem],
    existing_domains: Set[str],
    existing_x: Set[str],
    max_candidates: int,
    config_path: str,
) -> DiffusionReport:
    """Extract outbound domains from collected items and probe for RSS feeds.

    Algorithm (content-based recommendation):
      1. Extract all unique domains from item URLs.
      2. Remove existing source domains.
      3. Count link frequency per domain.
      4. For top domains, probe common RSS paths.
      5. Validate and score any discovered feeds.
    """
    # Phase 1: extract domains from collected items.
    domain_counts: Counter = Counter()
    for it in items:
        if it.url:
            domain = _domain(it.url)
            if domain and domain not in existing_domains:
                domain_counts[domain] += 1

    report = DiffusionReport(
        method="content_links",
        seeds_used=len(items),
        candidates_discovered=len(domain_counts),
        candidates_validated=0,
        candidates_passed=0,
    )

    if not domain_counts:
        return report

    # Phase 2: probe top domains for RSS feeds.
    for domain, link_count in domain_counts.most_common(40):
        if len([c for c in (report.passed + report.rejected) if c.url]) >= max_candidates * 2:
            break

        # Normalize domain for name.
        name = domain.replace(".com", "").replace(".org", "").replace(".io", "")
        name = name.replace("www.", "").replace("blog.", "").replace("news.", "")
        name = " ".join(w.capitalize() for w in re.split(r"[.-]", name) if w)

        # Probe RSS paths.
        feed_url = _probe_rss(domain)
        if not feed_url:
            continue

        ds = DiffusedSource(
            name=name,
            source_type="rss",
            url=feed_url,
            language=_guess_language(domain),
            category="media",
            reason=f"Linked from {link_count} collected AI news items",
            diffusion_method="content_link",
            link_count=link_count,
        )

        # Validate the feed.
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:
            ds.validation_note = f"feedparser error on {feed_url}"
            report.rejected.append(ds)
            continue

        if parsed.bozo and not parsed.entries:
            ds.validation_note = f"feed not parseable: {parsed.bozo_exception}"
            report.rejected.append(ds)
            continue

        entries = parsed.entries[:20]
        if not entries:
            ds.validation_note = "empty feed"
            report.rejected.append(ds)
            continue

        ds.reachable = True
        ds.validated = True
        ds.freshness_score = _feed_freshness(entries)
        ds.relevance_score = _feed_ai_relevance(entries)
        ds.overall_score = min(1.0,
            min(link_count / 10, 1.0) * 0.35
            + ds.freshness_score * 0.35
            + ds.relevance_score * 0.30
        )
        ds.validation_note = (
            f"OK: {len(entries)} entries, links={link_count}, "
            f"freshness={ds.freshness_score:.2f}, "
            f"relevance={ds.relevance_score:.2f}"
        )

        if ds.overall_score >= 0.25:
            report.passed.append(ds)
        else:
            report.rejected.append(ds)

    report.candidates_validated = len(report.passed) + len(report.rejected)
    report.candidates_passed = len(report.passed)
    report.yaml_snippet = _render_diffused_yaml(report.passed)

    return report


# ═══════════════════════════════════════════════════════════════════════
# X API helpers
# ═══════════════════════════════════════════════════════════════════════


def _resolve_user_id(client: httpx.Client, username: str) -> Optional[str]:
    try:
        resp = client.get(f"/users/by/username/{username}")
        if resp.status_code == 200:
            return resp.json().get("data", {}).get("id")
    except Exception:
        pass
    return None


def _get_following(
    client: httpx.Client, user_id: str, max_results: int = 200
) -> List[Tuple[str, str, int]]:
    """Get accounts a user follows. Returns [(username, name, follower_count), ...]."""
    results: List[Tuple[str, str, int]] = []
    try:
        resp = client.get(
            f"/users/{user_id}/following",
            params={
                "max_results": min(max_results, 200),
                "user.fields": "username,name,public_metrics",
            },
        )
        if resp.status_code != 200:
            return results
        for u in resp.json().get("data", []):
            metrics = u.get("public_metrics", {})
            results.append((
                u.get("username", ""),
                u.get("name", ""),
                metrics.get("followers_count", 0),
            ))
    except Exception:
        pass
    return results


def _get_user_profile(client: httpx.Client, user_id: str) -> Dict[str, Any]:
    try:
        resp = client.get(
            f"/users/{user_id}",
            params={"user.fields": "name,username,description,public_metrics"},
        )
        if resp.status_code == 200:
            return resp.json().get("data", {})
    except Exception:
        pass
    return {}


def _get_recent_tweets(
    client: httpx.Client, user_id: str, count: int = 10
) -> List[Dict[str, Any]]:
    try:
        resp = client.get(
            f"/users/{user_id}/tweets",
            params={
                "max_results": count,
                "tweet.fields": "created_at,text,lang",
                "exclude": "retweets,replies",
            },
        )
        if resp.status_code == 200:
            return resp.json().get("data", [])
    except Exception:
        pass
    return []


# ═══════════════════════════════════════════════════════════════════════
# RSS probing
# ═══════════════════════════════════════════════════════════════════════


def _probe_rss(domain: str, timeout: float = 8.0) -> Optional[str]:
    """Try common RSS paths on a domain. Return the first valid feed URL."""
    for path in _RSS_PROBE_PATHS:
        url = f"https://{domain}{path}"
        try:
            resp = httpx.get(url, timeout=timeout, follow_redirects=True)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                text = resp.text[:2000].strip()
                # Check for XML/RSS/Atom signatures.
                is_xml = "xml" in content_type or text.startswith("<?xml") or text.startswith("<rss") or text.startswith("<feed")
                if is_xml:
                    # Quick parse check.
                    try:
                        parsed = feedparser.parse(text)
                        if parsed.entries or (not parsed.bozo):
                            return url
                    except Exception:
                        pass
        except Exception:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════
# Scoring helpers
# ═══════════════════════════════════════════════════════════════════════

_AI_KEYWORDS_LOWER = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "llm", "gpt", "claude", "gemini", "deepseek", "qwen", "model",
    "transformer", "diffusion", "neural", "training", "fine-tune",
    "inference", "benchmark", "open source", "agent", "multimodal",
    "rag", "embedding", "copilot", "codex", "chatbot", "robotics",
    "alignment", "safety", "rlhf", "lora", "开源", "模型", "智能",
    "大模型", "推理", "训练", "智能体", "多模态", "语音",
]


def _is_ai_text(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _AI_KEYWORDS_LOWER)


def _tweet_freshness(newest_iso: str) -> float:
    if not newest_iso:
        return 0.2
    try:
        newest = datetime.fromisoformat(newest_iso.replace("Z", "+00:00"))
        h = (datetime.now(timezone.utc) - newest).total_seconds() / 3600
        if h <= 24:
            return 1.0
        if h <= 72:
            return 0.7
        if h <= 168:
            return 0.4
        return 0.1
    except Exception:
        return 0.2


def _feed_freshness(entries) -> float:
    newest_str = ""
    for e in entries:
        t = e.get("published_parsed") or e.get("updated_parsed")
        if t:
            try:
                ts = time.mktime(t)
                newest_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                break
            except Exception:
                pass
    return _tweet_freshness(newest_str)


def _feed_ai_relevance(entries) -> float:
    if not entries:
        return 0.0
    hits = sum(1 for e in entries if _is_ai_text(getattr(e, "title", "")))
    return hits / len(entries)


def _infer_account_type(followers: int) -> str:
    if followers > 500_000:
        return "official"
    if followers > 10_000:
        return "kol"
    return "media"


# ═══════════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════════


def _load_seed_x_accounts(config_path: str) -> List[Tuple[str, float]]:
    """Return [(username, weight), ...] sorted by weight desc for top seeds."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return []
    accounts: List[Tuple[str, float]] = []
    for s in cfg.get("sources", []):
        if s.get("type") == "x" and s.get("username"):
            accounts.append((s["username"], float(s.get("weight", 1.0))))
    accounts.sort(key=lambda x: -x[1])
    return accounts


def _load_existing_domains(config_path: str) -> Set[str]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return set()
    domains: Set[str] = set()
    for s in cfg.get("sources", []):
        if s.get("type") == "rss" and s.get("url"):
            d = _domain(s["url"])
            if d:
                domains.add(d)
    return domains


def _load_existing_x_usernames(config_path: str) -> Set[str]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return set()
    return {s["username"].lower().lstrip("@")
            for s in cfg.get("sources", [])
            if s.get("type") == "x" and s.get("username")}


def _domain(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        p = urlparse(url)
        return p.netloc.lower().replace("www.", "").replace("blog.", "")
    except Exception:
        return ""


def _guess_language(domain: str) -> str:
    cn_tlds = {".cn", ".com.cn", ".中国"}
    for tld in cn_tlds:
        if domain.endswith(tld):
            return "zh"
    return "en"


# ═══════════════════════════════════════════════════════════════════════
# Output rendering
# ═══════════════════════════════════════════════════════════════════════


def _render_diffused_yaml(sources: List[DiffusedSource]) -> str:
    if not sources:
        return ""
    lines = ["# ── Diffused sources (social graph + content link discovery) ──", ""]
    for c in sources:
        method_tag = "graph" if c.diffusion_method == "social_graph" else "link"
        lines.append(
            f"# [{c.overall_score:.2f}] [{method_tag}] {c.reason}"
        )
        if c.source_type == "rss":
            slug = re.sub(r"[^a-z0-9_]", "_", c.name.lower().replace(" ", "_"))[:30]
            lines.append(f"- id: \"diffused_{slug}\"")
            lines.append(f"  type: \"rss\"")
            lines.append(f"  url: \"{c.url}\"")
        elif c.source_type == "x":
            slug = re.sub(r"[^a-z0-9_]", "_", (c.username or c.name).lower())[:30]
            lines.append(f"- id: \"diffused_x_{slug}\"")
            lines.append(f"  type: \"x\"")
            lines.append(f"  username: \"{c.username}\"")
            lines.append(f"  account_type: \"{c.account_type}\"")
        lines.append(f"  weight: {0.6 + c.overall_score * 0.4:.1f}")
        lines.append(f"  max_items: {max(3, int(8 * c.overall_score))}")
        lines.append("")
    return "\n".join(lines)
