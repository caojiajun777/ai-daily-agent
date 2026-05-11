"""AI HOT Daily scraper — both a content source and a source-of-sources.

AI HOT (aihot.virxact.com/daily) is a curated Chinese AI daily report
with excellent source attribution. Each article shows exactly where it
came from (X account, RSS feed, GitHub Releases, etc.).

This adapter does two things:

  1. Content extraction — parses the daily page into RawItem entries
     so our pipeline can use AI HOT as another content feed.

  2. Source discovery — extracts the unique source names/URLs attributed
     by AI HOT, cross-references with our config, and outputs new
     candidate sources we haven't added yet.

Usage in default.yaml:
  - id: "aihot_daily"
    type: "aihot"
    url: "https://aihot.virxact.com/daily"
    weight: 1.0
    max_items: 25
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

import httpx

from agent.sources.base import RawItem


class AIHotAdapter:
    type_name = "aihot"

    def __init__(self, source_id: str, url: str) -> None:
        self.source_id = source_id
        self.url = url

    def fetch(self, *, max_items: int = 25) -> List[RawItem]:
        html = _fetch_html(self.url)
        if not html:
            return []
        items = _extract_articles(html, self.source_id)
        return items[:max_items]


def extract_sources_from_aihot(url: str = "https://aihot.virxact.com/daily") -> List[Dict[str, str]]:
    """Extract unique source attributions from AI HOT Daily.

    Returns a list of dicts with keys: name, category, type_hint.
    These can be fed into the source discoverer for validation.
    """
    html = _fetch_html(url)
    if not html:
        return []

    # AI HOT source blocks look like:
    #   <span class="role-tag">官方·X</span>
    #   <span>X：百度 Baidu (@Baidu_Inc)</span>
    #   <span class="role-tag">综合资讯</span>
    #   <span>IT之家（RSS）</span>
    #   <span class="role-tag">官方</span>
    #   <span>Claude Code：GitHub Releases（RSS）</span>

    sources: List[Dict[str, str]] = []
    seen: Set[str] = set()

    # Pattern: role-tag + following span text.
    pattern = re.compile(
        r'<span[^>]*class="[^"]*role-tag[^"]*"[^>]*>(.*?)</span>\s*<span>(.*?)</span>',
        re.DOTALL,
    )

    for m in pattern.finditer(html):
        role = m.group(1).strip()
        desc = m.group(2).strip()
        key = f"{role}|{desc}"
        if key in seen:
            continue
        seen.add(key)

        source_info = _parse_source_attribution(role, desc)
        if source_info:
            sources.append(source_info)

    return sources


# ── HTML fetching ───────────────────────────────────────────────────────


def _fetch_html(url: str, timeout: float = 15.0) -> str:
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html",
                },
            )
            if resp.status_code == 200:
                return resp.text
    except Exception:
        pass
    return ""


# ── Article extraction ──────────────────────────────────────────────────


def _extract_articles(html: str, source_id: str) -> List[RawItem]:
    """Parse AI HOT Daily page into RawItem entries."""
    items: List[RawItem] = []

    # AI HOT article blocks: <article class="daily-article">
    #   <h3><a href="...">Title</a></h3>
    #   <div class="daily-article-source"><span class="role-tag">...</span><span>...</span></div>
    #   <p class="daily-article-summary">Summary text</p>

    article_re = re.compile(
        r'<article\s[^>]*class="[^"]*daily-article[^"]*"[^>]*>(.*?)</article>',
        re.DOTALL,
    )
    title_re = re.compile(
        r'<h3[^>]*>\s*<a\s[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    summary_re = re.compile(
        r'<p\s[^>]*class="[^"]*daily-article-summary[^"]*"[^>]*>(.*?)</p>',
        re.DOTALL,
    )
    source_re = re.compile(
        r'<span[^>]*class="[^"]*role-tag[^"]*"[^>]*>(.*?)</span>\s*<span>(.*?)</span>',
        re.DOTALL,
    )

    for art_m in article_re.finditer(html):
        art_html = art_m.group(1)

        t_m = title_re.search(art_html)
        s_m = summary_re.search(art_html)
        src_m = source_re.search(art_html)

        if not t_m:
            continue

        url = t_m.group(1).strip()
        title = _strip_html(t_m.group(2)).strip()
        summary = _strip_html(s_m.group(1)).strip() if s_m else ""

        role_tag = src_m.group(1).strip() if src_m else ""
        source_name = _strip_html(src_m.group(2)).strip() if src_m else ""

        # Map AI HOT source names to our own source_ids when possible.
        mapped_source = _map_source_name(source_name, role_tag)

        if not title:
            continue

        items.append(RawItem(
            source_id=mapped_source,
            source_type="aihot",
            title=title,
            url=url,
            summary=summary[:800],
            published_at=datetime.now(timezone.utc).isoformat(),
            author="",
            tags=[role_tag, source_name],
        ))

    return items


# ── Source attribution parsing ──────────────────────────────────────────


def _parse_source_attribution(role: str, desc: str) -> Optional[Dict[str, str]]:
    """Parse an AI HOT source attribution line into structured info.

    Examples:
      "官方·X" + "X：百度 Baidu (@Baidu_Inc)" → {name: "百度 Baidu", type_hint: "x", username: "Baidu_Inc"}
      "综合资讯" + "IT之家（RSS）" → {name: "IT之家", type_hint: "rss"}
      "官方" + "Claude Code：GitHub Releases（RSS）" → {name: "Claude Code GitHub Releases", type_hint: "rss"}
    """
    info: Dict[str, str] = {"category": role}

    # Detect X/Twitter accounts.
    x_match = re.search(r"@(\w+)", desc)
    if x_match:
        info["type_hint"] = "x"
        info["username"] = x_match.group(1)
        info["name"] = desc.split("@")[0].strip().rstrip("(").strip()
        return info

    # Detect RSS feeds.
    if "RSS" in desc or "rss" in desc.lower() or "feed" in desc.lower():
        info["type_hint"] = "rss"
        info["name"] = re.sub(r"[（(][^)）]*[Rr][Ss][Ss][^)）]*[)）]", "", desc).strip()
        return info

    # Detect GitHub Releases.
    if "GitHub" in desc or "github" in desc.lower():
        info["type_hint"] = "github_releases"
        info["name"] = desc.strip()
        return info

    # Fallback: treat as media.
    info["type_hint"] = "media"
    info["name"] = desc.strip()
    return info


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


# ── Source name mapping ─────────────────────────────────────────────────
# Map AI HOT's source attribution text to our own source_ids.
# If no mapping is found, we use a cleaned version of the AI HOT attribution
# so the draft shows e.g. "X：百度 Baidu" instead of "aihot_daily".

_KNOWN_AIHOT_SOURCES: Dict[str, str] = {
    # X accounts
    "Baidu_Inc": "x_baidu",
    "alibaba_cloud": "x_alicloud",
    "StepFun_ai": "x_stepfun",
    "OpenRouter": "x_openrouter",
    "SiliconFlowAI": "x_siliconflow",
    "TencentHunyuan": "x_tencent_hunyuan",
    "Alibaba_Qwen": "x_qwen",
    "OpenAI": "x_openai",
    "AnthropicAI": "x_anthropic",
    "GoogleDeepMind": "x_googledeepmind",
    "AIatMeta": "x_metaai",
    "MistralAI": "x_mistralai",
    "StabilityAI": "x_stabilityai",
    "NousResearch": "x_nous",
    "OpenAIDevs": "x_openai_devs",
    "deepseek_ai": "x_deepseek",
    "ChatGLM": "x_zhipu",
    "kaboroje": "x_karpathy",
    "fchollet": "x_fchollet",
    "dotey": "x_dotey",
    "AYi_AInotes": "x_ayi",
    "emollick": "x_emollick",
    "rohanpaul_ai": "x_rohanpaul",
    "berryxia": "x_berryxia",
    "steipete": "x_steipete",
    "clem": "x_clem",
    # RSS sources
    "IT之家": "ithome",
    "量子位": "qbitai",
    "机器之心": "jiqizhixin",
    "Hugging Face": "huggingface_blog",
    "OpenAI": "openai_news",
    "Anthropic": "anthropic_news",
    "The Decoder": "the_decoder",
    "WIRED": "wired_ai",
    "MIT Technology Review": "mit_tech_review_ai",
    "VentureBeat": "venturebeat_ai",
    "Ars Technica": "ars_technica_ai",
    "Hacker News": "hackernews",
}


def _map_source_name(source_name: str, role_tag: str) -> str:
    """Map an AI HOT source attribution to our own source_id.

    Falls back to a cleaned display name if no mapping found.
    """
    # Try exact match first.
    clean = source_name.strip()
    if clean in _KNOWN_AIHOT_SOURCES:
        return _KNOWN_AIHOT_SOURCES[clean]

    # Try substring match — e.g. "X：百度 Baidu (@Baidu_Inc)" → find "Baidu_Inc"
    import re as _re
    username_m = _re.search(r"@(\w+)", clean)
    if username_m:
        uname = username_m.group(1)
        if uname in _KNOWN_AIHOT_SOURCES:
            return _KNOWN_AIHOT_SOURCES[uname]

    # Try matching the name part before "：" or "（"
    name_part = _re.split(r"[：（(]", clean)[0].strip()
    if name_part in _KNOWN_AIHOT_SOURCES:
        return _KNOWN_AIHOT_SOURCES[name_part]

    # Fallback: use a readable label derived from the AI HOT attribution.
    # This shows the original source info instead of "aihot_daily".
    short = name_part[:30]
    return f"aihot:{short}" if short else "aihot_daily"
