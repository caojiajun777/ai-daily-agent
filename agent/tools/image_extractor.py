"""Smart image extraction with Playwright fallback for JS-rendered pages.

Two-tier approach:
  1. Fast HTTP og:image extraction (traditional sites, ~1s per URL).
  2. Playwright headless browser fallback (JS SPAs, ~4-8s per URL).

Only sites whose HTTP extraction returns nothing are retried with Playwright.
This keeps the pipeline fast for traditional sites while getting images from
X/Twitter, OpenAI, qbitai, and other JS-heavy platforms.
"""

from __future__ import annotations

import re
from typing import List, Tuple
from urllib.parse import urljoin

import httpx

# ── Candidate extraction patterns ───────────────────────────────────────
_OG_IMAGE_RE = re.compile(
    r'<meta\s[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_TWITTER_IMAGE_RE = re.compile(
    r'<meta\s[^>]*name=["\']twitter:image["\'][^>]*content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_IMG_SRC_RE = re.compile(
    r'<img[^>]+src=["\']([^"\']+\.(?:png|jpg|jpeg|gif|webp|PNG|JPG|JPEG|GIF|WEBP))["\']',
)
# Lazy-loaded images often have the real URL in data-original or data-src.
_DATA_ORIGINAL_RE = re.compile(
    r'<img[^>]+data-original=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_DATA_SRC_RE = re.compile(
    r'<img[^>]+data-src=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_HEAD_CLOSE_RE = re.compile(r"</head>", re.IGNORECASE)

# ── Noise detection ─────────────────────────────────────────────────────
_NOISE_WORDS = [
    "logo", "icon", "avatar", "favicon", "t.png", "1x1", "pixel",
    "placeholder", "default", "blank", "spacer", "qrcode", "qr_code",
    "qr-code", "wechat", "微信", "扫码", "banner", "sidebar", "widget",
    "button", "badge", "thumb-", "-thumb", "loading", "spinner",
    "emoji", "svg", "profile_images", "arrow.png", "arrow-",
    "weixin", "girl.png", "boy.png", "qr-icon", "share-icon",
    "bg.", "_bg.", "header-bg", "thumbnail", "/thumb/", "thumb-",
    "150x150", "240x240", "80x80", "100x100", "_240.", "_150.",
    "related", "sidebar", "recommend",
]

# ── Quality signals ─────────────────────────────────────────────────────
_QUALITY_WORDS = [
    "hero", "header", "cover", "featured", "article", "post", "news",
    "illustration", "photo", "screenshot", "chart", "graph", "diagram",
    "figure", "render", "portrait",
]

# Playwright browser instance (lazy singleton).
_browser = None


def _get_browser():
    """Lazy-init headless browser. Tries Edge first (on Windows),
    then system Chrome, then Playwright-managed Chromium."""
    global _browser
    if _browser is not None:
        return _browser
    try:
        from playwright.sync_api import sync_playwright
        _pw = sync_playwright().start()
        # Try Edge first (pre-installed on Windows), then Chrome, then default Chromium.
        for channel in [None, "msedge", "chrome"]:
            try:
                kw = {
                    "headless": True,
                    "args": [
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-gpu",
                    ],
                }
                if channel:
                    kw["channel"] = channel
                _browser = _pw.chromium.launch(**kw)
                break
            except Exception:
                continue
    except Exception:
        _browser = False
    return _browser


def extract_image(url: str, timeout: float = 8.0) -> str:
    """Fetch a page and extract the best article image.

    Tries fast HTTP extraction first. Falls back to Playwright headless
    browser for JS-rendered pages (X, OpenAI, qbitai, etc.).
    """
    if not url.startswith(("http://", "https://")):
        return ""

    # Tier 1: fast HTTP extraction.
    img = _extract_via_http(url, timeout=min(timeout, 5.0))
    if img:
        return img

    # Tier 2: Playwright for JS SPAs.
    return _extract_via_playwright(url, timeout=timeout)


def _extract_via_http(url: str, timeout: float) -> str:
    """Fast HTTP og:image extraction from raw HTML."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; report-agent/0.1; "
                        "+https://github.com/caojiajun777/ai-daily-agent)"
                    ),
                    "Accept": "text/html",
                },
            )
            if resp.status_code != 200:
                return ""
            html = resp.text
    except Exception:
        return ""

    candidates = _collect_candidates(html, url)
    return _pick_best(candidates)


def _extract_via_playwright(url: str, timeout: float) -> str:
    """Headless browser extraction for JS-rendered pages.

    Uses 'load' (not 'networkidle') to stay fast. Scrolls once to
    trigger lazy-loaded images. For X/Twitter URLs, injects the
    user's auth_token cookie so we can see the full tweet content.
    """
    import os as _os

    browser = _get_browser()
    if not browser:
        return ""

    page = None
    context = None
    try:
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        domain = ""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
        except Exception:
            pass
        if domain in ("x.com", "twitter.com", "www.x.com", "www.twitter.com"):
            x_cookies = _load_x_cookies()
            if x_cookies:
                context.add_cookies(x_cookies)
        page = context.new_page()
        page.set_viewport_size({"width": 1280, "height": 900})
        page.goto(url, wait_until="load", timeout=timeout * 1000)
        # Give JS a moment to render tweet content.
        page.wait_for_timeout(1500)
        try:
            page.evaluate("window.scrollTo(0, 800)")
            page.wait_for_timeout(600)
        except Exception:
            pass
        html = page.content()
    except Exception:
        return ""
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass
        if context:
            try:
                context.close()
            except Exception:
                pass

    # After JS render: check og:image first, then all images.
    candidates: List[Tuple[str, int]] = []
    head_end = _HEAD_CLOSE_RE.search(html)
    head = html[: head_end.start()] if head_end else html[:20000]

    for m in _OG_IMAGE_RE.finditer(head):
        candidates.append((_abs_url(m.group(1), url), 1))
    for m in _TWITTER_IMAGE_RE.finditer(head):
        candidates.append((_abs_url(m.group(1), url), 2))
    for m in _IMG_SRC_RE.finditer(html[:100000]):
        candidates.append((_abs_url(m.group(1), url), 3))
    # Lazy-loaded: data-original, data-src (IT之家, qbitai, etc.)
    for m in _DATA_ORIGINAL_RE.finditer(html[:100000]):
        candidates.append((_abs_url(m.group(1), url), 3))
    for m in _DATA_SRC_RE.finditer(html[:100000]):
        candidates.append((_abs_url(m.group(1), url), 3))

    return _pick_best(candidates)


def _collect_candidates(html: str, base_url: str) -> List[Tuple[str, int]]:
    """Collect image candidates from HTML. Returns [(url, source_priority), ...]."""
    candidates: List[Tuple[str, int]] = []
    head_end = _HEAD_CLOSE_RE.search(html)
    head = html[: head_end.start()] if head_end else html[:20000]
    body = html[head_end.start():] if head_end else ""

    for m in _OG_IMAGE_RE.finditer(head):
        candidates.append((_abs_url(m.group(1), base_url), 1))
    for m in _TWITTER_IMAGE_RE.finditer(head):
        candidates.append((_abs_url(m.group(1), base_url), 2))
    for m in _IMG_SRC_RE.finditer(body[:80000]):
        candidates.append((_abs_url(m.group(1), base_url), 3))
    # Lazy-loaded: data-original, data-src (IT之家, qbitai, etc.)
    for m in _DATA_ORIGINAL_RE.finditer(body[:80000]):
        candidates.append((_abs_url(m.group(1), base_url), 3))
    for m in _DATA_SRC_RE.finditer(body[:80000]):
        candidates.append((_abs_url(m.group(1), base_url), 3))

    return candidates


def _pick_best(candidates: List[Tuple[str, int]]) -> str:
    """Score candidates and return the best image URL."""
    best_url = ""
    best_score = -1
    seen: set = set()

    for img_url, src_pri in candidates:
        if img_url in seen:
            continue
        seen.add(img_url)
        if _is_noise(img_url):
            continue
        score = _score_image(img_url, src_pri)
        if score > best_score:
            best_score = score
            best_url = img_url

    return best_url


# ── Scoring ─────────────────────────────────────────────────────────────


def _score_image(url: str, source_priority: int) -> float:
    score = 0.0
    if source_priority == 1:
        score += 0.20
    elif source_priority == 2:
        score += 0.15

    path = url.split("?")[0]
    filename = path.rstrip("/").split("/")[-1] if "/" in path else path
    name = re.sub(r"\.(png|jpg|jpeg|gif|webp)$", "", filename, flags=re.IGNORECASE)

    if len(name) >= 20:
        score += 0.25
    elif len(name) >= 10:
        score += 0.15
    elif len(name) >= 5:
        score += 0.05

    name_lower = name.lower()
    quality_hits = sum(1 for w in _QUALITY_WORDS if w in name_lower)
    score += min(quality_hits * 0.10, 0.30)

    # UUID filename → article content image (e.g. d24f9aa6-b115-4223...).
    if re.match(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", name, re.IGNORECASE):
        score += 0.20

    path_lower = path.lower()
    # IT之家 article body images (non-thumbnail CDN path).
    if "newsuploadfiles" in path_lower and "thumbnail" not in path_lower:
        score += 0.15
    # Tweet media images are high-quality content.
    if "pbs.twimg.com/media" in path_lower:
        score += 0.30
    if any(d in path_lower for d in ["/uploads/", "/content/", "/images/",
                                       "/wp-content/", "/media/", "/photos/",
                                       "/imagehub/"]):
        score += 0.15
    if any(d in path_lower for d in ["/assets/", "/static/", "/theme/",
                                       "/templates/", "/dist/", "/build/"]):
        score -= 0.15

    digits = sum(1 for c in name if c.isdigit())
    if len(name) > 0 and digits / len(name) > 0.6:
        score -= 0.10

    return score


def _is_noise(url: str) -> bool:
    lower = url.lower()
    return any(w in lower for w in _NOISE_WORDS)


def _load_x_cookies() -> list:
    """Load X/Twitter session cookies from environment variables.

    Expects these env vars (matching browser cookie names):
      X_AUTH_TOKEN, X_CT0, X_TWID, X_GUEST_ID, X_GUEST_ID_ADS,
      X_GUEST_ID_MARKETING, X_KDT, X_PERSONALIZATION_ID,
      X_CF_CLEARANCE, X_CF_BM, X_CUID, X_TWPID, X_GSTATE, X_LANG
    """
    import os as _os
    cookies: list = []
    cookie_map = {
        "X_AUTH_TOKEN": "auth_token",
        "X_CT0": "ct0",
        "X_TWID": "twid",
        "X_GUEST_ID": "guest_id",
        "X_GUEST_ID_ADS": "guest_id_ads",
        "X_GUEST_ID_MARKETING": "guest_id_marketing",
        "X_KDT": "kdt",
        "X_PERSONALIZATION_ID": "personalization_id",
        "X_CF_CLEARANCE": "cf_clearance",
        "X_CF_BM": "__cf_bm",
        "X_CUID": "__cuid",
        "X_TWPID": "_twpid",
        "X_GSTATE": "g_state",
        "X_LANG": "lang",
    }
    for env_var, cookie_name in cookie_map.items():
        val = _os.getenv(env_var, "")
        if val:
            cookies.append({
                "name": cookie_name,
                "value": val,
                "domain": ".x.com",
                "path": "/",
            })
    return cookies


def _abs_url(raw: str, base_url: str) -> str:
    try:
        url = urljoin(base_url, raw)
    except Exception:
        url = raw
    # Fix HTML entities and upgrade X image resolution.
    url = url.replace("&amp;", "&")
    if "pbs.twimg.com/media" in url and "name=small" in url:
        url = url.replace("name=small", "name=large")
    return url


def enrich_items_with_images(items: list, max_per_batch: int = 20) -> int:
    count = 0
    for item in items[:max_per_batch]:
        item_url = item.get("url", "") if isinstance(item, dict) else getattr(item, "url", "")
        if not item_url:
            continue
        img = extract_image(item_url)
        if img:
            if isinstance(item, dict):
                item["image_url"] = img
            else:
                item.image_url = img
            count += 1
    return count
