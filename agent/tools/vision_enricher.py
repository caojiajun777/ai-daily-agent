"""Vision enrichment using Qwen VL models via DashScope native API.

Describes images from source articles/tweets to provide richer context
for the writer LLM. Uses DashScope multimodal-generation endpoint
(NOT the OpenAI-compatible endpoint, which only supports text).

Native API: POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import httpx

_DASHSCOPE_MM_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        env_path = os.path.normpath(env_path)
        if os.path.exists(env_path):
            load_dotenv(env_path)
    except ImportError:
        pass


def describe_image(
    image_url: str,
    *,
    model: str = "qwen-vl-plus",
    max_tokens: int = 200,
    timeout: float = 45.0,
    title: str = "",
    article_text: str = "",
) -> str:
    """Get a relevant image description, or empty string if image adds no value.

    Returns empty string for:
      - Logos, avatars, generic stock photos, QR codes
      - Images unrelated to the article/tweet topic
      - Failed API calls

    Returns a natural Chinese sentence if the image provides news value:
      - Screenshots with key data or announcements
      - Product photos showing new features
      - Charts/diagrams with findings
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        _try_load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        return ""

    # Build prompt with context so Qwen can judge relevance.
    context_str = ""
    if title:
        context_str += f"这篇新闻的标题是：{title}。"
    if article_text:
        context_str += f"新闻内容是关于：{article_text[:200]}。"

    prompt = (
        f"你是一个AI新闻编辑，正在审核一张配图。{context_str}\n\n"
        "首先判断这张图片是否是以下类型之一（如果有价值则描述，无价值则回复SKIP）：\n"
        "- 产品截图/UI界面/功能演示 → 描述展示了什么功能\n"
        "- 数据图表/基准测试结果 → 提取关键数字和结论\n"
        "- 官方公告/推文截图 → 提取核心信息\n"
        "- 模型架构图/技术示意图 → 简述核心思想\n\n"
        "如果图片是以下类型，直接回复 SKIP：\n"
        "- Logo/头像/二维码\n"
        "- 通用配图/风景照/人物照\n"
        "- 与新闻主题无关的图片\n"
        "- 模糊到无法辨认的图片\n\n"
        "有价值的请用一句中文描述（不超过40字），融入新闻语境。"
        "不要加\"这张图片展示了\"之类的废话，直接给信息。"
    )

    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": image_url},
                        {"text": prompt},
                    ],
                }
            ]
        },
    }

    try:
        resp = httpx.post(
            _DASHSCOPE_MM_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            content=json.dumps(payload, ensure_ascii=False),
            timeout=timeout,
        )
        if resp.status_code != 200:
            return ""

        data = resp.json()
        choices = data.get("output", {}).get("choices", [])
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content", [])
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
            text = " ".join(texts).strip()
        elif isinstance(content, str):
            text = content.strip()
        else:
            return ""

        # Filter out SKIP or empty/boring responses.
        if not text or "SKIP" in text.upper() or len(text) < 5:
            return ""
        # Filter generic descriptions that add no news value.
        generic_starts = ("这是一张", "图片显示了", "图中是", "这是", "这张图片", "图中")
        if any(text.startswith(g) for g in generic_starts):
            return ""

        return text
    except Exception:
        return ""


def enrich_items(
    items: List[Dict],
    *,
    max_per_batch: int = 5,
) -> int:
    """Add image descriptions to items that have image_url set."""
    count = 0
    for item in items[:max_per_batch]:
        img_url = item.get("image_url", "")
        if not img_url:
            continue
        desc = describe_image(img_url)
        if desc:
            prev = item.get("summary", "")
            item["summary"] = f"{prev} [配图信息: {desc}]"
            count += 1
    return count
