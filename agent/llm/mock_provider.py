"""Mock provider for tests, CI and offline replay.

Deterministic by construction: response text is a function of the last user
message and the configured ``script``. No network, no env vars, no surprises.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from agent.harness.trace import estimate_tokens
from agent.llm.base import LLMMessage, LLMProvider, LLMResponse, ModelUnavailable


class MockLLMProvider:
    name = "mock"

    def __init__(
        self,
        model: str = "mock-model",
        responder: Optional[Callable[[List[LLMMessage]], str]] = None,
    ) -> None:
        if not model:
            raise ModelUnavailable("mock provider requires a non-empty model id")
        self.model = model
        self._responder = responder or _default_responder

    def complete(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        t0 = time.time()
        text = self._responder(messages)
        latency_ms = int((time.time() - t0) * 1000)
        joined_in = "\n".join(m.content for m in messages)
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            input_tokens_est=estimate_tokens(joined_in),
            output_tokens_est=estimate_tokens(text),
            raw={"mock": True},
        )


def _default_responder(messages: List[LLMMessage]) -> str:
    # Deterministic 6-section draft. URLs are extracted from the user message
    # so the critic's hallucination check passes.
    import json as _json
    import re as _re

    last = messages[-1].content if messages else ""
    # Try to find the items_json block and extract real URLs / titles.
    urls: List[str] = _re.findall(r'"url":\s*"(https?://[^"]+)"', last)
    titles: List[str] = _re.findall(r'"title":\s*"([^"]+)"', last)
    sources: List[str] = _re.findall(r'"source_id":\s*"([^"]+)"', last)

    # Build at least 6 slots; repeat the available items if fewer than 6.
    def _slot(i: int) -> dict:
        url = urls[i % len(urls)] if urls else f"https://example.com/{i}"
        title = titles[i % len(titles)] if titles else f"Mock item {i+1}"
        src = sources[i % len(sources)] if sources else "mock"
        return {"title": f"#{i+1} {title[:40]}", "summary": f"mock summary {i+1}", "url": url, "source": src}

    section_names = ["今日头条", "模型前沿", "工具与开源", "论文精选", "产品落地", "业界风向"]
    counter = [0]

    def _section(name: str) -> dict:
        item = _slot(counter[0])
        counter[0] += 1
        return {"heading": name, "items": [item]}

    payload = {
        "date": "1970-01-01",
        "title": "AI 早报 1970-01-01",
        "overview": "今日没有特别重大动态，各模型进展平稳。",
        "sections": [_section(n) for n in section_names],
    }
    return _json.dumps(payload, ensure_ascii=False)
