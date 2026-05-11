"""Stub for a generic OpenAI-compatible provider.

The DeepSeek path already covers OpenAI-compatible APIs; this module exists as
a clear extension point. Reserved for later.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from agent.llm.base import LLMMessage, LLMResponse, ModelUnavailable


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self, model: str, **kwargs) -> None:
        raise ModelUnavailable(
            "OpenAICompatibleProvider is not implemented in MVP. "
            "Use provider='deepseek' or provider='mock'."
        )

    def complete(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:  # pragma: no cover - never reached
        raise ModelUnavailable("OpenAICompatibleProvider is not implemented in MVP.")
