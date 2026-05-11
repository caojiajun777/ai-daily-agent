"""Provider factory.

Single entry point used by the orchestrator. New providers register here.
"""

from __future__ import annotations

from typing import Optional

from agent.llm.base import LLMProvider, ModelUnavailable


def build_provider(name: str, model: Optional[str] = None, **kwargs) -> LLMProvider:
    name = (name or "").lower()
    if name == "mock":
        from agent.llm.mock_provider import MockLLMProvider

        return MockLLMProvider(model=model or "mock-model", **kwargs)
    if name == "deepseek":
        from agent.llm.deepseek_provider import DeepSeekProvider

        return DeepSeekProvider(model=model or "deepseek-v4-pro", **kwargs)
    if name == "anthropic":
        from agent.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model or "claude-sonnet-4-6", **kwargs)
    if name == "openai_compatible":
        from agent.llm.openai_compatible_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider(model=model or "gpt-4o-mini", **kwargs)
    raise ModelUnavailable(
        f"unknown provider '{name}'. Valid: mock, deepseek, anthropic, openai_compatible"
    )
