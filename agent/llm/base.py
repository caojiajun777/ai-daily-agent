"""LLMProvider abstraction.

Agents NEVER import an SDK directly. They receive an ``LLMProvider`` from the
runtime and call ``complete(messages, ...) -> LLMResponse``. Providers are
responsible for:

  - resolving and validating the requested model at construction time
  - executing the request with retries and timeout
  - reporting concrete failures via ``LLMCallFailed``
  - never silently downgrading to a different model

Switching backends (DeepSeek, Anthropic, OpenAI, OpenAI-compatible) only
requires a new ``LLMProvider`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol


class LLMCallFailed(RuntimeError):
    """Raised after all retries are exhausted."""


class ModelUnavailable(RuntimeError):
    """Raised at provider construction when the requested model is not usable.

    Per project policy this is *fatal*: providers must not silently fall back
    to a different model.
    """


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    latency_ms: int
    input_tokens_est: int
    output_tokens_est: int
    raw: Optional[Dict[str, object]] = None


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse: ...
