"""DeepSeek provider via OpenAI-compatible API.

DeepSeek exposes an OpenAI-compatible chat completions endpoint. We use the
official ``openai`` SDK pointed at DeepSeek's base URL, which is the
recommended integration path and lets us reuse the same code path for any
OpenAI-compatible provider in the future.

Construction-time invariants:
  - ``DEEPSEEK_API_KEY`` must be set
  - the requested model id must exist in the provider's ``models.list()``
    response. If not, we raise ``ModelUnavailable``. We never silently pick a
    different model (per project policy).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from agent.harness.trace import estimate_tokens
from agent.llm.base import (
    LLMCallFailed,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ModelUnavailable,
)


DEFAULT_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider:
    name = "deepseek"

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout_s: int = 60,
        max_retries: int = 2,
        # Escape hatch for environments where listing models is not
        # available (corporate gateways). Off by default — real runs should
        # always validate.
        skip_model_check: bool = False,
    ) -> None:
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ModelUnavailable(
                "DEEPSEEK_API_KEY is not set; refusing to construct DeepSeekProvider"
            )
        base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ModelUnavailable(
                "the 'openai' package is required for DeepSeekProvider"
            ) from e
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=request_timeout_s
        )
        self.model = model
        self._max_retries = max_retries
        if not skip_model_check:
            self._verify_model()

    def _verify_model(self) -> None:
        try:
            resp = self._client.models.list()
            available = {m.id for m in resp.data}
        except Exception as e:
            raise ModelUnavailable(
                f"failed to list DeepSeek models for validation: {e}"
            ) from e
        if self.model not in available:
            raise ModelUnavailable(
                f"requested model '{self.model}' is not available on DeepSeek "
                f"(available: {sorted(available)}). Refusing to silently downgrade."
            )

    def complete(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        last_err: Optional[Exception] = None
        joined_in = "\n".join(m.content for m in messages)
        for attempt in range(1, self._max_retries + 2):
            t0 = time.time()
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": m.role, "content": m.content} for m in messages
                    ],
                    "temperature": temperature,
                    "max_tokens": max_output_tokens,
                }
                # deepseek-v4-pro is a reasoning model and does not support
                # response_format — passing it causes an empty completion.
                # JSON output is enforced via the system prompt instead.
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                latency_ms = int((time.time() - t0) * 1000)
                return LLMResponse(
                    text=text,
                    model=self.model,
                    provider=self.name,
                    latency_ms=latency_ms,
                    input_tokens_est=estimate_tokens(joined_in),
                    output_tokens_est=estimate_tokens(text),
                    raw={"id": getattr(resp, "id", None)},
                )
            except Exception as e:
                last_err = e
                continue
        raise LLMCallFailed(
            f"DeepSeek call failed after {self._max_retries + 1} attempts: {last_err}"
        )
