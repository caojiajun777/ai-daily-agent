"""Qwen / Tongyi Qianwen LLM provider via Alibaba Cloud Bailian (DashScope).

OpenAI-compatible API endpoint. Supports:
  - Deep thinking mode (enable_thinking=True in extra_body)
  - JSON structured output (response_format supported)
  - Streaming or non-streaming completion
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from agent.harness.trace import estimate_tokens
from agent.llm.base import LLMCallFailed, LLMMessage, LLMProvider, LLMResponse, ModelUnavailable


class QwenProvider:
    name = "qwen"

    def __init__(
        self,
        model: str = "qwen3.6-plus",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout_s: int = 120,
        max_retries: int = 1,
        skip_model_check: bool = False,
    ) -> None:
        self.model = model
        self._api_key: Optional[str] = None
        self._base_url: str = ""
        self._max_retries = max_retries
        self._request_timeout_s = request_timeout_s

        api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ModelUnavailable(
                "DASHSCOPE_API_KEY is not set. Get one from "
                "https://bailian.console.aliyun.com/"
            )
        self._api_key = api_key
        self._base_url = base_url or os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self._client: Any = None
        self._init_client(skip_model_check)

    def _init_client(self, skip_model_check: bool) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ModelUnavailable(
                "the 'openai' package is required for QwenProvider"
            ) from e
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._request_timeout_s,
        )
        if not skip_model_check:
            self._verify_model()

    def _verify_model(self) -> None:
        try:
            resp = self._client.models.list()
            available = {m.id for m in resp.data}
        except Exception as e:
            raise ModelUnavailable(
                f"failed to list Qwen models: {e}"
            ) from e
        if self.model not in available:
            raise ModelUnavailable(
                f"requested model '{self.model}' is not available "
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
                # Qwen supports response_format for JSON output.
                if response_format:
                    kwargs["response_format"] = response_format

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
            f"Qwen call failed after {self._max_retries + 1} attempts: {last_err}"
        )
