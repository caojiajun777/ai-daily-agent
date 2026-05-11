"""Tool registry with JSON-Schema input validation.

Tools are simple typed callables exposed to agents. Each tool declares an input
schema (JSON Schema dict) so the harness can validate arguments *before* the
tool runs. This is the foundation of a safe tool-use loop: even if an LLM emits
a malformed tool call, validation catches it instead of crashing the runtime.

The MVP uses tools internally (sources, persistence) rather than via free-form
LLM tool calls, but the same registry will plug into a function-calling loop in
later phases without API changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from jsonschema import Draft202012Validator, ValidationError


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]

    def to_function_schema(self) -> Dict[str, Any]:
        # Shape suitable for OpenAI/DeepSeek function-calling, useful later.
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolError(RuntimeError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> ToolSpec:
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        # Fail fast on malformed schemas.
        Draft202012Validator.check_schema(input_schema)
        spec = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def call(self, name: str, args: Optional[Dict[str, Any]] = None) -> Any:
        spec = self.get(name)
        args = args or {}
        try:
            Draft202012Validator(spec.input_schema).validate(args)
        except ValidationError as e:
            raise ToolError(f"invalid args for {name}: {e.message}") from e
        return spec.handler(**args)
