import pytest

from agent.harness.tools import ToolError, ToolRegistry


def test_register_and_call_ok():
    reg = ToolRegistry()
    reg.register(
        "echo",
        "echo string",
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        lambda text: text.upper(),
    )
    assert "echo" in reg.names()
    out = reg.call("echo", {"text": "hi"})
    assert out == "HI"


def test_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolError):
        reg.call("nope", {})


def test_invalid_args_rejected():
    reg = ToolRegistry()
    reg.register(
        "add",
        "add two ints",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        lambda a, b: a + b,
    )
    with pytest.raises(ToolError):
        reg.call("add", {"a": 1})  # missing b
    with pytest.raises(ToolError):
        reg.call("add", {"a": "x", "b": 2})  # wrong type


def test_duplicate_registration_rejected():
    reg = ToolRegistry()
    reg.register("t", "d", {"type": "object"}, lambda: None)
    with pytest.raises(ValueError):
        reg.register("t", "d", {"type": "object"}, lambda: None)


def test_function_schema_export():
    reg = ToolRegistry()
    spec = reg.register(
        "ping",
        "ping",
        {"type": "object", "properties": {}},
        lambda: "pong",
    )
    fn = spec.to_function_schema()
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "ping"
