import os

from agent.harness.trace import Tracer, prompt_hash, estimate_tokens


def test_trace_writes_jsonl(tmp_path):
    p = tmp_path / "t.jsonl"
    tr = Tracer(str(p), run_id="r-1")
    tr.log("foo", a=1)
    tr.log_stage("collect", "ok", count=5)
    tr.log_llm_call(
        provider="mock",
        model="m",
        prompt="hello",
        output="world",
        latency_ms=12,
        status="ok",
        stage="write",
    )
    events = tr.read_all()
    assert len(events) == 3
    assert events[0]["event"] == "foo"
    assert events[0]["a"] == 1
    assert events[1]["stage"] == "collect"
    assert events[1]["status"] == "ok"
    assert events[2]["event"] == "llm_call"
    assert events[2]["provider"] == "mock"
    assert events[2]["prompt_hash"] == prompt_hash("hello")
    assert events[2]["input_tokens_est"] >= 1
    assert events[2]["output_tokens_est"] >= 1


def test_estimate_tokens_handles_cjk():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") >= 1
    assert estimate_tokens("你好世界") >= 1
