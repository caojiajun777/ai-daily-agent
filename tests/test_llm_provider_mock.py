from agent.llm import LLMMessage, build_provider
from agent.llm.mock_provider import MockLLMProvider


def test_mock_provider_basic_call():
    provider = build_provider("mock", model="mock-x")
    assert provider.name == "mock"
    assert provider.model == "mock-x"
    resp = provider.complete([LLMMessage(role="user", content="hello")])
    assert resp.text
    assert resp.provider == "mock"
    assert resp.model == "mock-x"
    assert resp.input_tokens_est >= 1
    assert resp.output_tokens_est >= 1


def test_mock_provider_custom_responder():
    provider = MockLLMProvider(
        model="mock-y", responder=lambda msgs: "fixed-output"
    )
    resp = provider.complete([LLMMessage(role="user", content="anything")])
    assert resp.text == "fixed-output"
