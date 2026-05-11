"""DeepSeekProvider construction-time validation tests.

We don't hit the network. The point is to verify the policy:
  - missing API key → ModelUnavailable
  - unknown provider name → ModelUnavailable
  - factory wires deepseek correctly when env is set (model check skipped via
    monkeypatched stub client)
"""

import pytest

from agent.llm import build_provider
from agent.llm.base import ModelUnavailable


def test_deepseek_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ModelUnavailable):
        build_provider("deepseek", model="deepseek-v4-pro")


def test_unknown_provider_raises():
    with pytest.raises(ModelUnavailable):
        build_provider("not-a-real-provider", model="x")


def test_deepseek_unknown_model_rejected(monkeypatch):
    """If models.list() doesn't contain our model, construction must fail."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    class _Model:
        def __init__(self, mid):
            self.id = mid

    class _ModelList:
        def list(self):
            class _R:
                data = [_Model("deepseek-chat"), _Model("deepseek-reasoner")]

            return _R()

    class _StubClient:
        def __init__(self, **kwargs):
            self.models = _ModelList()

    import agent.llm.deepseek_provider as dp

    monkeypatch.setattr(dp, "DEFAULT_BASE_URL", "https://example.invalid")
    # Patch OpenAI inside deepseek_provider
    import openai as _openai

    monkeypatch.setattr(_openai, "OpenAI", _StubClient)

    with pytest.raises(ModelUnavailable):
        build_provider("deepseek", model="deepseek-v4-pro")


def test_deepseek_known_model_accepted(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    class _Model:
        def __init__(self, mid):
            self.id = mid

    class _ModelList:
        def list(self):
            class _R:
                data = [_Model("deepseek-v4-pro")]

            return _R()

    class _StubClient:
        def __init__(self, **kwargs):
            self.models = _ModelList()

    import openai as _openai

    monkeypatch.setattr(_openai, "OpenAI", _StubClient)

    provider = build_provider("deepseek", model="deepseek-v4-pro")
    assert provider.name == "deepseek"
    assert provider.model == "deepseek-v4-pro"


def test_deepseek_skip_model_check(monkeypatch):
    """skip_model_check escape hatch must skip models.list() entirely."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    class _StubClient:
        def __init__(self, **kwargs):
            class _ML:
                def list(self_inner):
                    raise RuntimeError("listing should not happen when skipped")

            self.models = _ML()

    import openai as _openai

    monkeypatch.setattr(_openai, "OpenAI", _StubClient)

    provider = build_provider(
        "deepseek", model="deepseek-v4-pro", skip_model_check=True
    )
    assert provider.model == "deepseek-v4-pro"
