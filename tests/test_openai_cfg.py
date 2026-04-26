"""Tests for the inference client configuration helpers."""

from typing import Any

import pytest

from nextext.core import openai_cfg
from nextext.core.openai_cfg import InferencePipeline
from nextext.utils.env_cfg import load_inference_env


class _RecordingCompletions:
    """Capture chat completion request kwargs for assertion."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        class _Msg:
            content = "ok"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _RecordingClient:
    """Minimal OpenAI client stub exposing a recording completions endpoint."""

    def __init__(self, completions: _RecordingCompletions) -> None:
        class _Chat:
            def __init__(self, c: _RecordingCompletions) -> None:
                self.completions = c

        self.chat = _Chat(completions)


def _install_recording_client(
    monkeypatch: pytest.MonkeyPatch, pipeline: InferencePipeline
) -> _RecordingCompletions:
    """Replace the pipeline's OpenAI client with a recording stub and bypass health."""
    completions = _RecordingCompletions()
    monkeypatch.setattr(pipeline, "_client", _RecordingClient(completions))
    monkeypatch.setattr(pipeline, "get_health", lambda: True)
    return completions


def test_client_uses_configured_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the client is constructed with OPENAI_API_KEY and OPENAI_API_BASE.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables and module attributes.
    """
    recorded_kwargs: dict[str, str] = {}

    class DummyClient:
        """Minimal OpenAI client stub for constructor assertions."""

        def __init__(self, **kwargs: str) -> None:
            """Store the client constructor arguments for inspection.

            Args:
                **kwargs (str): Arbitrary client keyword arguments.
            """
            recorded_kwargs.update(kwargs)

    monkeypatch.setattr(openai_cfg, "OpenAIClient", DummyClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://inference-server/v1")
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")

    pipeline = InferencePipeline()
    _ = pipeline.client

    assert recorded_kwargs["api_key"] == "test-key"
    assert recorded_kwargs["base_url"] == "http://inference-server/v1"


def test_client_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that accessing the client raises when OPENAI_API_KEY is not set.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    pipeline = InferencePipeline()

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _ = pipeline.api_key


def test_base_url_strips_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that base_url strips a trailing slash from OPENAI_API_BASE.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("OPENAI_API_BASE", "http://inference-server/v1/")

    pipeline = InferencePipeline()

    assert pipeline.base_url == "http://inference-server/v1"


def test_default_model_uses_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that TEXT_MODEL is used for the default model.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")

    pipeline = InferencePipeline()

    assert pipeline.default_model == "llama3.1:8b"


def test_default_model_requires_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that text analysis requires TEXT_MODEL.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("TEXT_MODEL", raising=False)

    pipeline = InferencePipeline()

    with pytest.raises(RuntimeError, match="TEXT_MODEL"):
        _ = pipeline.default_model


def test_translation_model_requires_translation_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that translation requires TRANSLATION_MODEL.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    monkeypatch.delenv("TRANSLATION_MODEL", raising=False)

    pipeline = InferencePipeline()

    with pytest.raises(RuntimeError, match="TRANSLATION_MODEL"):
        _ = pipeline.translation_model


def test_call_model_includes_system_message_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """call_model must send a system role by default for backward compatibility."""
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_model(prompt="hello")

    assert len(completions.calls) == 1
    messages = completions.calls[0]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == pipeline.sys_prompt
    assert messages[1] == {"role": "user", "content": "hello"}


def test_call_model_custom_system_prompt_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit system_prompt overrides the default without changing structure."""
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_model(prompt="hi", system_prompt="be terse")

    messages = completions.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[1] == {"role": "user", "content": "hi"}


def test_call_model_omits_system_message_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """include_system_prompt=False must produce a single user-only message."""
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_model(prompt="payload", include_system_prompt=False)

    messages = completions.calls[0]["messages"]
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "payload"}
    assert all(m.get("role") != "system" for m in messages)


def test_inference_pipeline_provider_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset INFERENCE_PROVIDER falls back to ollama."""
    monkeypatch.delenv("INFERENCE_PROVIDER", raising=False)

    pipeline = InferencePipeline()

    assert pipeline.provider == "ollama"


@pytest.mark.parametrize("value", ["ollama", "vllm", "openai"])
def test_inference_pipeline_provider_all_three(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """All three valid providers round-trip through the env var."""
    monkeypatch.setenv("INFERENCE_PROVIDER", value)

    pipeline = InferencePipeline()

    assert pipeline.provider == value


def test_inference_pipeline_provider_lowercases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uppercase INFERENCE_PROVIDER resolves to its canonical lowercase form."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "VLLM")

    pipeline = InferencePipeline()

    assert pipeline.provider == "vllm"


def test_inference_pipeline_provider_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown INFERENCE_PROVIDER values fall back to ollama."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "garbage")

    pipeline = InferencePipeline()

    assert pipeline.provider == "ollama"


def test_load_inference_env_returns_dataclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_inference_env returns a frozen InferenceConfig with the resolved provider."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")

    cfg = load_inference_env()

    assert cfg.provider == "vllm"


# ---------------------------------------------------------------------------
# think parameter — call_model forwarding
# ---------------------------------------------------------------------------


def test_call_model_forwards_think_false_via_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """call_model with think=False must attach extra_body={"think": False}.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables and pipeline internals.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    monkeypatch.delenv("OLLAMA_THINK", raising=False)
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_model("hi", think=False)

    assert completions.calls[0]["extra_body"] == {"think": False}


def test_call_model_forwards_think_true_via_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """call_model with think=True must attach extra_body={"think": True}.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables and pipeline internals.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    monkeypatch.delenv("OLLAMA_THINK", raising=False)
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_model("hi", think=True)

    assert completions.calls[0]["extra_body"] == {"think": True}


def test_call_model_omits_extra_body_when_think_none_and_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No think arg and OLLAMA_THINK unset must leave extra_body absent entirely.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables and pipeline internals.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    monkeypatch.delenv("OLLAMA_THINK", raising=False)
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_model("hi")

    assert "extra_body" not in completions.calls[0]


def test_call_model_uses_ollama_think_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OLLAMA_THINK=0 with no per-call think arg must produce extra_body={"think": False}.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables and pipeline internals.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_THINK", "0")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_model("hi")

    assert completions.calls[0]["extra_body"] == {"think": False}


def test_call_model_per_call_think_overrides_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-call think=True must win over OLLAMA_THINK=0 env default.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables and pipeline internals.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_THINK", "0")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_model("hi", think=True)

    assert completions.calls[0]["extra_body"] == {"think": True}


def test_call_model_preserves_existing_kwargs_when_think_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All standard request kwargs survive alongside extra_body when think is set.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables and pipeline internals.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    monkeypatch.delenv("OLLAMA_THINK", raising=False)
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_model(
        "payload",
        temperature=0.5,
        seed=7,
        num_predict=128,
        top_p=0.9,
        stop=["END"],
        think=False,
    )

    recorded = completions.calls[0]
    assert recorded["temperature"] == 0.5
    assert recorded["seed"] == 7
    assert recorded["max_tokens"] == 128
    assert recorded["top_p"] == 0.9
    assert recorded["stop"] == ["END"]
    assert recorded["extra_body"] == {"think": False}
