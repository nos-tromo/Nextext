"""Tests for the inference client configuration helpers."""

import io
from typing import Any, ClassVar

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
            choices: ClassVar[list[Any]] = [_Choice()]

        return _Resp()


class _RecordingClient:
    """Minimal OpenAI client stub exposing a recording completions endpoint."""

    def __init__(self, completions: _RecordingCompletions) -> None:
        class _Chat:
            def __init__(self, c: _RecordingCompletions) -> None:
                self.completions = c

        self.chat = _Chat(completions)


def _install_recording_client(monkeypatch: pytest.MonkeyPatch, pipeline: InferencePipeline) -> _RecordingCompletions:
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
def test_inference_pipeline_provider_all_three(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
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


# ---------------------------------------------------------------------------
# load_prompt locale resolution
# ---------------------------------------------------------------------------


def test_load_prompt_defaults_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no NEXTEXT_RESPONSE_LANGUAGE set, prompts come from the en/ locale.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("NEXTEXT_RESPONSE_LANGUAGE", raising=False)
    pipeline = InferencePipeline()

    assert "English" in pipeline.load_prompt("system")


def test_load_prompt_selects_german_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEXTEXT_RESPONSE_LANGUAGE=de selects the de/ system and summary prompts.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("NEXTEXT_RESPONSE_LANGUAGE", "de")
    pipeline = InferencePipeline()

    assert "deutscher Sprache" in pipeline.load_prompt("system")
    assert "Zusammenfassung" in pipeline.load_prompt("summary")


def test_post_init_loads_localized_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cached sys_prompt reflects the active locale at construction time.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("NEXTEXT_RESPONSE_LANGUAGE", "de")
    pipeline = InferencePipeline()

    assert "deutscher Sprache" in pipeline.sys_prompt


def test_load_prompt_falls_back_to_english_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """A keyword missing from the active locale falls back to en/ and warns.

    German ships no ``translation.txt``, so requesting it under ``de`` must
    return the English template and log a fallback warning.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    from loguru import logger

    monkeypatch.setenv("NEXTEXT_RESPONSE_LANGUAGE", "de")
    pipeline = InferencePipeline()

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        translation_prompt = pipeline.load_prompt("translation")
    finally:
        logger.remove(handler_id)

    assert "{SOURCE_LANG}" in translation_prompt
    log_output = sink.getvalue()
    assert "translation" in log_output
    assert "falling back" in log_output


def test_load_prompt_unknown_keyword_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A keyword absent from both the active and fallback locales raises.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("NEXTEXT_RESPONSE_LANGUAGE", raising=False)
    pipeline = InferencePipeline()

    with pytest.raises(FileNotFoundError):
        pipeline.load_prompt("does_not_exist")


# ---------------------------------------------------------------------------
# call_vision — multimodal content parts
# ---------------------------------------------------------------------------


def test_call_vision_sends_image_then_text_content_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """call_vision must send an image_url data URI part ahead of the instruction.

    Image-first ordering matches what document/vision model clients use, and
    the data URI is the only shape an OpenAI-compatible endpoint accepts for
    inline bytes.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("TEXT_MODEL", "vlm")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_vision(prompt="describe this", images=[b"\xff\xd8jpegbytes"])

    content = completions.calls[0]["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["text"] == "describe this"
    assert completions.calls[0]["messages"][-1]["role"] == "user"


def test_call_vision_encodes_the_actual_image_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The data URI payload must decode back to the exact bytes handed in."""
    import base64

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("TEXT_MODEL", "vlm")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)
    payload = b"\xff\xd8\x00binary\xfffra\x00me"

    pipeline.call_vision(prompt="p", images=[payload])

    url = completions.calls[0]["messages"][-1]["content"][0]["image_url"]["url"]
    assert base64.b64decode(url.split(",", 1)[1]) == payload


def test_call_vision_sends_every_image_before_the_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple frames are all attached, with the text instruction last."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("TEXT_MODEL", "vlm")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_vision(prompt="p", images=[b"one", b"two", b"three"])

    content = completions.calls[0]["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "image_url", "text"]


def test_call_vision_honours_custom_mime_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-supplied MIME type is what lands in the data URI."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("TEXT_MODEL", "vlm")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_vision(prompt="p", images=[b"png"], mime_type="image/png")

    url = completions.calls[0]["messages"][-1]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_call_vision_includes_system_prompt_and_maps_num_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """call_vision shares call_model's system-role and max_tokens conventions."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("TEXT_MODEL", "vlm")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_vision(prompt="p", images=[b"x"], num_predict=160, system_prompt="be terse")

    kwargs = completions.calls[0]
    assert kwargs["messages"][0] == {"role": "system", "content": "be terse"}
    assert kwargs["max_tokens"] == 160
    assert kwargs["model"] == "vlm"


def test_call_vision_omits_system_message_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """include_system_prompt=False sends a lone user message."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("TEXT_MODEL", "vlm")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_vision(prompt="p", images=[b"x"], include_system_prompt=False)

    assert [m["role"] for m in completions.calls[0]["messages"]] == ["user"]


def test_call_vision_forwards_think_via_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tri-state think field reaches the provider the same way as for text."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("TEXT_MODEL", "vlm")
    pipeline = InferencePipeline()
    completions = _install_recording_client(monkeypatch, pipeline)

    pipeline.call_vision(prompt="p", images=[b"x"], think=False)

    assert completions.calls[0]["extra_body"] == {"think": False}


def test_call_vision_requires_at_least_one_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty image list is a caller bug, not a request to send text-only."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("TEXT_MODEL", "vlm")
    pipeline = InferencePipeline()
    _install_recording_client(monkeypatch, pipeline)

    with pytest.raises(ValueError):
        pipeline.call_vision(prompt="p", images=[])


def test_call_vision_raises_when_provider_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable provider fails loudly; the caller decides to degrade."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("TEXT_MODEL", "vlm")
    pipeline = InferencePipeline()
    _install_recording_client(monkeypatch, pipeline)
    monkeypatch.setattr(pipeline, "get_health", lambda: False)

    with pytest.raises(RuntimeError):
        pipeline.call_vision(prompt="p", images=[b"x"])


# ---------------------------------------------------------------------------
# frame_caption prompt (visual context)
# ---------------------------------------------------------------------------


def test_frame_caption_prompt_loads_in_english(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caption instruction must exist in the fallback locale.

    ``load_prompt`` raises when neither the localized nor the English file is
    present, which would silently disable visual context for every job.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("RESPONSE_LANGUAGE", raising=False)
    prompt = InferencePipeline().load_prompt("frame_caption")
    assert prompt.strip()
    assert "{" not in prompt  # a plain instruction, not a format template


def test_frame_caption_prompt_is_localized_for_german(monkeypatch: pytest.MonkeyPatch) -> None:
    """German deployments caption in German, so the summary stays one language."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    english = "Describe what is visible"
    assert english not in InferencePipeline().load_prompt("frame_caption")


@pytest.mark.parametrize("language", ["en", "de"])
def test_summary_prompt_mentions_visual_context(monkeypatch: pytest.MonkeyPatch, language: str) -> None:
    """Both summary templates must tell the model how to use a visual block.

    Without the instruction the model tends to enumerate the frames verbatim
    instead of weaving what was shown into the prose summary.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars.
        language (str): Prompt locale under test.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("RESPONSE_LANGUAGE", language)
    prompt = InferencePipeline().load_prompt("summary")
    assert prompt.count("{text}") == 1
    assert prompt.count("{") == 1  # str.format would choke on any other brace
    keyword = "Visual context" if language == "en" else "Visueller Kontext"
    assert keyword in prompt
