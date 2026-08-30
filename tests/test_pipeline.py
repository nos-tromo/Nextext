"""Tests for the shared Nextext pipeline helpers."""

from pathlib import Path
from typing import Any, override

import httpx2
import openai
import pandas as pd
import pytest

from nextext import pipeline
from nextext.core.openai_cfg import InferencePipeline
from nextext.pipeline import transcript_txt_exports
from nextext.utils.env_cfg import SentenceRestoreConfig, WhisperClientConfig


@pytest.fixture
def disable_docker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable Docker environment detection by mocking the Path.exists method.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        """Mock the Path.exists method to simulate a non-Docker environment by returning False for the /.dockerenv path.

        Args:
            path (Path): The path to check for existence.

        Returns:
            bool: False if the path is /.dockerenv, otherwise the result of the original Path.exists method.
        """
        if path.as_posix() == "/.dockerenv":
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)


@pytest.fixture
def enable_docker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable Docker environment detection by mocking the Path.exists method.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if path.as_posix() == "/.dockerenv":
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)


def test_transcription_pipeline_invokes_transcriber_and_diarizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline diarizes with no speaker bounds and assigns speakers at the word level.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    monkeypatch.setattr(
        pipeline,
        "load_whisper_env",
        lambda: WhisperClientConfig(api_base="http://audio:8000/v1", api_key="k", model="test-model"),
    )

    class DummyTranscriber:
        """A dummy transcriber standing in for ExternalWhisperTranscriber.

        It records the parameters passed to it and whether its methods were called,
        allowing us to verify the pipeline's behavior.
        """

        instance: "DummyTranscriber"

        def __init__(self, **params: Any) -> None:
            """Initialize the dummy transcriber, recording the keyword params it was built with.

            Args:
                **params (Any): Keyword arguments forwarded by the pipeline.
            """
            DummyTranscriber.instance = self
            self.params = params
            self.transcription_called = False
            self.src_lang = "fr"
            self.skip_reason: str | None = None
            self.transcription_result: dict[str, Any] = {
                "segments": [{"start": 0.0, "end": 1.0, "text": "bonjour"}],
                "words": [{"word": "bonjour", "start": 0.0, "end": 1.0}],
            }

        def transcription(self) -> None:
            """Simulate transcription by flagging the call."""
            self.transcription_called = True

        def transcript_output(self) -> pd.DataFrame:
            """Return a dummy transcript DataFrame.

            Returns:
                pd.DataFrame: A DataFrame containing the dummy transcription text.
            """
            return pd.DataFrame({"text": ["bonjour"]})

    diarize_calls: dict[str, Any] = {}

    def fake_diarize_file(file_path: Path, **kwargs: Any) -> list[dict[str, Any]]:
        """Record that diarization ran (and any bound kwargs, expected to be none).

        Args:
            file_path (Path): Path forwarded by the pipeline.
            **kwargs (Any): Speaker-bound keyword arguments; expected to be empty
                since diarization now always auto-detects.

        Returns:
            list[dict[str, Any]]: A single fake speaker turn.
        """
        diarize_calls.update(kwargs)
        diarize_calls["called"] = True
        return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

    build_calls: list[Any] = []

    def fake_build(
        segments: list[dict[str, Any]],
        words: list[dict[str, Any]],
        turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Record the call and return the input segments unchanged.

        Args:
            segments (list[dict[str, Any]]): Whisper segments forwarded by the pipeline.
            words (list[dict[str, Any]]): Whisper words forwarded by the pipeline.
            turns (list[dict[str, Any]]): Canonicalized speaker turns forwarded by the pipeline.

        Returns:
            list[dict[str, Any]]: The input segments, unchanged.
        """
        build_calls.append((segments, words, turns))
        return segments

    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", DummyTranscriber)
    monkeypatch.setattr(pipeline, "diarize_file", fake_diarize_file)
    monkeypatch.setattr(pipeline, "canonicalize_speaker_labels", lambda turns: turns)
    monkeypatch.setattr(pipeline, "build_speaker_segments", fake_build)
    monkeypatch.setattr(
        pipeline, "load_sentence_restore_env", lambda: SentenceRestoreConfig(enabled=False, min_punct_ratio=0.01)
    )

    outcome = pipeline.transcription_pipeline(
        file_path=Path("/tmp/audio.wav"),
        src_lang="auto",
        diarize=True,
    )

    instance = DummyTranscriber.instance
    assert "n_speakers" not in instance.params  # no speaker-count is threaded any more
    assert instance.params["model_id"] == "test-model"
    assert instance.transcription_called is True
    assert diarize_calls.get("called") is True
    assert "max_speakers" not in diarize_calls and "num_speakers" not in diarize_calls  # auto-detect
    assert len(build_calls) == 1
    assert list(outcome.transcript["text"]) == ["bonjour"]
    assert outcome.src_lang == "fr"


def test_transcription_pipeline_falls_back_to_original_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``diarize=False`` request falls back to the source language and never diarizes.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    monkeypatch.setattr(
        pipeline,
        "load_whisper_env",
        lambda: WhisperClientConfig(api_base="http://audio:8000/v1", api_key="k", model="test-model"),
    )

    class DummyTranscriber:
        """A dummy transcriber that leaves src_lang unset to exercise the fallback path."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialise the dummy transcriber; args/kwargs accepted to match the real signature."""
            self.src_lang: str | None = None
            self.transcription_result: dict[str, Any] | None = None
            self.skip_reason: str | None = None

        def transcription(self) -> None:
            """Simulate transcription, populating a segment but leaving src_lang unset."""
            self.transcription_result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hola"}]}

        def transcript_output(self) -> pd.DataFrame:
            """Return a dummy transcript DataFrame.

            Returns:
                pd.DataFrame: A DataFrame containing the dummy transcription text.
            """
            return pd.DataFrame({"text": ["hola"]})

    def fail_diarize(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Fail if diarization is attempted when diarize=False.

        Args:
            *args (Any): Unused positional arguments.
            **kwargs (Any): Unused keyword arguments.

        Returns:
            list[dict[str, Any]]: Never returns; always fails the test.
        """
        pytest.fail("diarize_file should not be called when diarize=False")

    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", DummyTranscriber)
    monkeypatch.setattr(pipeline, "diarize_file", fail_diarize)

    outcome = pipeline.transcription_pipeline(
        file_path=Path("/tmp/audio.wav"),
        src_lang="es",
        diarize=False,
    )

    assert list(outcome.transcript["text"]) == ["hola"]
    assert outcome.src_lang == "es"


def test_transcription_pipeline_skips_diarization_for_empty_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty transcript skips diarization even when diarize=True is requested.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    monkeypatch.setattr(
        pipeline,
        "load_whisper_env",
        lambda: WhisperClientConfig(api_base="http://audio:8000/v1", api_key="k", model="test-model"),
    )

    class DummyTranscriber:
        """A dummy transcriber that yields an empty transcript."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialise the dummy transcriber; args/kwargs accepted to match the real signature."""
            self.src_lang = "en"
            self.transcription_result: dict[str, Any] | None = None
            self.skip_reason: str | None = None

        def transcription(self) -> None:
            """Simulate transcription that produced no segments (e.g. silent audio)."""
            self.transcription_result = {"segments": []}

        def transcript_output(self) -> pd.DataFrame:
            """Return an empty transcript DataFrame.

            Returns:
                pd.DataFrame: An empty-text DataFrame.
            """
            return pd.DataFrame({"text": []})

    def fail_diarize(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Fail if diarization is attempted for an empty transcript.

        Args:
            *args (Any): Unused positional arguments.
            **kwargs (Any): Unused keyword arguments.

        Returns:
            list[dict[str, Any]]: Never returns; always fails the test.
        """
        pytest.fail("diarize_file should not be called for an empty transcript")

    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", DummyTranscriber)
    monkeypatch.setattr(pipeline, "diarize_file", fail_diarize)

    outcome = pipeline.transcription_pipeline(
        file_path=Path("/tmp/audio.wav"),
        src_lang="en",
        diarize=True,
    )

    assert list(outcome.transcript["text"]) == []
    assert outcome.src_lang == "en"


class _RestorableTranscriber:
    """Transcriber stand-in with unpunctuated text and word timestamps."""

    def __init__(self, **params: Any) -> None:
        """Seed an unpunctuated one-segment result with words.

        Args:
            **params (Any): Ignored construction params.
        """
        self.src_lang = "ar"
        self.skip_reason: str | None = None
        self.transcription_result: dict[str, Any] = {
            "segments": [{"start": 0.0, "end": 6.0, "text": "a b c d e f"}],
            "words": [{"word": ch, "start": float(i), "end": float(i) + 0.5} for i, ch in enumerate("abcdef")],
        }

    def transcription(self) -> None:
        """No-op stand-in for the Whisper call."""

    def transcript_output(self) -> pd.DataFrame:
        """Return a one-row transcript frame.

        Returns:
            pd.DataFrame: A dummy transcript.
        """
        return pd.DataFrame({"text": ["x"]})


def _install_restorable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire a restorable transcriber and a no-network InferencePipeline.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr(
        pipeline, "load_whisper_env", lambda: WhisperClientConfig(api_base="http://a/v1", api_key="k", model="m")
    )
    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", _RestorableTranscriber)
    monkeypatch.setattr(pipeline, "InferencePipeline", lambda: object())


def test_transcription_pipeline_restores_when_low_punctuation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undiarized low-punctuation transcript → restoration runs with turns=None."""
    _install_restorable(monkeypatch)
    monkeypatch.setattr(
        pipeline, "load_sentence_restore_env", lambda: SentenceRestoreConfig(enabled=True, min_punct_ratio=0.01)
    )
    recorded: dict[str, Any] = {}

    def fake_restore(words: list[dict[str, Any]], turns: Any, inference_pipeline: Any) -> list[dict[str, Any]]:
        """Record the call and return one canned sentence segment.

        Args:
            words (list[dict[str, Any]]): Words forwarded by the pipeline.
            turns (Any): Speaker turns (or None) forwarded by the pipeline.
            inference_pipeline (Any): Inference client forwarded by the pipeline.

        Returns:
            list[dict[str, Any]]: A single restored segment.
        """
        recorded["called"] = True
        recorded["turns"] = turns
        return [{"start": 0.0, "end": 6.0, "text": "a b c d e f."}]

    monkeypatch.setattr(pipeline, "restore_sentence_segments", fake_restore)

    pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="ar", diarize=False)

    assert recorded["called"] is True
    assert recorded["turns"] is None


def test_transcription_pipeline_skips_restore_when_well_punctuated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A punctuated transcript is left alone (ratio above threshold)."""
    _install_restorable(monkeypatch)
    monkeypatch.setattr(
        pipeline, "load_sentence_restore_env", lambda: SentenceRestoreConfig(enabled=True, min_punct_ratio=0.01)
    )

    class _Punctuated(_RestorableTranscriber):
        """Restorable transcriber whose transcript is already punctuated."""

        def __init__(self, **params: Any) -> None:
            """Seed a punctuated segment (words inherited from the base).

            Args:
                **params (Any): Ignored construction params.
            """
            super().__init__(**params)
            self.transcription_result["segments"] = [{"start": 0.0, "end": 6.0, "text": "a. b. c. d. e. f."}]

    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", _Punctuated)

    def boom(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Fail if restoration is invoked on punctuated text.

        Args:
            *args (Any): Ignored.
            **kwargs (Any): Ignored.

        Returns:
            list[dict[str, Any]]: Never returns.
        """
        pytest.fail("restore_sentence_segments should not run on punctuated text")

    monkeypatch.setattr(pipeline, "restore_sentence_segments", boom)

    pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="ar", diarize=False)


def test_transcription_pipeline_restore_supersedes_build_when_diarized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diarized low-punctuation path uses restore (with turns), not build_speaker_segments."""
    _install_restorable(monkeypatch)
    monkeypatch.setattr(
        pipeline, "load_sentence_restore_env", lambda: SentenceRestoreConfig(enabled=True, min_punct_ratio=0.01)
    )
    monkeypatch.setattr(pipeline, "diarize_file", lambda fp: [{"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00"}])
    monkeypatch.setattr(pipeline, "canonicalize_speaker_labels", lambda turns: turns)

    def no_build(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Fail if build_speaker_segments runs when restoration supersedes it.

        Args:
            *args (Any): Ignored.
            **kwargs (Any): Ignored.

        Returns:
            list[dict[str, Any]]: Never returns.
        """
        pytest.fail("build_speaker_segments should not run when restoration supersedes it")

    monkeypatch.setattr(pipeline, "build_speaker_segments", no_build)
    recorded: dict[str, Any] = {}

    def fake_restore(words: list[dict[str, Any]], turns: Any, inference_pipeline: Any) -> list[dict[str, Any]]:
        """Record the turns passed to restoration and return a canned segment.

        Args:
            words (list[dict[str, Any]]): Words forwarded by the pipeline.
            turns (Any): Canonicalized speaker turns forwarded by the pipeline.
            inference_pipeline (Any): Inference client forwarded by the pipeline.

        Returns:
            list[dict[str, Any]]: A single speaker-labeled restored segment.
        """
        recorded["turns"] = turns
        return [{"start": 0.0, "end": 6.0, "text": "a b c d e f.", "speaker": "SPEAKER_00"}]

    monkeypatch.setattr(pipeline, "restore_sentence_segments", fake_restore)

    pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="ar", diarize=True)

    assert recorded["turns"] == [{"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00"}]


def test_transcription_pipeline_restore_failure_falls_back_to_build_speaker_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising restore_sentence_segments degrades to build_speaker_segments (fail-soft)."""
    _install_restorable(monkeypatch)
    monkeypatch.setattr(
        pipeline, "load_sentence_restore_env", lambda: SentenceRestoreConfig(enabled=True, min_punct_ratio=0.01)
    )
    monkeypatch.setattr(pipeline, "diarize_file", lambda fp: [{"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00"}])
    monkeypatch.setattr(pipeline, "canonicalize_speaker_labels", lambda turns: turns)

    def boom(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Raise to simulate a restoration failure exercising the fail-soft path.

        Args:
            *args (Any): Ignored.
            **kwargs (Any): Ignored.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("restoration exploded")

    monkeypatch.setattr(pipeline, "restore_sentence_segments", boom)

    build_calls: list[Any] = []

    def fake_build(
        segments: list[dict[str, Any]],
        words: list[dict[str, Any]],
        turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Record the call and return the input segments unchanged.

        Args:
            segments (list[dict[str, Any]]): Whisper segments forwarded by the pipeline.
            words (list[dict[str, Any]]): Whisper words forwarded by the pipeline.
            turns (list[dict[str, Any]]): Canonicalized speaker turns forwarded by the pipeline.

        Returns:
            list[dict[str, Any]]: The input segments, unchanged.
        """
        build_calls.append((segments, words, turns))
        return segments

    monkeypatch.setattr(pipeline, "build_speaker_segments", fake_build)

    pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="ar", diarize=True)

    assert len(build_calls) == 1


@pytest.mark.parametrize(
    ("task", "src_lang", "trg_lang", "expected"),
    [
        ("translate", "de", "en", True),  # English target is translated like any other
        ("translate", "en", "de", True),
        ("translate", "de", "de", False),  # same language is a no-op
        ("translate", "de-DE", "de", False),  # locale variants collapse to the base code
        ("translate", None, "de", True),  # unknown source still differs from the target
        ("transcribe", "de", "en", False),  # transcribe never translates
    ],
)
def test_should_translate(
    task: str,
    src_lang: str | None,
    trg_lang: str,
    expected: bool,
) -> None:
    """The LLM translation stage runs only for a translate task across languages.

    Whisper only transcribes, so translation is gated purely on the requested
    task and whether the resolved source differs from the target. An English
    target must translate (no Whisper translate hop exists anymore), while a
    same-language request must be a no-op.

    Args:
        task (str): Requested task, ``"transcribe"`` or ``"translate"``.
        src_lang (str | None): Resolved source language code.
        trg_lang (str): Target language code.
        expected (bool): Whether translation should run.
    """
    assert pipeline.should_translate(task, src_lang, trg_lang) is expected


def test_translation_pipeline_returns_input_when_language_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the translation pipeline to ensure it returns the input when the language matches.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.

    Raises:
        AssertionError: If the translation is attempted when languages match.
    """

    class DummyTranslator:
        """A dummy translator covering the case where detected source matches target language.

        Simulates the translator returning the target language as the detected source; the
        translate method asserts to verify the pipeline returns the input without translating.
        """

        def __init__(self, **kwargs: Any) -> None:
            """Initialise; kwargs accepted to match the real translator signature."""

        def detect_language(self, text: str) -> dict[str, str]:
            """Return a dict reporting ``"es"`` — matches the test's target so translation is skipped.

            Args:
                text (str): The text for which to detect the language.

            Returns:
                dict[str, str]: A dictionary containing the detected language code.
            """
            return {"code": "es"}

        def translate(self, trg_lang: str, text: str) -> str:
            """Simulate the translation process.

            In this dummy implementation, it raises an assertion
            error if called, because the translation pipeline should not attempt to translate when
            the detected source language matches the target language.

            Args:
                trg_lang (str): The target language for translation.

                text (str): The text to be translated.

            Returns:
                str: The translated text (not used in this dummy implementation).

            Raises:
                AssertionError: If the translate method is called when the source and target languages match.
            """
            raise AssertionError("translate should not be called when languages match")

    monkeypatch.setattr(pipeline, "Translator", DummyTranslator)
    df = pd.DataFrame({"text": ["hola"]})

    result = pipeline.translation_pipeline(df.copy(), "es")

    pd.testing.assert_frame_equal(result, df)


def test_translation_pipeline_translates_each_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the translation pipeline to ensure it translates each row correctly.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """

    class DummyTranslator:
        """A dummy translator covering the case where source != target language.

        Simulates the translator detecting a source different from the target and records each
        translate-method call so the test can assert per-row translation actually happened.
        """

        def __init__(self, **kwargs: Any) -> None:
            """Initialise; kwargs accepted to match the real translator signature."""
            self.calls: list[tuple[str, str, str]] = []

        def detect_language(self, text: str) -> dict[str, str]:
            """Return a dict reporting ``"en"`` (doesn't match the ``"es"`` target).

            Forces the pipeline to attempt translation for each row.

            Args:
                text (str): The text for which to detect the language.

            Returns:
                dict[str, str]: A dictionary containing the detected language code.
            """
            return {"code": "en"}

        def translate(self, trg_lang: str, text: str, src_lang: str | None = None) -> str:
            """Record args and return a synthetic translated string ``"{text} (to {trg_lang})"``.

            Lets the test assert per-row translation actually fires when source != target.

            Args:
                trg_lang (str): The target language for translation.
                text (str): The text to be translated.
                src_lang (str | None): The source language of the text. If None, it will be inferred.

            Returns:
                str: The translated text (not used in this dummy implementation).
            """
            self.calls.append((trg_lang, text, src_lang or ""))
            return f"{text}-{trg_lang}"

    translator = DummyTranslator()
    monkeypatch.setattr(pipeline, "Translator", lambda **kwargs: translator)
    df = pd.DataFrame({"text": ["hello", "world"]})

    result = pipeline.translation_pipeline(df.copy(), "es")

    assert translator.calls == [("es", "hello", "en"), ("es", "world", "en")]
    # The original transcript text is preserved untouched...
    assert list(result["text"]) == ["hello", "world"]
    # ...and the translation is written to its own column so both can be
    # cross-referenced in the output table.
    assert list(result["translation"]) == ["hello-es", "world-es"]


def test_effective_text_column_prefers_translation() -> None:
    """``effective_text_column`` should prefer ``translation`` when present, else ``text``."""
    transcribed_only = pd.DataFrame({"text": ["hello"]})
    assert pipeline.effective_text_column(transcribed_only) == "text"

    translated = pd.DataFrame({"text": ["hello"], "translation": ["hola"]})
    assert pipeline.effective_text_column(translated) == "translation"


def test_summarization_pipeline_formats_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the summarization pipeline to ensure it formats the prompt correctly.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    # Import the real InferencePipeline type for subclassing
    from nextext.core.openai_cfg import InferencePipeline

    class DummyPipeline(InferencePipeline):
        """A dummy inference pipeline class to test the summarization pipeline's prompt formatting behavior."""

        def __init__(self) -> None:
            """Initialise the pipeline and start the per-call prompt list."""
            self.prompts: list[str] = []

        @override
        def load_prompt(self, keyword: str = "system") -> str:
            """Load the prompt template for the given keyword.

            Args:
                keyword (str): The keyword for the prompt template.

            Returns:
                str: The prompt template.
            """
            assert keyword == "summary"
            return "Summarize: {text}"

        @override
        def call_model(
            self,
            prompt: str,
            model: str | None = None,
            temperature: float = 0.1,
            seed: int = 42,
            stop: list[str] | None = None,
            num_predict: int | None = None,
            top_p: float | None = None,
            system_prompt: str | None = None,
            include_system_prompt: bool = True,
            think: bool | None = None,
        ) -> str:
            """Simulate calling the model with the given prompt.

            Args:
                prompt (str): The prompt to pass to the model.
                model (str | None): Unused test double argument.
                temperature (float): Unused test double argument.
                seed (int): Unused test double argument.
                stop (list[str] | None): Unused test double argument.
                num_predict (int | None): Unused test double argument.
                top_p (float | None): Unused test double argument.
                system_prompt (str | None): Unused test double argument.
                include_system_prompt (bool): Unused test double argument.
                think (bool | None): Unused test double argument.

            Returns:
                str: The model's response.
            """
            del model, temperature, seed, stop, num_predict, top_p, system_prompt, include_system_prompt
            self.prompts.append(prompt)
            return "summary result"

    inference_pipeline = DummyPipeline()

    result = pipeline.summarization_pipeline("Important content", inference_pipeline)

    assert result == "summary result"
    assert inference_pipeline.prompts == ["Summarize: Important content"]


def test_summarization_pipeline_rejects_empty_text(
    ollama_pipeline: None = None,
) -> None:
    """Test the summarization pipeline to ensure it rejects empty text.

    Args:
        ollama_pipeline (None, optional): The Ollama pipeline instance. Defaults to None.

    Raises:
        ValueError: If the input text is empty.
    """
    from nextext.core.openai_cfg import InferencePipeline

    dummy_pipeline = InferencePipeline()
    with pytest.raises(ValueError):
        pipeline.summarization_pipeline("", dummy_pipeline)


def test_hate_speech_pipeline_returns_flagged_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that hate_speech_pipeline returns only rows flagged as hate speech.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    from nextext.core.openai_cfg import InferencePipeline

    responses = iter(
        [
            {
                "hate_speech": True,
                "category": "racism",
                "confidence": "high",
                "reason": "Contains slurs",
            },
            {
                "hate_speech": False,
                "category": "none",
                "confidence": "low",
                "reason": "",
            },
        ]
    )

    class DummyDetector:
        def __init__(self, inference_pipeline: Any, max_chars: int) -> None:
            pass

        def detect(self, text: str) -> dict[str, Any]:
            return next(responses)

    monkeypatch.setattr(pipeline, "HateSpeechDetector", DummyDetector)

    df = pd.DataFrame({"start": ["00:00:01", "00:00:05"], "text": ["bad text", "good text"]})
    dummy_ip = InferencePipeline.__new__(InferencePipeline)

    results = pipeline.hate_speech_pipeline(df, dummy_ip)

    assert len(results) == 1
    assert results[0]["category"] == "racism"
    assert results[0]["text"] == "bad text"
    assert results[0]["start"] == "00:00:01"


def test_hate_speech_pipeline_reads_translation_column_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detection runs against ``translation`` when that column is present.

    Guards the :func:`effective_text_column` wiring: before translation had
    its own column it overwrote ``text`` in place, so downstream analysis saw
    the translated content. That behavior must be preserved — detection reads
    the translated column and the flagged entry surfaces the translated text,
    not the original.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    from nextext.core.openai_cfg import InferencePipeline

    seen: list[str] = []

    class DummyDetector:
        def __init__(self, inference_pipeline: Any, max_chars: int) -> None:
            pass

        def detect(self, text: str) -> dict[str, Any]:
            seen.append(text)
            return {
                "hate_speech": True,
                "category": "other",
                "confidence": "low",
                "reason": "",
            }

    monkeypatch.setattr(pipeline, "HateSpeechDetector", DummyDetector)

    df = pd.DataFrame(
        {
            "start": ["00:00:01"],
            "text": ["original source text"],
            "translation": ["translated target text"],
        }
    )
    dummy_ip = InferencePipeline.__new__(InferencePipeline)

    results = pipeline.hate_speech_pipeline(df, dummy_ip)

    assert seen == ["translated target text"]
    assert results[0]["text"] == "translated target text"


def test_wordlevel_pipeline_invokes_all_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the word-level pipeline to ensure all steps are invoked.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """

    class DummyWordCounter:
        """A dummy word counter class to test the word-level pipeline's behavior."""

        def __init__(self, text: str, language: str) -> None:
            """Initialise the counter with text + language and start the step-invocation list.

            Args:
                text (str): The text to be processed.
                language (str): The language of the text.
            """
            self.text = text
            self.language = language
            self.steps: list[str] = []

        def text_to_doc(self) -> None:
            """Simulate the process of converting text to a document format.

            In this dummy implementation,
            it simply records that the text_to_doc step was invoked, allowing us to verify that the word-level
            pipeline correctly calls this step as part of its processing.
            """
            self.steps.append("text_to_doc")

        def lemmatize_doc(self) -> None:
            """Simulate the process of lemmatizing a document.

            In this dummy implementation, it simply records
            that the lemmatize_doc step was invoked, allowing us to verify that the word-level pipeline correctly
            calls this step as part of its processing.
            """
            self.steps.append("lemmatize_doc")

        def count_words(self) -> pd.DataFrame:
            """Simulate the process of counting words in the text.

            In this dummy implementation, it returns a
            DataFrame with a single word and its count, allowing us to verify that the word-level pipeline correctly
            calls this step and processes its output.

            Returns:
                pd.DataFrame: A DataFrame containing a single word and its count.
            """
            return pd.DataFrame({"word": ["test"], "count": [1]})

        def get_noun_sentiment(self) -> pd.DataFrame:
            """Simulate the process of getting noun sentiment.

            In this dummy implementation, it returns a DataFrame
            with a single noun and its sentiment score, allowing us to verify that the word-level pipeline correctly
            calls this step and processes its output.

            Returns:
                pd.DataFrame: A DataFrame containing a single noun and its sentiment score.
            """
            return pd.DataFrame({"noun": ["test"], "sentiment": [0.5]})

        def create_interactive_graph(self) -> str:
            """Simulate the process of creating an interactive graph.

            In this dummy implementation, it returns a string
            representing the path to the graph, allowing us to verify that the word-level pipeline correctly calls this
            step and processes its output.

            Returns:
                str: A string representing the path to the interactive graph.
            """
            return "graph.html"

        def create_wordcloud(self) -> str:
            """Simulate word-cloud creation; returns a path-like string for the test to assert on.

            Returns:
                str: A string representing the path to the word cloud.
            """
            return "wordcloud"

    monkeypatch.setattr(pipeline, "WordCounter", lambda text, language: DummyWordCounter(text, language))
    captured: dict[str, str] = {}

    def fake_extract_entities(text: str) -> pd.DataFrame:
        captured["text"] = text
        return pd.DataFrame({"entity": ["Test"]})

    monkeypatch.setattr(pipeline, "extract_entities", fake_extract_entities)
    df = pd.DataFrame({"text": ["alpha", "beta"]})

    counts, entities, wordcloud = pipeline.wordlevel_pipeline(df, "en")

    assert list(counts["word"]) == ["test"]
    assert list(entities["entity"]) == ["Test"]
    assert captured["text"] == "alpha beta"
    assert wordcloud == "wordcloud"  # type: ignore[comparison-overlap]


# ---------------------------------------------------------------------------
# summarization map-reduce + output cap
# ---------------------------------------------------------------------------


class _RecordingPipeline(InferencePipeline):
    """Inference double that records each ``call_model`` invocation.

    Attributes:
        calls: One dict per model call, capturing the ``prompt`` and the
            ``num_predict`` output cap it was invoked with.
    """

    def __init__(self, reply: str = "S") -> None:
        """Initialise the recorder with the canned reply returned per call.

        Args:
            reply (str): Fixed string returned from every ``call_model`` call.
        """
        self.calls: list[dict[str, Any]] = []
        self._reply = reply

    @override
    def load_prompt(self, keyword: str = "system") -> str:
        """Return a minimal ``{text}`` summary template.

        Args:
            keyword (str): The prompt keyword; must be ``"summary"``.

        Returns:
            str: A ``{text}`` template prefixed with ``Summarize: ``.
        """
        assert keyword == "summary"
        return "Summarize: {text}"

    @override
    def call_model(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.1,
        seed: int = 42,
        stop: list[str] | None = None,
        num_predict: int | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
        include_system_prompt: bool = True,
        think: bool | None = None,
    ) -> str:
        """Record the call and return the canned reply.

        Args:
            prompt (str): The prompt sent to the model.
            model (str | None): Unused test double argument.
            temperature (float): Unused test double argument.
            seed (int): Unused test double argument.
            stop (list[str] | None): Unused test double argument.
            num_predict (int | None): Recorded output-token cap.
            top_p (float | None): Unused test double argument.
            system_prompt (str | None): Unused test double argument.
            include_system_prompt (bool): Unused test double argument.
            think (bool | None): Unused test double argument.

        Returns:
            str: The canned reply.

        Raises:
            RuntimeError: If invoked more than 1000 times, a sign the
                map-reduce recursion failed to terminate.
        """
        del model, temperature, seed, stop, top_p, system_prompt, include_system_prompt, think
        if len(self.calls) >= 1000:
            raise RuntimeError("call_model invoked too many times; recursion likely unbounded")
        self.calls.append({"prompt": prompt, "num_predict": num_predict})
        return self._reply


def test_split_to_budget_packs_words_within_budget() -> None:
    """``_split_to_budget`` groups whole words into chunks no larger than the budget."""
    text = "alpha beta gamma delta epsilon zeta eta theta"

    chunks = pipeline._split_to_budget(text, 12)

    assert chunks
    assert all(len(chunk) <= 12 for chunk in chunks)
    # Every original word is preserved across the chunks, in order.
    assert " ".join(chunks).split() == text.split()


def test_split_to_budget_returns_single_chunk_when_text_fits() -> None:
    """Text within the budget yields exactly one chunk (single-shot path)."""
    chunks = pipeline._split_to_budget("short text", 100)

    assert chunks == ["short text"]


def test_split_to_budget_hard_slices_oversized_token() -> None:
    """A single token longer than the budget is sliced, never exceeding it."""
    chunks = pipeline._split_to_budget("abcdefghij", 3)

    assert all(len(chunk) <= 3 for chunk in chunks)
    assert "".join(chunks) == "abcdefghij"


def test_summarization_single_shot_applies_output_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short transcripts take the single-shot path with the output cap applied.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("SUMMARY_MAX_INPUT_TOKENS", raising=False)
    recorder = _RecordingPipeline(reply="the summary")

    result = pipeline.summarization_pipeline("a short transcript", recorder)

    assert result == "the summary"
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["num_predict"] == pipeline.SUMMARY_MAX_OUTPUT_TOKENS
    assert recorder.calls[0]["prompt"] == "Summarize: a short transcript"


def test_summarization_map_reduce_splits_long_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transcript larger than the budget is summarized hierarchically.

    Every request — the per-chunk map calls and the final reduce — stays within
    the character budget derived from ``SUMMARY_MAX_INPUT_TOKENS``, and the
    output cap is applied to all of them.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("SUMMARY_MAX_INPUT_TOKENS", "10")
    char_budget = int(10 * pipeline._CHARS_PER_TOKEN)
    recorder = _RecordingPipeline(reply="S")
    text = " ".join(f"word{i:02d}" for i in range(40))

    result = pipeline.summarization_pipeline(text, recorder)

    assert result == "S"
    assert len(recorder.calls) > 1
    payloads = [call["prompt"].removeprefix("Summarize: ") for call in recorder.calls]
    assert all(len(payload) <= char_budget for payload in payloads)
    assert all(call["num_predict"] == pipeline.SUMMARY_MAX_OUTPUT_TOKENS for call in recorder.calls)


def test_summarization_terminates_when_summaries_do_not_shrink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reduce-depth guard prevents unbounded recursion.

    When the model returns partial summaries larger than the budget (so the
    combined text never shrinks), summarization still terminates and returns a
    string rather than recursing forever.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("SUMMARY_MAX_INPUT_TOKENS", "10")
    recorder = _RecordingPipeline(reply="X" * 50)
    text = " ".join(f"word{i:02d}" for i in range(40))

    result = pipeline.summarization_pipeline(text, recorder)

    assert isinstance(result, str)
    assert result
    assert len(recorder.calls) < 500


class _OverflowingPipeline(InferencePipeline):
    """Inference double that raises on payloads above a character threshold.

    Simulates a backend whose real context window is smaller than the character
    budget assumed by the summarizer (e.g. token-dense scripts), so oversized
    requests raise instead of returning a summary.

    Attributes:
        max_payload_chars: Requests whose ``{text}`` payload exceeds this raise.
        overflow_count: Number of calls that raised the simulated error.
        success_count: Number of calls that returned the canned reply.
    """

    def __init__(
        self,
        max_payload_chars: int,
        reply: str = "S",
        error_message: str = "This model's maximum context length is 4096 tokens",
    ) -> None:
        """Configure the overflow threshold and the simulated error.

        Args:
            max_payload_chars (int): Payloads longer than this raise the error.
            reply (str): Canned reply returned for in-budget payloads.
            error_message (str): Message of the raised ``RuntimeError``.
        """
        self.max_payload_chars = max_payload_chars
        self._reply = reply
        self._error_message = error_message
        self.overflow_count = 0
        self.success_count = 0

    @override
    def load_prompt(self, keyword: str = "system") -> str:
        """Return a minimal ``{text}`` summary template.

        Args:
            keyword (str): The prompt keyword; must be ``"summary"``.

        Returns:
            str: A ``{text}`` template prefixed with ``Summarize: ``.
        """
        assert keyword == "summary"
        return "Summarize: {text}"

    @override
    def call_model(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.1,
        seed: int = 42,
        stop: list[str] | None = None,
        num_predict: int | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
        include_system_prompt: bool = True,
        think: bool | None = None,
    ) -> str:
        """Raise when the payload is too large, otherwise return the reply.

        Args:
            prompt (str): The prompt sent to the model.
            model (str | None): Unused test double argument.
            temperature (float): Unused test double argument.
            seed (int): Unused test double argument.
            stop (list[str] | None): Unused test double argument.
            num_predict (int | None): Unused test double argument.
            top_p (float | None): Unused test double argument.
            system_prompt (str | None): Unused test double argument.
            include_system_prompt (bool): Unused test double argument.
            think (bool | None): Unused test double argument.

        Returns:
            str: The canned reply for in-budget payloads.

        Raises:
            RuntimeError: When the payload exceeds ``max_payload_chars``.
        """
        del model, temperature, seed, stop, num_predict, top_p, system_prompt, include_system_prompt, think
        payload = prompt.removeprefix("Summarize: ")
        if len(payload) > self.max_payload_chars:
            self.overflow_count += 1
            raise RuntimeError(self._error_message)
        self.success_count += 1
        return self._reply


def test_is_context_length_error_detects_known_messages() -> None:
    """The predicate recognises provider overflow messages and ignores others."""
    assert pipeline._is_context_length_error(
        RuntimeError("This model's maximum context length is 4096 tokens, however you requested 9000")
    )
    assert pipeline._is_context_length_error(ValueError("context_length_exceeded"))
    assert not pipeline._is_context_length_error(RuntimeError("connection refused"))


def test_summarization_retries_with_smaller_budget_on_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A context-length error shrinks the budget and retries instead of crashing.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("SUMMARY_MAX_INPUT_TOKENS", "20")  # -> 60-char budget
    overflower = _OverflowingPipeline(max_payload_chars=35)
    text = " ".join(f"word{i:02d}" for i in range(30))

    result = pipeline.summarization_pipeline(text, overflower)

    assert result == "S"
    assert overflower.overflow_count >= 1  # the first, oversized attempt failed
    assert overflower.success_count >= 1  # the smaller-budget retry succeeded


def test_summarization_reraises_non_context_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors that are not context-length overflows propagate unchanged.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("SUMMARY_MAX_INPUT_TOKENS", raising=False)
    overflower = _OverflowingPipeline(max_payload_chars=-1, error_message="connection refused")

    with pytest.raises(RuntimeError, match="connection refused"):
        pipeline.summarization_pipeline("anything", overflower)


def test_summarization_degrades_to_empty_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent overflow degrades to an empty summary rather than crashing.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("SUMMARY_MAX_INPUT_TOKENS", "20")
    overflower = _OverflowingPipeline(max_payload_chars=-1)  # every request overflows

    result = pipeline.summarization_pipeline("some transcript text", overflower)

    assert result == ""
    assert overflower.overflow_count == pipeline._MAX_OVERFLOW_RETRIES + 1


class _ErroringPipeline(_OverflowingPipeline):
    """Inference double whose every model call raises a given exception.

    Attributes:
        error: The exception instance raised by :meth:`call_model`.
        call_count: Number of model calls attempted before raising.
    """

    def __init__(self, error: Exception) -> None:
        """Configure the exception raised on every model call.

        Args:
            error (Exception): The exception instance to raise.
        """
        super().__init__(max_payload_chars=-1)
        self.error = error
        self.call_count = 0

    @override
    def call_model(self, prompt: str, **kwargs: Any) -> str:  # pyrefly: ignore[bad-override]
        """Raise the configured exception unconditionally.

        Args:
            prompt (str): The prompt sent to the model (ignored).
            **kwargs (Any): Unused test double arguments.

        Returns:
            str: Never returns.

        Raises:
            Exception: The configured ``error`` instance, always.
        """
        del prompt, kwargs
        self.call_count += 1
        raise self.error


def _api_status_error(status_code: int, message: str) -> openai.APIStatusError:
    """Build an ``openai.APIStatusError`` (or subclass) for a status code.

    Args:
        status_code (int): The simulated HTTP status code.
        message (str): The error message carried by the exception.

    Returns:
        openai.APIStatusError: The SDK exception the client would raise for
            that status, e.g. :class:`openai.InternalServerError` for 500.
    """
    request = httpx2.Request("POST", "http://inference.invalid/v1/chat/completions")
    response = httpx2.Response(status_code, request=request)
    return openai.APIStatusError(message, response=response, body=None)


def test_summarization_degrades_to_empty_on_transient_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5xx from the inference endpoint fail-softs to an empty summary.

    Mirrors the production incident where the LLM router returned 500
    (upstream connection error) and the whole job crashed instead of
    degrading like the NER/diarization clients do.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("SUMMARY_MAX_INPUT_TOKENS", raising=False)
    erroring = _ErroringPipeline(_api_status_error(500, "InternalServerError - Connection error."))

    result = pipeline.summarization_pipeline("some transcript text", erroring)

    assert result == ""
    assert erroring.call_count == 1


def test_summarization_degrades_to_empty_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client-side connection failure fail-softs to an empty summary.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("SUMMARY_MAX_INPUT_TOKENS", raising=False)
    request = httpx2.Request("POST", "http://inference.invalid/v1/chat/completions")
    erroring = _ErroringPipeline(openai.APIConnectionError(request=request))

    result = pipeline.summarization_pipeline("some transcript text", erroring)

    assert result == ""


def test_summarization_reraises_non_transient_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration-shaped API errors (4xx) still fail the job loudly.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("SUMMARY_MAX_INPUT_TOKENS", raising=False)
    erroring = _ErroringPipeline(_api_status_error(401, "invalid api key"))

    with pytest.raises(openai.APIStatusError, match="invalid api key"):
        pipeline.summarization_pipeline("some transcript text", erroring)


def test_hate_speech_pipeline_keeps_partial_findings_on_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient inference failure mid-sweep returns the findings so far.

    A router outage during the per-segment loop must not crash the job (and
    with it the completed transcription); the sweep stops at the failing
    segment and the findings collected up to that point are kept.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    class DummyDetector:
        def __init__(self, inference_pipeline: Any, max_chars: int) -> None:
            self.calls = 0

        def detect(self, text: str) -> dict[str, Any]:
            if text == "second text":
                raise _api_status_error(500, "InternalServerError - Connection error.")
            return {
                "hate_speech": True,
                "category": "other",
                "confidence": "low",
                "reason": "",
            }

    monkeypatch.setattr(pipeline, "HateSpeechDetector", DummyDetector)
    df = pd.DataFrame(
        {
            "start": ["00:00:01", "00:00:05", "00:00:09"],
            "text": ["first text", "second text", "third text"],
        }
    )
    dummy_ip = InferencePipeline.__new__(InferencePipeline)

    results = pipeline.hate_speech_pipeline(df, dummy_ip)

    assert len(results) == 1
    assert results[0]["text"] == "first text"


def test_hate_speech_pipeline_reraises_non_transient_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration-shaped API errors (4xx) during detection still raise.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    class DummyDetector:
        def __init__(self, inference_pipeline: Any, max_chars: int) -> None:
            pass

        def detect(self, text: str) -> dict[str, Any]:
            raise _api_status_error(401, "invalid api key")

    monkeypatch.setattr(pipeline, "HateSpeechDetector", DummyDetector)
    df = pd.DataFrame({"start": ["00:00:01"], "text": ["some text"]})
    dummy_ip = InferencePipeline.__new__(InferencePipeline)

    with pytest.raises(openai.APIStatusError, match="invalid api key"):
        pipeline.hate_speech_pipeline(df, dummy_ip)


def test_transcript_txt_exports_transcribe_returns_single_block_file() -> None:
    """A transcribe-only transcript yields one readable 'transcript' block export."""
    df = pd.DataFrame(
        {
            "start": ["00:00:00"],
            "end": ["00:00:02"],
            "speaker": ["SPEAKER_00"],
            "text": ["Hello world."],
        }
    )
    exports = transcript_txt_exports(df)
    assert [label for label, _ in exports] == ["transcript"]
    _, block = exports[0]
    rule = "=" * 40
    assert block == f"{rule}\n[00:00:00 - 00:00:02]  SPEAKER_00\n{rule}\nHello world.\n"


def test_transcript_txt_exports_translate_splits_into_two_files() -> None:
    """A translated transcript yields separate 'transcript' and 'translation' block exports."""
    df = pd.DataFrame(
        {
            "start": ["00:00:00"],
            "end": ["00:00:02"],
            "speaker": ["SPEAKER_00"],
            "text": ["Hello world."],
            "translation": ["Hallo Welt."],
        }
    )
    assert [label for label, _ in transcript_txt_exports(df)] == ["transcript", "translation"]
    exports = dict(transcript_txt_exports(df))
    assert set(exports) == {"transcript", "translation"}
    rule = "=" * 40
    assert exports["transcript"] == f"{rule}\n[00:00:00 - 00:00:02]  SPEAKER_00\n{rule}\nHello world.\n"
    assert exports["translation"] == f"{rule}\n[00:00:00 - 00:00:02]  SPEAKER_00\n{rule}\nHallo Welt.\n"
    # Each file carries only its own text column, never the other.
    assert "Hallo Welt." not in exports["transcript"]
    assert "Hello world." not in exports["translation"]


def test_transcript_txt_exports_omits_speaker_tag_when_no_diarization() -> None:
    """With no ``speaker`` column, the header carries no ``(...)`` tag at all."""
    df = pd.DataFrame({"start": ["0:00:00"], "end": ["0:05:01"], "text": ["Praise be."]})
    ((_, block),) = transcript_txt_exports(df)
    rule = "=" * 40
    assert block == f"{rule}\n[0:00:00 - 0:05:01]\n{rule}\nPraise be.\n"
    # No ``]  speaker`` suffix on the header when the job was not diarized.
    assert "]  " not in block


def test_transcript_txt_exports_preserves_order_and_blank_line_separator() -> None:
    """Segments render in order, diarized labels pass through, blank line between blocks."""
    df = pd.DataFrame(
        {
            "start": ["0:00:00", "0:00:05"],
            "end": ["0:00:05", "0:00:09"],
            "speaker": ["SPEAKER_00", "SPEAKER_01"],
            "text": ["Hello there.", "General Kenobi."],
        }
    )
    assert [label for label, _ in transcript_txt_exports(df)] == ["transcript"]
    ((_, block),) = transcript_txt_exports(df)
    rule = "=" * 40
    assert block == (
        f"{rule}\n[0:00:00 - 0:00:05]  SPEAKER_00\n{rule}\nHello there.\n\n"
        f"{rule}\n[0:00:05 - 0:00:09]  SPEAKER_01\n{rule}\nGeneral Kenobi.\n"
    )


def test_transcript_txt_exports_banner_fences_header_around_paragraph_body() -> None:
    """A body with its own paragraph break stays unambiguous: rule lines fence the header.

    This is the motivating case — a blank line *inside* a segment's text must not
    read as a segment boundary, because the header is bannered above and below.
    """
    df = pd.DataFrame(
        {
            "start": ["00:00:00"],
            "end": ["00:00:04"],
            "speaker": ["SPEAKER_00"],
            "text": ["First paragraph.\n\nSecond paragraph."],
        }
    )
    ((_, block),) = transcript_txt_exports(df)
    rule = "=" * 40
    assert block == (f"{rule}\n[00:00:00 - 00:00:04]  SPEAKER_00\n{rule}\nFirst paragraph.\n\nSecond paragraph.\n")
    # The header is fenced above and below by rule lines, so the blank line
    # inside the body cannot be mistaken for a segment boundary.
    assert f"{rule}\n[00:00:00 - 00:00:04]  SPEAKER_00\n{rule}" in block
    assert "First paragraph.\n\nSecond paragraph." in block


class _GateTranscriber:
    """Minimal ExternalWhisperTranscriber stand-in with a single 0-10s segment."""

    def __init__(self, **params: Any) -> None:
        """Record construction params and seed a one-segment transcript result.

        Args:
            **params (Any): Keyword arguments forwarded by the pipeline.
        """
        self.params = params
        self.src_lang = "en"
        self.skip_reason: str | None = None
        self.transcription_result: dict[str, Any] = {
            "segments": [{"start": 0.0, "end": 10.0, "text": "x"}],
            "words": [],
        }

    def transcription(self) -> None:
        """No-op stand-in for the Whisper call."""

    def transcript_output(self) -> pd.DataFrame:
        """Return a dummy transcript.

        Returns:
            pd.DataFrame: A one-row transcript frame.
        """
        return pd.DataFrame({"text": ["x"]})


class _SpeakerReflectingTranscriber:
    """Transcriber stand-in whose transcript_output mirrors the segments' speaker column."""

    def __init__(self, **params: Any) -> None:
        """Seed a three-segment result so the diarize path runs.

        Args:
            **params (Any): Ignored construction params.
        """
        self.src_lang = "en"
        self.skip_reason: str | None = None
        self.transcription_result: dict[str, Any] = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "a"},
                {"start": 1.0, "end": 2.0, "text": "b"},
                {"start": 2.0, "end": 3.0, "text": "c"},
            ],
            "words": [],
        }

    def transcription(self) -> None:
        """No-op stand-in for the Whisper call."""

    def transcript_output(self) -> pd.DataFrame:
        """Return a frame mirroring each final segment's text and speaker.

        Returns:
            pd.DataFrame: One row per final segment with ``text`` and ``speaker``.
        """
        segments = self.transcription_result["segments"]
        return pd.DataFrame(
            {"text": [s["text"] for s in segments], "speaker": [s.get("speaker", "") for s in segments]}
        )


def test_transcription_pipeline_renumbers_speakers_by_transcript_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: alignment labels out of order are renumbered by first appearance in the output.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    monkeypatch.setattr(
        pipeline, "load_whisper_env", lambda: WhisperClientConfig(api_base="http://a/v1", api_key="k", model="m")
    )
    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", _SpeakerReflectingTranscriber)
    monkeypatch.setattr(pipeline, "diarize_file", lambda fp: [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}])
    monkeypatch.setattr(
        pipeline, "load_sentence_restore_env", lambda: SentenceRestoreConfig(enabled=False, min_punct_ratio=0.01)
    )

    def fake_build(
        segments: list[dict[str, Any]], words: list[dict[str, Any]], turns: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return segments whose speaker labels are NOT in first-appearance order.

        Args:
            segments (list[dict[str, Any]]): Whisper segments.
            words (list[dict[str, Any]]): Whisper words.
            turns (list[dict[str, Any]]): Speaker turns.

        Returns:
            list[dict[str, Any]]: Segments labelled Speaker 5, Speaker 2, Speaker 5.
        """
        return [
            {"start": 0.0, "end": 1.0, "text": "a", "speaker": "Speaker 5"},
            {"start": 1.0, "end": 2.0, "text": "b", "speaker": "Speaker 2"},
            {"start": 2.0, "end": 3.0, "text": "c", "speaker": "Speaker 5"},
        ]

    monkeypatch.setattr(pipeline, "build_speaker_segments", fake_build)

    outcome = pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="en", diarize=True)

    # Speaker 5 first-heard -> Speaker 1; Speaker 2 next-new -> Speaker 2; back to 5 -> Speaker 1.
    assert list(outcome.transcript["speaker"]) == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_transcription_pipeline_fills_unlabeled_segments_from_nearest_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a segment overlapping no diarize turn inherits the nearest turn's speaker.

    Runs the real alignment functions (no fakes for build/fill/renumber) so the
    middle segment — which overlaps neither turn — would previously stay
    unlabeled and render as ``Unknown``. With the nearest-turn fallback it
    inherits the closer (preceding) turn's label before renumbering.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    monkeypatch.setattr(
        pipeline, "load_whisper_env", lambda: WhisperClientConfig(api_base="http://a/v1", api_key="k", model="m")
    )
    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", _SpeakerReflectingTranscriber)
    monkeypatch.setattr(
        pipeline,
        "diarize_file",
        lambda fp: [
            {"start": 0.0, "end": 0.9, "speaker": "SPEAKER_00"},  # 0.1s before segment b
            {"start": 2.2, "end": 3.0, "speaker": "SPEAKER_01"},  # 0.2s after segment b
        ],
    )
    monkeypatch.setattr(
        pipeline, "load_sentence_restore_env", lambda: SentenceRestoreConfig(enabled=False, min_punct_ratio=0.01)
    )

    outcome = pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="en", diarize=True)

    # a overlaps SPEAKER_00, c overlaps SPEAKER_01; b overlaps nothing and
    # inherits the nearer preceding turn (SPEAKER_00) instead of staying blank.
    assert list(outcome.transcript["speaker"]) == ["Speaker 1", "Speaker 1", "Speaker 2"]


# ---------------------------------------------------------------------------
# Typed skip reasons carried out of transcription_pipeline
# ---------------------------------------------------------------------------


class _SkippingTranscriber:
    """Transcriber stand-in that produced no segments for a typed reason."""

    reason: str | None = "vad_no_speech"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the stand-in; args/kwargs match the real signature."""
        self.src_lang = "en"
        self.transcription_result: dict[str, Any] | None = None
        self.skip_reason: str | None = None

    def transcription(self) -> None:
        """Simulate a run that yielded no segments."""
        self.transcription_result = {"segments": [], "words": []}
        self.skip_reason = type(self).reason

    def transcript_output(self) -> pd.DataFrame:
        """Return an empty transcript DataFrame.

        Returns:
            pd.DataFrame: An empty-text DataFrame.
        """
        return pd.DataFrame({"text": []})


def test_transcription_pipeline_carries_skip_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason recorded by the transcriber must reach the caller.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides the Whisper env and transcriber.
    """
    monkeypatch.setattr(
        pipeline,
        "load_whisper_env",
        lambda: WhisperClientConfig(api_base="http://audio:8000/v1", api_key="k", model="test-model"),
    )
    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", _SkippingTranscriber)

    outcome = pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="en", diarize=False)

    assert outcome.transcript.empty
    assert outcome.src_lang == "en"
    assert outcome.skip_reason == "vad_no_speech"
    assert outcome.is_empty is True


def test_transcription_pipeline_defaults_skip_reason_for_empty_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty transcript with no recorded cause defaults to ``asr_empty_transcript``.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides the Whisper env and transcriber.
    """
    monkeypatch.setattr(
        pipeline,
        "load_whisper_env",
        lambda: WhisperClientConfig(api_base="http://audio:8000/v1", api_key="k", model="test-model"),
    )
    monkeypatch.setattr(_SkippingTranscriber, "reason", None)
    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", _SkippingTranscriber)

    outcome = pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="en", diarize=False)

    assert outcome.skip_reason == "asr_empty_transcript"


def test_transcription_pipeline_reports_no_skip_reason_for_real_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transcript with text must carry no skip reason and not read as empty.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides the Whisper env and transcriber.
    """
    monkeypatch.setattr(
        pipeline,
        "load_whisper_env",
        lambda: WhisperClientConfig(api_base="http://audio:8000/v1", api_key="k", model="test-model"),
    )

    class _SpeakingTranscriber(_SkippingTranscriber):
        """Stand-in that produced one real segment."""

        @override
        def transcription(self) -> None:
            """Simulate a successful run."""
            self.transcription_result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hola."}], "words": []}
            self.skip_reason = None

        @override
        def transcript_output(self) -> pd.DataFrame:
            """Return a one-row transcript.

            Returns:
                pd.DataFrame: A single-segment DataFrame.
            """
            return pd.DataFrame({"text": ["hola."]})

    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", _SpeakingTranscriber)

    outcome = pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="es", diarize=False)

    assert outcome.skip_reason is None
    assert outcome.is_empty is False


def test_transcription_outcome_is_empty_for_whitespace_only_text() -> None:
    """A transcript of only blank text must count as empty."""
    outcome = pipeline.TranscriptionOutcome(
        transcript=pd.DataFrame({"text": ["   ", ""]}),
        src_lang="en",
        skip_reason=None,
    )

    assert outcome.is_empty is True


# ---------------------------------------------------------------------------
# summarization_pipeline — visual context
# ---------------------------------------------------------------------------


def test_summarization_prepends_visual_context_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame captions reach the model ahead of the transcript, clearly labelled.

    The summary prompt tells the model to treat a leading "Visual context"
    section as a second source, so the block must be inside the ``{text}``
    payload — not a separate message the template never sees.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("SUMMARY_MAX_INPUT_TOKENS", raising=False)
    recorder = _RecordingPipeline(reply="s")

    pipeline.summarization_pipeline(
        "they discussed the roadmap",
        recorder,
        visual_context="[00:00] a slide titled Roadmap",
    )

    prompt = recorder.calls[0]["prompt"]
    assert "[00:00] a slide titled Roadmap" in prompt
    assert prompt.index("a slide titled Roadmap") < prompt.index("they discussed the roadmap")
    assert "Visual context" in prompt
    assert "Transcript" in prompt


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_summarization_without_visual_context_is_unchanged(monkeypatch: pytest.MonkeyPatch, empty: str | None) -> None:
    """An absent or blank block leaves the audio-only prompt byte-identical.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        empty (str | None): A falsy visual-context value.
    """
    monkeypatch.delenv("SUMMARY_MAX_INPUT_TOKENS", raising=False)
    recorder = _RecordingPipeline(reply="s")

    pipeline.summarization_pipeline("a short transcript", recorder, visual_context=empty)

    assert recorder.calls[0]["prompt"] == "Summarize: a short transcript"


def test_summarization_from_visual_context_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Captions with no transcript summarize on their own.

    A keyframes-only job — or a video whose audio held no speech — still has a
    summarizable source. The payload then carries no ``Transcript`` heading,
    which would otherwise invite the model to remark on the missing half.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("SUMMARY_MAX_INPUT_TOKENS", raising=False)
    recorder = _RecordingPipeline(reply="s")

    assert pipeline.summarization_pipeline("", recorder, visual_context="[00:00] a room") == "s"

    prompt = recorder.calls[0]["prompt"]
    assert "[00:00] a room" in prompt
    assert "Visual context" in prompt
    assert "Transcript" not in prompt


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_summarization_rejects_empty_text_and_empty_visual_context(
    monkeypatch: pytest.MonkeyPatch, empty: str | None
) -> None:
    """With neither a transcript nor captions there is nothing to summarize.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        empty (str | None): A falsy visual-context value.
    """
    monkeypatch.delenv("SUMMARY_MAX_INPUT_TOKENS", raising=False)
    recorder = _RecordingPipeline(reply="s")

    with pytest.raises(ValueError):
        pipeline.summarization_pipeline("", recorder, visual_context=empty)


def test_summarization_with_visual_context_still_chunks_long_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The block is part of the budgeted payload, not an exemption from it.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("SUMMARY_MAX_INPUT_TOKENS", "10")
    recorder = _RecordingPipeline(reply="partial")

    pipeline.summarization_pipeline(
        " ".join(["word"] * 400),
        recorder,
        visual_context="[00:00] a hallway",
    )

    assert len(recorder.calls) > 1  # map-reduce still engaged
