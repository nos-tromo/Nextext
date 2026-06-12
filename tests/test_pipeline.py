"""Tests for the shared Nextext pipeline helpers."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nextext import pipeline


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
    """A multi-speaker request transcribes, then diarizes via the /diarize client.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")

    class DummyTranscriber:
        """A dummy transcriber recording its parameters and exposing transcription_result."""

        instance: "DummyTranscriber"

        def __init__(
            self,
            file_path: Path,
            src_lang: str,
            n_speakers: int,
        ) -> None:
            """Initialize the dummy transcriber with the given parameters.

            Args:
                file_path (Path): Path to the audio file.
                src_lang (str): Source language code.
                n_speakers (int): Maximum speaker count for diarization.
            """
            self.params = {
                "file_path": file_path,
                "src_lang": src_lang,
                "n_speakers": n_speakers,
            }
            self.transcription_called = False
            self.transcription_result: dict[str, Any] | None = None
            self.src_lang = "fr"
            DummyTranscriber.instance = self

        def transcription(self) -> None:
            """Simulate transcription by flagging the call and populating one segment."""
            self.transcription_called = True
            self.transcription_result = {"segments": [{"start": 0.0, "end": 1.0, "text": "bonjour"}]}

        def transcript_output(self) -> pd.DataFrame:
            """Return a dummy transcript DataFrame.

            Returns:
                pd.DataFrame: A DataFrame containing the dummy transcription text.
            """
            return pd.DataFrame({"text": ["bonjour"]})

    diarize_calls: dict[str, Any] = {}

    def fake_diarize_file(file_path: Path, **kwargs: Any) -> list[dict[str, Any]]:
        """Record the requested speaker bound and return one speaker turn.

        Args:
            file_path (Path): Path forwarded by the pipeline.
            **kwargs (Any): Speaker-count keyword arguments.

        Returns:
            list[dict[str, Any]]: A single fake speaker turn.
        """
        diarize_calls.update(kwargs)
        return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

    assign_calls: list[Any] = []

    def fake_assign(transcription_segments: list[dict[str, Any]], diarize_segments: list[dict[str, Any]]) -> None:
        """Record that speaker alignment was invoked.

        Args:
            transcription_segments (list[dict[str, Any]]): Segments to label.
            diarize_segments (list[dict[str, Any]]): Speaker turns.
        """
        assign_calls.append((transcription_segments, diarize_segments))

    monkeypatch.setattr(pipeline, "WhisperTranscriber", DummyTranscriber)
    monkeypatch.setattr(pipeline, "diarize_file", fake_diarize_file)
    monkeypatch.setattr(pipeline, "assign_speakers_by_overlap", fake_assign)

    df, detected_lang = pipeline.transcription_pipeline(
        file_path=Path("/tmp/audio.wav"),
        src_lang="auto",
        n_speakers=2,
    )

    instance = DummyTranscriber.instance
    assert instance.params["n_speakers"] == 2
    assert instance.transcription_called is True
    assert diarize_calls["max_speakers"] == 2
    assert len(assign_calls) == 1
    assert list(df["text"]) == ["bonjour"]
    assert detected_lang == "fr"


def test_transcription_pipeline_falls_back_to_original_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-speaker requests fall back to the source language and never diarize.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")

    class DummyTranscriber:
        """A dummy transcriber that leaves src_lang unset to exercise the fallback path."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialise the dummy transcriber; args/kwargs accepted to match the real signature."""
            self.src_lang = None
            self.transcription_result: dict[str, Any] | None = None

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
        """Fail if diarization is attempted for a single-speaker request.

        Args:
            *args (Any): Unused positional arguments.
            **kwargs (Any): Unused keyword arguments.

        Returns:
            list[dict[str, Any]]: Never returns; always fails the test.
        """
        pytest.fail("diarize_file should not be called when n_speakers <= 1")

    monkeypatch.setattr(pipeline, "WhisperTranscriber", DummyTranscriber)
    monkeypatch.setattr(pipeline, "diarize_file", fail_diarize)

    df, detected_lang = pipeline.transcription_pipeline(
        file_path=Path("/tmp/audio.wav"),
        src_lang="es",
        n_speakers=1,
    )

    assert list(df["text"]) == ["hola"]
    assert detected_lang == "es"


def test_transcription_pipeline_skips_diarization_for_empty_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty transcript skips diarization even when many speakers are requested.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")

    class DummyTranscriber:
        """A dummy transcriber that yields an empty transcript."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialise the dummy transcriber; args/kwargs accepted to match the real signature."""
            self.src_lang = "en"
            self.transcription_result: dict[str, Any] | None = None

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

    monkeypatch.setattr(pipeline, "WhisperTranscriber", DummyTranscriber)
    monkeypatch.setattr(pipeline, "diarize_file", fail_diarize)

    df, detected_lang = pipeline.transcription_pipeline(
        file_path=Path("/tmp/audio.wav"),
        src_lang="en",
        n_speakers=3,
    )

    assert list(df["text"]) == []
    assert detected_lang == "en"


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
    assert list(result["text"]) == ["hello-es", "world-es"]


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

        def load_prompt(self, keyword: str = "system") -> str:
            """Load the prompt template for the given keyword.

            Args:
                keyword (str): The keyword for the prompt template.

            Returns:
                str: The prompt template.
            """
            assert keyword == "summary"
            return "Summarize: {text}"

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
            del (
                model,
                temperature,
                seed,
                stop,
                num_predict,
                top_p,
                system_prompt,
                include_system_prompt,
            )
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

        def named_entity_recognition(self) -> pd.DataFrame:
            """Simulate the process of named entity recognition.

            In this dummy implementation, it returns a DataFrame
            with a single entity, allowing us to verify that the word-level pipeline correctly calls this step and
            processes its output.

            Returns:
                pd.DataFrame: A DataFrame containing a single entity.
            """
            return pd.DataFrame({"entity": ["Test"]})

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
    df = pd.DataFrame({"text": ["alpha", "beta"]})

    counts, entities, wordcloud = pipeline.wordlevel_pipeline(df, "en")

    assert list(counts["word"]) == ["test"]
    assert list(entities["entity"]) == ["Test"]
    assert wordcloud == "wordcloud"  # type: ignore[comparison-overlap]
