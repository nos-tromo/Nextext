"""Tests for the shared Nextext pipeline helpers."""

from pathlib import Path

import pandas as pd
import pytest

from nextext import pipeline

pytest.importorskip("pandas")


@pytest.fixture
def disable_docker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable Docker environment detection by mocking the Path.exists method.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:  # type: ignore[override]
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

    def fake_exists(path: Path) -> bool:  # type: ignore[override]
        if path.as_posix() == "/.dockerenv":
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)


def test_transcription_pipeline_invokes_transcriber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the transcription pipeline to ensure it invokes the transcriber correctly.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """

    class DummyTranscriber:
        """A dummy transcriber class to test the transcription pipeline without relying
        on the actual WhisperX implementation. It records the parameters passed to it and
        whether its methods were called, allowing us to verify the pipeline's behavior.
        """

        instance: "DummyTranscriber"

        def __init__(
            self,
            file_path: Path,
            trg_lang: str,
            src_lang: str,
            model_id: str,
            task: str,
            n_speakers: int,
        ) -> None:
            """Initialize the dummy transcriber with the given parameters.

            Args:
                file_path (Path): Path to the audio file.
                trg_lang (str): Target language code.
                src_lang (str): Source language code.
                model_id (str): Model ID for the transcription model.
                task (str): Task to perform (transcribe or translate).
                n_speakers (int): Number of speakers for diarization.
            """
            self.params = {
                "file_path": file_path,
                "trg_lang": trg_lang,
                "src_lang": src_lang,
                "model_id": model_id,
                "task": task,
                "n_speakers": n_speakers,
            }
            self.transcription_called = False
            self.diarization_called = False
            self.src_lang = "fr"
            DummyTranscriber.instance = self

        def transcription(self) -> None:
            """Simulate the transcription process by setting a flag to indicate that the method was called."""
            self.transcription_called = True

        def diarization(self) -> None:
            """Simulate the diarization process by setting a flag to indicate that the method was called."""
            self.diarization_called = True

        def transcript_output(self) -> pd.DataFrame:
            """Simulate the output of the transcription process by returning a DataFrame with dummy text data.

            Returns:
                pd.DataFrame: A DataFrame containing the dummy transcription text.
            """
            return pd.DataFrame({"text": ["bonjour"]})

    monkeypatch.setattr(pipeline, "WhisperTranscriber", DummyTranscriber)

    df, detected_lang = pipeline.transcription_pipeline(
        file_path=Path("/tmp/audio.wav"),
        trg_lang="en",
        src_lang="auto",
        model_id="base",
        task="transcribe",
        n_speakers=2,
    )

    instance = DummyTranscriber.instance
    assert instance.params["n_speakers"] == 2
    assert instance.transcription_called is True
    assert instance.diarization_called is True
    assert list(df["text"]) == ["bonjour"]
    assert detected_lang == "fr"


def test_transcription_pipeline_falls_back_to_original_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test the transcription pipeline to ensure it falls back to the original language.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """

    class DummyTranscriber:
        """A dummy transcriber class to test the transcription pipeline's language fallback
        behavior. It simulates a scenario where the transcriber detects a source language
        but does not set it, allowing us to verify that the pipeline correctly falls back
        to the original source language provided as an argument.
        """

        def __init__(self, *args, **kwargs) -> None:
            """Initialize the dummy transcriber. The parameters are not used in this dummy
            implementation, but they are accepted to match the expected signature of the
            actual transcriber.
            """
            self.src_lang = None

        def transcription(self) -> None:
            """Simulate the transcription process. In this dummy implementation, it does not
            set the src_lang attribute, allowing us to test the fallback behavior in the pipeline.
            """
            pass

        def diarization(self) -> None:
            """Simulate the diarization process. In this dummy implementation, it should not be
            called when n_speakers <= 1."""
            pytest.fail("diarization should not be called when n_speakers <= 1")

        def transcript_output(self) -> pd.DataFrame:
            """Simulate the output of the transcription process by returning a DataFrame with dummy text data.

            Returns:
                pd.DataFrame: A DataFrame containing the dummy transcription text.
            """
            return pd.DataFrame({"text": ["hola"]})

    monkeypatch.setattr(pipeline, "WhisperTranscriber", DummyTranscriber)

    df, detected_lang = pipeline.transcription_pipeline(
        file_path=Path("/tmp/audio.wav"),
        trg_lang="en",
        src_lang="es",
        model_id="base",
        task="transcribe",
        n_speakers=1,
    )

    assert list(df["text"]) == ["hola"]
    assert detected_lang == "es"


def test_translation_pipeline_returns_input_when_language_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test the translation pipeline to ensure it returns the input when the language matches.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.

    Raises:
        AssertionError: If the translation is attempted when languages match.
    """

    class DummyTranslator:
        """A dummy translator class to test the translation pipeline's behavior when the
        detected source language matches the target language. It simulates a scenario where
        the translator detects the source language as the same as the target language, and
        it raises an assertion error if the translate method is called, allowing us to verify
        that the pipeline correctly returns the input without attempting translation.
        """

        def __init__(self, **kwargs) -> None:
            """Initialize the dummy translator. The parameters are not used in this dummy
            implementation, but they are accepted to match the expected signature of the
            actual translator.
            """
            pass

        def detect_language(self, text: str) -> dict[str, str]:
            """Simulate language detection by returning a dictionary indicating that the detected
            language code is "es", which matches the target language in the test, allowing us to
            verify that the translation pipeline correctly identifies the language match and returns
            the input without translation.

            Args:
                text (str): The text for which to detect the language.

            Returns:
                dict[str, str]: A dictionary containing the detected language code.
            """
            return {"code": "es"}

        def translate(self, trg_lang: str, text: str) -> str:
            """Simulate the translation process. In this dummy implementation, it raises an assertion
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
    """
    Test the translation pipeline to ensure it translates each row correctly.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """

    class DummyTranslator:
        """A dummy translator class to test the translation pipeline's behavior when the detected source
        language does not match the target language. It simulates a scenario where the translator detects
        a source language different from the target language and records the parameters passed to the
        translate method, allowing us to verify that the pipeline correctly attempts translation for each
        row of text.
        """

        def __init__(self, **kwargs) -> None:
            """Initialize the dummy translator. The parameters are not used in this dummy implementation,
            but they are accepted to match the expected signature of the actual translator.
            """
            self.calls: list[tuple[str, str, str]] = []

        def detect_language(self, text: str) -> dict[str, str]:
            """Simulate language detection by returning a dictionary indicating that the detected language
            code is "en", which does not match the target language "es" in the test, allowing us to verify
            that the translation pipeline correctly identifies the language mismatch and attempts translation
            for each row of text.

            Args:
                text (str): The text for which to detect the language.

            Returns:
                dict[str, str]: A dictionary containing the detected language code.
            """
            return {"code": "en"}

        def translate(
            self, trg_lang: str, text: str, src_lang: str | None = None
        ) -> str:
            """Simulate the translation process by recording the parameters passed to the translate method and returning a dummy translated string that combines the original text and the target language code. This allows us to verify that the translation pipeline correctly attempts translation for each row of text when the detected source language does not match the target language.

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
    """
    Test the summarization pipeline to ensure it formats the prompt correctly.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """

    # Import the real InferencePipeline type for subclassing
    from nextext.modules.openai_cfg import InferencePipeline

    class DummyPipeline(InferencePipeline):
        """A dummy inference pipeline class to test the summarization pipeline's prompt formatting behavior."""

        def __init__(self) -> None:
            """Initialize the dummy inference pipeline and set up a list to record the prompts passed to the
            call_model method.
            """
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
            think: bool = False,
            num_ctx: int = 32768,
            temperature: float = 0.1,
            seed: int = 42,
            stop: list[str] | None = None,
            num_predict: int | None = None,
            top_k: int | None = None,
            top_p: float | None = None,
            system_prompt: str | None = None,
        ) -> str:
            """Simulate calling the model with the given prompt.

            Args:
                prompt (str): The prompt to pass to the model.
                model (str | None): Unused test double argument.
                think (bool): Unused test double argument.
                num_ctx (int): Unused test double argument.
                temperature (float): Unused test double argument.
                seed (int): Unused test double argument.
                stop (list[str] | None): Unused test double argument.
                num_predict (int | None): Unused test double argument.
                top_k (int | None): Unused test double argument.
                top_p (float | None): Unused test double argument.
                system_prompt (str | None): Unused test double argument.

            Returns:
                str: The model's response.
            """
            del (
                model,
                think,
                num_ctx,
                temperature,
                seed,
                stop,
                num_predict,
                top_k,
                top_p,
                system_prompt,
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
    """
    Test the summarization pipeline to ensure it rejects empty text.

    Args:
        ollama_pipeline (None, optional): The Ollama pipeline instance. Defaults to None.

    Raises:
        ValueError: If the input text is empty.
    """
    from nextext.modules.openai_cfg import InferencePipeline

    dummy_pipeline = InferencePipeline()
    with pytest.raises(ValueError):
        pipeline.summarization_pipeline("", dummy_pipeline)


def test_wordlevel_pipeline_invokes_all_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test the word-level pipeline to ensure all steps are invoked.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture for modifying behavior.
    """

    class DummyWordCounter:
        """A dummy word counter class to test the word-level pipeline's behavior."""

        def __init__(self, text: str, language: str) -> None:
            """Initialize the dummy word counter with the given text and language, and set up a list to record
            the steps invoked.

            Args:
                text (str): The text to be processed.
                language (str): The language of the text.
            """
            self.text = text
            self.language = language
            self.steps: list[str] = []

        def text_to_doc(self) -> None:
            """Simulate the process of converting text to a document format. In this dummy implementation,
            it simply records that the text_to_doc step was invoked, allowing us to verify that the word-level
            pipeline correctly calls this step as part of its processing.
            """
            self.steps.append("text_to_doc")

        def lemmatize_doc(self) -> None:
            """Simulate the process of lemmatizing a document. In this dummy implementation, it simply records
            that the lemmatize_doc step was invoked, allowing us to verify that the word-level pipeline correctly
            calls this step as part of its processing.
            """
            self.steps.append("lemmatize_doc")

        def count_words(self) -> pd.DataFrame:
            """Simulate the process of counting words in the text. In this dummy implementation, it returns a
            DataFrame with a single word and its count, allowing us to verify that the word-level pipeline correctly
            calls this step and processes its output.

            Returns:
                pd.DataFrame: A DataFrame containing a single word and its count.
            """
            return pd.DataFrame({"word": ["test"], "count": [1]})

        def named_entity_recognition(self) -> pd.DataFrame:
            """Simulate the process of named entity recognition. In this dummy implementation, it returns a DataFrame
            with a single entity, allowing us to verify that the word-level pipeline correctly calls this step and
            processes its output.

            Returns:
                pd.DataFrame: A DataFrame containing a single entity.
            """
            return pd.DataFrame({"entity": ["Test"]})

        def get_noun_sentiment(self) -> pd.DataFrame:
            """Simulate the process of getting noun sentiment. In this dummy implementation, it returns a DataFrame
            with a single noun and its sentiment score, allowing us to verify that the word-level pipeline correctly
            calls this step and processes its output.

            Returns:
                pd.DataFrame: A DataFrame containing a single noun and its sentiment score.
            """
            return pd.DataFrame({"noun": ["test"], "sentiment": [0.5]})

        def create_interactive_graph(self) -> str:
            """Simulate the process of creating an interactive graph. In this dummy implementation, it returns a string
            representing the path to the graph, allowing us to verify that the word-level pipeline correctly calls this
            step and processes its output.

            Returns:
                str: A string representing the path to the interactive graph.
            """
            return "graph.html"

        def create_wordcloud(self) -> str:
            """Simulate the process of creating a word cloud. In this dummy implementation, it returns a string
            representing the path to the word cloud, allowing us to verify that the word-level pipeline correctly calls this
            step and processes its output.

            Returns:
                str: A string representing the path to the word cloud.
            """
            return "wordcloud"

    monkeypatch.setattr(
        pipeline, "WordCounter", lambda text, language: DummyWordCounter(text, language)
    )
    df = pd.DataFrame({"text": ["alpha", "beta"]})

    counts, entities, sentiments, graph, wordcloud = pipeline.wordlevel_pipeline(
        df, "en"
    )

    assert list(counts["word"]) == ["test"]
    assert list(entities["entity"]) == ["Test"]
    assert list(sentiments["noun"]) == ["test"]
    assert graph == "graph.html"
    assert wordcloud == "wordcloud"
