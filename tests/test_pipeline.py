from pathlib import Path

import pytest

from nextext import pipeline

pd = pytest.importorskip("pandas")


@pytest.fixture
def disable_docker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:  # type: ignore[override]
        if path.as_posix() == "/.dockerenv":
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)


@pytest.fixture
def enable_docker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:  # type: ignore[override]
        if path.as_posix() == "/.dockerenv":
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)


def test_get_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "find_dotenv", lambda: "")
    monkeypatch.setattr(pipeline, "load_dotenv", lambda path: None)
    monkeypatch.setenv("API_TOKEN", "secret")

    assert pipeline.get_api_key("API_TOKEN") == "secret"


def test_get_api_key_prompts_and_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, disable_docker_env: None
) -> None:
    dotenv_path = tmp_path / ".env"
    saved: dict[str, str] = {}

    monkeypatch.setattr(pipeline, "find_dotenv", lambda: str(dotenv_path))
    monkeypatch.setattr(pipeline, "load_dotenv", lambda path: None)
    monkeypatch.setattr(pipeline.getpass, "getpass", lambda prompt: "prompted-key")

    def fake_set_key(path: str, token: str, value: str) -> None:
        saved["path"] = path
        saved["token"] = token
        saved["value"] = value

    monkeypatch.setattr(pipeline, "set_key", fake_set_key)
    monkeypatch.delenv("TEST_TOKEN", raising=False)

    result = pipeline.get_api_key("TEST_TOKEN")

    assert result == "prompted-key"
    assert saved == {
        "path": str(dotenv_path),
        "token": "TEST_TOKEN",
        "value": "prompted-key",
    }


def test_get_api_key_docker_environment_raises(
    monkeypatch: pytest.MonkeyPatch, enable_docker_env: None
) -> None:
    monkeypatch.setattr(pipeline, "find_dotenv", lambda: "")
    monkeypatch.setattr(pipeline, "load_dotenv", lambda path: None)
    monkeypatch.delenv("API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        pipeline.get_api_key("API_KEY")


def test_transcription_pipeline_invokes_transcriber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyTranscriber:
        instance: "DummyTranscriber"

        def __init__(
            self,
            file_path: Path,
            auth_token: str,
            trg_lang: str,
            src_lang: str,
            model_id: str,
            task: str,
            n_speakers: int,
        ) -> None:
            self.params = {
                "file_path": file_path,
                "auth_token": auth_token,
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
            self.transcription_called = True

        def diarization(self) -> None:
            self.diarization_called = True

        def transcript_output(self) -> pd.DataFrame:
            return pd.DataFrame({"text": ["bonjour"]})

    monkeypatch.setattr(pipeline, "WhisperTranscriber", DummyTranscriber)

    df, detected_lang = pipeline.transcription_pipeline(
        file_path=Path("/tmp/audio.wav"),
        api_key="token",
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
    class DummyTranscriber:
        def __init__(self, *args, **kwargs) -> None:
            self.src_lang = None

        def transcription(self) -> None:
            pass

        def diarization(self) -> None:
            pytest.fail("diarization should not be called when n_speakers <= 1")

        def transcript_output(self) -> pd.DataFrame:
            return pd.DataFrame({"text": ["hola"]})

    monkeypatch.setattr(pipeline, "WhisperTranscriber", DummyTranscriber)

    df, detected_lang = pipeline.transcription_pipeline(
        file_path=Path("/tmp/audio.wav"),
        api_key="token",
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
    class DummyTranslator:
        def __init__(self) -> None:
            pass

        def detect_language(self, text: str) -> dict[str, str]:
            return {"code": "es"}

        def translate(self, trg_lang: str, text: str) -> str:
            raise AssertionError("translate should not be called when languages match")

    monkeypatch.setattr(pipeline, "Translator", DummyTranslator)
    df = pd.DataFrame({"text": ["hola"]})

    result = pipeline.translation_pipeline(df.copy(), "es")

    pd.testing.assert_frame_equal(result, df)


def test_translation_pipeline_translates_each_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyTranslator:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def detect_language(self, text: str) -> dict[str, str]:
            return {"code": "en"}

        def translate(self, trg_lang: str, text: str) -> str:
            self.calls.append((trg_lang, text))
            return f"{text}-{trg_lang}"

    translator = DummyTranslator()
    monkeypatch.setattr(pipeline, "Translator", lambda: translator)
    df = pd.DataFrame({"text": ["hello", "world"]})

    result = pipeline.translation_pipeline(df.copy(), "es")

    assert translator.calls == [("es", "hello"), ("es", "world")]
    assert list(result["text"]) == ["hello-es", "world-es"]


def test_summarization_pipeline_formats_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyPipeline:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def load_prompt(self, keyword: str) -> str:
            assert keyword == "summary"
            return "Summarize: {text}"

        def call_ollama_server(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "summary result"

    ollama_pipeline = DummyPipeline()

    result = pipeline.summarization_pipeline("Important content", ollama_pipeline)

    assert result == "summary result"
    assert ollama_pipeline.prompts == ["Summarize: Important content"]


def test_summarization_pipeline_rejects_empty_text(
    ollama_pipeline: None = None,
) -> None:
    with pytest.raises(ValueError):
        pipeline.summarization_pipeline("", object())


def test_wordlevel_pipeline_invokes_all_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyWordCounter:
        def __init__(self, text: str, language: str) -> None:
            self.text = text
            self.language = language
            self.steps: list[str] = []

        def text_to_doc(self) -> None:
            self.steps.append("text_to_doc")

        def lemmatize_doc(self) -> None:
            self.steps.append("lemmatize_doc")

        def count_words(self) -> pd.DataFrame:
            return pd.DataFrame({"word": ["test"], "count": [1]})

        def named_entity_recognition(self) -> pd.DataFrame:
            return pd.DataFrame({"entity": ["Test"]})

        def get_noun_sentiment(self) -> pd.DataFrame:
            return pd.DataFrame({"noun": ["test"], "sentiment": [0.5]})

        def create_interactive_graph(self) -> str:
            return "graph.html"

        def create_wordcloud(self) -> str:
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


def test_topics_pipeline_runs_all_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTopicModeling:
        instance: "DummyTopicModeling"

        def __init__(self, data: list[str], lang_code: str) -> None:
            self.data = data
            self.lang_code = lang_code
            self.loaded = False
            self.fitted = False
            DummyTopicModeling.instance = self

        def load_pipeline(self) -> None:
            self.loaded = True

        def fit_topic_model(self) -> None:
            self.fitted = True

        def summarize_topics(self, ollama_pipeline) -> list[tuple[str, str]]:
            assert self.loaded is True
            assert self.fitted is True
            assert ollama_pipeline == "ollama"
            return [("Topic", "Summary")]

    monkeypatch.setattr(pipeline, "TopicModeling", DummyTopicModeling)
    df = pd.DataFrame({"text": ["text one", "text two"]})

    result = pipeline.topics_pipeline(df, "en", "ollama")

    assert result == [("Topic", "Summary")]
    instance = DummyTopicModeling.instance
    assert instance.data == ["text one", "text two"]
    assert instance.lang_code == "en"


def test_hatespeech_pipeline_appends_results(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyDetector:
        def __init__(self, ollama_pipeline) -> None:
            self.ollama = ollama_pipeline

        def process(self, texts: list[str]) -> list[str]:
            return ["safe" for _ in texts]

    monkeypatch.setattr(pipeline, "HateSpeechDetector", DummyDetector)
    df = pd.DataFrame({"text": ["hello", "world"]})

    result = pipeline.hatespeech_pipeline("ollama", df.copy())

    assert list(result["hate_speech"]) == ["safe", "safe"]


def test_hatespeech_pipeline_no_results_keeps_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyDetector:
        def __init__(self, ollama_pipeline) -> None:
            pass

        def process(self, texts: list[str]) -> list[str]:
            return []

    monkeypatch.setattr(pipeline, "HateSpeechDetector", DummyDetector)
    df = pd.DataFrame({"text": ["hello"]})

    result = pipeline.hatespeech_pipeline("ollama", df.copy())

    assert "hate_speech" not in result
