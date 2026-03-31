import pandas as pd

from nextext.modules.transcription import WhisperTranscriber


def _build_transcriber() -> WhisperTranscriber:
    """Build a WhisperTranscriber instance for testing.

    Returns:
        WhisperTranscriber: A WhisperTranscriber instance with default column settings.
    """
    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.start_column = "start"
    transcriber.end_column = "end"
    transcriber.text_column = "text"
    transcriber.speaker_column = "speaker"
    return transcriber


def test_merge_transcriptions_keeps_final_sentence_without_terminal_punctuation() -> (
    None
):
    """Test that `_merge_transcriptions_by_sentence` correctly merges transcriptions into sentences
    even when the final sentence does not end with terminal punctuation.
    """
    transcriber = _build_transcriber()
    data = pd.DataFrame(
        [
            {"start": "0:00:00", "end": "0:00:01", "text": "hello"},
            {"start": "0:00:01", "end": "0:00:02", "text": "world"},
        ]
    )

    merged = transcriber._merge_transcriptions_by_sentence(data)

    assert list(merged.columns) == ["start", "end", "text"]
    assert merged.to_dict("records") == [
        {"start": "0:00:00", "end": "0:00:02", "text": "hello world"}
    ]


def test_merge_transcriptions_handles_arabic_question_mark() -> None:
    """Test that `_merge_transcriptions_by_sentence` correctly merges transcriptions into sentences
    when the text contains Arabic question marks (؟) as terminal punctuation.
    """
    transcriber = _build_transcriber()
    data = pd.DataFrame(
        [
            {"start": "0:00:00", "end": "0:00:01", "text": "مرحبا"},
            {"start": "0:00:01", "end": "0:00:02", "text": "كيف الحال؟"},
        ]
    )

    merged = transcriber._merge_transcriptions_by_sentence(data)

    assert merged.to_dict("records") == [
        {"start": "0:00:00", "end": "0:00:02", "text": "مرحبا كيف الحال؟"}
    ]


def test_configure_torch_safe_globals_registers_omegaconf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that Torch safe globals include OmegaConf config classes."""
    recorded_globals: list[type] = []

    def fake_add_safe_globals(classes: list[type]) -> None:
        recorded_globals.extend(classes)

    monkeypatch.setattr(
        transcription.torch.serialization,
        "add_safe_globals",
        fake_add_safe_globals,
    )

    _configure_torch_safe_globals()

    class_names = {registered.__name__ for registered in recorded_globals}
    assert "DictConfig" in class_names
    assert "ListConfig" in class_names
