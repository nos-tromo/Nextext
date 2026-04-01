"""Tests for the Nextext model preload utilities."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nextext.utils import model_loader


def test_get_spacy_model_dir_uses_env_var(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that the spaCy cache directory honors the environment override.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables.
        tmp_path (Path): The pytest fixture providing a temporary directory for testing.
    """
    cache_dir = tmp_path / "spacy-cache"
    monkeypatch.setenv(model_loader.SPACY_MODEL_DIR, str(cache_dir))

    assert model_loader.get_spacy_model_dir() == cache_dir


def test_download_spacy_model_skips_cached_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that cached spaCy models are not downloaded again.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables and functions.
        tmp_path (Path): The pytest fixture providing a temporary directory for testing.
    """
    cache_dir = tmp_path / "spacy-cache"
    model_dir = cache_dir / "en_core_web_sm"
    model_dir.mkdir(parents=True)
    monkeypatch.setenv(model_loader.SPACY_MODEL_DIR, str(cache_dir))

    def fail_run(*args: object, **kwargs: object) -> None:
        """Simulate a failure if subprocess.run is called, indicating that the download function attempted to run
        when it should have been skipped due to the model already being cached. This helps verify that the caching
        mechanism is working correctly and prevents unnecessary downloads of spaCy models that are already available
        in the specified cache directory.

        Raises:
            AssertionError: Always raised to indicate that subprocess.run should not be called for cached models.
        """
        raise AssertionError("subprocess.run should not be called for cached models")

    monkeypatch.setattr(model_loader.subprocess, "run", fail_run)

    model_loader.download_spacy_model("en_core_web_sm")

    assert str(cache_dir) in sys.path


def test_download_spacy_model_uses_persistent_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that spaCy downloads target the persistent cache directory.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables and functions.
        tmp_path (Path): The pytest fixture providing a temporary directory for testing.
    """
    cache_dir = tmp_path / "spacy-cache"
    monkeypatch.setenv(model_loader.SPACY_MODEL_DIR, str(cache_dir))
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        """Simulate the subprocess.run function to capture the command that would be executed for downloading
        a spaCy model. This allows us to verify that the download command is correctly targeting the specified
        cache directory, ensuring that the model is being downloaded to the intended location rather than a
        default or incorrect path.
        """
        captured.append(cmd)

    monkeypatch.setattr(model_loader.subprocess, "run", fake_run)
    monkeypatch.setattr(model_loader.importlib, "invalidate_caches", lambda: None)

    model_loader.download_spacy_model("en_core_web_sm")

    assert captured == [
        [
            sys.executable,
            "-m",
            "spacy",
            "download",
            "en_core_web_sm",
            "--target",
            str(cache_dir),
        ]
    ]


def test_get_whisper_model_ids_includes_detection_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that Whisper preload includes the language-detection model.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables and functions.
    """
    monkeypatch.setattr(
        model_loader,
        "load_mappings",
        lambda _: {"default_transcribe": "turbo", "default_translate": "large-v3"},
    )

    model_ids = model_loader.get_whisper_model_ids()

    assert model_ids == ["large-v3", "small", "turbo"]


def test_get_alignment_model_ids_combines_torch_and_hf_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that alignment model discovery merges Torch and HF defaults.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables and functions.
    """
    fake_alignment = SimpleNamespace(
        DEFAULT_ALIGN_MODELS_TORCH={"en": "WAV2VEC2_ASR_BASE_960H"},
        DEFAULT_ALIGN_MODELS_HF={"de": "VOXPOPULI_ASR_BASE_10K_DE"},
    )
    monkeypatch.setitem(sys.modules, "whisperx.alignment", fake_alignment)

    alignment_models = model_loader.get_alignment_model_ids()

    assert alignment_models == {
        "de": "VOXPOPULI_ASR_BASE_10K_DE",
        "en": "WAV2VEC2_ASR_BASE_960H",
    }


def test_main_preloads_expected_model_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the main preload routine covers all configured model groups.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables and functions.
    """
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        model_loader,
        "_get_default_device",
        lambda: "cpu",
    )
    monkeypatch.setattr(
        model_loader,
        "ensure_nltk_resources",
        lambda: calls.append(("nltk", "all")),
    )
    monkeypatch.setattr(
        model_loader,
        "get_spacy_model_ids",
        lambda: ["en_core_web_sm"],
    )
    monkeypatch.setattr(
        model_loader,
        "download_spacy_model",
        lambda model_id: calls.append(("spacy", model_id)),
    )
    monkeypatch.setattr(
        model_loader,
        "get_whisper_model_ids",
        lambda: ["small", "turbo"],
    )
    monkeypatch.setattr(
        model_loader,
        "preload_whisper_model",
        lambda model_id, device: calls.append((f"whisper:{device}", model_id)),
    )
    monkeypatch.setattr(
        model_loader,
        "get_alignment_model_ids",
        lambda: {"en": "WAV2VEC2_ASR_BASE_960H"},
    )
    monkeypatch.setattr(
        model_loader,
        "preload_alignment_model",
        lambda language_code, model_id, device: calls.append(
            (f"alignment:{device}", f"{language_code}:{model_id}")
        ),
    )
    monkeypatch.setattr(
        model_loader,
        "preload_diarization_model",
        lambda auth_token, device: calls.append(
            (f"diarization:{device}", auth_token or "")
        ),
    )
    monkeypatch.setenv("HF_HUB_TOKEN", "secret-token")

    model_loader.main()

    assert calls == [
        ("nltk", "all"),
        ("spacy", "en_core_web_sm"),
        ("whisper:cpu", "small"),
        ("whisper:cpu", "turbo"),
        ("alignment:cpu", "en:WAV2VEC2_ASR_BASE_960H"),
        ("diarization:cpu", "secret-token"),
    ]
