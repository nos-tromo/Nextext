"""Tests for the Nextext model preload utilities."""

import sys
from pathlib import Path

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
        """Fail if subprocess.run is called — the cached model should short-circuit the download.

        Lets the test assert the cache check ran and skipped the spaCy download.

        Raises:
            AssertionError: Always raised — subprocess.run must not be called for cached models.
        """
        raise AssertionError("subprocess.run should not be called for cached models")

    monkeypatch.setattr(model_loader.subprocess, "run", fail_run)

    model_loader.download_spacy_model("en_core_web_sm")

    assert str(cache_dir) in sys.path


def test_download_spacy_model_uses_persistent_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that spaCy wheel installs target the persistent cache directory.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables and functions.
        tmp_path (Path): The pytest fixture providing a temporary directory for testing.
    """
    cache_dir = tmp_path / "spacy-cache"
    monkeypatch.setenv(model_loader.SPACY_MODEL_DIR, str(cache_dir))
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        """Capture the command subprocess.run would execute for the spaCy download.

        Lets the test assert the command targets the configured cache directory.

        Args:
            cmd (list[str]): The command that would be executed by subprocess.run, captured for verification.

            check (bool): A flag indicating whether to check the command execution result, included for signature
        """
        captured.append(cmd)

    monkeypatch.setattr(model_loader.subprocess, "run", fake_run)
    monkeypatch.setattr(model_loader.importlib, "invalidate_caches", lambda: None)

    model_loader.download_spacy_model("en_core_web_sm")

    assert captured == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(cache_dir),
            (
                "https://github.com/explosion/spacy-models/releases/download/"
                "en_core_web_sm-3.8.0/"
                "en_core_web_sm-3.8.0-py3-none-any.whl"
            ),
        ]
    ]


def test_get_spacy_model_package_version_uses_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the spaCy model package version honors the environment override.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables.
    """
    monkeypatch.setenv(model_loader.SPACY_MODEL_PACKAGE_VERSION, "3.7.2")

    assert model_loader.get_spacy_model_package_version() == "3.7.2"


def test_get_spacy_model_package_version_uses_spacy_minor_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the spaCy model package version is derived from the installed spaCy version.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying module behavior.
    """
    monkeypatch.delenv(model_loader.SPACY_MODEL_PACKAGE_VERSION, raising=False)
    monkeypatch.setattr(model_loader, "package_version", lambda _: "3.8.4")

    assert model_loader.get_spacy_model_package_version() == "3.8.0"


def test_get_spacy_model_download_url_uses_override_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the direct spaCy download URL honors the configured base URL.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables.
    """
    monkeypatch.setenv(
        model_loader.SPACY_MODEL_DOWNLOAD_BASE_URL,
        "https://mirror.example.com/spacy",
    )
    monkeypatch.setenv(model_loader.SPACY_MODEL_PACKAGE_VERSION, "3.8.0")

    assert model_loader.get_spacy_model_download_url("en_core_web_sm") == (
        "https://mirror.example.com/spacy/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    )


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
        "LOCAL_WHISPER_MODEL_IDS",
        ("large-v3-turbo", "large-v3"),
    )
    monkeypatch.setattr(
        model_loader,
        "preload_whisper_model",
        lambda model_id, device: calls.append((f"whisper:{device}", model_id)),
    )
    monkeypatch.setattr(
        model_loader,
        "preload_diarization_model",
        lambda auth_token, device: calls.append((f"diarization:{device}", auth_token or "")),
    )
    monkeypatch.setattr(
        model_loader,
        "preload_gliner_model",
        lambda: calls.append(("gliner", model_loader.GLINER_MODEL_ID)),
    )
    monkeypatch.setattr(
        model_loader,
        "preload_translation_model",
        lambda **kwargs: calls.append(("translation", "")),
    )
    monkeypatch.setenv("HF_HUB_TOKEN", "secret-token")

    model_loader.main()

    assert calls == [
        ("nltk", "all"),
        ("spacy", "en_core_web_sm"),
        ("whisper:cpu", "large-v3-turbo"),
        ("whisper:cpu", "large-v3"),
        ("diarization:cpu", "secret-token"),
        ("gliner", model_loader.GLINER_MODEL_ID),
        ("translation", ""),
    ]


def test_preload_translation_model_skips_when_no_model_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that translation model preload is skipped when TRANSLATION_MODEL is unset.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment
            variables and functions.
    """
    monkeypatch.delenv(model_loader.TRANSLATION_MODEL_ENV, raising=False)

    def should_not_be_called(*args: object, **kwargs: object) -> None:
        """Raise if called.

        Raises:
            AssertionError: Always, to signal an unexpected call.
        """
        raise AssertionError("Should not be called when TRANSLATION_MODEL is unset")

    monkeypatch.setattr(model_loader.subprocess, "run", should_not_be_called)
    monkeypatch.setattr(model_loader.huggingface_hub, "snapshot_download", should_not_be_called)

    model_loader.preload_translation_model()


def test_preload_translation_model_ollama_pulls_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the Ollama provider pulls the translation model via the CLI.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment
            variables and functions.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv(model_loader.TRANSLATION_MODEL_ENV, "llama3")
    captured: list[tuple[list[str], bool]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        """Capture the subprocess command.

        Args:
            cmd (list[str]): The command passed to subprocess.run.
            check (bool): The check flag passed to subprocess.run.
        """
        captured.append((cmd, check))

    monkeypatch.setattr(model_loader.subprocess, "run", fake_run)

    model_loader.preload_translation_model()

    assert captured == [(["ollama", "pull", "llama3"], True)]


def test_preload_translation_model_vllm_downloads_from_hf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the vLLM provider downloads the translation model from Hugging Face Hub.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment
            variables and functions.
    """
    model = "Infomaniak-AI/vllm-translategemma-4b-it"
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv(model_loader.TRANSLATION_MODEL_ENV, model)
    captured: list[dict[str, object]] = []

    def fake_snapshot_download(model_id: str, token: str | None = None) -> None:
        """Capture the snapshot_download call.

        Args:
            model_id (str): The model ID passed to snapshot_download.
            token (str | None): The token passed to snapshot_download.
        """
        captured.append({"model_id": model_id, "token": token})

    monkeypatch.setattr(model_loader.huggingface_hub, "snapshot_download", fake_snapshot_download)

    model_loader.preload_translation_model(hf_token="hf-token")

    assert captured == [{"model_id": model, "token": "hf-token"}]


def test_preload_translation_model_openai_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the OpenAI provider does not trigger any local download.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment
            variables and functions.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.setenv(model_loader.TRANSLATION_MODEL_ENV, "gpt-4o")

    def should_not_be_called(*args: object, **kwargs: object) -> None:
        """Raise if called.

        Raises:
            AssertionError: Always, to signal an unexpected call.
        """
        raise AssertionError("No local download should occur for the OpenAI provider")

    monkeypatch.setattr(model_loader.subprocess, "run", should_not_be_called)
    monkeypatch.setattr(model_loader.huggingface_hub, "snapshot_download", should_not_be_called)

    model_loader.preload_translation_model()
