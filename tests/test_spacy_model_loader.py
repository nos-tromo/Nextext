import sys
from pathlib import Path

from nextext.utils import spacy_model_loader


def test_get_spacy_model_dir_uses_env_var(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that `get_spacy_model_dir` uses the environment variable if set.

    Args:
        monkeypatch (_pytest.monkeypatch.MonkeyPatch): Fixture to modify environment variables.
        tmp_path (Path): Temporary directory for test files.
    """
    cache_dir = tmp_path / "spacy-cache"
    monkeypatch.setenv(spacy_model_loader.SPACY_MODEL_DIR, str(cache_dir))

    assert spacy_model_loader.get_spacy_model_dir() == cache_dir


def test_download_spacy_model_skips_cached_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that `download_spacy_model` does not attempt to download a model that is already cached.

    Args:
        monkeypatch (_pytest.monkeypatch.MonkeyPatch): Fixture to modify environment variables.
        tmp_path (Path): Temporary directory for test files.

    Raises:
        AssertionError: If `subprocess.run` is called for a cached model.
    """
    cache_dir = tmp_path / "spacy-cache"
    model_dir = cache_dir / "en_core_web_sm"
    model_dir.mkdir(parents=True)
    monkeypatch.setenv(spacy_model_loader.SPACY_MODEL_DIR, str(cache_dir))

    def fail_run(*args, **kwargs) -> None:
        raise AssertionError("subprocess.run should not be called for cached models")

    monkeypatch.setattr(spacy_model_loader.subprocess, "run", fail_run)

    spacy_model_loader.download_spacy_model("en_core_web_sm")

    assert str(cache_dir) in sys.path


def test_download_spacy_model_uses_persistent_target(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that `download_spacy_model` uses the persistent cache directory as the target for downloading models.

    Args:
        monkeypatch (_pytest.monkeypatch.MonkeyPatch): Fixture to modify environment variables.
        tmp_path (Path): Temporary directory for test files.
    """
    cache_dir = tmp_path / "spacy-cache"
    monkeypatch.setenv(spacy_model_loader.SPACY_MODEL_DIR, str(cache_dir))
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        captured.append(cmd)

    monkeypatch.setattr(spacy_model_loader.subprocess, "run", fake_run)
    monkeypatch.setattr(spacy_model_loader.importlib, "invalidate_caches", lambda: None)

    spacy_model_loader.download_spacy_model("en_core_web_sm")

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
