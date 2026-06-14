"""Guards for the no-local-ML invariant.

All model inference runs on external endpoints, so the backend environment
must stay free of the heavyweight ML runtimes. camel-tools declares torch /
transformers in its metadata for morphology models Nextext never imports;
``[tool.uv] override-dependencies`` excludes them, and these tests pin that
invariant so a future lock regeneration cannot silently reintroduce the
packages.
"""

import importlib.util

import pytest


@pytest.mark.parametrize(
    "package",
    ["torch", "transformers", "pyannote", "gliner", "whisper", "accelerate"],
)
def test_local_ml_runtimes_are_not_installed(package: str) -> None:
    """The torch-family ML runtimes must be absent from the environment.

    Args:
        package (str): Importable package name that must not resolve.
    """
    assert importlib.util.find_spec(package) is None, (
        f"'{package}' is installed — the no-local-ML invariant is broken. "
        "Check [tool.uv] override-dependencies in pyproject.toml and the "
        "project dependency list."
    )


def test_camel_tools_tokenizer_works_without_torch() -> None:
    """The Arabic tokenizer path of camel-tools must work without torch.

    camel-tools is kept solely for ``simple_word_tokenize`` (a pure-Python
    regex tokenizer); this asserts the override of its torch / transformers
    requirements does not break that one code path.
    """
    from camel_tools.tokenizers.word import simple_word_tokenize

    assert simple_word_tokenize("مرحبا بالعالم.") == ["مرحبا", "بالعالم", "."]
