"""Packaging guard: the wheel ships the application, not its dev tooling."""

import tomllib
from fnmatch import fnmatch
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _discovered(name: str, include: list[str], exclude: list[str]) -> bool:
    """Apply setuptools' ``packages.find`` include/exclude patterns to a package name.

    Args:
        name (str): Dotted package name.
        include (list[str]): Include patterns (``["*"]`` when unset, as in setuptools).
        exclude (list[str]): Exclude patterns.

    Returns:
        bool: Whether package discovery would pick the package up.
    """
    return any(fnmatch(name, pattern) for pattern in include) and not any(fnmatch(name, pattern) for pattern in exclude)


def test_package_discovery_keeps_the_eval_harness_out_of_the_wheel() -> None:
    """``eval/`` is a dev-only directory, never a distributed package; ``nextext`` and its subpackages are."""
    find = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"]["setuptools"]["packages"]["find"]
    include = find.get("include", ["*"])
    exclude = find.get("exclude", [])

    assert _discovered("nextext", include, exclude)
    assert _discovered("nextext.core", include, exclude)
    assert not _discovered("eval", include, exclude)
    assert not _discovered("eval.hate_speech", include, exclude)
