"""Pure UI helpers shared by the Streamlit frontend.

These helpers were previously defined inline in ``nextext/app.py``; moving
them here removes their dependency on the in-process pipeline and keeps
the rendering logic side-effect free for testing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def check_batch_within_limit(
    files: Sequence[Any],
    max_total_bytes: int,
) -> str | None:
    """Validate that a selected upload batch fits the in-memory cap.

    Streamlit's ``file_uploader`` holds every selected file in the Streamlit
    server's memory at once, so a multi-GB batch can exhaust RAM before any
    pipeline work begins. This guard sums the reported file sizes and returns
    an actionable message when the batch exceeds ``max_total_bytes`` instead
    of letting the process get OOM-killed.

    Args:
        files: Selected Streamlit ``UploadedFile`` objects (anything exposing
            a numeric ``size`` attribute). Entries without a usable ``size``
            count as zero bytes.
        max_total_bytes: Maximum combined batch size, in bytes.

    Returns:
        str | None: ``None`` when the batch is within the cap; otherwise a
            human-readable error message naming the limit and the CLI escape
            hatch for large local batches.
    """
    total = 0
    for file in files:
        size = getattr(file, "size", 0)
        if isinstance(size, int) and size > 0:
            total += size
    if total <= max_total_bytes:
        return None
    gib = 1 << 30
    return (
        f"Selected files total {total / gib:.1f} GB, over the "
        f"{max_total_bytes / gib:.1f} GB limit. The web UI keeps the whole batch "
        "in memory — upload fewer files at once, raise NEXTEXT_MAX_BATCH_MB if the "
        "frontend has the RAM, or use `nextext-cli` for large local batches."
    )


def default_target_language(
    language_maps: dict[str, str],
    language_names: Sequence[str],
) -> tuple[int, str]:
    """Resolve a safe default target language for the translation UI.

    Args:
        language_maps: Mapping of language codes to labels.
        language_names: Sorted language labels shown in the UI.

    Returns:
        tuple[int, str]: The default selectbox index and language code.
    """
    preferred_codes = ("de-DE", "de", "en")
    for language_code in preferred_codes:
        language_name = language_maps.get(language_code)
        if language_name in language_names:
            return language_names.index(language_name), language_code

    if not language_names:
        return 0, "en"

    fallback_name = language_names[0]
    fallback_code = {v: k for k, v in language_maps.items()}.get(fallback_name, "en")
    return 0, fallback_code


def progress_value(
    file_index: int,
    total_files: int,
    stage_index: int,
    total_stages: int,
) -> float:
    """Calculate overall progress for sequential multi-file processing.

    Args:
        file_index: One-based file index currently being processed.
        total_files: Total number of files in the run.
        stage_index: Zero-based pipeline stage index.
        total_stages: Total number of pipeline stages.

    Returns:
        float: A normalized progress value between 0.0 and 1.0.

    Raises:
        ValueError: If any numeric input is outside the supported range.
    """
    if total_files <= 0:
        raise ValueError("total_files must be greater than zero.")
    if total_stages <= 0:
        raise ValueError("total_stages must be greater than zero.")
    if file_index < 1 or file_index > total_files:
        raise ValueError("file_index must be within the file count.")
    if stage_index < 0 or stage_index >= total_stages:
        raise ValueError("stage_index must be within the stage count.")

    completed_files = file_index - 1
    stage_progress = stage_index / total_stages
    return (completed_files + stage_progress) / total_files


def result_file_names(results: Sequence[dict[str, Any]]) -> list[str]:
    """Return stable labels for processed result entries.

    Args:
        results: Stored result entries.

    Returns:
        list[str]: File names displayed in the result selector.
    """
    file_names: list[str] = []
    for index, result in enumerate(results, start=1):
        file_names.append(str(result.get("file_name", f"File {index}")))
    return file_names


def download_file_name(result: dict[str, Any], suffix: str) -> str:
    """Build a download filename based on the original uploaded file name.

    Args:
        result: Stored result payload for one file.
        suffix: Output suffix including the desired extension.

    Returns:
        str: A download filename derived from the original upload name.
    """
    original_name = str(result.get("file_name", "result"))
    original_stem = Path(original_name).stem or "result"
    return f"{original_stem}_{suffix}"


def select_result(
    results: Sequence[dict[str, Any]],
    selected_file_name: str | None,
) -> dict[str, Any]:
    """Select the requested result entry or fall back to the first one.

    Args:
        results: Stored result entries.
        selected_file_name: Requested file name.

    Returns:
        dict[str, Any]: The matching result entry.

    Raises:
        ValueError: If no results are available.
    """
    if not results:
        raise ValueError("At least one result entry is required.")

    if selected_file_name:
        for result in results:
            if result.get("file_name") == selected_file_name:
                return result
    return dict(results[0])


def transcript_list_to_dataframe(segments: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Convert the JSON transcript list to a Pandas DataFrame for the UI.

    Args:
        segments: List of transcript segment dicts as returned by the API.

    Returns:
        pd.DataFrame: A DataFrame with ``start``, ``end``, ``speaker``, ``text``
            columns. ``speaker`` is dropped when every entry is ``None``.
    """
    if not segments:
        return pd.DataFrame(columns=["start", "end", "speaker", "text"])
    df = pd.DataFrame(list(segments))
    if "speaker" in df.columns and df["speaker"].isna().all():
        df = df.drop(columns=["speaker"])
    columns = [col for col in ("start", "end", "speaker", "text") if col in df.columns]
    return df[columns]


def word_counts_list_to_dataframe(
    rows: Sequence[dict[str, Any]] | None,
) -> pd.DataFrame:
    """Convert a word-counts JSON list to a Pandas DataFrame.

    Args:
        rows: List of ``{word, count}`` dicts from the API, or ``None``.

    Returns:
        pd.DataFrame: A DataFrame with the original two columns; empty when
            ``rows`` is ``None`` or empty.
    """
    if not rows:
        return pd.DataFrame(columns=["word", "count"])
    df = pd.DataFrame(list(rows))
    return df[["word", "count"]]


def named_entities_list_to_dataframe(
    rows: Sequence[dict[str, Any]] | None,
) -> pd.DataFrame:
    """Convert a named-entities JSON list to a Pandas DataFrame.

    Args:
        rows: List of ``{entity, category, frequency}`` dicts.

    Returns:
        pd.DataFrame: A DataFrame with the legacy ``Entity``/``Category``/``Frequency``
            column names that the rest of the UI references.
    """
    if not rows:
        return pd.DataFrame(columns=["Entity", "Category", "Frequency"])
    df = pd.DataFrame(list(rows))
    return df.rename(columns={"entity": "Entity", "category": "Category", "frequency": "Frequency"})[
        ["Entity", "Category", "Frequency"]
    ]
