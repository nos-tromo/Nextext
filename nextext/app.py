"""Compatibility shim for the legacy Streamlit entry point.

The Streamlit UI now lives in :mod:`nextext.frontend.app` and talks to the
FastAPI backend over HTTP. This module preserves the historical
``nextext.app`` import surface so the existing test suite and any external
``streamlit run nextext/app.py`` invocations continue to work without
modification.

It exposes:

- ``main`` / ``cli`` — the Streamlit entry points (re-exported).
- Pure UI helpers that previously lived here (re-exported from
  :mod:`nextext.frontend.state` and :mod:`nextext.core.docint_transcript`).
- ``_build_results_archive`` / ``_build_docint_jsonl_archive`` — the legacy
  multi-result ZIP builders, kept here because the test suite drives them
  directly. The runtime UI calls :mod:`nextext.api.artifacts` for per-job
  archives instead.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
from matplotlib.figure import Figure

from nextext.core.docint_transcript import (
    build_docint_jsonl,
)
from nextext.core.docint_transcript import (
    parse_hhmmss_to_seconds as _parse_hhmmss_to_seconds,
)
from nextext.core.docint_transcript import (
    transcript_segments_from_df as _transcript_segments_from_df,
)
from nextext.frontend.app import cli, main
from nextext.frontend.state import (
    default_target_language as _default_target_language,
)
from nextext.frontend.state import (
    download_file_name as _download_file_name,
)
from nextext.frontend.state import (
    progress_value as _progress_value,
)
from nextext.frontend.state import (
    result_file_names as _result_file_names,
)
from nextext.frontend.state import (
    select_result as _select_result,
)

__all__ = [
    "_build_docint_jsonl_archive",
    "_build_results_archive",
    "_default_target_language",
    "_download_file_name",
    "_parse_hhmmss_to_seconds",
    "_progress_value",
    "_result_file_names",
    "_select_result",
    "_transcript_segments_from_df",
    "cli",
    "main",
]


def _dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to XLSX bytes using openpyxl.

    Args:
        df: The DataFrame to serialize.

    Returns:
        bytes: XLSX bytes suitable for inclusion in a ZIP archive.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(cast(Any, buffer), engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buffer.getvalue()


def _build_results_archive(
    results: Sequence[dict[str, Any]],
    archive_stem: str,
) -> bytes:
    """Bundle every produced output for every processed file into a ZIP.

    Args:
        results: Stored result entries.
        archive_stem: Top-level directory name inside the ZIP.

    Returns:
        bytes: ZIP archive bytes ready to feed into ``st.download_button``.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, result in enumerate(results, start=1):
            original_name = str(result.get("file_name", f"result_{index}"))
            stem = Path(original_name).stem or f"result_{index}"
            base = f"{archive_stem}/{stem}/{stem}"

            transcript = result.get("transcript")
            if isinstance(transcript, pd.DataFrame) and not transcript.empty:
                zf.writestr(
                    f"{base}_transcript.csv",
                    transcript.to_csv(index=False).encode("utf-8"),
                )
                zf.writestr(
                    f"{base}_transcript.xlsx",
                    _dataframe_to_excel_bytes(transcript),
                )

            summary = result.get("summary")
            if isinstance(summary, str) and summary.strip():
                zf.writestr(f"{base}_summary.txt", summary.encode("utf-8"))

            word_counts = result.get("word_counts")
            if isinstance(word_counts, pd.DataFrame) and not word_counts.empty:
                zf.writestr(
                    f"{base}_words.csv",
                    word_counts.to_csv(index=False).encode("utf-8"),
                )
                zf.writestr(
                    f"{base}_words.xlsx",
                    _dataframe_to_excel_bytes(word_counts),
                )

            named_entities = result.get("named_entities")
            if isinstance(named_entities, pd.DataFrame) and not named_entities.empty:
                zf.writestr(
                    f"{base}_entities.csv",
                    named_entities.to_csv(index=False).encode("utf-8"),
                )
                zf.writestr(
                    f"{base}_entities.xlsx",
                    _dataframe_to_excel_bytes(named_entities),
                )

            wordcloud = result.get("wordcloud")
            if isinstance(wordcloud, Figure):
                png_buffer = io.BytesIO()
                wordcloud.savefig(png_buffer, format="png", bbox_inches="tight")
                zf.writestr(f"{base}_wordcloud.png", png_buffer.getvalue())

            findings = result.get("hate_speech_findings")
            if findings:
                findings_df = pd.DataFrame(findings)
                zf.writestr(
                    f"{base}_hate_speech.csv",
                    findings_df.to_csv(index=False).encode("utf-8"),
                )
                zf.writestr(
                    f"{base}_hate_speech.xlsx",
                    _dataframe_to_excel_bytes(findings_df),
                )

    return buffer.getvalue()


def _docint_language_for_result(result: dict[str, Any]) -> str | None:
    """Resolve the transcript language code for a single result entry.

    Args:
        result: Stored result payload.

    Returns:
        str | None: Normalized ISO 639-1 language code, or ``None``.
    """
    # Lazy import: the legacy shim is also imported by the frontend Docker
    # image (transitively via the tests' import surface) where the heavy
    # ``nextext.pipeline`` dependency tree is not installed. Keeping this
    # import inside the function lets the module load in lightweight contexts
    # too.
    from nextext.pipeline import normalize_language_code

    lang = result.get("transcript_language") or result.get("resolved_src_lang")
    if not lang:
        return None
    return normalize_language_code(str(lang))


def _docint_task_for_result(result: dict[str, Any]) -> str:
    """Resolve the task label for a single result entry.

    Args:
        result: Stored result payload.

    Returns:
        str: ``"transcribe"`` or ``"translate"``.
    """
    task = result.get("task") or "transcribe"
    return "translate" if str(task).lower() == "translate" else "transcribe"


def _docint_result_jsonl(result: dict[str, Any]) -> bytes:
    """Build the JSONL payload for a single processed result entry.

    Args:
        result: Stored result payload for one file.

    Returns:
        bytes: UTF-8 JSONL bytes; ``b""`` when the transcript has no rows.
    """
    transcript = result.get("transcript")
    if not isinstance(transcript, pd.DataFrame):
        return b""
    segments = _transcript_segments_from_df(transcript)
    if not segments:
        return b""
    return build_docint_jsonl(
        source_file=str(result.get("file_name", "")),
        source_file_hash=result.get("source_file_hash"),
        language=_docint_language_for_result(result),
        task=_docint_task_for_result(result),
        segments=segments,
    )


def _build_docint_jsonl_archive(
    results: Sequence[dict[str, Any]],
    archive_stem: str,
) -> tuple[bytes, str, str]:
    """Bundle docint JSONL output for every processed file.

    When the session contains exactly one processable result, the payload
    is a single ``.jsonl`` file; otherwise the per-file payloads are
    zipped under ``{archive_stem}/{stem}.jsonl``.

    Args:
        results: Stored result entries.
        archive_stem: Top-level directory name inside the ZIP.

    Returns:
        tuple[bytes, str, str]: ``(data, file_name, mime)``.
    """
    per_file: list[tuple[str, bytes]] = []
    for index, result in enumerate(results, start=1):
        payload = _docint_result_jsonl(result)
        if not payload:
            continue
        original_name = str(result.get("file_name", f"result_{index}"))
        stem = Path(original_name).stem or f"result_{index}"
        per_file.append((stem, payload))

    if not per_file:
        return b"", f"{archive_stem}.jsonl", "application/x-ndjson"

    if len(per_file) == 1:
        stem, payload = per_file[0]
        return payload, f"{stem}.docint.jsonl", "application/x-ndjson"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for stem, payload in per_file:
            zf.writestr(f"{archive_stem}/{stem}.jsonl", payload)
    return buffer.getvalue(), f"{archive_stem}_docint.zip", "application/zip"


if __name__ == "__main__":  # pragma: no cover
    main()
