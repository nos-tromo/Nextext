"""Tests for the CLI docint JSONL emitter (``nextext.cli._emit_docint_jsonl``)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nextext.cli import _emit_docint_jsonl


def _transcript_df() -> pd.DataFrame:
    """Build a tiny two-segment transcript DataFrame.

    Returns:
        pd.DataFrame: Transcript with ``start``/``end``/``text`` columns.
    """
    return pd.DataFrame(
        {
            "start": ["00:00:00", "00:00:02"],
            "end": ["00:00:02", "00:00:04"],
            "text": ["Hallo.", "Welt."],
        }
    )


def test_emit_docint_jsonl_writes_source_language(tmp_path: Path) -> None:
    """The emitted records carry the resolved source language."""
    source = tmp_path / "clip.wav"
    source.write_bytes(b"x")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _emit_docint_jsonl(
        transcript_df=_transcript_df(),
        source_path=source,
        output_path=out_dir,
        language="de",
    )

    target = out_dir / "clip.jsonl"
    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]
    assert records
    for record in records:
        assert record["language"] == "de"
