"""Dev-only hate-speech eval: score transcript classifiers against gold-labelled fixtures.

Not shipped — ``eval/`` is excluded from the Docker image. It runs against the
live chat model configured by the usual environment variables
(``OPENAI_API_BASE``, ``OPENAI_API_KEY``, ``TEXT_MODEL``, ``INFERENCE_PROVIDER``,
``OLLAMA_THINK``, ``RESPONSE_LANGUAGE``), so a run costs real inference.

Two strategies are compared on the same rows:

- ``baseline`` — a frozen copy of the classifier Nextext shipped before the
  windowed rewrite: one isolated request per row with the old GMF prompt
  (``prompts/baseline/``, byte-identical to docint's), the old system line, the
  old ``bool()`` verdict parse, and the translation judged instead of the
  original when one exists.
- ``windowed`` — the production :func:`nextext.pipeline.hate_speech_pipeline`.

Usage::

    uv run python eval/hate_speech/run.py
    uv run python eval/hate_speech/run.py --strategy windowed --window-tokens 1 --context-tokens 0
    uv run python eval/hate_speech/run.py --mhc eval/hate_speech/data/mhc_german.csv --by-tag
    uv run python eval/hate_speech/run.py --emit-docint-jsonl eval/hate_speech/reports/docint
"""

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from score import GOLD_VALUES, RowResult, markdown_table, score_by_tag, score_rows

from nextext.core.docint_transcript import build_docint_jsonl, transcript_segments_from_df
from nextext.core.openai_cfg import InferencePipeline
from nextext.pipeline import hate_speech_pipeline
from nextext.utils.env_cfg import load_hate_speech_env, load_language_env

_HERE = Path(__file__).resolve().parent
_FIXTURES_DIR = _HERE / "fixtures"
_REPORTS_DIR = _HERE / "reports"
_BASELINE_PROMPTS_DIR = _HERE / "prompts" / "baseline"
_BASELINE_SYSTEM_PROMPT = "You are a content moderation assistant. Respond only with valid JSON."
_BASELINE_MAX_CHARS = 2048
_ROW_SECONDS = 5
_MHC_COUNTER_FUNCTIONALITIES = frozenset({"counter_quote_nh", "counter_ref_nh"})

Classifier = Callable[[pd.DataFrame, str], set[int]]
"""Maps a transcript frame and its language code to the flagged row positions."""


def load_fixtures(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load fixture transcripts (one JSON object per line) and validate their gold labels.

    Args:
        paths (Sequence[Path]): Fixture JSONL files.

    Returns:
        list[dict[str, Any]]: Transcripts with ``id``, ``lang``, ``tags`` and
            ``rows`` (each row: ``text``, ``gold``, optional ``speaker``,
            ``translation``, ``needs_context``).

    Raises:
        ValueError: If a row carries a gold label outside :data:`GOLD_VALUES`.
    """
    transcripts: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            transcript = json.loads(line)
            for row in transcript["rows"]:
                if row["gold"] not in GOLD_VALUES:
                    raise ValueError(
                        f"{path.name}:{line_number}: unknown gold label {row['gold']!r} in {transcript['id']}"
                    )
            transcripts.append(transcript)
    return transcripts


def load_mhc(path: Path, *, lang: str = "de", limit: int | None = None) -> list[dict[str, Any]]:
    """Convert a locally downloaded Multilingual HateCheck CSV into one-row transcripts.

    Hateful cases become ``endorses``; the counter-speech functionalities
    (``counter_quote_nh``, ``counter_ref_nh``) become ``condemns_or_counters``;
    everything else is ``none``. Each case is tagged with its functionality.

    Args:
        path (Path): The MHC CSV (``functionality``, ``test_case``, ``label_gold``).
        lang (str): Language code of the suite. Defaults to ``"de"``.
        limit (int | None): Keep at most this many cases.

    Returns:
        list[dict[str, Any]]: The converted transcripts.
    """
    transcripts: list[dict[str, Any]] = []
    for position, record in enumerate(pd.read_csv(path).to_dict("records")):
        if limit is not None and position >= limit:
            break
        functionality = str(record["functionality"])
        if str(record["label_gold"]).strip().lower() == "hateful":
            gold = "endorses"
        elif functionality in _MHC_COUNTER_FUNCTIONALITIES:
            gold = "condemns_or_counters"
        else:
            gold = "none"
        transcripts.append(
            {
                "id": f"mhc-{position}",
                "lang": lang,
                "tags": [functionality],
                "rows": [{"text": str(record["test_case"]), "gold": gold}],
            }
        )
    return transcripts


def transcript_frame(transcript: dict[str, Any]) -> pd.DataFrame:
    """Build the transcript DataFrame the pipeline expects from a fixture transcript.

    Rows get unique ``H:MM:SS`` start stamps at 5-second steps; ``speaker`` and
    ``translation`` columns are added only when a row carries them.

    Args:
        transcript (dict[str, Any]): A fixture transcript.

    Returns:
        pd.DataFrame: Columns ``start``, ``end``, optional ``speaker``,
            ``text``, optional ``translation``.
    """
    rows = transcript["rows"]
    data: dict[str, list[str]] = {
        "start": [str(timedelta(seconds=i * _ROW_SECONDS)) for i in range(len(rows))],
        "end": [str(timedelta(seconds=i * _ROW_SECONDS + _ROW_SECONDS - 1)) for i in range(len(rows))],
    }
    if any(row.get("speaker") for row in rows):
        data["speaker"] = [row.get("speaker", "") for row in rows]
    data["text"] = [row["text"] for row in rows]
    if any(row.get("translation") for row in rows):
        data["translation"] = [row.get("translation", "") for row in rows]
    return pd.DataFrame(data)


def flagged_rows(df: pd.DataFrame, findings: Sequence[dict[str, Any]]) -> set[int]:
    """Map pipeline findings back to row positions via their unique start stamps.

    Args:
        df (pd.DataFrame): The frame the findings came from.
        findings (Sequence[dict[str, Any]]): ``hate_speech_pipeline`` findings.

    Returns:
        set[int]: Flagged row positions.
    """
    position_by_start = {start: position for position, start in enumerate(df["start"].tolist())}
    return {position_by_start[f["start"]] for f in findings if f.get("start") in position_by_start}


def baseline_flag(raw: str) -> bool:
    """Reproduce the removed per-row parser's verdict, quirks included.

    Mirrors the pre-rewrite ``_parse_hate_speech_payload``: whole-reply JSON,
    else the greedy ``{...}`` span, then ``bool(data["hate_speech"])`` — so the
    string ``"false"`` still counts as a positive. A non-object reply (which
    crashed the old job) counts as a negative here.

    Args:
        raw (str): The model reply.

    Returns:
        bool: The baseline verdict.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return False
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return False
    return isinstance(data, dict) and bool(data.get("hate_speech", False))


def baseline_classifier(pipeline: InferencePipeline) -> Classifier:
    """Build the frozen pre-rewrite per-row classifier.

    Args:
        pipeline (InferencePipeline): The live inference client.

    Returns:
        Classifier: One isolated request per row, legacy prompt and parse.
    """
    code = load_language_env().code
    prompt_path = _BASELINE_PROMPTS_DIR / code / "hate_speech.txt"
    if not prompt_path.exists():
        prompt_path = _BASELINE_PROMPTS_DIR / "en" / "hate_speech.txt"
    template = prompt_path.read_text(encoding="utf-8")

    def classify(df: pd.DataFrame, language: str) -> set[int]:
        """Classify every row in isolation, judging the translation when present.

        Args:
            df (pd.DataFrame): The transcript frame.
            language (str): Unused; the baseline never saw the language.

        Returns:
            set[int]: Flagged row positions.
        """
        del language
        column = "translation" if "translation" in df.columns else "text"
        flagged: set[int] = set()
        for position, text in enumerate(df[column].astype(str).tolist()):
            raw = pipeline.call_model(
                prompt=template.replace("{text}", text[:_BASELINE_MAX_CHARS]),
                system_prompt=_BASELINE_SYSTEM_PROMPT,
            )
            if baseline_flag(raw):
                flagged.add(position)
        return flagged

    return classify


def windowed_classifier(pipeline: InferencePipeline) -> Classifier:
    """Build the production windowed classifier.

    Args:
        pipeline (InferencePipeline): The live inference client.

    Returns:
        Classifier: :func:`hate_speech_pipeline` mapped back to row positions.
    """

    def classify(df: pd.DataFrame, language: str) -> set[int]:
        """Run the production pipeline on one transcript.

        Args:
            df (pd.DataFrame): The transcript frame.
            language (str): The transcript's language code.

        Returns:
            set[int]: Flagged row positions.
        """
        return flagged_rows(df, hate_speech_pipeline(df, pipeline, src_lang=language))

    return classify


def evaluate(transcripts: Sequence[dict[str, Any]], classify: Classifier) -> list[RowResult]:
    """Classify every transcript and pair each row's verdict with its gold label.

    Args:
        transcripts (Sequence[dict[str, Any]]): Fixture transcripts.
        classify (Classifier): The strategy under test.

    Returns:
        list[RowResult]: One scored row per transcript row.
    """
    results: list[RowResult] = []
    for transcript in transcripts:
        flagged = classify(transcript_frame(transcript), transcript["lang"])
        for position, row in enumerate(transcript["rows"]):
            results.append(
                RowResult(
                    transcript_id=transcript["id"],
                    row=position,
                    gold=row["gold"],
                    predicted=position in flagged,
                    tags=tuple(transcript["tags"]),
                    needs_context=bool(row.get("needs_context", False)),
                )
            )
    return results


def select_transcripts(
    *,
    fixtures: Sequence[Path] | None,
    mhc: Path | None,
    mhc_lang: str,
    tags: Sequence[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Assemble the transcripts to score.

    ``--mhc`` alone scores only the benchmark; the committed fixtures run when
    no benchmark is given, or alongside it when ``--fixtures`` names them.

    Args:
        fixtures (Sequence[Path] | None): Fixture files, or ``None`` for the default.
        mhc (Path | None): A Multilingual HateCheck CSV, or ``None``.
        mhc_lang (str): Language code of the MHC suite.
        tags (Sequence[str] | None): Keep only transcripts carrying one of these tags.
        limit (int | None): Keep at most this many transcripts.

    Returns:
        list[dict[str, Any]]: The transcripts.
    """
    transcripts: list[dict[str, Any]] = []
    if fixtures:
        transcripts += load_fixtures(fixtures)
    elif mhc is None:
        transcripts += load_fixtures(sorted(_FIXTURES_DIR.glob("*.jsonl")))
    if mhc is not None:
        transcripts += load_mhc(mhc, lang=mhc_lang)
    if tags:
        transcripts = [t for t in transcripts if set(tags) & set(t["tags"])]
    return transcripts[:limit] if limit is not None else transcripts


def run_settings() -> dict[str, Any]:
    """Describe the settings a run uses, for its report.

    Returns:
        dict[str, Any]: Effective window budgets (defaults included), prompt
            locale, provider and the Ollama think setting.
    """
    budgets = load_hate_speech_env()
    return {
        "window_tokens": budgets.window_tokens,
        "context_tokens": budgets.context_tokens,
        "prompt_lang": load_language_env().code,
        "provider": os.getenv("INFERENCE_PROVIDER", "ollama"),
        "ollama_think": os.getenv("OLLAMA_THINK"),
    }


def emit_docint_jsonl(transcripts: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    """Write each transcript as the ``docint.jsonl`` payload Nextext would export.

    For smoke-testing docint's transcript path on the same synthetic rows.

    Args:
        transcripts (Sequence[dict[str, Any]]): Fixture transcripts.
        out_dir (Path): Directory for the ``<id>.docint.jsonl`` files.

    Returns:
        list[Path]: The written files, in transcript order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for transcript in transcripts:
        payload = build_docint_jsonl(
            source_file=f"{transcript['id']}.wav",
            source_file_hash=None,
            language=transcript["lang"],
            segments=transcript_segments_from_df(transcript_frame(transcript)),
        )
        path = out_dir / f"{transcript['id']}.docint.jsonl"
        path.write_bytes(payload)
        written.append(path)
    return written


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv (Sequence[str] | None): Arguments, or ``None`` for ``sys.argv``.

    Returns:
        argparse.Namespace: The parsed options.
    """
    parser = argparse.ArgumentParser(description="Score hate-speech classifiers against gold-labelled transcripts.")
    parser.add_argument("--fixtures", nargs="+", type=Path, help="Fixture JSONL files (default: the committed ones).")
    parser.add_argument(
        "--mhc",
        type=Path,
        help="Local Multilingual HateCheck CSV (never committed); scored alone unless --fixtures is given.",
    )
    parser.add_argument("--mhc-lang", default="de", help="Language code of the MHC suite (default: de).")
    parser.add_argument(
        "--strategy",
        nargs="+",
        choices=("baseline", "windowed"),
        default=["baseline", "windowed"],
        help="Strategies to compare (default: both).",
    )
    parser.add_argument("--window-tokens", type=int, help="Override HATE_SPEECH_WINDOW_TOKENS.")
    parser.add_argument("--context-tokens", type=int, help="Override HATE_SPEECH_CONTEXT_TOKENS (0 = no context).")
    parser.add_argument("--prompt-lang", choices=("en", "de"), help="Override RESPONSE_LANGUAGE (prompt locale).")
    parser.add_argument("--tag", action="append", help="Only transcripts carrying this tag (repeatable).")
    parser.add_argument("--limit", type=int, help="Evaluate at most this many transcripts.")
    parser.add_argument("--by-tag", action="store_true", help="Also print one table per tag.")
    parser.add_argument("--out", type=Path, default=_REPORTS_DIR, help="Report directory.")
    parser.add_argument(
        "--emit-docint-jsonl",
        type=Path,
        metavar="DIR",
        help="Write docint-shaped JSONL per transcript and exit without model calls.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the eval and print one metrics table (plus per-tag tables on request).

    Args:
        argv (Sequence[str] | None): Arguments, or ``None`` for ``sys.argv``.

    Returns:
        int: Process exit code.
    """
    args = _parse_args(argv)
    if args.prompt_lang:
        os.environ["RESPONSE_LANGUAGE"] = args.prompt_lang
    if args.window_tokens is not None:
        os.environ["HATE_SPEECH_WINDOW_TOKENS"] = str(args.window_tokens)
    if args.context_tokens is not None:
        os.environ["HATE_SPEECH_CONTEXT_TOKENS"] = str(args.context_tokens)

    transcripts = select_transcripts(
        fixtures=args.fixtures, mhc=args.mhc, mhc_lang=args.mhc_lang, tags=args.tag, limit=args.limit
    )

    if args.emit_docint_jsonl:
        for path in emit_docint_jsonl(transcripts, args.emit_docint_jsonl):
            print(path)
        return 0

    pipeline = InferencePipeline()
    if not pipeline.get_health():
        print("The configured inference provider is not reachable.", file=sys.stderr)
        return 1

    builders: dict[str, Callable[[InferencePipeline], Classifier]] = {
        "baseline": baseline_classifier,
        "windowed": windowed_classifier,
    }
    report: dict[str, Any] = {
        "created": datetime.now(UTC).isoformat(),
        "model": pipeline.default_model,
        **run_settings(),
        "transcripts": len(transcripts),
        "strategies": {},
    }
    rows_by_strategy: dict[str, list[RowResult]] = {}
    for name in args.strategy:
        started = time.monotonic()
        rows = evaluate(transcripts, builders[name](pipeline))
        elapsed = time.monotonic() - started
        rows_by_strategy[name] = rows
        report["strategies"][name] = {
            "seconds": round(elapsed, 1),
            "metrics": asdict(score_rows(rows)),
            "rows": [asdict(row) for row in rows],
        }

    print(markdown_table({name: score_rows(rows) for name, rows in rows_by_strategy.items()}))
    if args.by_tag:
        for name, rows in rows_by_strategy.items():
            print(f"\n### {name} by tag\n")
            print(markdown_table(score_by_tag(rows)))

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.out / f"hs-{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
