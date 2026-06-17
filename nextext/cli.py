"""CLI entry point: single-file in-process pipeline runner (no backend required)."""

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
from loguru import logger

from nextext.core.docint_transcript import (
    build_docint_jsonl,
)
from nextext.core.docint_transcript import (
    language_name as _language_name,
)
from nextext.core.docint_transcript import (
    transcript_segments_from_df as _transcript_segments_from_df,
)
from nextext.core.openai_cfg import InferencePipeline
from nextext.core.processing import FileProcessor
from nextext.pipeline import (
    hate_speech_pipeline,
    normalize_language_code,
    should_translate,
    summarization_pipeline,
    transcription_pipeline,
    translation_pipeline,
    wordlevel_pipeline,
)
from nextext.utils.log_cfg import setup_logging

setup_logging()


def parse_arguments(args_list: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Nextext CLI.

    Sets up the argument parser with options for file processing, language
    settings, model selection, and analysis features. Supports a full-analysis
    shortcut that enables word statistics and summarization at once.

    Args:
        args_list (list | None): Drop-in list of arguments to parse. If
            ``None``, uses ``sys.argv`` to parse command-line arguments.
            Defaults to ``None``.

    Returns:
        argparse.Namespace: Parsed command-line arguments as a namespace
            object.

    Raises:
        argparse.ArgumentError: If there is an error in argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Audio transcription and analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-f",
        "--file",
        dest="file_path",
        type=Path,
        required=True,
        help="Specify the file path and name of the audio file to be transcribed.",
    )
    parser.add_argument(
        "-sl",
        "--src-lang",
        dest="src_lang",
        type=str,
        default=None,
        help="Specify the language code (ISO 639-1) of the source audio (default: None).",
    )
    parser.add_argument(
        "-tl",
        "--trg-lang",
        dest="trg_lang",
        type=str,
        default="de",
        help="Specify the language code (ISO 639-1) of the target language (default: 'de').",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        type=str,
        default="transcribe",
        help="Specify the task to perform: 'transcribe' (default), or 'translate'.",
    )
    parser.add_argument(
        "-s",
        "--speakers",
        dest="speakers",
        type=int,
        default=1,
        help="Specify the maximum number of speakers for diarization (default: 1).",
    )
    parser.add_argument(
        "-w",
        "--words",
        dest="words",
        action="store_true",
        help="Show most frequently used words (default: False).",
    )
    parser.add_argument(
        "-sum",
        "--summarize",
        dest="summarize",
        action="store_true",
        help="Additional transcript summarization (default: False).",
    )
    parser.add_argument(
        "-hs",
        "--hate-speech",
        dest="hate_speech",
        action="store_true",
        help="Detect hate speech in transcript segments via LLM (default: False).",
    )
    parser.add_argument(
        "-F",
        "--full-analysis",
        dest="full_analysis",
        action="store_true",
        help="Enable full analysis, equivalent to using -w -sum (default: False).",
    )
    parser.add_argument(
        "-ed",
        "--emit-docint-jsonl",
        dest="emit_docint_jsonl",
        type=Path,
        default=None,
        help=(
            "Write a docint-compatible JSONL transcript to this path. If the "
            "path points to an existing directory, the file is saved as "
            "'<source_stem>.jsonl' inside it."
        ),
    )
    parser.add_argument(
        "-fd",
        "--force-docint-jsonl",
        dest="force_docint_jsonl",
        action="store_true",
        help=("Overwrite the --emit-docint-jsonl target when it already exists (default: False)."),
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    if args.full_analysis:
        args.words = True
        args.summarize = True
        args.hate_speech = True

    return args


def _resolve_docint_output_path(
    output_path: Path,
    source_path: Path,
) -> Path:
    """Resolve the JSONL output path for the ``--emit-docint-jsonl`` flag.

    When ``output_path`` is an existing directory, the file is written as
    ``<source_stem>.jsonl`` inside it; otherwise ``output_path`` is taken
    verbatim. The returned path is resolved to an absolute form so callers
    can perform symlink / overwrite checks without surprises from
    relative-path lookups.

    Args:
        output_path (Path): The CLI-provided target path.
        source_path (Path): The input audio file path.

    Returns:
        Path: Absolute path to the JSONL file to write.

    Raises:
        ValueError: If either the target itself or its parent directory is
            a symlink; we refuse to follow symlinks at the write boundary.
    """
    if output_path.is_dir():
        target = output_path / f"{source_path.stem}.jsonl"
    else:
        target = output_path
    target = target.resolve(strict=False)
    if target.is_symlink():
        raise ValueError(f"Refusing to write docint JSONL via symlink target '{target}'.")
    if target.parent.is_symlink():
        raise ValueError(f"Refusing to write docint JSONL through a symlinked parent directory '{target.parent}'.")
    return target


def _emit_docint_jsonl(
    transcript_df: "pd.DataFrame",
    source_path: Path,
    output_path: Path,
    task: str,
    language: str | None,
    detected_language: str | None,
    force_overwrite: bool = False,
) -> None:
    """Write a docint JSONL payload for a completed transcription run.

    The write is atomic: the payload is materialized to ``<target>.tmp``
    and then :func:`os.replace` swaps it into place, so partial writes are
    never observable.

    Args:
        transcript_df (pd.DataFrame): Sentence-merged transcript.
        source_path (Path): Original input audio path.
        output_path (Path): Target JSONL path or parent directory.
        task (str): Task to perform, ``"transcribe"`` or ``"translate"``.
        language (str | None): Normalized ISO 639-1 code of the transcript
            text (the target for ``translate``; the source for
            ``transcribe``).
        detected_language (str | None): Normalized ISO 639-1 code of the
            auto-detected source audio language.
        force_overwrite (bool): When ``True``, overwrite an existing
            target file. When ``False`` (default), refuse and raise
            :class:`FileExistsError`.

    Raises:
        FileExistsError: If the resolved target already exists and
            ``force_overwrite`` is ``False``.
        ValueError: If the resolved target or its parent is a symlink.
    """
    segments = _transcript_segments_from_df(transcript_df)
    if not segments:
        logger.warning("No transcript segments available; skipping docint JSONL export.")
        return
    try:
        file_hash = f"sha256:{hashlib.sha256(source_path.read_bytes()).hexdigest()}"
    except OSError as exc:
        logger.warning("Could not hash source file ({}); omitting hash.", exc)
        file_hash = None
    payload = build_docint_jsonl(
        source_file=source_path.name,
        source_file_hash=file_hash,
        language=language,
        detected_language=detected_language,
        task=task,
        segments=segments,
    )
    target = _resolve_docint_output_path(output_path, source_path)
    if target.exists() and not force_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing docint JSONL at '{target}'. "
            "Remove the file or pass --force-docint-jsonl to overwrite."
        )
    # Record which parent directories we had to create so the user can
    # audit any new filesystem layout introduced by the export.
    missing_parents: list[Path] = []
    if not target.parent.exists():
        for ancestor in reversed(target.parent.parents):
            if not ancestor.exists():
                missing_parents.append(ancestor)
        missing_parents.append(target.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        for created in missing_parents:
            logger.info("Created directory '{}' for docint JSONL output.", created)
    # Atomic write: render to a tmp sibling then replace in one syscall.
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, target)
    logger.info("Wrote docint JSONL to '{}'.", target)


def main() -> None:
    """Run the Nextext pipeline for transcription, translation, and analysis.

    Orchestrates the entire process, including transcription, translation,
    word statistics, and summarization. Handles the command-line arguments
    and manages the flow of data through the various modules.

    Raises:
        ValueError: If an invalid task is specified or if the source language
            cannot be resolved for analysis.
        ConnectionError: If the configured inference provider is not
            reachable for analysis tasks.
    """
    logger.info("\n\nInitiating Nextext...\n")

    # Parse command-line arguments
    args = parse_arguments()

    _run_main(args)


def _run_main(args: argparse.Namespace) -> None:
    """Execute the main Nextext pipeline on the parsed CLI arguments.

    Args:
        args (argparse.Namespace): Parsed CLI arguments from
            :func:`parse_arguments`.

    Raises:
        ValueError: If ``args.task`` is not ``"transcribe"`` or
            ``"translate"``, or if the source language cannot be resolved.
        ConnectionError: If the configured inference provider is not reachable.
    """
    # Set up input/output directories and file paths
    file_processor = FileProcessor(args.file_path)

    # Transcribe and diarize the audio file
    if args.task in ["transcribe", "translate"]:
        transcript_df, updated_src_lang = transcription_pipeline(
            file_path=args.file_path,
            src_lang=args.src_lang,
            n_speakers=args.speakers,
        )
        args.src_lang = updated_src_lang  # Update source language if detected

        # Guard: stop early when the transcript contains no speech
        transcript_text = " ".join(transcript_df["text"].astype(str).tolist()).strip()
        if transcript_df.empty or not transcript_text:
            logger.warning(
                "No speech detected in '{}'. Writing empty transcript and skipping analysis.",
                args.file_path,
            )
            file_processor.write_file_output(transcript_df, "transcript")
            return
    else:
        logger.error("Invalid task specified: {}", args.task)
        raise ValueError("Invalid task. Please specify 'transcribe' or 'translate'.")

    # Machine translate the transcribed text. Whisper only transcribes; the LLM
    # performs every translation, directly from source to target. Engage it
    # whenever a translate task was requested and the resolved source differs
    # from the target (so an English target is translated too).
    inference_pipeline = None
    if should_translate(args.task, args.src_lang, args.trg_lang):
        inference_pipeline = InferencePipeline(out_language=_language_name(args.trg_lang))
        if not inference_pipeline.get_health():
            logger.error("The configured inference provider is not reachable.")
            raise ConnectionError(
                "The configured inference provider is not reachable. Please ensure it is running and accessible."
            )
        transcript_df = translation_pipeline(
            df=transcript_df,
            trg_lang=args.trg_lang,
            src_lang=args.src_lang,
            inference_pipeline=inference_pipeline,
        )

    # Streamline language for further processing
    if args.task == "transcribe":
        if args.src_lang is None:
            logger.error("Unable to resolve source language for downstream analysis.")
            raise ValueError("Source language could not be resolved for downstream analysis.")
        normalized_lang = normalize_language_code(args.src_lang)
        if normalized_lang is None:
            logger.error("Unable to normalize source language for downstream analysis.")
            raise ValueError("Source language could not be normalized for downstream analysis.")
    else:
        normalized_lang = normalize_language_code(args.trg_lang)
        if normalized_lang is None:
            logger.error("Unable to normalize target language for downstream analysis.")
            raise ValueError("Target language could not be normalized for downstream analysis.")
    transcript_lang: str = normalized_lang
    # The auto-detected (or pinned) source language, normalized. Equals
    # ``transcript_lang`` for ``transcribe``; differs for ``translate`` where
    # ``transcript_lang`` is the target.
    detected_src_lang = normalize_language_code(args.src_lang) if args.src_lang else None

    # Calculate word statistics
    if args.words:
        word_counts, named_entities, wordcloud_fig = wordlevel_pipeline(
            data=transcript_df,
            language=transcript_lang,
        )
        exports = [
            (word_counts, "words"),
            (named_entities, "entities"),
            (wordcloud_fig, "wordcloud"),
        ]
        for export, name in exports:
            if export is not None:
                file_processor.write_file_output(export, name)

    if args.summarize:
        if inference_pipeline is None:
            inference_pipeline = InferencePipeline(out_language=_language_name(transcript_lang))
        if not inference_pipeline.get_health():
            logger.error("The configured inference provider is not reachable.")
            raise ConnectionError(
                "The configured inference provider is not reachable. Please ensure it is running and accessible."
            )

        # Summarize the transcribed text
        if args.summarize:
            transcript_summary = summarization_pipeline(
                text=" ".join(transcript_df["text"].astype(str).tolist()),
                inference_pipeline=inference_pipeline,
            )
            if transcript_summary is not None:
                file_processor.write_file_output(transcript_summary, "summary")

    # Hate speech detection
    if args.hate_speech:
        if inference_pipeline is None:
            inference_pipeline = InferencePipeline(out_language=_language_name(transcript_lang))
        if not inference_pipeline.get_health():
            logger.error("The configured inference provider is not reachable.")
            raise ConnectionError(
                "The configured inference provider is not reachable. Please ensure it is running and accessible."
            )
        hate_speech_findings = hate_speech_pipeline(
            df=transcript_df,
            inference_pipeline=inference_pipeline,
        )
        if hate_speech_findings:
            file_processor.write_file_output(pd.DataFrame(hate_speech_findings), "hate_speech")
            logger.info("Hate speech detected in {} segment(s).", len(hate_speech_findings))
        else:
            logger.info("No hate speech detected.")

    # Save final transcript
    file_processor.write_file_output(transcript_df, "transcript")

    # Optional: emit a docint-compatible JSONL payload next to the transcript.
    # ``_transcript_segments_from_df`` returns integer-valued floats for
    # ``start_seconds`` / ``end_seconds`` — they parse the already-rounded
    # ``HH:MM:SS`` strings produced by ``_seconds_to_time``.
    if getattr(args, "emit_docint_jsonl", None) is not None:
        _emit_docint_jsonl(
            transcript_df=transcript_df,
            source_path=args.file_path,
            output_path=args.emit_docint_jsonl,
            task=args.task,
            language=transcript_lang,
            detected_language=detected_src_lang,
            force_overwrite=getattr(args, "force_docint_jsonl", False),
        )

    logger.info("The end of our elaborate plans, the end of everything that stands.")


if __name__ == "__main__":
    main()
