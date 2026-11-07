"""CLI entry point: single-file in-process pipeline runner (no backend required).

Every run that processes a file writes its own data-protection audit chain
(:mod:`nextext.audit`), attributed to ``AUDIT_CLI_USER`` via ``cli``: the
file's intake (``collect``), each model call (``ai_call``), the joined result
(``combine``) and every output file it writes (``disclose`` to
``file:<output>``). Inside the backend container (``make cli``) the chain goes
to the container's log; elsewhere to stdout, which the CLI uses for nothing
else. A run whose actions need a stated purpose (``AUDIT_REQUIRE_PURPOSE``)
takes it from ``AUDIT_CASE_REF`` / ``AUDIT_REASON`` and exits ``2`` without
one; a failed audit write stops the run with a non-zero exit.
"""

import argparse
import hashlib
import os
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from matplotlib.figure import Figure

from nextext.api import cli_files
from nextext.audit import (
    CAPTION_STAGE,
    FRAME_HATE_SPEECH_STAGE,
    HATE_SPEECH_STAGE,
    MEDIA,
    SUMMARY_STAGE,
    TRANSCRIPTION_STAGE,
    TRANSLATION_STAGE,
    VISUAL_SUMMARY_STAGE,
    WORDLEVEL_STAGE,
    AuditSink,
    JobAudit,
    Origin,
    build_emitter,
    emit_config_event,
    joined_analyses,
)
from nextext.core.docint_transcript import (
    build_docint_jsonl,
)
from nextext.core.docint_transcript import (
    transcript_segments_from_df as _transcript_segments_from_df,
)
from nextext.core.keyframes import extract_keyframe_samples
from nextext.core.openai_cfg import InferencePipeline
from nextext.core.outcomes import skip_reason_text
from nextext.core.processing import FileProcessor
from nextext.core.visual_context import FrameCaption, describe_keyframes, format_visual_context
from nextext.pipeline import (
    column_text,
    effective_text_column,
    frame_hate_speech_pipeline,
    hate_speech_pipeline,
    normalize_language_code,
    should_translate,
    summarization_pipeline,
    transcription_pipeline,
    translation_pipeline,
    wordlevel_pipeline,
)
from nextext.utils.audit_event import Actor, AuditEmitter, Purpose
from nextext.utils.audit_runtime import purpose_from_env
from nextext.utils.env_cfg import (
    load_audit_env,
    load_cli_audit_user,
    load_cli_retention_env,
    load_keyframe_defaults,
    load_language_env,
    load_visual_summary_env,
)
from nextext.utils.log_cfg import setup_logging

setup_logging()

#: What a CLI run does with the file: it reads it in, sends it to the models,
#: joins their results and writes output files. A purpose required for any of
#: these is required for the run.
CLI_ACTIONS: frozenset[str] = frozenset({"collect", "ai_call", "combine", "disclose"})
_HASH_CHUNK = 1 << 20


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
        description="Audio/video transcription, translation and analysis toolkit.",
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
        "--diarize",
        dest="diarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect and label speakers (default: on). Use --no-diarize to skip.",
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
        "-kf",
        "--keyframes",
        dest="keyframes",
        action="store_true",
        help=(
            "Sample video keyframes and describe them (default: False). Independent "
            "of --summarize: the descriptions are written on their own, and feed the "
            "summary when one is requested. Captioning obeys NEXTEXT_VISUAL_SUMMARY."
        ),
    )
    parser.add_argument(
        "-hs",
        "--hate-speech",
        dest="hate_speech",
        action="store_true",
        help=(
            "Detect hate speech the speakers endorse, judging each segment in context windows via LLM; "
            "with -kf, also judge the keyframe captions (default: False)."
        ),
    )
    parser.add_argument(
        "-F",
        "--full-analysis",
        dest="full_analysis",
        action="store_true",
        help="Enable full analysis, equivalent to using -w -sum -hs -kf (default: False).",
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
        args.keyframes = True

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
    language: str | None,
    force_overwrite: bool = False,
) -> Path | None:
    """Write a docint JSONL payload for a completed transcription run.

    The write is atomic: the payload is materialized to ``<target>.tmp``
    and then :func:`os.replace` swaps it into place, so partial writes are
    never observable.

    Args:
        transcript_df (pd.DataFrame): Sentence-merged transcript.
        source_path (Path): Original input audio path.
        output_path (Path): Target JSONL path or parent directory.
        language (str | None): Normalized ISO 639-1 code of the transcript
            text. Since the JSONL always carries the original transcript
            (never a Nextext translation), this is always the resolved
            source language.
        force_overwrite (bool): When ``True``, overwrite an existing
            target file. When ``False`` (default), refuse and raise
            :class:`FileExistsError`.

    Returns:
        Path | None: The file written, or ``None`` when there were no
            segments to export.

    Raises:
        FileExistsError: If the resolved target already exists and
            ``force_overwrite`` is ``False``.
        ValueError: If the resolved target or its parent is a symlink.
    """
    segments = _transcript_segments_from_df(transcript_df)
    if not segments:
        logger.warning("No transcript segments available; skipping docint JSONL export.")
        return None
    try:
        file_hash = f"sha256:{hashlib.sha256(source_path.read_bytes()).hexdigest()}"
    except OSError as exc:
        logger.warning("Could not hash source file ({}); omitting hash.", exc)
        file_hash = None
    payload = build_docint_jsonl(
        source_file=source_path.name,
        source_file_hash=file_hash,
        language=language,
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
    return target


def main() -> None:
    """Run the Nextext pipeline for transcription, translation, and analysis.

    Orchestrates the entire process, including transcription, translation,
    word statistics, and summarization. Handles the command-line arguments
    and manages the flow of data through the various modules.

    Exits with ``0`` on success and ``3`` when the file held no processable
    speech, so a batch caller can tell the two apart. ``2`` is left to
    argparse, which uses it for command-line usage errors.

    Raises:
        SystemExit: Always, carrying the exit code from :func:`_run_main`.
        ValueError: If an invalid task is specified or if the source language
            cannot be resolved for analysis.
        ConnectionError: If the configured inference provider is not
            reachable for analysis tasks.
    """
    logger.info("\n\nInitiating Nextext...\n")

    # Parse command-line arguments
    args = parse_arguments()

    raise SystemExit(run_cli(args))


def _refuse(message: str) -> int:
    """Print a one-line refusal to stderr and return the usage exit code.

    Args:
        message (str): The reason, naming no value.

    Returns:
        int: ``2``, the exit code argparse uses for usage errors.
    """
    print(f"nextext-cli: {message}", file=sys.stderr)
    return 2


def _file_sha256(path: Path) -> str:
    """Hash a file in chunks, so a multi-GB recording never sits in memory.

    Args:
        path (Path): The file to hash.

    Returns:
        str: The lowercase hex SHA-256 of its bytes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _cli_analyses(args: argparse.Namespace) -> list[str]:
    """Name the analyses a run asked for, for its ``collect`` event.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        list[str]: Transcription always, then each opt-in analysis requested.
    """
    requested = ["transcription"]
    for name, wanted in (
        ("diarization", args.diarize),
        ("translation", args.task == "translate"),
        ("word_level", args.words),
        ("summary", args.summarize),
        ("hate_speech", args.hate_speech),
        ("keyframes", args.keyframes),
    ):
        if wanted:
            requested.append(name)
    return requested


def _collect_criteria(args: argparse.Namespace) -> dict[str, object]:
    """Describe the input file for the run's ``collect`` event: size, hash, what was asked.

    Never the file's name or path, which can name the people in it.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        dict[str, object]: The event's criteria.

    Raises:
        OSError: If the file cannot be read.
    """
    path: Path = args.file_path
    return {
        "files": 1,
        "bytes": path.stat().st_size,
        "source_sha256": _file_sha256(path),
        "analyses": _cli_analyses(args),
        "options": {
            "src_lang": args.src_lang,
            "trg_lang": args.trg_lang,
            "task": args.task,
            "diarize": args.diarize,
            "words": args.words,
            "summarization": args.summarize,
            "hate_speech": args.hate_speech,
            "keyframes": args.keyframes,
            "docint_jsonl": getattr(args, "emit_docint_jsonl", None) is not None,
        },
    }


def run_cli(
    args: argparse.Namespace,
    *,
    emitter_factory: Callable[[str | None], AuditEmitter] = build_emitter,
) -> int:
    """Run the pipeline on one file inside its own audit chain.

    The chain starts before the file is read and closes when the run ends,
    whether it succeeds, finds no speech, or fails.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        emitter_factory (Callable[[str | None], AuditEmitter]): Creates the
            run's audit emitter from the ``AUDIT_OUTPUT`` mode.

    Returns:
        int: ``0`` for a normal run, ``3`` when the file held no processable
            speech, ``2`` when the audit settings or the stated purpose are
            malformed, or a purpose is required and missing.

    Raises:
        AuditWriteError: If an audit event cannot be written; the run stops.
        OSError: If the audit stream or the input file cannot be opened.
    """
    try:
        audit_cfg = load_audit_env()
        case_ref, reason = purpose_from_env(os.environ)
    except ValueError as exc:  # PurposeError is a ValueError; neither names a value.
        return _refuse(str(exc))
    purpose = Purpose(case_ref=case_ref, reason=reason) if case_ref or reason else None
    actor = Actor(load_cli_audit_user(), "cli")
    emitter = emitter_factory(audit_cfg.output)
    emitter.start()
    try:
        # A run has no identity seam, so only the purpose requirement states its mode.
        emit_config_event(emitter, {"require_purpose": sorted(audit_cfg.require_purpose)})
        if purpose is None and audit_cfg.require_purpose & CLI_ACTIONS:
            emitter.emit("collect", MEDIA, actor, outcome="denied", criteria={"files": 1})
            return _refuse("this run needs a stated purpose: set AUDIT_CASE_REF and/or AUDIT_REASON.")
        audit = JobAudit(AuditSink(emitter), uuid.uuid4().hex, Origin(actor=actor, purpose=purpose))
        audit.collect(_collect_criteria(args))
        return _run_main(args, audit)
    finally:
        emitter.close()


def _keyframe_step(
    args: argparse.Namespace,
    file_processor: FileProcessor,
    ensure_inference: Callable[[], InferencePipeline],
    audit: JobAudit,
) -> list[FrameCaption]:
    """Sample and describe video keyframes when ``--keyframes`` was given.

    The step is independent of ``--summarize``: the sampled JPEGs and the
    timestamped descriptions are written as outputs in their own right, and
    the returned captions feed the summary and the hate-speech pass only when
    those were also requested.
    Captioning within the step obeys ``NEXTEXT_VISUAL_SUMMARY`` and is
    fail-soft — a caption outage still leaves the frames on disk.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        file_processor (FileProcessor): Writer for this run's output files.
        ensure_inference (Callable[[], InferencePipeline]): Accessor for the
            run's inference pipeline, created on first use.
        audit (JobAudit): The run's audit trail.

    Returns:
        list[FrameCaption]: The captions, empty when the step was off, the
            file has no frames, or nothing was captioned.
    """
    if not args.keyframes:
        return []
    rates = load_keyframe_defaults()
    samples = extract_keyframe_samples(
        args.file_path,
        per_minute=rates.per_minute,
        max_frames=rates.max_frames,
    )
    if not samples:
        logger.info("No keyframes sampled from '{}' (no video stream, or sampling is off).", args.file_path)
        return []
    file_processor.write_keyframes([sample.jpeg for sample in samples], [sample.time_sec for sample in samples])
    audit.disclose_file("keyframes")

    visual_cfg = load_visual_summary_env()
    if not visual_cfg.enabled:
        return []
    caption_pipeline = ensure_inference()
    with audit.stage(CAPTION_STAGE) as notes:
        captions = describe_keyframes(
            samples,
            caption_pipeline,
            max_frames=visual_cfg.max_frames,
            max_side=visual_cfg.max_side,
        )
        notes.result("chat", text=format_visual_context(captions) if captions else None)
    if captions:
        _write_output(file_processor, audit, format_visual_context(captions), "visual_context")
    return captions


def _scratch_dir() -> Path | None:
    """Return where the temporary audio re-encode goes: the CLI's own directory under ``make cli``.

    In ``/tmp/nextext-cli`` the backend's sweep (and ``make cli-clean``) reach
    the copy if a run dies; anywhere else it is the system's temp directory.

    Returns:
        Path | None: The directory, or ``None`` for the default.
    """
    root = cli_files.CLI_ROOT
    return root if Path.cwd() == root else None


def _utc_now() -> datetime:
    """Return the current time in UTC (a seam for tests).

    Returns:
        datetime: Now.
    """
    return datetime.now(UTC)


def _deletion_notice() -> str | None:
    """Say when this run's outputs are deleted, in the response language.

    The backend deletes what CLI runs left on the scratch volume once the
    retention period (``NEXTEXT_CLI_RETENTION_DAYS``) has passed, by the rule
    its state records (read here, never written); this line is the notice to
    the person who ran it. The daily sweep can land up to a day later.

    Returns:
        str | None: The sentence, or ``None`` when retention is off or its state cannot be read.
    """
    config = load_cli_retention_env()
    if config.days is None:
        return None
    now = _utc_now().timestamp()
    try:
        rule = cli_files.read_rule(cli_files.CLI_ROOT, days=config.days, notice_days=config.notice_days, now=now)
    except (OSError, ValueError):
        return None
    day = datetime.fromtimestamp(rule.deadline(now), UTC).date().isoformat()
    if load_language_env().code == "de":
        return f"Die Ausgaben werden frühestens am {day} gelöscht (Aufbewahrungsfrist {config.days} Tage)."
    return f"Outputs are deleted not before {day} (retention period {config.days} days)."


def _write_output(
    file_processor: FileProcessor,
    audit: JobAudit,
    data: str | pd.DataFrame | Figure,
    label: str,
) -> None:
    """Write one output file and record its disclosure (``file:<label>``).

    Args:
        file_processor (FileProcessor): Writer for this run's output files.
        audit (JobAudit): The run's audit trail.
        data (str | pd.DataFrame | Figure): What to write.
        label (str): The output's name, e.g. ``summary``.
    """
    file_processor.write_file_output(data, label)
    audit.disclose_file(label)


def _write_transcript(file_processor: FileProcessor, audit: JobAudit, transcript_df: pd.DataFrame) -> None:
    """Write the transcript files and record their disclosure.

    Args:
        file_processor (FileProcessor): Writer for this run's output files.
        audit (JobAudit): The run's audit trail.
        transcript_df (pd.DataFrame): The transcript, optionally translated.
    """
    file_processor.write_transcript_output(transcript_df)
    audit.disclose_file("transcript")
    if "translation" in transcript_df.columns:
        audit.disclose_file("translation")


def _write_docint(args: argparse.Namespace, audit: JobAudit, transcript_df: pd.DataFrame, language: str | None) -> None:
    """Write the docint JSONL export when asked for, and record it.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        audit (JobAudit): The run's audit trail.
        transcript_df (pd.DataFrame): The transcript.
        language (str | None): The transcript's source language.
    """
    if getattr(args, "emit_docint_jsonl", None) is None:
        return
    written = _emit_docint_jsonl(
        transcript_df=transcript_df,
        source_path=args.file_path,
        output_path=args.emit_docint_jsonl,
        language=language,
        force_overwrite=getattr(args, "force_docint_jsonl", False),
    )
    if written is not None:
        audit.disclose_file("docint.jsonl")


def _record_combination(
    audit: JobAudit,
    transcript_df: pd.DataFrame,
    *,
    word_counts: bool = False,
    named_entities: bool = False,
    summary: bool = False,
    captions: bool = False,
    hate_speech: bool = False,
) -> None:
    """Record the run's ``combine`` event, if its result joins several analyses.

    Args:
        audit (JobAudit): The run's audit trail.
        transcript_df (pd.DataFrame): The transcript; empty when none.
        word_counts (bool): Whether word counts were produced.
        named_entities (bool): Whether named entities were produced.
        summary (bool): Whether a summary was produced.
        captions (bool): Whether keyframe captions were produced.
        hate_speech (bool): Whether hate-speech detection ran.
    """
    analyses, speaker_attributed = joined_analyses(
        transcript_columns=[] if transcript_df.empty else list(transcript_df.columns),
        word_counts=word_counts,
        named_entities=named_entities,
        summary=summary,
        frame_captions=captions,
        faces=False,
        hate_speech=hate_speech,
    )
    audit.combine(analyses, speaker_attributed=speaker_attributed)


def _run_main(args: argparse.Namespace, audit: JobAudit) -> int:
    """Execute the pipeline, then say when what it left is deleted, however it ended.

    Args:
        args (argparse.Namespace): Parsed CLI arguments from
            :func:`parse_arguments`.
        audit (JobAudit): The run's audit trail.

    Returns:
        int: As :func:`_run_steps`.
    """
    try:
        return _run_steps(args, audit)
    finally:
        # The notice never changes how the run ended.
        try:
            notice = _deletion_notice()
            if notice is not None:
                print(notice, file=sys.stderr)
        except Exception as exc:
            logger.warning("No deletion notice could be given ({}).", type(exc).__name__)


def _run_steps(args: argparse.Namespace, audit: JobAudit) -> int:
    """Execute the main Nextext pipeline on the parsed CLI arguments.

    Args:
        args (argparse.Namespace): Parsed CLI arguments from
            :func:`parse_arguments`.
        audit (JobAudit): The run's audit trail; every model call, the
            joined result and every output file are recorded on it.

    Returns:
        int: ``0`` for a normal run; ``3`` when the file yielded no
            transcript (see :mod:`nextext.core.outcomes`). ``2`` is reserved
            by argparse for usage errors, so it is deliberately not reused.

    Raises:
        ValueError: If ``args.task`` is not ``"transcribe"`` or
            ``"translate"``, or if the source language cannot be resolved.
        ConnectionError: If the configured inference provider is not reachable.
    """
    # Set up input/output directories and file paths
    file_processor = FileProcessor(args.file_path)

    inference_pipeline: InferencePipeline | None = None

    def ensure_inference() -> InferencePipeline:
        """Return the run's inference pipeline, creating it on first use.

        The provider is health-checked once, when the pipeline is built, so a
        step that needs inference on an unreachable provider stops the run
        instead of quietly producing nothing.

        Returns:
            InferencePipeline: The shared pipeline for this run.

        Raises:
            ConnectionError: If the configured provider is unreachable.
        """
        nonlocal inference_pipeline
        if inference_pipeline is None:
            candidate = InferencePipeline()
            if not candidate.get_health():
                logger.error("The configured inference provider is not reachable.")
                raise ConnectionError(
                    "The configured inference provider is not reachable. Please ensure it is running and accessible."
                )
            inference_pipeline = candidate
        return inference_pipeline

    # Transcribe and diarize the audio file
    if args.task in ["transcribe", "translate"]:
        with audit.stage(TRANSCRIPTION_STAGE) as notes:
            outcome = transcription_pipeline(
                file_path=args.file_path,
                src_lang=args.src_lang,
                diarize=args.diarize,
                scratch_dir=_scratch_dir(),
            )
            notes.result("asr", text=column_text(outcome.transcript, "text"), segments=len(outcome.transcript))
        transcript_df = outcome.transcript
        args.src_lang = outcome.src_lang  # Update source language if detected

        # Keyframes: an opt-in step of its own. It runs ahead of the no-speech
        # guard below, so a silent video still yields what could be seen — and
        # its captions feed the summary and the hate-speech pass when requested.
        captions = _keyframe_step(args, file_processor, ensure_inference, audit)
        visual_context = format_visual_context(captions) or None

        # Guard: stop early when the transcript contains no speech
        if outcome.is_empty:
            logger.warning(
                "No transcript produced for '{}' (reason: {}): {} Skipping analysis.",
                args.file_path,
                outcome.skip_reason or "unknown",
                skip_reason_text(outcome.skip_reason),
            )
            _write_transcript(file_processor, audit, transcript_df)
            # The captions are a summarizable source in their own right, so a
            # requested summary is still written — from the visuals alone.
            visual_summary = ""
            if args.summarize and visual_context:
                summary_pipeline = ensure_inference()
                with audit.stage(VISUAL_SUMMARY_STAGE) as notes:
                    visual_summary = summarization_pipeline(
                        text="",
                        inference_pipeline=summary_pipeline,
                        visual_context=visual_context,
                    )
                    notes.result("chat", text=visual_summary, inputs=["visual_context"])
                if visual_summary:
                    _write_output(file_processor, audit, visual_summary, "summary")
            # Likewise the captions can show hate a silent clip never speaks.
            judged_frames = bool(args.hate_speech and captions)
            if judged_frames:
                frame_pipeline = ensure_inference()
                with audit.stage(FRAME_HATE_SPEECH_STAGE) as notes:
                    frame_findings = frame_hate_speech_pipeline(captions, inference_pipeline=frame_pipeline)
                    notes.result("chat", frames=len(captions))
                if frame_findings:
                    _write_output(file_processor, audit, pd.DataFrame(frame_findings), "hate_speech")
            _record_combination(
                audit, transcript_df, summary=bool(visual_summary), captions=bool(captions), hate_speech=judged_frames
            )
            # Honour an explicit export request even here: the helper warns
            # that there are no segments rather than leaving the caller to
            # wonder why its target file never appeared.
            _write_docint(args, audit, transcript_df, None)
            return 3
    else:
        logger.error("Invalid task specified: {}", args.task)
        raise ValueError("Invalid task. Please specify 'transcribe' or 'translate'.")

    # Machine translate the transcribed text. Whisper only transcribes; the LLM
    # performs every translation, directly from source to target. Engage it
    # whenever a translate task was requested and the resolved source differs
    # from the target (so an English target is translated too).
    if should_translate(args.task, args.src_lang, args.trg_lang):
        translation_inference = ensure_inference()
        with audit.stage(TRANSLATION_STAGE) as notes:
            transcript_df = translation_pipeline(
                df=transcript_df,
                trg_lang=args.trg_lang,
                src_lang=args.src_lang,
                inference_pipeline=translation_inference,
            )
            notes.result(
                "chat",
                text=column_text(transcript_df, "translation"),
                src_lang=args.src_lang,
                trg_lang=args.trg_lang,
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
    has_word_counts = has_entities = False
    if args.words:
        with audit.stage(WORDLEVEL_STAGE):
            word_counts, named_entities, wordcloud_fig = wordlevel_pipeline(
                data=transcript_df,
                language=transcript_lang,
            )
        has_word_counts = isinstance(word_counts, pd.DataFrame) and not word_counts.empty
        has_entities = isinstance(named_entities, pd.DataFrame) and not named_entities.empty
        exports = [
            (word_counts, "words"),
            (named_entities, "entities"),
            (wordcloud_fig, "wordcloud"),
        ]
        for export, name in exports:
            if export is not None:
                _write_output(file_processor, audit, export, name)

    transcript_summary = ""
    if args.summarize:
        # Summarize the transcribed text (translated text when available),
        # together with the keyframe captions when the keyframe step ran.
        summary_text_column = effective_text_column(transcript_df)
        summary_inference = ensure_inference()
        with audit.stage(SUMMARY_STAGE) as notes:
            transcript_summary = summarization_pipeline(
                text=" ".join(transcript_df[summary_text_column].astype(str).tolist()),
                inference_pipeline=summary_inference,
                visual_context=visual_context,
            )
            summary_input = "translation" if summary_text_column == "translation" else "transcript"
            notes.result(
                "chat",
                text=transcript_summary,
                inputs=[summary_input, *(["visual_context"] if visual_context else [])],
            )
        if transcript_summary:
            _write_output(file_processor, audit, transcript_summary, "summary")

    # Hate speech detection — recorded by what was judged, never by what was found.
    if args.hate_speech:
        hate_speech_inference = ensure_inference()
        with audit.stage(HATE_SPEECH_STAGE) as notes:
            hate_speech_findings = hate_speech_pipeline(
                df=transcript_df,
                inference_pipeline=hate_speech_inference,
                src_lang=args.src_lang,
            )
            hate_speech_findings += frame_hate_speech_pipeline(captions, inference_pipeline=hate_speech_inference)
            notes.result("chat", segments=len(transcript_df), frames=len(captions))
        if hate_speech_findings:
            _write_output(file_processor, audit, pd.DataFrame(hate_speech_findings), "hate_speech")
            logger.info("Hate speech detected in {} segment(s).", len(hate_speech_findings))
        else:
            logger.info("No hate speech detected.")

    _record_combination(
        audit,
        transcript_df,
        word_counts=has_word_counts,
        named_entities=has_entities,
        summary=bool(transcript_summary),
        captions=bool(captions),
        hate_speech=bool(args.hate_speech),
    )

    # Save final transcript
    _write_transcript(file_processor, audit, transcript_df)

    # Optional: emit a docint-compatible JSONL payload next to the transcript.
    # ``_transcript_segments_from_df`` returns integer-valued floats for
    # ``start_seconds`` / ``end_seconds`` — they parse the already-rounded
    # ``HH:MM:SS`` strings produced by ``_seconds_to_time``. docint always
    # receives the original (untranslated) transcript text — see
    # ``transcript_segments_from_df`` — so ``language`` is the resolved
    # source language here, not ``transcript_lang`` (which is the target for
    # a ``translate`` task and only describes the downstream-analysis text).
    _write_docint(args, audit, transcript_df, detected_src_lang)

    logger.info("The end of our elaborate plans, the end of everything that stands.")
    return 0


if __name__ == "__main__":
    main()
