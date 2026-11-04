"""Data-protection audit log: the process's chain and one job's events.

Nextext records its processing of personal data (§76 BDSG: collection,
query, disclosure, combination, erasure, and every transfer to a model) as
hash-chained audit events, written by the federation library vendored in
``nextext/utils/audit_event.py``; ``nextext/utils/audit_runtime.py`` holds the
parts every emitting app does the same way (the output stream, the purpose
header, request ids). Events carry references, counts and parameters — never
media, transcripts, prompts or model output. The store, its retention and who
may read it live in obs-plane (its ``docs/audit.md``); ``docs/audit.md`` in
this repo lists what Nextext records.

This module is shared by the API and the CLI:

* :func:`build_emitter` creates the one emitter of a process, and
  :func:`emit_config_event` records the mode it runs in;
* :class:`AuditSink` puts the emitter behind a gate that closes before the
  chain does, so a pipeline thread still running at shutdown cannot write an
  event after the chain's last one;
* :class:`JobAudit` writes every event of one job (an API job or a CLI run),
  all carrying the actor, purpose and request id of the request that started
  it (:class:`Origin`);
* the ``*_STAGE`` tables say how the inference requests of each pipeline stage
  are recorded: one ``ai_call`` event per task, built from the stage's
  :mod:`nextext.core.inference_trace` tally.

Writing an event that fails raises (:class:`~nextext.utils.audit_event.AuditWriteError`,
or ``RuntimeError`` from a closed emitter) and the operation fails with it: a
request answers 500 without its data, a job fails, a CLI run exits non-zero.
"""

from __future__ import annotations

import threading
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Final

from nextext.core.inference_trace import InferenceTrace, trace_inference
from nextext.utils.audit_event import Actor, AiCall, AuditEmitter, Purpose, Target, sha256_text
from nextext.utils.audit_runtime import id_chunks, open_audit_stream

__all__ = [
    "CAPTION_STAGE",
    "FRAME_HATE_SPEECH_STAGE",
    "HATE_SPEECH_STAGE",
    "SERVICE",
    "SUMMARY_STAGE",
    "TRANSCRIPTION_STAGE",
    "TRANSLATION_STAGE",
    "VISUAL_SUMMARY_STAGE",
    "WORDLEVEL_STAGE",
    "AiSpec",
    "AuditSink",
    "JobAudit",
    "Origin",
    "StageNotes",
    "build_emitter",
    "emit_config_event",
    "emit_with_ids",
    "joined_analyses",
    "service_version",
]

#: The compose project name, which the audit verifier binds every chain to.
SERVICE: Final = "nextext"
#: The operation that brings a job's media in; media-reading stages cite it.
MEDIA: Final = "job.create"
#: Target kind of an API job and of a CLI run.
JOB_KIND: Final = "job"


def service_version() -> str:
    """Return the installed package version, which every event carries.

    Returns:
        str: The ``nextext`` distribution version, or ``0+unknown`` when the
            package is not installed (running from a bare source tree).
    """
    try:
        return version("nextext")
    except PackageNotFoundError:
        return "0+unknown"


def build_emitter(output: str | None, *, heartbeat_interval_s: float = 60.0) -> AuditEmitter:
    """Create the process's emitter, writing where ``AUDIT_OUTPUT`` says.

    Args:
        output: The ``AUDIT_OUTPUT`` value: ``None`` or ``stdout`` for the
            process's stdout, ``container`` for the container's main stdout
            under the container-wide lock.
        heartbeat_interval_s: Seconds between heartbeat events.

    Returns:
        AuditEmitter: The emitter, not yet started.

    Raises:
        ValueError: If ``output`` names neither mode.
        OSError: If the container stream cannot be opened; audited work must
            not start then.
    """
    return AuditEmitter(
        SERVICE,
        service_version(),
        stream=open_audit_stream(output),
        heartbeat_interval_s=heartbeat_interval_s,
    )


def emit_config_event(emitter: AuditEmitter, criteria: Mapping[str, object]) -> dict[str, object]:
    """Record the operating mode the chain starts in (``admin`` / ``config.audit``).

    Args:
        emitter: The started emitter.
        criteria: The mode, e.g. whether the edge secret is enforced and which
            actions need a purpose — never secrets or paths.

    Returns:
        dict[str, object]: The event as written.
    """
    return emitter.emit("admin", "config.audit", Actor(None, "system"), criteria=criteria)


class AuditSink:
    """The emitter behind a gate that closes before the chain does.

    Pipeline threads cannot be stopped, so a job still running at shutdown
    could otherwise write a stage event after the chain's ``shutdown`` event.
    Every job event goes through :meth:`emit`; :meth:`seal` waits for an event
    being written and refuses all later ones, and the owner then writes its
    final events on :attr:`emitter` itself before closing it.
    """

    def __init__(self, emitter: AuditEmitter) -> None:
        """Wrap an emitter; the gate starts open.

        Args:
            emitter: The process's started emitter.
        """
        self._emitter = emitter
        self._lock = threading.Lock()
        self._sealed = False

    @property
    def emitter(self) -> AuditEmitter:
        """The wrapped emitter, for events written after :meth:`seal`.

        Returns:
            AuditEmitter: The emitter.
        """
        return self._emitter

    def seal(self) -> None:
        """Close the gate: later :meth:`emit` calls raise ``RuntimeError``."""
        with self._lock:
            self._sealed = True

    def emit(
        self,
        action: str,
        operation: str,
        actor: Actor,
        *,
        outcome: str = "ok",
        request_id: str | None = None,
        parent_event_id: str | None = None,
        purpose: Purpose | None = None,
        target: Target | None = None,
        criteria: Mapping[str, object] | None = None,
        recipient: str | None = None,
        ai: AiCall | None = None,
    ) -> dict[str, object]:
        """Write one event, unless the gate is sealed.

        Args:
            action: The audit action.
            operation: The dotted operation name.
            actor: Who caused it.
            outcome: ``ok``, ``denied`` or ``error``.
            request_id: The correlation id of the originating request.
            parent_event_id: The event this one belongs to.
            purpose: The stated purpose.
            target: What was touched, by reference.
            criteria: Parameters and counts.
            recipient: Where data went.
            ai: Model provenance.

        Returns:
            dict[str, object]: The event as written.

        Raises:
            RuntimeError: If the gate is sealed or the emitter closed.
            AuditWriteError: If the line could not be written.
        """
        with self._lock:
            if self._sealed:
                raise RuntimeError("audit sink is sealed")
            return self._emitter.emit(
                action,
                operation,
                actor,
                outcome=outcome,
                request_id=request_id,
                parent_event_id=parent_event_id,
                purpose=purpose,
                target=target,
                criteria=criteria,
                recipient=recipient,
                ai=ai,
            )


def emit_with_ids(
    sink: AuditSink | AuditEmitter,
    action: str,
    operation: str,
    actor: Actor,
    *,
    kind: str,
    ids: Sequence[str],
    container: str | None = None,
    outcome: str = "ok",
    request_id: str | None = None,
    purpose: Purpose | None = None,
    criteria: Mapping[str, object] | None = None,
    recipient: str | None = None,
) -> dict[str, object]:
    """Write an event naming any number of ids, never cutting the list.

    The first :data:`~nextext.utils.audit_runtime.MAX_IDS_PER_EVENT` ids go on
    the event itself, which counts all of them; each further chunk goes on a
    continuation event (``<operation>.ids``) pointing at it.

    Args:
        sink: Where to write (an :class:`AuditSink` or the emitter itself).
        action: The audit action.
        operation: The dotted operation name.
        actor: Who caused it.
        kind: The target kind of every id.
        ids: The ids, possibly more than fit one event.
        container: The enclosing object of the targets, if any.
        outcome: ``ok``, ``denied`` or ``error``.
        request_id: The correlation id.
        purpose: The stated purpose.
        criteria: Parameters, on the first event only.
        recipient: Where data went.

    Returns:
        dict[str, object]: The first event as written.
    """
    chunks = id_chunks(ids) or [[]]
    primary = sink.emit(
        action,
        operation,
        actor,
        outcome=outcome,
        request_id=request_id,
        purpose=purpose,
        target=Target(kind=kind, ids=chunks[0], count=len(ids), container=container),
        criteria=criteria,
        recipient=recipient,
    )
    for chunk in chunks[1:]:
        sink.emit(
            action,
            f"{operation}.ids",
            actor,
            outcome=outcome,
            request_id=request_id,
            parent_event_id=str(primary["event_id"]),
            purpose=purpose,
            target=Target(kind=kind, ids=chunk, count=len(chunk), container=container),
            recipient=recipient,
        )
    return primary


@dataclass(frozen=True)
class Origin:
    """Who started a job, for what stated purpose, under which request.

    A job's events are written in the background, long after the request
    returned; they all carry its origin.

    Attributes:
        actor: The caller, as the identity seam established it.
        purpose: The caller's stated purpose, if any.
        request_id: The gateway's correlation id, if the request had one.
    """

    actor: Actor
    purpose: Purpose | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class AiSpec:
    """How one inference task of a stage is recorded.

    Attributes:
        operation: The event's operation, e.g. ``ai.transcribe``.
        recipient: Where the content went, ``inference:<task>``.
        consumes: Earlier operations whose output the task was given; their
            event ids become the event's ``ai.sources``. ``job.create`` stands
            for the uploaded media.
    """

    operation: str
    recipient: str
    consumes: tuple[str, ...] = ()


_TEXT_SOURCES: Final = ("ai.transcribe", "ai.sentence_restore", "ai.translate")

#: VAD guard, Whisper, diarization and sentence restoration (stage 1).
TRANSCRIPTION_STAGE: Final[Mapping[str, AiSpec]] = {
    "vad": AiSpec("ai.vad", "inference:vad", (MEDIA,)),
    "asr": AiSpec("ai.transcribe", "inference:asr", (MEDIA,)),
    "diarize": AiSpec("ai.diarize", "inference:diarize", (MEDIA,)),
    "chat": AiSpec("ai.sentence_restore", "inference:chat", ("ai.transcribe", "ai.diarize")),
}
#: Keyframe captions: the sampled frames go to the vision-capable chat model.
CAPTION_STAGE: Final[Mapping[str, AiSpec]] = {
    "chat": AiSpec("ai.caption_frames", "inference:chat", (MEDIA,)),
}
#: Segment translation by the chat model.
TRANSLATION_STAGE: Final[Mapping[str, AiSpec]] = {
    "chat": AiSpec("ai.translate", "inference:translate", ("ai.transcribe", "ai.sentence_restore")),
}
#: Word-level analysis: named entities from the ``/gliner`` service.
WORDLEVEL_STAGE: Final[Mapping[str, AiSpec]] = {
    "ner": AiSpec("ai.ner", "inference:ner", _TEXT_SOURCES),
}
#: Summary of the transcript and, when there are any, the frame captions.
SUMMARY_STAGE: Final[Mapping[str, AiSpec]] = {
    "chat": AiSpec("ai.summarize", "inference:chat", (*_TEXT_SOURCES, "ai.caption_frames")),
}
#: Summary of a silent clip, written from the frame captions alone.
VISUAL_SUMMARY_STAGE: Final[Mapping[str, AiSpec]] = {
    "chat": AiSpec("ai.summarize", "inference:chat", ("ai.caption_frames",)),
}
#: Hate-speech classification of the transcript windows and the frame captions.
HATE_SPEECH_STAGE: Final[Mapping[str, AiSpec]] = {
    "chat": AiSpec("ai.hate_speech", "inference:chat", (*_TEXT_SOURCES, "ai.diarize", "ai.caption_frames")),
}
#: Hate-speech classification of a silent clip's frame captions alone.
FRAME_HATE_SPEECH_STAGE: Final[Mapping[str, AiSpec]] = {
    "chat": AiSpec("ai.hate_speech", "inference:chat", ("ai.caption_frames",)),
}


@dataclass
class StageNotes:
    """What a stage reports about its results, for its ``ai_call`` events.

    Attributes:
        texts: Per task, the text the task produced; recorded only as its
            SHA-256 (``ai.output_sha256``).
        counts: Per task, extra counts for the event's criteria.
    """

    texts: dict[str, str] = field(default_factory=dict)
    counts: dict[str, dict[str, object]] = field(default_factory=dict)

    def result(self, task: str, *, text: str | None = None, **counts: object) -> None:
        """Note a task's text output and counts.

        Args:
            task: The inference task the note is for, e.g. ``asr``.
            text: The text the task produced, hashed into the event; ``None``
                or empty records no hash.
            **counts: Counts or parameters for the event's criteria — never
                content.
        """
        if text:
            self.texts[task] = text
        if counts:
            self.counts.setdefault(task, {}).update(counts)


def joined_analyses(
    *,
    transcript_columns: Collection[str],
    word_counts: bool,
    named_entities: bool,
    summary: bool,
    frame_captions: bool,
    hate_speech: bool,
) -> tuple[list[str], bool]:
    """Name the analyses a job's result joins, for its ``combine`` event.

    Args:
        transcript_columns: The columns of the job's transcript; empty when
            it produced none. Speaker labels (``speaker``) mean diarization
            was joined to the transcript, a ``translation`` column its
            translation.
        word_counts: Whether the result holds word counts.
        named_entities: Whether the result holds named entities.
        summary: Whether the result holds a summary.
        frame_captions: Whether the result holds keyframe captions.
        hate_speech: Whether hate-speech detection ran — whatever it found,
            so the event never shows whether it found anything.

    Returns:
        tuple[list[str], bool]: The analyses joined, and whether the results
            carry speaker labels.
    """
    columns = set(transcript_columns)
    speaker_attributed = "speaker" in columns
    present = (
        ("transcript", bool(columns)),
        ("diarization", speaker_attributed),
        ("translation", "translation" in columns),
        ("word_counts", word_counts),
        ("named_entities", named_entities),
        ("summary", summary),
        ("frame_captions", frame_captions),
        ("hate_speech", hate_speech),
    )
    return [name for name, joined in present if joined], speaker_attributed


class JobAudit:
    """Every audit event of one job: an API job or a CLI run.

    The events target the job (``kind="job"``), carry the origin's actor,
    purpose and request id, and point at the job's ``collect`` event as their
    parent, so one job's trail can be followed from the upload on.
    """

    def __init__(self, sink: AuditSink, job_id: str, origin: Origin) -> None:
        """Prepare the job's audit trail; nothing is written yet.

        Args:
            sink: Where the events go.
            job_id: The job (or CLI run) id.
            origin: Who started the job, why, and under which request.
        """
        self._sink = sink
        self.job_id = job_id
        self.origin = origin
        # Operation -> event id of the job's successful events, for parents
        # and model sources.
        self._produced: dict[str, str] = {}

    @property
    def target(self) -> Target:
        """The job, as every one of its events names it.

        Returns:
            Target: ``kind="job"``, the job id, count 1.
        """
        return Target(kind=JOB_KIND, ids=[self.job_id], count=1)

    @property
    def collect_event_id(self) -> str | None:
        """The id of the job's ``collect`` event, once written.

        Returns:
            str | None: The event id, or ``None`` before :meth:`collect`.
        """
        return self._produced.get(MEDIA)

    def emit(
        self,
        action: str,
        operation: str,
        *,
        outcome: str = "ok",
        criteria: Mapping[str, object] | None = None,
        recipient: str | None = None,
        ai: AiCall | None = None,
    ) -> dict[str, object]:
        """Write one event about the job.

        Args:
            action: The audit action.
            operation: The dotted operation name.
            outcome: ``ok``, ``denied`` or ``error``.
            criteria: Parameters and counts — never content.
            recipient: Where data went.
            ai: Model provenance.

        Returns:
            dict[str, object]: The event as written.
        """
        return self._sink.emit(
            action,
            operation,
            self.origin.actor,
            outcome=outcome,
            request_id=self.origin.request_id,
            parent_event_id=self.collect_event_id,
            purpose=self.origin.purpose,
            target=self.target,
            criteria=criteria,
            recipient=recipient,
            ai=ai,
        )

    def collect(self, criteria: Mapping[str, object]) -> dict[str, object]:
        """Record that the job's media entered the system (``collect`` / ``job.create``).

        Args:
            criteria: File count and size, the media's hash and the options.

        Returns:
            dict[str, object]: The event as written; later events name it as
                their parent.
        """
        event = self.emit("collect", MEDIA, criteria=criteria)
        self._produced[MEDIA] = str(event["event_id"])
        return event

    @contextmanager
    def stage(self, specs: Mapping[str, AiSpec]) -> Iterator[StageNotes]:
        """Record every inference request a pipeline stage makes.

        Opens an inference trace around the block; when the block ends —
        normally or by an exception — one ``ai_call`` event is written per task
        that sent requests, in the order the tasks were first used.

        Args:
            specs: How each task of the stage is recorded; a task without an
                entry is recorded as ``ai.<task>`` to ``inference:<task>``.

        Yields:
            StageNotes: Where the block notes its outputs and counts.
        """
        notes = StageNotes()
        with trace_inference() as trace:
            try:
                yield notes
            except BaseException:
                self._record_stage(trace, specs, notes)
                raise
            self._record_stage(trace, specs, notes)

    def _record_stage(self, trace: InferenceTrace, specs: Mapping[str, AiSpec], notes: StageNotes) -> None:
        """Write the ``ai_call`` events of a finished stage.

        A task is ``ok`` when at least one of its requests returned a usable
        reply, ``error`` when none did; the counts say how many were sent and
        how many failed.

        Args:
            trace: The stage's inference trace.
            specs: How each task is recorded.
            notes: What the stage reported about its results.
        """
        for tally in trace.tallies:
            spec = specs.get(tally.task) or AiSpec(f"ai.{tally.task}", f"inference:{tally.task}")
            outcome = "ok" if tally.succeeded > 0 else "error"
            criteria: dict[str, object] = {"requests": tally.requests}
            if tally.failed:
                criteria["failed_requests"] = tally.failed
            criteria.update(notes.counts.get(tally.task, {}))
            text = notes.texts.get(tally.task) if outcome == "ok" else None
            input_tokens, output_tokens = tally.tokens
            ai = AiCall(
                model_alias=tally.model or tally.task,
                model_id=tally.served_model,
                sources=[self._produced[op] for op in spec.consumes if op in self._produced],
                output_sha256=sha256_text(text) if text else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            event = self.emit(
                "ai_call", spec.operation, outcome=outcome, criteria=criteria, recipient=spec.recipient, ai=ai
            )
            if outcome == "ok":
                self._produced[spec.operation] = str(event["event_id"])

    def combine(self, analyses: Collection[str], *, speaker_attributed: bool) -> dict[str, object] | None:
        """Record that the job's result joins several analyses (``combine`` / ``job.combine``).

        Args:
            analyses: The analyses the result joins, e.g. ``transcript``,
                ``diarization``, ``named_entities``, ``hate_speech``. Names
                only — never what an analysis found.
            speaker_attributed: Whether the results carry speaker labels.

        Returns:
            dict[str, object] | None: The event as written, or ``None`` when
                fewer than two analyses were joined (nothing was combined).
        """
        if len(analyses) < 2:
            return None
        return self.emit(
            "combine",
            "job.combine",
            criteria={"analyses": sorted(analyses), "speaker_attributed": speaker_attributed},
        )

    def disclose_file(self, output: str) -> dict[str, object]:
        """Record that a CLI run wrote an output file (``disclose`` / ``output.write``).

        Args:
            output: What the file holds, e.g. ``transcript`` or ``summary``;
                never its path.

        Returns:
            dict[str, object]: The event as written.
        """
        return self.emit("disclose", "output.write", recipient=f"file:{output}")
