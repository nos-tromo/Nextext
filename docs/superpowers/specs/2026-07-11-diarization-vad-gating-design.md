# Diarization VAD-gating — design

**Date:** 2026-07-11
**Status:** Approved
**Branch:** `feature/diarization-always-on-wordlevel` (adds to PR #91)

## Problem

The `/diarize` backend (stock pyannote) over-detects music/noise as speech —
its *speaker*-segmentation is permissive on non-speech. Measured on VoxConverse:
false alarm is the dominant DER term on clean content (53% of the error) and is
**model-independent** (identical for 3.1 and community-1). It is the client's
*"sung/music intro scored as the narrator"* defect. Silero VAD rejects music/noise
far better; gating the diarization by VAD cut DER −12.5% (FA −35%) in the
`vllm-service` eval harness (report:
`vllm-service/eval/reports/2026-07-11-false-alarm-vad-gating.md`). Nextext already
calls the `/vad` service (as a pre-Whisper guard) — this reuses it to gate
diarization.

## Approach (decided)

Intersect the `/diarize` turns with the `/vad` speech timeline **before**
canonicalization/alignment, using tuned Silero params (threshold ≈ 0.4,
`speech_pad_ms` ≈ 100 — Silero's default 0.5 over-cuts real speech). **On by
default**; **crop-to-speech** semantics (a turn straddling a music gap splits
into its speech-only pieces — the measured `crop(vad)` behavior).

## Components

### `nextext/core/vad.py`
Add `speech_segments(file_path, *, threshold: float, pad_ms: int) -> list[tuple[float, float]] | None`.
Calls `POST {base}/vad` with `threshold` + `speech_pad_ms` form fields; parses the
response `segments` (each `{start,end}`) into `(start, end)` tuples. **Fail-open
for gating:** returns `None` when no endpoint resolves (off token / no central
fallback), on any transport/HTTP error, or a malformed payload — the caller then
skips gating and keeps every turn (a VAD outage must never silently drop speaker
labels). `has_speech` is unchanged.

### `nextext/core/diarization.py`
Add `gate_turns_by_vad(turns: list[dict], vad_intervals: list[tuple[float, float]]) -> list[dict]`.
For each turn, intersect `[start, end]` with each speech interval and emit one
turn per overlapping piece (`{**turn, "start": max(...), "end": min(...)}`),
preserving `speaker`. A turn entirely in non-speech yields nothing (dropped); a
turn spanning a gap yields multiple fragments. Empty `vad_intervals` → return
`turns` unchanged (fail-safe: never blank a transcript on an empty VAD result).

### `nextext/pipeline.py`
In `transcription_pipeline`, gate between `diarize_file` and
`canonicalize_speaker_labels`:

```python
turns = diarize_file(file_path)
if turns and gate.enabled:
    vad_segs = speech_segments(file_path, threshold=gate.threshold, pad_ms=gate.pad_ms)
    if vad_segs is not None:
        turns = gate_turns_by_vad(turns, vad_segs)
turns = canonicalize_speaker_labels(turns)
if turns:
    transcriber.transcription_result["segments"] = build_speaker_segments(segments, words, turns)
```

So canonical `Speaker N` numbering reflects the gated turns. Costs one extra
`/vad` call per diarized job (decoupled from the `has_speech` guard for clarity).

### `nextext/utils/env_cfg.py`
`load_diarize_vad_gate_env() -> DiarizeVadGateConfig(enabled: bool, threshold: float, pad_ms: int)`:
- `NEXTEXT_DIARIZE_VAD_GATE` — on/off (`1/true/yes/on` vs `0/false/no/off`), **default on**.
- `VAD_GATE_THRESHOLD` — float, default `0.4`; invalid/≤0 or >1 warns → default.
- `VAD_GATE_PAD_MS` — int, default `100`; invalid/<0 warns → default.

## Error handling

| Situation | Behavior |
|---|---|
| `NEXTEXT_DIARIZE_VAD_GATE` off | No gating; turns unchanged. |
| `/vad` unset/off/unreachable/malformed | `speech_segments` → `None` → gating skipped (all turns kept). |
| VAD returns empty segments | `gate_turns_by_vad` returns turns unchanged (fail-safe). |
| `diarize=False` / empty transcript | Diarization (and gating) skipped, as today. |

## Tests (TDD)

- `tests/test_vad.py` — `speech_segments`: parses segments + passes `threshold`/`speech_pad_ms` (respx); `None` on off-token / HTTP error / non-dict / missing `segments`.
- `tests/test_diarization.py` — `gate_turns_by_vad`: turn straddling a gap → split; turn fully in non-speech → dropped; turn fully in speech → unchanged; empty intervals → passthrough; speaker preserved.
- `tests/test_pipeline.py` — gating applied when enabled + turns + non-None VAD; skipped when disabled or VAD `None`; canonicalize runs on gated turns.
- `tests/test_env_cfg.py` — the gate config loader defaults + parsing/fallbacks.
- Docs: `.env.example`, `CLAUDE.md`, `AGENTS.md` (VAD agent I/O contract).

## Out of scope

- Reusing the `has_speech` guard's VAD call (a later optimization; kept separate).
- Backend-side gating in `diarize_server` (the client-side placement was chosen).
- Tuning threshold/pad on client-specific content (operators tune via env).
