# Faces in videos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A job that asks for `faces` gets the faces of its video grouped per
person, with appearances that seek the player, a best shot per person to pass
on, and the faces in `faces.zip`, `faces.csv`, `face_NNN.jpg` and `archive.zip`.

**Architecture:** A time-based frame sampler (PyAV) feeds an HTTP client for
vllm-service's `/faces/detect` and `/faces/group`; pure-geometry tracking and
quality scoring run in-process (NumPy/Pillow, no model); a new job stage at
index 2 records `ai.detect_faces`/`ai.group_faces`; the SPA gets a Faces tab.

**Tech Stack:** Python 3.12, PyAV 19, Pillow, NumPy, httpx, FastAPI/Pydantic;
React 19, TanStack Query, vitest.

**Spec:** `docs/superpowers/specs/2026-11-07-face-detection-design.md`

## Global Constraints

- No model runs in-process; `tests/test_no_torch.py` forbids `cv2`, `onnxruntime`, `insightface`, `mediapipe`, `dlib`, `face_recognition` too.
- `NEXTEXT_FACES=off|detect|group`, default `off`, read once in the lifespan; `faces=true` while `off` → `422` `"Invalid request."` before `allocate_tmpdir`, no `collect` event.
- Stage label `"Detecting faces"` at index 2 of `PIPELINE_STAGE_LABELS`; it never calls `_ensure_inference()`; it runs before `if outcome.is_empty:`.
- Env: `FACES_API_BASE` (URL / off token / central root), `FACES_TIMEOUT` 300, `FACES_INTERVAL_S` 1.0 (> 0), `FACES_MAX_FRAMES` 600 (ceiling 3600). Constants: batch 8 images ≤ 1280 px, source budget 100 MiB per batch, request budget 16 MiB, ≤ 20 faces/frame, face ≥ 40 px, tracks ≤ 500, entries ≤ 100, IoU ≥ 0.5, size ratio ≤ 1.5, gap ≤ 1 sample, cut threshold 30 (mean |Δ| on a 32×18 grey thumbnail), good face score ≥ 0.85 / ≥ 64 px / frontality ≥ 0.5, ≤ 3 faces per track to `/faces/group`.
- Crops: best = box + 50 % per side, ≤ 1024 px, JPEG q90; crop = box + 25 %, ≤ 256 px, q85; group crop = box + 25 %, ≤ 160 px, q90; thumbnail = box + 15 %, ≤ 128 px, q80. No EXIF.
- Audit by what was judged: `frames` on `face_detect`, `tracks` on `face_group`; never persons, counts of faces or hashes.
- Logs: `error_label(exc)` only; no image bytes, base64 or file names beyond `file_path.name` at INFO as existing stages do.
- Thumbnails only in the finished snapshot (`JobResult`), never in SSE deltas or `GET /jobs` items.
- No CLI flag; `-F` unchanged and documented as never including faces.
- No real faces or videos in tests; synthetic PyAV clips and stubbed clients only.
- Python 3.12 Google docstrings on every new/changed function; `uv run pre-commit run --all-files`, `make verify`, `uv run pytest`, `cd frontend && pnpm test` green before each commit that touches them.
- Version 1.17.0 → 1.18.0.

## Review Focus

- A portrait phone video (display rotation 90°): frames and crops come out upright (Task 2, `test_sample_frames_applies_display_rotation`).
- A clip longer than `FACES_MAX_FRAMES × FACES_INTERVAL_S`: samples spread over the whole clip, not its first ten minutes (Task 2, `test_sample_frames_stretch_interval_over_long_clip`).
- A crowd: more than 20 faces per frame and more than 500 tracks stay bounded and set `truncated` (Task 5, `test_crowd_is_capped_and_truncated`).
- The faces service down for some batches: the rest still yields faces with `outcome="partial"`; all down gives `"unavailable"` and no crash (Task 5, `test_partial_outage_is_partial`, `test_total_outage_is_unavailable`).
- A face touching the frame edge: every crop stays inside the frame (Task 5, `test_edge_face_crop_is_clamped`).

---

### Task 1: Named stage indices

**Files:**
- Modify: `nextext/api/jobs.py` (`PIPELINE_STAGE_LABELS` L85-92; the 16 literal `_notify(n)`/`_complete(n)` calls in `_run_pipeline_blocking` L605-835)
- Test: `tests/test_api/test_stage_sequence.py` (new)

**Interfaces:**
- Produces: `STAGE_TRANSCRIBE`, `STAGE_KEYFRAMES`, `STAGE_TRANSLATE`, `STAGE_WORDS`, `STAGE_SUMMARY`, `STAGE_HATE_SPEECH: Final[int]`, each `PIPELINE_STAGE_LABELS.index(<label>)`.

- [ ] **Step 1: Write the failing test** — run `_run_pipeline_blocking` with the stubbed stages of `tests/test_api/test_audit_pipeline.py`'s `stubbed` fixture pattern (all options on) and collect pushed events:

```python
def test_stage_started_sequence_matches_labels(...) -> None:
    started = [payload["stage"] for name, payload in events if name == "stage_started"]
    assert started == list(PIPELINE_STAGE_LABELS)
    assert [p["stage_index"] for n, p in events if n == "stage_started"] == list(range(len(PIPELINE_STAGE_LABELS)))

def test_stage_constants_name_their_labels() -> None:
    assert PIPELINE_STAGE_LABELS[STAGE_KEYFRAMES] == "Describing keyframes"
```

- [ ] **Step 2: Run it** — `uv run pytest tests/test_api/test_stage_sequence.py -q` → FAIL (`ImportError: STAGE_KEYFRAMES`).
- [ ] **Step 3: Implement** the constants and replace every literal, including the silent-clip branch.
- [ ] **Step 4: Run** `uv run pytest -q` → all pass.
- [ ] **Step 5: Commit** — `git commit -m "refactor(jobs): name the pipeline stage indices"` (with the test).

---

### Task 2: Time-based frame sampler

**Files:**
- Modify: `nextext/core/keyframes.py` (new `SampledFrame`, `sample_frames`, `_sample_time`; export in `__all__`)
- Test: `tests/test_frame_sampler.py` (new; synthesize clips with PyAV like `tests/test_keyframes.py::_make_mjpeg_ramp`)

**Interfaces:**
- Produces: `@dataclass(frozen=True) class SampledFrame: index: int; time_sec: float; image: PIL.Image.Image` (RGB, upright, long side ≤ `max_side`); `sample_frames(file_path: Path, *, interval_s: float, max_frames: int, max_side: int = 3840) -> Iterator[SampledFrame]` — `index` counts yielded samples from 0.

- [ ] **Step 1: Write the failing tests**

```python
def test_sample_frames_one_per_interval(tmp_path) -> None:      # 10 s mpeg4 clip, gop_size=12 at 24 fps, interval 1.0
    frames = list(sample_frames(clip, interval_s=1.0, max_frames=600))
    assert [f.index for f in frames] == list(range(len(frames)))
    assert 9 <= len(frames) <= 11
    assert all(b.time_sec > a.time_sec for a, b in zip(frames, frames[1:]))

def test_sample_frames_stretch_interval_over_long_clip(tmp_path) -> None:  # 60 s clip, max_frames=10
    frames = list(sample_frames(clip, interval_s=1.0, max_frames=10))
    assert len(frames) <= 10 and frames[-1].time_sec >= 50.0

def test_sample_frames_keyframe_path_when_gops_are_short(tmp_path) -> None:  # all-intra MJPEG: same count, keyframe decode chosen
def test_sample_frames_applies_display_rotation(tmp_path) -> None:  # 64x32 clip, stream.set_display_rotation(90) (or side-data equivalent) → image.size == (32, 64)
def test_sample_frames_offsets_start_time(tmp_path) -> None:   # first sample's time_sec < interval even when the stream starts at a non-zero pts
def test_sample_frames_audio_only_yields_nothing(tmp_path) -> None:
def test_sample_frames_corrupt_file_yields_what_it_has(tmp_path) -> None:  # proxy container raising mid-decode, as test_keyframes does
def test_sample_frames_downscales_to_max_side(tmp_path) -> None:
```

- [ ] **Step 2: Run them** — `uv run pytest tests/test_frame_sampler.py -q` → FAIL (`ImportError`).
- [ ] **Step 3: Implement** — open with PyAV; no video stream → return. Set `codec_context.thread_type = "AUTO"` and `thread_count = max(1, (os.cpu_count() or 1) // load_job_concurrency())` before decoding. Effective interval = `max(interval_s, duration / max_frames)` when the duration is known. One demux pass collects keyframe pts; if the median keyframe gap (s) ≤ 2 × interval, decode with `skip_frame="NONKEY"`, else `"NONREF"` (reopen the container for the decode pass). Time = `frame.time`, else `index / average_rate`, minus `stream.start_time * time_base` when set; yield the first frame at or after each tick. Apply `frame.rotation` (rotate the PIL image so it displays upright; pin the direction with the rotated-clip test). Downscale with `frame.reformat(width=…, height=…)` before `to_image()`. Catch `(FFmpegError, ValueError, OSError)` → log `error_label`-style class name, stop.
- [ ] **Step 4: Run** `uv run pytest tests/test_frame_sampler.py tests/test_keyframes.py -q` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(core): time-based video frame sampler for the face stage"`.

---

### Task 3: Faces client and configuration

**Files:**
- Create: `nextext/core/faces.py`
- Modify: `nextext/utils/env_cfg.py` (after `load_vad_env`), `tests/test_no_torch.py`
- Test: `tests/test_faces_client.py` (new), `tests/test_env_cfg.py`, `tests/test_inference_trace.py` (L196-261 parametrization)

**Interfaces:**
- Produces (`env_cfg`): `FacesMode = Literal["off", "detect", "group"]`; `load_faces_mode() -> FacesMode` (`NEXTEXT_FACES`; off tokens → `"off"`; unknown → warning, `"off"`); `@dataclass(frozen=True) class FacesConfig: api_base: str; api_key: str; timeout: float`; `load_faces_env() -> FacesConfig` (`DEFAULT_FACES_TIMEOUT = 300.0`); `@dataclass(frozen=True) class FacesSamplingConfig: interval_s: float; max_frames: int`; `load_faces_sampling_env() -> FacesSamplingConfig`.
- Produces (`faces.py`): `FACES_MODEL_ALIAS = "faces"`; `MAX_REQUEST_BYTES = 16 * 2**20`; `@dataclass(frozen=True) class DetectedFace: box: tuple[float, float, float, float]; score: float; landmarks: tuple[tuple[float, float], ...]` (all normalized to the sent image); `@dataclass(frozen=True) class GroupFace: jpeg: bytes; landmarks: tuple[tuple[float, float], ...]` (crop pixels); `@dataclass(frozen=True) class GroupTrack: samples: tuple[int, ...]; faces: tuple[GroupFace, ...]`; `detect_faces(images: Sequence[bytes], *, min_score: float = 0.7) -> list[list[DetectedFace] | None] | None` (outer `None`: no endpoint or the request failed; inner `None`: that image was undecodable); `group_faces(tracks: Sequence[GroupTrack]) -> list[int] | None` (`None`: no endpoint, failure, malformed reply, label count mismatch, or body over `MAX_REQUEST_BYTES` — the last without a request).

- [ ] **Step 1: Write the failing tests** (monkeypatch `faces.httpx.post`, as `tests/test_vad.py`)

```python
def test_detect_posts_json_with_bearer_and_parses(monkeypatch) -> None:
    # FACES_API_BASE=http://router:4000/ → url "http://router:4000/faces/detect"; json == {"images": [b64...], "min_score": 0.7}
def test_detect_unset_base_with_off_token_makes_no_request(monkeypatch) -> None:
def test_detect_falls_back_to_central_root(monkeypatch) -> None:  # OPENAI_API_BASE=http://r:4000/v1 → http://r:4000/faces/detect
def test_detect_marks_undecodable_entries_none(monkeypatch) -> None:
def test_detect_http_error_is_none_and_logged_by_class(monkeypatch, log_lines) -> None:  # 500 with an echoed body; the body is not in the log
def test_group_returns_labels(monkeypatch) -> None:
def test_group_label_count_mismatch_is_none(monkeypatch) -> None:
def test_group_over_budget_sends_nothing(monkeypatch) -> None:
def test_faces_mode_values(monkeypatch) -> None:  # "", "off", "0" → off; "detect"; "GROUP" → group; "maybe" → off + warning
def test_faces_sampling_env_defaults_and_clamps(monkeypatch) -> None:  # 1.0/600; "0" interval → 1.0 + warning; 9999 frames → 3600 + warning
```

Add `("faces", {"FACES_API_BASE": ...}, call, "face_detect", "faces", ok_payload)` to the trace parametrization; add the six module names to `test_no_torch.py`.

- [ ] **Step 2: Run them** → FAIL (`ModuleNotFoundError: nextext.core.faces`).
- [ ] **Step 3: Implement** — `faces.py` mirrors `vad.py` (module docstring carries the contract, `import httpx as httpx`, `correlation_headers()`, bearer from config, `record_inference("face_detect" | "face_group", FACES_MODEL_ALIAS, ok=…)` once per request, `error_label(exc)` in warnings); images go as `base64.b64encode(...).decode("ascii")`. `env_cfg` loaders copy `load_vad_env`'s shape and use `_load_positive_int` for `FACES_MAX_FRAMES` with `ceiling=3600`.
- [ ] **Step 4: Run** `uv run pytest tests/test_faces_client.py tests/test_env_cfg.py tests/test_inference_trace.py tests/test_no_torch.py -q` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(core): HTTP client for the faces service and its configuration"`.

---

### Task 4: Tracks and quality (pure geometry)

**Files:**
- Create: `nextext/core/face_tracks.py`
- Test: `tests/test_face_tracks.py`

**Interfaces:**
- Produces: `Box = tuple[int, int, int, int]` (x, y, w, h source pixels); `@dataclass(frozen=True) class Detection: sample: int; time_sec: float; box: Box; score: float; landmarks: tuple[tuple[float, float], ...]` (source pixels); `@dataclass class Track: id: int; detections: list[Detection]; closed: bool = False` with properties `first_time`, `last_time`, `last_sample`, `mean_area`; `iou(a: Box, b: Box) -> float`; `grey_thumb(image: Image.Image) -> np.ndarray` (32×18 float32); `is_cut(prev: np.ndarray | None, cur: np.ndarray) -> bool` (mean |Δ| > 30); `assign_tracks(tracks: list[Track], detections: Sequence[Detection], *, sample: int, cut: bool) -> list[Track]` (mutates `tracks`, returns the track per detection); `frontality(landmarks) -> float`; `sharpness(grey: np.ndarray) -> float`; `quality(score: float, short_side: int, frontal: float, sharp: float) -> float`; `merge_spans(spans: Sequence[tuple[float, float]], *, gap: float) -> list[tuple[float, float]]`.

Formulas the tests pin:

```python
frontality = 0.0 if eye_dist == 0 else max(0.0, 1.0 - abs(nose_x - (eye0_x + eye1_x) / 2) / (0.5 * eye_dist))
sharpness = min(1.0, laplacian_variance(grey) / 100.0)   # 4-neighbour Laplacian via numpy slicing
quality = score * min(1.0, short_side / 112) * (0.5 + 0.5 * frontal) * (0.5 + 0.5 * sharp)
```

`assign_tracks`: on `cut`, close every open track first; candidates are open tracks with `last_sample >= sample - 2`, `iou >= 0.5` and an area ratio ≤ 1.5² (side ratio 1.5); greedy by descending IoU; a track takes at most one detection per sample; the rest open new tracks with the next id.

- [ ] **Step 1: Write the failing tests** — `test_iou_identical_and_disjoint`, `test_steady_face_forms_one_track`, `test_gap_of_one_sample_keeps_track_two_breaks_it`, `test_cut_breaks_every_track`, `test_size_jump_starts_new_track`, `test_two_faces_two_tracks_no_swap`, `test_frontality_frontal_vs_profile`, `test_sharpness_flat_vs_checkerboard`, `test_quality_orders_larger_sharper_frontal_first`, `test_merge_spans_joins_within_gap`, `test_is_cut_on_scene_change_not_on_noise`.
- [ ] **Step 2: Run** `uv run pytest tests/test_face_tracks.py -q` → FAIL.
- [ ] **Step 3: Implement** `nextext/core/face_tracks.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(core): face tracks, shot cuts and crop quality"`.

---

### Task 5: Face extraction

**Files:**
- Create: `nextext/core/face_extraction.py`
- Modify: `nextext/pipeline.py` (add `faces_pipeline`)
- Test: `tests/test_face_extraction.py`

**Interfaces:**
- Consumes: `sample_frames` (Task 2), `detect_faces`/`group_faces`/`GroupTrack`/`GroupFace` (Task 3), Task 4's functions.
- Produces: `@dataclass(frozen=True) class FaceCrop: time_sec: float; box: Box; score: float; quality: float; jpeg: bytes`; `@dataclass(frozen=True) class FaceEntry: number: int; assigned: bool; appearances: tuple[tuple[float, float], ...]; detections: int; best: FaceCrop; crops: tuple[FaceCrop, ...]; thumbnail: bytes`; `@dataclass(frozen=True) class DetectionRow: time_sec: float; entry: int; box: Box; score: float`; `FacesOutcome = Literal["ok", "partial", "unavailable"]`; `@dataclass(frozen=True) class FacesResult: outcome: FacesOutcome; grouped: bool; truncated: bool; entries: tuple[FaceEntry, ...]; rows: tuple[DetectionRow, ...]; frames_sent: int; tracks_sent: int`; `extract_faces(file_path: Path, *, mode: Literal["detect", "group"], interval_s: float, max_frames: int) -> FacesResult | None` (`None` when no frame was sampled); `pipeline.faces_pipeline(file_path: Path, *, mode: Literal["detect", "group"]) -> FacesResult | None` (reads `load_faces_sampling_env()`).

Flow: batch samples (8 or the 100 MiB source budget) → send each as JPEG q85 scaled to ≤ 1280 px → map normalized boxes to source pixels → filter (score order × area, top 20, ≥ 40 px) → cut check → `assign_tracks` → cut crops from the source frame while it is in memory, keeping per track the best crop + thumbnail, the top-3 crops and the top-3 good group crops → after the last batch: cap tracks (500 by duration × `mean_area`) → `group` mode: `group_faces` on tracks with ≥ 1 good face (labels `-1` and unsent tracks stay single) → entries (a label's tracks merged; else one track each) → cap entries (100 by total appearance time × area) → number by first appearance (ties by x) → appearances = `merge_spans([(t.first_time, t.last_time + interval)], gap=2 * interval)` → rows for kept entries. `grouped` is true only in `group` mode with a reply. `outcome`: every batch answered → `ok`; some batches failed or images undecodable → `partial`; none answered → `unavailable` with no entries.

- [ ] **Step 1: Write the failing tests** (monkeypatch `face_extraction.sample_frames` with synthetic `SampledFrame`s and `detect_faces`/`group_faces` with fakes):

```python
def test_one_person_two_shots_groups_to_person_1() -> None:       # group mode, labels [0, 0] → one entry, number 1, assigned True, two appearances
def test_detect_mode_never_calls_group(monkeypatch) -> None:      # grouped False, entries assigned False
def test_group_failure_falls_back_to_tracks() -> None:            # group_faces → None: grouped False, one entry per track
def test_cooccurring_tracks_send_their_samples() -> None:          # GroupTrack.samples carry the sample indices
def test_numbering_follows_first_appearance() -> None:
def test_best_crop_is_highest_quality_with_margin_and_bounded() -> None:  # decoded best.jpg long side ≤ 1024, larger than the box
def test_edge_face_crop_is_clamped() -> None:                      # box at x=0: crop box within the frame
def test_thumbnail_and_crops_are_bounded_jpegs_without_exif() -> None:
def test_crowd_is_capped_and_truncated() -> None:                  # 30 faces/frame over many cuts → ≤ 20 per frame kept, ≤ 500 tracks, ≤ 100 entries, truncated True
def test_partial_outage_is_partial() -> None:
def test_total_outage_is_unavailable() -> None:
def test_no_frames_is_none() -> None:
def test_group_body_over_budget_is_ungrouped() -> None:
```

- [ ] **Step 2: Run** `uv run pytest tests/test_face_extraction.py -q` → FAIL.
- [ ] **Step 3: Implement** `nextext/core/face_extraction.py` and `pipeline.faces_pipeline`.
- [ ] **Step 4: Run** → PASS; then `uv run pytest -q` → all pass.
- [ ] **Step 5: Commit** — `git commit -m "feat(pipeline): extract, track and group the faces of a video"`.

---

### Task 6: Audit specs

**Files:**
- Modify: `nextext/audit.py` (new `FACE_STAGE` after `CAPTION_STAGE`, `__all__`; `joined_analyses` gains `faces: bool`, listed after `frame_captions`)
- Modify: `nextext/api/jobs.py` `_joined_analyses` (pass `faces=`) and `nextext/cli.py` `_record_combination` (pass `faces=False`)
- Test: `tests/test_audit.py` or the existing `joined_analyses` tests, `tests/test_api/test_audit_pipeline.py`

**Interfaces:**
- Produces: `FACE_STAGE: Final[Mapping[str, AiSpec]] = {"face_detect": AiSpec("ai.detect_faces", "inference:faces", (MEDIA,)), "face_group": AiSpec("ai.group_faces", "inference:faces", (MEDIA, "ai.detect_faces"))}`; `joined_analyses(..., faces: bool)`.

- [ ] **Step 1: Write the failing tests** — `test_face_stage_records_frames_and_tracks_only` (a `JobAudit` stage with two `record_inference` calls and `notes.result("face_detect", frames=12)`, `notes.result("face_group", tracks=3)` → two `ai_call` events, operations, recipients, `ai.sources` = the upload / upload + detect event; `assert_no_content`); `test_joined_analyses_lists_faces`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/test_api/test_audit_pipeline.py tests -k "audit or joined" -q` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(audit): record the face stage by what it judged"`.

---

### Task 7: Option, switch, stage and result

**Files:**
- Modify: `nextext/api/schemas.py` (`JobOptions.faces: bool = False`; `FaceAppearanceOut`, `FaceEntryOut`, `FacesOut`; `JobResult.faces: FacesOut | None = None`; `ConfigResponse.faces: Literal["off", "detect", "group"] = "off"`)
- Modify: `nextext/api/jobs.py` (`PIPELINE_STAGE_LABELS` gains `"Detecting faces"` after `"Describing keyframes"` and `STAGE_FACES`; `JobState.faces_mode: FacesMode = "off"`; `JobManager.__init__(…, faces_mode: FacesMode = "off")` stored as `self.faces_mode` and copied onto each `JobState`; `file_opts["faces"]`; the stage block; `"faces"` in both result dicts; `_serialize_result`; `_requested_analyses` adds `("faces", options.faces)`)
- Modify: `nextext/api/main.py` (lifespan: `faces_mode = load_faces_mode()`, `app.state.faces_mode`, `"faces": faces_mode` in `emit_config_event`, `JobManager(..., faces_mode=faces_mode)`)
- Modify: `nextext/api/routes/jobs.py` (`create_job`: after `_parse_options`, `if parsed_options.faces and manager.faces_mode == "off": raise HTTPException(422, "Invalid request.")`), `nextext/api/routes/health.py` (`get_config` reads `request.app.state.faces_mode`)
- Test: `tests/test_api/test_pipeline_faces.py` (new), `tests/test_api/test_config.py` or the existing `/config` tests, `tests/test_api/test_job_options_keyframes.py` (extra-forbid still holds)

**Interfaces:**
- Consumes: `faces_pipeline`, `FacesResult` (Task 5), `FACE_STAGE` (Task 6).
- Produces: `FaceAppearanceOut(start_seconds: float, end_seconds: float)`; `FaceEntryOut(number: int, assigned: bool, first_seen_seconds: float, last_seen_seconds: float, appearances: list[FaceAppearanceOut], detections: int, thumbnail: str, best_artifact: str)` with `thumbnail = "data:image/jpeg;base64,…"` and `best_artifact = f"face_{number:03d}.jpg"`; `FacesOut(outcome: Literal["ok", "partial", "unavailable"], grouped: bool, truncated: bool, entries: list[FaceEntryOut])`; `state.result["faces"]: FacesResult | None`.

Stage block (between the keyframe block and `visual_context = …`):

```python
_notify(STAGE_FACES)
faces_result: FacesResult | None = None
if file_opts["faces"] and state.faces_mode != "off":
    with audit.stage(FACE_STAGE) as notes:
        try:
            faces_result = faces_pipeline(state.file_path, mode=state.faces_mode)
        except Exception as exc:  # fail-soft: faces never fail the job
            logger.warning("Job {} faces unavailable: {}", state.job_id, error_label(exc))
        if faces_result is not None:
            notes.result("face_detect", frames=faces_result.frames_sent)
            if faces_result.tracks_sent:
                notes.result("face_group", tracks=faces_result.tracks_sent)
    _complete(STAGE_FACES, {"face_frames": faces_result.frames_sent, "faces": len(faces_result.entries)} if faces_result else {"skipped": True})
else:
    _complete(STAGE_FACES, {"skipped": True})
```

- [ ] **Step 1: Write the failing tests**

```python
def test_faces_stage_runs_at_index_2_and_lands_in_result(...) -> None:   # stub faces_pipeline; snapshot.result.faces.entries[0].thumbnail startswith "data:image/jpeg;base64,"
def test_silent_video_keeps_its_faces(...) -> None:                      # transcription empty → skipped job still has result.faces
def test_faces_failure_does_not_fail_the_job(...) -> None:               # faces_pipeline raises → job completed, faces None
def test_sse_delta_carries_counts_only(...) -> None:                     # no "thumbnail"/"jpeg" anywhere in the event history bytes
def test_audio_only_has_no_faces_tab(...) -> None:                       # faces_pipeline → None: result.faces is None, delta {"skipped": True}
def test_faces_requested_while_off_is_422_and_not_collected(client, audit_capture) -> None:
def test_config_reports_faces_mode(client) -> None:                      # NEXTEXT_FACES=group → GET /config → "faces": "group"
def test_config_event_records_faces_mode(...) -> None:
def test_list_items_carry_no_thumbnails(client) -> None:
```

- [ ] **Step 2: Run** `uv run pytest tests/test_api/test_pipeline_faces.py -q` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q` → all pass (Task 1's sequence test now sees seven labels).
- [ ] **Step 5: Commit** — `git commit -m "feat(api): faces option, operator switch and the face stage"`.

---

### Task 8: Face artifacts

**Files:**
- Modify: `nextext/api/artifacts.py` (`faces.zip`, `faces.csv`, `face_NNN.jpg` in `render_artifact`; `SUPPORTED_ARTIFACTS` adds `faces.zip`, `faces.csv`; `FACE_IMAGE_ARTIFACT = re.compile(r"face_\d{3}\.jpg")`; `_render_archive_members` adds `faces/…` and `{stem}_faces.csv`; `_APP_JPEG = "image/jpeg"`)
- Modify: `nextext/api/routes/jobs.py` (`download_artifact`: accept `FACE_IMAGE_ARTIFACT.fullmatch(name)` as well)
- Test: `tests/test_api/test_faces_artifact.py` (new)

**Interfaces:**
- Consumes: `state.result["faces"]: FacesResult | None`.
- Produces: `faces_members(result: FacesResult) -> list[tuple[str, bytes]]` — `person_NNN/` (assigned) or `face_NNN/` folders with `best.jpg`, `crop_1.jpg`…`crop_3.jpg`, plus `manifest.json` = `{"outcome", "grouped", "truncated", "entries": [{"number", "assigned", "folder", "appearances": [[s, e]], "files": [{"file", "time_sec", "box": [x, y, w, h], "score", "quality"}]}]}`; `faces_csv(result) -> bytes` with header `time_sec,entry,x,y,w,h,score`.

- [ ] **Step 1: Write the failing tests** — `test_faces_zip_layout_and_manifest`, `test_faces_csv_one_row_per_detection`, `test_face_best_image_by_number` (`face_001.jpg` → the entry's best JPEG; `face_999.jpg` → 404; `face_1.jpg` → 404 unsupported), `test_archive_holds_faces_folder_and_csv`, `test_batch_archive_holds_faces`, `test_docint_jsonl_has_no_faces`, `test_face_download_is_disclosed_as_jpg` (audit recipient `download:jpg`, criteria `artifact: face_001.jpg`), `test_no_faces_result_404s_face_artifacts`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/test_api -q` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(artifacts): faces.zip, faces.csv and each person's best image"`.

---

### Task 9: Frontend

**Files:**
- Modify: `frontend/src/api/types.ts` (`JobOptions.faces`, `FaceAppearance`, `FaceEntry`, `FacesResult`, `JobResult.faces`, `AppConfig.faces`)
- Modify: `frontend/src/components/upload/UploadForm.tsx` (+ test): `useAppConfig()`; `faces` state; toggle after keyframes, shown only when `config.faces !== 'off'`; payload includes `faces`
- Modify: `frontend/src/components/results/TimeCell.tsx`: extract `TimeButton({label, seconds, jobId, fileName, mediaUrl})` (button, or plain `<span>` when not playable); `TimeCell` wraps it in the `<td>`
- Create: `frontend/src/components/results/FacesTab.tsx` (+ `FacesTab.test.tsx`)
- Modify: `frontend/src/components/results/ResultPanel.tsx` (+ test): `TabId` `'faces'`, tab after `visual_context` when `result.faces` is not null
- Modify: `frontend/src/i18n/en.ts`, `de.ts`; the six `JobResult` fixtures (`faces: null`)

**Interfaces:**
- Consumes: `FacesOut` JSON from Task 7; artifacts from Task 8 via `downloadArtifact(jobId, name, fileName)` / `DownloadButtons`.
- i18n keys (en / de): `options.faces` "Faces" / "Gesichter"; `results.tab_faces` "Faces" / "Gesichter"; `results.face_person` "Person {n}" / "Person {n}"; `results.face_single` "Face {n}" / "Gesicht {n}"; `results.face_sightings_one` "{count} sighting" / "{count} Sichtung"; `results.face_sightings_other` "{count} sightings" / "{count} Sichtungen"; `results.face_seen` "First seen {first} · last seen {last}" / "Zuerst {first} · zuletzt {last}"; `results.faces_none` "No faces found." / "Keine Gesichter gefunden."; `results.faces_partial` "Face detection was not reachable for part of the video." / "Die Gesichtserkennung war für einen Teil des Videos nicht erreichbar."; `results.faces_unavailable` "Face detection was not reachable." / "Die Gesichtserkennung war nicht erreichbar."; `results.faces_truncated` "Only the most prominent faces are shown." / "Es werden nur die auffälligsten Gesichter gezeigt."; `results.faces_not_grouped` "Faces were not grouped by person." / "Gesichter wurden nicht nach Personen gruppiert."; `artifacts.download_faces` "Download all faces (ZIP)" / "Alle Gesichter herunterladen (ZIP)"; `artifacts.download_faces_csv` "Download face table (CSV)" / "Gesichtertabelle herunterladen (CSV)"; `artifacts.download_face_best` "Download best image" / "Bestes Bild herunterladen".

FacesTab: banners (`Banner` from `@infra/ui`) for `partial`/`unavailable`, `truncated`, and "not grouped" in group-capable results; a responsive grid of cards: `<img src={entry.thumbnail} alt={label}>` (data URI), the label (`face_person` when `assigned`, else `face_single`), sightings, first/last via `formatStamp`, appearances as `TimeButton`s (`seconds = start_seconds`), and a "best image" `DownloadButtons` item `{name: entry.best_artifact, fileName: `${stem}_${entry.best_artifact}`}`; the card whose appearance contains the player's `currentTime` (this job playing) gets `aria-current="true"` and a ring. Tab-level downloads `faces.zip` and `faces.csv`. Icons keep `h-4 w-4` when given a className.

- [ ] **Step 1: Write the failing tests** — FacesTab: `renders one card per entry with thumbnail and localized label`, `appearance button seeks the player`, `highlights the entry on screen`, `shows the partial, truncated and not-grouped banners`, `shows faces_none for an empty ok result`; UploadForm: `hides the faces toggle when the switch is off`, `shows it after keyframes and sends faces`; ResultPanel: `adds the faces tab after visual context`, `keeps faces under the skipped banner`.
- [ ] **Step 2: Run** `cd frontend && pnpm test` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cd frontend && pnpm test && pnpm lint && pnpm typecheck && pnpm build` → all pass.
- [ ] **Step 5: Commit** — `git commit -m "feat(frontend): faces tab with appearances, best images and the upload toggle"`.

---

### Task 10: Documentation and compose

**Files:** `CLAUDE.md` (pipeline step "Faces", key modules `faces.py`/`face_tracks.py`/`face_extraction.py`, the env vars, API artifacts and `GET /config`, "five inference clients" → six), `docs/architecture.md`, `docs/configuration.md` (a Faces section; the dedicated-endpoints table and "three non-OpenAI services" wording), `docs/audit.md` (the `ai_call` table rows, `download:jpg`, `faces` in `combine` and `config.audit`, `face_NNN.jpg` in the route table), `docs/cli.md` (`-F` never includes faces; no faces in the CLI yet), `.env.example` (`NEXTEXT_FACES`, `FACES_API_BASE`, `FACES_TIMEOUT`, `FACES_INTERVAL_S`, `FACES_MAX_FRAMES`), `docker/compose.yaml` (the five variables as `${X:-}` in the backend environment, alphabetical), the `_central_endpoint_root` docstring.

- [ ] **Step 1: Write the docs.**
- [ ] **Step 2: Check** `uv run pre-commit run --all-files` and `docker compose --env-file /dev/null -f docker/compose.yaml config -q` → pass.
- [ ] **Step 3: Commit** — `git commit -m "docs: faces in videos — stage, switch, artifacts, audit"`.

---

### Task 11: Release

- [ ] **Step 1:** `pyproject.toml` `version = "1.18.0"`, `uv lock` (only the version line changes).
- [ ] **Step 2:** `make verify && uv run pytest -q && (cd frontend && pnpm test)` → green.
- [ ] **Step 3: Commit** — `git commit -m "chore: release v1.18.0"`.

---

## Addendum: weak tracks (after the first end-to-end run)

A first run on a two-person clip left two half-profile tracks of an already
grouped person as faces of their own: Task 6 sent only tracks with at least one
good face, so a track without one was never compared, although its features
matched its person well above the threshold and the other person far below it.

The fix keeps the good-face rule but turns it into a role instead of an entry
ticket. Each track now goes to `/faces/group`: one with a good face as an
anchor, with only its good crops; one without as a weak track
(`GroupTrack.anchor=False`), with its best crops by quality. The service
clusters the anchors into people and lets each weak track join the one person
it matches and was not on screen with, or nobody; a weak track never founds or
links a person (vllm-service `faces_cluster.py`). Tests:
`test_tracks_without_a_good_face_are_sent_as_weak`,
`test_profile_track_joins_the_person_the_service_names`,
`test_mixed_track_sends_only_its_good_crops`, `test_group_marks_weak_tracks`.

## Addendum: faces seen once, and faces too small to compare

A second end-to-end run, on hand-held video with hard cuts, split people
again: every track there is a single sample (the cut detector fires on almost
every sample, and boxes a second apart barely overlap), so a small or blurred
face that passes the good-face bar founds a person of its own, and profile
tracks fell short of joining at the group threshold. Denser sampling did not
link them. The fix is in the faces service (vllm-service: a cluster seen in a
single sample joins like a weak track; joining compares a piece with the
person's three closest anchors and has its own, looser threshold,
`FACE_JOIN_THRESHOLD`). On this side, `_group` no longer sends a weak track
none of whose faces reaches `GOOD_MIN_PX` (64 px): the feature model works on
112 px faces, and below that the features are too noisy to match anyone.
Tests: `test_tracks_too_small_to_compare_are_not_sent`,
`test_tracks_without_a_good_face_are_sent_as_weak`.

