# Faces in videos — design

**Date:** 2026-11-07
**Status:** Approved
**Depends on:** vllm-service `faces` (`/faces/detect`, `/faces/group`; its spec
`docs/superpowers/specs/2026-11-07-faces-service-design.md` in that repo)

## Problem

Nextext should find the faces in an uploaded video and hand them out. Two uses,
both decided: an overview of who is on screen when, and picture material to pass
on — the best single shot of each person as a file, for example for a match in
another system. Faces of the same person within one video are grouped
("Person 1, 2 …", numbered by first appearance like the speaker labels).

## Decisions

1. **The models live in vllm-service.** Nextext stays model-free; a new HTTP
   client talks to `/faces/detect` and `/faces/group` like `vad.py` talks to
   `/vad`. `tests/test_no_torch.py` additionally forbids `cv2`, `onnxruntime`,
   `insightface`, `mediapipe`, `dlib` and `face_recognition`.
2. **Biometric features never reach Nextext.** `/faces/group` takes face crops
   and returns labels only.
3. **An operator switch `NEXTEXT_FACES=off|detect|group`, default `off`,** read
   once at startup and recorded in `config.audit`. `detect` gives crops without
   any biometric processing; `group` adds grouping, which is biometric
   processing: the controller extends the data-protection impact assessment
   before switching it on (§ 67 BDSG, with the DPO involved); the software
   enforces nothing. Plus a per-job option `faces`, default off. `faces=true`
   while the switch is `off` is refused with `422` before anything is stored.
4. **No CLI flag in this release.** What `make cli` writes stays on disk up to
   `NEXTEXT_CLI_RETENTION_DAYS`; face pictures there need their own decision.
   `-F` never includes faces.
5. **Faces go into the regular `archive.zip`** (folder `faces/`, beside
   `keyframes/`) and so into the batch ZIP; never into `docint.jsonl`. Every
   download is its own `disclose` event.
6. **Thumbnails are `data:` URIs in the finished snapshot** (≤ 128 px), so
   looking at them is the one `query`/`job.get` event, not N disclosures. The CSP
   already allows `img-src data:`. Never in SSE frames or list items.
7. **Out of scope:** attributes (age, sex, emotion, origin), matching across
   videos or against lists, anything kept beyond the job, blurring, the
   rotation fix for keyframes (faces only), audio-only files (the stage is
   skipped and no tab appears).

## Pipeline

A new stage, `"Detecting faces"`, sits right after the keyframes and before the
no-speech short-circuit, so a silent video still yields faces. It never calls
`_ensure_inference()`; it is fail-soft inside its `audit.stage`.

1. **Sample** frames by time: every `FACES_INTERVAL_S` (1.0 s), at most
   `FACES_MAX_FRAMES` (600, ceiling 3600) — a longer clip stretches the interval
   so the samples span it. A demux pass measures keyframe spacing: when the
   median gap is at most twice the interval, only keyframes are decoded,
   otherwise the reference frames. Times fall back to index ÷ average rate and
   are offset by the stream's start time, so they match the player's
   `currentTime`. The display rotation is applied, so portrait phone videos are
   upright. Source frames are kept at their own resolution only per batch
   (≈ 100 MB).
2. **Detect** in batches of 8 images, each ≤ 1280 px.
3. **Filter:** at most 20 faces per frame by score × area; at least 40 px.
4. **Track:** link detections of consecutive samples (IoU ≥ 0.5, size within
   ×1.5, a gap of at most one sample); break every track at a shot cut,
   detected from the grey-level difference of tiny thumbnails.
5. **Score quality** per crop: detection score × size × frontality (from the
   landmarks) × sharpness (Laplacian variance).
6. **Group** (`group` mode): up to 3 faces per track, ≤ 160 px JPEG, with each
   track's sample indices so the service keeps co-occurring tracks apart. A
   track with a good face (score ≥ 0.85, ≥ 64 px, roughly frontal) is an
   anchor and sends only its good faces; a track without one is weak and sends
   its best faces with `"anchor": false`, provided one of its faces is at
   least 64 px. Anchors make up the people; a weak track, and an anchor seen
   in a single sample, may join the one person it matches but never founds or
   links one (added after end-to-end runs left half-profile tracks of a
   grouped person as faces of their own, and, on hand-held, hard-cut video,
   small or blurred single-sample faces as people of their own).
7. **Number** entries by first appearance; merge each entry's spans into
   appearances.

Kept per entry: the best shot from the source with a generous margin (≈ 50 % of
the box per side, ≤ 1024 px) so another system can detect on it, 3 more crops
(≤ 256 px) and the thumbnail. Caps: 500 tracks, 100 entries, ranked by
duration × size; over either, `truncated` is set. A weak track that matches no
person (or no person clearly), a weak track whose faces all stay under 64 px, a
track without a decodable face, or every track
when grouping is off, failed or over its byte budget, stays a single entry
(`assigned=false`, shown as "Face N").

## Result

`JobResult.faces: FacesOut | None` — `outcome` (`ok` | `partial` | `unavailable`),
`grouped`, `truncated`, `entries[]` with `number`, `assigned`,
`first_seen_seconds`, `last_seen_seconds`, `appearances[]`
(`start_seconds`/`end_seconds`), `detections`, `thumbnail` (data URI) and
`best_artifact` (`face_NNN.jpg`).

Artifacts: `faces.zip` (`person_NNN/` or `face_NNN/` folders with `best.jpg` and
`crop_N.jpg`, plus `manifest.json` with file, entry, time, box in source pixels,
score and quality), `faces.csv` (one row per detection, no images),
`face_NNN.jpg` (one entry's best shot, recipient `download:jpg`), and `faces/`
plus `{stem}_faces.csv` inside `archive.zip`. No EXIF, no source file name.

## Audit

`FACE_STAGE`: `face_detect` → `ai.detect_faces` to `inference:faces`, sources the
upload; `face_group` → `ai.group_faces` to `inference:faces`, sources the upload
and `ai.detect_faces`. Recorded by what was judged — `frames`, `tracks` as the
size of the transfer — never by what was found. `combine` lists `faces` whenever
the stage ran.

## Configuration

`NEXTEXT_FACES` (`off`), `FACES_API_BASE` (URL, an off token, or the central
endpoint with one `/v1` stripped), `FACES_TIMEOUT` (300 s), `FACES_INTERVAL_S`
(1.0), `FACES_MAX_FRAMES` (600, ceiling 3600). `GET /config` reports `faces`.
