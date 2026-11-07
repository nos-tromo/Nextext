# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Data confidentiality — hard rule

**NEVER expose actual production or testing data in any file committed or
pushed to git.** This covers not only file contents but also metadata that
references real data: filenames, file descriptions, social-media account
names or handles, user identifications, sample records, log excerpts, and
screenshots. It applies everywhere git sees — source code, tests, fixtures,
docs, examples, configs, commit messages, and CI files. Use fully synthetic,
invented placeholders instead.

**Likewise, NEVER expose local filepaths from development machines** —
absolute paths or home directories such as `/Users/<name>/...`,
`/home/<name>/...`, or `C:\Users\...` — anywhere git sees. The only
permitted paths are relative project paths starting from the project's
root (e.g. `docker/compose.yaml`).

## Planning

For any non-trivial change (>1 file or any root-cause fix), present a plan and wait for approval BEFORE editing code. Do not start with Edit/Write on a fresh investigation.

## Project Overview

Nextext is a modular audio analysis toolkit that transcribes, translates, and analyzes natural language from audio/video files. All model inference runs on external endpoints: Whisper transcription via an OpenAI-compatible audio API, voice-activity detection (`/vad`), speaker diarization (`/diarize`), GLiNER NER (`/gliner`), and face detection and grouping (`/faces/detect`, `/faces/group`) via dedicated out-of-process HTTP services, and LLMs (Ollama, vLLM, or OpenAI-compatible endpoints) for translation, summarization, and hate-speech detection. Only spaCy word-level NLP runs in-process; every upload is re-encoded to 16 kHz mono FLAC via the PyAV wheel (bundled ffmpeg) before transcription. The backend ships no model weights and needs no GPU — PyAV is the only local media dependency, and no apt audio tooling is installed.

## Project Context

- Local ML runtimes (openai-whisper, pyannote, GLiNER, Silero VAD, torch) have been removed; every model call is an HTTP request to an external endpoint. `tests/test_no_torch.py` pins the no-torch invariant (and the absence of in-process face/CV runtimes: `cv2`, `onnxruntime`, `insightface`, `mediapipe`, `dlib`, `face_recognition`) — camel-tools' torch/transformers requirements are excluded via `[tool.uv] override-dependencies`.
- Docker base images are digest-pinned on the `FROM` lines and Dependabot-bumped, with one deliberate exception: the backend builds and runs on the `python:3.12-slim-trixie` line (uv is copied in from `ghcr.io/astral-sh/uv` for the build stage only), and that interpreter line is fixed by `requires-python`, `uv.lock` and the CI matrix — the builder runs `uv` with `UV_PYTHON_DOWNLOADS=0`, so any other `python:*` tag fails `uv sync` by construction. `.github/dependabot.yml` therefore ignores minor/major bumps of `library/python` and only its digest refreshes; moving to another Python is a migration (lockfile, wheels, CI matrix, Dockerfile, docs), never an automated bump.
- The frontend builds on the current `node:*-alpine` and is served by `nginx-unprivileged:*-alpine`; both move freely with Dependabot. Node ≥25 images ship no Corepack, so the builder installs pnpm with `npm install -g pnpm@<pin>` — keep that pin in step with `packageManager` in `frontend/package.json` and `pnpm-version` in `.github/workflows/ci.yml`.

## Commands

```bash
# Install dependencies
uv sync                    # production deps
uv sync --group dev        # include dev deps (pytest, ruff, pyrefly, pre-commit)

# Run the app
uv run nextext-api         # FastAPI backend on port 8000
uv run nextext-cli -f <file> [args]  # CLI mode (in-process, no backend required)

# Frontend (React SPA) — run via Docker or the Vite dev server
make build && make up-dev  # Docker: build images, then run detached; publishes on NEXTEXT_HOST_PORT (or: make dev)
# cd frontend && pnpm dev  # local Vite dev server (proxies /api/v1 to localhost:8000)

# Preload spaCy language resources (the only local downloads)
make preload

# Run nextext-cli inside the running backend (its audit chain lands in the backend's log;
# the compose default AUDIT_REQUIRE_PURPOSE=query,disclose makes CASE_REF or REASON required)
make cli ARGS='-f /tmp/nextext-cli/<file> -sum' CASE_REF=… REASON=…
make cli-clean [RUN=output/<stem>] [LEGACY=1]   # delete what CLI runs left now, recorded (else after NEXTEXT_CLI_RETENTION_DAYS)

# Tests
uv run pytest              # run full test suite
uv run pytest tests/test_pipeline.py  # single test file
uv run pytest -k "test_name"          # single test by name

# Linting & formatting (also enforced by pre-commit hooks)
ruff check --fix           # lint with auto-fix
ruff format                # format code
uv run pyrefly check
```

## Testing

- Always run the full test suite (`pytest`) after making changes and report pass/fail counts.
- When tests fail, fix the root cause rather than patching tests to match stale/removed code.
- Verify with `pre-commit run --all-files` (pyrefly, lint, docstrings) before declaring work complete.

Tests are in `tests/` using pytest with monkeypatch fixtures and `respx` for mocking the HTTP inference clients (Whisper, NER, diarization). Tests simulate Docker detection and environment configuration. No GPU, no network, and no model downloads required for tests.

Hate-speech accuracy is measured by the dev-only harness `eval/hate_speech/` (not shipped; excluded from the image): `run.py` compares a frozen copy of the pre-rewrite per-row classifier with the windowed pipeline on invented, stance-labelled transcripts against the live `TEXT_MODEL`, reporting precision/recall and the use/mention false-positive rate. Its offline unit tests run with the suite.

Several modules call `load_dotenv()` at import time (`nextext/utils/env_cfg.py`, `nextext/core/openai_cfg.py`, `nextext/core/words.py`, `nextext/utils/model_loader.py`), which copies a local, uncommitted `.env`'s values straight into the real process environment the first time one of those modules is imported — not into a test-scoped sandbox, and not reverted by `monkeypatch`. A developer `.env` with e.g. `RESPONSE_LANGUAGE=de` (a normal local-dev setup, see `.env.example`) then silently outranks any test asserting "unset" default behavior, for the rest of that pytest session. `tests/conftest.py` carries an autouse fixture that clears `RESPONSE_LANGUAGE`/`NEXTEXT_RESPONSE_LANGUAGE` before every test so the suite is hermetic against ambient `.env` state; add the same pattern there for any other env var a test needs to assume is unset by default.

Before claiming a check is green, run the actual command and paste the output. `git ls-files` does not cover untracked files — use `pre-commit run --all-files`. After opening a PR, confirm CI actually triggered on the latest push before declaring done.

## Docstrings & Style

- All new/modified Python functions must have Google-style docstrings.
- Python 3.12 is the target; prefer explicit types and distinct variable names across branches to satisfy pyrefly.

## Logging

Operational logs carry no upstream error bodies or transcript text, and the access logs no query strings. The backend's operational logs go to **stderr only**: its stdout carries the data-protection audit log alone (`docs/audit.md`), whose lines must never interleave with anything else, so `setup_logging()` also moves every stdlib handler writing to stdout — uvicorn's default config puts its access log there — to stderr. Never `print()` in the backend. An error message can quote the request it rejected, so log an upstream or model failure with `error_label(exc)` (exception class plus HTTP status) and an unexpected one with `exception_frames(exc)` (traceback frames, no message) from `nextext/utils/log_cfg.py` — never the response body, `str(exc)`, or `logger.exception`. `setup_logging()` also strips the query string (which can carry the media token or a playback grant) from uvicorn's access log and the exception message from uvicorn's error log; the frontend nginx writes its access log in the `scrubbed` format (`frontend/nginx/default.conf`), which records no query string, referer or client address. nginx's error log has a fixed format: an error logged for a request (e.g. the backend unreachable or timing out) still carries the full request line, query string included, with the client address and referer.

## Architecture

**Agent-based design:** Each feature is a stateless agent (module) with narrow input/output. The FastAPI backend orchestrates them; the React frontend is a static SPA served by nginx that never imports the pipeline directly.

**Service split (Docker):**

- **Backend** (`backend`) — FastAPI app exposing `/api/v1`. Owns the pipeline and the HTTP inference clients (Whisper, NER, diarization). CPU-only, built from `docker/Dockerfile.backend`.
- **Frontend** (`frontend`) — React SPA (node → nginx multi-stage build) in `frontend/`. Serves the compiled bundle and proxies `/api/v1` same-origin to the backend — no `BACKEND_HOST` env var needed. Built from `docker/Dockerfile.frontend`; `cd frontend && pnpm {dev,build,test,lint,typecheck}` for local development.

`nextext-cli` keeps the in-process path: it imports `nextext.pipeline` directly and runs end-to-end without needing a backend container. Lives in the backend image alongside the API.

**Playback.** The SPA has a slide-in media player (`frontend/src/components/player/MediaPanel.tsx`, mounted once in the Shell, driven by `lib/mediaPlayerStore.ts`). Clicking a timestamp in the Transcript or Visual context tab, or a face's appearance in the Faces tab (`components/results/FacesTab.tsx`, whose card is marked `aria-current` while the face is on screen), opens it and seeks there; the row under the playhead is highlighted and scrolled into view as playback advances (`hooks/useFollowActiveRow.ts`), so the highlight cannot drift off the fold. It scrolls only when the row has actually left the viewport, and centres it so the surrounding lines come along. Following pauses while the reader scrolls the page by hand — but not for a wheel over the player panel, nor for Space/arrows aimed at a control — and resumes on either signal: a timestamp click (the store's `followSeq` counter) or the playhead's row coming back into view. That second signal is load-bearing: a one-way pause left following dead for the rest of the session after a single trackpad flick. It is non-modal by design — no backdrop, no `aria-modal` — and the canvas reserves room for it on wide screens, so the page stays usable. `TranscriptSegment` carries `start_seconds`/`end_seconds` (numeric twins of the `str(timedelta)` display strings) so the client seeks without re-implementing the parser. A media element can send neither the identity nor the stated purpose, so the panel asks for a playback grant (`POST /jobs/{id}/media-grant`, through the API client) before it sets `src`, and plays the grant's URL, never the snapshot's token; when the element fails (an expired grant refuses the next range) it asks for a fresh grant once and resumes at the same position — an error within `REGRANT_INTERVAL_MS` (30 s) of the last fresh grant is the file's and shows as unplayable, so an undecodable file cannot loop.

**Purpose bar.** The header carries `@infra/ui`'s `PurposeBar` (scope `nextext`, mounted in `Shell.tsx` via `AppShell`'s `headerActions`, labels from the i18n catalogs). Every request goes through `apiRequest` in `frontend/src/api/client.ts` — `apiGet`/`apiSend`/`apiFetch`, the SSE helper and the blob downloads all call it — which adds the stated purpose as `X-Audit-Purpose` and sends through `withPurpose`, so a `428` opens the bar's dialog and the request is resent once. Never `fetch` the backend directly, never link to it (`<a href>`, `<img src>`, `window.open`): such a request cannot carry the purpose. `required` comes from `GET /config`'s `require_purpose` (`hooks/useAppConfig.ts`, shared with the language provider). The labels object is typed as the full `PurposeBarLabels`, so a label infra-ui adds fails the build until it is translated. A declined prompt shows `common.error_purpose_required`, not the generic error; `hooks/useRetryOnPurpose.ts` refetches the queries a `428` refused once a purpose is stated (failed mutations and downloads stay failed); the player shows `player.purpose_needed` and asks for its grant again once a purpose is stated. The SPA sets no request timeouts, so none runs on while the dialog is open.

**Pipeline flow (server-side):**

1. **Transcription** (always-on) → every upload is re-encoded to 16 kHz mono FLAC (`nextext/core/audio.py`, PyAV) so libsndfile-only Whisper servers can decode it → external Whisper API (`/v1/audio/transcriptions`, always in the source language) behind an external `/vad` speech guard (defaults to the central endpoint; `VAD_API_BASE=off` skips it), + speaker diarization via the out-of-process `/diarize` HTTP service — on by default and auto-detecting the speaker count (no speaker bounds sent), bypassable per job (`diarize=false` / CLI `--no-diarize`); VAD-gating of the turns (cropping to the Silero speech timeline, dropping music/noise the diarizer over-detects as speech) happens server-side in the diarize backend (`DIARIZE_VAD_URL`, on by default in the full vllm-service stack), so turns arrive pre-gated; they are then aligned onto the transcript at the word level (falling back to segment-level overlap when the endpoint returns no word timestamps), with segments overlapping no turn inheriting the temporally nearest turn's speaker instead of rendering as `Unknown`, and finally renumbered to contiguous `Speaker N` by first appearance **in the assembled transcript**, so labels always read in order (Speaker 1, 2, 3 … top to bottom); the speaker column is omitted entirely when ≤1 distinct speaker is detected; low-punctuation transcripts (e.g. Arabic) are re-segmented into one sentence per row via `TEXT_MODEL` before merge/translate (`NEXTEXT_SENTENCE_RESTORE`, default on) → `pd.DataFrame`
2. **Keyframes / visual context** (optional, `keyframes=false` by default) → samples video keyframes across the whole clip (`nextext/core/keyframes.py`, PyAV) and captions each one request at a time via `TEXT_MODEL`'s vision path (`call_vision`), producing `keyframes.zip`, `JobResult.frame_captions` and the `visual_context.txt` block. It is a stage of its own and **independent of summarization in both directions**: a summary no longer samples frames, and frames no longer need a summary. With the option off nothing is sampled and no video is decoded. Runs **before** the no-speech short-circuit, so a silent video still yields its visuals. Captioning has two gates, either of which skips it while sampling continues: the per-job `visual_context` (default on — how a client asks for frames without words) and the operator's `NEXTEXT_VISUAL_SUMMARY` (default on). Needs a vision-capable `TEXT_MODEL`; fail-soft: a text-only model aborts captioning after one request and the frames stay downloadable
3. **Faces** (optional, `faces=false` by default, and only while the operator switch `NEXTEXT_FACES` is `detect` or `group`; `faces=true` while it is `off` → `422` before the upload is stored, no `collect`) → samples the video by time (`sample_frames` in `nextext/core/keyframes.py`: every `FACES_INTERVAL_S`, at most `FACES_MAX_FRAMES`, stretched to span the clip — duration from container, stream or the largest packet pts; keyframe-only decode when keyframes are dense; display rotation applied; times from the stream start; cover art (`attached_pic`) never sampled), sends batches of 8 images ≤ 1280 px to the out-of-process `/faces/detect` service, filters (≥ 40 px, ≤ 20 per frame by score × area), tracks in-process (`nextext/core/face_tracks.py`: IoU ≥ 0.5, area within ×1.5, ≤ 1 missing sample, every track broken at a shot cut), scores crops (score × size × frontality × sharpness) and — in `group` mode — asks `/faces/group` to label the tracks per person (crops in, labels out; no biometric feature reaches Nextext; over the 16 MiB budget it retries with one crop per track, then stays ungrouped). Entries are numbered by first appearance ("Person N" when grouped, else "Face N"); caps 500 tracks / 100 entries set `truncated`, as does a frame with more than 20 faces. Boxes are reported in the source's own pixels. Runs **before** the no-speech short-circuit, never calls `_ensure_inference()`, and is fail-soft: a failed batch → `partial`, a `503` with `Retry-After` ≤ 10 s is retried once, three failed batches in a row stop sampling, none answered → `unavailable`, an unexpected error → no faces (logged with `exception_frames`). Result: `JobResult.faces` (thumbnails as `data:` URIs, only in the finished snapshot), `faces.zip`, `faces.csv`, `face_NNN.jpg`, and `faces/` in `archive.zip`; never in `docint.jsonl`, never in the CLI (`-F` never includes faces).
4. **Translation** (optional) → LLM-based segment translation, directly source → target for any target language, via `InferencePipeline`. Whisper's audio-translate task is not used.
5. **Word-level analysis** (optional) → word counts, named entities via the out-of-process `/gliner` HTTP service, word clouds
6. **Summarization** (optional) → LLM summary via `InferencePipeline`, over whatever the job produced: the transcript (its translation when there is one) and/or the keyframe captions prepended as a "Visual context" block, so the summary covers what was shown as well as what was said. Captions alone are a valid input — a video whose audio held no speech is summarized from its visuals — and only a request with neither text nor captions raises
7. **Hate-speech detection** (optional) → windowed, stance-aware LLM classification: rows are judged in context windows — a labelled core (`HATE_SPEECH_WINDOW_TOKENS`) framed by read-only neighbouring rows (`HATE_SPEECH_CONTEXT_TOKENS`) — one request per window; the model names the targeted group and the speaker's stance for each candidate row and returns row indices, never text; only rows whose speaker **endorses** group-focused enmity become findings (quoting, reporting, condemning or analysing hate is not hate); the original `text` is judged, with the translation shown as an aid; findings stay per row (`start`, `speaker`, `translation`)
8. **Artifacts** → backend renders `.txt`, `.csv`, `.xlsx`, `.png`, `.jsonl`, ZIP on demand at `/api/v1/jobs/{id}/artifacts/{name}`

**HTTP API (`/api/v1`):**

- `POST /jobs` (multipart: `file` + JSON `options`) — queue a new job; returns `{job_id}`. `options` is `JobOptions` (`extra="forbid"`): `src_lang`, `trg_lang`, `task`, `diarize` (default on), `words`, `summarization`, `hate_speech`, `keyframes`, `faces` (all default off; `faces=true` while `NEXTEXT_FACES` is `off` → `422`), plus the `keyframes_per_minute` / `keyframes_max` sampling rates.
- `GET /jobs` — list the caller's in-memory jobs, newest first. The frontend calls this on load to re-discover and resume its jobs after a browser reload.
- `GET /jobs/{id}` — point-in-time snapshot (owner-scoped).
- `GET /jobs/{id}/events` — SSE stream of stage transitions (owner-scoped); replays event history on connect so a reattached client resumes mid-run.
- `POST /jobs/{id}/media-grant` — a playback grant (owner-scoped, needs the purpose a query needs): `{media_url, expires_at}`, the media URL carrying a fresh `secrets.token_urlsafe(32)` bound in memory (`nextext/api/media_grants.py`, on the `JobManager`) to the job and to this request's caller, purpose and request id, usable until `NEXTEXT_MEDIA_GRANT_TTL_S` (900 s) runs out. Issuing it is not recorded; refusals are (`media.grant`, denied).
- `GET /jobs/{id}/media?grant=…` or `?token=…` — the original upload, streamed for in-browser playback with HTTP Range (`206`) so the player can seek. Authorized by a capability in the URL, **not** by the principal: a `<video>`/`<audio>` element cannot attach the trusted header or a purpose. `grant` is what the SPA plays: each range is recorded with the grant request's purpose and request id and `criteria.grant` (the grant's id, never the token); with the edge secret set a grant plays only for its holder (another verified identity: `404` + denied event). `token` is the per-job capability minted at job creation, handed out only on the owner-scoped snapshot (`JobResult.media_url`), kept off `GET /jobs`, and revoked by `DELETE` with the job's grants; it names no purpose, so it gets `428` while `query` is required. Exactly one of the two; every failure is a `404` so a wrong token or grant cannot probe which jobs exist.
- `GET /jobs/{id}/artifacts/{name}` — binary download (transcript.csv/xlsx/txt, translation.txt, summary.txt, visual_context.txt, wordcounts.csv/xlsx, entities.csv/xlsx, wordcloud.png, keyframes.zip, hate_speech.csv/xlsx, docint.jsonl, archive.zip, faces.zip, faces.csv, and `face_NNN.jpg` — one face entry's best shot, recipient `download:jpg`). Owner-scoped. The face artifacts `404` for a result without faces; `faces.zip` holds `person_NNN/` or `face_NNN/` folders (`best.jpg`, `crop_N.jpg`) and a `manifest.json` (file, time, box in source pixels, score, quality); `faces.csv` is one row per detection, no images. `keyframes.zip` also carries a `manifest.json` (`file`/`index`/`time_sec` per frame; `keyframes/manifest.json` inside `archive.zip`) so a consumer that reads frames without captions can still place each one in the clip — omitted rather than guessed when the times do not pair with the frames.
- `DELETE /jobs/{id}` — cleanup (owner-scoped).
- `GET /health`, `GET /languages` — meta endpoints. `GET /config` — the SPA's bootstrap config: UI `language`, `require_purpose` (the parsed `AUDIT_REQUIRE_PURPOSE`, sorted) and `faces` (the operator switch `NEXTEXT_FACES`; the SPA shows the Faces toggle unless `off`); unaudited.
- `GET /metrics` — Prometheus exposition (aggregate request/latency counters plus the job-outcome counters below; no transcript or user data); unauthenticated, scraped by obs-plane over `inference-net`.

**Empty / not-processed outcomes.** A file with no processable speech is not a failure: the job **completes**, and `skipped: true` plus a typed `skip_reason_code` say why. The vocabulary lives in `nextext/core/outcomes.py`: `vad_no_speech` (the `/vad` guard rejected the file), `asr_empty_transcript` (Whisper returned nothing), `asr_all_segments_filtered` (every segment failed the `no_speech_prob` filter). Failures carry a typed `error_code` — `undecodable_media` or `internal` — while `error` stays the static `"Job failed."` so no detail leaks. Both codes appear on `GET /jobs/{id}`, on `GET /jobs` list items (the SPA's only source after a browser reload), and on the terminal `job_completed` / `job_failed` SSE frames; the frontend localizes them (`frontend/src/lib/outcomeMessages.ts`) and never renders the backend's own English prose. The worker logs one job-scoped warning per skipped or failed job and increments `nextext_jobs_total{outcome}`, `nextext_jobs_skipped_total{reason}`, `nextext_jobs_failed_total{code}` (`nextext/api/metrics.py`). A skipped job renders no *transcript-derived* artifacts — `archive.zip` 404s like its siblings rather than returning an empty ZIP — but the flag means "no transcript", not "no result": a silent video that asked for keyframes still carries its frames, their descriptions, and a summary written from them, and the SPA shows the banner above those tabs rather than instead of them. `nextext-cli` reports the same outcome as **exit code 3** (not `2`, which argparse uses for usage errors).

Identity is resolved per request by `resolve_principal`, in order: a service token (`Authorization: Bearer <token>` matching an entry of `NEXTEXT_SERVICE_TOKENS`) resolves to the service principal `service:<name>`, and the identity header on such a request is ignored — a service cannot act as a user; then the trusted header (`NEXTEXT_AUTH_HEADER`, default `X-Auth-User`), which with `NEXTEXT_EDGE_SECRET` set counts only on a request whose `X-Edge-Secret` matches the secret (constant-time; otherwise it is ignored and the method and path logged, never a value); then `NEXTEXT_DEFAULT_IDENTITY` (the dev / header-less fallback), honoured only while no edge secret is set — with both set it is ignored, startup warns once, and such a request gets `401`; else `401`. The `service:` namespace belongs to service tokens alone: a header value in it is ignored like an unverified header, and a default identity in it fails startup, so a user named like a service (`docint`) never shares the service's jobs, which hold other users' media and transcripts. The value scopes the caller's in-memory jobs; cross-owner reads return `404` so existence never leaks. The React frontend mints a per-browser id and carries it in its URL (`?owner=<id>`) on first visit, reading it back on every reload so the identity survives browser refreshes; behind the edge gateway, the gateway replaces it with the signed-in user. Only the gateway holds the edge secret, so with it set nothing else that can reach the backend can assert a user identity; without it, the header is trusted from any caller that can reach the backend, and startup warns unless `NEXTEXT_DEFAULT_IDENTITY` marks a dev setup. There is no durable storage: jobs live only in memory.

**Data-protection audit log** (`docs/audit.md`). Every audited request writes one hash-chained event (`nextext/api/audit.py`: `audited(...)` resolves the actor from the identity seam — `edge`, `service`, or `system` for the default identity — reads `X-Audit-Purpose` (`400` when malformed), keeps `X-Request-ID`, and refuses with `428` plus a `denied` event when `AUDIT_REQUIRE_PURPOSE` lists the action and no purpose is stated); routes emit after the data is fetched and before it is returned. The job manager writes the job's own events with the creating request's actor, purpose and request id: `collect` at creation, one `ai_call` per inference task of every stage (`nextext/audit.py` stage tables over the `nextext/core/inference_trace.py` tally the six inference clients report into, which also forward `X-Request-ID`), `combine` for the joined result, and `delete` for owner deletes, the shutdown purge and the startup sweep. Status polls (an unfinished job's snapshot) and the SSE streams carry no personal data and are not recorded; cross-owner attempts are recorded as `denied`. A deployment requires a purpose for `query` and `disclose` (`docker/compose.yaml` default, colon-less so an empty value switches it off): the SPA states it from the header's purpose bar, playback carries it through a media grant, and `make cli` takes `CASE_REF`/`REASON`. Fail closed: a failed audit write fails the request (generic `500`, no data), the job, the CLI run, or startup. Hate-speech results are never recorded — only what was judged; the face stage likewise records `frames` (`ai.detect_faces`) and `tracks` (`ai.group_faces`), never what was found, and `combine` lists `faces` whenever the stage produced a result. The library is vendored byte-identically (`nextext/utils/audit_event.py`, `nextext/utils/audit_runtime.py`) — never edit or reformat it.

**Key modules:**

- `nextext/api/main.py` — FastAPI factory, lifespan (opens the audit chain and its `config.audit` event, boots the in-memory `JobManager`, closes the chain on shutdown). `create_app(audit_emitter_factory=…, pipeline_runner=…)` are the test seams.
- `nextext/api/jobs.py` — `JobManager`, async workers bounded by a configurable `asyncio.Semaphore` (`NEXTEXT_JOB_CONCURRENCY`, default `1` = single in-flight job), SSE event broker. Holds all jobs in memory, and their playback grants (`media_grants`, revoked on delete and at shutdown); `list_for_owner` powers the frontend's reload re-discovery. Writes each job's audit events (collect, per-stage `ai_call`, combine, delete), the owner record next to each upload, the startup sweep of leftover job directories and the shutdown purge; seals its `AuditSink` before the chain closes.
- `nextext/api/job_record.py` — the owner record (`job.json`: job id, owner, created) written atomically into each job directory, which the startup sweep reads to attribute what it deletes.
- `nextext/api/audit.py` — request-side audit: the `audited(...)` dependency (actor, purpose, request id, `428`), `media_audit`, `RequestAudit.emit` / `emit_ids`, `RequestAudit.authorized_by` (a grant-played stream takes the grant request's purpose and request id).
- `nextext/api/media_grants.py` — `MediaGrants`, the in-memory playback grants: keyed by the token's SHA-256 (never the token), bound to the job and the issuing request's `Origin`, expiring after `NEXTEXT_MEDIA_GRANT_TTL_S`, at most 16 live per job, revoked with the job.
- `nextext/audit.py` — the shared audit glue for API and CLI: `build_emitter` (`AUDIT_OUTPUT`), `emit_config_event`, `AuditSink`, `JobAudit` (a job's or CLI run's events), the per-stage `AiSpec` tables, `emit_with_ids` (continuation events past 200 ids), `joined_analyses`.
- `nextext/core/inference_trace.py` — the context-var tally every inference client reports each request into (task, model, ok, tokens) and the `X-Request-ID` the clients forward.
- `nextext/api/identity.py` — `resolve_principal` FastAPI dependency: service token (`NEXTEXT_SERVICE_TOKENS`, principal `service:<name>`) → trusted header (`NEXTEXT_AUTH_HEADER`, default `X-Auth-User`; edge-verified via `X-Edge-Secret` when `NEXTEXT_EDGE_SECRET` is set, and never a `service:` value) → `NEXTEXT_DEFAULT_IDENTITY` → `401`. `check_principal_config` runs in the lifespan: a malformed secret or token list, or a `service:` default identity, stops startup (`PrincipalConfigError`, whose messages never carry a value); the production shape without an edge secret logs one warning, and so does a default identity set next to an edge secret (it is ignored). `identity_headers_trusted` applies the same edge check to the decorative `X-Auth-Name` that `/whoami` echoes. The React frontend carries the identity in its URL (`?owner=<id>`); there are no server-managed cookies. Routes and ownership checks see only the resolved principal, so this is the single identity seam.
- `nextext/api/routes/` — `health`, `jobs` routers. Per-route ownership checks return `404` on cross-owner access so existence never leaks.
- `nextext/api/routes/jobs.py` — also serves `POST /jobs/{id}/media-grant` and `GET /jobs/{id}/media` via `FileResponse` (Starlette answers `Range` with `206`), with the MIME type guessed from the upload name; `_authorize_grant` / `_authorize_token` are the media route's two capability paths.
- `nextext/api/artifacts.py` — Per-job artifact byte materializers (CSV/XLSX/PNG/JSONL/ZIP) rendered on demand from the in-memory `state.result`.
- `nextext/api/schemas.py` — Pydantic request/response models for jobs, snapshots, and the SSE event payloads.
- `nextext/api/metrics.py` — job-outcome Prometheus counters (`nextext_jobs_total`, `nextext_jobs_skipped_total`, `nextext_jobs_failed_total`), registered on the default registry the instrumentator exposes. Typed-code labels only.
- `nextext/core/outcomes.py` — the typed `SkipReason` / `FailureCode` vocabulary shared by the pipeline, API, and CLI, plus the English fallback prose for non-UI consumers.
- `nextext/cli.py` — CLI entry point (argparse), single-file processing in-process. `run_cli` wraps each run in its own audit chain (`via="cli"`, `AUDIT_CLI_USER`, purpose from `AUDIT_CASE_REF`/`AUDIT_REASON`, exit `2` on a refusal) and records every output file it writes (`file:<output>`). `_keyframe_step` runs the opt-in keyframe/captioning step (`-kf/--keyframes`) ahead of the no-speech guard, writing `{stem}_keyframes/frame_NNN.jpg` and `{stem}_visual_context.txt`.
- `nextext/pipeline.py` — Shared pipeline functions connecting all agents. `transcription_pipeline` returns a `TranscriptionOutcome` (transcript, resolved source language, `skip_reason`); its `is_empty` property is the single definition of "no usable text" that the API worker and CLI both branch on.
- `nextext/core/transcription.py` — `ExternalWhisperTranscriber` (OpenAI-compatible audio API); the pre-upload speech guard is delegated to the external `/vad` service via `core/vad.py`. Records `skip_reason` when a run yields no segments, so the three causes stay distinguishable downstream.
- `nextext/core/audio.py` — audio-normalization agent: re-encodes any upload to 16 kHz mono FLAC via PyAV (bundled ffmpeg) before the Whisper call; fail-closed (`AudioDecodeError`) on undecodable input.
- `nextext/core/vad.py` — voice-activity-detection agent: fail-open HTTP client for the out-of-process `/vad` service — `has_speech` (pre-Whisper guard); an unset/unreachable endpoint transcribes everything.
- `nextext/core/diarization.py` — speaker-diarization agent: HTTP client for the out-of-process `/diarize` service; aligns turns onto the transcript at the word level (`build_speaker_segments`), falling back to segment-level overlap (`assign_speakers_by_overlap`) when word timestamps are unavailable, labels segments that overlap no turn with the temporally nearest turn's speaker (`fill_speakers_by_nearest_turn`, preferring the preceding turn on a tie, so no segment renders as `Unknown` while others carry speakers), then renumbers the finished transcript's speakers to contiguous `Speaker N` by first appearance in reading order (`renumber_speakers_by_appearance`; supersedes the earlier turn-order `canonicalize_speaker_labels` for what the transcript displays).
- `nextext/core/visual_context.py` — visual-context agent: captions sampled
  video keyframes via `InferencePipeline.call_vision` (`describe_keyframes`,
  one request per frame), downscales each frame first (`prepare_frame`), and
  renders the `[mm:ss] caption` block that is both an artifact of its own and
  an optional summarizer input (`format_visual_context`). Fail-soft throughout — a per-frame outage skips
  that caption, and a client rejection on the first frame (a text-only
  `TEXT_MODEL`) aborts the run after one round-trip.
- `nextext/core/keyframes.py` — keyframe sampler: `extract_keyframe_samples`
  returns timestamped `Keyframe` objects spanning the clip's full duration;
  `extract_keyframes` is the bytes-only wrapper the `keyframes.zip` artifact
  uses, and `keyframe_manifest` renders the frame/time index both the artifact
  and the CLI write. `sample_frames` is the face stage's time-based sampler
  (`SampledFrame`: index, time, upright RGB image, its `scale` against the
  source, the clip's measured duration); `_video_stream` skips cover art.
- `nextext/core/faces.py` — faces agent: HTTP client for the out-of-process
  `/faces/detect` and `/faces/group` service (`detect_faces`, `group_faces`,
  `group_request_body`); fail-soft (`None` on no endpoint, failure or a broken
  contract), one retry on a short `503 Retry-After`, `MAX_REQUEST_BYTES`
  (16 MiB) budget, errors logged by `error_label` only.
- `nextext/core/face_tracks.py` — pure geometry: IoU, shot cuts on grey
  thumbnails, `assign_tracks` (keeps a caller's open-track list in step),
  frontality, sharpness, quality, `merge_spans`.
- `nextext/core/face_extraction.py` — `extract_faces`: batching within the
  source budget, filtering, tracking, crops (best/crops/group/thumbnail),
  online pruning of closed tracks beyond the cap, the three-failed-batches
  breaker, grouping with its one-crop fallback, caps, numbering, appearances;
  `FacesResult` / `FaceEntry` / `FaceCrop` / `DetectionRow`.
  `pipeline.faces_pipeline` reads `load_faces_sampling_env()` and calls it.
- `nextext/core/sentence_segmentation.py` — sentence-restoration agent: for
  low-punctuation transcripts (e.g. Arabic), re-segments the word stream into
  one segment per sentence via `TEXT_MODEL` (`restore_sentence_segments`), which
  returns `index:code` boundaries (never text) and appends the classified
  terminal mark (`.`/`؟`/`!`). Gated on `terminal_punctuation_ratio`; fail-soft.
- `nextext/core/ner.py` — named-entity-recognition agent: HTTP client for the out-of-process `/gliner` service (`extract_entities`).
- `nextext/core/translation.py` — LLM translation with prompt templates.
- `nextext/core/words.py` — NLP word-level analysis (spaCy word counts + word clouds).
- `nextext/core/hate_speech.py` — hate-speech agent: `next_window` builds a labelled core plus read-only context margins within character budgets; `render_window_prompt` fills the `hate_speech_transcript` prompt in one regex pass; `window_response_format` is a strict JSON schema whose `index` enum admits only core rows; `parse_window_reply` derives the verdict from stance alone (`endorses`), with category/confidence allow-lists; `classify_window` sends one temperature-0 request without a system role. Orchestration, structured-output fallbacks and overflow retries live in `hate_speech_pipeline` (`nextext/pipeline.py`). The prompt is byte-identical to docint's `hate_speech_transcript.txt` — change both repos together.
- `nextext/core/openai_cfg.py` — `InferencePipeline` for OpenAI-compatible LLM calls.
- `nextext/core/processing.py` — File I/O and export formatting (CLI).
- `nextext/utils/mappings/` — JSON config files for Whisper/spaCy model names, language codes.
- `nextext/utils/prompts/` — LLM prompt templates (system, translation, summary, hate_speech_transcript, sentence_segment, frame_caption), localized per language under `en/` and `de/` (selected by `RESPONSE_LANGUAGE`, with `NEXTEXT_RESPONSE_LANGUAGE` honored as a deprecated fallback; English fallback).

## Environment

`.env.example` follows deploy's shared format (deploy `docs/runbooks/env-sync.md`, For member authors); check with `python3 -B ../deploy/scripts/check_env_examples.py .`.

Key env vars (see `.env.example`):

- `OPENAI_API_KEY`, `OPENAI_API_BASE` — the **central** OpenAI-compatible endpoint; carries translation, summarization, and hate-speech detection (all on `TEXT_MODEL`), supplies the bearer token reused by the NER, diarization, and VAD clients, and is the fallback for every per-model endpoint below (Whisper verbatim; NER/diarization/VAD with one trailing `/v1` stripped, since they speak a plain service root).
- `WHISPER_API_BASE` / `WHISPER_API_KEY` / `WHISPER_MODEL` — dedicated Whisper endpoint (OpenAI SDK base incl. `/v1`); falls back to the central pair. Model defaults: `whisper-1` (openai), `openai/whisper-large-v3` (vllm). `INFERENCE_PROVIDER=ollama` has no transcription API, so it requires explicit `WHISPER_API_BASE` + `WHISPER_MODEL` (`load_whisper_env` raises otherwise).
- `NER_API_BASE` — root URL of the out-of-process `/gliner` NER service (e.g. `http://vllm-router:4000`); the client appends `/gliner`. Defaults to the central `OPENAI_API_BASE` (one trailing `/v1` stripped); set it only to point NER elsewhere. NER issues a request only when a job requests entities. The bearer token is reused from `OPENAI_API_KEY`. Fail-soft: errors degrade to empty entities. `NER_TIMEOUT` — per-request (per-chunk) timeout in seconds (default `120`).
- `DIARIZE_API_BASE` — root URL of the out-of-process `/diarize` service (e.g. `http://vllm-router:4000`); the client appends `/diarize`. Defaults to the central `OPENAI_API_BASE` (one trailing `/v1` stripped); set it only to override. Diarization runs by default for every job (auto-detecting the speaker count) unless the job sets `diarize=false`. Nextext requests word timestamps (`timestamp_granularities=["segment","word"]`) from the Whisper call and degrades gracefully to segment-level alignment when the ASR endpoint returns none. The bearer token is reused from `OPENAI_API_KEY`. Fail-soft: errors degrade to an unlabelled transcript. `DIARIZE_TIMEOUT` — per-request timeout in seconds (default `600`). See `nextext/core/diarization.py` for the `/diarize` request/response contract.
- `INFERENCE_PROVIDER` — `ollama` (code default), `vllm`, or `openai`; `docker/compose.yaml` defaults it to `vllm`, and `OPENAI_API_BASE` to the router `http://vllm-router:4000/v1`. Selects the Whisper model default and the Ollama `think` handling; prompts are provider-independent.
- `TEXT_MODEL` — LLM model name shared by translation, summarization, and hate-speech detection
- `RESPONSE_LANGUAGE` (backend + CLI + frontend) — Uniform federation UI-language switch: `en` (English, default) or `de` (German). Controls both the LLM output language (summaries, hate-speech rationales, prompts) and the SPA UI language. Backend selects the localized prompt subdirectory (`nextext/utils/prompts/<code>/`), frontend applies the `LanguageProvider` i18n translation. Missing locale files fall back to English; unrecognized values warn and fall back to `en`. Resolved by `load_language_env` in `nextext/utils/env_cfg.py` (backend + CLI) and the `LanguageProvider` (frontend).
- `NEXTEXT_RESPONSE_LANGUAGE` (backend + CLI, deprecated) — Legacy name for the output language setting. Kept as a fallback for one release; new deployments should use `RESPONSE_LANGUAGE` instead. If both are set, `RESPONSE_LANGUAGE` takes precedence.
- `SUMMARY_MAX_INPUT_TOKENS` (backend + CLI) — Max transcript tokens sent to `TEXT_MODEL` in a single summarize request. Longer transcripts are summarized map-reduce (chunk → summarize each → recursively summarize the combined partials) so no request overflows the chat model's context window; short transcripts take a single-shot path. Every request also caps output at 1024 tokens (`SUMMARY_MAX_OUTPUT_TOKENS` in `nextext/pipeline.py`). The token budget is converted to a character budget with a conservative ratio (`_CHARS_PER_TOKEN`), so lower it for token-dense scripts (e.g. CJK) or small `max_model_len` backends and raise it to reduce chunking. If a request still overflows, the budget auto-halves and retries (up to 4×), then fail-soft degrades to an empty summary rather than crashing the job. Invalid/≤0 values warn and fall back. Defaults to `6000`.
- `HATE_SPEECH_WINDOW_TOKENS` / `HATE_SPEECH_CONTEXT_TOKENS` (backend + CLI) — Context-window budgets for hate-speech detection: estimated tokens of rows labelled per request (default `1000`) and of read-only neighbouring rows shown on each side (default `300`; `0` disables context). Converted to characters with `_CHARS_PER_TOKEN`. Requests carry a JSON-schema `response_format`; a 400/422 rejection or an unparseable constrained reply switches the rest of the sweep to unconstrained requests, a context overflow halves both budgets and retries, and a reply cut off at the output cap (`finish_reason: length`, via `InferencePipeline.call_model_reply`) halves the core and asks again (the cap scales at 80 tokens per core row, min 1024). Invalid values warn and fall back. Resolved by `load_hate_speech_env`.
- `OLLAMA_THINK` — tri-state default for the Ollama `think` request field forwarded by `InferencePipeline.call_model` via `extra_body`. Accepts `1`/`true`/`yes`/`on` (enable), `0`/`false`/`no`/`off` (disable), or unset (omit field, model default). Honoured by Ollama-hosted reasoning models such as Qwen3; a no-op for `vllm`/`openai` providers. Per-call `think=` overrides the env default.
- `VAD_API_BASE` — root URL of the out-of-process `/vad` speech-guard service (e.g. `http://vllm-router:4000`); the client appends `/vad`. Defaults to the central `OPENAI_API_BASE` (one trailing `/v1` stripped), so the guard runs ahead of every transcription; set `VAD_API_BASE=off` (or `false`/`no`/`0`) to switch it off, or a URL to override. The bearer token is reused from `OPENAI_API_KEY`. Fail-open: an unreachable service degrades to transcribing anyway. `VAD_TIMEOUT` — per-request timeout in seconds (default `60`). See `nextext/core/vad.py` for the `/vad` request/response contract.
- `NEXTEXT_SENTENCE_RESTORE` / `SENTENCE_RESTORE_MIN_PUNCT_RATIO` (backend + CLI) —
  Sentence restoration for punctuation-poor transcripts. When on (default) and a
  transcript's terminal-punctuation density (marks ÷ words) is below
  `SENTENCE_RESTORE_MIN_PUNCT_RATIO` (default `0.01`), each contiguous speaker
  run is re-segmented into whole sentences by `TEXT_MODEL`, so rows are one
  sentence each (granular and a coherent translation unit) instead of a
  whole-speaker-turn blob. The model returns `index:code` boundaries — never
  text — so words/timestamps stay untouched; questions get `؟`, exclamations
  `!`, else `.`. Fail-soft: a model outage degrades to today's behavior. Resolved
  by `load_sentence_restore_env`. Set `NEXTEXT_SENTENCE_RESTORE=off` to disable.
- `NEXTEXT_FACES` (backend only) — operator switch for the face stage: `off` (default; a job asking for faces gets `422`), `detect` (find and cut out faces; no biometric processing) or `group` (also grouped per person — biometric processing: the controller extends the DPIA first, § 67 BDSG; the software enforces nothing). Read once at startup by `load_faces_mode`, recorded in `config.audit`, reported by `GET /config`; on with no resolvable endpoint → one startup warning. `FACES_API_BASE` (service root; unset → central `OPENAI_API_BASE` with one `/v1` stripped; an off token → no endpoint), `FACES_TIMEOUT` (`300` s, read once per video), `FACES_INTERVAL_S` (`1.0`), `FACES_MAX_FRAMES` (`600`, ceiling `3600`) — resolved by `load_faces_env` / `load_faces_sampling_env`. `tests/conftest.py` clears all five before every test.
- `NEXTEXT_VISUAL_SUMMARY` / `VISUAL_SUMMARY_MAX_FRAMES` / `VISUAL_SUMMARY_IMAGE_MAX_SIDE` (backend + CLI) —
  The captioning half of the keyframe step, which the job itself opts into
  (`keyframes=true` / `nextext-cli -kf`). When on (default), each sampled frame
  is captioned by `TEXT_MODEL`; the captions surface as
  `JobResult.frame_captions`, the `visual_context.txt` artifact, and — when a
  summary was also requested — one of its sources. Setting
  `NEXTEXT_VISUAL_SUMMARY=off` skips captioning only: the frames are still
  sampled and downloadable as `keyframes.zip`. It is the **operator** half of a
  two-part gate; the per-job `visual_context` option (default `true`) is the
  client half, for a caller that wants the frames but not the words. Its
  default cannot be `false`: `JobOptions` forbids extra fields, so a client
  that predates the option cannot send it, and any other default would change
  what the SPA already gets. Either gate off means `_ensure_inference()` is
  never reached — which matters beyond cost, since that call sits outside the
  captioning fail-soft guard and the keyframe stage runs first, so it was the
  one thing that could fail a frames-only job on an unhealthy chat router and
  discard its transcript. Requires a vision-capable
  `TEXT_MODEL` — a text-only one degrades to an uncaptioned run with one
  warning. `VISUAL_SUMMARY_MAX_FRAMES` (default `12`, clamped to `200`) bounds
  the per-job cost at one inference request per frame;
  `VISUAL_SUMMARY_IMAGE_MAX_SIDE` (default `1024`) bounds each frame's upload
  size. Resolved by `load_visual_summary_env`.
- `NEXTEXT_OFFLINE=1` (default) — gates the spaCy downloads (`is_offline()`); the only local downloads left. Offline + uncached spaCy model raises an actionable error.
- `NEXTEXT_HOST_PORT` (frontend, dev/override only) — host port published by `make up-dev` for the nginx frontend container. Defaults to `8501`; maps to nginx port `8080` (the unprivileged nginx image listens there — see Container hardening below).
- `NEXTEXT_CLIENT_MAX_BODY_SIZE` (frontend) — nginx `client_max_body_size` for the `/api/v1` upload proxy. Defaults to `8192m`.
- `NEXTEXT_API_HOST` / `NEXTEXT_API_PORT` (backend only) — uvicorn bind address. Defaults to `0.0.0.0:8000`.
- `NEXTEXT_DEFAULT_TARGET_LANG` (backend only) — Initial translation target language code surfaced by `GET /languages` as `default_target` and used to seed the frontend's "Target language" dropdown on a fresh browser. Must be a supported target code; an unsupported (or unset) value falls back to English (`en`). The frontend persists the user's own selection per-browser (localStorage), so it survives reloads and takes precedence over this default. Defaults to `en`.
- `NEXTEXT_MAX_UPLOAD_MB` (backend only) — Hard cap on per-file upload bytes. The backend streams the upload to disk in 1 MiB chunks (`_stream_upload_to_disk` in `nextext/api/routes/jobs.py`) and returns `413` once the cap is exceeded; unparseable values fall back to the default and `<1` clamps to `1`. The React frontend mirrors the *default* as an advisory client-side check only (`DEFAULT_MAX_FILE_MB` in `frontend/src/lib/uploadGuard.ts`) — it does not read the env var, and the backend is the enforcement point. The separate nginx body limit is `NEXTEXT_CLIENT_MAX_BODY_SIZE`. Defaults to `8192`.
- `NEXTEXT_JOB_CONCURRENCY` (backend only) — Max jobs the in-memory `JobManager` runs concurrently (`asyncio.Semaphore`). Defaults to `1` (serial, one in-flight job — the historical behavior); raise it to overlap jobs, bounded by container CPU (PyAV decode per job) and the external inference services' capacity. Unparseable/`<1` values clamp to `1`. Resolved by `load_job_concurrency` in `nextext/utils/env_cfg.py`.
- `KEYFRAMES_PER_MINUTE` (backend + CLI) — Default keyframes sampled per minute of video, applied to `JobOptions.keyframes_per_minute` only when a job-creation request omits the field (an explicit per-request value always wins), and used directly by `nextext-cli -kf`. Invalid values warn and fall back; negatives clamp to `0`. Resolved by `load_keyframe_defaults` in `nextext/utils/env_cfg.py`. Defaults to `4`. Both rates apply only to a job that asked for keyframes.
- `KEYFRAMES_MAX` (backend + CLI) — Default hard ceiling on keyframes returned per clip, applied to `JobOptions.keyframes_max` only when a request omits it. Clamped to `[0, 200]` (the schema's hard cap; larger values warn and clamp to `200`); an explicit per-request value still overrides. Defaults to `20`.
- There is no combined-batch size cap. The old `NEXTEXT_MAX_BATCH_MB` existed for Streamlit's `file_uploader`, which held a whole multi-file selection in the frontend process's memory; the React SPA streams each file through nginx instead, so only the **per-file** guard above applies. Large local batches still belong in `nextext-cli`, which reads from disk and never buffers whole files.
- `NEXTEXT_AUTH_HEADER` (backend only) — Name of the trusted identity header. Defaults to `X-Auth-User`. The SPA (`frontend/src/api/client.ts`) and the edge gateway always send `X-Auth-User`, so another name suits only other callers.
- `NEXTEXT_DEFAULT_IDENTITY` (backend only) — Fallback identity for header-less / developer callers. Unset by default, so a request without the trusted header gets `401`. Honoured only while `NEXTEXT_EDGE_SECRET` is unset: with both set it is ignored (one startup warning naming both) and an unverified request gets `401`. Must not start with `service:` (reserved for service tokens), else startup fails.
- `NEXTEXT_EDGE_SECRET` (backend only) — Secret the edge gateway sends as `X-Edge-Secret`; when set, the trusted header counts only on requests carrying it. At least 32 characters, else startup fails. Generated with `make secret` in edge-plane, whose `.env` holds the same value under the same name. Unset keeps the header trusted from any caller (startup warning in the production shape). Rollout order: `docs/configuration.md` § Identity.
- `NEXTEXT_SERVICE_TOKENS` (backend only) — Comma-separated `name=token` pairs for services that call the API directly; `Authorization: Bearer <token>` resolves to the principal `service:<name>` and ignores the identity header. Names match `[A-Za-z0-9._-]+` and are unique; tokens are ≥ 32 characters, unique, and never the edge secret — anything else fails startup. docint's entry carries the value of docint's `NEXTEXT_API_KEY`.
- `AUDIT_OUTPUT` (backend + CLI) — where the audit chain goes: unset/`stdout` for the process's stdout, `container` for the container's main stdout under a container-wide lock (`docker/compose.yaml` sets it for the backend, so `make cli` runs land in its log). Anything else stops startup.
- `AUDIT_REQUIRE_PURPOSE` (backend + CLI) — comma-separated audit actions that need a stated purpose. Empty in code (tests, bare `uvicorn`); `docker/compose.yaml` defaults it to `query,disclose` while unset (`${AUDIT_REQUIRE_PURPOSE-query,disclose}`, colon-less so `AUDIT_REQUIRE_PURPOSE=` empty switches it off). A request without one gets `428`, a CLI run exits `2`; an unknown action stops startup. Job creation counts as `collect`, `ai_call` and `combine`; a CLI run as those plus `disclose`, so under the default every run needs a purpose. Exposed to the SPA as `GET /config`'s `require_purpose`.
- `NEXTEXT_MEDIA_GRANT_TTL_S` (backend only) — seconds a media playback grant stays usable; default `900`, at most `86400`, invalid values fall back. Resolved once per `JobManager` by `load_media_grant_ttl`.
- `AUDIT_CLI_USER` / `AUDIT_CASE_REF` / `AUDIT_REASON` (CLI only) — the operator a run is attributed to and its stated purpose; `make cli` sets them (`CASE_REF=… REASON=…`).
- `NEXTEXT_CLI_RETENTION_DAYS` / `RETENTION_NOTICE_DAYS` (backend + CLI) — what `make cli` leaves in `/tmp/nextext-cli` (and `/tmp/output` of earlier releases) is deleted that many days after it was written (365; `off` keeps it, development only), by `nextext/api/cli_files.py` at startup and daily, recorded (`cli.sweep`); a new period waits the notice (30 days), a changed one never deletes sooner than announced: longer applies at once, shorter gives the notice only to what it makes due sooner (`RetentionRule`, state in `/tmp/nextext-cli/.retention-since`; off forgets it). `make cli-clean` (`nextext/cli_clean.py`) deletes now (`cli.clean`). Invalid values stop startup. Each CLI run prints the earliest date its outputs are deleted, however it ended.

## Docker

Docker assets live under `docker/`. `docker/compose.yaml` defines two services — no profiles, no GPU reservations:

- `backend` — built from `docker/Dockerfile.backend`, multi-stage `uv` build (no extras; runtime apt is `curl` only — all inference, including the VAD guard, is external; audio normalization uses the PyAV wheel, so no apt audio tooling is added). Runs `uvicorn nextext.api.main:app` with a `HEALTHCHECK` against `/api/v1/health`. Reachable only on the `nextext-net` network by default; no host port is published. `AUDIT_OUTPUT: container` sends every process's audit chain to the container's stdout; `make cli` runs `nextext-cli` in it via `docker compose exec`. `AUDIT_REQUIRE_PURPOSE` defaults to `query,disclose` here (empty in `.env` switches it off); the dev overlay does not override it.
- `frontend` — React SPA compiled and served by nginx. Built from `docker/Dockerfile.frontend` (node build → nginx image). The nginx config proxies `/api/v1` same-origin to the backend, so browser uploads stream through nginx without buffering whole files in any Python process. The base `docker/compose.yaml` is the production shape and publishes no host ports; `docker/compose.override.yaml` (layered by `make up-dev`) publishes nginx on `${NEXTEXT_HOST_PORT:-8501}`.

The stack shares `inference-net` with the inference provider (vllm-service / Ollama). The `Makefile` is the entry point — it points Compose at `docker/compose.yaml`, since a bare `docker compose` from the repo root no longer finds it. Run `make volumes` (one-time, creates the external `spacy-cache` volume), then `make build && make up` for production shape, or `make build && make up-dev` (or just `make dev`) to publish the frontend on the host. `make up`/`make up-dev` are detached and never build (`--no-build`), so build the images first (in prod, load or pull them). `make bundle` writes an image tarball built from the latest annotated release tag (production); `make bundle-dev` bundles the current working tree instead (dev/soak).

The React SPA source lives in `frontend/`; run `cd frontend && pnpm {dev,build,test,lint,typecheck}` for local development without Docker.

**Container hardening (deploy ADR 0001):** both containers run non-root with
read-only root filesystems — the backend as uid `10001` (`app`,
`HOME=/home/app`; `SPACY_MODEL_DIR` points at the cache volume
under `/home/app`, `/tmp` is a disk-backed scratch volume sized for multi-GB
job media, and `MPLCONFIGDIR=/tmp/matplotlib` keeps matplotlib's import-time
config/font cache off the read-only `$HOME/.config`), the frontend on
`nginxinc/nginx-unprivileged` as uid `101`
listening on **:8080** (the edge gateway's `nextext-frontend` upstream must
match). Compose applies `no-new-privileges` + `cap_drop: ALL` via the
`x-hardened` anchor. On existing hosts the `spacy-cache` volume needs a
one-time `chown -R 10001:10001` (runbook in the `deploy` repo). The
`nltk-cache` volume is no longer mounted or declared; it stays on hosts that
already have it until an operator removes it (`docker volume rm nltk-cache`),
since compose cannot drop external volume state it does not own.

## Persistence model

Jobs live only in memory. `JobManager` holds them in a dict keyed by `job_id` and scoped by `owner_id`; there is no SQLite index, no on-disk artifacts, and no TTL sweeper. A job is retained until the owner `DELETE`s it or the backend process exits — nothing ever cuts off a long-running job. The one thing kept on disk besides the upload is its owner record (`job.json`, `nextext/api/job_record.py`), so that after a crash the next startup can remove the leftover job directories and attribute their deletion in the audit log; a directory without a valid record is removed too, recorded without job or owner.

Reload resilience comes from the identity, not from storage. The owner id survives a browser refresh in the page URL (`?owner=<id>`), so on load the frontend calls `GET /jobs` to re-discover the caller's jobs and resumes them: it re-subscribes to any still running (the SSE broker replays each job's event history on connect) and re-renders those already finished. A run therefore survives a browser reload during processing, but not a backend restart.

The upload itself outlives the pipeline: it stays at `state.file_path` until the owner `DELETE`s the job or the backend exits (or, after a crash, the next startup's sweep), which is what makes playback possible after a job completes. Nothing else reclaims it. The FLAC re-encode for Whisper is written into the same job directory (`scratch_dir`), so it goes with the upload.

Artifacts (`.csv`/`.xlsx`/`.png`/`.jsonl`/`.zip`) are materialised on demand from the in-memory `state.result` by `nextext/api/artifacts.py`; they are never written to disk.

## Deployment: edge-plane gateway sub-path

The Nextext SPA is served in production under the canonical `/nextext/`
sub-path behind the `edge-plane` gateway, not at its own vhost root. The
`frontend` service joins the external `edge-net` network (alongside its
existing `nextext-net` membership) as alias `nextext-frontend`, which is how
the gateway reaches it. Vite is built with `base: '/nextext/'`, `API_BASE`
derives from `BASE_URL` (`frontend/src/api/client.ts`'s `apiBase()`), the
`BrowserRouter` uses a matching `basename`, and the frontend's nginx config
strips the `/nextext` prefix internally before falling through to the
existing root-anchored locations (the SSE job-events endpoint included),
redirecting bare `/` to `/nextext/`. The gateway is the sole production entry
point and is what injects `X-Auth-User` for the backend's trusted-header
principal seam (`nextext/api/identity.py`), along with `X-Edge-Secret`, which
the frontend's nginx passes through and the backend requires before trusting
the header once `NEXTEXT_EDGE_SECRET` is set — requests without a trusted
header are then rejected as unauthenticated, and `NEXTEXT_DEFAULT_IDENTITY`
is ignored while the secret is set, so a dev-only fallback stays dev-only.

## Git & PR Workflow

- Prefer multiple small topical commits over a single catch-all commit.
- Each commit message should describe a single logical change (refactor, fix, feat, docs, test).
- Never commit directly to `main`; always branch (`feat/`, `fix/`) and open a PR.
- Never create a NEW PR when an existing PR for the work is open — push additional commits to that branch.
- Release order is strict: bump VERSION file -> commit -> tag. Never tag before the VERSION bump.
- Use single, non-compound shell commands for `gh` operations (no `&&` chains); if `gh pr merge` is blocked, fall back to the GitHub MCP merge tool.

## Communication Style

Keep changes minimal and scoped. Do not add explanatory code comments for trivial or self-evident changes. Do not overwrite existing test files with Write — use Edit to append or modify tests.
