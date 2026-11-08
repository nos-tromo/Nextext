# Architecture

Nextext runs as two cooperating containers on the `nextext-net` compose
network, plus the external seams it joins (`inference-net` for the model
endpoints, `edge-net` for the production gateway).

## Service split

The stack brings up two containers:

- **Backend** (`backend`) — FastAPI on port 8000 (internal). Owns the pipeline, the HTTP inference clients, and the in-memory job store. Exposes `/api/v1/health`, `/api/v1/languages`, `/api/v1/jobs/*`. Not published to the host by default.
- **Frontend** (`frontend`) — React SPA served by nginx on port 8080 (internal). nginx proxies `/api/v1` to the backend same-origin, so browser uploads stream through nginx without buffering whole files in Python.

Both images run non-root with read-only root filesystems (deploy ADR 0001):
the backend as uid `10001` (cache env vars point at the writable mounts —
`SPACY_MODEL_DIR` at the cache volume, `MPLCONFIGDIR` at
`/tmp/matplotlib` on the scratch volume), the frontend on
`nginxinc/nginx-unprivileged` as uid `101` listening on `:8080`. With `make up-dev`, the frontend is published
on `http://localhost:${NEXTEXT_HOST_PORT:-8501}/`, which maps to that nginx
port 8080.

`nextext-cli` keeps a third, container-free path: it imports the pipeline
directly and runs end-to-end in-process, without a backend. It ships inside the
backend image alongside the API — see [cli.md](cli.md).

## Keyframes and visual context

For video, transcription only hears the file. **Describing keyframes** is the
pipeline's second stage, between Transcribing and Translating: it samples
frames across the whole clip and describes each through `TEXT_MODEL`'s vision
path — one request per frame — producing the `keyframes.zip` archive, the
`[mm:ss] caption` block behind `visual_context.txt`, and the SPA's Visual
context tab.

The archive carries a `manifest.json` naming each frame's `file`, `index` and
`time_sec`, so a consumer can place a frame in the clip without downloading the
captions. It is omitted when the sampling times are unknown, never guessed.

It is asked for per job (`JobOptions.keyframes`, the **Keyframes** checkbox,
`nextext-cli -kf`) and is **off by default**: with the option off, nothing is
sampled and no video is decoded. It is independent of summarization in both
directions — a summary no longer pulls frames in behind your back, and frames
no longer need a summary to be worth producing. When both were asked for, the
caption block is prepended to the transcript inside the summarizer's own
budgeted payload, so the summary covers slides, scenes and legible on-screen
text alongside what was said.

Because the stage runs before the no-speech short-circuit, a video whose audio
held nothing still yields its frames and their descriptions — and, if a summary
was requested, one written from those descriptions alone. Such a job is still
reported as `skipped` with its typed code: that flag means "no transcript",
not "no result".

Captioning needs a vision-capable `TEXT_MODEL` and is fail-soft — a text-only
model aborts it after one request, leaving the sampled frames downloadable and
one warning behind. `NEXTEXT_VISUAL_SUMMARY` is the operator kill-switch for
captioning alone (sampling continues), and `VISUAL_SUMMARY_MAX_FRAMES` bounds
the cost — see
[configuration.md](configuration.md#visual-context-keyframe-descriptions).

## Faces

**Detecting faces** is the third stage, right after the keyframes and, like
them, before the no-speech short-circuit, so a silent video still yields its
faces. It runs only when the job asks for it (`JobOptions.faces`, the
**Faces** toggle) *and* the operator switch `NEXTEXT_FACES` is `detect` or
`group`; a job asking for faces while the switch is `off` is refused with
`422` before its upload is stored. There is no CLI option.

No face model runs in Nextext. The models live behind the inference router in
vllm-service (`/faces/detect`, `/faces/group`); `nextext/core/faces.py` is the
HTTP client, and everything else is arithmetic on boxes and grey levels:

1. **Sample** (`sample_frames` in `nextext/core/keyframes.py`): one frame every
   `FACES_INTERVAL_S`, at most `FACES_MAX_FRAMES`; a longer clip stretches the
   interval so the samples span it — its duration taken from the container,
   the stream, or, for a file that records none (a browser recording), the
   largest packet timestamp. When keyframes come at most two intervals apart
   only they are decoded, otherwise the reference frames. Times count from the
   stream's start, so they match the player's clock; the display rotation is
   applied, so portrait phone videos are upright. Cover art (an attached
   picture in an audio file) is never sampled, so an audio file gets no Faces
   tab. Frames larger than 3840 px are scaled down, and every reported box is
   mapped back to the source's own pixels.
2. **Detect** in batches of 8 images, each scaled to at most 1280 px, while
   at most about 100 MB of source frames are held at once.
3. **Filter**: at least 40 px; at most 20 faces a frame, the most prominent
   (score times area) kept.
4. **Track** (`nextext/core/face_tracks.py`): a detection continues a track
   when it overlaps its last box by an IoU of at least 0.5, its area is within
   1.5 times that box's, and at most one sample went by without it. A shot cut
   — a large grey-level change between tiny thumbnails of consecutive samples
   — closes every track.
5. **Score** each crop: detection score times size, frontality (from the
   landmarks) and sharpness (Laplacian variance).
6. **Group** (`group` mode only): each track goes with up to three crops and
   its sample indices, so tracks on screen together are never merged. A track
   with a good face (score at least 0.85, at least 64 px, roughly frontal) is
   an anchor and sends only its good crops; the anchors make up the people. A
   track without one (a profile, say) is weak and sends its best crops anyway:
   it may join the one person it matches, but never founds or links one, so a
   person seen from the side is not split off as a face of its own. The
   service returns one label per track and nothing else. A request over the
   16 MiB budget is retried with one crop per track; over even that, nothing is
   grouped.
7. **Number** the entries by first appearance and merge each entry's spans into
   appearances, each ending at the clip's end at the latest but never before
   the face was last seen.

Each entry keeps its best shot (a wide margin, at most 1024 px, so another
system can detect on it), three more crops (at most 256 px) and a thumbnail
(at most 128 px). Caps: 500 tracks and 100 entries, ranked by duration times
size; over either — or over 20 faces in a frame — the result is `truncated`.
Tracks without a usable face, or every track when grouping is off, failed or
over budget, stay single entries (`assigned=false`, "Face N").

The thumbnails travel in the finished snapshot as `data:` URIs
(`JobResult.faces`), never in an SSE frame or a `GET /jobs` item. The
artifacts are `faces.zip` (`person_NNN/` or `face_NNN/` folders with
`best.jpg` and `crop_N.jpg`, plus a `manifest.json` of each file's time, box,
score and quality), `faces.csv` (one row per detection, no images),
`face_NNN.jpg` (one entry's best shot) and `faces/` plus `{stem}_faces.csv`
inside `archive.zip`; none of them exists for a result without faces. Faces
never reach `docint.jsonl`.

The stage is fail-soft and never touches the chat provider. A failed batch
leaves its frames without faces (`outcome="partial"`); a busy service that asks
for a short wait (`503`, `Retry-After` at most 10 s) gets that batch once more;
after three failed batches in a row the rest of the video is neither decoded
nor sent, so a hung service cannot hold the job for hours. No answered batch
at all gives `outcome="unavailable"`. An unexpected error leaves the job
without faces and is logged with its traceback frames.

## Playback

The original upload outlives the pipeline — it is removed only when the owner
deletes the job or the backend restarts — so a finished job can still be played
back. `GET /jobs/{id}/media` streams it through Starlette's `FileResponse`,
which answers `Range` requests with `206`; that is what lets the player jump to
a timestamp without re-fetching the whole recording.

The route is authorized by a capability in the URL rather than by the request
principal. A `<video>`/`<audio>` element can attach neither the trusted identity
header nor the stated purpose, and the blob-URL workaround used for the word
cloud would buffer the entire recording in memory and forfeit seeking. So
before the player sets `src`, it asks for a playback grant
(`POST /jobs/{id}/media-grant`): an ordinary, owner-scoped request through the
SPA's API client, which states the purpose like any other. The grant is a fresh
random token, bound in the backend's memory to the job and to that request's
caller, purpose and request id, and usable for every range request of the
playback until it expires (`NEXTEXT_MEDIA_GRANT_TTL_S`, 900 s by default); each
streamed range is recorded with that purpose
([audit.md](audit.md#playback)). When a grant expires mid-playback, the element
fails on its next range request; the player then asks for a fresh grant once
and resumes where it stopped. With the edge secret set, a grant plays only for
the identity it was issued to.

The job's own token — minted at job creation, handed out on the owner-scoped
snapshot as `media_url`, never listed — still plays, but it states no purpose,
so it is refused while `AUDIT_REQUIRE_PURPOSE` lists `query`, as a deployment's
default does; the SPA no longer plays it. Grants and token die with the job,
and every failure answers `404`, so neither can be used to probe which jobs
exist. Both travel in the query string, which the app's own access logs — the
frontend's nginx and the backend's uvicorn — drop; nginx's error log still
records it when it logs an error for the request (see
[configuration.md](configuration.md#logs)).

In the SPA, the transcript row (or frame caption) under the playhead is
highlighted and scrolled into view as playback advances, so the highlight
stays on screen without the reader chasing it. It scrolls only once the row has
left the viewport, and centres it, so the page moves a screenful at a time
rather than a row at a time.

Auto-scrolling pauses as soon as the reader scrolls the page by hand — a wheel
or touch drag over the page, or a scroll keypress that no control is consuming
— so reading back over an earlier passage is never interrupted. It resumes on
either of two signals: a timestamp click, or the playhead's row coming back
into view. The second matters as much as the first; without it a single
trackpad flick left following switched off for the rest of the session, which
read as the feature simply not working.

The frontend's nginx gives the route its own location with `proxy_buffering
off` — the default would spill a multi-GB body into the container's 16 MB
tmpfs and withhold bytes the player wants immediately.

## Jobs and identity

Jobs live only in memory — there is no durable storage and no TTL, so a long-running job is never cut off and is retained until you delete it or the backend restarts. Each upload's directory also holds a small owner record, so the uploads a crashed backend leaves behind are removed at the next startup and the removal is attributed to their owners (see [audit.md](audit.md#the-owner-record)). Without the edge gateway, identity is anonymous: the frontend mints a per-browser id and stamps it into the URL (`?owner=<id>`) on first visit, sending it to the backend as the trusted identity header (`X-Auth-User` by default) to scope your jobs. Because that id survives a refresh, reloading the page mid-run re-discovers your jobs and resumes the live progress view; closing the tab and reopening the bare host starts a fresh identity. Developers calling the API directly can skip the header and set `NEXTEXT_DEFAULT_IDENTITY` instead; it applies only while no edge secret is set. Behind the gateway, the header carries the signed-in user. With `NEXTEXT_EDGE_SECRET` set, the backend believes the header only on requests carrying the gateway's `X-Edge-Secret`; without it, it believes whoever can reach the backend. Services that call the API directly, such as docint, authenticate with a token of their own (`NEXTEXT_SERVICE_TOKENS`) and act only as themselves, under a `service:` principal no user can claim — see [configuration.md § Identity](configuration.md#identity).

A job whose file held no processable speech still **completes** — it is a
result, not a failure — and carries `skipped: true` with a typed
`skip_reason_code` (`vad_no_speech`, `asr_empty_transcript`,
`asr_all_segments_filtered`) on the snapshot, the job list, and the
`job_completed` event. Failures carry a typed `error_code`
(`undecodable_media`, `internal`) beside the static `"Job failed."` message.
The SPA localizes these codes; the backend logs one warning per outcome and
counts them in `/metrics` (see
[configuration.md](configuration.md#metrics)).

Cross-owner reads return `404` rather than `403`, so the existence of another
owner's job never leaks. In production the `edge-plane` gateway is what injects
the trusted header; see
[configuration.md](configuration.md#production-sub-path).

## Audit log

The backend records every collection, query, download, combination and
deletion of a job, and every transfer of job content to a model, in a
hash-chained data-protection audit log on its stdout — one chain per process,
with the CLI's runs writing their own. The log collector takes the lines to
obs-plane's audit store. What each endpoint and stage records:
[audit.md](audit.md).

## Image tagging

Each build is tagged `nextext-{backend,frontend}:${NEXTEXT_VERSION}`, where
`NEXTEXT_VERSION` defaults to `latest`. Override it (e.g. for releases)
by exporting `NEXTEXT_VERSION` before running `make` (or a raw
`docker compose -f docker/compose.yaml` invocation).

When invoked through `make`, `NEXTEXT_VERSION` defaults to
`YYYY-MM-DD-<short-sha>` so each build gets a traceable tag. Export
`NEXTEXT_VERSION=…` beforehand to pin a specific version.
