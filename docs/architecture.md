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

## Playback

The original upload outlives the pipeline — it is removed only when the owner
deletes the job or the backend restarts — so a finished job can still be played
back. `GET /jobs/{id}/media` streams it through Starlette's `FileResponse`,
which answers `Range` requests with `206`; that is what lets the player jump to
a timestamp without re-fetching the whole recording.

The route is authorized by a per-job capability token in the URL rather than by
the request principal. A `<video>`/`<audio>` element cannot attach the trusted
identity header, and the blob-URL workaround used for the word cloud would
buffer the entire recording in memory and forfeit seeking. The token is minted
at job creation, handed out only on the owner-scoped snapshot, never listed,
and dies with the job; every failure answers `404`, so a wrong token cannot be
used to probe which jobs exist. Its one cost is that it appears in proxy access
logs.

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

Jobs live only in memory — there is no durable storage and no TTL, so a long-running job is never cut off and is retained until you delete it or the backend restarts. Identity is anonymous: the frontend mints a per-browser id and stamps it into the URL (`?owner=<id>`) on first visit, sending it to the backend as the trusted identity header (`X-Auth-User` by default) to scope your jobs. Because that id survives a refresh, reloading the page mid-run re-discovers your jobs and resumes the live progress view; closing the tab and reopening the bare host starts a fresh identity. Developers calling the API directly can skip the header and set `NEXTEXT_DEFAULT_IDENTITY` instead. There is no authentication — the backend trusts whoever can reach `inference-net`.

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

## Image tagging

Each build is tagged `nextext-{backend,frontend}:${NEXTEXT_VERSION}`, where
`NEXTEXT_VERSION` defaults to `latest`. Override it (e.g. for releases)
by exporting `NEXTEXT_VERSION` before running `make` (or a raw
`docker compose -f docker/compose.yaml` invocation).

When invoked through `make`, `NEXTEXT_VERSION` defaults to
`YYYY-MM-DD-<short-sha>` so each build gets a traceable tag. Export
`NEXTEXT_VERSION=…` beforehand to pin a specific version.
