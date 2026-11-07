# Configuration

Nextext is configured entirely through environment variables — no code changes
are required to swap inference providers, endpoints, or output language. The
canonical, annotated list of every variable lives in
[`.env.example`](../.env.example); this document explains the groups that need
more than a one-line comment.

## Inference provider and localization

Nextext communicates with any OpenAI-compatible inference provider via `OPENAI_API_BASE` and `OPENAI_API_KEY`. Provider selection is handled entirely through environment variables — no code changes required.

**Language selection:** Set `RESPONSE_LANGUAGE` to `en` (English, default) or `de` (German) to control both the LLM output language (summaries, hate-speech rationales, prompts) and the SPA UI language. Missing or unrecognized values fall back to English. The legacy `NEXTEXT_RESPONSE_LANGUAGE` is deprecated but accepted as a fallback for one release; prefer `RESPONSE_LANGUAGE` in new deployments.

## Dedicated endpoints

Every model class can also be re-pointed at a **dedicated endpoint**, falling back to the central pair when unset:

| Model | Dedicated env vars | Endpoint shape |
|-------|--------------------|----------------|
| Whisper transcription | `WHISPER_API_BASE` / `WHISPER_API_KEY` / `WHISPER_MODEL` | OpenAI SDK base incl. `/v1` |
| GLiNER NER | `NER_API_BASE` (+ `NER_TIMEOUT`) | service root, `POST {base}/gliner` |
| Speaker diarization | `DIARIZE_API_BASE` (+ `DIARIZE_TIMEOUT`) | service root, `POST {base}/diarize` |
| VAD speech guard | `VAD_API_BASE` (+ `VAD_TIMEOUT`) | service root, `POST {base}/vad` |
| Faces | `FACES_API_BASE` (+ `FACES_TIMEOUT`) | service root, `POST {base}/faces/detect`, `POST {base}/faces/group` |

The NER, diarization, VAD and faces services speak a plain service root rather than the OpenAI `/v1` shape, so the central fallback strips one trailing `/v1` from `OPENAI_API_BASE` before appending the service path (`http://vllm-router:4000/v1` → `http://vllm-router:4000/gliner`). Whisper, which speaks `/v1`, uses `OPENAI_API_BASE` verbatim. The four non-OpenAI services reuse `OPENAI_API_KEY` as their bearer token; none takes a dedicated key.

NER issues a request only when a job asks for entities, so it needs no off switch. Diarization runs by default for every job, auto-detecting the speaker count (no bounds sent), and is bypassed per job with `diarize=false` (CLI `--no-diarize`). The VAD guard runs ahead of every transcription (fail-open: an unreachable service transcribes anyway); switch it off with `VAD_API_BASE=off` (also `false` / `no` / `0`).

> **Diarization** and **VAD** reach plain `POST /diarize` and `POST /vad` services (multipart `file` + form fields → JSON). Point `DIARIZE_API_BASE` / `VAD_API_BASE` — or the central endpoint — at a service implementing them; the full request/response contracts live in `nextext/core/diarization.py` and `nextext/core/vad.py`.

Speaker diarization runs out-of-process against an HTTP `/diarize` service. It uses the central endpoint by default, or set `DIARIZE_API_BASE` to a dedicated service root; the gated-model agreements and any Hugging Face token live on the service side, not in Nextext.

## Provider setup

The Nextext compose services join an external Docker network (`inference-net`) so they can reach whichever inference container you deploy on that network. **Create the network and start your inference provider before running the compose stack.**

**Ollama (text tasks only — needs a separate Whisper endpoint):**

```bash
# Create Docker network and persistent cache
docker network create inference-net

# Run the Ollama service
docker run -d \
  --network inference-net \
  --name ollama \
  --gpus all \
  -v ollama-cache:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:0.32.15@sha256:57d60e686821ea81a7748a3ec8141308c8b8f95b27105713954abf7a6529e700
```

Then configure Nextext to reach it by adding the following to your `.env` file (Ollama serves no transcription API, so Whisper needs an explicit dedicated endpoint):

```bash
OPENAI_API_BASE=http://ollama:11434/v1
OPENAI_API_KEY=ollama
WHISPER_API_BASE=http://<your-whisper-host>:8000/v1
WHISPER_MODEL=openai/whisper-large-v3
```

**Hosted OpenAI API:**

```bash
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=your-key
```

Any other OpenAI-compatible endpoint (vLLM, LiteLLM, etc.) works the same way — set `OPENAI_API_BASE` to the `/v1` endpoint and `OPENAI_API_KEY` to whatever the provider expects.

## Recommended Ollama models

The following models are recommended and tested for this application (select depending on your hardware setup):

| Purpose | Model |
|---------|-------|
| Summarization / general | [`gemma3:27b-it-qat`](https://ollama.com/library/gemma3), [`gemma3:12b-it-qat`](https://ollama.com/library/gemma3), [`gemma3n:e4b`](https://ollama.com/library/gemma3n) |
| Translation | [`translategemma:27b`](https://ollama.com/library/translategemma), [`translategemma:12b`](https://ollama.com/library/translategemma), [`translategemma:4b`](https://ollama.com/library/translategemma) |

Pull models into the running Ollama container:

```bash
docker exec ollama ollama pull gemma3:12b-it-qat
```

Then set the model names in `.env`:

```bash
TEXT_MODEL=gemma3:12b-it-qat
```

## Upload size limits

The nginx upload limit is controlled by the `NEXTEXT_CLIENT_MAX_BODY_SIZE` env var (defaults to `8192m`). Override it in your `.env` file:

```bash
NEXTEXT_CLIENT_MAX_BODY_SIZE=16384m   # 16 GB
```

The backend's hard cap is set independently by `NEXTEXT_MAX_UPLOAD_MB` (defaults to `8192` MB).

## Job concurrency

`NEXTEXT_JOB_CONCURRENCY` (backend only) — Max jobs the in-memory `JobManager`
runs concurrently (`asyncio.Semaphore`). Defaults to `1` (serial, one in-flight
job — the historical behavior); raise it to overlap jobs, bounded by container
CPU (PyAV decode per job) and the external inference services' capacity.
Unparseable / `<1` values clamp to `1`. Resolved by `load_job_concurrency` in
`nextext/utils/env_cfg.py`.

## Sentence restoration

`NEXTEXT_SENTENCE_RESTORE` / `SENTENCE_RESTORE_MIN_PUNCT_RATIO` (backend + CLI) —
Sentence restoration for punctuation-poor transcripts. When on (default) and a
transcript's terminal-punctuation density (marks ÷ words) is below
`SENTENCE_RESTORE_MIN_PUNCT_RATIO` (default `0.01`), each contiguous speaker
run is re-segmented into whole sentences by `TEXT_MODEL`, so rows are one
sentence each (granular and a coherent translation unit) instead of a
whole-speaker-turn blob. The model returns `index:code` boundaries — never
text — so words/timestamps stay untouched; questions get `؟`, exclamations
`!`, else `.`. Fail-soft: a model outage degrades to today's behavior. Resolved
by `load_sentence_restore_env`. Set `NEXTEXT_SENTENCE_RESTORE=off` to disable.

## Hate-speech detection

`HATE_SPEECH_WINDOW_TOKENS` / `HATE_SPEECH_CONTEXT_TOKENS` (backend + CLI) —
context-window budgets for the hate-speech stage. Transcript rows are one
sentence each, too little for the model to tell a speaker who *condemns* hate
("Das ist antisemitisch.") from one who spreads it. So rows are judged in
windows:

- a **core** of about `HATE_SPEECH_WINDOW_TOKENS` (default `1000`, docint's
  fine-chunk scale) of consecutive rows that the model labels, one request per
  window;
- read-only **context** of about `HATE_SPEECH_CONTEXT_TOKENS` (default `300`)
  on each side. This lets the model resolve who is speaking, what is being
  answered, and what "die" / "they" refers to. `0` disables the context.

The model names the targeted group and the speaker's stance for every
candidate row. Only rows whose speaker *endorses* group-focused enmity are
reported; quoting, reporting, condemning, analysing, or asking about hate is
not hate.

It judges the original wording. When the job translated, the translation is
shown as an aid and carried in the finding, next to the speaker.

Requests use a JSON-schema `response_format`. A provider that rejects it, or
whose constrained reply cannot be parsed, is served unconstrained for the rest
of the sweep. A context overflow halves both budgets and retries.

A reply cut off at the output cap (`finish_reason: length`) is not trusted
either: the core is halved and the window asked again. The cap grows with the
core, at 80 tokens per row and at least 1024. A single-row window keeps
whatever complete items its reply held.

Invalid values warn and fall back. Resolved by `load_hate_speech_env`. The
prompt is `nextext/utils/prompts/<code>/hate_speech_transcript.txt`; docint
keeps a byte-identical copy for its own transcript windows.

**What a video shows.** A picture whose hate is purely visual leaves nothing in
the transcript. So when a job captioned its keyframes, the stage also judges
each caption, one request per caption, with
`nextext/utils/prompts/<code>/hate_speech_image.txt`. That file is a
byte-identical copy of docint's chunk prompt (both repos pin its hash), whose
image rule judges the message a picture conveys: a hate symbol shown with no
distancing is a finding. The caption is labelled `Image description:`
(`Bildbeschreibung:`), as docint labels its images.

- Findings carry `source: frame`, the caption as `text` and the frame's time as
  `start`; transcript findings carry `source: transcript`.
- A silent clip is judged from its captions alone.
- No vision request is added: the stage reads only the captions the keyframe
  step already wrote, so a job without captions judges no frames.
- An unconstrained reply that lists one verdict per statement is read as the
  whole caption: it is a finding when any listed verdict endorses.

**Ollama:** a window prompt is roughly 3k tokens. Give the model a context of
at least 8k tokens (`OLLAMA_CONTEXT_LENGTH` on the Ollama server), or Ollama
silently drops the start of the prompt.

The dev-only harness `eval/hate_speech/` measures the classifier against
labelled synthetic transcripts. It is not shipped; see its README.

## Video keyframes

Keyframe extraction is a step of its own, requested per job
(`JobOptions.keyframes`, the SPA's **Keyframes** checkbox, or the CLI's
`-kf/--keyframes`) and **off by default**. When it is off no frames are
sampled at all — there is no `keyframes.zip` and no visual context. It is
independent of summarization in both directions: a summary no longer samples
frames, and frames no longer need a summary to be described.

Describing those frames is a second, separate request-level switch,
`JobOptions.visual_context` (**on by default**, API-only — the SPA has no
control for it). Turning it off keeps the sampling and skips the captions, for
a client that wants the JPEGs and not the words; see
[Visual context](#visual-context-keyframe-descriptions) below.

Both knobs below supply a **default** that applies only when a job-creation
request omits the corresponding field — an explicit per-request value always
wins — and they bite only when the job asked for keyframes. Resolved by
`load_keyframe_defaults` in `nextext/utils/env_cfg.py`; the CLI reads them too.

- `KEYFRAMES_PER_MINUTE` (backend only) — Default keyframes sampled per minute
  of video, applied to `JobOptions.keyframes_per_minute` only when a
  job-creation request omits the field. Invalid values warn and fall back;
  negatives clamp to `0`. Defaults to `4`.
- `KEYFRAMES_MAX` (backend only) — Default hard ceiling on keyframes returned
  per clip, applied to `JobOptions.keyframes_max` only when a request omits it.
  Clamped to `[0, 200]` (the schema's hard cap; larger values warn and clamp to
  `200`). Defaults to `20`.

## Visual context (keyframe descriptions)

Within the keyframe step, each sampled frame is described by `TEXT_MODEL` and
the timestamped captions become a result in their own right: they surface on
`GET /jobs/{id}` as `frame_captions`, as the `visual_context.txt` artifact,
and in the SPA's **Visual context** tab. A job that *also* asked for a summary
has them folded into it, so the summary covers what was shown as well as what
was said — and for a video whose audio held no speech, the captions are the
summary's only source. A job that asked for hate-speech detection has them
judged as well ([Hate-speech detection](#hate-speech-detection)); the caption
prompt asks for recognisable symbols, emblems, flags, gestures and codes by
name, so what a frame shows reaches that pass in words.

This needs a **vision-capable** `TEXT_MODEL` (the shared `vllm-service` chat
endpoint serves one). Captioning is fail-soft end to end: a per-frame outage
costs that caption, and a model that rejects image input aborts after a single
request, leaving an uncaptioned run plus one warning — the sampled frames are
still downloadable as `keyframes.zip`. Each captioned frame costs one
inference request, which is what the frame budget below bounds. Resolved by
`load_visual_summary_env` in `nextext/utils/env_cfg.py`; `nextext-cli` honours
the same settings.

- `NEXTEXT_VISUAL_SUMMARY` (backend + CLI) — Operator kill-switch for the
  captioning half of the keyframe step; sampling is unaffected. Only an
  explicit falsy token (`0`/`false`/`no`/`off`) disables captioning;
  unrecognised values warn and keep the default. Defaults to on. It is the
  operator half of a two-part gate: a job may also opt out on its own with
  `visual_context: false`, which is how an API client asks for the frames
  without the words (docint runs its own image pipeline over them and never
  reads the captions). Either gate off skips the provider entirely — which
  also means such a job cannot be failed by an unreachable chat model, since
  the keyframe stage runs first and is otherwise the one that resolves it.
- `VISUAL_SUMMARY_MAX_FRAMES` (backend + CLI) — Maximum frames captioned per
  job; a longer clip's frames are subsampled evenly so coverage still spans the
  whole file. Clamped to `200` (the keyframe ceiling); non-integer or
  non-positive values warn and fall back. Defaults to `12`.
- `VISUAL_SUMMARY_IMAGE_MAX_SIDE` (backend + CLI) — Longest edge in pixels each
  frame is downscaled to before upload, bounding request size and image-token
  count. Frames already within budget are not upscaled. Defaults to `1024`.

## Faces

A job that asks for faces (`JobOptions.faces`, the SPA's **Faces** toggle —
shown only while the switch below is on) gets the faces of its video: an
overview of who is on screen when, grouped per person in `group` mode, and
each one's best shot to pass on. How the stage works is in
[architecture.md](architecture.md#faces); what it records in
[audit.md](audit.md). There is no CLI option, and `-F` never includes faces.

`NEXTEXT_FACES` (backend only) is the operator switch, read once at startup,
recorded in `config.audit` and reported by `GET /config` as `faces`:

| Value | Effect |
|---|---|
| `off` (default; also an empty value or `0`/`false`/`no`) | A job asking for faces is refused with `422` before anything is stored |
| `detect` | Faces are found, tracked and cut out; no biometric processing. Entries read "Face N" |
| `group` | Also grouped per person by the faces service — **biometric processing**. The controller extends the data-protection impact assessment before switching it on (§ 67 BDSG, with the DPO involved); the software enforces nothing. Entries the service grouped read "Person N" |

An unknown value warns and stays `off`. With `detect` or `group` set but no
faces endpoint resolving, startup logs one warning naming `NEXTEXT_FACES` and
`FACES_API_BASE`; such jobs then report their faces as unavailable.

- `FACES_API_BASE` (backend only) — the faces service root; the client appends
  `/faces/detect` and `/faces/group`. Unset: the central `OPENAI_API_BASE` with
  one trailing `/v1` stripped. An off token (`off`/`false`/`no`/`0`) leaves no
  endpoint. The bearer token is `OPENAI_API_KEY`.
- `FACES_TIMEOUT` (backend only) — seconds per request. Defaults to `300`;
  values that are not positive and finite warn and fall back. Read once per
  video.
- `FACES_INTERVAL_S` (backend only) — seconds between sampled frames.
  Defaults to `1.0`; a clip longer than `FACES_MAX_FRAMES` intervals stretches
  it so the samples span the whole clip. Values that are not positive and
  finite warn and fall back.
- `FACES_MAX_FRAMES` (backend only) — most frames sampled per video. Defaults
  to `600`, clamped to `3600`.

The stage never fails a job. A failed batch leaves its frames without faces
(`partial`); a `503` whose `Retry-After` asks for at most 10 s is waited out and
that batch sent once more; after three failed batches in a row the rest of the
video is neither decoded nor sent, so a hung service costs at most three
timeouts. No answered batch gives `unavailable`. A grouping request over the
16 MiB budget is retried with one crop per track, and left ungrouped only when
even that is too large.

Boxes in `faces.csv` and the `manifest.json` of `faces.zip` are in the source
video's own pixels (upright), also for a source larger than the 3840 px the
frames are sampled at. `last_seen_seconds` is the time of the entry's last
detection; an appearance runs to the next sample after it, at most to the
clip's end. `faces.zip`, `faces.csv` and `face_NNN.jpg` answer `404` for a
result without faces (none found, or the service unavailable), and the SPA
then offers no download.

Tests start without `NEXTEXT_FACES` or any `FACES_*` variable: an autouse
fixture in `tests/conftest.py` clears them, so a developer `.env` cannot switch
the stage on for the suite.

## Production sub-path

The Nextext SPA is served in production under the canonical `/nextext/`
sub-path behind the `edge-plane` gateway, not at its own vhost root. The
`frontend` service joins the external `edge-net` network (alongside its
existing `nextext-net` membership) as alias `nextext-frontend`, which is how
the gateway reaches it. Vite is built with `base: '/nextext/'`, `API_BASE`
derives from `BASE_URL` (`frontend/src/api/client.ts`'s `apiBase()`), the
`BrowserRouter` uses a matching `basename`, and the frontend's nginx config
strips the `/nextext` prefix internally before falling through to the
existing root-anchored locations (the SSE job-events endpoint included),
redirecting bare `/` to `/nextext/`.

The gateway is the sole production entry point and is what injects
`X-Auth-User` — and `X-Edge-Secret` — for the backend's trusted-header
principal seam; see [Identity](#identity).

## Identity

Each request resolves to one principal, which scopes that caller's in-memory
jobs (`nextext/api/identity.py`):

1. **Service token.** `Authorization: Bearer <token>` matching an entry of
   `NEXTEXT_SERVICE_TOKENS` resolves to the service principal
   `service:<name>`. The identity header on such a request is ignored: a
   service acts only as itself, never as a user.
2. **Trusted header** (`NEXTEXT_AUTH_HEADER`, default `X-Auth-User`), which the
   edge gateway sets to the signed-in user. With `NEXTEXT_EDGE_SECRET` set, it
   counts only on a request whose `X-Edge-Secret` header matches the secret;
   on any other request it is ignored as if absent, and a warning records the
   method and path — never a header value. A value starting with `service:` is
   ignored the same way on every request.
3. **`NEXTEXT_DEFAULT_IDENTITY`**, the dev / header-less fallback, honoured
   only while `NEXTEXT_EDGE_SECRET` is unset. With both set it is ignored —
   startup logs one warning naming both variables — so a request with neither
   a service token nor an edge-verified header gets `401`. It may not start
   with `service:` either; a value that does stops the backend at startup.
   Production leaves it unset, so a request with none of the above gets
   `401`.

The `service:` namespace keeps services and users apart: a user called
`docint` is a different principal from the docint service (`service:docint`)
and cannot list or read its jobs, which hold other users' media and
transcripts.

`NEXTEXT_AUTH_HEADER` is read by the backend only. The SPA
(`frontend/src/api/client.ts`) and the edge gateway always send `X-Auth-User`,
so another name suits only other callers. `GET /jobs/{id}/media` is
authorized by its per-job token, not by the principal, and is unaffected by
all of this.

### Edge secret

`NEXTEXT_EDGE_SECRET` is shared with the edge gateway, which sends it as
`X-Edge-Secret` on every request it proxies; the frontend's nginx passes it
through to the backend. Generate it with `make secret` in edge-plane and set
the same value, under the same name, in both repos' `.env`; then run `make up`
in each, since a plain restart keeps the old value. A value shorter than 32
characters stops the backend at startup; the error never shows the value.

Without the secret, the trusted header is believed from any caller that can
reach the backend — anything on `inference-net`, or anything that can reach
the frontend — and startup logs a warning saying so, unless
`NEXTEXT_DEFAULT_IDENTITY` marks a dev setup. With it set, only the gateway
can assert a user: a request that reaches the frontend or the backend some
other way (the SPA under `make up-dev`, say) carries no usable identity and
gets `401` — `NEXTEXT_DEFAULT_IDENTITY` does not apply while the secret is
set, or it would authenticate every such request. Service tokens work the
same either way. The gateway's side of the contract is edge-plane's
`docs/identity-contract.md`.

| `NEXTEXT_EDGE_SECRET` | `NEXTEXT_DEFAULT_IDENTITY` | A request without a token or a trusted header |
|---|---|---|
| unset | unset | `401` |
| unset | set | the default identity (dev) |
| set | unset | `401` |
| set | set | `401`; startup warns that the default identity is ignored |

### Service tokens

`NEXTEXT_SERVICE_TOKENS` lists the services that call the API directly, as
comma-separated `name=token` pairs:

```bash
NEXTEXT_SERVICE_TOKENS=docint=<token>
```

Names use letters, digits, `.`, `_` and `-` and are unique; tokens are at
least 32 characters, unique, and never equal to the edge secret. Anything
else stops the backend at startup, with an error that names the entry by its
position and never echoes a name or token. Generate each token with
`make secret` in edge-plane.

docint sends its `NEXTEXT_API_KEY` as the bearer token, so its entry carries
that value, and its calls then resolve to `service:docint`. A bearer token
that matches no entry is not an error: the request continues with the
trusted header. Tokens are compared in constant time and never logged.

### Rolling out

Each step must not lock out a caller that is still on the old path:

1. **Service token first.** Set the same token as docint's `NEXTEXT_API_KEY`
   and in Nextext's `NEXTEXT_SERVICE_TOKENS`, then recreate both. docint's
   calls keep working, but their principal changes from `docint` to
   `service:docint`, and a job docint started under the old principal is no
   longer visible to it. Recreating the backend drops in-memory jobs anyway,
   so do this while docint has no transcription in flight.
2. **Gateway before backend.** Set `NEXTEXT_EDGE_SECRET` in edge-plane's
   `.env` and recreate the gateway, so it already sends the secret; then set
   it in Nextext's `.env` and recreate the backend. In the reverse order,
   every request through the gateway gets `401` until the gateway catches up.
   A full `deploy` bring-up starts the gateway last, so it has no such
   window.

Without a matching token, docint's calls fall back to the trusted header,
which the backend ignores once the edge secret is set — they then get `401`.

A service states its purpose like any caller: under the deployment default
`AUDIT_REQUIRE_PURPOSE=query,disclose`, docint's reads of a finished job's
snapshot and of its artifacts carry `X-Audit-Purpose` — the purpose of the
docint job that started the ingest. A docint release that does not send it
yet gets `428` on those reads, so upgrade docint before Nextext, or switch the
requirement off until then (see [audit.md](audit.md#purpose)).

## Data-protection audit log

The backend and every `nextext-cli` run record who collected, queried,
downloaded, combined or deleted which job, and every transfer of job content
to a model, as hash-chained JSON lines on stdout — see
[audit.md](audit.md) for what each endpoint, stage and CLI run records. The
store, its retention and who may read it are obs-plane's.

| Variable | Default | Meaning |
|---|---|---|
| `AUDIT_OUTPUT` | `stdout` | `container` writes to the container's main stdout under a container-wide lock, so a `make cli` run lands in the backend's log; `docker/compose.yaml` sets it for the backend |
| `AUDIT_REQUIRE_PURPOSE` | `query,disclose` (`docker/compose.yaml`, while unset); empty outside compose | Comma-separated actions that need a stated purpose (`X-Audit-Purpose`, which the SPA sends from its header's purpose bar; `AUDIT_CASE_REF` / `AUDIT_REASON` for the CLI). Without one a request gets `428` and a CLI run exits `2`; an unknown action stops startup. Set it empty (`AUDIT_REQUIRE_PURPOSE=`) to switch the requirement off |
| `AUDIT_CLI_USER`, `AUDIT_CASE_REF`, `AUDIT_REASON` | unset | CLI only, set per run by `make cli`: the operator and the stated purpose |
| `NEXTEXT_MEDIA_GRANT_TTL_S` | `900` | Seconds a media playback grant stays usable (at most `86400`); invalid values fall back to the default |
| `NEXTEXT_CLI_RETENTION_DAYS` | `365` | Days after which what `make cli` left in `/tmp/nextext-cli` (and `/tmp/output` from earlier releases) is deleted by the backend, recorded; `off` keeps it (development only). Anything else stops startup |
| `RETENTION_NOTICE_DAYS` | `30` | Days a new `NEXTEXT_CLI_RETENTION_DAYS` deletes nothing after it took effect, and a shortened one spares what it makes due sooner (1–365). Changing it alone moves no date. |

Under the deployment default, listing jobs, reading a finished job's result,
playing a recording, every download and every CLI run need a stated purpose;
uploads and deletes do not ([audit.md](audit.md#purpose) has the full list).
To see what the backend gets:

```bash
docker compose --env-file .env -f docker/compose.yaml config backend | grep AUDIT_REQUIRE_PURPOSE
#       AUDIT_REQUIRE_PURPOSE: query,disclose   (unset in .env)
#       AUDIT_REQUIRE_PURPOSE: ""               (AUDIT_REQUIRE_PURPOSE= in .env: off)
```

A recording plays through a short-lived grant that carries the purpose,
because a media element cannot send the header
([audit.md](audit.md#playback)).

A failed audit write fails the operation: a request answers the generic `500`
without its data, a job fails, a CLI run exits non-zero.

## Metrics

`GET /metrics` (app root, not under `/api/v1`) serves a Prometheus exposition
of aggregate request/latency counters only — no transcript or user data. It is
unauthenticated by design: the `obs-plane` scraper, like every inference
caller, reaches the backend over `inference-net`.

Alongside the HTTP series from `prometheus_fastapi_instrumentator`, the backend
publishes job-outcome counters (`nextext/api/metrics.py`). Their labels are
typed codes only — never a filename, owner id, or transcript text:

| Metric | Labels | What it counts |
|--------|--------|----------------|
| `nextext_jobs_total` | `outcome` = `completed` \| `skipped` \| `failed` | Every job that reached a terminal state. |
| `nextext_jobs_skipped_total` | `reason` = `vad_no_speech` \| `asr_empty_transcript` \| `asr_all_segments_filtered` \| `unknown` | Jobs that completed without a transcript. |
| `nextext_jobs_failed_total` | `code` = `undecodable_media` \| `internal` | Failed jobs, by typed cause. |

A rising `nextext_jobs_skipped_total{reason="vad_no_speech"}` usually means the
`/vad` endpoint is over-reporting silence; a rising
`nextext_jobs_failed_total{code="undecodable_media"}` means users are uploading
containers PyAV cannot decode. Both are visible without reading logs.

## Logs

The backend logs to stderr for the container logging driver — its stdout
carries the [audit log](audit.md) alone, so uvicorn's access log, which its
default configuration sends to stdout, is moved to stderr at startup. The
frontend's nginx logs to stdout/stderr. Those operational logs carry no
upstream error bodies or transcript text, and the access logs no query
strings:

- A failed call to an inference endpoint — Whisper, `/vad`, `/diarize`,
  `/gliner`, or the chat model — is logged by exception class and HTTP status,
  never by its message or response body: a rejection can quote the request it
  rejected.
- An unexpected job or request failure is logged by exception class and
  traceback frames, without the message.
- The backend's uvicorn access log and the frontend's nginx access log record
  the request path without its query string, which can carry the per-job
  media token or a playback grant (see
  [architecture.md](architecture.md#playback)). An nginx
  access line reads `time "METHOD path protocol" status bytes request-time
  "X-Request-Id"` — no referer or client address.

nginx's error log has a fixed format: an error logged for a request — the
backend unreachable or timing out, say — still carries the full request line,
query string included, with the client address and referer.
