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

The NER, diarization, and VAD services speak a plain service root rather than the OpenAI `/v1` shape, so the central fallback strips one trailing `/v1` from `OPENAI_API_BASE` before appending the service path (`http://vllm-router:4000/v1` → `http://vllm-router:4000/gliner`). Whisper, which speaks `/v1`, uses `OPENAI_API_BASE` verbatim. The three non-OpenAI services reuse `OPENAI_API_KEY` as their bearer token; none takes a dedicated key.

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
summary's only source.

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
`X-Auth-User` for the backend's trusted-header principal seam
(`nextext/api/identity.py`) — production leaves `NEXTEXT_DEFAULT_IDENTITY`
unset so requests without that header are rejected as unauthenticated; any
dev-only fallback stays dev-only. `NEXTEXT_AUTH_HEADER` (backend + frontend,
default `X-Auth-User`) names the header; both sides read the same variable so
they agree on it.

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
