# Data-protection audit log

Nextext records its processing of personal data — recordings, their
transcripts, and everything derived from them — in a data-protection audit
log along the categories of §76 BDSG: collection, query, disclosure (including
every transfer to a model), combination and erasure, each with the person, the
time, the stated purpose and the recipient.

Every event is one hash-chained JSON line, written by the federation's audit
library (`nextext/utils/audit_event.py` and `nextext/utils/audit_runtime.py`,
vendored verbatim from `nos-tromo/.github` and drift-checked in CI). Each
process writes its own chain: a `start` event, then its events, a heartbeat
every 60 s, and a `shutdown` event. The log collector routes the lines to the
audit store; the store, its retention, who may read it and how the chains are
verified are obs-plane's (`docs/audit.md` there) and are not repeated here.

Events carry **references, counts and parameters — never content**: no media,
transcript, translation, summary, caption, prompt, model reply, entity,
hate-speech finding, file name or path, and never the media token or a
playback grant. Where a model produced text, its SHA-256 is recorded
(`ai.output_sha256`), so a later copy of the result can be matched without the
log holding it.

## Requests

The backend writes one event per audited request, once the data is fetched and
before the response is returned (`nextext/api/audit.py`). Every such event
carries the actor, the stated purpose and the gateway's `X-Request-ID` — for a
stream played with a grant, those of the request the grant was issued to.

| Endpoint | Action / operation | Target | Recorded |
|---|---|---|---|
| `POST /api/v1/jobs` | `collect` / `job.create` | the job | `files`, `bytes`, `source_sha256` of the upload, the `analyses` asked for and the job `options`. An upload rejected for its size (`413`) or failing while it is stored (a full disk, say) is removed and recorded with `outcome="error"` and a typed `failure`. A job asking for `faces` while `NEXTEXT_FACES` is `off` is refused with `422` before anything is stored, and nothing is collected |
| `GET /api/v1/jobs` | `query` / `job.list` | every job listed | — |
| `GET /api/v1/jobs/{id}` | `query` / `job.get` | the job | only once the job has finished and its snapshot carries the result; until then the snapshot is a status poll (status, progress, the caller's own options) and is not recorded. The face thumbnails travel inside this snapshot as `data:` URIs, so looking at the faces is this one query, not a disclosure per image |
| `DELETE /api/v1/jobs/{id}` | `delete` / `job.delete` | the job | the job's `status` when it was deleted. The SPA's "Clear" actions send one such request per job |
| `GET /api/v1/jobs/{id}/artifacts/{name}` | `disclose` / `artifact.download` | the job | `recipient="download:<format>"` (`csv`, `xlsx`, `txt`, `png`, `jsonl`, `zip`, `jpg`), the `artifact` name. Each face's best image (`face_NNN.jpg`, recipient `download:jpg`) is a download, and so a disclosure, of its own; so are `faces.zip` and `faces.csv` |
| `GET /api/v1/jobs/batch/{name}` | `disclose` / `artifact.download_batch` | the completed jobs the bundle was built from | as above |
| `GET /api/v1/jobs/{id}/media` | `query` / `media.stream` | the job | one event per request that streams bytes — a player fetches several ranges — with `ranged`. Played with a grant (`?grant=`, see [Playback](#playback)), the event carries the purpose and request id of the request the grant was issued to, and the grant's id as `grant` — never the grant itself |
| `POST /api/v1/jobs/{id}/media-grant` | `query` / `media.grant` | the job | refusals only: another owner's job, or a purpose `AUDIT_REQUIRE_PURPOSE` asks for and the request lacks (`428`) |

An attempt on another owner's job on `job.get`, `job.delete`,
`artifact.download` or `media.grant` still answers `404` (existence never
leaks), but is recorded with `outcome="denied"`, naming the caller and the
job.

Not recorded, because they carry no personal data or touch none:

| Endpoint or case | Why |
|---|---|
| `GET /api/v1/health`, `/version`, `/config`, `/languages`; `GET /metrics` | health, configuration, static mappings, aggregate counters |
| `GET /api/v1/whoami` | echoes the caller's own identity |
| `GET /api/v1/jobs/events`, `GET /api/v1/jobs/{id}/events` | progress streams: stage names, progress, counts and typed codes |
| `POST /api/v1/jobs/{id}/media-grant` answered `201` | the grant hands out no personal data; each stream it opens is recorded when it is played |
| a media request with an unknown or expired grant, or one issued for another job (`404`) | it names no one |
| a request answered `401` | it has no actor; recording it would let anyone on the networks flood the trail |
| `404` for an unknown job, an unsupported or unproduced artifact; `409` for an unfinished job | no data was touched |
| `422` for invalid job options, including `faces` while `NEXTEXT_FACES` is `off` | refused before the upload is stored |

When more than 200 jobs are named, the first event holds 200 ids and counts
all of them; each further chunk follows as `<operation>.ids` with the first
event as `parent_event_id`.

## A job's own events

A job runs in the background, after its request has returned. Its events carry
the actor, purpose and request id of the request that created it, target the
job, and name the job's `collect` event as `parent_event_id`.

**Model calls.** Every stage that sends job content to a model writes one
`ai_call` event per inference task when it ends — also when it fails — built
from a tally every inference client reports into
(`nextext/core/inference_trace.py`). The event counts the `requests` sent and
any `failed_requests`; it is `ok` when at least one request returned a usable
reply, `error` when none did. Token counts are recorded when every reply
reported them, and `ai.model_id` when the endpoint named the model it served.
`ai.sources` names the events whose output the model was given.

| Stage | Operation | Recipient | `ai.model_alias` | `ai.output_sha256` over | `ai.sources` |
|---|---|---|---|---|---|
| Speech guard | `ai.vad` | `inference:vad` | `vad` | — | the upload |
| Transcription | `ai.transcribe` | `inference:asr` | `WHISPER_MODEL` (or the provider's default) | the finished transcript's `text` column, one row per line | the upload |
| Diarization | `ai.diarize` | `inference:diarize` | `diarize` | — | the upload |
| Sentence restoration | `ai.sentence_restore` | `inference:chat` | `TEXT_MODEL` | — (the model returns boundaries, not text) | transcription, diarization |
| Keyframe captions | `ai.caption_frames` | `inference:chat` | `TEXT_MODEL` | the `visual_context.txt` block | the upload |
| Translation | `ai.translate` | `inference:translate` | `TEXT_MODEL` | the `translation` column, one row per line | transcription, restoration |
| Named entities | `ai.ner` | `inference:ner` | `gliner` | — | transcription, restoration, translation |
| Summary | `ai.summarize` | `inference:chat` | `TEXT_MODEL` | the summary (`summary.txt`) | the transcript or translation, the captions |
| Hate-speech detection | `ai.hate_speech` | `inference:chat` | `TEXT_MODEL` | never | the transcript, diarization, the captions |
| Face detection | `ai.detect_faces` | `inference:faces` | `faces` | never | the upload |
| Face grouping (`NEXTEXT_FACES=group`) | `ai.group_faces` | `inference:faces` | `faces` | never | the upload, face detection |

The `/vad`, `/diarize`, `/gliner` and `/faces` services take no model field,
so their alias is the endpoint's name. The `upload` is the job's `collect` event. A
stage that sent nothing (the speech guard switched off, diarization not asked
for) writes no event.

Hate-speech categories can amount to special categories of personal data, so
that stage is recorded by what it judged — `segments` and `frames` — and never
by what it found: no categories, no counts of findings, no hash of them.

The face stage is recorded the same way. `ai.detect_faces` counts the
`frames` sent for detection (frames of a batch that failed included — what was
transferred); `ai.group_faces` counts the `tracks` sent for grouping, in
`group` mode only. Neither records what was found: no number of faces or
persons, no boxes, no hash. A busy service (`503` with a short `Retry-After`)
is retried once, and both requests count. Grouping takes face crops and
returns labels only; no biometric feature reaches Nextext or this log.

**Combination.** When a finished job's result joins two or more analyses —
speaker labels from diarization onto the transcript, named entities, the
hate-speech classification, captions, the faces, a summary — one `combine` /
`job.combine` event names them (`analyses`) and says whether the results carry
speaker labels (`speaker_attributed`). Hate-speech detection is listed whenever
it ran, whatever it found; so is `faces`, whenever the face stage produced a
result (even one without faces, or one the service never answered). After an
unexpected error in the face stage there is no result and nothing of it was
joined, so `faces` is not listed — its `ai_call` events still are.

**Erasure.** Besides `job.delete` by the owner, the system erases jobs in two
places, each as one `delete` event per owner, by `Actor(None, "system")`, with
the owner as the target's `container` (`owner:<id>`):

- `job.purge` (`trigger="shutdown"`): a clean shutdown removes every remaining
  job and its upload.
- `job.sweep_orphans` (`trigger="startup"`): a backend that died without
  shutting down leaves uploads on the `tmp-jobs` volume that nothing in memory
  points at; the next startup removes them before it serves a request.

What `make cli` runs leave in `/tmp/nextext-cli` (and `/tmp/output` from
earlier releases) is erased too, recorded *before* it goes, naming each entry
as `path-sha256:<hex>` of its path (a file name can name a person), with
`kind: cli_file`:

- `cli.sweep` (by the system, `period_days`): at startup and then daily, what
  is older than `NEXTEXT_CLI_RETENTION_DAYS`, by the rule in
  [cli.md](cli.md) (a new period's notice, `RETENTION_NOTICE_DAYS`, first);
- `cli.clean` (by the operator, via `cli`, with any stated purpose):
  `make cli-clean`.

An entry that could not be removed is recorded again with `outcome: error`.

## The owner record

Jobs live in memory, so a restart would lose whose upload a leftover directory
held. Each job directory therefore holds a small owner record, `job.json` —
the job id, the owner, when it was created — written atomically (temporary
file, flushed, renamed) when the job is created and before its `collect`
event; if it cannot be written, the job is not created. The startup sweep reads
it to name each leftover job and its owner (`nextext/api/job_record.py`).

A directory without a valid record — an upload interrupted before its job
existed, or one left by a release that wrote no records — is never served
(nothing in memory points at it) and is removed by the same sweep; its
deletion is recorded as `target.kind="upload"` with a count, no job id and no
owner, and `owner_recorded: false`. The audio re-encode of a running job lives
in its job directory too, so the sweep removes it with the upload.

The sweep runs only when the backend holds the job directory's lock; a second
process on the same directory sweeps nothing.

## Actor

| Caller | `actor` |
|---|---|
| The trusted identity header (edge-verified when `NEXTEXT_EDGE_SECRET` is set) | `{"user": <X-Auth-User>, "via": "edge"}` |
| `NEXTEXT_DEFAULT_IDENTITY`, honoured only while no edge secret is set | `{"user": <default identity>, "via": "system"}` |
| A service token (`NEXTEXT_SERVICE_TOKENS`) | `{"user": "service:<name>", "via": "service"}` |
| A CLI run | `{"user": AUDIT_CLI_USER or null, "via": "cli"}` |
| Startup sweep, shutdown purge | `{"user": null, "via": "system"}` |

The media route is authorized by what its URL carries — a playback grant or
the job's own token — not by the caller. A request that resolves to an
identity other than the job's owner is recorded with `on_behalf_of: <owner>`.
With `NEXTEXT_EDGE_SECRET` set, a media request without a verified identity
did not come through the gateway and is refused (`404`, not recorded), and a
grant plays only for the identity it was issued to: another verified identity
is refused (`404`) and recorded with `outcome="denied"`. Without the secret —
where every identity is unverified anyway — a header-less `<video>` request is
attributed to whom the grant or token was issued: the grant's requester, or
the job's creator.

## Purpose

A caller states its purpose in the `X-Audit-Purpose` header: the URL-encoded
form of up to two fields, `case_ref` and `reason` — what
`new URLSearchParams({case_ref, reason}).toString()` produces. It is recorded
on the request's event, on every event of a job the request created, and on
every stream played with a grant issued to it. A malformed value (repeated or
oversized fields, control characters, non-ASCII, not form-encoded) is answered
`400` with a generic message and nothing is recorded. The purpose goes into
the audit log only, never into the operational logs.

`AUDIT_REQUIRE_PURPOSE` lists the actions that need a stated purpose. A request
whose action is listed but states none is refused with `428`, and the refusal
is recorded with `outcome="denied"`. Creating a job counts as `collect`,
`ai_call` and `combine`, since the job will call the models and join their
results; a job snapshot needs the purpose only once it carries the result.
An unknown action stops the backend at startup.

A deployment requires `query,disclose`: `docker/compose.yaml` sets that
default while the variable is unset, since §76 BDSG Abs. 2 asks the reason for
queries and disclosures. Under it:

| Needs a stated purpose | Needs none |
|---|---|
| listing jobs; a finished job's result; a playback grant; every download, the word-cloud image included; every CLI run, since it writes output files | uploading (`collect`); deleting; polling an unfinished job; the progress streams; the configuration and identity endpoints |

Playing with the job's own token (`?token=`) is refused while `query` is
listed, since a media element states no purpose; the SPA plays with a grant
instead ([Playback](#playback)). `AUDIT_REQUIRE_PURPOSE=` (empty) in `.env`
switches the requirement off; outside compose — bare `uvicorn`, the tests — the
list is empty unless set. `GET /api/v1/config` reports the parsed list as
`require_purpose`, so the SPA knows a purpose is required.

Machine clients state a purpose like anyone else. A service that calls with
its own token (`NEXTEXT_SERVICE_TOKENS`) sends `X-Audit-Purpose` on its
`query` and `disclose` requests, and under the default is refused (`428`)
without one. docint's media ingest is the example: creating the job needs no
purpose (`collect`), but reading the finished job's snapshot and downloading
its `docint.jsonl` and `keyframes.zip` do, and docint states the purpose of the
docint job that started the ingest — or `reason="docint ingest"` when no
request did.

## The purpose in the SPA

The SPA's header carries a purpose bar (`@infra/ui`'s `PurposeBar`, scope
`nextext`), where the user states a case reference, a reason, or both, once
per browser tab. Every request the SPA makes goes through one function
(`apiRequest` in `frontend/src/api/client.ts`), which sends the stated purpose
as `X-Audit-Purpose`; when the backend answers `428`, the bar's dialog opens
and the request is sent once more with what the user saved there. A declined
prompt leaves the request refused, with its own message ("This action needs a
stated purpose…") rather than a generic error. A refused query — the job list,
a result — loads by itself once a purpose is stated; a refused download stays
refused until it is clicked again; the player says that playing needs a
purpose, and starts once one is stated. No request times out while the dialog
is open. While `require_purpose` is non-empty, the empty bar is a warning and
a blank purpose cannot be saved. Signing out through the header's account
menu clears it.

Every request to an audited endpoint, and how it carries the purpose:

| What the SPA does | Endpoint | Recorded as | Carries the purpose |
|---|---|---|---|
| list the jobs, on load and after a change | `GET /jobs` | `query` / `job.list` | header, through the API client |
| show a finished job's result | `GET /jobs/{id}` | `query` / `job.get` | header, through the API client |
| upload a file | `POST /jobs` | `collect` / `job.create` | header, through the API client (not required by default) |
| delete a job, "Clear" | `DELETE /jobs/{id}` | `delete` / `job.delete` | header, through the API client (not required by default) |
| download an artifact | `GET /jobs/{id}/artifacts/{name}` | `disclose` / `artifact.download` | header: fetched through the API client and saved from a blob, never an `<a href>` to the backend |
| show the word cloud | `GET /jobs/{id}/artifacts/wordcloud.png` | `disclose` / `artifact.download` | header: fetched through the API client; the `<img>` shows a blob URL |
| download all jobs | `GET /jobs/batch/{name}` | `disclose` / `artifact.download_batch` | header: fetched and saved from a blob, as above |
| open the player | `POST /jobs/{id}/media-grant` | refusals only | header, through the API client |
| play a recording | `GET /jobs/{id}/media?grant=…` | `query` / `media.stream` | the grant: the `<video>`/`<audio>` element sends no header |

The progress streams are read with `fetch`, not `EventSource`, through the same
client. Nothing else reaches the backend around it: there is no `<a href>`,
`<img src>`, `window.open` or form post to the backend, and the media element
is the one request the browser makes on its own — which is why it plays a
grant.

## Playback

A `<video>`/`<audio>` element sends neither the identity header nor
`X-Audit-Purpose`. Before the player sets `src`, the SPA therefore asks for a
playback grant, `POST /api/v1/jobs/{id}/media-grant`: an ordinary request,
owner-scoped like the snapshot, which states the purpose and needs the one a
query needs. It returns a media URL carrying a fresh random token
(`secrets.token_urlsafe(32)`), bound in the backend's memory to the job and to
that request's caller, purpose and request id, until it expires
(`NEXTEXT_MEDIA_GRANT_TTL_S`, 900 s by default). Every range request of the
playback presents it, and each is recorded as `query` / `media.stream` with
the grant request's purpose and request id and the grant's own random id in
`criteria.grant`, which ties the ranges of one playback together.

When a grant expires mid-playback, the next range request is refused (`404`,
not recorded), the element fails, and the SPA asks for a fresh grant once and
picks playback up where it stopped; an error soon after a fresh grant is the
file's and is reported as unplayable. Grants live in memory like the jobs:
deleting a job revokes its grants, a restart ends them all, and a job keeps
at most 16. Only a hash of each token is kept.

The grant rides in the query string, like the token: the backend's and the
frontend's access logs drop the query string, and obs-plane redacts every
query value before operational logs are stored; nginx's error log, whose
format is fixed, still carries the request line when it logs an error for
the request. It never appears in an event.

## Correlation

The gateway's `X-Request-ID` is recorded as `request_id` on every request
event and on the job events it caused — and on the streams a grant opened, as
the request id of the grant's request — and is forwarded as `X-Request-ID` on
every inference request the job makes — Whisper, `/vad`, `/diarize`,
`/gliner` and the chat model — so the router's logs can be matched with the
trail. No identity and no purpose are sent to the router.

Chat calls also carry `X-Cache-Scope`, a pseudonym of the job's owner: an
HMAC-SHA256 of the owner id under a key the backend draws at startup and never
lets out (`nextext/core/inference_trace.py`). vllm-service gives each scope
its own share of the chat model's prefix cache, so one owner cannot time
another owner's cached prompts. Outside the process a scope cannot be mapped
back to an owner, and a restart draws new scopes. The other inference requests
do not carry it, and CLI runs send none. Jobs docint starts run under its
service owner (`service:docint`), so they share one scope; their prompts are
the pipeline's own, not ones a user writes.

## Command line

Every `nextext-cli` run writes a chain of its own, attributed to
`AUDIT_CLI_USER` via `cli`: the file's intake (`collect` / `job.create`, with
the file's size and hash, never its name or path), every model call as above,
the combination, and each output file it writes (`disclose` / `output.write`
to `file:<output>` — `transcript`, `translation`, `summary`, `words`,
`entities`, `wordcloud`, `hate_speech`, `keyframes`, `visual_context`,
`docint.jsonl`). A run's events target the run under a fresh id.

Run it inside the running backend with `make cli`, so the chain lands in the
backend's own log (`AUDIT_OUTPUT=container`), attributed to the host account
that ran it — a one-off container can exit before the log collector sees it:

```bash
docker compose --env-file .env -f docker/compose.yaml cp recording.mp4 backend:/tmp/nextext-cli/
make cli ARGS='-f /tmp/nextext-cli/recording.mp4 -sum' CASE_REF=case-0001 REASON='…'
```

The purpose comes from `AUDIT_CASE_REF` / `AUDIT_REASON` (`make cli` passes
`CASE_REF` / `REASON`). When `AUDIT_REQUIRE_PURPOSE` lists one of a run's
actions (`collect`, `ai_call`, `combine`, `disclose`) and no purpose is set,
the run records the refusal and exits `2` with a one-line message; a malformed
purpose or requirement exits `2` before anything is recorded. Every run writes
output files, a disclosure, so under the deployment default (`query,disclose`)
every `make cli` run needs `CASE_REF=` or `REASON=` (either is enough). Run on a host
with `uv run nextext-cli`, the chain goes to the CLI's stdout, which carries
nothing else. `load-models` (`make preload`) reads no personal data and writes
no chain.

## Fail closed

A failed audit write fails the operation, so nothing is processed or handed
out unrecorded:

| Where | What happens |
|---|---|
| A request | the generic `500`, without the data |
| Job creation | the upload is removed and no job exists |
| A job's stage or combination | the job fails and offers no result |
| A CLI run | it stops with a non-zero exit |
| Startup | the backend does not start |

Operational logs never stand in for the audit log. They go to stderr only;
stdout carries the audit lines alone, and uvicorn's access log, which its
default configuration sends to stdout, is moved to stderr at startup.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `AUDIT_OUTPUT` | `stdout` | Where the lines go: the process's stdout, or `container` — the container's main stdout under a container-wide lock, so every process in it (the API, `make cli` runs) reaches the container's log without interleaving. `docker/compose.yaml` sets `container` for the backend |
| `AUDIT_REQUIRE_PURPOSE` | `query,disclose` in `docker/compose.yaml` while unset; empty outside compose | Comma-separated actions that need a stated purpose (`collect`, `query`, `disclose`, `combine`, `delete`, `ai_call`, `modify`, `auth`, `admin`); empty switches the requirement off |
| `NEXTEXT_MEDIA_GRANT_TTL_S` | `900` | Seconds a playback grant stays usable, at most 86400 |
| `AUDIT_CLI_USER` | unset | CLI only: the operator a run is attributed to; `make cli` sets the host account |
| `AUDIT_CASE_REF`, `AUDIT_REASON` | unset | CLI only: the run's stated purpose |

The chain's first event after `start`, `admin` / `config.audit`, records the
mode the backend runs in: `edge_secret_enforced`, `default_identity_enabled`
(the dev fallback is in effect), `require_purpose`, `service_tokens` (how
many services may call as themselves — a count, never a name or token),
`cli_retention_days`, `retention_notice_days`, and `faces` (the operator switch
`NEXTEXT_FACES`: `off`, `detect` or `group`). A CLI run's records
`require_purpose` only.

## Limits

- A job deleted while it runs is recorded at once; its upload is removed when
  the pipeline step holding it returns.
- A pipeline step still running at shutdown cannot write after the chain's
  end: its job fails instead.
- A multipart upload is buffered by the web framework before the route sees
  it; a backend that dies during that upload leaves the partial buffer in the
  system temp directory, outside any job directory, and the startup sweep does
  not see it.
