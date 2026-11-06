# CLI reference

`nextext-cli` runs the same pipeline as the backend, but in-process: it imports
`nextext.pipeline` directly and needs no backend container. It ships inside the
backend image alongside the API, and is the right tool for very large local
files, since it reads from disk instead of streaming an upload. Every run
writes its own [audit chain](#audit-log).

> `uv run nextext-cli --help` is the authoritative flag list — `nextext/cli.py`
> is the source of truth, and the listing below is a snapshot of it.

## Arguments

Running `uv run nextext-cli [ARGS]` from the command line supports the following arguments:

```bash
-h, --help            show this help message and exit
-f, --file            Specify the file path and name of the audio file to be transcribed.
-sl, --src-lang       Specify the language code (ISO 639-1) of the source audio (default: None).
-tl, --trg-lang       Specify the language code (ISO 639-1) of the target language (default: 'de').
-t, --task            Specify the task to perform: 'transcribe' (default), or 'translate'.
--[no-]diarize        Detect and label speakers (default: on).
-w, --words           Show most frequently used words (default: False).
-sum, --summarize     Additional transcript summarization (default: False).
-hs, --hate-speech    Detect hate speech the speakers endorse, judging each segment in context windows via LLM; with -kf, also judge the keyframe captions (default: False).
-kf, --keyframes      Sample video keyframes and describe them (default: False).
-F, --full-analysis   Enable full analysis, equivalent to using -w -sum -hs -kf (default: False).
-ed, --emit-docint-jsonl   Write a docint-compatible JSONL transcript to this path.
-fd, --force-docint-jsonl  Overwrite the --emit-docint-jsonl target if it exists (default: False).
```

## Outputs

Every run writes into an output directory named after the source file. The
transcript always lands there (`{stem}_transcript.txt`/`.csv`/`.xlsx`, plus
`{stem}_translation.txt` for a translate task); the optional steps add their
own files (`{stem}_summary.txt`, `{stem}_words.*`, `{stem}_entities.*`,
`{stem}_wordcloud.png`, `{stem}_hate_speech.*`). The hate-speech files list
only segments whose speaker endorses hate, with the original `text`, `start`,
and — when present — `speaker` and the `translation` the classifier saw as an aid,
plus `source` (`transcript`). With `-kf`, keyframe captions that endorse hate
are listed too, with `source` `frame` and the frame's time as `start`.

`-kf/--keyframes` adds two: the sampled frames as
`{stem}_keyframes/frame_NNN.jpg` plus a `manifest.json` of their sampling times
— the same layout the API's `keyframes.zip` artifact uses — and their
descriptions as `{stem}_visual_context.txt`. The
descriptions are written whether or not a summary was asked for; with `-sum`
they also feed the summary.

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | The run produced a transcript. |
| `1`  | The run failed (unhandled error, e.g. an undecodable file or an unreachable endpoint). |
| `2`  | Command-line usage error — argparse's own code (unknown flag, missing `-f`) — or a refusal before any processing: a malformed `AUDIT_*` setting, or a stated purpose that `AUDIT_REQUIRE_PURPOSE` requires and the run lacks. Not a pipeline outcome. |
| `3`  | The file held no processable speech: nothing was transcribed and the text analysis stages were skipped. An empty transcript is still written, and the warning names which of the three causes fired (`vad_no_speech`, `asr_empty_transcript`, `asr_all_segments_filtered`). `-kf` still runs, so a silent video still yields its keyframes and their descriptions, plus a summary written from them with `-sum` and hate-speech findings judged from them with `-hs`. |

Exit `3` is what lets a batch loop tell "nothing to transcribe" apart from a
successful run — kept distinct from argparse's `2` so a mistyped flag is never
mistaken for a speech-free file.

## Batch processing

In CLI mode, you can let Nextext iterate over a directory to batch process files:

```bash
for file in path/to/your/directory/*; do
    uv run nextext-cli -f $file [ARGS]
done
```

To collect the files that yielded nothing, branch on the exit code:

```bash
for file in path/to/your/directory/*; do
    uv run nextext-cli -f "$file" [ARGS]
    case $? in
        0) ;;
        3) echo "no speech: $file" >> no_speech.txt ;;
        *) echo "failed:    $file" >> failed.txt ;;
    esac
done
```

## Audit log

Every run that processes a file writes its own data-protection audit chain,
attributed to `AUDIT_CLI_USER` (via `cli`): the file's intake (its size and
hash, never its name or path), each model call, the joined result, and each
output file it writes (`file:transcript`, `file:summary`, …). The stated
purpose comes from `AUDIT_CASE_REF` / `AUDIT_REASON`. What is recorded:
[audit.md](audit.md#command-line).

On the production host, run it inside the running backend, so the chain lands
in the backend's own log:

```bash
docker compose --env-file .env -f docker/compose.yaml cp recording.mp4 backend:/tmp/nextext-cli/
make cli ARGS='-f /tmp/nextext-cli/recording.mp4 -w -sum' CASE_REF=case-0001 REASON='Review of a reported clip'
```

A deployment requires a stated purpose for queries and disclosures
(`AUDIT_REQUIRE_PURPOSE=query,disclose`, the `docker/compose.yaml` default),
and every run writes output files, so `make cli` needs `CASE_REF=` or
`REASON=` — either is enough. Without one, the run records the refusal, prints
one line and exits `2` before it reads the file:

```text
nextext-cli: this run needs a stated purpose: set AUDIT_CASE_REF and/or AUDIT_REASON.
```

`make cli` runs `docker compose exec` with `/tmp/nextext-cli` (on the writable
scratch volume) as the working directory, so the outputs land under
`/tmp/nextext-cli/output/<stem>/`; it names the host account that ran it as
`AUDIT_CLI_USER`. Each run ends by saying when its outputs are deleted. Copy
the outputs out with `docker compose … cp` before then.

- **Automatically:** the backend deletes what runs left there, inputs and
  outputs, `NEXTEXT_CLI_RETENTION_DAYS` (default 365) after it was written, at
  startup and then daily, recorded (`delete` / `cli.sweep`, naming each entry by
  a hash of its path, never by name). Outputs earlier releases wrote under
  `/tmp/output/` follow the same rule. A new period deletes nothing for
  `RETENTION_NOTICE_DAYS` (30) after it took effect. A changed period never
  postpones what the period in force already made due, and never deletes sooner
  than announced: a longer one applies at once (within a running first notice,
  at its end), and a shorter one gives the notice only to what it makes due
  sooner, never ending before a notice already running. Changing
  `RETENTION_NOTICE_DAYS` alone moves nothing. The state of the release before
  is rewritten with its notice's end; a state with impossible values is
  refused. Switching retention off forgets when it took effect, so switching it
  on again gives a fresh notice.
- **The notice:** every run, however it ends, prints the earliest day its
  outputs are deleted; the daily sweep can land a day later. A rerun into an
  existing output directory makes it young again.
- **Now:** `make cli-clean` deletes all of it, `RUN=output/<stem>` one run,
  recorded with your account (`delete` / `cli.clean`). `LEGACY=1` also removes
  the inputs earlier releases read from `/tmp` itself (regular files directly in
  `/tmp`, never the backend's own lock file or directories).

Run directly with `uv run nextext-cli`, the chain goes to
stdout, which the CLI uses for nothing else (its logs go to stderr).

## Related

- [configuration.md](configuration.md) — the env vars the CLI reads
  (`RESPONSE_LANGUAGE`, `TEXT_MODEL`, the endpoint group, sentence restoration)
- [architecture.md](architecture.md) — how the in-process CLI path relates to
  the backend service
