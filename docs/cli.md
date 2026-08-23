# CLI reference

`nextext-cli` runs the same pipeline as the backend, but in-process: it imports
`nextext.pipeline` directly and needs no backend container. It ships inside the
backend image alongside the API, and is the right tool for very large local
files, since it reads from disk instead of streaming an upload.

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
-o, --output          Specify the output directory (default: output).
-F, --full-analysis   Enable full analysis, equivalent to using -w -sum (default: False).
```

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | The run produced a transcript. |
| `2`  | The file held no processable speech — nothing was transcribed and the analysis stages were skipped. An empty transcript is still written, and the warning names which of the three causes fired (`vad_no_speech`, `asr_empty_transcript`, `asr_all_segments_filtered`). |
| `1`  | The run failed (unhandled error, e.g. an undecodable file or an unreachable endpoint). |

Exit `2` is what lets a batch loop tell "nothing to transcribe" apart from a
successful run.

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
        2) echo "no speech: $file" >> no_speech.txt ;;
        *) echo "failed:    $file" >> failed.txt ;;
    esac
done
```

## Related

- [configuration.md](configuration.md) — the env vars the CLI reads
  (`RESPONSE_LANGUAGE`, `TEXT_MODEL`, the endpoint group, sentence restoration)
- [architecture.md](architecture.md) — how the in-process CLI path relates to
  the backend service
