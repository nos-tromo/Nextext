# Hate-speech eval harness

Dev-only harness that scores Nextext's hate-speech classification against
gold-labelled transcripts. It answers one question: does the windowed,
stance-aware classifier stop flagging people who *talk about* hate, without
missing the ones who *spread* it?

**Not shipped.** `eval/` is excluded from the Docker image (`.dockerignore`)
and is not a Python package, so it never reaches the wheel. Nothing here runs
at serve time.

## Strategies

| Strategy | What it runs |
|---|---|
| `baseline` | A frozen copy of the classifier Nextext shipped before the windowed rewrite: one isolated request per row, the old GMF prompt (`prompts/baseline/`, byte-identical to docint's current `hate_speech.txt`), the old system line, the old `bool()` verdict parse, and the translation judged instead of the original when one exists. |
| `windowed` | The production `nextext.pipeline.hate_speech_pipeline`. |

Both run against the live chat model, configured by the usual environment
variables (`OPENAI_API_BASE`, `OPENAI_API_KEY`, `TEXT_MODEL`,
`INFERENCE_PROVIDER`, `OLLAMA_THINK`, `RESPONSE_LANGUAGE`). A run costs real
inference, so it is manual — CI only runs the offline unit tests
(`test_hate_speech_eval.py`).

## Run

```bash
# Both strategies over the committed fixtures, one table
uv run python eval/hate_speech/run.py

# Per-tag breakdown, German prompt
uv run python eval/hate_speech/run.py --by-tag --prompt-lang de

# Ablation: the new prompt without context margins (separates prompt from context effect)
uv run python eval/hate_speech/run.py --strategy windowed --context-tokens 0

# Smaller windows (if recall drops on long transcripts)
uv run python eval/hate_speech/run.py --strategy windowed --window-tokens 400
```

Each run prints a Markdown table and writes a JSON report with every row's
verdict to `eval/hate_speech/reports/` (gitignored). Run with the production
`OLLAMA_THINK` setting: thinking tokens can count against the output cap.

## Metrics

Gold labels are stance-level: `none`, or the speaker's stance — `endorses`,
`quotes_or_reports`, `condemns_or_counters`, `analyzes_or_discusses`,
`unclear`. Only `endorses` is a positive.

- **precision / recall / F1 / FPR** — over all rows.
- **mention FPR** — the flagged share of rows that quote, report, condemn,
  analyse, or are unclear. This is the use/mention error the rewrite targets.
- **neutral FPR** — the flagged share of rows with no hostile content.
- **context recall** — recall on endorsing rows that are only hateful in
  context (`needs_context`), e.g. "Die gehören alle weg." after a row naming
  the group.

## Fixtures

`fixtures/*.jsonl` hold **invented** transcripts only — never real data, never
real slurs. One transcript per line:

```json
{"id": "de-context-hate", "lang": "de", "tags": ["context_hate"], "rows": [
  {"speaker": "Speaker 1", "text": "…", "gold": "none"},
  {"speaker": "Speaker 1", "text": "…", "gold": "endorses", "needs_context": true, "category": "ethnicity"}]}
```

Optional per row: `speaker`, `translation` (judged as an aid), `needs_context`,
`category`. The harness generates unique `start`/`end` stamps at 5-second steps.

## External data (local only, never committed)

Place downloaded sets under `eval/hate_speech/data/` (gitignored by `**/data/`).

- **Multilingual HateCheck** (German suite): `--mhc eval/hate_speech/data/<file>.csv`
  reads the `functionality`, `test_case` and `label_gold` columns. Hateful cases
  become `endorses`, the counter-speech functionalities (`counter_quote_nh`,
  `counter_ref_nh`, i.e. MHC F18/F19) become `condemns_or_counters`, and the rest
  `none`. Each case is a one-row transcript tagged with its functionality, so
  `--by-tag` reports per functionality.
- **GAHD** (German adversarial set with contrastive pairs) or a private
  labelled sample: convert to the fixture format above and pass it via
  `--fixtures`.

## docint smoke test

```bash
uv run python eval/hate_speech/run.py --emit-docint-jsonl /tmp/hs-docint
```

writes each fixture as the `docint.jsonl` payload Nextext would export (one
file per transcript, no model calls), for ingesting into a scratch docint
collection with hate-speech detection enabled.
