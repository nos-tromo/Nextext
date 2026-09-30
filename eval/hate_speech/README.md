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
| `baseline` | A frozen copy of the classifier Nextext shipped before the windowed rewrite: one isolated request per row, the old GMF prompt (`prompts/baseline/` — the per-chunk prompt Nextext and docint both shipped before the rewrite), the old system line, the old `bool()` verdict parse, and the translation judged instead of the original when one exists. |
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
```

**Ablation ladder** — each step changes one thing, so the tables separate the
prompt effect from the context effect:

```bash
# 1. baseline: old prompt, one isolated row per request
uv run python eval/hate_speech/run.py --strategy baseline
# 2. new prompt, still one isolated row per request (prompt effect = 2 vs 1)
uv run python eval/hate_speech/run.py --strategy windowed --window-tokens 1 --context-tokens 0
# 3. new prompt, one row per request with neighbours as context (context effect = 3 vs 2)
uv run python eval/hate_speech/run.py --strategy windowed --window-tokens 1
# 4. production windows (batching effect = 4 vs 3)
uv run python eval/hate_speech/run.py --strategy windowed
```

A labelled row is never clipped below 2048 characters, so `--window-tokens 1`
is a true one-row-per-request run. At default budgets `de-long-panel` spans
three windows, which exercises window edges; the short transcripts fit in one.
If recall drops on long transcripts, try smaller windows (`--window-tokens 400`).

Each run prints a Markdown table and writes a JSON report with every row's
verdict to `eval/hate_speech/reports/` (gitignored), along with the effective
window budgets, prompt locale, provider and `OLLAMA_THINK` setting. Run with the
production think setting: thinking tokens count against the output cap.

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
real slurs. They name real protected groups, because a model reacts to group
terms, not to placeholders. They never repeat the prompt's few-shot examples; a
test rejects any row that shares 60% of its words with one, since a copied
example measures memorisation, not judgement. One transcript per line:

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
  reads the `functionality`, `test_case` and `label_gold` columns and scores the
  benchmark on its own (add `--fixtures …` to run both). Hateful cases
  become `endorses`, the counter-speech functionalities (`counter_quote_nh`,
  `counter_ref_nh`, i.e. MHC F18/F19) become `condemns_or_counters`, and the rest
  `none`. Each case is a one-row transcript tagged with its functionality, so
  `--by-tag` reports per functionality.
- **GAHD** (German adversarial set with contrastive pairs) or a private
  labelled sample: convert to the fixture format above and pass it via
  `--fixtures`.

## docint smoke test

```bash
uv run python eval/hate_speech/run.py --emit-docint-jsonl eval/hate_speech/reports/docint
```

writes each fixture as the `docint.jsonl` payload Nextext would export (one
file per transcript, no model calls), for ingesting into a scratch docint
collection with hate-speech detection enabled.
