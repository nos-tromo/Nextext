"""Tests for the dev-only hate-speech eval harness (no network, no model calls)."""

import json
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import run as hs_run
import score as hs_score

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _transcript(rows: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    """Build an in-memory fixture transcript.

    Args:
        rows (list[dict[str, Any]]): The transcript rows.
        **fields (Any): Overrides for ``id``, ``lang`` and ``tags``.

    Returns:
        dict[str, Any]: The transcript.
    """
    return {"id": "t", "lang": "de", "tags": ["x"], "rows": rows, **fields}


# ---------------------------------------------------------------------------
# score.py
# ---------------------------------------------------------------------------


def test_score_rows_counts_hand_built_confusions() -> None:
    """Counts and rates match a hand-checked confusion table."""
    rows = [
        hs_score.RowResult("t", 0, "endorses", True),
        hs_score.RowResult("t", 1, "endorses", False, needs_context=True),
        hs_score.RowResult("t", 2, "condemns_or_counters", True),
        hs_score.RowResult("t", 3, "quotes_or_reports", False),
        hs_score.RowResult("t", 4, "none", True),
        hs_score.RowResult("t", 5, "none", False),
    ]

    metrics = hs_score.score_rows(rows)

    assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == (1, 2, 1, 2)
    assert metrics.precision == pytest.approx(1 / 3)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.4)
    assert metrics.fpr == pytest.approx(0.5)
    assert metrics.mention_fpr == pytest.approx(0.5)
    assert metrics.mention_rows == 2
    assert metrics.neutral_fpr == pytest.approx(0.5)
    assert metrics.neutral_rows == 2
    assert metrics.context_recall == 0.0
    assert metrics.context_rows == 1


def test_score_rows_leaves_undefined_rates_empty() -> None:
    """Rates without a denominator are ``None``, not zero."""
    metrics = hs_score.score_rows([hs_score.RowResult("t", 0, "none", False)])

    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.f1 is None
    assert metrics.fpr == 0.0


def test_score_by_tag_splits_rows_per_tag() -> None:
    """Each tag gets its own metrics, sorted by tag name."""
    rows = [
        hs_score.RowResult("a", 0, "endorses", True, tags=("direct_hate",)),
        hs_score.RowResult("b", 0, "condemns_or_counters", True, tags=("condemns",)),
    ]

    by_tag = hs_score.score_by_tag(rows)

    assert list(by_tag) == ["condemns", "direct_hate"]
    assert by_tag["condemns"].fp == 1
    assert by_tag["direct_hate"].tp == 1


def test_markdown_table_marks_undefined_rates() -> None:
    """Undefined rates render as a dash."""
    table = hs_score.markdown_table({"baseline": hs_score.score_rows([hs_score.RowResult("t", 0, "none", False)])})

    assert "| baseline | 1 | 0 | 0 | 0 | — | — | — | 0.00 |" in table


# ---------------------------------------------------------------------------
# run.py helpers
# ---------------------------------------------------------------------------


def test_committed_fixtures_are_well_formed() -> None:
    """The committed synthetic fixtures load, cover both classes and both languages."""
    transcripts = hs_run.load_fixtures(sorted(_FIXTURES.glob("*.jsonl")))

    ids = [t["id"] for t in transcripts]
    golds = [row["gold"] for t in transcripts for row in t["rows"]]
    assert len(ids) == len(set(ids))
    assert {"none", "endorses", "quotes_or_reports", "condemns_or_counters", "analyzes_or_discusses"} <= set(golds)
    assert {"de", "en"} <= {t["lang"] for t in transcripts}
    assert any(row.get("needs_context") for t in transcripts for row in t["rows"])
    assert any(row.get("translation") for t in transcripts for row in t["rows"])


def test_load_fixtures_rejects_unknown_gold_labels(tmp_path: Path) -> None:
    """A typo in a gold label fails loudly instead of skewing the scores.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_transcript([{"text": "a", "gold": "condemns"}])) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="condemns"):
        hs_run.load_fixtures([path])


def test_transcript_frame_adds_optional_columns_only_when_used() -> None:
    """Speaker and translation columns appear only when a row carries them."""
    plain = _transcript([{"text": "a", "gold": "none"}, {"text": "b", "gold": "none"}])
    rich = _transcript(
        [{"text": "a", "gold": "none", "speaker": "Speaker 1", "translation": "A"}, {"text": "b", "gold": "none"}]
    )

    plain_df = hs_run.transcript_frame(plain)
    rich_df = hs_run.transcript_frame(rich)

    assert list(plain_df.columns) == ["start", "end", "text"]
    assert plain_df["start"].tolist() == ["0:00:00", "0:00:05"]
    assert list(rich_df.columns) == ["start", "end", "speaker", "text", "translation"]
    assert rich_df["speaker"].tolist() == ["Speaker 1", ""]


def test_flagged_rows_maps_findings_back_by_start() -> None:
    """Findings identify their rows by the unique start stamp the harness generates."""
    df = hs_run.transcript_frame(_transcript([{"text": t, "gold": "none"} for t in ("a", "b", "c")]))

    assert hs_run.flagged_rows(df, [{"start": "0:00:05", "text": "b"}]) == {1}


def test_evaluate_scores_each_row_with_the_given_classifier() -> None:
    """Rows are scored against their gold label with the classifier's flags."""
    transcripts = [_transcript([{"text": "a", "gold": "endorses"}, {"text": "b", "gold": "condemns_or_counters"}])]

    rows = hs_run.evaluate(transcripts, lambda df, lang: {0})

    assert [(r.row, r.gold, r.predicted) for r in rows] == [(0, "endorses", True), (1, "condemns_or_counters", False)]


def test_baseline_flag_reproduces_the_removed_parser() -> None:
    """The frozen baseline keeps the old verdict logic, including its string-"false" quirk."""
    assert hs_run.baseline_flag('{"hate_speech": true, "category": "religion"}') is True
    assert hs_run.baseline_flag('{"hate_speech": "false"}') is True
    assert hs_run.baseline_flag('Sure: {"hate_speech": false} done') is False
    assert hs_run.baseline_flag("I cannot answer that.") is False
    assert hs_run.baseline_flag("[1, 2]") is False


def test_load_mhc_maps_functionalities_to_gold_stances(tmp_path: Path) -> None:
    """Hateful cases are positives, counter-speech is condemnation, the rest is neutral.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    path = tmp_path / "mhc_german.csv"
    pd.DataFrame(
        {
            "functionality": ["derog_neg_emote_h", "counter_ref_nh", "counter_quote_nh", "ident_neutral_nh"],
            "test_case": ["A", "B", "C", "D"],
            "label_gold": ["hateful", "non-hateful", "non-hateful", "non-hateful"],
        }
    ).to_csv(path, index=False)

    transcripts = hs_run.load_mhc(path)

    assert [t["rows"][0]["gold"] for t in transcripts] == [
        "endorses",
        "condemns_or_counters",
        "condemns_or_counters",
        "none",
    ]
    assert transcripts[1]["tags"] == ["counter_ref_nh"]
    assert transcripts[0]["rows"][0]["text"] == "A"


def test_emit_docint_jsonl_writes_one_file_per_transcript(tmp_path: Path) -> None:
    """Fixtures convert to docint-shaped JSONL, one line per row, for docint smoke ingests.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    transcripts = [_transcript([{"text": "a", "gold": "none", "speaker": "Speaker 1"}, {"text": "b", "gold": "none"}])]

    written = hs_run.emit_docint_jsonl(transcripts, tmp_path)

    assert written == [tmp_path / "t.docint.jsonl"]
    records = [json.loads(line) for line in written[0].read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in records] == ["a", "b"]
    assert records[0]["speaker"] == "Speaker 1"
    assert records[0]["language"] == "de"


_PROMPTS = Path(__file__).resolve().parents[2] / "nextext" / "utils" / "prompts"


def _words(text: str) -> set[str]:
    """Return the lower-cased word set of a sentence.

    Args:
        text (str): The sentence.

    Returns:
        set[str]: Its words.
    """
    return set(re.findall(r"\w+", text.lower()))


def _prompt_example_sentences() -> list[str]:
    """Collect the few-shot example rows of every transcript prompt.

    Returns:
        list[str]: The example sentences (speaker label removed).
    """
    sentences: list[str] = []
    for path in sorted(_PROMPTS.glob("*/hate_speech_transcript.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\[[A-Z]\] (?:Speaker \d+: )?(.+)$", line)
            if match:
                sentences.append(match.group(1))
    return sentences


def test_committed_fixtures_do_not_copy_the_prompt_examples() -> None:
    """A fixture that repeats (or nearly repeats) a few-shot example measures memorisation, not judgement."""
    examples = [_words(sentence) for sentence in _prompt_example_sentences()]
    rows = [row["text"] for t in hs_run.load_fixtures(sorted(_FIXTURES.glob("*.jsonl"))) for row in t["rows"]]
    assert examples

    overlapping = [
        text
        for text in rows
        if any(len(_words(text) & ex) / len(_words(text) | ex) >= 0.6 for ex in examples if _words(text))
    ]

    assert overlapping == []


def test_committed_fixtures_include_a_transcript_longer_than_one_default_window() -> None:
    """At default budgets at least one transcript spans several windows, so window edges and context get exercised."""
    default_core_chars = 1000 * 3
    longest = max(
        sum(len(row["text"]) + 12 for row in t["rows"]) for t in hs_run.load_fixtures(sorted(_FIXTURES.glob("*.jsonl")))
    )

    assert longest > 2 * default_core_chars


def test_select_transcripts_runs_mhc_alone_unless_fixtures_are_named(tmp_path: Path) -> None:
    """``--mhc`` scores the benchmark on its own; committed fixtures join only when asked for.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    mhc = tmp_path / "mhc.csv"
    pd.DataFrame({"functionality": ["counter_ref_nh"], "test_case": ["B"], "label_gold": ["non-hateful"]}).to_csv(
        mhc, index=False
    )
    fixture = tmp_path / "one.jsonl"
    fixture.write_text(json.dumps(_transcript([{"text": "a", "gold": "none"}], id="own")) + "\n", encoding="utf-8")

    mhc_only = hs_run.select_transcripts(fixtures=None, mhc=mhc, mhc_lang="de", tags=None, limit=None)
    both = hs_run.select_transcripts(fixtures=[fixture], mhc=mhc, mhc_lang="de", tags=None, limit=None)

    assert [t["id"] for t in mhc_only] == ["mhc-0"]
    assert [t["id"] for t in both] == ["own", "mhc-0"]


def test_run_settings_record_the_effective_budgets_and_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reports state the budgets actually used (defaults included), the provider and the think setting.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("HATE_SPEECH_WINDOW_TOKENS", raising=False)
    monkeypatch.setenv("HATE_SPEECH_CONTEXT_TOKENS", "0")
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv("OLLAMA_THINK", "0")

    settings = hs_run.run_settings()

    assert settings["window_tokens"] == 1000
    assert settings["context_tokens"] == 0
    assert settings["provider"] == "vllm"
    assert settings["ollama_think"] == "0"


# ---------------------------------------------------------------------------
# Chunk mode (docint's per-chunk prompt)
# ---------------------------------------------------------------------------

_CHUNK_FIXTURES = Path(__file__).resolve().parent / "chunk_fixtures"


def test_committed_chunk_fixtures_cover_the_chunk_prompt_risks() -> None:
    """Chunk fixtures load and include the cases where a stance rule can suppress real hate."""
    chunks = hs_run.load_chunk_fixtures(sorted(_CHUNK_FIXTURES.glob("*.jsonl")))

    tags = {tag for chunk in chunks for tag in chunk["tags"]}
    golds = [chunk["rows"][0]["gold"] for chunk in chunks]
    assert {"report_framed_hate", "reporting", "rhetorical_question", "coded_ideology", "image"} <= tags
    assert "endorses" in golds
    assert "quotes_or_reports" in golds
    assert all(len(chunk["rows"]) == 1 for chunk in chunks)


def test_load_chunk_fixtures_rejects_unknown_gold_labels(tmp_path: Path) -> None:
    """A typo in a chunk's gold label fails loudly.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "c", "lang": "de", "tags": ["x"], "text": "a", "gold": "hate"}) + "\n")

    with pytest.raises(ValueError, match="hate"):
        hs_run.load_chunk_fixtures([path])


class _ChunkPipeline:
    """Structural stand-in for ``InferencePipeline`` recording chunk requests."""

    def __init__(self, reply: str) -> None:
        """Store the canned reply.

        Args:
            reply (str): The reply every call returns.
        """
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def call_model(self, prompt: str, **kwargs: Any) -> str:
        """Record the call and return the canned reply.

        Args:
            prompt (str): The rendered chunk prompt.
            **kwargs (Any): Request keyword arguments.

        Returns:
            str: The canned reply.
        """
        self.calls.append({"prompt": prompt, **kwargs})
        return self.reply


def _one_chunk(text: str) -> pd.DataFrame:
    """Build the one-row frame a chunk is scored as.

    Args:
        text (str): The chunk text.

    Returns:
        pd.DataFrame: The frame.
    """
    return hs_run.transcript_frame(_transcript([{"text": text, "gold": "none"}]))


def test_chunk_classifier_flags_only_an_endorsing_stance(tmp_path: Path) -> None:
    """The new chunk prompt's verdict is the stance: endorsement flags, condemnation does not.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    prompt = tmp_path / "hate_speech.txt"
    prompt.write_text("Classify:\n{text}", encoding="utf-8")
    endorsing = _ChunkPipeline(json.dumps({"stance": "endorses", "category": "ethnicity"}))
    condemning = _ChunkPipeline(json.dumps({"stance": "condemns_or_counters", "category": "ethnicity"}))

    flagged = hs_run.chunk_classifier(cast(Any, endorsing), prompt)(_one_chunk("Eins."), "de")
    clean = hs_run.chunk_classifier(cast(Any, condemning), prompt)(_one_chunk("Zwei."), "de")

    assert flagged == {0}
    assert clean == set()
    assert endorsing.calls[0]["prompt"] == "Classify:\nEins."
    assert endorsing.calls[0]["include_system_prompt"] is False
    assert endorsing.calls[0]["response_format"]["json_schema"]["strict"] is True


def test_chunk_baseline_classifier_reproduces_docints_old_call() -> None:
    """The frozen chunk baseline sends the old prompt without a system role and trusts the boolean."""
    pipeline = _ChunkPipeline('{"hate_speech": "false", "category": "none"}')

    flagged = hs_run.chunk_baseline_classifier(cast(Any, pipeline))(_one_chunk("Drei."), "de")

    assert flagged == {0}
    assert pipeline.calls[0]["include_system_prompt"] is False
    assert "Drei." in pipeline.calls[0]["prompt"]
