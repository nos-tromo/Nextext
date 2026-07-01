# Standalone HTTP NER Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route named-entity recognition through the out-of-process `/gliner` HTTP service (selected by `NER_API_BASE`), mirroring the `/diarize` architecture, and remove the in-process GLiNER path and its dependency.

**Architecture:** A new stateless agent `nextext/core/ner.py` POSTs transcript chunks to `{NER_API_BASE}/gliner` and tallies entities into the existing `[Category, Entity, Frequency]` DataFrame. `wordlevel_pipeline` calls it instead of `WordCounter.named_entity_recognition()`. When `NER_API_BASE` is unset, NER yields an empty table (exactly like diarization with `DIARIZE_API_BASE` unset). The in-process GLiNER model, its offline loader, its preload, and the `gliner` dependency are deleted.

**Tech Stack:** Python 3.12, httpx, pandas, pytest (monkeypatch), loguru, uv.

**Reference spec:** `docs/superpowers/specs/2026-06-13-standalone-ner-design.md`

**Verified service contract:** `POST {base}/gliner` with `{"text": str, "labels": [str]}` → `{"entities": [{"start": int, "end": int, "text": str, "label": str, "score": float}, ...]}`. Reachable from the backend at `http://vllm-router:4000/gliner`.

---

## File Structure

**Create:**
- `nextext/core/ner.py` — the NER HTTP agent: `extract_entities(text) -> pd.DataFrame`, plus `_chunk_text`, `_NER_LABELS`, `_NER_THRESHOLD`, `_NER_WORD_BUDGET`, `_SENTENCE_RE` (relocated from `words.py`).
- `tests/test_ner.py` — unit tests mirroring `tests/test_diarization.py`.

**Modify:**
- `nextext/utils/env_cfg.py` — add `NerConfig`, `DEFAULT_NER_TIMEOUT`, `load_ner_env`.
- `tests/test_env_cfg.py` — add `load_ner_env` tests.
- `nextext/pipeline.py` — import + call `extract_entities`; stop calling `WordCounter.named_entity_recognition`.
- `tests/test_pipeline.py` — patch `pipeline.extract_entities` in the word-level test.
- `nextext/core/words.py` — delete all in-process GLiNER machinery and the `named_entity_recognition` method.
- `nextext/utils/model_loader.py` — delete `preload_gliner_model`, `GLINER_MODEL_ID`, the `gliner`/`gc` imports, and the preload call.
- `tests/test_model_loader.py` — drop the GLiNER preload assertions.
- `tests/test_model_registry.py` — drop `test_gliner_spec_opts_out_of_mps` and the `words` side-effect import.
- `nextext/utils/model_registry.py` — drop GLiNER from example docstrings.
- `pyproject.toml` + `uv.lock` — drop `gliner>=0.2.0`.
- `docker/compose.yaml` — replace `NER_MODEL` passthrough with `NER_API_BASE` + `NER_TIMEOUT`.
- `.env.example` — drop `NER_MODEL`/`MODEL_RESIDENCY_GLINER`, add an NER section.
- `CLAUDE.md`, `AGENTS.md` — document the HTTP NER agent.

**Task order rationale:** config → agent → pipeline rewire → remove in-process GLiNER → remove preload → drop dependency → docs. The pipeline is rewired (Task 3) before the method is deleted (Task 4) so tests stay green at every step. The dependency is dropped (Task 6) only after no code imports `gliner`.

---

## Task 1: Add `NerConfig` + `load_ner_env` to env config

**Files:**
- Modify: `nextext/utils/env_cfg.py` (add after `DiarizationConfig` ~line 121, after `DEFAULT_DIARIZE_TIMEOUT` ~line 129, after `load_diarization_env` ~line 264)
- Test: `tests/test_env_cfg.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_env_cfg.py`, extend the import block (currently importing `DEFAULT_DIARIZE_TIMEOUT, load_diarization_env, …`) to add `DEFAULT_NER_TIMEOUT` and `load_ner_env`:

```python
from nextext.utils.env_cfg import (
    DEFAULT_DIARIZE_TIMEOUT,
    DEFAULT_NER_TIMEOUT,
    load_diarization_env,
    load_inference_env,
    load_ner_env,
    load_transcription_env,
)
```

Append these tests to the end of the file (`io` and `pytest` are already imported):

```python
# ---------------------------------------------------------------------------
# load_ner_env
# ---------------------------------------------------------------------------


def test_load_ner_env_unset_disables_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset NER_API_BASE disables NER and uses the default timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("NER_API_BASE", raising=False)
    monkeypatch.delenv("NER_TIMEOUT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cfg = load_ner_env()

    assert cfg.api_base == ""
    assert cfg.api_key == ""
    assert cfg.timeout == DEFAULT_NER_TIMEOUT


def test_load_ner_env_strips_whitespace_and_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NER_API_BASE is normalised: surrounding whitespace and trailing '/' removed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("NER_API_BASE", "  http://vllm-router:4000/  ")

    cfg = load_ner_env()

    assert cfg.api_base == "http://vllm-router:4000"


def test_load_ner_env_reuses_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bearer token is taken from OPENAI_API_KEY (the shared router credential).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("NER_API_BASE", "http://vllm-router:4000")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")

    cfg = load_ner_env()

    assert cfg.api_key == "sk-secret"


def test_load_ner_env_parses_valid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid NER_TIMEOUT is parsed as a float.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("NER_TIMEOUT", "45")

    cfg = load_ner_env()

    assert cfg.timeout == 45.0


@pytest.mark.parametrize("raw_value", ["not-a-number", "0", "-5"])
def test_load_ner_env_invalid_timeout_warns_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    """Non-numeric or non-positive NER_TIMEOUT warns and falls back to the default.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        raw_value (str): An invalid timeout token.
    """
    from loguru import logger

    monkeypatch.setenv("NER_TIMEOUT", raw_value)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        cfg = load_ner_env()
    finally:
        logger.remove(handler_id)

    assert cfg.timeout == DEFAULT_NER_TIMEOUT
    assert "NER_TIMEOUT" in sink.getvalue()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_env_cfg.py -k ner -v`
Expected: FAIL at import — `ImportError: cannot import name 'load_ner_env' from 'nextext.utils.env_cfg'`.

- [ ] **Step 3: Implement `NerConfig`, `DEFAULT_NER_TIMEOUT`, `load_ner_env`**

In `nextext/utils/env_cfg.py`, add the dataclass immediately after the `DiarizationConfig` class:

```python
@dataclass(frozen=True)
class NerConfig:
    """Dataclass for the named-entity-recognition service configuration.

    NER runs out-of-process against an HTTP ``/gliner`` endpoint (e.g.
    ``nos-tromo/vllm-service`` behind the LiteLLM router); Nextext no longer
    hosts GLiNER in-process.

    Attributes:
        api_base: Root URL of the NER service (e.g. ``http://vllm-router:4000``);
            the client appends ``/gliner``. An empty string disables NER.
        api_key: Bearer token forwarded to the service, reused from
            ``OPENAI_API_KEY``. An empty string sends no ``Authorization`` header.
        timeout: Per-request timeout in seconds for each chunk's ``/gliner`` call.
    """

    api_base: str
    api_key: str
    timeout: float
```

Add the default constant next to `DEFAULT_DIARIZE_TIMEOUT`:

```python
DEFAULT_NER_TIMEOUT: float = 120.0
```

Add the loader immediately after `load_diarization_env`:

```python
def load_ner_env() -> NerConfig:
    """Loads named-entity-recognition service configuration from environment variables.

    Returns:
        NerConfig: Dataclass containing the resolved settings.
        - api_base (str): ``NER_API_BASE`` with surrounding whitespace and any
          trailing ``/`` removed. Empty (the default) disables NER, so callers
          skip the HTTP call and emit no entities.
        - api_key (str): ``OPENAI_API_KEY`` (the NER service shares the inference
          router's credentials). Empty sends no bearer token.
        - timeout (float): ``NER_TIMEOUT`` seconds. Defaults to
          :data:`DEFAULT_NER_TIMEOUT`; non-numeric or non-positive values warn
          and fall back to the default.
    """
    api_base = os.getenv("NER_API_BASE", "").strip().rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    timeout = DEFAULT_NER_TIMEOUT
    raw_timeout = os.getenv("NER_TIMEOUT", "").strip()
    if raw_timeout:
        try:
            parsed = float(raw_timeout)
            if parsed <= 0:
                raise ValueError
            timeout = parsed
        except ValueError:
            logger.warning(
                "Invalid NER_TIMEOUT '{}'. Falling back to {}s.",
                raw_timeout,
                DEFAULT_NER_TIMEOUT,
            )

    return NerConfig(api_base=api_base, api_key=api_key, timeout=timeout)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_env_cfg.py -k ner -v`
Expected: PASS (7 tests: 4 single + 3 parametrized invalid-timeout cases).

- [ ] **Step 5: Commit**

```bash
git add nextext/utils/env_cfg.py tests/test_env_cfg.py
git commit -m "feat(ner): add NerConfig and load_ner_env"
```

---

## Task 2: Add the `nextext/core/ner.py` HTTP agent

**Files:**
- Create: `nextext/core/ner.py`
- Test: `tests/test_ner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ner.py`:

```python
"""Tests for the named-entity-recognition agent (HTTP /gliner client)."""

from typing import Any

import httpx
import pytest

from nextext.core import ner
from nextext.core.ner import extract_entities


def test_extract_entities_returns_empty_when_base_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset NER_API_BASE disables NER and never issues a request.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.delenv("NER_API_BASE", raising=False)

    def fail_post(url: str, **kwargs: Any) -> httpx.Response:
        raise AssertionError("httpx.post must not be called when NER_API_BASE is unset")

    monkeypatch.setattr(ner.httpx, "post", fail_post)

    df = extract_entities("Barack Obama visited Berlin.")

    assert list(df.columns) == ["Category", "Entity", "Frequency"]
    assert df.empty


def test_extract_entities_posts_correctly_and_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request targets /gliner with text + labels, bearer auth, and timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000/")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.delenv("NER_TIMEOUT", raising=False)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(
            200,
            json={
                "entities": [
                    {"start": 0, "end": 12, "text": "Barack Obama", "label": "person", "score": 0.99},
                    {"start": 21, "end": 27, "text": "Berlin", "label": "loc", "score": 0.97},
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    df = extract_entities("Barack Obama visited Berlin.")

    assert captured["url"] == "http://router:4000/gliner"
    assert captured["json"]["text"] == "Barack Obama visited Berlin."
    assert captured["json"]["labels"] == ner._NER_LABELS
    assert captured["headers"]["Authorization"] == "Bearer sk-secret"
    assert captured["timeout"] == 120.0
    rows = {(c, e): f for c, e, f in df.itertuples(index=False)}
    assert rows[("PERSON", "Barack Obama")] == 1
    assert rows[("LOC", "Berlin")] == 1


def test_extract_entities_omits_authorization_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Authorization header is sent when OPENAI_API_KEY is empty.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return httpx.Response(200, json={"entities": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    extract_entities("Some text with no entities of interest.")

    assert "Authorization" not in captured["headers"]


def test_extract_entities_filters_low_score_and_short_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entities below the score threshold or shorter than 3 chars are dropped.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "entities": [
                    {"text": "Berlin", "label": "loc", "score": 0.9},   # kept
                    {"text": "Bonn", "label": "loc", "score": 0.1},     # dropped: score < 0.3
                    {"text": "UN", "label": "org", "score": 0.95},      # dropped: len < 3
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    df = extract_entities("Berlin Bonn UN are words.")

    pairs = {(c, e) for c, e, _ in df.itertuples(index=False)}
    assert pairs == {("LOC", "Berlin")}


def test_extract_entities_chunks_long_text_into_multiple_posts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text beyond the word budget is split across multiple /gliner requests.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")
    posts: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        posts.append(kwargs["json"]["text"])
        return httpx.Response(200, json={"entities": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(ner.httpx, "post", fake_post)
    # Two sentences, each well over the 512-word budget, force >= 2 chunks.
    long_text = ("word " * 600).strip() + ". " + ("term " * 600).strip() + "."

    extract_entities(long_text)

    assert len(posts) >= 2


def test_extract_entities_swallows_http_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-2xx response is logged and yields an empty DataFrame.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(500, text="boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    df = extract_entities("Barack Obama visited Berlin.")

    assert df.empty


def test_extract_entities_swallows_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport error (e.g. connection refused) is logged and yields an empty DataFrame.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    df = extract_entities("Barack Obama visited Berlin.")

    assert df.empty


def test_extract_entities_handles_non_dict_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JSON payload that is not an object is rejected and yields an empty DataFrame.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"], request=httpx.Request("POST", url))

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    df = extract_entities("Barack Obama visited Berlin.")

    assert df.empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ner.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'nextext.core.ner'`.

- [ ] **Step 3: Implement `nextext/core/ner.py`**

Create `nextext/core/ner.py`:

```python
"""Named-entity-recognition agent: HTTP client for the out-of-process ``/gliner`` service.

Nextext no longer hosts GLiNER in-process. NER runs against an HTTP ``/gliner``
endpoint (e.g. ``nos-tromo/vllm-service`` behind the LiteLLM router) that accepts
text plus a label set and returns scored entities. This module owns the wire call
and the client-side tally onto the ``[Category, Entity, Frequency]`` table that
the pipeline, CLI exporter, and API schemas already consume.

The endpoint is located via ``NER_API_BASE``; when it is unset NER is disabled and
callers receive an empty table (see :func:`nextext.utils.env_cfg.load_ner_env`).
Failures are logged and swallowed: a transcript without entities is preferable to
a failed job.
"""

import re
from collections import Counter

import httpx as httpx  # explicit re-export so tests can monkeypatch ner.httpx
import pandas as pd
from loguru import logger

from nextext.utils.env_cfg import load_ner_env

__all__ = ["extract_entities"]

_NER_LABELS = [
    "date",
    "event",
    "fac",
    "group",
    "loc",
    "money",
    "org",
    "person",
    "time",
]
_NER_THRESHOLD = 0.3
_NER_WORD_BUDGET = 512
_SENTENCE_RE = re.compile(
    r".+?(?:[.!?]+[\"')\]]*(?=\s+|$)|\n{2,}|$)",
    re.DOTALL,
)


def _chunk_text(text: str, word_budget: int = _NER_WORD_BUDGET) -> list[str]:
    """Split text into sentence-packed chunks within a word budget.

    Args:
        text (str): Raw input text.
        word_budget (int): Maximum whitespace-delimited words per chunk.

    Returns:
        list[str]: Ordered list of text chunks suitable for repeated NER inference.
    """
    sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text.strip()) if m.group(0).strip()] or [
        text.strip()
    ]
    chunks: list[str] = []
    current_words: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(current_words) + len(words) > word_budget:
            if current_words:
                chunks.append(" ".join(current_words))
            current_words = words[:word_budget]
        else:
            current_words.extend(words)
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def extract_entities(text: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Tally named entities for ``text`` via the out-of-process ``/gliner`` service.

    The service URL is ``{NER_API_BASE}/gliner``. When ``NER_API_BASE`` is unset
    NER is disabled: a warning is logged and an empty table is returned so callers
    proceed without entities. Each 512-word chunk is sent as a separate request;
    per-chunk transport/HTTP/JSON errors are logged and skipped so one bad chunk
    never loses the whole transcript's entities. Entities are kept when their
    score is at least :data:`_NER_THRESHOLD` and their text is at least 3
    characters long; labels are upper-cased and counts tallied.

    Args:
        text (str): The transcript text to analyse.
        columns (list[str] | None): Output column names. Defaults to
            ``["Category", "Entity", "Frequency"]``.

    Returns:
        pd.DataFrame: A ``[Category, Entity, Frequency]`` DataFrame. Empty (with
            those columns) when NER is disabled, the text is blank, or every
            request fails.
    """
    if columns is None:
        columns = ["Category", "Entity", "Frequency"]
    empty = pd.DataFrame(columns=pd.Index(columns)).reset_index(drop=True)

    config = load_ner_env()
    if not config.api_base:
        logger.warning(
            "NER requested but NER_API_BASE is unset; returning no entities. "
            "Set NER_API_BASE to enable named-entity recognition."
        )
        return empty
    if not text or not text.strip():
        return empty

    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    url = f"{config.api_base}/gliner"
    all_entities: list[tuple[str, str]] = []
    for chunk in _chunk_text(text):
        try:
            response = httpx.post(
                url,
                json={"text": chunk, "labels": _NER_LABELS},
                headers=headers,
                timeout=config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "NER request to {} failed ({}): {}",
                url,
                exc.response.status_code,
                exc.response.text[:500],
            )
            continue
        except (httpx.HTTPError, ValueError, OSError) as exc:
            logger.error("NER request to {} failed: {}", url, exc)
            continue

        if not isinstance(payload, dict):
            logger.error("NER response from {} was not a JSON object; ignoring.", url)
            continue

        for entity in payload.get("entities", []):
            text_val = str(entity.get("text", "")).strip()
            label = str(entity.get("label", "")).strip()
            score = float(entity.get("score", 0.0))
            if text_val and label and len(text_val) >= 3 and score >= _NER_THRESHOLD:
                all_entities.append((label.upper(), text_val))

    if not all_entities:
        return empty

    entity_counts = Counter(all_entities)
    return pd.DataFrame(
        [(label, entity, count) for (label, entity), count in entity_counts.items()],
        columns=pd.Index(columns),
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ner.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add nextext/core/ner.py tests/test_ner.py
git commit -m "feat(ner): add /gliner HTTP client agent"
```

---

## Task 3: Route `wordlevel_pipeline` through `extract_entities`

**Files:**
- Modify: `nextext/pipeline.py` (import near line 13; call at line 218)
- Test: `tests/test_pipeline.py` (`test_wordlevel_pipeline_invokes_all_steps`, lines 557-660)

- [ ] **Step 1: Update the test to assert the new wiring (failing)**

In `tests/test_pipeline.py`, inside `test_wordlevel_pipeline_invokes_all_steps`:

1. Delete the `named_entity_recognition` method from `DummyWordCounter` (lines 608-618).
2. Replace the setup/assert block (currently lines 652-658) with a patched `extract_entities` that captures its argument:

```python
    monkeypatch.setattr(pipeline, "WordCounter", lambda text, language: DummyWordCounter(text, language))
    captured: dict[str, str] = {}

    def fake_extract_entities(text: str) -> pd.DataFrame:
        captured["text"] = text
        return pd.DataFrame({"entity": ["Test"]})

    monkeypatch.setattr(pipeline, "extract_entities", fake_extract_entities)
    df = pd.DataFrame({"text": ["alpha", "beta"]})

    counts, entities, wordcloud = pipeline.wordlevel_pipeline(df, "en")

    assert list(counts["word"]) == ["test"]
    assert list(entities["entity"]) == ["Test"]
    assert captured["text"] == "alpha beta"
    assert wordcloud == "wordcloud"  # type: ignore[comparison-overlap]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py::test_wordlevel_pipeline_invokes_all_steps -v`
Expected: FAIL — `AttributeError: <module 'nextext.pipeline'> does not have the attribute 'extract_entities'` (monkeypatch.setattr raises because the name is not yet imported into the pipeline module).

- [ ] **Step 3: Wire the pipeline to `extract_entities`**

In `nextext/pipeline.py`, add the import next to the existing word/diarization imports (the file already has `from nextext.core.words import WordCounter` at line 13 and `from nextext.core.diarization import assign_speakers_by_overlap, diarize_file` at line 9):

```python
from nextext.core.ner import extract_entities
```

Then in `wordlevel_pipeline`, replace line 218:

```python
    named_entities = word_analysis.named_entity_recognition()
```

with:

```python
    named_entities = extract_entities(word_analysis.text)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py::test_wordlevel_pipeline_invokes_all_steps -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nextext/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): route NER through the /gliner service"
```

---

## Task 4: Remove in-process GLiNER from `words.py` (and its registry test)

**Files:**
- Modify: `nextext/core/words.py`
- Modify: `tests/test_model_registry.py` (drop `words` import line 11; drop `test_gliner_spec_opts_out_of_mps`, ~lines 210-219)
- Modify: `nextext/utils/model_registry.py` (drop GLiNER from example docstrings, lines ~3 and ~47)

> The registry test and the registration must change together: once `words.py` stops registering the `gliner` spec, `REGISTRY._specs["gliner"]` no longer exists, so `test_gliner_spec_opts_out_of_mps` would `KeyError`.

- [ ] **Step 1: Replace the `words.py` import header**

Replace lines 1-36 (module docstring through `load_dotenv()`) of `nextext/core/words.py` with exactly:

```python
"""Word-level analysis: counts and word clouds via spaCy + NLTK."""

from collections import Counter

import arabic_reshaper
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from bidi.algorithm import get_display
from camel_tools.tokenizers.word import simple_word_tokenize
from dotenv import load_dotenv
from loguru import logger
from matplotlib.figure import Figure
from spacy.language import Language
from spacy.tokens import Doc
from wordcloud import WordCloud

from nextext.utils.font_loader import load_font_file
from nextext.utils.mappings_loader import load_mappings
from nextext.utils.model_loader import (
    download_spacy_model,
    ensure_spacy_model_path,
)

load_dotenv()
```

- [ ] **Step 2: Delete the GLiNER section**

Delete the entire GLiNER block that previously sat between the imports and `class WordCounter` — every line from the `# ---- GLiNER NER ----` comment banner through the end of `_chunk_text` (the original `_GLINER_*` constants, `_resolve_hf_cache_dir`, `_resolve_hf_cache_path`, `_load_gliner_config`, `_link_or_copy_path`, `_resolve_local_gliner_dependency`, `_materialize_offline_gliner_dir`, `_prepare_local_gliner_model_dir`, `_resolve_gliner_load_target`, `_load_gliner`, `_move_gliner`, the `REGISTRY.register(ModelSpec(name="gliner", …))` call, and `_chunk_text`). After this, `class WordCounter` follows directly after `load_dotenv()` (with two blank lines between).

- [ ] **Step 3: Delete the `named_entity_recognition` method and fix the class docstring**

In `class WordCounter`, delete the whole `named_entity_recognition` method (originally lines 508-553). In the class docstring's `Methods:` block, delete the two lines:

```python
        named_entity_recognition(columns=...) -> pd.DataFrame:
            Run NER and return a DataFrame.
```

- [ ] **Step 4: Update the registry test and registry docstrings**

In `tests/test_model_registry.py`, delete the import line:

```python
from nextext.core import words  # noqa: F401 — registers the gliner spec
```

and delete the entire `test_gliner_spec_opts_out_of_mps` function (the docstring + the two-line body asserting `model_registry.REGISTRY._specs["gliner"].mps_compatible is False`).

In `nextext/utils/model_registry.py`, edit the two docstring mentions so GLiNER is no longer used as an example:
- Module docstring (line ~3): change `Models that can live on GPU (Whisper, diarization, GLiNER, ...) register a` to `Models that can live on GPU (e.g. Whisper) register a`.
- `ModelSpec` attribute docstring (line ~47): change `name: Registry key, e.g. ``"gliner"``, ``"whisper_turbo"``.` to `name: Registry key, e.g. ``"whisper_turbo"``.`

- [ ] **Step 5: Verify no dangling references, types, or unused imports**

Run: `uv run ruff check nextext/core/words.py tests/test_model_registry.py nextext/utils/model_registry.py`
Expected: PASS — in particular **no `F401` unused-import** warnings (confirms the pruned `words.py` import list is correct). If ruff reports an unused import, remove it.

Run: `uv run mypy --no-incremental --ignore-missing-imports --disable-error-code=import-untyped --disable-error-code=attr-defined --disable-error-code=assignment nextext/`
Expected: PASS.

Run: `uv run pytest tests/test_words.py tests/test_model_registry.py tests/test_pipeline.py -v`
Expected: PASS (no test references the removed method or spec).

- [ ] **Step 6: Commit**

```bash
git add nextext/core/words.py tests/test_model_registry.py nextext/utils/model_registry.py
git commit -m "refactor: remove in-process GLiNER NER"
```

---

## Task 5: Remove the GLiNER preload from `model_loader.py`

**Files:**
- Modify: `nextext/utils/model_loader.py` (imports lines 4 & 18 & 32; `preload_gliner_model` ~lines 247-257; `main()` call ~lines 302-304)
- Modify: `tests/test_model_loader.py` (~lines 195-206)

- [ ] **Step 1: Update the preload test (failing)**

In `tests/test_model_loader.py`, delete the `preload_gliner_model` monkeypatch block:

```python
    monkeypatch.setattr(
        model_loader,
        "preload_gliner_model",
        lambda: calls.append(("gliner", model_loader.GLINER_MODEL_ID)),
    )
```

and delete the GLiNER row from the expected `calls` assertion so it reads:

```python
    assert calls == [
        ("nltk", "all"),
        ("spacy", "en_core_web_sm"),
        ("whisper:cpu", "large-v3-turbo"),
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_model_loader.py -v`
Expected: FAIL — `main()` still calls the real `preload_gliner_model`, appending the `("gliner", …)` tuple, so the trimmed assertion mismatches.

- [ ] **Step 3: Remove the GLiNER preload from `model_loader.py`**

In `nextext/utils/model_loader.py`:

1. Delete the import line 18: `from gliner import GLiNER`.
2. Delete the import line 4: `import gc as gc` (its only use is in `preload_gliner_model`).
3. Delete the constant line 32: `GLINER_MODEL_ID = "gliner-community/gliner_large-v2.5"`.
4. Delete the entire `preload_gliner_model` function:

```python
def preload_gliner_model(model_id: str = GLINER_MODEL_ID) -> None:
    """Download and cache the GLiNER NER model.

    Args:
        model_id (str): The Hugging Face model ID for GLiNER.
    """
    logger.info("Loading GLiNER model '{}'.", model_id)
    model = GLiNER.from_pretrained(model_id)
    del model
    gc.collect()
    logger.info("GLiNER model '{}' cached.", model_id)
```

5. Delete the `try/except` block in `main()` that calls it:

```python
    try:
        preload_gliner_model()
    except Exception as exc:
        failures.append(f"GLiNER {GLINER_MODEL_ID} ({exc})")
```

- [ ] **Step 4: Verify**

Run: `uv run ruff check nextext/utils/model_loader.py`
Expected: PASS — no `F401` (confirms `gc` and `GLiNER` were the only users removed).

Run: `uv run pytest tests/test_model_loader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nextext/utils/model_loader.py tests/test_model_loader.py
git commit -m "refactor: drop the GLiNER preload"
```

---

## Task 6: Drop the `gliner` dependency

**Files:**
- Modify: `pyproject.toml` (line 19)
- Modify: `uv.lock` (regenerated)

- [ ] **Step 1: Remove the dependency line**

In `pyproject.toml`, delete the line from the `dependencies` array:

```toml
    "gliner>=0.2.0",
```

- [ ] **Step 2: Re-lock**

Run: `uv lock`
Expected: completes; `uv.lock` no longer contains a `[[package]] name = "gliner"` entry. Verify: `grep -c '^name = "gliner"' uvlock_check; rm -f uvlock_check` — or simply `grep -n 'name = "gliner"' uv.lock` returns nothing.

- [ ] **Step 3: Verify the suite still imports and passes without gliner installed**

Run: `uv sync --group dev`
Run: `uv run pytest`
Expected: PASS — full suite green; no module imports `gliner` anymore.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): drop the gliner dependency"
```

---

## Task 7: Documentation and configuration

**Files:**
- Modify: `docker/compose.yaml` (line 37)
- Modify: `.env.example` (line 12; lines 62-63; diarization section ~66-76)
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: compose.yaml — swap the env passthrough**

In `docker/compose.yaml`, in the `&backend-env` block, replace:

```yaml
      NER_MODEL: ${NER_MODEL:-}
```

with (keep alphabetical order — place beside the other NER/NEXTEXT keys):

```yaml
      NER_API_BASE: ${NER_API_BASE:-}
      NER_TIMEOUT: ${NER_TIMEOUT:-120}
```

- [ ] **Step 2: .env.example — remove GLiNER knobs, add an NER section**

Delete line 12: `# NER_MODEL=gliner-community/gliner_large-v2.5`.

In the residency block, delete `# MODEL_RESIDENCY_GLINER=evict` and change the "Supported names" comment from `# Supported names: gliner, whisper_turbo.` to `# Supported names: whisper_turbo.`

After the diarization section (after the `# DIARIZE_TIMEOUT=600 …` line), add:

```bash
# =============================================================================
# Named-entity recognition (out-of-process /gliner HTTP service)
# =============================================================================
# NER runs against an HTTP /gliner endpoint (e.g. nos-tromo/vllm-service behind
# the LiteLLM router) instead of an in-process GLiNER model. Set NER_API_BASE to
# the service ROOT (the client appends /gliner); leave it unset to disable NER,
# in which case transcripts carry no entity table. The bearer token is reused
# from OPENAI_API_KEY when that is set.
# NER_API_BASE=http://vllm-router:4000       # or http://ner-only:8000 for a standalone service
# NER_TIMEOUT=120                            # Per-request timeout in seconds (per text chunk)
```

- [ ] **Step 3: CLAUDE.md — update prose**

Make these edits in `CLAUDE.md`:
- Overview (line ~7): change `GLiNER for named-entity recognition` to `an out-of-process /gliner HTTP service for named-entity recognition`.
- Pipeline step 3 (line ~70): change `word counts, GLiNER named entities, word clouds` to `word counts, named entities via the /gliner HTTP service, word clouds`.
- Key modules: add a line `- ` + backtick `nextext/core/ner.py` + backtick ` — named-entity-recognition agent: HTTP client for the out-of-process /gliner service.` and change the `nextext/core/words.py` line to `— NLP word-level analysis (spaCy word counts + word clouds).`
- Environment section: add two bullets mirroring `DIARIZE_API_BASE`:
  - `` - `NER_API_BASE` — root URL of the out-of-process `/gliner` NER service (e.g. `http://vllm-router:4000`); the client appends `/gliner`. Unset (default) disables NER, so transcripts carry no entity table. The bearer token is reused from `OPENAI_API_KEY`. `NER_TIMEOUT` — per-request (per-chunk) timeout in seconds (default `120`). ``
  - In the `MODEL_RESIDENCY_STRATEGY` bullet, remove `MODEL_RESIDENCY_GLINER, ` from the per-model overrides list.
- Memory management section: change the GPU-resident models list from ``(`whisper_turbo`, `gliner`)`` to ``(`whisper_turbo`)``.

- [ ] **Step 4: AGENTS.md — document the NER agent**

In `AGENTS.md`, update the "Word Intelligence" table row (line ~17) so the entity table is no longer attributed to `words.py`, and add a new agent row (matching the existing table's column format: Agent | Files | Inputs | Outputs | Entry points):

```markdown
| Named Entity Recognition | `nextext/core/ner.py` → `extract_entities`, `wordlevel_pipeline` | Transcript text | Entity table (`[Category, Entity, Frequency]`) via the out-of-process `/gliner` service | CLI `-w/--words`, Streamlit "Word-level analysis" |
```

Update the Word Intelligence row's Outputs cell to drop "entity table" (it now lists word counts, noun sentiment table, graph HTML, word cloud only). Leave other rows untouched.

- [ ] **Step 5: Verify docs don't break tooling**

Run: `uv run pytest`
Expected: PASS (docs-only changes; full suite still green).

Run: `pre-commit run --all-files`
Expected: PASS (ruff, mypy, docstrings, markdown hooks).

- [ ] **Step 6: Commit**

```bash
git add docker/compose.yaml .env.example CLAUDE.md AGENTS.md
git commit -m "docs: document HTTP NER and NER_API_BASE"
```

---

## Final Verification

- [ ] Run the full suite and report counts: `uv run pytest`
- [ ] Run the full gate: `pre-commit run --all-files`
- [ ] Confirm no `gliner` references remain in code: `grep -rniI "gliner" nextext/ tests/ pyproject.toml` returns nothing (docs/spec history aside).
- [ ] Manual smoke (optional, requires the service): with `NER_API_BASE=http://vllm-router:4000`, run `uv run nextext-cli -f <clip> -w` and confirm an entities artifact is produced; with `NER_API_BASE` unset, confirm the run completes with an empty entity table and a single warning.

---

## Self-Review

**Spec coverage:** §4 architecture → Tasks 2-3; §5.1 `ner.py` → Task 2; §5.2 config → Task 1; §5.3 pipeline → Task 3; §6 removal → Tasks 4-6; §8 config/compose/.env → Task 7; §9 docs → Task 7; §10 testing → tests in every task; §11 commit plan → Tasks 1-7 (commit 1 of the spec is split into Tasks 1-2 for finer granularity). All spec sections map to a task.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; deletion steps name exact symbols + line anchors and are backstopped by a `ruff F401` check.

**Type consistency:** `extract_entities(text: str, columns=None) -> pd.DataFrame` is defined identically in Task 2 and called as `extract_entities(word_analysis.text)` in Task 3 and patched as `fake_extract_entities(text)` in the Task 3 test. `NerConfig(api_base, api_key, timeout)` / `load_ner_env` / `DEFAULT_NER_TIMEOUT` names match across Tasks 1-2. `_NER_LABELS` is referenced by the Task 2 test exactly as defined.
