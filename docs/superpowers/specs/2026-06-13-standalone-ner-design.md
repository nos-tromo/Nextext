# Standalone HTTP NER — Design Spec

- **Date:** 2026-06-13
- **Status:** Approved (pending spec review)
- **Topic:** Route named-entity recognition through an out-of-process `/gliner` HTTP service, mirroring the `/diarize` architecture, and remove the in-process GLiNER path.

> **Amendment 2026-06-14 (PR #37):** NER is no longer disabled when `NER_API_BASE`
> is unset. `load_ner_env()` now falls back to the central `OPENAI_API_BASE` (one
> trailing `/v1` stripped), so NER reaches `{central-root}/gliner` by default and is
> disabled only when *neither* `NER_API_BASE` nor `OPENAI_API_BASE` is set.
> Diarization and VAD gained the same central fallback. The "unset → disabled"
> statements below (Goals §2, §7, §12, the `load_ner_env` snippet) are updated to
> match; the per-chunk POSTs, client-side threshold, and output contract are
> unchanged.

## 1. Context & Motivation

Today NER is **in-process only**: `WordCounter.named_entity_recognition()`
(`nextext/core/words.py`) acquires the GPU-resident `gliner` model from the
process-wide registry, chunks the transcript, and calls
`predict_entities(chunk, LABELS, threshold=0.3)`. This forces the backend to
carry the `gliner` package, the model weights, and a non-trivial offline
model-loading apparatus.

Diarization already moved out-of-process: it runs against an HTTP `/diarize`
service selected by `DIARIZE_API_BASE`, with no in-process fallback (see
`nextext/core/diarization.py`). This spec applies the **same pattern** to NER so
Nextext can run "without vLLM"/without the local ML stack by pointing at a
standalone NER service, and so the backend image sheds the `gliner` dependency.

The standalone service already exists in the `nos-tromo/vllm-service` stack as
`vllm-service-gliner-1` and is reachable from the backend through the LiteLLM
router.

## 2. Goals & Non-Goals

**Goals**
- NER always runs over HTTP against a `/gliner` service selected by `NER_API_BASE`.
- When `NER_API_BASE` is unset, NER falls back to the central `OPENAI_API_BASE` (one trailing `/v1` stripped); it is disabled — empty entity table — only when neither is set. Diarization and VAD share the same fallback.
- Remove the in-process GLiNER machinery and drop the `gliner` dependency from the backend.
- Preserve the existing output contract: a `[Category, Entity, Frequency]` DataFrame consumed by the pipeline, CLI exporter, API schemas, and `nextext/app.py`.

**Non-Goals**
- No change to spaCy word counts, the word cloud, or any other `WordCounter` behavior.
- No new Docker compose service for NER (the service is external, like `/diarize`).
- No unrelated cleanup of pre-existing `AGENTS.md` staleness (e.g. "noun sentiment table / graph HTML" entries that no longer exist in `words.py`).
- `torch` is **not** removed — Whisper transcription still runs in-process and needs it.

## 3. Service Contract (verified 2026-06-13)

- **Endpoint:** `POST {NER_API_BASE}/gliner`
- **Request body (JSON):** `{"text": "<chunk>", "labels": ["person", "loc", ...]}`
- **Response (JSON):** `{"entities": [{"start": int, "end": int, "text": str, "label": str, "score": float}, ...]}`
- **Reachability:** from the backend via the router at `http://vllm-router:4000/gliner` (same host:port as `OPENAI_API_BASE`), or a standalone deployment at `http://<ner-host>:8000`. The router-proxied route returned `405` on GET and `422` on an empty POST (route live), and a well-formed POST returned scored entities.
- **Auth:** the `/gliner` passthrough was keyless in testing (returned `422`, not `401`). The client still sends `Authorization: Bearer {OPENAI_API_KEY}` when set, which is harmless and matches diarization.

## 4. Architecture

```
wordlevel_pipeline(text, language, …)        nextext/pipeline.py
  ├─ WordCounter.count_words()               spaCy — unchanged, in-process
  ├─ ner.extract_entities(text)              NEW — HTTP POST {NER_API_BASE}/gliner
  └─ WordCounter.create_wordcloud()          unchanged
```

A new stateless agent module `nextext/core/ner.py` is the NER analog of
`nextext/core/diarization.py`. `WordCounter` loses its NER responsibility and
keeps only word counts + word cloud.

## 5. Component Specs

### 5.1 `nextext/core/ner.py` (new)

Module-level constants (moved verbatim from `words.py`):
- `_NER_LABELS = ["date", "event", "fac", "group", "loc", "money", "org", "person", "time"]`
- `_NER_THRESHOLD = 0.3`
- `_NER_WORD_BUDGET = 512`
- `_SENTENCE_RE` (the existing sentence splitter)
- `_chunk_text(text, word_budget=_NER_WORD_BUDGET) -> list[str]` (moved verbatim)

```python
def extract_entities(text: str, columns: list[str] | None = None) -> pd.DataFrame:
    """POST transcript chunks to the /gliner service and tally entities.

    Returns a [Category, Entity, Frequency] DataFrame. Returns an empty frame
    (same columns) when NER_API_BASE is unset, the text is empty, or every
    request fails — NER failures must never abort a job.
    """
```

Behavior:
1. `config = load_ner_env()`. If `not config.api_base`: log a warning and return an empty `[Category, Entity, Frequency]` DataFrame (NER disabled). Mirrors `diarize_file` returning `[]`.
2. If `text` is empty/blank: return the empty frame.
3. For each `chunk` in `_chunk_text(text)`: `httpx.post(f"{config.api_base}/gliner", json={"text": chunk, "labels": _NER_LABELS}, headers=<bearer if key>, timeout=config.timeout)`, `raise_for_status()`, parse JSON.
4. For each entity: keep when `float(score) >= _NER_THRESHOLD` **and** `len(text.strip()) >= 3`; append `(label.strip().upper(), text.strip())`.
5. `Counter` the `(label, entity)` pairs → DataFrame `[Category, Entity, Frequency]`.
6. Error handling mirrors `diarization.py`: `httpx.HTTPStatusError` logs status + body snippet; `(httpx.HTTPError, ValueError, OSError)` logs the exception. On any error, that chunk contributes nothing; a total failure yields the empty frame. (Decision: per-chunk failures are logged and skipped so one bad chunk doesn't lose the whole transcript's entities.)

Design notes:
- **Per-chunk requests** (one POST per 512-word chunk) are intentional: the service is GLiNER under the hood and would truncate long input, hurting recall. Trade-off: N round-trips per file.
- **Threshold is applied client-side** on the returned `score`, so semantics are identical regardless of service internals.

### 5.2 `nextext/utils/env_cfg.py`

Add, mirroring `DiarizationConfig` / `load_diarization_env`:

```python
@dataclass(frozen=True)
class NerConfig:
    api_base: str
    api_key: str
    timeout: float

DEFAULT_NER_TIMEOUT: float = 120.0

def load_ner_env() -> NerConfig:
    api_base = os.getenv("NER_API_BASE", "").strip().rstrip("/") or _central_endpoint_root()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    # NER_TIMEOUT parsed like DIARIZE_TIMEOUT: positive float, else warn + default
    ...
    return NerConfig(api_base=api_base, api_key=api_key, timeout=timeout)
```

`NER_TIMEOUT` default is **120s** (diarize uses 600s; NER per-chunk requests are fast). Non-numeric/non-positive values warn and fall back, identical to `DIARIZE_TIMEOUT`.

### 5.3 `nextext/pipeline.py`

Add a module-level import beside the existing agent imports (mirroring
`pipeline.py:9` `from nextext.core.diarization import … diarize_file`):
```python
from nextext.core.ner import extract_entities
```
Then in `wordlevel_pipeline` (~line 218) replace:
```python
named_entities = word_analysis.named_entity_recognition()
```
with:
```python
named_entities = extract_entities(text)
```
(`text` is already the function's transcript input; NER needs no language — the service is multilingual, matching the old GLiNER call which also passed no language.) The module-level import means tests patch `nextext.pipeline.extract_entities`.

## 6. Removal Scope (in-process GLiNER)

- **`nextext/core/words.py`** — delete: `from gliner import GLiNER`; `REGISTRY`/`ModelSpec`/`Strategy` import; all `_GLINER_*` constants; the offline-cache/loader helpers (`_resolve_hf_cache_dir`, `_resolve_hf_cache_path`, `_load_gliner_config`, `_link_or_copy_path`, `_resolve_local_gliner_dependency`, `_materialize_offline_gliner_dir`, `_prepare_local_gliner_model_dir`, `_resolve_gliner_load_target`, `_load_gliner`, `_move_gliner`); the `REGISTRY.register(ModelSpec(name="gliner", …))` block; `_SENTENCE_RE` + `_chunk_text` (relocated to `ner.py`); and the `named_entity_recognition` method. Update the module docstring and the `WordCounter` class docstring (drop NER). Prune now-unused imports (`hashlib`, `json`, `re`, `shutil`, `tempfile`, `warnings`, `os`, `cast`, `Path` if unused) — verified by `ruff --fix` + `mypy`.
- **`nextext/utils/model_loader.py`** — delete `from gliner import GLiNER`, `GLINER_MODEL_ID`, `preload_gliner_model()`, and its call + failure handling in the preload routine.
- **`nextext/utils/model_registry.py`** — remove GLiNER from example docstrings (lines ~3, ~47). The registry stays (still hosts `whisper_turbo`).
- **`pyproject.toml`** — remove `"gliner>=0.2.0"`; re-lock `uv.lock` via `uv lock`.

## 7. Behavior When Unset / CLI Implications

- `NER_API_BASE` unset → falls back to the central `OPENAI_API_BASE`; only when neither is set do the backend and `nextext-cli` return an empty entities table (no entities artifact). Diarization behaves identically with its own central fallback.
- The CLI runs in-process, so it produces entities whenever an NER endpoint resolves — `NER_API_BASE` or the central `OPENAI_API_BASE`. Consistent with diarization/VAD, which share the same fallback.

## 8. Configuration Changes

- **Add:** `NER_API_BASE` (root URL; client appends `/gliner`), `NER_TIMEOUT` (default 120s).
- **Remove:** `NER_MODEL` and `MODEL_RESIDENCY_GLINER` (the service owns the model; no in-process GLiNER to keep resident).
- **`docker/compose.yaml`** — replace `NER_MODEL: ${NER_MODEL:-}` with `NER_API_BASE: ${NER_API_BASE:-}` (and `NER_TIMEOUT: ${NER_TIMEOUT:-}` if surfaced), mirroring the `DIARIZE_API_BASE` passthrough.
- **`.env.example`** — remove the `NER_MODEL` line, remove `MODEL_RESIDENCY_GLINER` and drop `gliner` from the "Supported names" comment; add an NER section mirroring the diarization block, with a `vllm-router:4000` router example and a standalone example.

## 9. Documentation Changes (`CLAUDE.md`, `AGENTS.md`)

- `CLAUDE.md`: overview line (GLiNER → out-of-process `/gliner` HTTP NER service); pipeline step 3; key-modules list (add `nextext/core/ner.py`, adjust `words.py`); Environment section (add `NER_API_BASE`/`NER_TIMEOUT`, drop `MODEL_RESIDENCY_GLINER`); Memory-management section (drop `gliner` from the registry-models list).
- `AGENTS.md`: add an NER agent entry and adjust the "Word Intelligence" row so the entity table is attributed to the new `/gliner` agent. Leave unrelated stale lines untouched.

## 10. Testing Strategy

- **`tests/test_ner.py` (new)** — mirror `tests/test_diarization.py` by monkeypatching `ner.httpx`:
  - success → correct `[Category, Entity, Frequency]` DataFrame;
  - `NER_API_BASE` unset → empty frame + warning, no HTTP call;
  - HTTP status error / transport error → empty frame, swallowed;
  - score `< 0.3` dropped; entity text `< 3` chars dropped;
  - label uppercasing and frequency tallying;
  - long text → multiple POSTs (chunking);
  - bearer header present when `OPENAI_API_KEY` set, absent otherwise.
- **`tests/test_env_cfg.py`** — add `load_ner_env` cases (api_base strip/rstrip, key reuse, `NER_TIMEOUT` parse + fallback), mirroring the diarization env tests.
- **`tests/test_pipeline.py`** — repoint the word-level test to `monkeypatch.setattr(pipeline, "extract_entities", …)` (mirroring the existing `diarize_file` patch at `test_pipeline.py:132`) instead of stubbing `WordCounter.named_entity_recognition`.
- **`tests/test_model_loader.py`** — remove the `preload_gliner_model` assertions.
- **`tests/test_model_registry.py`** — remove `test_gliner_spec_opts_out_of_mps` and the "registers the gliner spec" import comment; keep the `whisper_turbo` coverage.
- Full `uv run pytest`, then `pre-commit run --all-files` (ruff, mypy, docstrings) must be green.

## 11. Commit Plan (mirrors the diarization series)

1. `feat(ner): add /gliner HTTP client agent and config` — `nextext/core/ner.py`, `NerConfig`/`load_ner_env` in `env_cfg.py`, `tests/test_ner.py`, `tests/test_env_cfg.py`.
2. `refactor(pipeline): route NER through the /gliner service` — `nextext/pipeline.py`, `tests/test_pipeline.py`.
3. `refactor: remove in-process GLiNER NER` — `words.py`, `model_loader.py`, `model_registry.py`, `tests/test_model_loader.py`, `tests/test_model_registry.py`.
4. `chore(deps): drop the gliner dependency` — `pyproject.toml`, `uv.lock`.
5. `docs: document HTTP NER and NER_API_BASE` — `CLAUDE.md`, `.env.example`, `docker/compose.yaml`, `AGENTS.md`.

## 12. Risks & Rollback

- **Latency:** per-chunk requests add round-trips on long transcripts. Mitigation: keep chunk budget at 512; requests are individually fast. Revisit a batch endpoint only if the service grows one.
- **Behavioral change:** environments that ran in-process NER now reach the external `/gliner` service — via `NER_API_BASE` or, by default, the central `OPENAI_API_BASE`. Called out in docs and `.env.example`.
- **Rollback:** the change is a clean series of 5 commits; reverting commits 3–4 restores in-process GLiNER if needed.

## 13. Open Decisions (defaults chosen, override on review)

- Per-chunk POST requests (recall over round-trips) — **chosen**.
- `NER_TIMEOUT` default **120s**.
- Branch: NER work is logically separate from the current `feat/http-diarization` branch; a dedicated `feat/http-ner` branch is recommended before implementation.
