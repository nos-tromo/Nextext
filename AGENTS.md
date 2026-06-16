# Nextext Agents

Nextext breaks the speech-to-insight workflow into specialized agents (self-contained modules) coordinated by the shared pipelines in `nextext/pipeline.py`. Each agent consumes a narrow input, produces a deterministic artifact, and can be toggled on/off from three orchestrators:

- The CLI (`nextext/cli.py`) — runs the pipeline in-process; no backend required.
- The FastAPI backend (`nextext/api/`) — wraps the same pipeline behind a job-based HTTP API (`POST /api/v1/jobs` → SSE `/events` → `/artifacts/{name}`).
- The React frontend (`frontend/`) — a static SPA served by nginx that talks to `/api/v1` same-origin; never imports the pipeline directly.

This document describes every agent, how they interact, and what they expect from the runtime environment.

## Agent Directory

| Agent | Module & entry point | Consumes | Produces | Activated by |
| --- | --- | --- | --- | --- |
| Transcription & Diarization | `nextext/core/transcription.py` → `ExternalWhisperTranscriber`, `transcription_pipeline`; `nextext/core/diarization.py` → `diarize_file`, `assign_speakers_by_overlap` | Audio file path, Whisper endpoint config, speaker count (`DIARIZE_API_BASE` for diarization) | Timestamped transcript `pd.DataFrame` (`start`, `end`, `speaker`, `text`) | Always-on |
| Translation | `nextext/core/translation.py` → `Translator`, `translation_pipeline` | Transcript `pd.DataFrame`, source and target ISO codes, inference provider | Mutated `pd.DataFrame` translated segment-wise | CLI `--task translate`, React UI task option |
| Word Intelligence | `nextext/core/words.py` → `WordCounter`, `wordlevel_pipeline` | Transcript text, resolved language | Word counts, noun sentiment table, graph HTML, word cloud `Figure` | CLI `-w/--words`, React UI word-level analysis option |
| Named Entity Recognition | `nextext/core/ner.py` → `extract_entities` (called by `wordlevel_pipeline`) | Transcript text | Entity table (`[Category, Entity, Frequency]`) via the out-of-process `/gliner` service | CLI `-w/--words`, React UI word-level analysis option |
| Summarization | `nextext/pipeline.py` → `summarization_pipeline` + `InferencePipeline` | Full transcript text | Summary string in the configured output language | CLI `-sum/--summarize`, React UI summarization option |
| Hate Speech Detection | `nextext/core/hate_speech.py` → `HateSpeechDetector`, `hate_speech_pipeline` | Transcript `pd.DataFrame`, inference provider | List of flagged segment dicts (`hate_speech`, `category`, `confidence`, `reason`, `text`) | CLI `-hs/--hate-speech`, React UI hate speech option |
| File Export | `nextext/core/processing.py` → `FileProcessor.write_file_output` | Any agent result | Files in `output/<input-file>/` (`.txt`, `.csv`, `.xlsx`, `.png`) | CLI workflow |
| Inference Service | `nextext/core/openai_cfg.py` → `InferencePipeline` | Prompt template + runtime options | Raw string from an OpenAI-compatible chat endpoint | Shared dependency for translation, summary, and hate speech agents |

## Execution Flow

1. **Interface orchestrator** (CLI or React frontend via the backend API) collects user options and instantiates `transcription_pipeline`.
2. **Transcription agent** returns a `pd.DataFrame` and detected language code; the orchestration state is updated with the resolved language.
3. **Optional agents** (translation, word-level, summary) fire in the order shown in `nextext/cli.py` (CLI) or the backend job worker (`nextext/api/jobs.py`), each mutating or extending the transcript payload.
4. **Outputs** are held in the backend job store for the UI (served via `/api/v1/jobs/{id}/artifacts/{name}`) or routed through the `FileProcessor` for CLI batch exports.

> The agents themselves remain stateless; any UI or CLI state is maintained by the orchestrator wrappers.

## Transcription & Diarization Agent

- **Key files:** `nextext/core/transcription.py`, `nextext/core/vad.py`, `nextext/core/diarization.py`, `transcription_pipeline()` (`nextext/pipeline.py`).
- **Responsibilities:** Screen the file with the external `/vad` speech guard, forward it to an OpenAI-compatible `/v1/audio/transcriptions` endpoint, optionally label speakers via the out-of-process `/diarize` HTTP service (`n_speakers > 1`), and emit a normalized DataFrame used by every downstream agent.
- **Inputs:** `Path` to audio/video, optional source code, speaker count. (Whisper always transcribes; the `transcribe`/`translate` task is a job-level flag that gates the downstream Translation agent, not a transcription input.)
- **Outputs:** `pd.DataFrame` with `start`, `end`, `speaker`, `text`; the source language resolves from the API response (`ExternalWhisperTranscriber.src_lang`).
- **Endpoints:** Whisper resolves via `load_whisper_env()` — `WHISPER_API_BASE`/`WHISPER_API_KEY` with central `OPENAI_API_BASE`/`OPENAI_API_KEY` fallback; model defaults `whisper-1` (openai) / `openai/whisper-large-v3` (vllm), overridable via `WHISPER_MODEL`; `INFERENCE_PROVIDER=ollama` requires both explicitly. NER, diarization, and VAD likewise default to the central endpoint (one trailing `/v1` stripped) when their dedicated `*_API_BASE` is unset. The VAD guard posts the file to `{VAD_API_BASE}/vad` (contract in `nextext/core/vad.py`); it is fail-open — any error transcribes the file anyway, and `VAD_API_BASE=off` switches the guard off. Diarization posts the file to `{DIARIZE_API_BASE}/diarize` (contract in `nextext/core/diarization.py`); it is fail-soft — when no endpoint resolves or the request fails it logs and continues without speaker labels.
- **Dependencies:** `openai` SDK, `httpx`. No torch, no GPU, no local audio tooling.
- **Operational notes:** Speech-free audio (per the `/vad` service) never reaches the remote endpoint; surviving segments pass a `no_speech_prob` post-filter. Diarization assigns speakers via maximum segment overlap (`assign_speakers_by_overlap`) and is skipped when `n_speakers == 1` or no segments survived transcription. The speaker column is omitted when no segment was labelled.

## Translation Agent

- **Key files:** `nextext/core/translation.py`, `translation_pipeline()` (`nextext/pipeline.py`).
- **Responsibilities:** Detect the transcript language via `langdetect` when needed, prompt the shared chat model via the inference service, and rewrite each transcript segment to the requested target language.
- **Inputs:** Transcript `pd.DataFrame` (only the `text` column is used), target ISO 639-1 code, optional resolved source code from transcription, shared `InferencePipeline`.
- **Outputs:** In-place replacement of the `text` column; `Translator.src_lang` is populated for logging and downstream toggles.
- **Dependencies:** `langdetect`, `pycountry`, plus an OpenAI-compatible chat completions backend selected by `INFERENCE_PROVIDER` (`ollama` by default, `vllm`, or `openai`).
- **Operational notes:** This is the only translation path — Whisper's audio-translate task is no longer used — so English targets are translated here too. Translation is skipped only when the resolved source language already equals the target (see `should_translate()` in `nextext/pipeline.py`). Translation runs on `TEXT_MODEL` — the same model used for summarization — over the templated prompt in `nextext/utils/prompts/translation.txt` plus a translation system prompt, identically for every `INFERENCE_PROVIDER`.

## Word Intelligence Agent

- **Key files:** `nextext/core/words.py`, `wordlevel_pipeline()` (`nextext/pipeline.py`), spaCy mappings in `nextext/utils/mappings/spacy_models.json`.
- **Responsibilities:** Turn transcript text into linguistic diagnostics—top words, named entities (via the out-of-process GLiNER `/gliner` service), and a matplotlib word cloud.
- **Inputs:** `pd.DataFrame` with `text`, resolved language code (source for `transcribe`, target for `translate`).
- **Outputs:** Tuple `(word_counts_df, entities_df, wordcloud_figure)` — the entity DataFrame is populated by the out-of-process NER agent (`nextext/core/ner.py`), empty when no NER endpoint resolves (no dedicated `NER_API_BASE` and no central `OPENAI_API_BASE`).
- **Dependencies:** `spacy`, `matplotlib`, `camel_tools` (pure-Python tokenizer only), `arabic_reshaper`; fonts loaded via `nextext/utils/font_loader.py` to keep multilingual rendering stable. NER runs remotely through `nextext/core/ner.py` (`POST {NER_API_BASE}/gliner`) — fail-soft: errors degrade to an empty entity table.
- **Operational notes:** Call `text_to_doc()` then `lemmatize_doc()` before counting; spaCy models download on demand only when `NEXTEXT_OFFLINE=0` (offline + uncached raises actionably). `extract_entities` chunks text into ≤512-word sentence-packed chunks before each `/gliner` request.


## Summarization Agent

- **Key files:** `summarization_pipeline()` (`nextext/pipeline.py`), prompt template `nextext/utils/prompts/summary.txt`.
- **Responsibilities:** Format the entire transcript as a prompt, apply the system instruction defined in `InferencePipeline`, and return a concise summary limited to 15 sentences.
- **Inputs:** Concatenated transcript string, `InferencePipeline`.
- **Outputs:** Summary string; orchestrators attach it to session state or export it via `FileProcessor`.
- **Dependencies:** OpenAI-compatible chat completions via the backend selected by `INFERENCE_PROVIDER` (Ollama on `http://localhost:11434/v1` by default, a LiteLLM-fronted vLLM stack, or the hosted OpenAI API); prompt language controlled by `InferencePipeline.out_language`. Summarization always uses the templated prompt + system message regardless of provider.
- **Operational notes:** The pipeline raises `ValueError` when text is empty; make sure optional agents check for data before calling.

## Hate Speech Detection Agent

- **Key files:** `nextext/core/hate_speech.py`, `hate_speech_pipeline()` (`nextext/pipeline.py`), prompt template `nextext/utils/prompts/hate_speech.txt`.
- **Responsibilities:** Iterate over each transcript segment, send it to an LLM with the hate speech prompt, and return a list of flagged entries with structured metadata.
- **Inputs:** Transcript `pd.DataFrame` (only the `text` column is used), shared `InferencePipeline`, optional `max_chars` limit (default 2048).
- **Outputs:** `list[dict]` — each entry contains `hate_speech=True`, `category`, `confidence` (`high`/`medium`/`low`), `reason`, and the original `text`. Only flagged segments are included; an empty list means no hate speech was found.
- **Categories:** racism, sexism, homophobia, religious hatred, xenophobia, disability discrimination, none.
- **Dependencies:** OpenAI-compatible chat completions via `InferencePipeline`; same provider configuration as translation and summarization.
- **Operational notes:** The LLM response is parsed as JSON with regex fallback for prose-wrapped output. Parsing failures produce a safe default (`hate_speech=False`). Text is truncated to `max_chars` before sending to the model. CLI results are exported as a CSV via `FileProcessor`; the React UI downloads the hate speech CSV via the backend artifacts endpoint.

## Inference Service Agent

- **Key files:** `nextext/core/openai_cfg.py` and prompts directory `nextext/utils/prompts/`.
- **Responsibilities:** Construct prompts, resolve the configured `TEXT_MODEL`, create an OpenAI-compatible client, perform provider health checks, and expose `call_model()` to translation and summarization.
- **Inputs:** Prompt keyword (`system`, `summary`, `translation`, `hate_speech`), runtime options (temperature, stop tokens, max tokens, `include_system_prompt`), provider configuration (`INFERENCE_PROVIDER`, `OPENAI_API_BASE`, `OPENAI_API_KEY`).
- **Outputs:** Raw string response from the configured chat completion endpoint; `sys_prompt` ensures outputs are emitted in the configured language when the system role is included.
- **Operational notes:** Ollama remains the default provider and is reached through its `/v1/chat/completions` compatibility layer. `INFERENCE_PROVIDER=vllm` targets a LiteLLM-fronted `nos-tromo/vllm-service` stack (LiteLLM dispatches by `model` field, so `TEXT_MODEL` must be registered on the endpoint). `INFERENCE_PROVIDER=openai` targets the hosted OpenAI API. `call_model(include_system_prompt=False)` sends a single user message for models or endpoints that accept only `user`/`assistant` roles; all standard callers keep the default system prompt. `OPENAI_API_KEY` is required for every provider. `OLLAMA_THINK` (tri-state: `1`/`true`/`yes`/`on` to enable, `0`/`false`/`no`/`off` to disable, unset to omit) sets a process-wide default for the Ollama `think` request field; `call_model(think=...)` overrides it per call. Forwarded via `extra_body`, so it is a no-op on vLLM/OpenAI.

## File Export Agent

- **Key files:** `nextext/core/processing.py`.
- **Responsibilities:** Create `output/<input-file>/` directories and serialize any agent output (text, list, DataFrame, Matplotlib figure) to disk.
- **Inputs:** Label describing the payload, optional target language suffix.
- **Outputs:** `.txt`, `.csv`, `.xlsx`, or `.png` files depending on type; log statements confirm each save path.
- **Operational notes:** Only the CLI uses `FileProcessor`; the React UI downloads artifacts via the backend's `/api/v1/jobs/{id}/artifacts/{name}` endpoints, which materialise them on demand from the in-memory job result.

## Orchestrators (CLI & React frontend)

- **React frontend (`frontend/`):** A static SPA served by nginx. The user selects options in the upload form, the browser posts the file to `POST /api/v1/jobs` (same-origin, proxied by nginx), and subscribes to the SSE event stream (`GET /api/v1/jobs/{id}/events`) for live progress. Completed artifacts are downloaded from `/api/v1/jobs/{id}/artifacts/{name}`. The frontend never imports the Python pipeline.
- **CLI (`nextext/cli.py`):** `parse_arguments()` maps command-line flags to agent toggles. `main()` wires the pipelines sequentially and persists outputs via `FileProcessor`.
- **Shared behaviour:** Both orchestrators guard optional agents behind flags, update the resolved language after transcription, and instantiate a shared `InferencePipeline` when translation or summarization is requested.

## Adding or Modifying Agents

1. **Create the module:** Place new agent code under `nextext/core/` (or `nextext/pipeline.py` if it is a thin wrapper). Keep the public function signature narrow and return plain Python or pandas objects.
2. **Expose a pipeline hook:** Add a helper function in `nextext/pipeline.py` so both orchestrators can call the agent without duplicating logic.
3. **Wire toggles:** Update the backend job worker (`nextext/api/jobs.py`) to honour the new option flag, and add the corresponding UI option in the React frontend (`frontend/src/`) and CLI argument in `nextext/cli.py`.
4. **Document configuration:** Extend this file and, if needed, add prompt templates or mapping entries under `nextext/utils/`.
5. **Persist outputs:** Decide whether the result is UI-only or needs a saved artifact; if so, register it in `nextext/api/artifacts.py` (for the React frontend) and `FileProcessor.write_file_output()` (for CLI) to follow the existing naming convention.

By keeping each agent isolated and documented here, Nextext can scale its audio-analysis capabilities without creating tight coupling between new features and the React frontend or CLI.
