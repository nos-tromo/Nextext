# Nextext Documentation

This directory contains the in-repo reference manual for **Nextext**, the audio
and video transcription, translation, and analysis toolkit. It complements the
top-level [`README.md`](../README.md) (which focuses on prerequisites, the
quick start, and first run) with topic-by-topic deep dives.

## Table of contents

| Document | What it covers |
|---|---|
| [configuration.md](configuration.md) | Inference provider selection, dedicated per-model endpoints, localization, provider setup recipes, upload limits, job concurrency, sentence restoration, keyframes, the production sub-path, metrics |
| [architecture.md](architecture.md) | The backend/frontend service split, the in-memory job and anonymous-identity model, image tagging |
| [airgap.md](airgap.md) | Offline behaviour, preloading the language caches, `make bundle`, loading the tarballs on a disconnected host |
| [cli.md](cli.md) | `nextext-cli` — flags and directory batch processing |

Design history (dated plans and specs for individual features) lives alongside
this manual under `superpowers/`; those files record how a decision was made at
a point in time and are not kept current.

## Who this is for

- **Operators** deploying Nextext against their own inference stack — start
  with the top-level [`README.md`](../README.md) quick start, then
  [configuration.md](configuration.md) and, for disconnected hosts,
  [airgap.md](airgap.md).
- **Backend developers** extending the pipeline or the API — start with
  [architecture.md](architecture.md), then the module map in
  [`CLAUDE.md`](../CLAUDE.md).
- **Command-line users** processing local files or batches without running the
  stack — go straight to [cli.md](cli.md).

## Conventions used in these docs

- **Source references** use repo-relative paths (for example
  `nextext/core/diarization.py`) so editors can jump directly to the module.
- **Environment variables** are named exactly as they appear in
  [`.env.example`](../.env.example), which stays the canonical annotated list;
  these docs explain only the groups that need more than a one-line comment.
- **Endpoint paths** are written as the client sees them — `/api/v1/...` for
  the application API, bare `/metrics` for the Prometheus exposition mounted at
  the app root.
- Documentation is plain Markdown (GitHub Flavored). No docs build step is
  required.
