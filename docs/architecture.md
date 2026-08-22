# Architecture

Nextext runs as two cooperating containers on the `nextext-net` compose
network, plus the external seams it joins (`inference-net` for the model
endpoints, `edge-net` for the production gateway).

## Service split

The stack brings up two containers:

- **Backend** (`backend`) — FastAPI on port 8000 (internal). Owns the pipeline, the HTTP inference clients, and the in-memory job store. Exposes `/api/v1/health`, `/api/v1/languages`, `/api/v1/jobs/*`. Not published to the host by default.
- **Frontend** (`frontend`) — React SPA served by nginx on port 80 (internal). nginx proxies `/api/v1` to the backend same-origin, so browser uploads stream through nginx without buffering whole files in Python.

With `make up-dev`, the frontend is published on `http://localhost:${NEXTEXT_HOST_PORT:-8501}/` (nginx → React SPA).

`nextext-cli` keeps a third, container-free path: it imports the pipeline
directly and runs end-to-end in-process, without a backend. It ships inside the
backend image alongside the API — see [cli.md](cli.md).

## Jobs and identity

Jobs live only in memory — there is no durable storage and no TTL, so a long-running job is never cut off and is retained until you delete it or the backend restarts. Identity is anonymous: the frontend mints a per-browser id and stamps it into the URL (`?owner=<id>`) on first visit, sending it to the backend as the trusted identity header (`X-Auth-User` by default) to scope your jobs. Because that id survives a refresh, reloading the page mid-run re-discovers your jobs and resumes the live progress view; closing the tab and reopening the bare host starts a fresh identity. Developers calling the API directly can skip the header and set `NEXTEXT_DEFAULT_IDENTITY` instead. There is no authentication — the backend trusts whoever can reach `inference-net`.

Cross-owner reads return `404` rather than `403`, so the existence of another
owner's job never leaks. In production the `edge-plane` gateway is what injects
the trusted header; see
[configuration.md](configuration.md#production-sub-path).

## Image tagging

Each build is tagged `nextext-{backend,frontend}:${NEXTEXT_VERSION}`, where
`NEXTEXT_VERSION` defaults to `latest`. Override it (e.g. for releases)
by exporting `NEXTEXT_VERSION` before running `make` (or a raw
`docker compose -f docker/compose.yaml` invocation).

When invoked through `make`, `NEXTEXT_VERSION` defaults to
`YYYY-MM-DD-<short-sha>` so each build gets a traceable tag. Export
`NEXTEXT_VERSION=…` beforehand to pin a specific version.
