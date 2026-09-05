# Air-gapped delivery

Nextext is built to ship to hosts with no internet access. Nothing is fetched
at runtime: all model inference is an HTTP call to an endpoint you already
operate, and the only assets downloaded locally are the spaCy language
resources, which are preloaded ahead of time and shipped as a Docker volume.

## Offline usage

`NEXTEXT_OFFLINE=1` is the default: spaCy downloads are skipped and an
uncached spaCy model raises an actionable error instead of attempting a
doomed download. Fill the cache on a connected host:

```bash
make preload
```

`load-models` preloads the configured spaCy packages — the only assets
fetched locally; all model inference is remote. It is a download-once step:
the packages live in the external `spacy-cache` volume (`make volumes`) and
survive image upgrades, so a release bundle never carries them.

`make preload` runs the command in a one-off container **through compose**,
which matters on a fresh volume: a newly created Docker volume is root-owned
and the backend runs as uid 10001, so the compose `volume-permissions`
service has to chown it first. A bare `docker run` skips that and every
install fails on permissions. On a `uv run` host, run the command directly
instead:

```bash
NEXTEXT_OFFLINE=0 uv run load-models
```

Alternatively, ship the populated `spacy-cache` volume alongside the image
bundle.

## Bundling the images

To ship Nextext to a host without internet access, run the bundler on a
machine that *does* have access:

```bash
make bundle
```

The script builds the local Nextext image, pulls any externally hosted
images referenced by the compose file, and writes them to two versioned
tarballs in the project root:

- `nextext-built-{version}.tar.gz` — locally built Nextext images
- `nextext-pulled-{version}.tar.gz` — images pulled from registries

`make bundle` is the **production** shape: it builds the latest annotated
release tag reachable from `HEAD` (checking it out, building, then restoring
your branch) and refuses on a dirty tree or when no tag is reachable, so a
production artifact is always tag-versioned. `make bundle-dev` bundles the
current working tree as-is for dev iteration and staging soak.

## Loading on the target host

Copy the tarballs (and your `.env` plus the `docker/` directory) to the
target host, load them, and bring up the stack without rebuilding. The
target host runs the production shape — `docker/compose.yaml` without the
dev override — so no host ports are published:

```bash
docker load < nextext-built-<version>.tar.gz
docker load < nextext-pulled-<version>.tar.gz   # may be empty for the default compose
export NEXTEXT_VERSION=<version>
docker compose --env-file .env -f docker/compose.yaml up -d --no-build
```
