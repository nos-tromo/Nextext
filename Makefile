# Build-host helpers for nextext.
#
# Single CPU-only image pair (backend + frontend): all model inference runs
# on external endpoints, so there is no cpu/cuda profile split.
#
# The compose lifecycle (network/volumes/build/bundle/up/up-dev/stop/down/
# logs/pre-commit/test) + the versioned image tag come from make/common.mk,
# vendored from nos-tromo/.github. Only nextext-specific config and the help
# text live here.

.DEFAULT_GOAL := help

REPO     := nextext
NETWORKS := inference-net edge-net
VOLUMES  := spacy-cache
include make/common.mk

.PHONY: help preload cli

help:
	@echo "nextext — build-host helpers"
	@echo
	@echo "  make network    create the external inference-net + edge-net"
	@echo "  make volumes    create the external Docker volumes"
	@echo "  make build      build the backend + frontend images"
	@echo "  make bundle     ship the built images as a versioned .tar.gz (latest annotated release tag)"
	@echo "  make bundle-dev like 'bundle', but from the current working tree (dev/soak)"
	@echo "  make up         run the stack detached (no build, no host ports)"
	@echo "  make up-dev     like 'up' (detached, no build); publishes the frontend port on the host"
	@echo "  make dev        build, then up-dev"
	@echo "  make stop       stop the containers"
	@echo "  make down       stop + remove the containers"
	@echo "  make logs       tail combined logs"
	@echo "  make preload    fill the spacy-cache volume (connected host, one-off container)"
	@echo "  make cli ARGS='-f /tmp/<file> ...' CASE_REF=... REASON=..."
	@echo "                  run nextext-cli inside the running backend (audited; the backend must be up;"
	@echo "                  copy the file in first: docker compose --env-file .env -f docker/compose.yaml cp <file> backend:/tmp/)."
	@echo "                  CASE_REF and/or REASON state the run's purpose, which the default"
	@echo "                  AUDIT_REQUIRE_PURPOSE=query,disclose requires (without one the run exits 2), e.g."
	@echo "                  make cli ARGS='-f /tmp/clip.mp4 -sum' CASE_REF=case-0001 REASON='Review of a reported clip'"
	@echo "  make pre-commit run pre-commit hooks (ruff + pyrefly)"
	@echo "  make verify     pre-push gate: pre-commit + frontend lint/build; mirrors CI's lint gate"
	@echo "  make test       run pytest + vitest (test-backend / test-frontend for one)"

# Fill the spacy-cache volume on a connected host. Runs through compose so the
# volume-permissions service chowns the fresh (root-owned) volume first.
preload:
	$(COMPOSE) run --rm --no-TTY -e NEXTEXT_OFFLINE=0 backend load-models

# Run nextext-cli inside the running backend, not a one-off container, so the
# run's audit chain lands in the backend's own log (AUDIT_OUTPUT=container) —
# a one-off container can exit before the log collector sees it. The run is
# attributed to the host account that started it; CASE_REF / REASON state its
# purpose. Every run writes output files (a disclosure), so under the compose
# default AUDIT_REQUIRE_PURPOSE=query,disclose a run without either exits 2.
# The file must be inside the container: /tmp is the writable scratch volume,
# and outputs land under /tmp/output/<stem>/.
cli:
	$(COMPOSE) exec -w /tmp \
		-e AUDIT_CLI_USER="$${SUDO_USER:-$$(id -un)}" \
		-e AUDIT_CASE_REF="$(CASE_REF)" -e AUDIT_REASON="$(REASON)" \
		backend nextext-cli $(ARGS)
