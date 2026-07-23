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
VOLUMES  := nltk-cache spacy-cache
include make/common.mk

.PHONY: help

help:
	@echo "nextext — build-host helpers"
	@echo
	@echo "  make network    create the external inference-net"
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
	@echo "  make pre-commit run pre-commit hooks (ruff + pyrefly)"
	@echo "  make verify     pre-push gate: pre-commit + frontend lint/build; mirrors CI's lint gate"
	@echo "  make test       run the test suite"
