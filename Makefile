# Build-host helpers for nextext.
#
# The Docker profile (cpu/cuda) is read from PROFILE in .env, so plain
# `make up` follows the host's hardware. Override per-invocation with
# `make up PROFILE=cuda`.

.DEFAULT_GOAL := help

.PHONY: help network volumes build bundle up up-dev stop down logs pre-commit test

# Docker profile (cpu/cuda). Read from .env, default cpu. Override on the
# command line: make up PROFILE=cuda
PROFILE ?= $(or $(strip $(shell test -f .env && grep -E '^PROFILE=' .env | cut -d= -f2)),cpu)

# Versioned image tag.
# On production: read from .nextext-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting NEXTEXT_VERSION before invoking make.
NEXTEXT_VERSION ?= $(shell \
    cat .nextext-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export NEXTEXT_VERSION

COMPOSE      := docker compose --env-file .env -f docker/compose.yaml
COMPOSE_DEV  := docker compose --env-file .env -f docker/compose.yaml -f docker/compose.override.yaml
PROFILE_FLAG := --profile $(PROFILE)

help:
	@echo "nextext — build-host helpers. Active profile: $(PROFILE)"
	@echo
	@echo "  make network    create the external inference-net"
	@echo "  make volumes    create the external Docker volumes"
	@echo "  make build      build images for the $(PROFILE) profile"
	@echo "  make bundle     ship images as a versioned .tar.gz pair ($(PROFILE))"
	@echo "  make up         run the $(PROFILE) profile (no rebuild, no host ports)"
	@echo "  make up-dev     like 'up', but publishes the frontend port on the host"
	@echo "  make stop       stop the $(PROFILE) profile containers"
	@echo "  make down       stop + remove the $(PROFILE) profile containers"
	@echo "  make logs       tail combined logs for the $(PROFILE) profile"
	@echo "  make pre-commit run pre-commit hooks (ruff + mypy)"
	@echo "  make test       run the test suite"
	@echo
	@echo "Set PROFILE=cpu|cuda in .env, or override: make up PROFILE=cuda"

# Create the external Docker network (one-time per host; idempotent).
network:
	docker network create inference-net >/dev/null 2>&1 || true

# Create the external Docker volumes (one-time per host; idempotent).
volumes:
	./scripts/create_docker_volumes.sh

# Build images for the active profile.
build:
	DOCKER_BUILDKIT=1 $(COMPOSE) $(PROFILE_FLAG) build

# Build images and ship as a versioned .tar.gz pair (built + pulled).
bundle:
	./scripts/bundle_images.sh $(PROFILE)

# Run the active profile without rebuilding images (production shape, no host ports).
up:
	DOCKER_BUILDKIT=1 $(COMPOSE) $(PROFILE_FLAG) up --no-build

# Like 'up' but layers compose.override.yaml on top to publish the
# Streamlit frontend port on the host.
up-dev:
	DOCKER_BUILDKIT=1 $(COMPOSE_DEV) $(PROFILE_FLAG) up --no-build

# Stop the active profile's containers.
stop:
	$(COMPOSE) $(PROFILE_FLAG) stop

# Stop + remove the active profile's containers. All Nextext volumes
# are declared external, so this never destroys cached models or data.
down:
	$(COMPOSE) $(PROFILE_FLAG) down

# Tail combined logs for the active profile.
logs:
	$(COMPOSE) $(PROFILE_FLAG) logs -f --tail=100

# Run pre-commit hooks (ruff + mypy).
pre-commit:
	uv run pre-commit run --all-files

# Run the test suite.
test:
	uv run pytest -q
