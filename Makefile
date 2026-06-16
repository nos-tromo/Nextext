# Build-host helpers for nextext.
#
# Single CPU-only image pair (backend + frontend): all model inference runs
# on external endpoints, so there is no cpu/cuda profile split.

.DEFAULT_GOAL := help

.PHONY: help network volumes build bundle up up-dev stop down logs pre-commit test

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

help:
	@echo "nextext — build-host helpers"
	@echo
	@echo "  make network    create the external inference-net"
	@echo "  make volumes    create the external Docker volumes"
	@echo "  make build      build the backend + frontend images"
	@echo "  make bundle     ship the built images as a versioned .tar.gz"
	@echo "  make up         run the stack (no rebuild, no host ports)"
	@echo "  make up-dev     like 'up', but publishes the frontend port on the host"
	@echo "  make stop       stop the containers"
	@echo "  make down       stop + remove the containers"
	@echo "  make logs       tail combined logs"
	@echo "  make pre-commit run pre-commit hooks (ruff + mypy)"
	@echo "  make test       run the test suite"

# Create the external Docker network (one-time per host; idempotent).
network:
	docker network create inference-net >/dev/null 2>&1 || true

# Create the external Docker volumes (one-time per host; idempotent).
volumes:
	@for volume_name in \
		nltk-cache \
		spacy-cache; do \
		docker volume create "$$volume_name" >/dev/null 2>&1 || true; \
		printf 'Ensured Docker volume exists: %s\n' "$$volume_name"; \
	done

# Build the backend + frontend images.
build:
	DOCKER_BUILDKIT=1 $(COMPOSE) build

# Build images and ship them as a versioned .tar.gz of locally-built images.
bundle:
	./scripts/bundle_images.sh

# Run the stack without rebuilding images (production shape, no host ports).
up:
	DOCKER_BUILDKIT=1 $(COMPOSE) up --no-build

# Like 'up' but layers compose.override.yaml on top to publish the
# frontend (React SPA) port on the host.
up-dev:
	DOCKER_BUILDKIT=1 $(COMPOSE_DEV) up --no-build

# Stop the containers.
stop:
	$(COMPOSE) stop

# Stop + remove the containers. All Nextext volumes are declared
# external, so this never destroys cached language resources.
down:
	$(COMPOSE) down

# Tail combined logs.
logs:
	$(COMPOSE) logs -f --tail=100

# Run pre-commit hooks (ruff + mypy).
pre-commit:
	uv run pre-commit run --all-files

# Run the test suite.
test:
	uv run pytest -q

