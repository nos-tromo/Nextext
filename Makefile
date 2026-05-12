# Build-host helpers for nextext.

.PHONY: network volumes bundle bundle-cuda build build-cuda up up-cuda stop stop-cuda

# Versioned image tag.
# On production: read from .nextext-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting NEXTEXT_VERSION before invoking make.
NEXTEXT_VERSION ?= $(shell \
    cat .nextext-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export NEXTEXT_VERSION

# Create the external Docker network (one-time per host; idempotent)
network:
	DOCKER_BUILDKIT=1 docker network create inference-net

# Create the external Docker volumes (one-time per host; idempotent).
volumes:
	./scripts/create_docker_volumes.sh

# Build CPU stack and ship as versioned .tar.gz pair (built + pulled).
bundle:
	./scripts/bundle_images.sh cpu

# Build CUDA stack and ship as versioned .tar.gz pair (built + pulled).
bundle-cuda:
	./scripts/bundle_images.sh cuda

# Build the CPU profile
build:
	DOCKER_BUILDKIT=1 docker compose --profile cpu build

# Build the CUDA profile
build-cuda:
	DOCKER_BUILDKIT=1 docker compose --profile cuda build

# Run the CPU profile (backend, frontend, qdrant) without building.
up:
	DOCKER_BUILDKIT=1 docker compose --profile cpu up -d --no-build

# Run the CUDA profile (backend-cuda, frontend-cuda) without building.
up-cuda:
	DOCKER_BUILDKIT=1 docker compose --profile cuda up -d --no-build

# Stop the CPU profile containers.
stop:
	docker compose --profile cpu stop

# Stop the CUDA profile containers.
stop-cuda:
	docker compose --profile cuda stop