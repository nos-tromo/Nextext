# Build-host helpers for nextext.

.PHONY: volumes build-cpu build-cuda bundle-cpu bundle-cuda no-build-cpu no-build-cuda up-cpu up-cuda

# Versioned image tag.
# On production: read from .nextext-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting NEXTEXT_VERSION before invoking make.
NEXTEXT_VERSION ?= $(shell \
    cat .nextext-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export NEXTEXT_VERSION

# Create the external Docker volumes (one-time per host; idempotent).
volumes:
	./scripts/create_docker_volumes.sh

# Build the CPU profile
build-cpu:
	DOCKER_BUILDKIT=1 docker compose --profile cpu build

# Build the CUDA profile
build-cuda:
	DOCKER_BUILDKIT=1 docker compose --profile cuda build

# Build CPU stack and ship as versioned .tar.gz pair (built + pulled).
bundle-cpu:
	./scripts/bundle_images.sh cpu

# Build CUDA stack and ship as versioned .tar.gz pair (built + pulled).
bundle-cuda:
	./scripts/bundle_images.sh cuda

# Run the CPU profile (backend-cpu, frontend-cpu, qdrant-cpu) without building.
no-build-cpu:
	DOCKER_BUILDKIT=1 docker compose --profile cpu up -d --no-build

# Run the CUDA profile (backend-cuda, frontend-cuda) without building.
no-build-cuda:
	DOCKER_BUILDKIT=1 docker compose --profile cuda up -d --no-build

# Build and run the CPU profile (backend-cpu, frontend-cpu, qdrant-cpu).
up-cpu:
	DOCKER_BUILDKIT=1 docker compose --profile cpu up

# Build and run the CUDA profile (backend-cuda, frontend-cuda, qdrant-cuda).
up-cuda:
	DOCKER_BUILDKIT=1 docker compose --profile cuda up
