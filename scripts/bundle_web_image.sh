#!/usr/bin/env bash
# Save the nextext-web image to a gzipped tarball for airgapped delivery.
# Mirrors the existing image-bundle flow; deps are baked at build time.
set -euo pipefail

VERSION="${NEXTEXT_VERSION:-$(cat .nextext-version 2>/dev/null || date +%Y-%m-%d)}"
IMAGE="nextext-web:${VERSION}"
OUT="nextext-web-${VERSION}.tar.gz"

echo "Building ${IMAGE}…"
DOCKER_BUILDKIT=1 docker compose --env-file .env -f docker/compose.yaml build web

echo "Saving ${IMAGE} -> ${OUT}…"
docker save "${IMAGE}" | gzip > "${OUT}"
echo "Wrote ${OUT}"
