#!/usr/bin/env bash
set -euo pipefail

COMPOSE="docker compose --env-file .env -f docker/compose.yaml"

# Always compute a fresh version from git so repeated bundle runs produce
# distinct tags. Uses the commit date (not the build date) for reproducibility.
# Falls back to today's date when not in a git repo.
# .nextext-version (if present) is never used as input here — it is only
# written as output for production hosts.
# To pin a specific tag, set NEXTEXT_VERSION_OVERRIDE in your shell before
# invoking make.
if [[ -n "${NEXTEXT_VERSION_OVERRIDE:-}" ]]; then
  export NEXTEXT_VERSION="$NEXTEXT_VERSION_OVERRIDE"
else
  _git_sha=$(git rev-parse --short HEAD 2>/dev/null || true)
  _git_date=$(git log -1 --format=%cs 2>/dev/null || true)
  _date="${_git_date:-$(date +%Y-%m-%d)}"
  export NEXTEXT_VERSION="${_date}${_git_sha:+-${_git_sha}}"
fi
echo "NEXTEXT_VERSION=$NEXTEXT_VERSION"

# Persist the version so production hosts can run 'make up' without
# git or the original build date. Copy this file alongside docker/compose.yaml.
echo "$NEXTEXT_VERSION" > .nextext-version

# Build locally-defined services (backend + frontend).
$COMPOSE build

# Collect the built image names so docker save can bundle them. Every
# service in this compose file is locally built (backend + frontend);
# all model inference runs on external endpoints, so there are no
# stateful/remote images to pull.
built=()
while IFS= read -r img; do
  [[ -z "$img" ]] && continue
  built+=("$img")
done < <($COMPOSE config --images)

echo "Built images: ${built[*]:-<none>}"

if (( ${#built[@]} > 0 )); then
  docker save "${built[@]}" | gzip > "nextext-built-${NEXTEXT_VERSION}.tar.gz"
fi

echo "Wrote: nextext-built-${NEXTEXT_VERSION}.tar.gz"
