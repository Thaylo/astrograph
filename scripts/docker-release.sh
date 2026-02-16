#!/usr/bin/env bash
set -euo pipefail

# Build and push multi-arch Docker image for astrograph.
# Requires: docker buildx, a builder with linux/amd64+linux/arm64 support.
#
# Usage:
#   ./scripts/docker-release.sh          # build + push :VERSION and :latest
#   ./scripts/docker-release.sh --dry    # build only, no push

IMAGE="thaylo/astrograph"
PLATFORMS="linux/amd64,linux/arm64"

VERSION=$(python3 -c "
import tomllib, pathlib
data = tomllib.loads(pathlib.Path('pyproject.toml').read_text())
print(data['project']['version'])
")

DRY=false
if [[ "${1:-}" == "--dry" ]]; then
    DRY=true
fi

echo "==> Building ${IMAGE}:${VERSION} for ${PLATFORMS}"

PUSH_FLAG=()
if [[ "$DRY" == false ]]; then
    PUSH_FLAG=(--push)
fi

docker buildx build \
    --platform "${PLATFORMS}" \
    --tag "${IMAGE}:${VERSION}" \
    --tag "${IMAGE}:latest" \
    "${PUSH_FLAG[@]}" \
    .

if [[ "$DRY" == true ]]; then
    echo "==> Dry run complete (not pushed)"
else
    echo "==> Pushed ${IMAGE}:${VERSION} and ${IMAGE}:latest"
fi
