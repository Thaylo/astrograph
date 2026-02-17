#!/usr/bin/env bash
set -euo pipefail

# ─── ASTrograph Release Script ───────────────────────────────────────────────
#
# Automates the full release pipeline:
#   1. Bump version in pyproject.toml + __init__.py
#   2. Lock dependencies
#   3. Run tests + pre-commit
#   4. Commit, tag, push
#   5. Build multi-arch Docker images (arm64 on Mac, amd64 on Ubuntu)
#   6. Push to Docker Hub with a multi-arch manifest
#
# Usage:
#   ./scripts/release.sh 0.5.72           # full release
#   ./scripts/release.sh 0.5.72 --dry     # stop before git push + docker
#   ./scripts/release.sh 0.5.72 --docker  # skip git, docker only
#
# Prerequisites:
#   - Docker Hub auth on Mac (docker login)
#   - Docker Hub auth on Ubuntu (docker login)
#   - SSH access: thaylo@192.168.15.10
#   - Repo clone on Ubuntu at ~/Projects/astrograph
# ──────────────────────────────────────────────────────────────────────────────

IMAGE="thaylo/astrograph"
REMOTE_HOST="thaylo@192.168.15.10"
REMOTE_REPO="~/Projects/astrograph"
PYPROJECT="pyproject.toml"
INIT_PY="src/astrograph/__init__.py"

# ─── Parse arguments ─────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <version> [--dry|--docker]"
    echo ""
    echo "  <version>   Semantic version (e.g. 0.5.72)"
    echo "  --dry       Validate everything but don't push or build docker"
    echo "  --docker    Skip git steps, build and push docker only"
    exit 1
fi

VERSION="$1"
MODE="${2:-full}"

if [[ "$MODE" != "--dry" && "$MODE" != "--docker" && "$MODE" != "full" ]]; then
    echo "Unknown flag: $MODE (expected --dry or --docker)"
    exit 1
fi

# Validate version format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: version must be semver (e.g. 0.5.72), got: $VERSION"
    exit 1
fi

CURRENT_VERSION=$(python3 -c "
import tomllib, pathlib
data = tomllib.loads(pathlib.Path('$PYPROJECT').read_text())
print(data['project']['version'])
")

echo "==> Current version: $CURRENT_VERSION"
echo "==> Target version:  $VERSION"

if [[ "$CURRENT_VERSION" == "$VERSION" && "$MODE" != "--docker" ]]; then
    echo "Error: version $VERSION is already set — bump to a new version"
    exit 1
fi

# ─── Git phase ───────────────────────────────────────────────────────────────

if [[ "$MODE" != "--docker" ]]; then
    echo ""
    echo "==> [1/4] Bumping version to $VERSION"

    # Update pyproject.toml
    sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" "$PYPROJECT"

    # Update __init__.py
    sed -i '' "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" "$INIT_PY"

    # Verify both files match
    TOML_VER=$(python3 -c "
import tomllib, pathlib
data = tomllib.loads(pathlib.Path('$PYPROJECT').read_text())
print(data['project']['version'])
")
    INIT_VER=$(python3 -c "
import re, pathlib
text = pathlib.Path('$INIT_PY').read_text()
print(re.search(r'__version__ = \"(.+?)\"', text).group(1))
")
    if [[ "$TOML_VER" != "$VERSION" || "$INIT_VER" != "$VERSION" ]]; then
        echo "Error: version mismatch after bump — pyproject=$TOML_VER, __init__=$INIT_VER"
        exit 1
    fi
    echo "    Version set to $VERSION in both files"

    echo ""
    echo "==> [2/4] Locking dependencies"
    uv lock --quiet

    echo ""
    echo "==> [3/4] Running tests"
    uv run pytest -q --no-cov --tb=line 2>&1 | tail -3
    echo "    Tests passed"

    if [[ "$MODE" == "--dry" ]]; then
        echo ""
        echo "==> Dry run complete. Changes staged but NOT committed/pushed."
        echo "    To continue: git add -A && git commit && ./scripts/release.sh $VERSION --docker"
        exit 0
    fi

    echo ""
    echo "==> [4/4] Committing and tagging"
    git add "$PYPROJECT" "$INIT_PY" uv.lock
    git commit -m "release: v$VERSION"
    git tag "v$VERSION"
    git push origin main --tags
    echo "    Pushed v$VERSION to origin/main"
fi

# ─── Docker phase ────────────────────────────────────────────────────────────

echo ""
echo "==> [Docker] Building multi-arch images for $VERSION"

# Step 1: Build ARM image on Mac (DOCKER_BUILDKIT=0 for plain manifest)
echo ""
echo "--- Building arm64 (local Mac) ---"
DOCKER_BUILDKIT=0 docker build \
    -t "${IMAGE}:${VERSION}-arm64" \
    .
echo "    arm64 image built"

# Step 2: Sync repo on Ubuntu and build AMD64
echo ""
echo "--- Building amd64 (${REMOTE_HOST}) ---"
ssh "$REMOTE_HOST" "cd $REMOTE_REPO && git pull --quiet && docker build -t ${IMAGE}:${VERSION}-amd64 . 2>&1 | tail -3"
echo "    amd64 image built"

# Step 3: Push platform-specific images
echo ""
echo "--- Pushing platform images ---"
docker push "${IMAGE}:${VERSION}-arm64" 2>&1 | tail -2
ssh "$REMOTE_HOST" "docker push ${IMAGE}:${VERSION}-amd64 2>&1 | tail -2"
echo "    Both platform images pushed"

# Step 4: Create and push multi-arch manifests
echo ""
echo "--- Creating multi-arch manifests ---"

# Remove stale local manifests (ignore errors if they don't exist)
docker manifest rm "${IMAGE}:${VERSION}" 2>/dev/null || true
docker manifest rm "${IMAGE}:latest" 2>/dev/null || true

docker manifest create "${IMAGE}:${VERSION}" \
    "${IMAGE}:${VERSION}-arm64" \
    "${IMAGE}:${VERSION}-amd64"

docker manifest create "${IMAGE}:latest" \
    "${IMAGE}:${VERSION}-arm64" \
    "${IMAGE}:${VERSION}-amd64"

docker manifest push "${IMAGE}:${VERSION}"
docker manifest push "${IMAGE}:latest"

echo ""
echo "==> Release v$VERSION complete!"
echo "    Docker Hub: ${IMAGE}:${VERSION} (amd64 + arm64)"
echo "    Docker Hub: ${IMAGE}:latest    (amd64 + arm64)"
