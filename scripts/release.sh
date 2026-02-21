#!/usr/bin/env bash
set -euo pipefail

# ─── ASTrograph Release Script ───────────────────────────────────────────────
#
# Automates release preparation:
#   1. Bump version in pyproject.toml + __init__.py
#   2. Lock dependencies
#   3. Run local quality checks
#   4. Commit, tag, push
#   5. GitHub Actions publishes Docker Hub images for that tag
#
# Usage:
#   ./scripts/release.sh 0.5.72       # full release
#   ./scripts/release.sh 0.5.72 --dry # stop before commit/tag/push
#
# Prerequisites:
#   - Working tree is clean
#   - GitHub Actions secrets configured:
#       DOCKERHUB_USERNAME, DOCKERHUB_TOKEN
# ──────────────────────────────────────────────────────────────────────────────

PYPROJECT="pyproject.toml"
INIT_PY="src/astrograph/__init__.py"

# ─── Parse arguments ─────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <version> [--dry]"
    echo ""
    echo "  <version>   Semantic version (e.g. 0.5.72)"
    echo "  --dry       Validate everything but don't commit/tag/push"
    exit 1
fi

VERSION="$1"
MODE="${2:-full}"

if [[ "$MODE" != "--dry" && "$MODE" != "full" ]]; then
    echo "Unknown flag: $MODE (expected --dry)"
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

if [[ "$CURRENT_VERSION" == "$VERSION" ]]; then
    echo "Error: version $VERSION is already set — bump to a new version"
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working tree is not clean. Commit or stash changes first."
    exit 1
fi

echo ""
echo "==> [1/4] Bumping version to $VERSION"
python3 - "$VERSION" <<'PY'
import pathlib
import re
import sys

version = sys.argv[1]

pyproject = pathlib.Path("pyproject.toml")
pyproject_text = pyproject.read_text()
updated_pyproject, pyproject_count = re.subn(
    r'(?m)^version = "[^"]+"$',
    f'version = "{version}"',
    pyproject_text,
    count=1,
)
if pyproject_count != 1:
    raise SystemExit("Failed to update version in pyproject.toml")
pyproject.write_text(updated_pyproject)

init_py = pathlib.Path("src/astrograph/__init__.py")
init_text = init_py.read_text()
updated_init, init_count = re.subn(
    r'(?m)^__version__ = "[^"]+"$',
    f'__version__ = "{version}"',
    init_text,
    count=1,
)
if init_count != 1:
    raise SystemExit("Failed to update version in src/astrograph/__init__.py")
init_py.write_text(updated_init)
PY

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
echo "==> [3/4] Running local quality checks (linting + types only)"
echo "    Full test suite + coverage enforced by CI after tag push."
uv run ruff check .
uv run mypy
echo "    Local checks passed"

if [[ "$MODE" == "--dry" ]]; then
    echo ""
    echo "==> Dry run complete. Version files + lock updated locally."
    echo "    Next: review changes, then run ./scripts/release.sh $VERSION"
    exit 0
fi

echo ""
echo "==> [4/4] Committing and tagging"
git add "$PYPROJECT" "$INIT_PY" uv.lock
git commit -m "release: v$VERSION"
git tag "v$VERSION"
git push origin main
git push origin "v$VERSION"

echo ""
echo "==> Release tag pushed: v$VERSION"
echo "    GitHub Actions will now validate and publish Docker Hub tags:"
echo "      - thaylo/astrograph:$VERSION"
echo "      - thaylo/astrograph:v$VERSION"
echo "      - thaylo/astrograph:${VERSION%.*}"
echo "      - thaylo/astrograph:${VERSION%%.*}"
echo "      - thaylo/astrograph:latest"
