#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${1:-$ROOT/resources/python/cpython/bin/python3}"
REQUIREMENTS="$ROOT/resources/python/requirements.lock"
DEST_DIR="${2:-$ROOT/resources/python/wheelhouse}"

if [[ ! -x "$PYTHON_BIN" ]] || ! "$PYTHON_BIN" --version >/dev/null 2>&1; then
  "$ROOT/packaging/macos/bootstrap_python_runtime.sh"
fi

rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

"$PYTHON_BIN" -m pip download \
  --only-binary=:all: \
  --dest "$DEST_DIR" \
  -r "$REQUIREMENTS"

echo "Built wheelhouse at $DEST_DIR"
