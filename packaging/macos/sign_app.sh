#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="${1:-$ROOT/target/macos-bundle/TritPack.app}"
IDENTITY="${APPLE_SIGNING_IDENTITY:?Set APPLE_SIGNING_IDENTITY}"

codesign --force --options runtime --deep --timestamp --sign "$IDENTITY" "$APP_DIR"
codesign --verify --deep --strict "$APP_DIR"
echo "Signed $APP_DIR"
