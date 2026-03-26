#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="${1:-$ROOT/target/macos-bundle/TritPack.app}"
ZIP_PATH="${APP_DIR%.app}.zip"
PROFILE="${APPLE_NOTARY_PROFILE:?Set APPLE_NOTARY_PROFILE}"

ditto -c -k --keepParent "$APP_DIR" "$ZIP_PATH"
xcrun notarytool submit "$ZIP_PATH" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$APP_DIR"
echo "Notarized and stapled $APP_DIR"
