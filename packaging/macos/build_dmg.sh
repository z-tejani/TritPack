#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="${1:-$ROOT/target/macos-bundle/TritPack.app}"
DMG_PATH="${2:-$ROOT/target/macos-bundle/TritPack.dmg}"

rm -f "$DMG_PATH"
hdiutil create -volname "TritPack" -srcfolder "$APP_DIR" -ov -format UDZO "$DMG_PATH"
echo "Created $DMG_PATH"
