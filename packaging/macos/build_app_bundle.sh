#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_NAME="TritPack.app"
BUILD_DIR="$ROOT/target/macos-bundle"
APP_DIR="$BUILD_DIR/$APP_NAME"
RESOURCES_DIR="$APP_DIR/Contents/Resources"
MACOS_DIR="$APP_DIR/Contents/MacOS"

"$ROOT/packaging/macos/vendor_llamacpp.sh"
"$ROOT/packaging/macos/bootstrap_python_runtime.sh"

cargo build --release -p tritpack-desktop --manifest-path "$ROOT/Cargo.toml"

rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

cp "$ROOT/target/release/tritpack-desktop" "$MACOS_DIR/tritpack-desktop"
cp "$ROOT/packaging/macos/Info.plist" "$APP_DIR/Contents/Info.plist"

mkdir -p "$RESOURCES_DIR/llama.cpp" "$RESOURCES_DIR/python" "$RESOURCES_DIR/integration"
cp -R "$ROOT/resources/llama.cpp/." "$RESOURCES_DIR/llama.cpp/"
cp -R "$ROOT/resources/python/." "$RESOURCES_DIR/python/"
cp -R "$ROOT/python/tritpack" "$RESOURCES_DIR/python/"
cp -R "$ROOT/integration/." "$RESOURCES_DIR/integration/"

echo "Built app bundle at $APP_DIR"
