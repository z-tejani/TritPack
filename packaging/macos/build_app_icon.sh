#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_IMAGE="$ROOT/logo.jpeg"
ICON_DIR="$ROOT/target/macos-icon.iconset"
OUTPUT_DIR="$ROOT/resources/app-icon"
OUTPUT_ICON="$OUTPUT_DIR/TritPack.icns"

if [[ ! -f "$SOURCE_IMAGE" ]]; then
  SOURCE_IMAGE="$ROOT/desktop/ui-slint/ui/assets/logo.jpeg"
fi

if [[ ! -f "$SOURCE_IMAGE" ]]; then
  echo "missing logo source image" >&2
  exit 1
fi

rm -rf "$ICON_DIR"
mkdir -p "$ICON_DIR" "$OUTPUT_DIR"

for size in 16 32 128 256 512; do
  sips -s format png -z "$size" "$size" "$SOURCE_IMAGE" --out "$ICON_DIR/icon_${size}x${size}.png" >/dev/null
  retina_size=$((size * 2))
  sips -s format png -z "$retina_size" "$retina_size" "$SOURCE_IMAGE" --out "$ICON_DIR/icon_${size}x${size}@2x.png" >/dev/null
done

iconutil -c icns "$ICON_DIR" -o "$OUTPUT_ICON"
echo "built app icon at $OUTPUT_ICON"
