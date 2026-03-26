#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT/resources/llama.cpp/manifest.toml"
DEST_DIR="${1:-$ROOT/resources/llama.cpp}"
TMP_DIR="${TMPDIR:-/tmp}/tritpack-llama-vendor"

python3 - "$MANIFEST" "$DEST_DIR" "$TMP_DIR" <<'PY'
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
from pathlib import Path

manifest_path = Path(sys.argv[1])
dest_dir = Path(sys.argv[2])
tmp_dir = Path(sys.argv[3])
manifest = tomllib.loads(manifest_path.read_text())

archive_path = tmp_dir / manifest["archive_name"]
extract_dir = tmp_dir / "extract"
tmp_dir.mkdir(parents=True, exist_ok=True)
extract_dir.mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(manifest["archive_url"]) as response, archive_path.open("wb") as handle:
    shutil.copyfileobj(response, handle)

digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
if digest != manifest["sha256"]:
    raise SystemExit(f"sha mismatch for {archive_path.name}: {digest} != {manifest['sha256']}")

with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(extract_dir)

payload_root = next((path for path in extract_dir.iterdir() if path.is_dir()), None)
if payload_root is None:
    raise SystemExit("could not locate extracted llama.cpp payload")

dest_dir.mkdir(parents=True, exist_ok=True)
for existing in dest_dir.iterdir():
    if existing.name == "manifest.toml":
        continue
    if existing.is_dir() and not existing.is_symlink():
        shutil.rmtree(existing)
    else:
        existing.unlink()

for item in payload_root.iterdir():
    target = dest_dir / item.name
    if item.is_symlink():
        target.symlink_to(os.readlink(item))
    elif item.is_dir():
        shutil.copytree(item, target, symlinks=True)
    else:
        shutil.copy2(item, target, follow_symlinks=False)

for binary_name in manifest["binary_names"]:
    target = dest_dir / binary_name
    if not target.exists():
        raise SystemExit(f"missing expected binary {binary_name} in vendored archive")
    target.chmod(0o755)
    result = subprocess.run([str(target), "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{binary_name} failed --version: {result.stderr.strip()}")
    print(f"vendored {binary_name}: {(result.stdout or result.stderr).strip()}")
PY
