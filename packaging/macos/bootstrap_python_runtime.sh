#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT/resources/python/runtime-manifest.toml"
DEST_DIR="${1:-$ROOT/resources/python/cpython}"
TMP_DIR="${TMPDIR:-/tmp}/tritpack-python-vendor"

python3 - "$MANIFEST" "$DEST_DIR" "$TMP_DIR" <<'PY'
import hashlib
import shutil
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
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir(parents=True, exist_ok=True)
extract_dir.mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(manifest["archive_url"]) as response, archive_path.open("wb") as handle:
    shutil.copyfileobj(response, handle)

digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
if digest != manifest["sha256"]:
    raise SystemExit(f"sha mismatch for {archive_path.name}: {digest} != {manifest['sha256']}")

with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(extract_dir)

install_root = extract_dir / "python"
if not install_root.exists():
    install_root = next((path for path in extract_dir.iterdir() if path.is_dir()), None)
if install_root is None:
    raise SystemExit("could not locate extracted CPython runtime")

if dest_dir.exists():
    shutil.rmtree(dest_dir)
shutil.copytree(install_root, dest_dir)
python_bin = dest_dir / "bin/python3"
if not python_bin.exists():
    raise SystemExit(f"expected interpreter at {python_bin}")
if not any((dest_dir / "lib").glob("libpython*.dylib")):
    raise SystemExit(f"expected a libpython dylib under {dest_dir / 'lib'}")
python_bin.chmod(0o755)
print(f"vendored python runtime: {python_bin}")
PY
