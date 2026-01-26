#!/usr/bin/env bash
set -euo pipefail

# Upgrade all dependency lockfiles and re-sync the shared repo-root venv.
#
# This repo is a mono-repo:
# - `camera/` and `server/` each have their own `uv.lock`
# - the repo root is a small meta package depending on both local packages
# - the single source of truth for the dev environment is the repo-root `.venv`

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Error: shared virtual environment not found at '$VENV_DIR'." >&2
    echo "Run 'scripts/setup.sh' first to create and populate it." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is required but was not found in PATH." >&2
    echo "Install it (e.g. 'python -m pip install uv') or run 'scripts/setup.sh'." >&2
    exit 1
fi

echo "Upgrading lockfiles..."
for package_dir in camera server; do
    echo " - $package_dir/uv.lock"
    (cd "$PROJECT_ROOT/$package_dir" && uv lock --upgrade)
done

echo ""
echo "Verifying security baseline (protobuf advisory)..."
python - <<'PY'
from __future__ import annotations

import re
from pathlib import Path

lock_path = Path("camera/uv.lock")
text = lock_path.read_text(encoding="utf-8")

match = re.search(r'(?ms)^\[\[package\]\]\s*\nname = "protobuf"\s*\nversion = "([^"]+)"\s*$', text)
if not match:
    print("✓ protobuf not present in camera/uv.lock")
    raise SystemExit(0)

version = match.group(1)
print(f"Found protobuf=={version} in camera/uv.lock")
print("This repo's policy is to keep the lockfile free of protobuf until the upstream DoS")
print("advisory for google.protobuf.json_format.ParseDict() has a patched release.")
print("If you intentionally need Kaggle tooling, install it ad-hoc (see scripts/download_test_dataset.sh).")
raise SystemExit(1)
PY

echo ""
echo "Re-syncing shared venv from repo root..."
source "$VENV_DIR/bin/activate"
uv sync --active --dev --inexact --index-strategy unsafe-best-match

echo ""
echo "Done."
