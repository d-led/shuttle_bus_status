#!/usr/bin/env bash
set -euo pipefail

# Run all tests for the camera plate detection project
cd "$(dirname "$0")/.." || exit 1

python3 -m pytest tests/ "$@"
