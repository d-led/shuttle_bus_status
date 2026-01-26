#!/usr/bin/env bash
# Start the camera server
# Works on Raspberry Pi, macOS, and Linux

set -euo pipefail

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Source common environment setup
source "$SCRIPT_DIR/common_env.sh"

# Check if virtual environment was activated
if [[ "$VIRTUAL_ENV_ACTIVATED" != "true" ]]; then
    echo "⚠️  Virtual environment not found. Please run scripts/setup.sh first"
    exit 1
fi

# Start the camera server
echo "Starting camera server..."
echo "Platform: $(uname -s) ($(uname -m))"
echo "Server will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

exec shuttle-bus-status-camera-server
