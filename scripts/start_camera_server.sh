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
    echo "⚠️  Virtual environment not found. Please run scripts/setup_server.sh first"
    exit 1
fi

# Check if server package is installed
if ! python -c "from server.main import main_camera" 2>/dev/null; then
    echo "⚠️  Server package not installed. Installing..."
    cd server
    if command -v uv &> /dev/null; then
        uv sync --dev --index-strategy unsafe-best-match || uv pip install -e ".[dev]" || python -m pip install -e ".[dev]"
    else
        python -m pip install -e ".[dev]"
    fi
    cd "$PROJECT_ROOT"
fi

# Start the camera server
echo "Starting camera server..."
echo "Platform: $(uname -s) ($(uname -m))"
echo "Server will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Set PYTHONPATH to include server/src
export PYTHONPATH="$PROJECT_ROOT/server/src:$PYTHONPATH"

python -c "from server.main import main_camera; main_camera()"
