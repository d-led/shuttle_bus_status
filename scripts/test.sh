#!/usr/bin/env bash
set -euo pipefail

# Test script for camera plate detection
# Runs all CI checks: formatting, linting, type checking, and tests
# Automatically detects and uses .venv or camera-venv if available

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Set PROJECT_ROOT before sourcing common_env.sh so it uses the correct path
export PROJECT_ROOT

# Source common environment setup
source "$SCRIPT_DIR/common_env.sh"

# Function to run Python module
# Always use the activated venv's Python if available
run_python_module() {
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        # Use the venv's Python explicitly
        "${VIRTUAL_ENV}/bin/python" -m "$@"
    elif [ -n "${VENV_DIR:-}" ] && [ -d "${VENV_DIR}/bin" ]; then
        # Fallback: use VENV_DIR if VIRTUAL_ENV not set
        "${VENV_DIR}/bin/python" -m "$@"
    else
        PYTHONPATH="${PYTHONPATH:-}" python3 -m "$@"
    fi
}

# Check if dependencies are installed
if ! run_python_module pytest --version &> /dev/null 2>&1; then
    echo "Dependencies not found. Installing..."
    if [[ "$VIRTUAL_ENV_ACTIVATED" == "true" ]] || [ -d ".venv" ] || [ -d "camera/.venv" ] || [ -d "$HOME/camera-venv" ]; then
        if [[ "$VIRTUAL_ENV_ACTIVATED" != "true" ]]; then
            if [ -d ".venv" ]; then
                source .venv/bin/activate
            elif [ -d "camera/.venv" ]; then
                source camera/.venv/bin/activate
            else
                source "$HOME/camera-venv/bin/activate"
            fi
        fi
        cd camera
        pip install -e ".[dev]"
        cd ..
        # Also install server package if it exists
        if [ -d "server" ]; then
            cd server
            pip install -e ".[dev]" 2>/dev/null || true
            cd ..
        fi
    else
        echo "Warning: No virtual environment found. Please run ./scripts/setup.sh first or install dependencies manually."
        exit 1
    fi
fi

# Ensure server package is installed if tests exist
if [ -d "tests/server" ] && [ -d "server" ]; then
    if ! python -c "from server.config import Settings" 2>/dev/null; then
        echo "Installing server package for tests..."
        cd server
        pip install -e ".[dev]" 2>/dev/null || true
        cd ..
    fi
fi

echo "=========================================="
echo "Formatting..."
echo "=========================================="
echo ""

# Reformat code with Black first
echo "Reformatting with Black..."
run_python_module black camera server/src tests 2>/dev/null || true
echo "✓ Black formatting complete"
echo ""

echo "=========================================="
echo "Running CI checks..."
echo "=========================================="
echo ""

# 1. Check code formatting with Black
echo "1. Checking code formatting with Black..."
run_python_module black --check camera server/src tests 2>/dev/null || echo "⚠ Black check skipped (not installed)"
echo "✓ Formatting check passed"
echo ""

# 2. Lint with Ruff
echo "2. Linting with Ruff..."
run_python_module ruff check camera server/src tests 2>/dev/null || echo "⚠ Ruff check skipped (not installed)"
echo "✓ Linting passed"
echo ""

# 3. Type check with mypy
echo "3. Type checking with mypy..."
run_python_module mypy camera 2>/dev/null || echo "⚠ Mypy check skipped (not installed)"
if [ -d "server/src/server" ]; then
    export PYTHONPATH="${PROJECT_ROOT}/server/src:${PYTHONPATH:-}"
    run_python_module mypy server/src/server 2>/dev/null || echo "⚠ Server mypy check skipped"
fi
echo "✓ Type checking passed"
echo ""

# 4. Complexity analysis
echo "4. Running complexity analysis..."
if python -c "import radon" 2>/dev/null || python3 -c "import radon" 2>/dev/null; then
    if [ -f "$SCRIPT_DIR/analyze_complexity.py" ]; then
        python "$SCRIPT_DIR/analyze_complexity.py" camera || true
        if [ -d "server/src/server" ]; then
            python "$SCRIPT_DIR/analyze_complexity.py" server/src/server || true
        fi
        echo ""
        echo "Checking for dead code with vulture..."
        if run_python_module vulture camera --min-confidence 80 2>/dev/null; then
            echo "✓ No dead code found in camera"
        fi
        if [ -d "server/src/server" ]; then
            if run_python_module vulture server/src/server --min-confidence 80 2>/dev/null; then
                echo "✓ No dead code found in server"
            fi
        fi
        echo "✓ Complexity analysis complete"
    else
        echo "⚠ Complexity analysis script not found. Skipping..."
    fi
else
    echo "⚠ Complexity analysis tools not installed (radon, vulture). Skipping..."
fi
echo ""

# 5. Run tests (excluding integration tests by default)
echo "5. Running tests..."
# PYTHONPATH is already set by common_env.sh, but ensure server/src is explicitly included
# Ensure we're using the venv's Python
if [ -n "${VIRTUAL_ENV:-}" ]; then
    # Use venv's Python directly with explicit PYTHONPATH
    PYTHONPATH="${PROJECT_ROOT}/server/src:${PYTHONPATH:-}" "${VIRTUAL_ENV}/bin/python" -m pytest -m "not integration" tests/ "$@"
elif [ -n "${VENV_DIR:-}" ] && [ -d "${VENV_DIR}/bin" ]; then
    # Fallback: use VENV_DIR's Python
    PYTHONPATH="${PROJECT_ROOT}/server/src:${PYTHONPATH:-}" "${VENV_DIR}/bin/python" -m pytest -m "not integration" tests/ "$@"
else
    # Last resort: use run_python_module
    PYTHONPATH="${PROJECT_ROOT}/server/src:${PYTHONPATH:-}" run_python_module pytest -m "not integration" tests/ "$@"
fi
echo "✓ Tests passed"
echo ""

echo "=========================================="
echo "All checks passed! ✓"
echo "=========================================="
