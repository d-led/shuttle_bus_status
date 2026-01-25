#!/usr/bin/env bash
set -euo pipefail

# Test script for camera plate detection
# Runs all CI checks: formatting, linting, type checking, and tests
# Automatically detects and uses .venv or camera-venv if available

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Detect and activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$HOME/camera-venv" ]; then
    source "$HOME/camera-venv/bin/activate"
fi

# Function to run Python module
run_python_module() {
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        python -m "$@"
    else
        python3 -m "$@"
    fi
}

# Check if dependencies are installed
if ! run_python_module pytest --version &> /dev/null 2>&1; then
    echo "Dependencies not found. Installing..."
    if [ -d ".venv" ] || [ -d "$HOME/camera-venv" ]; then
        if [ -d ".venv" ]; then
            source .venv/bin/activate
        else
            source "$HOME/camera-venv/bin/activate"
        fi
        cd camera
        pip install -e ".[dev]"
        cd ..
    else
        echo "Warning: No virtual environment found. Please run ./scripts/setup.sh first or install dependencies manually."
        exit 1
    fi
fi

echo "=========================================="
echo "Formatting..."
echo "=========================================="
echo ""

# Reformat code with Black first
echo "Reformatting with Black..."
run_python_module black camera tests
echo "✓ Black formatting complete"
echo ""

echo "=========================================="
echo "Running CI checks..."
echo "=========================================="
echo ""

# 1. Check code formatting with Black
echo "1. Checking code formatting with Black..."
run_python_module black --check camera tests
echo "✓ Formatting check passed"
echo ""

# 2. Lint with Ruff
echo "2. Linting with Ruff..."
run_python_module ruff check camera tests
echo "✓ Linting passed"
echo ""

# 3. Type check with mypy
echo "3. Type checking with mypy..."
run_python_module mypy camera
echo "✓ Type checking passed"
echo ""

# 4. Complexity analysis
echo "4. Running complexity analysis..."
if python -c "import radon" 2>/dev/null || python3 -c "import radon" 2>/dev/null; then
    if [ -f "$SCRIPT_DIR/analyze_complexity.py" ]; then
        python "$SCRIPT_DIR/analyze_complexity.py" camera || true
        echo ""
        echo "Checking for dead code with vulture..."
        if run_python_module vulture camera --min-confidence 80 2>/dev/null; then
            echo "✓ No dead code found"
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
run_python_module pytest -m "not integration" "$@"
echo "✓ Tests passed"
echo ""

echo "=========================================="
echo "All checks passed! ✓"
echo "=========================================="
