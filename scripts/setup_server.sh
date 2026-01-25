#!/usr/bin/env bash
set -euo pipefail

# Setup script for server component
# Works on macOS, Ubuntu, Raspberry Pi, and CI environments
# Installs system and Python dependencies for the server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Detect environment
IS_CI="${CI:-false}"
OS_TYPE="unknown"

if [[ "$IS_CI" == "true" ]] || [[ -n "${CI:-}" ]]; then
    OS_TYPE="ci"
elif [[ -f /etc/os-release ]]; then
    if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null || grep -qi "raspberry" /etc/os-release; then
        OS_TYPE="raspberrypi"
    else
        OS_TYPE="linux"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
fi

echo "=========================================="
echo "Server Setup"
echo "=========================================="
echo "Detected environment: $OS_TYPE"
echo ""

# Install system dependencies
if [[ "$OS_TYPE" == "raspberrypi" ]] || [[ "$OS_TYPE" == "linux" ]]; then
    echo ""
    echo "Installing system dependencies (Linux/Raspberry Pi)..."
    if command -v sudo &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv
    else
        apt-get update
        apt-get install -y python3 python3-pip python3-venv
    fi
    echo "✓ System dependencies installed"
elif [[ "$OS_TYPE" == "macos" ]]; then
    echo ""
    echo "Installing system dependencies (macOS)..."
    if ! command -v python3 &> /dev/null; then
        if command -v brew &> /dev/null; then
            brew install python3
        else
            echo "Error: python3 not found and Homebrew not available."
            echo "Please install Python 3.13+ manually or install Homebrew first."
            exit 1
        fi
    fi
    echo "✓ System dependencies installed"
elif [[ "$OS_TYPE" == "ci" ]]; then
    echo ""
    echo "Skipping system dependencies (CI environment)..."
    echo "Assuming Python and pip are pre-installed"
fi

# Create virtual environment
echo ""
echo "Setting up virtual environment..."

if [[ "$OS_TYPE" == "raspberrypi" ]]; then
    VENV_DIR="$HOME/server-venv"
else
    VENV_DIR="$PROJECT_ROOT/.venv"
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at $VENV_DIR"
else
    echo "✓ Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."

# Check if uv is available, use it if so (faster, especially in CI)
if command -v uv &> /dev/null; then
    echo "Using uv for package management..."
    cd server
    if [[ "$IS_CI" == "true" ]] || [[ -n "${CI:-}" ]]; then
        uv sync --dev
    else
        uv sync --dev || uv pip install -e ".[dev]"
    fi
    echo "✓ Dependencies installed with uv"
    cd "$PROJECT_ROOT"
else
    echo "Using pip for package management..."
    cd server
    python -m pip install -e ".[dev]"
    echo "✓ Dependencies installed with pip"
    cd "$PROJECT_ROOT"
fi

# Verify installation
echo ""
echo "Verifying installation..."

if python -c "import pyview" 2>/dev/null; then
    echo "✓ pyview-web installed"
else
    echo "✗ pyview-web not found"
    exit 1
fi

if python -c "import uvicorn" 2>/dev/null; then
    echo "✓ uvicorn installed"
else
    echo "✗ uvicorn not found"
    exit 1
fi

if python -c "from server.main import create_app" 2>/dev/null; then
    echo "✓ server package importable"
else
    echo "✗ server package not importable"
    exit 1
fi

echo ""
echo "=========================================="
echo "Server setup complete! ✓"
echo "=========================================="
echo ""
echo "To activate the virtual environment:"
if [[ "$OS_TYPE" == "raspberrypi" ]]; then
    echo "  source $HOME/server-venv/bin/activate"
else
    echo "  source .venv/bin/activate"
fi
echo ""
echo "To run the server:"
echo "  python -m server.main"
echo "  # or"
echo "  shuttle-bus-status-server"
echo ""
