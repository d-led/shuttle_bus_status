#!/usr/bin/env bash
set -euo pipefail

# Setup script for server component
# Works on macOS, Ubuntu, Raspberry Pi, and CI environments
# Installs system and Python dependencies for the server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Source common environment setup
# Don't suppress errors - we want to see what's happening
source "$SCRIPT_DIR/common_env.sh"

echo "=========================================="
echo "Server Setup"
echo "=========================================="
echo "Detected environment: $OS_TYPE"
echo ""

# Install system dependencies
if [[ "$OS_TYPE" == "raspberrypi" ]] || [[ "$OS_TYPE" == "linux" ]] || [[ "$OS_TYPE" == "ci" ]]; then
    echo ""
    if [[ "$OS_TYPE" == "ci" ]]; then
        echo "Installing system dependencies (CI environment)..."
    else
        echo "Installing system dependencies (Linux/Raspberry Pi)..."
    fi
    # In CI, Python is usually pre-installed by actions/setup-python, but we verify
    # and install any missing dependencies. For server, we mainly need Python and pip.
    if command -v sudo &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv || true
    else
        # In CI, we might not need sudo (running as root or with permissions)
        apt-get update || true
        apt-get install -y python3 python3-pip python3-venv || true
    fi
    echo "✓ System dependencies installed/verified"
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
fi

# Create virtual environment
echo ""
echo "Setting up virtual environment..."

# Use VENV_DIR from common_env.sh if available, otherwise determine it
if [ -z "${VENV_DIR:-}" ]; then
    if [[ "$OS_TYPE" == "raspberrypi" ]]; then
        VENV_DIR="$HOME/server-venv"
    else
        VENV_DIR="$PROJECT_ROOT/.venv"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at $VENV_DIR"
else
    echo "✓ Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV_ACTIVATED=true

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
    # Always use unsafe-best-match to check all indexes
    # This ensures piwheels (ARM-only) doesn't cause issues on macOS/x86_64
    # On Raspberry Pi, piwheels will still be used as fallback when needed
    uv sync --dev --index-strategy unsafe-best-match || uv pip install -e ".[dev]" || python -m pip install -e ".[dev]"
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
