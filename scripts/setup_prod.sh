#!/usr/bin/env bash
set -euo pipefail

# Production setup script for camera plate detection
# Installs ONLY runtime dependencies (no dev tools: pytest, black, ruff, mypy, etc.)
# Optimized for Raspberry Pi deployment
# Works on Raspberry Pi, macOS, Ubuntu, and other Linux systems

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Source common setup functions
. "${SCRIPT_DIR}/setup-common.sh"

# Setup virtual environment
setup_venv "$PROJECT_ROOT"

# Detect environment
OS_TYPE="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" == "raspbian" ]] || [[ "$ID" == "debian" ]] && [[ -f /proc/device-tree/model ]] && grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
            OS_TYPE="raspberrypi"
        else
            OS_TYPE="linux"
        fi
    else
        OS_TYPE="linux"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
fi

# Single source of truth: always use repo-root `.venv` everywhere.
VENV_DIR="$PROJECT_ROOT/.venv"

echo "Detected environment: $OS_TYPE"
echo "Production mode: Installing runtime dependencies only (no dev tools)"

# Hard requirement: the camera ML dependencies (Ultralytics -> PyTorch) need 64-bit Linux wheels.
# Raspberry Pi 3 can run either 32-bit (armv7l) or 64-bit (aarch64) OS. We require a 64-bit OS.
if [[ "$OS_TYPE" == "raspberrypi" ]]; then
    MACHINE="$(uname -m)"
    if [[ "$MACHINE" != "aarch64" ]]; then
        echo ""
        echo "Error: Raspberry Pi detected, but OS architecture is '$MACHINE'."
        echo "This project requires a 64-bit Raspberry Pi OS (aarch64) to install ML dependencies (PyTorch/Ultralytics) on Python 3.13."
        echo ""
        echo "Fix:"
        echo "  - Install a 64-bit Raspberry Pi OS image, or"
        echo "  - Switch the project to a Python/ML stack that supports 32-bit armv7l."
        exit 1
    fi
fi

# Install system dependencies
if [[ "$OS_TYPE" == "raspberrypi" ]] || [[ "$OS_TYPE" == "linux" ]]; then
    echo ""
    echo "Installing system dependencies..."
    required_apt_packages=(
        python3-dev
        python3-pip
        python3-venv
        build-essential
        cmake
        pkg-config
        libjpeg-dev
        libpng-dev
        libtiff-dev
        libv4l-dev
        v4l-utils
    )

    missing_apt_packages=()
    for pkg in "${required_apt_packages[@]}"; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            missing_apt_packages+=("$pkg")
        fi
    done

    if [ "${#missing_apt_packages[@]}" -eq 0 ]; then
        echo "✓ System dependencies already installed"
    else
        echo "Missing system dependencies: ${missing_apt_packages[*]}"
        if command -v sudo &> /dev/null; then
            if sudo -n true 2>/dev/null || [[ "${CI:-false}" == "true" ]]; then
                sudo apt-get update
                sudo apt-get install -y "${missing_apt_packages[@]}"
                echo "✓ System dependencies installed"
            else
                echo "⚠️  sudo requires password. Please install system dependencies manually:"
                echo "   sudo apt-get install -y ${missing_apt_packages[*]}"
            fi
        else
            echo "⚠️  sudo not available. Please install system dependencies manually:"
            echo "   sudo apt-get install -y ${missing_apt_packages[*]}"
        fi
    fi
elif [[ "$OS_TYPE" == "macos" ]]; then
    echo ""
    echo "Installing system dependencies (macOS)..."
    if command -v brew &> /dev/null; then
        brew install pkg-config jpeg libpng libtiff || true
        echo "✓ macOS dependencies installed"
    else
        echo "⚠️  Homebrew not found. Install it from https://brew.sh or install dependencies manually"
    fi
fi

# Activate virtual environment (created by setup_venv above)
source "$VENV_DIR/bin/activate"

# Install Python dependencies (PRODUCTION ONLY - no dev tools)
echo ""
echo "Installing Python dependencies (production mode, no dev tools)..."

# Check if uv is available, use it if so (faster, especially on RPi)
if command -v uv &> /dev/null; then
    echo "Using uv for package management..."
    # Install WITHOUT --dev flag - only runtime dependencies
    # Single source of truth: repo-root pyproject.toml.
    # It depends on both local packages (camera + server).
    #
    # - NO --dev flag: excludes pytest/mypy/ruff/black/radon/vulture
    # - Use --inexact to avoid pruning in the shared repo-root venv.
    uv sync --active --inexact --index-strategy unsafe-best-match
    echo "✓ Production dependencies installed"
else
    echo "Using pip for package management..."
    python -m pip install --upgrade pip setuptools wheel
    
    # Without uv, pip can't resolve `tool.uv.sources`. Install local packages explicitly first.
    # Install WITHOUT [dev] extras - only runtime dependencies
    python -m pip install -e "camera"
    python -m pip install -e "server"
    # Keep the meta package installed too (small, but convenient for `pip list` / tooling).
    python -m pip install -e "."
    echo "✓ Production dependencies installed with pip"
fi

echo ""
echo "Production setup complete!"
echo ""
echo "Installed packages:"
echo "  - Runtime dependencies only (no dev tools)"
echo "  - Camera plate detection"
echo "  - Server with live view"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To start the camera server:"
echo "  scripts/start_camera_server.sh"
