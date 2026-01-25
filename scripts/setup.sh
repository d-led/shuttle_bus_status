#!/usr/bin/env bash
set -euo pipefail

# Unified setup script for camera plate detection
# Works on Raspberry Pi, macOS, Ubuntu (CI), and other Linux systems

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

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

# Detect if we're in CI (GitHub Actions, etc.)
IS_CI="${CI:-false}"
if [[ "$IS_CI" == "true" ]] || [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    IS_CI=true
    VENV_DIR="$PROJECT_ROOT/.venv"
else
    IS_CI=false
    if [[ "$OS_TYPE" == "raspberrypi" ]]; then
        VENV_DIR="$HOME/camera-venv"
    else
        VENV_DIR="$PROJECT_ROOT/.venv"
    fi
fi

echo "Detected environment: $OS_TYPE"
if [[ "$IS_CI" == "true" ]]; then
    echo "Running in CI mode"
fi

# Install system dependencies
if [[ "$OS_TYPE" == "raspberrypi" ]] || [[ "$OS_TYPE" == "linux" ]]; then
    echo ""
    echo "Installing system dependencies..."
    # In CI, we have sudo available, so install dependencies
    # For local Raspberry Pi, also install
    if command -v sudo &> /dev/null; then
        # Test if sudo works (in CI it will, on local RPi user may need password)
        if sudo -n true 2>/dev/null || [[ "$IS_CI" == "true" ]]; then
            sudo apt-get update
            sudo apt-get install -y \
                python3-dev \
                python3-pip \
                python3-venv \
                build-essential \
                cmake \
                pkg-config \
                libjpeg-dev \
                libpng-dev \
                libtiff-dev \
                libv4l-dev \
                v4l-utils
            echo "✓ System dependencies installed"
        else
            echo "⚠️  sudo requires password. Please install system dependencies manually:"
            echo "   sudo apt-get install -y python3-dev python3-pip python3-venv build-essential cmake pkg-config libjpeg-dev libpng-dev libtiff-dev libv4l-dev v4l-utils"
        fi
    else
        echo "⚠️  sudo not available. Please install system dependencies manually:"
        echo "   sudo apt-get install -y python3-dev python3-pip python3-venv build-essential cmake pkg-config libjpeg-dev libpng-dev libtiff-dev libv4l-utils"
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

# Create virtual environment
echo ""
echo "Creating virtual environment at: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."

# Check if uv is available, use it if so (faster, especially in CI)
if command -v uv &> /dev/null; then
    echo "Using uv for package management..."
    # Install camera package
    cd camera
    if [[ "$IS_CI" == "true" ]] || [[ -n "${CI:-}" ]]; then
        # In CI, use unsafe-best-match to check all indexes (piwheels may have older versions)
        uv sync --dev --index-strategy unsafe-best-match
    else
        uv sync --dev || uv pip install -e ".[dev]"
    fi
    echo "✓ Camera dependencies installed with uv"
    
    # Install server package
    cd "$PROJECT_ROOT/server"
    if [[ "$IS_CI" == "true" ]] || [[ -n "${CI:-}" ]]; then
        # In CI, use unsafe-best-match to check all indexes
        uv sync --dev --index-strategy unsafe-best-match
    else
        uv sync --dev || uv pip install -e ".[dev]"
    fi
    echo "✓ Server dependencies installed with uv"
    cd "$PROJECT_ROOT"
else
    echo "Using pip for package management..."
    python -m pip install --upgrade pip setuptools wheel
    
    # Install camera package
    cd camera
    python -m pip install -e ".[dev]"
    echo "✓ Camera dependencies installed with pip"
    
    # Install server package
    cd "$PROJECT_ROOT/server"
    python -m pip install -e ".[dev]"
    echo "✓ Server dependencies installed with pip"
    cd "$PROJECT_ROOT"
fi

# Verify installation
echo ""
echo "Verifying installation..."
cd "$PROJECT_ROOT"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

python -c "import numpy; print(f'✓ NumPy version: {numpy.__version__}')" || echo "⚠ NumPy not available"
python -c "import structlog; print('✓ structlog installed')" || echo "⚠ structlog not installed"

# OpenCV may not be available in CI without camera hardware
if python -c "import cv2; print(f'✓ OpenCV version: {cv2.__version__}')" 2>/dev/null; then
    :
else
    if [[ "$IS_CI" == "true" ]]; then
        echo "⚠ OpenCV not available (expected in CI without camera)"
    else
        echo "⚠ OpenCV not available - camera functionality will be limited"
    fi
fi

# Only check heavy ML dependencies if not in CI (they take time to install)
if [[ "$IS_CI" != "true" ]]; then
    python -c "import ultralytics; print('✓ Ultralytics installed')" 2>/dev/null || echo "⚠ Ultralytics not installed (may take time to download)"
    python -c "import easyocr; print('✓ EasyOCR installed')" 2>/dev/null || echo "⚠ EasyOCR not installed (may take time to download)"
fi

echo ""
echo "Setup complete!"
if [[ "$OS_TYPE" == "raspberrypi" ]]; then
    echo "To activate the virtual environment, run:"
    echo "  source ~/camera-venv/bin/activate"
else
    echo "To activate the virtual environment, run:"
    echo "  source $VENV_DIR/bin/activate"
fi
