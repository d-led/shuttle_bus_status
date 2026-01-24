#!/usr/bin/env bash
set -euo pipefail

# Setup script for camera plate detection dependencies
# Run this on the Raspberry Pi

echo "Installing system dependencies..."
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

echo "Creating virtual environment..."
cd ~
if [ ! -d "camera-venv" ]; then
    python3 -m venv camera-venv
fi

source ~/camera-venv/bin/activate

echo "Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing Python dependencies (this may take several minutes)..."
cd "$(dirname "$0")/../camera" || exit 1
python -m pip install -r requirements.txt

echo "Verifying installation..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import ultralytics; print('Ultralytics installed')"
python -c "import easyocr; print('EasyOCR installed')"

echo ""
echo "Setup complete!"
echo "To activate the virtual environment, run:"
echo "  source ~/camera-venv/bin/activate"
