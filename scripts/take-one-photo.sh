#!/usr/bin/env bash
set -euo pipefail

# Take a single photo from the USB camera and save it with a human-readable timestamp
# Usage: ./scripts/take-one-photo.sh [output_directory]

OUTPUT_DIR="${1:-/tmp}"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE="${OUTPUT_DIR}/camera_${TIMESTAMP}.jpg"

# Resolve device from config.toml (no inline Python).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEVICE="$(PYTHONPATH="${PROJECT_ROOT}" python "${SCRIPT_DIR}/detect_camera_device.py")"

echo "Capturing photo from ${DEVICE}..."
echo "Output: ${OUTPUT_FILE}"

if [[ "${DEVICE}" == avfoundation:* ]]; then
    # macOS (AVFoundation). The part after ":" is a numeric device index.
    AV_INDEX="${DEVICE#avfoundation:}"
    ffmpeg -f avfoundation \
        -framerate 30 \
        -video_size 1920x1080 \
        -i "${AV_INDEX}" \
        -frames:v 1 \
        -q:v 2 \
        -y "${OUTPUT_FILE}" \
        > /dev/null 2>&1
else
    # Linux/RPi (V4L2).
    ffmpeg -f v4l2 \
        -input_format mjpeg \
        -video_size 1920x1080 \
        -i "${DEVICE}" \
        -frames:v 1 \
        -q:v 2 \
        -y "${OUTPUT_FILE}" \
        > /dev/null 2>&1
fi

if [ -f "${OUTPUT_FILE}" ]; then
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    echo "Photo saved successfully: ${OUTPUT_FILE} (${FILE_SIZE})"
    exit 0
else
    echo "Error: Failed to capture photo"
    exit 1
fi
