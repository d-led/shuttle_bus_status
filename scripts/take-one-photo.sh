#!/usr/bin/env bash
set -euo pipefail

# Take a single photo from the USB camera and save it with a human-readable timestamp
# Usage: ./scripts/take-one-photo.sh [output_directory]

OUTPUT_DIR="${1:-/tmp}"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE="${OUTPUT_DIR}/camera_${TIMESTAMP}.jpg"

echo "Capturing photo from /dev/video0..."
echo "Output: ${OUTPUT_FILE}"

# Use ffmpeg to capture a single frame
ffmpeg -f v4l2 \
    -input_format mjpeg \
    -video_size 1920x1080 \
    -i /dev/video0 \
    -frames:v 1 \
    -q:v 2 \
    -y "${OUTPUT_FILE}" \
    > /dev/null 2>&1

if [ -f "${OUTPUT_FILE}" ]; then
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    echo "Photo saved successfully: ${OUTPUT_FILE} (${FILE_SIZE})"
    exit 0
else
    echo "Error: Failed to capture photo"
    exit 1
fi
