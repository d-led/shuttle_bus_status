#!/usr/bin/env bash
set -euo pipefail

# Run dataset-based quality tests.
#
# Requirements:
# - Download datasets (default path: data/test_images/german_plates)
# - Provide a plate-trained YOLO weights file via PLATE_MODEL_PATH
#
# Example:
#   export PLATE_DATASET_DIR="$PWD/data/test_images/german_plates"
#   export PLATE_MODEL_PATH="$PWD/models/best.pt"
#   scripts/test_datasets.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

export PROJECT_ROOT
source "$SCRIPT_DIR/common_env.sh"

if [[ "$VIRTUAL_ENV_ACTIVATED" != "true" ]]; then
  echo "❌ Virtual environment not active. Run: scripts/setup.sh"
  exit 1
fi

REPORT_MAX_IMAGES="${PLATE_REPORT_MAX_IMAGES:-50}"
REPORT_PATH="${PROJECT_ROOT}/reports/datasets/plate_report.html"

echo ""
echo "Dataset tests + report"
echo "  - Report max images: ${REPORT_MAX_IMAGES}"
echo "  - Tip: change it via: PLATE_REPORT_MAX_IMAGES=<n> scripts/test_datasets.sh"

python -m pytest -m integration tests/datasets "$@"

echo ""
echo "Generating HTML report (max images: ${REPORT_MAX_IMAGES})..."
python "${PROJECT_ROOT}/scripts/generate_detection_report.py" \
  --out "${REPORT_PATH}" \
  --max-images "${REPORT_MAX_IMAGES}"

echo ""
echo "✓ Reports written:"
echo "  - Detection report: ${REPORT_PATH}"
echo "  - OCR accuracy report: ${PROJECT_ROOT}/reports/datasets/ocr_accuracy_report.html"
echo ""
echo "Open in browser:"
echo "  file://${REPORT_PATH}"
echo "  file://${PROJECT_ROOT}/reports/datasets/ocr_accuracy_report.html"
echo ""
echo "Want a larger/smaller report? Example:"
echo "  PLATE_REPORT_MAX_IMAGES=200 scripts/test_datasets.sh"

