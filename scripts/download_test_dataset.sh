#!/usr/bin/env bash
set -euo pipefail

# Download script for German license plate test/detection datasets
# Downloads from multiple open-source sources
# Usage: ./scripts/download_test_dataset.sh [output_directory]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${1:-${PROJECT_ROOT}/data/test_images}"
DATASET_DIR="${OUTPUT_DIR}/german_plates"

echo "=========================================="
echo "German License Plate Dataset Downloader"
echo "=========================================="
echo "Output directory: ${DATASET_DIR}"
echo ""

# Create output directory
mkdir -p "${DATASET_DIR}"

# Check for required tools
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "⚠️  $1 not found. Some download methods may not work."
        return 1
    fi
    return 0
}

check_command curl || true
check_command wget || true
check_command unzip || true
check_command git || true

# Function to download from URL
download_file() {
    local url="$1"
    local output_file="$2"
    local description="$3"
    
    echo "Downloading ${description}..."
    if command -v curl &> /dev/null; then
        curl -L -o "${output_file}" "${url}" || return 1
    elif command -v wget &> /dev/null; then
        wget -O "${output_file}" "${url}" || return 1
    else
        echo "Error: Neither curl nor wget available"
        return 1
    fi
    echo "✓ Downloaded: ${output_file}"
}

# Function to extract archive
extract_archive() {
    local archive="$1"
    local output_dir="$2"
    
    echo "Extracting ${archive}..."
    if [[ "${archive}" == *.zip ]]; then
        unzip -q -o "${archive}" -d "${output_dir}" || return 1
    elif [[ "${archive}" == *.tar.gz ]] || [[ "${archive}" == *.tgz ]]; then
        tar -xzf "${archive}" -C "${output_dir}" || return 1
    elif [[ "${archive}" == *.tar ]]; then
        tar -xf "${archive}" -C "${output_dir}" || return 1
    else
        echo "⚠️  Unknown archive format: ${archive}"
        return 1
    fi
    echo "✓ Extracted to: ${output_dir}"
}

# Method 1: OpenALPR Train-OCR (if available)
download_openalpr() {
    local openalpr_dir="${DATASET_DIR}/openalpr"
    mkdir -p "${openalpr_dir}"
    
    echo ""
    echo "Method 1: Attempting to download OpenALPR training data..."
    echo "Note: OpenALPR training data may need to be cloned from repository"
    
    if command -v git &> /dev/null; then
        if [ ! -d "${openalpr_dir}/train-ocr" ]; then
            echo "Cloning OpenALPR train-ocr repository..."
            git clone --depth 1 https://github.com/openalpr/train-ocr.git "${openalpr_dir}/train-ocr" 2>&1 | grep -v "Cloning into" || {
                echo "⚠️  OpenALPR repository clone failed or already exists"
                return 1
            }
            echo "✓ OpenALPR data cloned"
        else
            echo "✓ OpenALPR repository already exists"
        fi
    else
        echo "⚠️  git not available, skipping OpenALPR"
    fi
}

# Method 2: Sample images from GitHub repositories
download_github_samples() {
    local github_dir="${DATASET_DIR}/github_samples"
    mkdir -p "${github_dir}"
    
    echo ""
    echo "Method 2: Downloading sample images from GitHub repositories..."
    
    # Try to download from known repositories with sample images
    # Note: These may change, so we'll try a few common patterns
    
    # German License Plate Recognition repository
    if command -v git &> /dev/null; then
        local repo_dir="${github_dir}/german_license_plate_recognition"
        if [ ! -d "${repo_dir}" ]; then
            echo "Cloning German License Plate Recognition repository..."
            git clone --depth 1 https://github.com/aboerzel/German_License_Plate_Recognition.git "${repo_dir}" 2>&1 | grep -v "Cloning into" || {
                echo "⚠️  Repository clone failed (may not exist or be accessible)"
                return 1
            }
            echo "✓ Repository cloned"
        else
            echo "✓ Repository already exists"
        fi
    fi
}

# Method 3: Create a README with links to other datasets
create_dataset_info() {
    local info_file="${DATASET_DIR}/README.md"
    
    cat > "${info_file}" << 'EOF'
# German License Plate Test Dataset

This directory contains test images for German license plate detection and recognition.

## Sources

### Open Source Options:

1. **OpenALPR Train-OCR**
   - Repository: https://github.com/openalpr/train-ocr
   - Contains training data for various countries including EU regions
   - License: AGPL-3.0

2. **German License Plate Recognition**
   - Repository: https://github.com/aboerzel/German_License_Plate_Recognition
   - Contains notebooks and sample data for German plates

### Commercial/Paid Options:

1. **UniData Germany License Plate Dataset**
   - 177,827 images with OCR labeling
   - Format: PNG images with CSV annotations
   - Website: https://unidata.pro/datasets/germany-license-plate-detection-dataset/

### Synthetic Data Generation:

1. **ALPR Dataset Generator**
   - Repository: https://github.com/markusrussold/ALPRDataset
   - Can generate synthetic German license plates
   - Supports EuroPlate font format

## Usage

For testing and development, you can:
1. Use the downloaded sample images
2. Generate synthetic plates using ALPR Dataset Generator
3. Use your own camera to capture test images with `scripts/take-one-photo.sh`

## Notes

- Most open-source datasets are limited in size
- For production training, consider combining multiple sources
- Always verify license compatibility for your use case
EOF

    echo "✓ Created dataset info: ${info_file}"
}

# Method 4: Download sample images from public sources (if URLs are available)
download_public_samples() {
    local samples_dir="${DATASET_DIR}/public_samples"
    mkdir -p "${samples_dir}"
    
    echo ""
    echo "Method 3: Attempting to download public sample images..."
    echo "Note: Direct image URLs may not be available or may change"
    
    # This is a placeholder - actual URLs would need to be verified
    # For now, we'll create a script that users can customize
    local download_script="${samples_dir}/download_samples.sh"
    
    cat > "${download_script}" << 'EOFSCRIPT'
#!/usr/bin/env bash
# Custom script to download sample images
# Add your own URLs here

SAMPLES_DIR="$(dirname "$0")"

# Example: Download from a public image hosting service
# Uncomment and modify URLs as needed:

# curl -L -o "${SAMPLES_DIR}/sample1.jpg" "https://example.com/german-plate-1.jpg"
# curl -L -o "${SAMPLES_DIR}/sample2.jpg" "https://example.com/german-plate-2.jpg"

echo "Add your own image URLs to this script to download samples"
EOFSCRIPT

    chmod +x "${download_script}"
    echo "✓ Created custom download script: ${download_script}"
}

# Main execution
echo "Starting dataset download..."
echo ""

# Try different methods
download_openalpr || echo "⚠️  OpenALPR download failed or skipped"
download_github_samples || echo "⚠️  GitHub samples download failed or skipped"
download_public_samples || echo "⚠️  Public samples download failed or skipped"
create_dataset_info

echo ""
echo "=========================================="
echo "Download Summary"
echo "=========================================="
echo "Dataset directory: ${DATASET_DIR}"
echo ""

# Count downloaded images
IMAGE_COUNT=$(find "${DATASET_DIR}" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.bmp" \) 2>/dev/null | wc -l | tr -d ' ')

if [ "${IMAGE_COUNT}" -gt 0 ]; then
    echo "✓ Found ${IMAGE_COUNT} image(s)"
else
    echo "⚠️  No images found. You may need to:"
    echo "   1. Check the cloned repositories for image files"
    echo "   2. Add custom download URLs to ${DATASET_DIR}/public_samples/download_samples.sh"
    echo "   3. Use scripts/take-one-photo.sh to capture your own test images"
fi

echo ""
echo "Next steps:"
echo "  1. Review ${DATASET_DIR}/README.md for dataset information"
echo "  2. Check cloned repositories for available images"
echo "  3. Consider using synthetic data generation for more samples"
echo ""
