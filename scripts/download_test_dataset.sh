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

# Method 1: Download sample images from known sources
download_sample_images() {
    local samples_dir="${DATASET_DIR}/samples"
    mkdir -p "${samples_dir}"
    
    echo ""
    echo "Method 1: Downloading sample images from public sources..."
    
    # Try to download individual sample images
    # These are example URLs - actual availability may vary
    # Add known sample image URLs here
    # Example format:
    # local image_urls=("https://raw.githubusercontent.com/user/repo/main/images/sample1.jpg")
    
    local image_urls=()
    
    # Check if array has any elements before iterating
    if [ ${#image_urls[@]} -eq 0 ]; then
        echo "⚠️  No image URLs configured. Add URLs to download_sample_images() function"
        return 0
    fi
    
    local downloaded=0
    for url in "${image_urls[@]}"; do
        if [ -n "$url" ]; then
            local filename=$(basename "$url")
            local output_file="${samples_dir}/${filename}"
            
            if [ ! -f "${output_file}" ]; then
                echo "  Downloading ${filename}..."
                if download_file "$url" "$output_file" "$filename"; then
                    ((downloaded++)) || true
                fi
            else
                echo "  ✓ ${filename} already exists"
            fi
        fi
    done
    
    if [ $downloaded -gt 0 ]; then
        echo "✓ Downloaded ${downloaded} sample image(s)"
    else
        echo "⚠️  No sample images downloaded (URLs may need to be added)"
    fi
}

# Method 2: Extract images from GitHub repositories (download specific image files)
download_github_images() {
    local github_dir="${DATASET_DIR}/github_images"
    mkdir -p "${github_dir}"
    
    echo ""
    echo "Method 2: Downloading images from GitHub repositories..."
    echo "Note: This downloads specific image files, not entire repositories"
    
    # Use GitHub API or raw.githubusercontent.com to download specific image files
    # Example: Download images from a repository's images/ or data/ directory
    
    # German License Plate Recognition - try to find and download sample images
    # Note: Actual paths need to be verified for each repository
    local repo_base="https://raw.githubusercontent.com"
    # Format: "user/repo/branch/path/to/image.jpg"
    # Add specific image file paths here when known
    local repos=()
    
    # Check if array has any elements before iterating
    if [ ${#repos[@]} -eq 0 ]; then
        echo "⚠️  No GitHub image paths configured. Add paths to download_github_images() function"
        return 0
    fi
    
    local downloaded=0
    for repo_path in "${repos[@]}"; do
        if [ -n "$repo_path" ]; then
            local url="${repo_base}/${repo_path}"
            local filename=$(basename "$repo_path")
            local output_file="${github_dir}/${filename}"
            
            if [ ! -f "${output_file}" ]; then
                echo "  Downloading ${filename}..."
                if download_file "$url" "$output_file" "$filename"; then
                    ((downloaded++)) || true
                fi
            else
                echo "  ✓ ${filename} already exists"
            fi
        fi
    done
    
    if [ $downloaded -gt 0 ]; then
        echo "✓ Downloaded ${downloaded} image(s) from GitHub"
    else
        echo "⚠️  No GitHub images downloaded (specific image paths may need to be added)"
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

1. **Kaggle: European License Plates Dataset** (Recommended)
   - Dataset: https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset
   - Includes German and other European license plates
   - Requires Kaggle API: `pip install kaggle`
   - Download: `kaggle datasets download -d abdelhamidzakaria/european-license-plates-dataset`
   - This downloads actual image files, not repositories

2. **GitHub Sample Images**
   - Downloads specific image files from GitHub repositories using raw.githubusercontent.com
   - Add image URLs to the script to download specific samples
   - No need to clone entire repositories

3. **OpenALPR Train-OCR**
   - Repository: https://github.com/openalpr/train-ocr
   - Contains training data for various countries including EU regions
   - License: AGPL-3.0
   - Note: You may need to manually download specific image files from this repo

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
1. Use the downloaded sample images (from Kaggle, GitHub, or custom URLs)
2. Add your own image URLs to `scripts/download_test_dataset.sh` in the appropriate arrays
3. Generate synthetic plates using ALPR Dataset Generator
4. Use your own camera to capture test images with `scripts/take-one-photo.sh`

## Adding Custom Image URLs

To download specific images, edit `scripts/download_test_dataset.sh` and add URLs to:
- `image_urls` array in `download_sample_images()` function
- `repos` array in `download_github_images()` function (format: "user/repo/branch/path/to/image.jpg")

## Notes

- Most open-source datasets are limited in size
- For production training, consider combining multiple sources
- Always verify license compatibility for your use case
EOF

    echo "✓ Created dataset info: ${info_file}"
}

# Method 4: Download from Kaggle (requires kaggle API)
download_kaggle() {
    local kaggle_dir="${DATASET_DIR}/kaggle"
    mkdir -p "${kaggle_dir}"
    
    echo ""
    echo "Method 3: Attempting to download from Kaggle..."
    
    if ! command -v kaggle &> /dev/null; then
        echo "⚠️  Kaggle CLI not found. Install with: pip install kaggle"
        echo "   Then authenticate: kaggle datasets download -d abdelhamidzakaria/european-license-plates-dataset"
        return 1
    fi
    
    # Check if kaggle credentials are set
    if [ ! -f "${HOME}/.kaggle/kaggle.json" ]; then
        echo "⚠️  Kaggle credentials not found. Please:"
        echo "   1. Get API token from https://www.kaggle.com/settings"
        echo "   2. Save to ~/.kaggle/kaggle.json"
        echo "   3. Run: chmod 600 ~/.kaggle/kaggle.json"
        return 1
    fi
    
    echo "Downloading European License Plates Dataset from Kaggle..."
    cd "${kaggle_dir}"
    
    # European License Plates Dataset (includes German plates)
    if kaggle datasets download -d abdelhamidzakaria/european-license-plates-dataset -p . --unzip 2>&1; then
        echo "✓ Kaggle dataset downloaded and extracted"
    else
        echo "⚠️  Kaggle download failed. Check your credentials and dataset availability"
        return 1
    fi
    
    cd "${PROJECT_ROOT}"
}

# Method 5: Download sample images from public sources (if URLs are available)
download_public_samples() {
    local samples_dir="${DATASET_DIR}/public_samples"
    mkdir -p "${samples_dir}"
    
    echo ""
    echo "Method 4: Creating custom download script for public samples..."
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
download_sample_images || echo "⚠️  Sample images download failed or skipped"
download_github_images || echo "⚠️  GitHub images download failed or skipped"
download_kaggle || echo "⚠️  Kaggle download failed or skipped"
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
