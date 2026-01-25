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

# Check for Python (needed for Roboflow API)
if command -v python3 &> /dev/null; then
    echo "✓ Python3 found"
else
    echo "⚠️  Python3 not found. Some download methods may not work."
fi

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
    
    # Sample image URLs from public sources
    # These are example/test images that may be available
    local image_urls=(
        # Add any direct image URLs here when you find them
        # Example format:
        # "https://raw.githubusercontent.com/user/repo/main/images/sample1.jpg"
    )
    
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
    local repos=(
        # Example (uncomment and verify these paths exist):
        # "aboerzel/German_License_Plate_Recognition/master/data/sample1.jpg"
    )
    
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

1. **Roboflow: German License Plates** (Easiest - Public Domain)
   - Dataset: https://universe.roboflow.com/max-mustermann-gmm7j/german-license-plates-hptbz
   - 1,200 images for object detection
   - License: Public Domain
   - Download: Visit the link and click "Download Dataset" button
   - Or use Roboflow API with API key (see script output for instructions)

2. **Kaggle: European License Plates Dataset** (Largest - Requires Account)
   - Dataset: https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset
   - Includes German and other European license plates
   - Requires Kaggle account and API setup
   - Install: `pip install kaggle`
   - Download: `kaggle datasets download -d abdelhamidzakaria/european-license-plates-dataset`
   - This downloads actual image files

3. **THI License Plate Dataset (TLPD)** (Academic - Requires Contact)
   - 17,000+ vehicle images, 18,000+ labeled plates
   - Contact: Alessandro.Zimmer@thi.de
   - Website: https://www.thi.de/forschung/carissma/c-isafe/thi-license-plate-dataset/

4. **GitHub Sample Images**
   - Downloads specific image files from GitHub repositories using raw.githubusercontent.com
   - Add image URLs to the script to download specific samples
   - No need to clone entire repositories

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
1. **Roboflow** (Recommended for quick start): Visit the link above and download manually
2. **Kaggle**: Set up Kaggle API and run the script (it will download automatically)
3. Add your own image URLs to `scripts/download_test_dataset.sh` in the appropriate arrays
4. Generate synthetic plates using ALPR Dataset Generator
5. Use your own camera to capture test images with `scripts/take-one-photo.sh`

## Quick Start

**Easiest option**: Visit https://universe.roboflow.com/max-mustermann-gmm7j/german-license-plates-hptbz
and download the dataset manually, then extract to `data/test_images/german_plates/roboflow/`

**Automated option**: Set up Kaggle API credentials and the script will download automatically.

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

# Method 3: Download from Roboflow (public domain dataset)
download_roboflow() {
    local roboflow_dir="${DATASET_DIR}/roboflow"
    mkdir -p "${roboflow_dir}"
    
    echo ""
    echo "Method 3: Attempting to download from Roboflow..."
    echo "Dataset: German License Plates (1.2k images, Public Domain)"
    
    # Try to use Roboflow Python API if available
    if python3 -c "import roboflow" 2>/dev/null; then
        echo "Roboflow Python package found. Attempting download..."
        
        # Create a temporary Python script to download
        local download_script="${roboflow_dir}/_download_roboflow.py"
        cat > "${download_script}" << 'PYEOF'
import os
import sys

try:
    from roboflow import Roboflow
    
    # Public dataset - no API key needed for public datasets
    # But we'll try without API key first
    try:
        rf = Roboflow()
    except:
        # If that fails, try with empty key (some public datasets work this way)
        rf = Roboflow(api_key="")
    
    # Try to access the public dataset
    try:
        project = rf.workspace("max-mustermann-gmm7j").project("german-license-plates-hptbz")
        dataset = project.version(1).download("yolov8", location=sys.argv[1] if len(sys.argv) > 1 else ".")
        print("SUCCESS")
    except Exception as e:
        print(f"API_ERROR: {e}")
        print("Trying alternative method...")
        # Alternative: direct download URL (if available)
        raise
except ImportError:
    print("ROBOFLOW_NOT_INSTALLED")
except Exception as e:
    print(f"ERROR: {e}")
PYEOF

        cd "${roboflow_dir}"
        local result=$(python3 "${download_script}" "${roboflow_dir}" 2>&1)
        
        if echo "$result" | grep -q "SUCCESS"; then
            echo "✓ Roboflow dataset downloaded successfully"
            rm -f "${download_script}"
            cd "${PROJECT_ROOT}"
            return 0
        elif echo "$result" | grep -q "ROBOFLOW_NOT_INSTALLED"; then
            echo "⚠️  Roboflow Python package not installed"
            echo "   Install with: pip install roboflow"
            echo "   Or download manually from:"
        else
            echo "⚠️  Roboflow API download failed, trying manual method..."
        fi
        rm -f "${download_script}"
        cd "${PROJECT_ROOT}"
    else
        echo "⚠️  Roboflow Python package not installed"
        echo "   Install with: pip install roboflow"
        echo "   Or download manually:"
    fi
    
    # Fallback: Manual download instructions
    echo ""
    echo "   Manual download (recommended if API doesn't work):"
    echo "   1. Visit: https://universe.roboflow.com/max-mustermann-gmm7j/german-license-plates-hptbz"
    echo "   2. Click 'Download Dataset' button"
    echo "   3. Select format: YOLOv8"
    echo "   4. Download the ZIP file"
    echo "   5. Extract to: ${roboflow_dir}"
    echo ""
    
    # Check if dataset was manually downloaded
    if [ -d "${roboflow_dir}" ] && [ "$(find "${roboflow_dir}" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l | tr -d ' ')" -gt 0 ]; then
        echo "✓ Found images in Roboflow directory (manually downloaded)"
        return 0
    fi
    
    return 1
}

# Method 4: Download from Kaggle (requires kaggle API)
download_kaggle() {
    local kaggle_dir="${DATASET_DIR}/kaggle"
    mkdir -p "${kaggle_dir}"
    
    echo ""
    echo "Method 4: Attempting to download from Kaggle..."
    echo "Dataset: European License Plates (includes German plates)"
    
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
        cd "${PROJECT_ROOT}"
        return 0
    else
        echo "⚠️  Kaggle download failed. Check your credentials and dataset availability"
        cd "${PROJECT_ROOT}"
        return 1
    fi
}

# Method 5: Download sample images from public sources (if URLs are available)
download_public_samples() {
    local samples_dir="${DATASET_DIR}/public_samples"
    mkdir -p "${samples_dir}"
    
    echo ""
    echo "Method 5: Creating custom download script for public samples..."
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
download_roboflow || echo "⚠️  Roboflow download requires manual steps (see instructions above)"
download_kaggle || echo "⚠️  Kaggle download failed or skipped (requires API setup)"
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
