#!/usr/bin/env bash
set -euo pipefail

# Download script for German license plate test/detection datasets
# Downloads from multiple open-source sources
# Usage: ./scripts/download_test_dataset.sh [output_directory]
#
# QUICK START (Easiest option):
#   1. Install Roboflow: pip install roboflow
#   2. Run this script - it will try to download automatically
#   OR manually download from:
#      https://universe.roboflow.com/max-mustermann-gmm7j/german-license-plates-hptbz
#      Extract to: data/test_images/german_plates/roboflow/
#
# ALTERNATIVE (Largest dataset):
#   1. Install Kaggle: pip install kaggle
#   2. Get API key from https://www.kaggle.com/settings
#   3. Save to ~/.kaggle/kaggle.json and chmod 600 it
#   4. Run this script - it will download automatically

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${1:-${PROJECT_ROOT}/data/test_images}"
DATASET_DIR="${OUTPUT_DIR}/german_plates"

echo "=========================================="
echo "German License Plate Dataset Downloader"
echo "=========================================="
echo "Output directory: ${DATASET_DIR}"
echo ""
echo "Recommended: Install 'roboflow' for automatic download"
echo "  pip install roboflow"
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
    
    # Try to install roboflow if not available
    if ! python3 -c "import roboflow" 2>/dev/null; then
        echo "Installing roboflow package..."
        if python3 -m pip install --quiet roboflow 2>/dev/null; then
            echo "✓ roboflow installed"
        else
            echo "⚠️  Failed to install roboflow automatically"
            echo "   Try manually: pip install roboflow"
            echo "   Or download manually from: https://universe.roboflow.com/max-mustermann-gmm7j/german-license-plates-hptbz"
            return 1
        fi
    fi
    
    # Try to use Roboflow Python API
    if python3 -c "import roboflow" 2>/dev/null; then
        echo "Roboflow Python package found. Attempting download..."
        
        # Create a temporary Python script to download
        local download_script="${roboflow_dir}/_download_roboflow.py"
        cat > "${download_script}" << 'PYEOF'
import os
import sys

try:
    from roboflow import Roboflow
    
    # Roboflow requires an API key even for public datasets
    # Check for API key in environment variable or config
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    
    if not api_key:
        # Try to read from config file if it exists
        config_path = os.path.expanduser("~/.roboflow/config")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        if line.startswith("api_key"):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            except:
                pass
    
    if not api_key:
        print("NO_API_KEY")
        print("Roboflow requires an API key even for public datasets.")
        print("Get your API key from: https://app.roboflow.com/")
        print("Then set: export ROBOFLOW_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Initialize Roboflow with API key
    rf = Roboflow(api_key=api_key)
    
    # Try to access the public dataset
    try:
        project = rf.workspace("max-mustermann-gmm7j").project("german-license-plates-hptbz")
        # Try version 1 first, if that fails try latest version
        try:
            dataset = project.version(1).download("yolov8", location=sys.argv[1] if len(sys.argv) > 1 else ".")
        except:
            # If version 1 doesn't exist, try downloading without specifying version (gets latest)
            dataset = project.download("yolov8", location=sys.argv[1] if len(sys.argv) > 1 else ".")
        print("SUCCESS")
    except Exception as e:
        print(f"API_ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
except ImportError:
    print("ROBOFLOW_NOT_INSTALLED")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

        cd "${roboflow_dir}"
        local result=$(python3 "${download_script}" "${roboflow_dir}" 2>&1)
        
        if echo "$result" | grep -q "SUCCESS"; then
            echo "✓ Roboflow dataset downloaded successfully"
            rm -f "${download_script}"
            cd "${PROJECT_ROOT}"
            return 0
        elif echo "$result" | grep -q "NO_API_KEY"; then
            echo ""
            echo "⚠️  Roboflow API key required"
            echo "   Roboflow requires an API key even for public datasets."
            echo "   Get your API key from: https://app.roboflow.com/"
            echo "   Then set: export ROBOFLOW_API_KEY='your-api-key'"
            echo "   Or download manually from:"
        elif echo "$result" | grep -q "ROBOFLOW_NOT_INSTALLED"; then
            echo "⚠️  Roboflow Python package not installed"
            echo "   Install with: pip install roboflow"
            echo "   Or download manually from:"
        else
            echo "⚠️  Roboflow API download failed:"
            echo "$result" | grep -E "ERROR|API_ERROR" | head -3
            echo "   Trying manual method..."
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
    
    # Try to install kaggle if not available
    if ! command -v kaggle &> /dev/null; then
        if ! python3 -c "import kaggle" 2>/dev/null; then
            echo "Installing kaggle package..."
            if python3 -m pip install --quiet kaggle 2>/dev/null; then
                echo "✓ kaggle installed"
            else
                echo "⚠️  Failed to install kaggle automatically"
                echo "   Try manually: pip install kaggle"
                return 1
            fi
        fi
    fi
    
    # Check if kaggle credentials are set (either JSON file or environment variable)
    HAS_CREDENTIALS=false
    
    # Check for JSON file first (takes precedence)
    if [ -f "${HOME}/.kaggle/kaggle.json" ]; then
        HAS_CREDENTIALS=true
        echo "✓ Found Kaggle credentials in ~/.kaggle/kaggle.json"
    # Check for environment variable if JSON file doesn't exist
    elif [ -n "${KAGGLE_API_TOKEN:-}" ]; then
        echo "✓ Found Kaggle API token in KAGGLE_API_TOKEN environment variable"
        # Create temporary JSON file from environment variable
        mkdir -p "${HOME}/.kaggle"
        # KAGGLE_API_TOKEN format: username:token
        if [[ "${KAGGLE_API_TOKEN}" == *":"* ]]; then
            KAGGLE_USERNAME="${KAGGLE_API_TOKEN%%:*}"
            KAGGLE_KEY="${KAGGLE_API_TOKEN#*:}"
            cat > "${HOME}/.kaggle/kaggle.json" << EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
            chmod 600 "${HOME}/.kaggle/kaggle.json" 2>/dev/null || true
            echo "✓ Created kaggle.json from environment variable"
            HAS_CREDENTIALS=true
        else
            echo "⚠️  KAGGLE_API_TOKEN should be in format 'username:token'"
            HAS_CREDENTIALS=false
        fi
    fi
    
    if [ "$HAS_CREDENTIALS" = false ]; then
        echo "⚠️  Kaggle credentials not found. Skipping Kaggle download."
        echo "   To use Kaggle, choose one method:"
        echo ""
        echo "   Method 1 (JSON file):"
        echo "   1. Get API token from https://www.kaggle.com/settings"
        echo "   2. Save to ~/.kaggle/kaggle.json with format:"
        echo "      {\"username\":\"your-username\",\"key\":\"your-api-key\"}"
        echo "   3. Run: chmod 600 ~/.kaggle/kaggle.json"
        echo ""
        echo "   Method 2 (Environment variable):"
        echo "   export KAGGLE_API_TOKEN='your-username:your-api-key'"
        echo ""
        echo "   Then run this script again"
        return 1
    fi
    
    # Verify credentials are valid
    if ! kaggle datasets list --max-size 1 &> /dev/null; then
        echo "⚠️  Kaggle credentials appear invalid. Please check your credentials"
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
    echo ""
    echo "Image locations:"
    find "${DATASET_DIR}" -type d -exec sh -c 'count=$(find "$1" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.bmp" \) 2>/dev/null | wc -l | tr -d " "); if [ "$count" -gt 0 ]; then echo "  $1: $count images"; fi' _ {} \; 2>/dev/null | grep -v ": 0 images" || true
    echo ""
    echo "Sample image files:"
    find "${DATASET_DIR}" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | head -5 | while read -r img; do
        echo "  ${img}"
    done
    if [ "${IMAGE_COUNT}" -gt 5 ]; then
        echo "  ... and $((IMAGE_COUNT - 5)) more"
    fi
else
    echo "⚠️  No images found. You may need to:"
    echo "   1. Install roboflow: pip install roboflow (then run script again)"
    echo "   2. Manually download from: https://universe.roboflow.com/max-mustermann-gmm7j/german-license-plates-hptbz"
    echo "   3. Set up Kaggle API (see instructions above)"
    echo "   4. Add custom download URLs to ${DATASET_DIR}/public_samples/download_samples.sh"
    echo "   5. Use scripts/take-one-photo.sh to capture your own test images"
fi

echo ""
echo "Next steps:"
echo "  1. Review ${DATASET_DIR}/README.md for dataset information"
if [ "${IMAGE_COUNT}" -gt 0 ]; then
    echo "  2. Images are ready in: ${DATASET_DIR}"
    echo "  3. Use these images for testing your license plate detection"
else
    echo "  2. Follow the download instructions above to get test images"
    echo "  3. Consider using synthetic data generation for more samples"
fi
echo ""
