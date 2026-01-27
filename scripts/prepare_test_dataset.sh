#!/usr/bin/env bash
# Prepare a self-contained test dataset for the test camera device

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TEST_DATA_DIR="data/test_camera"
SOURCE_DIR="data/test_images/german_plates/kaggle/dataset_final/test"

echo "Preparing test camera dataset..."

# Create test data directory
mkdir -p "$TEST_DATA_DIR"

# Use Python to select and copy good images
python3 <<'PYTHON_SCRIPT'
import re
import shutil
from pathlib import Path

def is_german_plate_format(text: str) -> bool:
    if not text:
        return False
    normalized = re.sub(r'[^A-Z0-9]+', '', text.upper())
    if len(normalized) < 5 or len(normalized) > 10:
        return False
    if re.match(r'^\d{4,}', normalized):
        return False
    if re.match(r'^[A-Z]{4,}\d', normalized):
        return False
    if sum(1 for c in normalized if c.isdigit()) > 4:
        return False
    if re.match(r'^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}$', normalized):
        return True
    if re.match(r'^[A-Z]{1,3}\d{1,4}$', normalized):
        return True
    if re.match(r'^[A-Z]{1,2}\d{1,4}$', normalized):
        return True
    if re.match(r'^\d{1,3}[A-Z]{2,3}$', normalized) and 5 <= len(normalized) <= 6:
        return True
    return False

# Find images for test dataset
# Prioritize Kaggle dataset (has German plates with annotated filenames - ground truth!)
# Then add Roboflow (has actual German plates like LIP*, but filenames are hashed)
roboflow_dir = Path('data/test_images/german_plates/roboflow')
kaggle_dir = Path('data/test_images/german_plates/kaggle')
test_dir = Path('data/test_camera')

images = []
# First prioritize Kaggle dataset (has plate numbers in filenames - annotated ground truth!)
if kaggle_dir.exists():
    for ext in ['.jpg', '.JPG', '.png', '.PNG']:
        for img_path in kaggle_dir.rglob(f'*{ext}'):
            stem = img_path.stem
            # Remove common suffixes like _1, _2, etc.
            clean_stem = re.sub(r'_[0-9]+$', '', stem)
            if is_german_plate_format(clean_stem):
                images.append(img_path)
            if len(images) >= 20:
                break
        if len(images) >= 20:
            break

# Then add Roboflow (has actual German plates like LIP*, mixed in)
# Roboflow filenames are hashed, so we can't extract plate numbers from them,
# but the images themselves contain German plates
if len(images) < 20 and roboflow_dir.exists():
    # Prefer test split, then valid, then train
    for split in ['test', 'valid', 'train']:
        split_dir = roboflow_dir / 'german-license-plates-7' / split / 'images'
        if split_dir.exists():
            for ext in ['.jpg', '.JPG', '.png', '.PNG']:
                for img_path in split_dir.glob(f'*{ext}'):
                    if img_path in images:
                        continue  # Skip duplicates
                    # Check if image has a corresponding label (indicates it has a plate)
                    label_path = split_dir.parent / 'labels' / f'{img_path.stem}.txt'
                    if label_path.exists():
                        images.append(img_path)
                    if len(images) >= 20:
                        break
                if len(images) >= 20:
                    break
        if len(images) >= 20:
            break

# Select diverse, good quality images
selected = []
seen_plates = set()
for img_path in sorted(images):
    stem = img_path.stem
    clean_stem = re.sub(r'_[0-9]+$', '', stem).upper()
    if clean_stem in seen_plates:
        continue  # Skip duplicates
    size = img_path.stat().st_size
    # Accept wider size range to get more images
    if 5_000 < size < 150_000:  # Reasonable size (5KB-150KB)
        selected.append(img_path)
        seen_plates.add(clean_stem)
    if len(selected) >= 20:
        break

# If we still don't have 20, relax the size constraint
if len(selected) < 20:
    for img_path in sorted(images):
        stem = img_path.stem
        clean_stem = re.sub(r'_[0-9]+$', '', stem).upper()
        if clean_stem in seen_plates:
            continue
        if img_path not in selected:
            selected.append(img_path)
            seen_plates.add(clean_stem)
        if len(selected) >= 20:
            break

# Copy selected images
print(f"Copying {len(selected)} images to {test_dir}...")
for img_path in selected:
    dest = test_dir / img_path.name
    shutil.copy2(img_path, dest)
    print(f"  Copied: {img_path.name} ({img_path.stat().st_size // 1024}KB)")

print(f"\n✓ Test dataset prepared: {test_dir}")
print(f"  Total images: {len(selected)}")
print(f"\n⚠️  License Notice:")
if len(selected) > 0:
    # Check if we used Roboflow (has .rf. in filename) or Kaggle
    roboflow_count = sum(1 for img in selected if '.rf.' in img.name.lower())
    kaggle_count = len(selected) - roboflow_count
    if kaggle_count > 0:
        print(f"  Images are primarily from: Kaggle European License Plates Dataset (annotated filenames)")
        print(f"  Source: https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset")
    if roboflow_count > 0:
        if kaggle_count > 0:
            print(f"  Mixed with: Roboflow German License Plates (Public Domain)")
        else:
            print(f"  Images are from: Roboflow German License Plates (Public Domain)")
        print(f"  Source: https://universe.roboflow.com/max-mustermann-gmm7j/german-license-plates-hptbz")
print(f"  See {test_dir}/LICENSE.md for attribution requirements")
PYTHON_SCRIPT

echo ""
echo "✓ Test camera dataset prepared successfully!"
echo "  Location: $TEST_DATA_DIR"
echo ""
echo "To use the test camera device, set in config.toml:"
echo "  [camera]"
echo "  device = \"test:data/test_camera\""
