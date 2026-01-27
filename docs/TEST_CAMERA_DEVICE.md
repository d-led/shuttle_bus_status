# Test Camera Device

## Overview

The test camera device allows you to test the plate detection and recognition system without a real camera. It cycles through images from a directory, **lingering on each image for 1-15 cycles** (read() calls) before switching to the next image.

**Status**: ✅ Implemented and tested

## Usage

### Configuration

Set the camera device in `config.toml`:

```toml
[camera]
device = "test:data/test_camera"  # Use self-contained test dataset
# Or use full dataset:
# device = "test:data/test_images/german_plates/kaggle"

[test_camera]
# Minimum cycles to linger on each image (number of read() calls before switching)
min_duration_seconds = 1.0  # Interpreted as cycles, not seconds

# Maximum cycles to linger on each image (number of read() calls before switching)
max_duration_seconds = 15.0  # Interpreted as cycles, not seconds
```

The format is: `test:<path_to_image_directory>`

**Self-contained test dataset**: Run `scripts/prepare_test_dataset.sh` to create a small test dataset with ~20 good plate images in `data/test_camera/`. This is perfect for quick testing without needing the full dataset.

### How It Works

1. **Image Loading**: Scans the specified directory for images (`.jpg`, `.jpeg`, `.png`)
2. **Random Cycles**: Each image is shown for a random number of cycles (read() calls) between `min_duration_seconds` and `max_duration_seconds` (default: 1-15 cycles)
3. **Lingering**: The same image is returned for multiple consecutive read() calls, allowing multiple detection attempts on the same image
4. **Cycling**: After the cycle count expires, automatically switches to the next image
5. **Shuffling**: Images are shuffled for randomness

### Features

- **OpenCV-Compatible**: Implements the same interface as `cv2.VideoCapture`, so it works seamlessly with the existing camera stream controller
- **Automatic Detection**: The test device appears in the camera device selector in the UI if the dataset exists
- **Configurable**: Supports width/height resizing, duration ranges

### Example

```toml
[camera]
# Use test camera with images from Kaggle dataset
device = "test:data/test_images/german_plates/kaggle"

# Optional: resize images
width = 1920
height = 1080
```

### Benefits

1. **No Hardware Required**: Test the system without a physical camera
2. **Reproducible**: Use the same test images for consistent testing
3. **Realistic**: Images linger for multiple cycles, allowing multiple detection attempts on the same image (simulating a plate staying in view)
4. **Development**: Perfect for development and debugging
5. **Cycle-Based**: Each image is shown for multiple read() cycles, giving the detection system multiple chances to detect the same plate

### Integration

The test device is automatically detected and listed in the camera device selector if:
- The dataset directory exists at `data/test_camera` (self-contained) or `data/test_images/german_plates/kaggle` (full dataset)
- Images are found in the directory

**Important**: 
- Test devices appear **first** in the UI dropdown for easy selection
- When `device = "auto"` is configured, the system will **prefer real cameras** over test devices
- Test device is only used with "auto" if no real camera is available
- To explicitly use test device, set `device = "test:data/test_camera"` in config

You can select it from the UI dropdown just like a real camera device.
