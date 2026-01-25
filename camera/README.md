# German License Plate Detection System

A polling-based system for detecting and logging German license plates from a USB camera on Raspberry Pi 3.

## Setup

### System Dependencies

Install system dependencies on the Raspberry Pi:

```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip python3-venv build-essential cmake pkg-config libjpeg-dev libpng-dev libtiff-dev libv4l-dev v4l-utils
```

### Python Dependencies

The project uses a virtual environment to avoid conflicts with system packages.

**Option 1: Use the setup script (recommended)**

From the project root, copy the files to the Pi and run:

```bash
scp -r camera scripts dled@dledpi:~/shuttle_bus_status/
ssh dled@dledpi
cd ~/shuttle_bus_status
./scripts/setup.sh
```

**Option 2: Manual installation**

```bash
# On the Raspberry Pi
python3 -m venv ~/camera-venv
source ~/camera-venv/bin/activate
cd ~/shuttle_bus_status/camera
pip install -e ".[dev]"
```

### Verify Camera Access

Test that the camera is accessible:

```bash
# List video devices
ls -la /dev/video*

# Check camera capabilities
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Take a test photo with timestamp (recommended)
./scripts/take-one-photo.sh

# Or capture to a specific directory
./scripts/take-one-photo.sh ~/photos
```

## Architecture

- `camera_capture.py` - USB camera access via OpenCV
- `plate_detector.py` - YOLOv8 model for license plate detection
- `plate_recognizer.py` - EasyOCR for German text extraction
- `plate_tracker.py` - Debouncing and state management
- `logger.py` - Structured logging of plate events
- `main.py` - Main polling loop
- `config.toml` - Configuration file

## Configuration

Configuration is managed via `config.toml` in the project root. The configuration includes:

- **Camera settings**: Device path (use "auto" for automatic detection), resolution, frame rate
- **Plate detection**: YOLOv8 model size, confidence threshold, polling interval
- **Plate recognition**: OCR languages, minimum confidence
- **Debouncing**: Appearance/disappearance thresholds
- **Logging**: Log file path, log level, console output

Example configuration:

```toml
[camera]
device = "auto"  # or "/dev/video0" for specific device
width = 1920
height = 1080
fps = 30
```

See `config.toml` in the project root for all available options.

## Running

Run from the project root:

```bash
source ~/camera-venv/bin/activate
cd ~/shuttle_bus_status
python -m camera.main
```

Or use the entry point:

```bash
source ~/camera-venv/bin/activate
camera-plate-detection
```

## Testing

Run tests from the project root:

```bash
./scripts/test.sh
```

## Dependencies

All dependencies are specified in `pyproject.toml`. The packages have been selected to use prebuilt aarch64 wheels where available to minimize compilation time on the Raspberry Pi.

- **opencv-python**: Camera access and image processing (has aarch64 wheels)
- **ultralytics**: YOLOv8 model framework (platform-agnostic wheels)
- **easyocr**: OCR for German text recognition (platform-agnostic wheels)
- **numpy**: Image array handling
- **pillow**: Image preprocessing
- **pydantic**: Configuration management with type validation
