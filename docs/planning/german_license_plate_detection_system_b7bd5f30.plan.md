---
name: German License Plate Detection System
overview: Set up a polling system on Raspberry Pi 3 to detect and log German license plates from a USB camera with debouncing to avoid flaky detections.
todos:
  - id: setup-project
    content: Create camera/ folder structure and requirements.txt with OpenCV, Ultralytics, EasyOCR dependencies
    status: pending
  - id: camera-capture
    content: Implement camera_capture.py using OpenCV to access /dev/video0 USB camera
    status: pending
  - id: plate-detection
    content: Implement plate_detector.py with YOLOv8 model for license plate detection
    status: pending
  - id: text-recognition
    content: Implement plate_recognizer.py with EasyOCR for German plate text extraction
    status: pending
  - id: debouncing
    content: Implement plate_tracker.py with debouncing logic for appearance/disappearance events
    status: pending
  - id: logging
    content: Implement logger.py for structured logging of plate events
    status: pending
  - id: main-loop
    content: Implement main.py polling loop integrating all components
    status: pending
  - id: testing
    content: Test on Raspberry Pi with actual camera feed and verify detection accuracy
    status: pending
isProject: false
---

# German License Plate Detection System

## Technology Stack Decision

Based on research and the Raspberry Pi 3 environment:

### Recommended Approach: Python with YOLOv8 + EasyOCR

**Rationale:**

- Python 3.13.5 is already installed on the Pi
- YOLOv8 + EasyOCR is a proven combination for German license plates (documented in IEEE research)
- OpenCV provides straightforward USB camera access via v4l2
- Good balance of accuracy, performance, and maintainability
- Active community support and pre-trained models available

**Alternative Considered:**

- UltimateALPR-SDK: Fastest option (12fps on RPi4, likely 6-8fps on RPi3) but:
  - Only C++/Python/Java/C# bindings (no Go/Rust)
  - More complex setup and licensing considerations
  - May be overkill for a polling-based system

**Go/Rust Consideration:**

- Go/Rust not installed on Pi (would require installation)
- Limited ANPR library support in Go/Rust ecosystems
- Would need to call Python libraries via FFI/subprocess, adding complexity
- Python ecosystem has mature, optimized libraries for this use case

## Architecture

```
camera/
├── main.py              # Main polling loop with debouncing
├── camera_capture.py     # USB camera access via OpenCV
├── plate_detector.py     # YOLOv8 plate detection
├── plate_recognizer.py   # EasyOCR text extraction
├── plate_tracker.py      # Debouncing and state management
├── logger.py             # Logging appearance/disappearance events
├── requirements.txt      # Python dependencies
└── README.md            # Setup and usage instructions
```

## Implementation Plan

### 1. Camera Access (`camera_capture.py`)

- Use OpenCV (`cv2.VideoCapture`) to access `/dev/video0`
- Configure for 1080p capture (camera supports Full HD)
- Handle H.264 format if available, fallback to MJPEG/raw
- Implement frame buffering for consistent polling

### 2. License Plate Detection (`plate_detector.py`)

- Use Ultralytics YOLOv8 model (pre-trained or fine-tuned for plates)
- Load model on startup (one-time cost)
- Process frames at configurable interval (e.g., 1-2 fps for polling)
- Return bounding boxes for detected plates

### 3. Text Recognition (`plate_recognizer.py`)

- Use EasyOCR with German language support
- Preprocess detected plate regions (grayscale, contrast enhancement)
- Extract text and validate German plate format (e.g., `AB-C 1234` pattern)
- Return normalized plate strings

### 4. State Tracking & Debouncing (`plate_tracker.py`)

- Maintain set of currently visible plates
- Implement debouncing logic:
  - Plate must be detected N times in M seconds to "appear"
  - Plate must be absent for K seconds to "disappear"
- Track timestamps and detection counts per plate

### 5. Event Logging (`logger.py`)

- Log appearance events: timestamp, plate number, confidence
- Log disappearance events: timestamp, plate number, duration visible
- Output format: JSON lines or structured log file
- Optional: stdout for real-time monitoring

### 6. Main Polling Loop (`main.py`)

- Initialize all components
- Poll camera at configured interval (e.g., every 0.5-1 second)
- Process frame through detection → recognition → tracking pipeline
- Handle errors gracefully (camera disconnection, model failures)
- Clean shutdown on SIGTERM/SIGINT

## Dependencies

**Core:**

- `opencv-python` - Camera access and image processing
- `ultralytics` - YOLOv8 model framework
- `easyocr` - OCR for German text recognition
- `numpy` - Image array handling

**Optional:**

- `Pillow` - Additional image preprocessing
- `python-dateutil` - Timestamp handling

## Performance Considerations

- **Raspberry Pi 3 limitations:**
  - CPU: ARM Cortex-A53 (4 cores @ 1.2GHz)
  - RAM: 1GB (shared with GPU)
  - Expected performance: 1-2 FPS for full pipeline (detection + OCR)

- **Optimization strategies:**
  - Use YOLOv8 nano model (smallest, fastest)
  - Process every Nth frame (skip frames for faster polling)
  - Reduce input resolution if needed (720p instead of 1080p)
  - Cache OCR model in memory
  - Consider running detection and OCR in separate threads

## Debouncing Configuration

Default debouncing parameters:

- **Appearance threshold**: Plate detected 3 times in 2 seconds
- **Disappearance threshold**: Plate absent for 5 seconds
- Configurable via command-line args or config file

## Next Steps

1. Create project structure in `camera/` folder
2. Set up `requirements.txt` with dependencies
3. Implement camera capture module
4. Integrate YOLOv8 plate detection
5. Add EasyOCR text recognition
6. Implement debouncing logic
7. Create main polling loop
8. Add logging functionality
9. Test on Raspberry Pi with actual camera feed