# YOLO-Based Character Recognition Approach

## Overview

This document describes an alternative approach to license plate character recognition using YOLO models instead of OCR engines. This approach is used by projects like [aimanelias/license-plate-recognition](https://github.com/aimanelias/license-plate-recognition).

## Two-Stage YOLO Approach

### Current Approach (OCR-Based)
1. **Stage 1**: YOLO model detects license plate bounding boxes
2. **Stage 2**: OCR engines (EasyOCR, PaddleOCR, Tesseract) recognize characters

### Alternative Approach (YOLO-Based)
1. **Stage 1**: YOLO model detects license plate bounding boxes (same as current)
2. **Stage 2**: Separate YOLO model trained specifically for character recognition

## Advantages of YOLO Character Recognition

1. **Higher Accuracy**: Models trained specifically on license plate characters often outperform general-purpose OCR
2. **Consistent Format**: YOLO models can be trained to recognize the exact character set and format of German plates
3. **Better Performance**: Can be optimized for specific hardware (e.g., TensorRT on NVIDIA GPUs)
4. **End-to-End Training**: Can fine-tune on your specific dataset

## Implementation Example

Based on [aimanelias/license-plate-recognition](https://github.com/aimanelias/license-plate-recognition):

### Model Structure
- **Plate Detection Model**: `best_16.pt` - Detects license plate bounding boxes
- **Character Recognition Model**: `best.pt` - Recognizes individual characters on cropped plates

### Training Data
- Annotated 500+ images for character recognition
- Labels include individual character bounding boxes (A-Z, 0-9)

### Workflow
1. Detect plate using YOLO detection model
2. Crop plate region from image
3. Run character recognition YOLO model on cropped plate
4. Parse character detections into final plate text

## Integration with Current System

To integrate this approach into our system:

1. **Add YOLO Character Recognizer**:
   ```python
   class YoloCharacterRecognizer:
       def __init__(self, model_path: Path, confidence_threshold: float = 0.5):
           from ultralytics import YOLO
           self._model = YOLO(str(model_path))
           self._confidence_threshold = confidence_threshold
       
       def recognize(self, plate_bgr: np.ndarray) -> tuple[str | None, float | None, dict]:
           # Run YOLO on cropped plate
           results = self._model.predict(plate_bgr, conf=self._confidence_threshold)
           # Parse character detections
           # Sort by x-coordinate to get left-to-right order
           # Combine into final text
   ```

2. **Update `create_plate_recognizer`** to support `engine="yolo_chars"`

3. **Training a Custom Model**:
   - Collect German license plate images
   - Annotate with character-level bounding boxes
   - Train YOLOv8/YOLOv11 model on character classes (A-Z, 0-9)

## Comparison with OCR

| Aspect | OCR (Current) | YOLO Characters |
|--------|---------------|-----------------|
| **Accuracy** | 35% (EasyOCR) | Potentially 70-90%+ (if well-trained) |
| **Training Required** | No | Yes (500+ annotated images) |
| **Speed** | Fast | Fast (with GPU optimization) |
| **Language Support** | Multi-language | Single format (German plates) |
| **Hardware** | CPU/GPU | GPU optimized (TensorRT) |

## Recommendations

1. **Short-term**: Continue with OCR approach (EasyOCR at 35% is acceptable for now)
2. **Medium-term**: If accuracy needs improvement, consider:
   - Training a YOLO character recognition model on German plates
   - Using the pre-trained model from aimanelias if compatible
3. **Long-term**: Hybrid approach - use YOLO for character recognition, OCR as fallback

## Resources

- [aimanelias/license-plate-recognition](https://github.com/aimanelias/license-plate-recognition) - Complete DeepStream implementation
- [Ultralytics YOLO](https://docs.ultralytics.com/) - YOLO training and inference
- [Roboflow](https://roboflow.com/) - Dataset annotation and training platform

## Notes

- The aimanelias project uses DeepStream (NVIDIA-specific), which may not be suitable for Raspberry Pi
- Our current approach is more portable (works on CPU/any GPU)
- YOLO character recognition would require GPU acceleration for real-time performance
- Consider the trade-off between accuracy and deployment complexity
