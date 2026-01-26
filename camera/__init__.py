"""Camera plate detection package."""

__version__ = "0.0.1"

from camera.plate_pipeline import (  # noqa: F401
    BBox,
    EasyOcrPlateRecognizer,
    ImagePlateDetections,
    PlateCandidate,
    PlateDetection,
    UltralyticsYoloPlateDetector,
    detect_plates_in_image,
)
