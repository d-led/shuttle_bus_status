"""Camera plate detection package."""

from camera.plate_pipeline import (  # noqa: F401
    BBox,
    EasyOcrPlateRecognizer,
    EnsemblePlateRecognizer,
    ImagePlateDetections,
    PlateCandidate,
    PlateDetection,
    PlateOcr,
    UltralyticsYoloPlateDetector,
    create_plate_recognizer,
    create_plate_recognizer_from_config,
    detect_plates_in_image,
)

__all__ = [
    "EasyOcrPlateRecognizer",
    "EnsemblePlateRecognizer",
    "PlateOcr",
    "create_plate_recognizer",
    "create_plate_recognizer_from_config",
]

__version__ = "0.0.1"
