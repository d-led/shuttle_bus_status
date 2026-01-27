from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from camera.plate_pipeline import (
    BBox,
    ImagePlateDetections,
    PlateCandidate,
    detect_plates_in_image,
)


@dataclass(frozen=True)
class _FakeDetector:
    def detect(self, _image_bgr: np.ndarray) -> list[PlateCandidate]:
        return [
            PlateCandidate(
                bbox=BBox(x1=1, y1=2, x2=6, y2=7),
                confidence=0.9,
                metadata={"source": "fake"},
            )
        ]


@dataclass(frozen=True)
class _FakeOcr:
    def recognize(self, _plate_bgr: np.ndarray):  # type: ignore[no-untyped-def]
        return (
            "M AB 1234",
            0.8,
            {"engine": "fake", "raw_text": "M AB 1234", "raw_confidence": 0.8},
        )


def test_detect_plates_in_image_returns_typed_result_with_timestamp() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    result = detect_plates_in_image(
        image_bgr=image,
        detector=_FakeDetector(),
        ocr=_FakeOcr(),
        captured_at=ts,
    )

    assert isinstance(result, ImagePlateDetections)
    assert result.captured_at == ts
    assert result.image_size == (10, 10)
    assert result.plate_count == 1
    det = result.detections[0]
    assert det.text == "M AB 1234"
    assert det.detection_confidence == 0.9
    assert det.ocr_confidence == 0.8
    assert det.raw_text == "M AB 1234"
    assert det.raw_ocr_confidence == 0.8
    assert det.reliability == 0.9 * 0.8
    assert det.crop_jpeg_b64 is None


def test_detect_plates_in_image_clamps_bbox_to_image() -> None:
    image = np.zeros((5, 5, 3), dtype=np.uint8)

    class _OutOfBoundsDetector:
        def detect(self, _image_bgr: np.ndarray) -> list[PlateCandidate]:
            return [
                PlateCandidate(bbox=BBox(-10, -10, 50, 50), confidence=0.5, metadata={})
            ]

    class _NoopOcr:
        def recognize(self, _plate_bgr: np.ndarray):  # type: ignore[no-untyped-def]
            return None, None, {}

    result = detect_plates_in_image(
        image_bgr=image,
        detector=_OutOfBoundsDetector(),
        ocr=_NoopOcr(),
        captured_at=datetime.now(timezone.utc),
    )

    bbox = result.detections[0].bbox
    assert bbox.x1 == 0
    assert bbox.y1 == 0
    assert bbox.x2 == 5
    assert bbox.y2 == 5
