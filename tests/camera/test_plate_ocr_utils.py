from __future__ import annotations

import numpy as np

from camera import plate_pipeline


def test_normalize_plate_text_uppercases_and_strips_non_alnum() -> None:
    assert plate_pipeline._normalize_plate_text("  lIp-mh 328 ") == "LIPMH328"
    assert plate_pipeline._normalize_plate_text("M AB 1234") == "MAB1234"


def test_preprocess_plate_for_ocr_returns_bgr_image() -> None:
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    out = plate_pipeline._preprocess_plate_for_ocr(img)
    assert out.ndim == 3
    assert out.shape[2] == 3
    assert out.dtype == np.uint8
