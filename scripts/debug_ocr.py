#!/usr/bin/env python3
"""Debug OCR to see what's actually being detected."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.plate_pipeline import EasyOcrPlateRecognizer


def main() -> int:
    # Test the problematic image from the report
    test_img_path = "data/test_images/german_plates/roboflow/german-license-plates-7/test/images/10650668_jpg.rf.22344aab0e953a367c653edbdc45646a.jpg"
    img = cv2.imread(test_img_path)
    if img is None:
        print(f"Could not load: {test_img_path}")
        return 1

    # Extract plate region (from report: bbox x1=164, y1=180, x2=249, y2=200)
    crop = img[180:200, 164:249]
    print(f"Crop size: {crop.shape}")
    print(f"Expected: 'LIP VE 351'")
    print()

    # Test with preprocessing disabled first
    ocr_no_prep = EasyOcrPlateRecognizer(
        languages=["en"],
        min_confidence=0.1,
        preprocess=False,
        normalize=False,  # Don't normalize to see raw results
    )

    print("=== Without Preprocessing ===")
    plate_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    kwargs = {"paragraph": False, "width_ths": 0.05, "height_ths": 0.05, "detail": 1}
    results = ocr_no_prep._reader.readtext(plate_rgb, **kwargs)
    print(f"EasyOCR detected {len(results)} text regions:")
    for i, item in enumerate(results):
        bbox, text, conf = item[0], item[1], item[2]
        print(f"  {i+1}. Text: {text!r}, Confidence: {conf:.3f}")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            y_coords = [pt[1] for pt in bbox if isinstance(pt, (list, tuple)) and len(pt) >= 2]
            x_coords = [pt[0] for pt in bbox if isinstance(pt, (list, tuple)) and len(pt) >= 2]
            if y_coords and x_coords:
                y_center = sum(y_coords) / len(y_coords)
                x_center = sum(x_coords) / len(x_coords)
                print(f"      Position: x={x_center:.1f}, y={y_center:.1f}")

    text, conf, meta = ocr_no_prep.recognize(crop)
    print(f"\nFinal result: {text} (conf={conf:.3f})")
    print()

    # Test with preprocessing enabled
    print("=== With Multiple Preprocessing Strategies ===")
    ocr_prep = EasyOcrPlateRecognizer(
        languages=["en"],
        min_confidence=0.1,
        preprocess=True,
        normalize=False,  # Don't normalize to see raw results
    )

    text, conf, meta = ocr_prep.recognize(crop)
    print(f"Final result: {text} (conf={conf:.3f})")
    print(f"Strategy used: {meta.get('preprocessing_strategy_used', 'unknown')}")
    if meta.get("all_strategy_results"):
        print(f"\nAll strategy results:")
        for r in meta.get("all_strategy_results", []):
            print(f"  {r['strategy']}: {r['text']!r} (conf={r['conf']:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
