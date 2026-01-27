#!/usr/bin/env python3
"""Diagnostic script to understand OCR failures."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.config import Settings
from camera.plate_pipeline import BBox, PlateCandidate, create_plate_recognizer, detect_plates_from_candidates


def main() -> int:
    settings = Settings.load_from_project_root()
    rec = settings.plate_recognition

    # Find a few sample images
    kaggle_dir = PROJECT_ROOT / "data/test_images/german_plates/kaggle"
    if not kaggle_dir.exists():
        print(f"Kaggle dataset not found: {kaggle_dir}")
        return 1

    images = sorted(list(kaggle_dir.rglob("*.jpg")) + list(kaggle_dir.rglob("*.png")))[:10]

    from camera.plate_pipeline import create_plate_recognizer_from_config
    
    ocr = create_plate_recognizer_from_config(
        ocr_engine=rec.ocr_engine,
        languages=rec.languages,
        min_confidence=0.1,  # Lower threshold to see all results
        preprocess=rec.preprocess,
        allowlist=rec.allowlist,
        normalize=rec.normalize,
    )

    print(f"OCR Engine: {rec.ocr_engine}")
    print(f"Preprocess: {rec.preprocess}, Allowlist: {rec.allowlist}, Normalize: {rec.normalize}")
    print(f"\nAnalyzing {len(images)} sample images:\n")

    for img_path in images:
        # Extract ground truth
        stem = img_path.stem.replace("_1", "").replace("_2", "")
        gt = stem.upper()

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        aspect = w / h if h > 0 else 0

        # Use full image as candidate (images are already cropped plates)
        candidate = PlateCandidate(
            bbox=BBox(x1=0, y1=0, x2=w, y2=h),
            confidence=1.0,
            metadata={"source": "full_image"},
        )

        res = detect_plates_from_candidates(
            image_bgr=img,
            candidates=[candidate],
            ocr=ocr,
            include_crops=False,
        )

        if res.detections:
            det = res.detections[0]
            match = "✓" if det.text and det.text.upper().replace(" ", "") == gt.replace(" ", "") else "✗"
            print(
                f"{match} {img_path.name:30s} GT: {gt:12s} OCR: {det.text or '(none)':20s} "
                f"Raw: {det.raw_text or '(none)':20s} Conf: {det.raw_ocr_confidence or 0:.3f} "
                f"Size: {w}x{h} (aspect: {aspect:.2f})"
            )
        else:
            print(f"✗ {img_path.name:30s} GT: {gt:12s} OCR: (no detection)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
