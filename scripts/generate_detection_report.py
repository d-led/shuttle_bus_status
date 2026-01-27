#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.plate_pipeline import (
    BBox,
    EasyOcrPlateRecognizer,
    PlateCandidate,
    UltralyticsYoloPlateDetector,
    detect_plates_from_candidates,
    detect_plates_in_image,
)
from camera.reporting import write_detection_report_html, write_detection_report_md
from camera.config import Settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an HTML/Markdown report for plate detections."
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("PLATE_DATASET_DIR", "data/test_images/german_plates"),
        help="Root folder containing images (recursively).",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("PLATE_MODEL_PATH"),
        help="Path to plate-trained YOLO weights file (best.pt). Optional if dataset provides YOLO labels.",
    )
    parser.add_argument(
        "--out",
        default="reports/plate_report.html",
        help="Output report path (.html or .md).",
    )
    parser.add_argument(
        "--also-md",
        action="store_true",
        help="Also write a Markdown report next to the HTML report.",
    )
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ocr-min-conf", type=float, default=0.5)
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Do not embed crop previews in the report.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset dir not found: {dataset_dir}")

    settings = Settings.load_from_project_root()
    rec = settings.plate_recognition
    ocr = EasyOcrPlateRecognizer(
        languages=list(rec.languages),
        min_confidence=float(args.ocr_min_conf),
        preprocess=bool(rec.preprocess),
        allowlist=bool(rec.allowlist),
        allowlist_chars=str(rec.allowlist_chars),
        normalize=bool(rec.normalize),
    )

    pairs = _collect_images_with_optional_labels(dataset_dir)[: args.max_images]
    images = [p[0] for p in pairs]
    if not images:
        raise SystemExit(f"No images found under: {dataset_dir}")

    results = []
    include_crops = not args.no_crops
    detector = _build_detector_if_available(args.model_path, conf=args.conf)
    for img_path, label_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ts = datetime.now(UTC)
        if label_path is not None:
            candidates = _candidates_from_yolo_labels(label_path, img_w=img.shape[1], img_h=img.shape[0])
            res = detect_plates_from_candidates(
                image_bgr=img,
                candidates=candidates,
                ocr=ocr,
                captured_at=ts,
                image_path=img_path,
                include_crops=include_crops,
            )
        else:
            if detector is None:
                raise SystemExit(
                    "No YOLO labels found for images and no model was provided. "
                    "Set PLATE_MODEL_PATH or pass --model-path."
                )
            res = detect_plates_in_image(
                image_bgr=img,
                detector=detector,
                ocr=ocr,
                captured_at=ts,
                image_path=img_path,
                include_crops=include_crops,
            )
        results.append(res)

    out_path = Path(args.out)
    title = f"Plate detection report ({len(results)} images)"
    if out_path.suffix.lower() == ".md":
        write_detection_report_md(out_path=out_path, results=results, title=title)
    else:
        write_detection_report_html(out_path=out_path, results=results, title=title)
        if args.also_md:
            write_detection_report_md(
                out_path=out_path.with_suffix(".md"),
                results=results,
                title=title,
            )

    print(f"Wrote report: {out_path}")
    return 0


def _collect_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def _collect_images_with_optional_labels(root: Path) -> list[tuple[Path, Path | None]]:
    images = _collect_images(root)
    out: list[tuple[Path, Path | None]] = []
    for img in images:
        label = _find_yolo_label_for_image(img)
        out.append((img, label))
    # Prefer images that have labels (so it works out-of-the-box).
    out.sort(key=lambda p: p[1] is None)
    return out


def _find_yolo_label_for_image(img: Path) -> Path | None:
    # Roboflow layout: .../<split>/images/foo.jpg and .../<split>/labels/foo.txt
    if img.parent.name == "images":
        candidate = img.parent.parent / "labels" / f"{img.stem}.txt"
        if candidate.exists():
            return candidate
    # Fallback: look for sibling .txt
    candidate2 = img.with_suffix(".txt")
    return candidate2 if candidate2.exists() else None


def _build_detector_if_available(model_path: str | None, *, conf: float):
    if not model_path:
        return None
    weights = Path(model_path)
    if not weights.exists():
        return None
    return UltralyticsYoloPlateDetector(model_path=weights, confidence_threshold=conf)


def _candidates_from_yolo_labels(label_path: Path, *, img_w: int, img_h: int) -> list[PlateCandidate]:
    fixed: list[PlateCandidate] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        xc = float(x) * img_w
        yc = float(y) * img_h
        bw = float(w) * img_w
        bh = float(h) * img_h
        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)
        fixed.append(
            PlateCandidate(
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2).clamp(width=img_w, height=img_h),
                confidence=1.0,
                metadata={"class_id": int(float(cls)), "source": "yolo_label"},
            )
        )
    return fixed


if __name__ == "__main__":
    raise SystemExit(main())

