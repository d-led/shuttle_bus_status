from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from camera.plate_pipeline import (
    BBox,
    EasyOcrPlateRecognizer,
    PlateCandidate,
    detect_plates_from_candidates,
)


@pytest.mark.integration
def test_plate_ocr_success_rate_on_labeled_dataset_is_reasonable() -> None:
    """Dataset-only quality gate (out-of-the-box).

    Uses YOLO label boxes from the downloaded dataset, runs OCR on crops and asserts
    we get *some* readable text on a reasonable fraction of labeled plates.
    """
    dataset_root = Path("data/test_images/german_plates")
    if not dataset_root.exists():
        pytest.skip(f"Dataset not found: {dataset_root}")

    max_images = 30
    min_ocr_success = 0.2
    pairs = _find_yolo_image_label_pairs(dataset_root, limit=max_images)
    if not pairs:
        pytest.skip(f"No YOLO image/label pairs found under {dataset_root}")

    ocr = EasyOcrPlateRecognizer(languages=["de", "en"], min_confidence=0.3)

    total_plates = 0
    ok_plates = 0
    for image_path, label_path in pairs:
        image_bgr = _read_image_bgr(image_path)
        h, w = image_bgr.shape[:2]
        gt_boxes = _read_yolo_labels(label_path, image_w=w, image_h=h)
        if not gt_boxes:
            continue

        candidates = [
            PlateCandidate(bbox=b, confidence=1.0, metadata={"source": "yolo_label"})
            for b in gt_boxes
        ]
        res = detect_plates_from_candidates(
            image_bgr=image_bgr,
            candidates=candidates,
            ocr=ocr,
            include_crops=False,
        )

        total_plates += len(res.detections)
        ok_plates += sum(1 for d in res.detections if d.text)

    if total_plates == 0:
        pytest.skip("No labeled plates found.")

    rate = ok_plates / total_plates
    assert (
        rate >= min_ocr_success
    ), f"ocr_success_rate={rate:.3f} < {min_ocr_success:.3f}"


def _read_image_bgr(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _find_yolo_image_label_pairs(root: Path, *, limit: int) -> list[tuple[Path, Path]]:
    image_exts = {".jpg", ".jpeg", ".png"}
    pairs: list[tuple[Path, Path]] = []

    # Common Roboflow YOLO layout:
    # - train/images + train/labels, valid/images + valid/labels, test/images + test/labels
    for split in ("train", "valid", "test"):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        pairs.extend(
            _pairs_from_dirs(
                images_dir, labels_dir, image_exts, limit=limit - len(pairs)
            )
        )
        if len(pairs) >= limit:
            return pairs[:limit]

    # Fallback: search any *images*/*labels* folder pair.
    for images_dir in root.rglob("images"):
        labels_dir = images_dir.parent / "labels"
        pairs.extend(
            _pairs_from_dirs(
                images_dir, labels_dir, image_exts, limit=limit - len(pairs)
            )
        )
        if len(pairs) >= limit:
            return pairs[:limit]

    return pairs[:limit]


def _pairs_from_dirs(
    images_dir: Path, labels_dir: Path, image_exts: set[str], *, limit: int
) -> list[tuple[Path, Path]]:
    if limit <= 0:
        return []
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return []

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in image_exts:
            continue
        label = labels_dir / (img.stem + ".txt")
        if label.exists():
            pairs.append((img, label))
        if len(pairs) >= limit:
            break
    return pairs


def _read_yolo_labels(label_path: Path, *, image_w: int, image_h: int) -> list[BBox]:
    boxes: list[BBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _cls, x, y, w, h = parts
        xc = float(x) * image_w
        yc = float(y) * image_h
        bw = float(w) * image_w
        bh = float(h) * image_h
        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)
        boxes.append(
            BBox(x1=x1, y1=y1, x2=x2, y2=y2).clamp(width=image_w, height=image_h)
        )
    return boxes


#
# Note: we intentionally avoid model-based detection tests here to keep the suite
# runnable "out of the box" after dataset download (no weights/config needed).
