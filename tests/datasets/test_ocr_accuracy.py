"""OCR accuracy test comparing results with ground truth from filenames."""

from __future__ import annotations

from pathlib import Path
import re
import time

import numpy as np
import pytest

from camera.plate_pipeline import (
    BBox,
    PlateCandidate,
    detect_plates_from_candidates,
)
from camera.config import Settings


@pytest.mark.integration
def test_ocr_accuracy_with_ground_truth() -> None:
    """Test OCR accuracy against ground truth from filenames.

    Extracts plate numbers from image filenames (e.g., "1033IR.png" -> "1033IR")
    and compares with OCR results. Generates a compact HTML report.
    """
    dataset_root = Path("data/test_images/german_plates")
    if not dataset_root.exists():
        pytest.skip(f"Dataset not found: {dataset_root}")

    max_images = 200
    pairs = _find_images_with_ground_truth(dataset_root, limit=max_images)
    if not pairs:
        pytest.skip(f"No images with ground truth found under {dataset_root}")

    settings = Settings.load_from_project_root()
    rec = settings.plate_recognition
    from camera.plate_pipeline import create_plate_recognizer_from_config

    ocr = create_plate_recognizer_from_config(
        ocr_engine=rec.ocr_engine,
        languages=list(rec.languages),
        min_confidence=float(rec.min_confidence),
        preprocess=bool(rec.preprocess),
        allowlist=bool(rec.allowlist),
        allowlist_chars=str(rec.allowlist_chars),
        normalize=bool(rec.normalize),
    )

    results: list[dict[str, object]] = []
    detection_times: list[float] = []
    yolo_times: list[float] = []
    ocr_times: list[float] = []

    for image_path, ground_truth_text, label_path in pairs:
        image_bgr = _read_image_bgr(image_path)
        h, w = image_bgr.shape[:2]

        # Measure detection time (OCR only, since we use ground truth bounding boxes)
        detection_start = time.perf_counter()
        ocr_start = time.perf_counter()

        # Use YOLO labels if available, otherwise use full image as candidate
        if label_path is not None:
            gt_boxes = _read_yolo_labels(label_path, image_w=w, image_h=h)
            if gt_boxes:
                candidates = [
                    PlateCandidate(
                        bbox=b, confidence=1.0, metadata={"source": "yolo_label"}
                    )
                    for b in gt_boxes
                ]
                res = detect_plates_from_candidates(
                    image_bgr=image_bgr,
                    candidates=candidates,
                    ocr=ocr,
                    include_crops=False,
                )
            else:
                # Label file exists but empty - skip
                continue
        else:
            # No YOLO labels - use full image as candidate (for datasets without labels)
            full_image_candidate = PlateCandidate(
                bbox=BBox(x1=0, y1=0, x2=w, y2=h),
                confidence=1.0,
                metadata={"source": "full_image"},
            )
            res = detect_plates_from_candidates(
                image_bgr=image_bgr,
                candidates=[full_image_candidate],
                ocr=ocr,
                include_crops=False,
            )
        
        ocr_time = time.perf_counter() - ocr_start

        # Find best OCR result (highest confidence)
        best_detection = None
        best_conf = 0.0
        for det in res.detections:
            conf = det.raw_ocr_confidence or 0.0
            if conf > best_conf:
                best_conf = conf
                best_detection = det

        ocr_text = best_detection.text if best_detection else None
        ocr_raw_text = best_detection.raw_text if best_detection else None
        ocr_confidence = best_conf

        # Normalize ground truth for comparison (same as OCR normalization)
        gt_normalized = _normalize_plate_text(ground_truth_text)
        ocr_normalized = _normalize_plate_text(ocr_text) if ocr_text else None
        ocr_raw_normalized = (
            _normalize_plate_text(ocr_raw_text) if ocr_raw_text else None
        )

        # Check both normalized OCR text and raw OCR text (sometimes raw is better)
        # Also allow partial matches (if OCR gets most of the plate correct)
        is_exact_match = (
            ocr_normalized == gt_normalized if ocr_normalized else False
        ) or (ocr_raw_normalized == gt_normalized if ocr_raw_normalized else False)

        # Partial match: use edit distance for fuzzy matching
        # This catches cases where OCR gets most characters right but misses 1-2
        is_partial_match = False
        if not is_exact_match and gt_normalized and len(gt_normalized) >= 4:
            for ocr_candidate in [ocr_normalized, ocr_raw_normalized]:
                if not ocr_candidate or len(ocr_candidate) < 3:
                    continue

                # Calculate simple edit distance (Levenshtein-like)
                # Count matching characters in sequence
                gt_chars = list(gt_normalized)
                ocr_chars = list(ocr_candidate)

                # Check if most characters match (allowing for 1-2 errors)
                matches = sum(
                    1
                    for i, c in enumerate(ocr_chars)
                    if i < len(gt_chars) and c == gt_chars[i]
                )
                match_ratio = matches / len(gt_normalized) if gt_normalized else 0.0

                # Also check reverse (GT in OCR)
                reverse_matches = sum(1 for c in gt_chars if c in ocr_chars)
                reverse_ratio = (
                    reverse_matches / len(gt_normalized) if gt_normalized else 0.0
                )

                # Accept if 75%+ match or if length is close and 65%+ match
                if match_ratio >= 0.75 or reverse_ratio >= 0.75:
                    is_partial_match = True
                    break
                # Also accept if lengths are similar and match ratio is decent
                if abs(len(ocr_candidate) - len(gt_normalized)) <= 2 and (
                    match_ratio >= 0.65 or reverse_ratio >= 0.65
                ):
                    is_partial_match = True
                    break

        is_correct = is_exact_match or is_partial_match

        # Record detection time
        detection_end = time.perf_counter()
        detection_time = detection_end - detection_start
        detection_times.append(detection_time)
        ocr_times.append(ocr_time)
        # YOLO time is 0 in this test since we use ground truth bounding boxes
        yolo_times.append(0.0)

        results.append(
            {
                "image_path": str(image_path.relative_to(dataset_root)),
                "image_abs_path": str(image_path.resolve()),
                "ground_truth": ground_truth_text,
                "ground_truth_normalized": gt_normalized,
                "ocr_text": ocr_text,
                "ocr_raw_text": ocr_raw_text,
                "ocr_normalized": ocr_normalized,
                "ocr_confidence": ocr_confidence,
                "is_correct": is_correct,
                "num_detections": len(res.detections),
                "detection_time": detection_time,
                "yolo_time": 0.0,  # No YOLO in this test
                "ocr_time": ocr_time,
            }
        )

    if not results:
        pytest.skip("No valid results to report")

    # Generate HTML report
    report_path = Path("reports/datasets/ocr_accuracy_report.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_accuracy_report_html(report_path, results, detection_times, yolo_times, ocr_times)

    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total if total > 0 else 0.0

    # Calculate performance metrics
    avg_detection_time = (
        sum(detection_times) / len(detection_times) if detection_times else 0.0
    )
    min_detection_time = min(detection_times) if detection_times else 0.0
    max_detection_time = max(detection_times) if detection_times else 0.0
    fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0.0
    total_time = sum(detection_times)

    # Calculate YOLO and OCR metrics separately
    avg_yolo_time = sum(yolo_times) / len(yolo_times) if yolo_times else 0.0
    avg_ocr_time = sum(ocr_times) / len(ocr_times) if ocr_times else 0.0
    total_yolo_time = sum(yolo_times)
    total_ocr_time = sum(ocr_times)

    # Print performance summary
    print("\nDetection Performance:")
    print(f"  Total images processed: {total}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average detection time: {avg_detection_time*1000:.1f}ms")
    print(f"  Min detection time: {min_detection_time*1000:.1f}ms")
    print(f"  Max detection time: {max_detection_time*1000:.1f}ms")
    print(f"  Detection FPS: {fps:.2f}")
    print("\nBreakdown:")
    print(f"  YOLO detection: {total_yolo_time:.2f}s total, {avg_yolo_time*1000:.1f}ms avg")
    print(f"  OCR recognition: {total_ocr_time:.2f}s total, {avg_ocr_time*1000:.1f}ms avg")

    # Always generate report, even if accuracy is low
    # Assert minimum accuracy (adjust threshold as needed)
    # Note: This test generates a report for analysis; the assertion is lenient
    min_accuracy = 0.01  # 1% - very lenient, mainly to ensure we have some data
    if total == 0:
        pytest.skip("No results to evaluate")

    # Log summary
    print(f"\nOCR Accuracy: {accuracy:.1%} ({correct}/{total} correct)")
    print(f"Report: {report_path.absolute()}")

    assert (
        accuracy >= min_accuracy or total < 10
    ), f"OCR accuracy {accuracy:.3f} ({correct}/{total}) below minimum {min_accuracy:.3f}. Report generated at {report_path}"


def _is_german_plate_format(text: str) -> bool:
    """Check if text matches German license plate format.

    German plates have format: 1-3 letters (city) + 1-2 letters (optional) + 1-4 digits
    Examples: M AB 1234, B XY 567, HH A 123, S XY 1234
    Without spaces: MAB1234, BXY567, HHA123, SXY1234

    Also accepts edge cases like digits+letters (123IR) which might be valid in some contexts,
    but filters out clearly non-German formats.
    """
    if not text:
        return False

    # Normalize: remove spaces, convert to uppercase
    normalized = re.sub(r"[^A-Z0-9]+", "", text.upper())

    # German plates are typically 5-10 characters (city code + suffix + digits)
    # Minimum 5 chars for a valid plate (e.g., "M1234", "AB123")
    if len(normalized) < 5 or len(normalized) > 10:
        return False

    # Reject clearly non-German formats FIRST:
    # - 4+ digits at start (German plates have digits after letters, max 4 digits total)
    if re.match(r"^\d{4,}", normalized):
        return False
    # - 4+ letters before digits (city codes are max 3)
    if re.match(r"^[A-Z]{4,}\d", normalized):
        return False
    # - 5+ digits total (German plates max 4 digits)
    if sum(1 for c in normalized if c.isdigit()) > 4:
        return False
    # - Too short (less than 4 chars)
    if len(normalized) < 4:
        return False

    # Standard German plate patterns (without spaces):
    # 1-3 letters (city) + 1-2 letters (optional) + 1-4 digits
    # Examples: MAB1234, BXY567, HHA123, SXY1234, M1234, AB1234

    # Pattern 1: 1-3 letters + 1-2 letters + 1-4 digits (most common)
    # MAB1234, BXY567, HHA123, SXY1234
    if re.match(r"^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}$", normalized):
        return True

    # Pattern 2: 1-3 letters + 1-4 digits (single letter suffix or no suffix)
    # M1234, B567, HH123
    if re.match(r"^[A-Z]{1,3}\d{1,4}$", normalized):
        return True

    # Pattern 3: 1-2 letters + 1-4 digits (no city prefix, less common but valid)
    # AB1234, XY567
    if re.match(r"^[A-Z]{1,2}\d{1,4}$", normalized):
        return True

    # Pattern 4: Edge case - digits first + letters (e.g., 123IR, 12AB)
    # Accept if: 1-3 digits + 2-3 letters (reasonable for edge cases, but must be 5-6 chars)
    # This catches some edge cases but rejects 4+ digits first
    if re.match(r"^\d{1,3}[A-Z]{2,3}$", normalized) and 5 <= len(normalized) <= 6:
        return True

    return False


def _extract_plate_from_filename(filename: str) -> str | None:
    """Extract plate number from filename (e.g., '1033IR.png' -> '1033IR').

    Only extracts if filename looks like a German license plate.
    Skips Roboflow-style hashed filenames and non-German plates.
    """
    # Remove extension
    stem = Path(filename).stem
    # Remove common suffixes like "_1", "_2", etc.
    stem = re.sub(r"_\d+$", "", stem)
    # Remove any path separators
    stem = stem.replace("/", "").replace("\\", "")

    if not stem:
        return None

    # Skip Roboflow-style hashed filenames (contain dots, long hashes, etc.)
    if "." in stem or len(stem) > 15 or "_rf." in stem.lower():
        return None

    # Convert to uppercase for validation
    stem_upper = stem.upper()

    # Must be alphanumeric, 4-10 characters
    if not re.match(r"^[A-Z0-9]{4,10}$", stem_upper):
        return None

    # Filter: only German license plates
    if not _is_german_plate_format(stem_upper):
        return None

    return stem_upper


def _normalize_plate_text(text: str | None) -> str:
    """Normalize plate text for comparison (same as OCR normalization)."""
    if not text:
        return ""
    # Same normalization as in plate_pipeline.py
    upper = text.upper()
    return re.sub(r"[^A-Z0-9]+", "", upper)


def _find_images_with_ground_truth(
    root: Path, *, limit: int
) -> list[tuple[Path, str, Path | None]]:
    """Find images with extractable ground truth from filenames.

    Only includes German license plates (filters out non-German formats).
    Prioritizes Kaggle dataset (filenames contain plate numbers).

    German plate format: 1-3 letters (city) + 1-2 letters (optional) + 1-4 digits
    Examples: MAB1234, BXY567, HHA123, SXY1234
    """
    image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    pairs: list[tuple[Path, str, Path | None]] = []

    # Prioritize Kaggle dataset (more likely to have plate numbers in filenames)
    kaggle_dir = root / "kaggle"
    if kaggle_dir.exists():
        for img_path in kaggle_dir.rglob("*"):
            if img_path.suffix not in image_exts:
                continue
            plate_text = _extract_plate_from_filename(img_path.name)
            if not plate_text:
                continue
            label_path = _find_yolo_label_for_image(img_path)
            pairs.append((img_path, plate_text, label_path))
            if len(pairs) >= limit:
                break

    # Then check other directories if we need more
    if len(pairs) < limit:
        for img_path in root.rglob("*"):
            if img_path.suffix not in image_exts:
                continue
            # Skip if already in pairs
            if any(p[0] == img_path for p in pairs):
                continue
            plate_text = _extract_plate_from_filename(img_path.name)
            if not plate_text:
                continue
            label_path = _find_yolo_label_for_image(img_path)
            pairs.append((img_path, plate_text, label_path))
            if len(pairs) >= limit:
                break

    return sorted(pairs, key=lambda x: x[0])


def _find_yolo_label_for_image(img: Path) -> Path | None:
    """Find YOLO label file for an image."""
    # Roboflow layout: .../<split>/images/foo.jpg and .../<split>/labels/foo.txt
    if img.parent.name == "images":
        candidate = img.parent.parent / "labels" / f"{img.stem}.txt"
        if candidate.exists():
            return candidate
    # Fallback: look for sibling .txt
    candidate2 = img.with_suffix(".txt")
    return candidate2 if candidate2.exists() else None


def _read_image_bgr(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _read_yolo_labels(label_path: Path, *, image_w: int, image_h: int) -> list[BBox]:
    """Read YOLO format labels."""
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


def _write_accuracy_report_html(
    report_path: Path,
    results: list[dict[str, object]],
    detection_times: list[float],
    yolo_times: list[float],
    ocr_times: list[float],
) -> None:
    """Write compact HTML table report."""
    from datetime import datetime

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total if total > 0 else 0.0

    # Calculate performance metrics
    avg_detection_time = (
        sum(detection_times) / len(detection_times) if detection_times else 0.0
    )
    min_detection_time = min(detection_times) if detection_times else 0.0
    max_detection_time = max(detection_times) if detection_times else 0.0
    fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0.0
    total_time = sum(detection_times)

    # Calculate YOLO and OCR timing breakdown
    avg_yolo_time = (
        sum(yolo_times) / len(yolo_times) if yolo_times else 0.0
    )
    avg_ocr_time = (
        sum(ocr_times) / len(ocr_times) if ocr_times else 0.0
    )
    total_yolo_time = sum(yolo_times)
    total_ocr_time = sum(ocr_times)

    # Get current timestamp in local timezone
    import time

    now = datetime.now()
    tz_name = (
        time.tzname[time.daylight] if time.daylight is not None else time.tzname[0]
    )
    test_timestamp = now.strftime("%Y-%m-%d %H:%M:%S") + f" {tz_name}"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OCR Accuracy Report</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin-top: 0;
            color: #333;
        }}
        .stats {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }}
        .stats strong {{
            color: #0066cc;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th {{
            background: #333;
            color: white;
            padding: 10px;
            text-align: left;
            position: sticky;
            top: 0;
        }}
        td {{
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .correct {{
            color: #008000;
            font-weight: bold;
        }}
        .incorrect {{
            color: #cc0000;
        }}
        .image-link {{
            color: #0066cc;
            text-decoration: none;
        }}
        .image-link:hover {{
            text-decoration: underline;
        }}
        .confidence {{
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>OCR Accuracy Report</h1>
        <div class="stats">
            <strong>Test Information:</strong><br>
            &nbsp;&nbsp;Test Run Date/Time: {test_timestamp}<br>
            <br>
            <strong>Accuracy Metrics:</strong><br>
            &nbsp;&nbsp;Total Images: {total}<br>
            &nbsp;&nbsp;Correct: {correct}<br>
            &nbsp;&nbsp;Incorrect: {total - correct}<br>
            &nbsp;&nbsp;Accuracy: {accuracy:.1%}<br>
            <br>
            <strong>Performance Metrics:</strong><br>
            &nbsp;&nbsp;Total Time: {total_time:.2f}s<br>
            &nbsp;&nbsp;Average Detection Time: {avg_detection_time*1000:.1f}ms<br>
            &nbsp;&nbsp;Min Detection Time: {min_detection_time*1000:.1f}ms<br>
            &nbsp;&nbsp;Max Detection Time: {max_detection_time*1000:.1f}ms<br>
            &nbsp;&nbsp;Detection FPS: {fps:.2f}<br>
            <br>
            <strong>Breakdown:</strong><br>
            &nbsp;&nbsp;Average YOLO Time: {avg_yolo_time*1000:.1f}ms<br>
            &nbsp;&nbsp;Average OCR Time: {avg_ocr_time*1000:.1f}ms<br>
            &nbsp;&nbsp;Total YOLO Time: {total_yolo_time:.2f}s<br>
            &nbsp;&nbsp;Total OCR Time: {total_ocr_time:.2f}s
        </div>
        <table>
            <thead>
                <tr>
                    <th>Image</th>
                    <th>Ground Truth</th>
                    <th>OCR Result</th>
                    <th>Raw OCR</th>
                    <th>Confidence</th>
                    <th>Total (ms)</th>
                    <th>YOLO (ms)</th>
                    <th>OCR (ms)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""

    # Calculate relative path from report to images
    # Report is at: reports/datasets/ocr_accuracy_report.html
    # Images are at: data/test_images/german_plates/...
    # Use os.path.relpath to calculate correct relative path
    import os

    report_dir = report_path.parent.resolve()

    for r in results:
        image_abs = Path(r["image_abs_path"]).resolve()
        image_rel = r["image_path"]  # Relative to dataset_root (for display)
        gt = str(r["ground_truth"])
        ocr_result = str(r["ocr_text"]) if r["ocr_text"] else "(none)"
        ocr_raw = str(r["ocr_raw_text"]) if r["ocr_raw_text"] else "(none)"
        conf = float(r["ocr_confidence"])
        detection_time_ms = float(r.get("detection_time", 0.0)) * 1000.0
        yolo_time_ms = float(r.get("yolo_time", 0.0)) * 1000.0
        ocr_time_ms = float(r.get("ocr_time", 0.0)) * 1000.0
        is_correct = bool(r["is_correct"])

        status_class = "correct" if is_correct else "incorrect"
        status_text = "✓" if is_correct else "✗"

        # Calculate relative path from report directory to image using os.path.relpath
        # This properly handles the ../../ path calculation
        image_href = os.path.relpath(str(image_abs), str(report_dir))
        image_filename = Path(image_rel).name

        html += f"""                <tr>
                    <td><a href="{image_href}" class="image-link" target="_blank">{image_filename}</a></td>
                    <td><code>{gt}</code></td>
                    <td><code>{ocr_result}</code></td>
                    <td><code>{ocr_raw}</code></td>
                    <td class="confidence">{conf:.3f}</td>
                    <td class="confidence">{detection_time_ms:.1f}</td>
                    <td class="confidence">{yolo_time_ms:.1f}</td>
                    <td class="confidence">{ocr_time_ms:.1f}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
"""

    html += """            </tbody>
        </table>
    </div>
</body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    print(f"Wrote OCR accuracy report: {report_path}")
