"""Plate detection + OCR pipeline.

This module is intentionally:
- **Typed**
- **Testable** (pluggable detector / OCR engine)
- **Side-effect free** (no global model loading)
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class BBox:
    """Pixel-space bounding box (x1,y1,x2,y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, *, width: int, height: int) -> BBox:
        x1 = max(0, min(self.x1, width - 1))
        y1 = max(0, min(self.y1, height - 1))
        x2 = max(0, min(self.x2, width))
        y2 = max(0, min(self.y2, height))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass(frozen=True)
class PlateDetection:
    captured_at: datetime
    bbox: BBox
    detection_confidence: float
    text: str | None
    ocr_confidence: float | None
    raw_text: str | None
    raw_ocr_confidence: float | None
    reliability: float
    crop_jpeg_b64: str | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class ImagePlateDetections:
    captured_at: datetime
    image_size: tuple[int, int]  # (width, height)
    image_path: Path | None
    detections: list[PlateDetection]

    @property
    def plate_count(self) -> int:
        return len(self.detections)


@dataclass(frozen=True)
class PlateCandidate:
    bbox: BBox
    confidence: float
    metadata: dict[str, object]


class PlateDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[PlateCandidate]: ...


class PlateOcr(Protocol):
    def recognize(
        self, plate_bgr: np.ndarray
    ) -> tuple[str | None, float | None, dict[str, object]]: ...


class UltralyticsYoloPlateDetector:
    """YOLO plate detector powered by Ultralytics.

    Requires a *plate-trained* weights file.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        confidence_threshold: float = 0.5,
        class_ids: set[int] | None = None,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO  # heavy import, but explicit dependency at startup

        self._model = YOLO(str(model_path))
        self._confidence_threshold = float(confidence_threshold)
        self._class_ids = class_ids
        self._device = device

    def detect(self, image_bgr: np.ndarray) -> list[PlateCandidate]:
        # Ultralytics works with BGR numpy arrays.
        preds = self._model.predict(
            source=image_bgr,
            conf=self._confidence_threshold,
            verbose=False,
            device=self._device,
        )
        boxes = _first_prediction_boxes(preds)
        if boxes is None:
            return []

        arrays = _extract_box_arrays(boxes)
        if arrays is None:
            return []
        xyxy_np, conf_np, cls_np = arrays
        return _candidates_from_arrays(
            xyxy_np=xyxy_np,
            conf_np=conf_np,
            cls_np=cls_np,
            class_filter=self._class_ids,
        )


def _first_prediction_boxes(preds: object) -> object | None:
    if not isinstance(preds, list) or not preds:
        return None
    first = preds[0]
    return getattr(first, "boxes", None)


def _extract_box_arrays(boxes: object) -> tuple[object, object, object | None] | None:
    xyxy = getattr(boxes, "xyxy", None)
    conf = getattr(boxes, "conf", None)
    cls = getattr(boxes, "cls", None)
    if xyxy is None or conf is None:
        return None

    return (
        _to_numpy_maybe(xyxy),
        _to_numpy_maybe(conf),
        _to_numpy_maybe(cls) if cls is not None else None,
    )


def _to_numpy_maybe(value: object) -> object:
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.numpy() if hasattr(value, "numpy") else value


def _candidates_from_arrays(
    *,
    xyxy_np: object,
    conf_np: object,
    cls_np: object | None,
    class_filter: set[int] | None,
) -> list[PlateCandidate]:
    try:
        count = len(conf_np)  # type: ignore[arg-type]
    except Exception:
        return []

    candidates: list[PlateCandidate] = []
    for i in range(count):
        class_id = int(cls_np[i]) if cls_np is not None else -1  # type: ignore[index]
        if class_filter is not None and class_id not in class_filter:
            continue

        row = xyxy_np[i]  # type: ignore[index]
        # Ultralytics uses tensors/arrays that expose `tolist()`.
        values = row.tolist() if hasattr(row, "tolist") else list(row)
        x1, y1, x2, y2 = (int(v) for v in values)
        confidence = float(conf_np[i])  # type: ignore[index]
        candidates.append(
            PlateCandidate(
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=confidence,
                metadata={"class_id": class_id},
            )
        )
    return candidates


class EasyOcrPlateRecognizer:
    def __init__(
        self,
        *,
        languages: list[str],
        min_confidence: float = 0.5,
        model_storage_directory: Path | None = None,
        download_enabled: bool = True,
        preprocess: bool = False,
        allowlist: bool = False,
        allowlist_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        normalize: bool = False,
    ) -> None:
        import easyocr  # heavy import, but explicit dependency at startup

        storage_dir = model_storage_directory or _default_easyocr_storage_directory()
        storage_dir.mkdir(parents=True, exist_ok=True)

        self._reader = easyocr.Reader(
            languages,
            model_storage_directory=str(storage_dir),
            download_enabled=download_enabled,
            verbose=False,
        )
        self._min_confidence = float(min_confidence)
        self._preprocess = bool(preprocess)
        self._allowlist = bool(allowlist)
        self._allowlist_chars = str(allowlist_chars)
        self._normalize = bool(normalize)

    def recognize(
        self, plate_bgr: np.ndarray
    ) -> tuple[str | None, float | None, dict[str, object]]:
        plate_rgb = self._prepare_plate_rgb_for_ocr(plate_bgr)
        results = self._reader.readtext(plate_rgb, **self._easyocr_kwargs())
        best = _best_easyocr_candidate(results)
        if best is None:
            return (
                None,
                None,
                {"candidates": len(results), "raw_text": None, "raw_confidence": None},
            )

        raw_text, raw_conf = best
        normalized_text = (
            _normalize_plate_text(raw_text) if self._normalize else raw_text
        )
        meta: dict[str, object] = {
            "candidates": len(results),
            "raw_text": raw_text,
            "raw_confidence": raw_conf,
            "normalized_text": normalized_text if self._normalize else None,
            "preprocess": self._preprocess,
            "allowlist": self._allowlist,
        }

        if raw_conf < self._min_confidence:
            return None, raw_conf, meta

        return (normalized_text if self._normalize else raw_text), raw_conf, meta

    def _prepare_plate_rgb_for_ocr(self, plate_bgr: np.ndarray) -> np.ndarray:
        processed = (
            _preprocess_plate_for_ocr(plate_bgr) if self._preprocess else plate_bgr
        )
        return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    def _easyocr_kwargs(self) -> dict[str, object]:
        if not self._allowlist:
            return {}
        return {"allowlist": self._allowlist_chars}


def _best_easyocr_candidate(results: object) -> tuple[str, float] | None:
    if not isinstance(results, list) or not results:
        return None
    best_text: str | None = None
    best_conf: float | None = None
    for item in results:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text_str = str(item[1]).strip()
        if not text_str:
            continue
        conf_f = float(item[2])
        if best_conf is None or conf_f > best_conf:
            best_text = text_str
            best_conf = conf_f
    if best_text is None or best_conf is None:
        return None
    return best_text, best_conf


_PLATE_NORMALIZE_RE = re.compile(r"[^A-Z0-9]+")


def _normalize_plate_text(text: str) -> str:
    upper = text.upper()
    return _PLATE_NORMALIZE_RE.sub("", upper)


def _preprocess_plate_for_ocr(plate_bgr: np.ndarray) -> np.ndarray:
    if plate_bgr.size == 0:
        return plate_bgr

    h, w = plate_bgr.shape[:2]
    up = cv2.resize(
        plate_bgr,
        (max(1, int(w * 2.0)), max(1, int(h * 2.0))),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    sharp = cv2.filter2D(
        eq,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.int16),
    )
    thr = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


def detect_plates_in_image(
    *,
    image_bgr: np.ndarray,
    detector: PlateDetector,
    ocr: PlateOcr,
    captured_at: datetime | None = None,
    image_path: Path | None = None,
    include_crops: bool = False,
    crop_max_width: int = 320,
    crop_jpeg_quality: int = 75,
) -> ImagePlateDetections:
    """Detect plates and run OCR for each detected plate candidate."""
    ts = captured_at or datetime.now(UTC)
    if ts.tzinfo is None:
        # Enforce timezone-aware timestamps (privacy-safe + consistent logs).
        ts = ts.replace(tzinfo=UTC)

    candidates = detector.detect(image_bgr)
    return detect_plates_from_candidates(
        image_bgr=image_bgr,
        candidates=candidates,
        ocr=ocr,
        captured_at=ts,
        image_path=image_path,
        include_crops=include_crops,
        crop_max_width=crop_max_width,
        crop_jpeg_quality=crop_jpeg_quality,
    )


def detect_plates_from_candidates(
    *,
    image_bgr: np.ndarray,
    candidates: list[PlateCandidate],
    ocr: PlateOcr,
    captured_at: datetime | None = None,
    image_path: Path | None = None,
    include_crops: bool = False,
    crop_max_width: int = 320,
    crop_jpeg_quality: int = 75,
) -> ImagePlateDetections:
    """Run OCR (and optional crop previews) for provided plate candidates.

    This is useful for dataset evaluation where bounding boxes come from labels
    rather than a detector model.
    """
    ts = captured_at or datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)

    height, width = image_bgr.shape[:2]
    detections: list[PlateDetection] = []
    for cand in candidates:
        bbox = cand.bbox.clamp(width=width, height=height)
        crop = image_bgr[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        text, ocr_conf, ocr_meta = ocr.recognize(crop)
        raw_text_value = ocr_meta.get("raw_text")
        raw_text = raw_text_value if isinstance(raw_text_value, str) else text
        raw_confidence = ocr_meta.get("raw_confidence")
        raw_ocr_conf = (
            float(raw_confidence)
            if isinstance(raw_confidence, float | int)
            else ocr_conf
        )
        crop_b64 = (
            _crop_preview_jpeg_b64(
                crop,
                max_width=crop_max_width,
                quality=crop_jpeg_quality,
            )
            if include_crops
            else None
        )

        reliability = _combine_reliability(
            detection_confidence=cand.confidence,
            ocr_confidence=ocr_conf,
        )
        detections.append(
            PlateDetection(
                captured_at=ts,
                bbox=bbox,
                detection_confidence=cand.confidence,
                text=text,
                ocr_confidence=ocr_conf,
                raw_text=raw_text,
                raw_ocr_confidence=raw_ocr_conf,
                reliability=reliability,
                crop_jpeg_b64=crop_b64,
                metadata={
                    "detector": cand.metadata,
                    "ocr": ocr_meta,
                    "image_path": str(image_path) if image_path is not None else None,
                },
            )
        )

    return ImagePlateDetections(
        captured_at=ts,
        image_size=(width, height),
        image_path=image_path,
        detections=detections,
    )


def _combine_reliability(
    *, detection_confidence: float, ocr_confidence: float | None
) -> float:
    """Combine detection + OCR confidence into a single metric.

    We keep it intentionally simple and monotonic.
    """
    det = float(detection_confidence)
    if ocr_confidence is None:
        return det * 0.5
    return det * float(ocr_confidence)


def _crop_preview_jpeg_b64(
    crop_bgr: np.ndarray, *, max_width: int, quality: int
) -> str | None:
    if crop_bgr.size == 0:
        return None

    preview = crop_bgr
    if max_width > 0:
        h, w = preview.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            new_w = max_width
            new_h = max(1, int(h * scale))
            preview = cv2.resize(preview, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(
        ".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _default_easyocr_storage_directory() -> Path:
    # Keep OCR model/cache data inside the repo so it works on systems where $HOME
    # isn't writable (e.g. restricted environments).
    return _find_project_root() / ".cache" / "easyocr"


def _find_project_root() -> Path:
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "config.toml").exists():
            return parent
    return current
