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
from typing import Any, Protocol, cast

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

        # EasyOCR initialization can take 30-60 seconds on first run (downloads models)
        import logging
        import time as time_module

        logger = logging.getLogger(__name__)
        logger.info(
            "Initializing EasyOCR (this may take 30-60s on first run to download models)..."
        )
        init_start = time_module.perf_counter()

        self._reader = easyocr.Reader(
            languages,
            model_storage_directory=str(storage_dir),
            download_enabled=download_enabled,
            verbose=False,
        )

        init_time = time_module.perf_counter() - init_start
        logger.info("EasyOCR initialized in %.2fs", init_time)
        self._min_confidence = float(min_confidence)
        self._preprocess = bool(preprocess)
        self._allowlist = bool(allowlist)
        self._allowlist_chars = str(allowlist_chars)
        self._normalize = bool(normalize)

    def recognize(
        self, plate_bgr: np.ndarray
    ) -> tuple[str | None, float | None, dict[str, object]]:
        # Try multiple preprocessing strategies and pick the best result
        if self._preprocess:
            return self._recognize_with_multiple_preprocessing(plate_bgr)

        # Single strategy (original or basic preprocessing)
        plate_rgb = self._prepare_plate_rgb_for_ocr(plate_bgr)
        return self._recognize_single(plate_rgb, strategy_name="original")

    def _recognize_with_multiple_preprocessing(
        self, plate_bgr: np.ndarray
    ) -> tuple[str | None, float | None, dict[str, object]]:
        """Try multiple preprocessing strategies and return best result."""
        h, w = plate_bgr.shape[:2]
        is_very_small = h < 30 or w < 100
        is_small = h < 50 or w < 150
        strategies_to_try = _select_preprocessing_strategies(
            plate_bgr, is_very_small=is_very_small, is_small=is_small
        )

        all_results: list[tuple[str | None, float | None, dict[str, object]]] = []
        fast_strategies = ["original", "upscale_clahe", "remove_vignette"]

        for strategy_name, processed in strategies_to_try:
            plate_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            text, conf, meta = self._recognize_single(
                plate_rgb, strategy_name=strategy_name
            )
            if text is not None:
                all_results.append((text, conf, meta))
                if _should_early_exit_preprocessing(
                    is_very_small,
                    strategy_name,
                    fast_strategies,
                    conf,
                    text,
                    len(strategies_to_try) - len(all_results),
                ):
                    break

        if not all_results:
            return (
                None,
                None,
                {
                    "preprocessing_strategies_tried": len(strategies_to_try),
                    "all_failed": True,
                },
            )

        best_result = max(
            all_results,
            key=lambda x: _score_preprocessing_result(x, is_very_small),
        )
        best_text, best_conf, best_meta = best_result
        all_strategies = _preprocess_plate_strategies(plate_bgr)
        best_meta["preprocessing_strategies_tried"] = len(strategies_to_try)
        best_meta["preprocessing_strategies_available"] = len(all_strategies)
        best_meta["preprocessing_strategy_used"] = best_meta.get(
            "strategy_name", "unknown"
        )
        best_meta["all_strategy_results"] = [
            {"strategy": m.get("strategy_name", "unknown"), "text": t, "conf": c}
            for t, c, m in all_results
        ]
        return best_text, best_conf, best_meta

    def _recognize_single(
        self, plate_rgb: np.ndarray, strategy_name: str = "unknown"
    ) -> tuple[str | None, float | None, dict[str, object]]:
        """Recognize text from a single preprocessed image."""
        # Always upscale very small images for better OCR (even if not preprocessing)
        h, w = plate_rgb.shape[:2]
        if w < 100 or h < 30:
            # Upscale small crops significantly
            scale = max(
                3.0, 200.0 / w, 60.0 / h
            )  # Target at least 200px wide or 60px tall
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            plate_rgb = cv2.resize(
                plate_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC
            )

        # EasyOCR parameters optimized for license plates
        kwargs = self._easyocr_kwargs()
        kwargs["paragraph"] = False  # Treat as single line
        kwargs["width_ths"] = (
            0.05  # Very low threshold for wide plates (handles spaces)
        )
        kwargs["height_ths"] = 0.05  # Very low threshold for height
        kwargs["detail"] = 1  # Get detailed results
        # More aggressive text detection for plates with spaces/separators
        kwargs["slope_ths"] = 0.1  # Allow more slanted text
        kwargs["ycenter_ths"] = 0.7  # More permissive vertical alignment (was 0.5)
        # Additional parameters to improve detection of spaced text
        kwargs["mag_ratio"] = 1.5  # Magnification ratio for better small text detection

        results = self._reader.readtext(plate_rgb, **kwargs)
        best = _best_easyocr_candidate(results)
        if best is None:
            return (
                None,
                None,
                {
                    "candidates": len(results),
                    "raw_text": None,
                    "raw_confidence": None,
                    "strategy_name": strategy_name,
                },
            )

        raw_text, raw_conf = best

        # Remove vignette characters (spurious single chars from circular seals)
        text_no_vignette = _remove_vignette_characters(raw_text)

        # Apply position-based character correction
        corrected_text = _correct_plate_characters_by_position(text_no_vignette)

        # Normalize if enabled
        normalized_text = (
            _normalize_plate_text(corrected_text) if self._normalize else corrected_text
        )

        meta: dict[str, object] = {
            "candidates": len(results),
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "raw_confidence": raw_conf,
            "normalized_text": normalized_text if self._normalize else None,
            "preprocess": self._preprocess,
            "allowlist": self._allowlist,
            "strategy_name": strategy_name,
        }

        if raw_conf < self._min_confidence:
            return None, raw_conf, meta

        return (normalized_text if self._normalize else corrected_text), raw_conf, meta

    def _prepare_plate_rgb_for_ocr(self, plate_bgr: np.ndarray) -> np.ndarray:
        """Prepare plate image for OCR (legacy single-strategy path)."""
        # When preprocessing is enabled, use multiple strategies (handled in recognize())
        # This path is for when preprocessing is disabled
        return cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)

    def _easyocr_kwargs(self) -> dict[str, object]:
        if not self._allowlist:
            return {}
        return {"allowlist": self._allowlist_chars}


def _easyocr_bbox_center(bbox: object) -> tuple[float, float]:
    """Extract (y_center, x_center) from bbox [[x,y], ...] or return (0, 0)."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0.0, 0.0
    if not all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in bbox):
        return 0.0, 0.0
    y_coords = [pt[1] for pt in bbox]
    x_coords = [pt[0] for pt in bbox]
    return sum(y_coords) / len(y_coords), sum(x_coords) / len(x_coords)


def _easyocr_valid_results(
    results: list[object],
) -> list[tuple[str, float, float, float]]:
    """Parse EasyOCR result list into (text, conf, y_center, x_center) tuples."""
    out: list[tuple[str, float, float, float]] = []
    for item in results:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text_str = str(item[1]).strip()
        if not text_str:
            continue
        conf_f = float(item[2])
        bbox = item[0] if len(item) > 0 else None
        y_center, x_center = _easyocr_bbox_center(bbox)
        out.append((text_str, conf_f, y_center, x_center))
    return out


def _easyocr_y_threshold(valid_results: list[tuple[str, float, float, float]]) -> float:
    """Compute y-threshold for grouping same-line detections (10-30px, or 10% of span)."""
    if not valid_results:
        return 20.0
    max_y = max(r[2] for r in valid_results)
    min_y = min(r[2] for r in valid_results)
    estimated_height = max_y - min_y if max_y > min_y else 100.0
    return max(10.0, min(30.0, estimated_height * 0.1))


def _easyocr_find_group_index(
    groups: list[list[tuple[str, float, float, float]]],
    y: float,
    y_threshold: float,
) -> int | None:
    """Index of group with similar y, or None."""
    for idx, group in enumerate(groups):
        if group and abs(group[0][2] - y) < y_threshold:
            return idx
    return None


def _easyocr_group_by_line(
    valid_results: list[tuple[str, float, float, float]],
    y_threshold: float,
) -> list[list[tuple[str, float, float, float]]]:
    """Group (text, conf, y, x) by similar y (same line)."""
    groups: list[list[tuple[str, float, float, float]]] = []
    for result in valid_results:
        _, _, y, _ = result
        idx = _easyocr_find_group_index(groups, y, y_threshold)
        if idx is not None:
            groups[idx].append(result)
        else:
            groups.append([result])
    return groups


def _easyocr_combine_groups(
    groups: list[list[tuple[str, float, float, float]]],
) -> list[tuple[str, float]]:
    """Sort each group by x and combine text; return (combined_text, avg_conf)."""
    combined: list[tuple[str, float]] = []
    for group in groups:
        group.sort(key=lambda r: r[3])
        combined.append(
            (" ".join(r[0] for r in group), sum(r[1] for r in group) / len(group))
        )
    return combined


def _easyocr_single_result(item: object) -> tuple[str, float] | None:
    """Parse single EasyOCR result item. Returns (text, conf) or None."""
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return None
    text_str = str(item[1]).strip()
    return (text_str, float(item[2])) if text_str else None


def _easyocr_best_single(
    valid_results: list[tuple[str, float, float, float]],
) -> tuple[str, float] | None:
    """Return (text, conf) with highest conf from valid_results, or None."""
    if not valid_results:
        return None
    best = max(valid_results, key=lambda r: r[1])
    return (best[0], best[1])


def _best_easyocr_candidate(results: object) -> tuple[str, float] | None:
    """Select best OCR result, combining multiple detections if on same line."""
    if not isinstance(results, list) or not results:
        return None
    if len(results) == 1:
        return _easyocr_single_result(results[0])
    valid_results = _easyocr_valid_results(results)
    if not valid_results:
        return None
    y_threshold = _easyocr_y_threshold(valid_results)
    groups = _easyocr_group_by_line(valid_results, y_threshold)
    combined_results = _easyocr_combine_groups(groups)
    if combined_results:
        return max(combined_results, key=lambda x: x[1])
    return _easyocr_best_single(valid_results)


_PLATE_NORMALIZE_RE = re.compile(r"[^A-Z0-9]+")

# Character confusion mapping for position-based correction
_CHAR_TO_INT = {
    "O": "0",  # O in digit position is likely 0
    "I": "1",  # I in digit position is likely 1
    "J": "3",  # J in digit position is likely 3
    "A": "4",  # A in digit position is likely 4
    "G": "6",  # G in digit position is likely 6
    "S": "5",  # S in digit position is likely 5
}

_INT_TO_CHAR = {
    "0": "O",  # 0 in letter position is likely O
    "1": "I",  # 1 in letter position is likely I
    "3": "J",  # 3 in letter position is likely J
    "4": "A",  # 4 in letter position is likely A
    "6": "G",  # 6 in letter position is likely G
    "5": "S",  # 5 in letter position is likely S
}


def _normalize_plate_text(text: str) -> str:
    """Normalize plate text: uppercase, remove non-alphanumeric."""
    upper = text.upper()
    return _PLATE_NORMALIZE_RE.sub("", upper)


# Characters commonly misread from vignettes (circular/oval shapes on German plates)
_VIGNETTE_CHARS = frozenset(
    {"E", "B", "O", "0", "8", "6", "G", "Q", "D", "P", "I", "1"}
)


def _vignette_parts_step(parts: list[str], i: int, result_parts: list[str]) -> int:
    """Process one part; merge vignette or append. Returns next index."""
    if (
        i > 0
        and i < len(parts) - 1
        and len(parts[i]) == 1
        and parts[i].upper() in _VIGNETTE_CHARS
        and len(parts[i - 1]) >= 2
        and len(parts[i + 1]) >= 2
    ):
        result_parts[-1] = result_parts[-1] + parts[i + 1]
        return i + 2
    if (
        i > 0
        and len(parts[i]) >= 3
        and parts[i][0].upper() in _VIGNETTE_CHARS
        and len(parts[i - 1]) >= 2
    ):
        result_parts[-1] = result_parts[-1] + parts[i][1:]
        return i + 1
    result_parts.append(parts[i])
    return i + 1


def _remove_vignette_from_parts(parts: list[str]) -> str:
    """Merge or strip vignette chars from space-separated parts."""
    result_parts: list[str] = []
    i = 0
    while i < len(parts):
        i = _vignette_parts_step(parts, i, result_parts)
    return " ".join(result_parts)


def _vignette_regex_replace(match: re.Match[str]) -> str:
    """Regex replacer: drop single vignette char between two groups if safe."""
    prefix = match.group(1)
    single_char = match.group(2)
    suffix = match.group(3)
    if single_char.upper() not in _VIGNETTE_CHARS:
        return match.group(0)
    char_pos = len(prefix)
    if char_pos <= 2:
        if len(prefix) >= 3:
            return match.group(0)
        if suffix and suffix[0].isalpha():
            return match.group(0)
    return prefix + suffix


def _remove_vignette_characters(text: str) -> str:
    """Remove spurious single characters that are likely vignettes (circular seals).

    German license plates have vignettes between letter groups that can be
    misread as single characters like 'E' or 'B'. Removes isolated single
    characters in the middle that are likely vignettes.
    """
    if not text or len(text) < 4:
        return text

    parts = text.split()
    if len(parts) >= 2:
        text = _remove_vignette_from_parts(parts)

    text_clean = text.replace(" ", "").replace(".", "").replace("-", "")
    if len(text_clean) < 4:
        return text_clean

    vignette_pattern = "".join(sorted(_VIGNETTE_CHARS))
    result = re.sub(
        f"([A-Z0-9]{{2,}})([{vignette_pattern}])([A-Z0-9]{{2,}})",
        _vignette_regex_replace,
        text_clean,
        flags=re.IGNORECASE,
    )
    if result != text_clean and len(result) >= 3:
        return result
    return text_clean


def _parse_paddle_line(line: object) -> tuple[str | None, float]:
    """Parse one PaddleOCR result line. Returns (text, conf) or (None, 0.0)."""
    if not isinstance(line, (list, tuple)) or len(line) < 2:
        return None, 0.0
    if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
        text_val, conf_val = line[1][0], line[1][1]
        text = str(text_val).strip() if text_val else None
        try:
            return text, float(conf_val)
        except (ValueError, TypeError):
            return text, 0.0
    if len(line) >= 3:
        text_val = line[1]
        conf_val = line[2] if len(line) > 2 else 0.0
        text = str(text_val).strip() if text_val else None
        try:
            return text, float(conf_val)
        except (ValueError, TypeError):
            return text, 0.0
    return None, 0.0


def _best_paddle_line(lines: list[object]) -> tuple[str | None, float]:
    """Return (best_text, best_conf) from PaddleOCR lines, or (None, 0.0)."""
    best_text: str | None = None
    best_conf: float = 0.0
    for line in lines:
        text, conf = _parse_paddle_line(line)
        if text and conf > best_conf:
            best_text = text
            best_conf = conf
    return best_text, best_conf


def _tesseract_parse_image_to_data(
    data: dict[str, object],
) -> tuple[str, float] | None:
    """Parse Tesseract image_to_data dict. Returns (combined_text, avg_conf) or None."""
    texts: list[str] = []
    confidences: list[float] = []
    text_list = data.get("text", [])
    conf_list = data.get("conf", [0])
    if not isinstance(text_list, list) or not isinstance(conf_list, list):
        return None
    for i in range(len(text_list)):
        text = text_list[i] if i < len(text_list) else ""
        text = str(text).strip() if text else ""
        if not text:
            continue
        conf_val = int(conf_list[i]) if i < len(conf_list) else 0
        if conf_val > 0:
            texts.append(text)
            confidences.append(float(conf_val) / 100.0)
    if not texts:
        return None
    combined = " ".join(texts).strip()
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
    return combined, avg_conf


def _tesseract_try_psm(
    pt: Any,
    plate_rgb: np.ndarray,
    lang: str,
    psm: str,
    allowlist: bool,
    allowlist_chars: str,
) -> tuple[str, float] | None:
    """Run Tesseract with one PSM; return (combined_text, avg_conf) or None."""
    psm_config = f"--psm {psm}"
    if allowlist:
        psm_config += f" -c tessedit_char_whitelist={allowlist_chars}"
    try:
        data = pt.image_to_data(
            plate_rgb, lang=lang, config=psm_config, output_type=pt.Output.DICT
        )
        return _tesseract_parse_image_to_data(data)
    except Exception:
        return None


def _tesseract_fallback_string(
    pt: Any, plate_rgb: np.ndarray, lang: str, config_str: str
) -> tuple[str | None, float]:
    """Fallback: image_to_string. Returns (text, 0.5) or (None, 0.0)."""
    try:
        simple_text = pt.image_to_string(
            plate_rgb, lang=lang, config=config_str
        ).strip()
        return (simple_text, 0.5) if simple_text else (None, 0.0)
    except Exception:
        return None, 0.0


def _tesseract_best_result(
    pytesseract: object,
    plate_rgb: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[str | None, float]:
    """Try PSM modes then image_to_string; return (best_text, best_conf)."""
    pt: Any = pytesseract
    lang = cfg.get("lang", "")
    allowlist = bool(cfg.get("allowlist", False))
    allowlist_chars = str(cfg.get("allowlist_chars", ""))
    config_str = str(cfg.get("config", "--psm 7"))
    best_text: str | None = None
    best_conf: float = 0.0
    for psm in ["7", "11", "6"]:
        parsed = _tesseract_try_psm(
            pt, plate_rgb, lang, psm, allowlist, allowlist_chars
        )
        if parsed and parsed[1] > best_conf:
            best_text, best_conf = parsed[0], parsed[1]
    if not best_text:
        best_text, best_conf = _tesseract_fallback_string(
            pt, plate_rgb, lang, config_str
        )
    return best_text, best_conf


def _dots_ocr_extract_text(result: object) -> str:
    """Extract text from dots.ocr result (markdown or json layout fallback)."""
    text = result.markdown.strip() if hasattr(result, "markdown") else ""
    if text:
        return text
    if not hasattr(result, "json"):
        return ""
    import json

    raw = getattr(result, "json", None)
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict):
        return ""
    texts = []
    for elem in data.get("layout", []):
        if (
            isinstance(elem, dict)
            and elem.get("category") == "Text"
            and elem.get("text")
        ):
            texts.append(elem["text"])
    return " ".join(texts).strip()


def _plate_ocr_post_process(
    best_text: str,
    best_conf: float,
    candidates: int,
    engine: str,
    *,
    allowlist: bool,
    allowlist_chars: str,
    normalize: bool,
    min_confidence: float,
    preprocess: bool,
) -> tuple[str | None, float | None, dict[str, object]]:
    """Apply vignette removal, correction, normalization and build meta. Shared by OCR recognizers."""
    if allowlist:
        best_text = "".join(c for c in best_text if c.upper() in allowlist_chars)
    text_no_vignette = _remove_vignette_characters(best_text)
    corrected_text = _correct_plate_characters_by_position(text_no_vignette)
    normalized_text = (
        _normalize_plate_text(corrected_text) if normalize else corrected_text
    )
    meta: dict[str, object] = {
        "candidates": candidates,
        "raw_text": best_text,
        "corrected_text": corrected_text,
        "raw_confidence": best_conf,
        "normalized_text": normalized_text if normalize else None,
        "preprocess": preprocess,
        "allowlist": allowlist,
        "engine": engine,
    }
    if best_conf < min_confidence:
        return None, best_conf, meta
    return (normalized_text if normalize else corrected_text), best_conf, meta


def _digit_section_start(text: str) -> int:
    """Index where digit section starts (first of two consecutive digits), or len(text)."""
    for i in range(len(text) - 1):
        if text[i].isdigit() and text[i + 1].isdigit():
            return i
    return len(text)


def _correct_char_for_position(char: str, i: int, digit_start: int) -> str:
    """Correct one character by position (letter vs digit section)."""
    if i < digit_start:
        return _INT_TO_CHAR[char] if char.isdigit() and char in _INT_TO_CHAR else char
    return _CHAR_TO_INT[char] if char.isalpha() and char in _CHAR_TO_INT else char


def _correct_plate_characters_by_position(text: str) -> str:
    """Apply position-based character correction for German plates."""
    if not text or len(text) < 3:
        return text
    digit_start = _digit_section_start(text)
    corrected = [
        _correct_char_for_position(char, i, digit_start) for i, char in enumerate(text)
    ]
    return "".join(corrected)


def _preprocess_plate_for_ocr(plate_bgr: np.ndarray) -> np.ndarray:
    """Basic preprocessing pipeline (legacy - kept for backward compatibility)."""
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


def _preprocess_plate_strategy_original(plate_bgr: np.ndarray) -> np.ndarray:
    """Strategy 1: Original image (no preprocessing)."""
    return plate_bgr


def _preprocess_plate_strategy_upscale_clahe_threshold(
    plate_bgr: np.ndarray,
) -> np.ndarray:
    """Strategy 2: Upscale + CLAHE + Adaptive threshold (current default)."""
    if plate_bgr.size == 0:
        return plate_bgr

    h, w = plate_bgr.shape[:2]
    # More aggressive upscaling for small images
    scale = max(3.0, 300.0 / w, 80.0 / h)  # Target at least 300px wide or 80px tall
    up = cv2.resize(
        plate_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
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


def _preprocess_plate_strategy_to_zero_180(plate_bgr: np.ndarray) -> np.ndarray:
    """Strategy 3: 'To zero' thresholding with threshold=180 (best in Nature paper)."""
    if plate_bgr.size == 0:
        return plate_bgr

    h, w = plate_bgr.shape[:2]
    # More aggressive upscaling for small images
    scale = max(3.0, 300.0 / w, 80.0 / h)  # Target at least 300px wide or 80px tall
    up = cv2.resize(
        plate_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    # "To zero" thresholding (threshold=180 as per Nature paper)
    _, thr = cv2.threshold(eq, 180, 255, cv2.THRESH_TOZERO)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


def _preprocess_plate_strategy_morphological(plate_bgr: np.ndarray) -> np.ndarray:
    """Strategy 4: Thresholding + morphological opening (noise reduction)."""
    if plate_bgr.size == 0:
        return plate_bgr

    h, w = plate_bgr.shape[:2]
    # More aggressive upscaling for small images
    scale = max(3.0, 300.0 / w, 80.0 / h)  # Target at least 300px wide or 80px tall
    up = cv2.resize(
        plate_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    # Adaptive threshold
    thr = cv2.adaptiveThreshold(
        eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)


def _preprocess_plate_strategy_otsu(plate_bgr: np.ndarray) -> np.ndarray:
    """Strategy 5: Otsu's binary thresholding."""
    if plate_bgr.size == 0:
        return plate_bgr

    h, w = plate_bgr.shape[:2]
    # More aggressive upscaling for small images
    scale = max(3.0, 300.0 / w, 80.0 / h)  # Target at least 300px wide or 80px tall
    up = cv2.resize(
        plate_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    # Otsu's method
    _, thr = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


def _flatten_strategy_upscale_and_flatten(
    plate_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Upscale, denoise, flatten background, CLAHE. Returns (eq_after_clahe, up)."""
    h, w = plate_bgr.shape[:2]
    scale = max(6.0, 500.0 / w, 150.0 / h)
    up = cv2.resize(
        plate_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_LANCZOS4,
    )
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(
        gray, h=10, templateWindowSize=7, searchWindowSize=21
    )
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bg_morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel_bg)
    bg_morph = cv2.GaussianBlur(bg_morph, (31, 31), 0)
    bg_blur = cv2.GaussianBlur(denoised, (51, 51), 0)
    background = cv2.addWeighted(bg_morph, 0.5, bg_blur, 0.5, 0)
    flattened = cv2.subtract(denoised, background)
    flattened = cv2.normalize(flattened, None, 0.0, 255.0, cv2.NORM_MINMAX)  # type: ignore[call-overload]
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    eq = clahe.apply(flattened)
    return eq, up


def _flatten_strategy_sharpen_threshold(eq: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Sharpen, adaptive threshold, morphological cleanup; return BGR."""
    kernel_sharpen = (
        np.array(
            [
                [-1, -1, -1, -1, -1],
                [-1, 2, 2, 2, -1],
                [-1, 2, 8, 2, -1],
                [-1, 2, 2, 2, -1],
                [-1, -1, -1, -1, -1],
            ],
            dtype=np.float32,
        )
        / 8.0
    )
    sharp = cv2.filter2D(eq, -1, kernel_sharpen)
    block_size = max(11, int(up.shape[1] / 8))
    if block_size % 2 == 0:
        block_size += 1
    thr = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        10,
    )
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel_clean)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def _preprocess_plate_strategy_flatten(plate_bgr: np.ndarray) -> np.ndarray:
    """Aggressive upscaling + background flattening for very small crops."""
    if plate_bgr.size == 0:
        return plate_bgr
    eq, up = _flatten_strategy_upscale_and_flatten(plate_bgr)
    return _flatten_strategy_sharpen_threshold(eq, up)


def _preprocess_plate_strategy_smooth_sharpen(plate_bgr: np.ndarray) -> np.ndarray:
    """Strategy 7: Smoothing + sharpening for better edge contrast.

    This strategy focuses on:
    1. Aggressive upscaling for small crops
    2. Edge-preserving smoothing to reduce noise while keeping edges
    3. Unsharp masking for controlled sharpening
    4. Contrast enhancement with CLAHE
    5. Adaptive thresholding optimized for edge clarity

    Unlike flattening (which subtracts background), this preserves the full
    image structure while enhancing edges through smoothing + sharpening.
    """
    if plate_bgr.size == 0:
        return plate_bgr

    h, w = plate_bgr.shape[:2]

    # Aggressive upscaling for tiny crops
    scale = max(6.0, 500.0 / w, 150.0 / h)
    up = cv2.resize(
        plate_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_LANCZOS4,  # Best quality for upscaling
    )

    # Convert to grayscale
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    # Step 1: Initial denoising (light, preserves edges)
    denoised = cv2.fastNlMeansDenoising(
        gray, h=7, templateWindowSize=5, searchWindowSize=15
    )

    # Step 2: Edge-preserving smoothing with bilateral filter
    # Parameters tuned for license plates: preserve edges, smooth background
    smoothed = cv2.bilateralFilter(denoised, d=5, sigmaColor=50, sigmaSpace=50)

    # Step 3: Unsharp masking for controlled sharpening
    # Create a slightly blurred version
    blurred = cv2.GaussianBlur(smoothed, (3, 3), 1.0)
    # Unsharp mask: original + (original - blurred) * amount
    unsharp = cv2.addWeighted(smoothed, 1.0 + 0.8, blurred, -0.8, 0)

    # Step 4: Additional light sharpening with kernel (more subtle)
    kernel_sharpen = np.array(
        [[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32
    )
    sharpened = cv2.filter2D(unsharp, -1, kernel_sharpen)

    # Step 5: Contrast enhancement with CLAHE (moderate, not too aggressive)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)

    # Step 6: Normalize to full dynamic range
    normalized = cv2.normalize(enhanced, None, 0.0, 255.0, cv2.NORM_MINMAX)  # type: ignore[call-overload]

    # Step 7: Adaptive thresholding optimized for edge clarity
    # Block size based on image size (smaller for better edge detection)
    block_size = max(11, int(up.shape[1] / 10))
    if block_size % 2 == 0:
        block_size += 1
    thr = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        8,  # Moderate C value for balanced contrast
    )

    # Very light morphological cleanup (minimal, preserve edges)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel_clean, iterations=1)

    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def _vignette_strategy_build_mask(
    gray: np.ndarray, contours: Any, h_img: int, w_img: int
) -> np.ndarray:
    """Build mask (255=keep, 0=replace) for circular/oval vignette regions."""
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    center_y, center_x = h_img // 2, w_img // 2
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50 or area > (h_img * w_img * 0.3):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        x_center_dist = abs(cx - center_x) / w_img
        y_center_dist = abs(cy - center_y) / h_img
        if x_center_dist < 0.3 and y_center_dist < 0.4:
            cv2.drawContours(mask, [contour], -1, (0.0,), -1)
    return mask


def _vignette_strategy_apply_mask_and_postprocess(
    gray: np.ndarray, mask: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """Replace masked regions with background, then CLAHE + sharpen + threshold."""
    edge_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    background_color = int(np.median(edge_pixels))
    gray_masked = gray.copy()
    gray_masked[mask == 0] = background_color
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_masked)
    kernel_sharpen = np.array(
        [[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32
    )
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    block_size = max(11, int(up.shape[1] / 10))
    if block_size % 2 == 0:
        block_size += 1
    thr = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        8,
    )
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


def _preprocess_plate_strategy_remove_vignette(plate_bgr: np.ndarray) -> np.ndarray:
    """Remove German plate vignette (circular seal) then standard preprocessing."""
    if plate_bgr.size == 0:
        return plate_bgr
    h, w = plate_bgr.shape[:2]
    scale = max(4.0, 400.0 / w, 100.0 / h)
    up = cv2.resize(
        plate_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_LANCZOS4,
    )
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape[:2]
    mask = _vignette_strategy_build_mask(gray, contours, h_img, w_img)
    return _vignette_strategy_apply_mask_and_postprocess(gray, mask, up)


def _preprocess_plate_strategies(plate_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Generate multiple preprocessing strategies for ensemble OCR."""
    strategies = [
        ("original", _preprocess_plate_strategy_original),
        ("upscale_clahe_threshold", _preprocess_plate_strategy_upscale_clahe_threshold),
        ("to_zero_180", _preprocess_plate_strategy_to_zero_180),
        ("morphological", _preprocess_plate_strategy_morphological),
        ("otsu", _preprocess_plate_strategy_otsu),
        (
            "flatten",
            _preprocess_plate_strategy_flatten,
        ),  # Aggressive upscaling + background flattening
        (
            "smooth_sharpen",
            _preprocess_plate_strategy_smooth_sharpen,
        ),  # Smoothing + sharpening for edge contrast
        (
            "remove_vignette",
            _preprocess_plate_strategy_remove_vignette,
        ),  # Remove German plate vignette
    ]
    return [(name, func(plate_bgr)) for name, func in strategies]


def _select_preprocessing_strategies(
    plate_bgr: np.ndarray,
    *,
    is_very_small: bool,
    is_small: bool,
) -> list[tuple[str, np.ndarray]]:
    """Select which preprocessing strategies to try (all / most / fast+few)."""
    all_strategies = _preprocess_plate_strategies(plate_bgr)
    fast_strategies = ["original", "upscale_clahe", "remove_vignette"]

    if is_very_small:
        return all_strategies
    if is_small:
        return [s for s in all_strategies if s[0] not in ["flatten", "smooth_sharpen"]]

    strategies_to_try: list[tuple[str, np.ndarray]] = []
    for name, processed in all_strategies:
        if name in fast_strategies or name.startswith("upscale_clahe"):
            strategies_to_try.append((name, processed))
    remaining = [
        s
        for s in all_strategies
        if s[0] not in fast_strategies and not s[0].startswith("upscale_clahe")
    ]
    strategies_to_try.extend(remaining[:2])
    return strategies_to_try


def _should_early_exit_preprocessing(
    is_very_small: bool,
    strategy_name: str,
    fast_strategies: list[str],
    conf: float | None,
    text: str,
    remaining_count: int,
) -> bool:
    """Return True if we can skip remaining strategies (good result from fast strategy)."""
    if is_very_small or remaining_count <= 0:
        return False
    if strategy_name not in fast_strategies and not strategy_name.startswith(
        "upscale_clahe"
    ):
        return False
    if conf is None or conf < 0.7 or len(text) < 4:
        return False
    import logging

    logger = logging.getLogger(__name__)
    logger.debug(
        "Early exit: got good result (conf=%.2f, text=%s) from strategy %s, skipping %d remaining strategies",
        conf,
        text,
        strategy_name,
        remaining_count,
    )
    return True


def _score_preprocessing_result(
    x: tuple[str | None, float | None, dict[str, object]],
    is_very_small: bool,
) -> tuple[float, float, float]:
    """Score (text, conf, meta) for choosing best: (priority, length_or_0, confidence)."""
    min_conf_for_length = 0.3
    text, conf, meta = x
    text_len = len(text or "")
    conf_val = conf or 0.0
    strategy = meta.get("strategy_name", "unknown") if meta else "unknown"
    strategy_boost = 0.15 if strategy == "flatten" and is_very_small else 0.0
    conf_val_boosted = conf_val + strategy_boost
    if conf_val_boosted < min_conf_for_length:
        return (conf_val_boosted, 0.0, 0.0)
    return (1.0, text_len, conf_val_boosted)


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
    import time

    ts = captured_at or datetime.now(UTC)
    if ts.tzinfo is None:
        # Enforce timezone-aware timestamps (privacy-safe + consistent logs).
        ts = ts.replace(tzinfo=UTC)

    # Time YOLO detection
    yolo_start = time.perf_counter()
    candidates = detector.detect(image_bgr)
    yolo_time = time.perf_counter() - yolo_start

    import logging

    logger = logging.getLogger(__name__)
    logger.info("YOLO detected %d candidates in %.3fs", len(candidates), yolo_time)

    # Time OCR (will be measured inside detect_plates_from_candidates)
    return detect_plates_from_candidates(
        image_bgr=image_bgr,
        candidates=candidates,
        ocr=ocr,
        captured_at=ts,
        image_path=image_path,
        include_crops=include_crops,
        crop_max_width=crop_max_width,
        crop_jpeg_quality=crop_jpeg_quality,
        yolo_time=yolo_time,  # Pass YOLO time for logging
    )


def _process_one_plate_candidate(
    cand: PlateCandidate,
    image_bgr: np.ndarray,
    width: int,
    height: int,
    ocr: PlateOcr,
    ts: datetime,
    *,
    include_crops: bool,
    crop_max_width: int,
    crop_jpeg_quality: int,
    image_path: Path | None,
) -> PlateDetection:
    """Run OCR on one candidate and return PlateDetection."""
    bbox = cand.bbox.clamp(width=width, height=height)
    crop = image_bgr[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
    text, ocr_conf, ocr_meta = ocr.recognize(crop)
    raw_text_value = ocr_meta.get("raw_text")
    raw_text = raw_text_value if isinstance(raw_text_value, str) else text
    raw_confidence = ocr_meta.get("raw_confidence")
    raw_ocr_conf = (
        float(raw_confidence) if isinstance(raw_confidence, float | int) else ocr_conf
    )
    crop_b64 = (
        _crop_preview_jpeg_b64(
            crop, max_width=crop_max_width, quality=crop_jpeg_quality
        )
        if include_crops
        else None
    )
    reliability = _combine_reliability(
        detection_confidence=cand.confidence, ocr_confidence=ocr_conf
    )
    return PlateDetection(
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
    yolo_time: float | None = None,
) -> ImagePlateDetections:
    """Run OCR (and optional crop previews) for provided plate candidates."""
    import time

    ts = captured_at or datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    ocr_start = time.perf_counter()
    height, width = image_bgr.shape[:2]
    detections = [
        _process_one_plate_candidate(
            cand,
            image_bgr,
            width,
            height,
            ocr,
            ts,
            include_crops=include_crops,
            crop_max_width=crop_max_width,
            crop_jpeg_quality=crop_jpeg_quality,
            image_path=image_path,
        )
        for cand in candidates
    ]
    ocr_time = time.perf_counter() - ocr_start
    if yolo_time is not None:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            "Detection timing: YOLO=%.3fs, OCR=%.3fs, total=%.3fs (candidates=%d)",
            yolo_time,
            ocr_time,
            yolo_time + ocr_time,
            len(candidates),
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


class PaddleOcrPlateRecognizer:
    """PaddleOCR plate recognizer - often better than EasyOCR for license plates."""

    def __init__(
        self,
        *,
        languages: list[str],
        min_confidence: float = 0.5,
        preprocess: bool = False,
        allowlist: bool = False,
        allowlist_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        normalize: bool = False,
    ) -> None:
        # Disable model source check to avoid connectivity delays
        import os

        original_disable = os.environ.get("DISABLE_MODEL_SOURCE_CHECK")
        os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            if original_disable is None:
                os.environ.pop("DISABLE_MODEL_SOURCE_CHECK", None)
            else:
                os.environ["DISABLE_MODEL_SOURCE_CHECK"] = original_disable
            raise ImportError(
                "PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr"
            ) from None

        # Use English for alphanumeric license plates
        lang_map = {"de": "en", "en": "en", "fr": "en", "it": "en", "es": "en"}
        ocr_langs = [lang_map.get(lang.lower(), "en") for lang in languages]
        ocr_lang = ocr_langs[0] if ocr_langs else "en"

        try:
            self._ocr = PaddleOCR(lang=ocr_lang)
        except Exception as e:
            if original_disable is None:
                os.environ.pop("DISABLE_MODEL_SOURCE_CHECK", None)
            else:
                os.environ["DISABLE_MODEL_SOURCE_CHECK"] = original_disable
            raise RuntimeError(f"Failed to initialize PaddleOCR: {e}") from e

        self._min_confidence = float(min_confidence)
        self._preprocess = bool(preprocess)
        self._allowlist = bool(allowlist)
        self._allowlist_chars = str(allowlist_chars)
        self._normalize = bool(normalize)

    def recognize(
        self, plate_bgr: np.ndarray
    ) -> tuple[str | None, float | None, dict[str, object]]:
        plate_rgb = self._prepare_plate_rgb_for_ocr(plate_bgr)
        # PaddleOCR API: use predict() in newer versions
        try:
            results = self._ocr.predict(plate_rgb)
        except (AttributeError, TypeError):
            results = self._ocr.ocr(plate_rgb)

        if not results or not results[0]:
            return (
                None,
                None,
                {"candidates": 0, "raw_text": None, "raw_confidence": None},
            )

        best_text, best_conf = _best_paddle_line(results[0])
        if not best_text:
            return (
                None,
                None,
                {
                    "candidates": len(results[0]),
                    "raw_text": None,
                    "raw_confidence": None,
                },
            )

        return _plate_ocr_post_process(
            best_text,
            best_conf,
            len(results[0]),
            "paddleocr",
            allowlist=self._allowlist,
            allowlist_chars=self._allowlist_chars,
            normalize=self._normalize,
            min_confidence=self._min_confidence,
            preprocess=self._preprocess,
        )

    def _prepare_plate_rgb_for_ocr(self, plate_bgr: np.ndarray) -> np.ndarray:
        h, w = plate_bgr.shape[:2]
        should_preprocess = self._preprocess and (w < 150 or h < 50)
        processed = (
            _preprocess_plate_for_ocr(plate_bgr) if should_preprocess else plate_bgr
        )
        return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)


class TesseractOcrPlateRecognizer:
    """Tesseract OCR plate recognizer - classic, well-established OCR engine."""

    def __init__(
        self,
        *,
        languages: list[str],
        min_confidence: float = 0.5,
        preprocess: bool = False,
        allowlist: bool = False,
        allowlist_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        normalize: bool = False,
    ) -> None:
        try:
            import pytesseract
        except ImportError as e:
            raise ImportError(
                "pytesseract not installed. Install with: pip install pytesseract. "
                "Also install Tesseract: brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)"
            ) from e

        self._pytesseract = pytesseract
        lang_map = {"de": "deu", "en": "eng"}
        ocr_langs = [lang_map.get(lang.lower(), lang.lower()) for lang in languages]
        self._lang = "+".join(ocr_langs) if ocr_langs else "eng"
        self._min_confidence = float(min_confidence)
        self._preprocess = bool(preprocess)
        self._allowlist = bool(allowlist)
        self._allowlist_chars = str(allowlist_chars)
        self._normalize = bool(normalize)

    def recognize(
        self, plate_bgr: np.ndarray
    ) -> tuple[str | None, float | None, dict[str, object]]:
        plate_rgb = self._prepare_plate_rgb_for_ocr(plate_bgr)

        config = "--psm 7"
        if self._allowlist:
            config += f" -c tessedit_char_whitelist={self._allowlist_chars}"
        cfg = {
            "lang": self._lang,
            "allowlist": self._allowlist,
            "allowlist_chars": self._allowlist_chars,
            "config": config,
        }
        best_text, best_conf = _tesseract_best_result(self._pytesseract, plate_rgb, cfg)
        if not best_text:
            return (
                None,
                None,
                {"candidates": 0, "raw_text": None, "raw_confidence": None},
            )

        return _plate_ocr_post_process(
            best_text,
            best_conf,
            1,
            "tesseract",
            allowlist=self._allowlist,
            allowlist_chars=self._allowlist_chars,
            normalize=self._normalize,
            min_confidence=self._min_confidence,
            preprocess=self._preprocess,
        )

    def _prepare_plate_rgb_for_ocr(self, plate_bgr: np.ndarray) -> np.ndarray:
        h, w = plate_bgr.shape[:2]
        should_preprocess = self._preprocess and (w < 150 or h < 50)
        processed = (
            _preprocess_plate_for_ocr(plate_bgr) if should_preprocess else plate_bgr
        )
        return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)


class DotsOcrPlateRecognizer:
    """dots.ocr plate recognizer - state-of-the-art multilingual document OCR."""

    def __init__(
        self,
        *,
        languages: list[str],  # noqa: ARG002
        min_confidence: float = 0.5,
        preprocess: bool = False,
        allowlist: bool = False,
        allowlist_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        normalize: bool = False,
    ) -> None:
        try:
            from dots_ocr import DotsOCRParser  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError(
                "dots.ocr not installed. Install with: pip install dots-ocr"
            ) from err

        self._parser = DotsOCRParser()
        self._min_confidence = float(min_confidence)
        self._preprocess = bool(preprocess)
        self._allowlist = bool(allowlist)
        self._allowlist_chars = str(allowlist_chars)
        self._normalize = bool(normalize)

    def recognize(
        self, plate_bgr: np.ndarray
    ) -> tuple[str | None, float | None, dict[str, object]]:
        plate_rgb = self._prepare_plate_rgb_for_ocr(plate_bgr)
        from PIL import Image

        pil_image = Image.fromarray(plate_rgb)
        try:
            result = self._parser.parse(
                pil_image,
                prompt_mode="prompt_ocr",
            )
            text = _dots_ocr_extract_text(result)
            if not text:
                return (
                    None,
                    None,
                    {"candidates": 0, "raw_text": None, "raw_confidence": None},
                )
            return _plate_ocr_post_process(
                text,
                0.7,
                1,
                "dotsocr",
                allowlist=self._allowlist,
                allowlist_chars=self._allowlist_chars,
                normalize=self._normalize,
                min_confidence=self._min_confidence,
                preprocess=self._preprocess,
            )
        except Exception as e:
            return (
                None,
                None,
                {
                    "candidates": 0,
                    "raw_text": None,
                    "raw_confidence": None,
                    "error": str(e),
                },
            )

    def _prepare_plate_rgb_for_ocr(self, plate_bgr: np.ndarray) -> np.ndarray:
        h, w = plate_bgr.shape[:2]
        should_preprocess = self._preprocess and (w < 150 or h < 50)
        processed = (
            _preprocess_plate_for_ocr(plate_bgr) if should_preprocess else plate_bgr
        )
        return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)


class ChandraOcrPlateRecognizer:
    """Chandra OCR plate recognizer - handles complex documents, handwriting, tables."""

    def __init__(
        self,
        *,
        languages: list[str],  # noqa: ARG002
        min_confidence: float = 0.5,
        preprocess: bool = False,
        allowlist: bool = False,
        allowlist_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        normalize: bool = False,
    ) -> None:
        try:
            from chandra.model import InferenceManager  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError(
                "chandra-ocr not installed. Install with: pip install chandra-ocr"
            ) from err

        self._manager = InferenceManager(method="hf")
        self._min_confidence = float(min_confidence)
        self._preprocess = bool(preprocess)
        self._allowlist = bool(allowlist)
        self._allowlist_chars = str(allowlist_chars)
        self._normalize = bool(normalize)

    def recognize(
        self, plate_bgr: np.ndarray
    ) -> tuple[str | None, float | None, dict[str, object]]:
        plate_rgb = self._prepare_plate_rgb_for_ocr(plate_bgr)
        from PIL import Image

        pil_image = Image.fromarray(plate_rgb)
        try:
            results = self._manager.generate([pil_image])
            if not results or not results[0]:
                return (
                    None,
                    None,
                    {"candidates": 0, "raw_text": None, "raw_confidence": None},
                )
            text = (
                results[0].markdown.strip() if hasattr(results[0], "markdown") else ""
            )
            if not text:
                return (
                    None,
                    None,
                    {"candidates": 0, "raw_text": None, "raw_confidence": None},
                )
            return _plate_ocr_post_process(
                text,
                0.7,
                1,
                "chandra",
                allowlist=self._allowlist,
                allowlist_chars=self._allowlist_chars,
                normalize=self._normalize,
                min_confidence=self._min_confidence,
                preprocess=self._preprocess,
            )
        except Exception as e:
            return (
                None,
                None,
                {
                    "candidates": 0,
                    "raw_text": None,
                    "raw_confidence": None,
                    "error": str(e),
                },
            )

    def _prepare_plate_rgb_for_ocr(self, plate_bgr: np.ndarray) -> np.ndarray:
        h, w = plate_bgr.shape[:2]
        should_preprocess = self._preprocess and (w < 150 or h < 50)
        processed = (
            _preprocess_plate_for_ocr(plate_bgr) if should_preprocess else plate_bgr
        )
        return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)


def create_plate_recognizer(
    *,
    engine: str,
    languages: list[str],
    min_confidence: float = 0.5,
    model_storage_directory: Path | None = None,
    download_enabled: bool = True,
    preprocess: bool = False,
    allowlist: bool = False,
    allowlist_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    normalize: bool = False,
) -> PlateOcr:
    """Factory function to create a plate recognizer based on engine name."""
    if engine == "easyocr":
        return EasyOcrPlateRecognizer(
            languages=languages,
            min_confidence=min_confidence,
            model_storage_directory=model_storage_directory,
            download_enabled=download_enabled,
            preprocess=preprocess,
            allowlist=allowlist,
            allowlist_chars=allowlist_chars,
            normalize=normalize,
        )
    if engine == "paddleocr":
        return PaddleOcrPlateRecognizer(
            languages=languages,
            min_confidence=min_confidence,
            preprocess=preprocess,
            allowlist=allowlist,
            allowlist_chars=allowlist_chars,
            normalize=normalize,
        )
    if engine == "tesseract":
        return TesseractOcrPlateRecognizer(
            languages=languages,
            min_confidence=min_confidence,
            preprocess=preprocess,
            allowlist=allowlist,
            allowlist_chars=allowlist_chars,
            normalize=normalize,
        )
    if engine == "dotsocr":
        return DotsOcrPlateRecognizer(
            languages=languages,
            min_confidence=min_confidence,
            preprocess=preprocess,
            allowlist=allowlist,
            allowlist_chars=allowlist_chars,
            normalize=normalize,
        )
    if engine == "chandra":
        return ChandraOcrPlateRecognizer(
            languages=languages,
            min_confidence=min_confidence,
            preprocess=preprocess,
            allowlist=allowlist,
            allowlist_chars=allowlist_chars,
            normalize=normalize,
        )
    raise ValueError(
        f"Unknown OCR engine: {engine}. Choose from: easyocr, paddleocr, tesseract, dotsocr, chandra"
    )


def _consensus_match_score(normalized: str, other_normalized: str) -> float:
    """Return 1.0 for exact match, 0.5 for partial (>=75% similarity), 0.0 otherwise."""
    if normalized == other_normalized:
        return 1.0
    if not normalized or not other_normalized:
        return 0.0
    common_chars = sum(1 for c in normalized if c in other_normalized)
    similarity = common_chars / max(len(normalized), len(other_normalized))
    return 0.5 if similarity >= 0.75 else 0.0


class EnsemblePlateRecognizer:
    """Ensemble OCR recognizer that tries multiple engines and picks the best result."""

    def __init__(
        self,
        *,
        engines: list[str],
        languages: list[str],
        min_confidence: float = 0.5,
        model_storage_directory: Path | None = None,
        download_enabled: bool = True,
        preprocess: bool = False,
        allowlist: bool = False,
        allowlist_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        normalize: bool = False,
    ) -> None:
        self._engines = engines
        self._recognizers: list[PlateOcr] = []
        for engine in engines:
            try:
                recognizer = create_plate_recognizer(
                    engine=engine,
                    languages=languages,
                    min_confidence=min_confidence,
                    model_storage_directory=model_storage_directory,
                    download_enabled=download_enabled,
                    preprocess=preprocess,
                    allowlist=allowlist,
                    allowlist_chars=allowlist_chars,
                    normalize=normalize,
                )
                self._recognizers.append(recognizer)
            except Exception as e:
                # Log but continue with other engines
                import logging

                logging.warning(f"Failed to initialize OCR engine {engine}: {e}")

        if not self._recognizers:
            raise ValueError(f"Failed to initialize any OCR engines from: {engines}")

    def recognize(
        self, plate_bgr: np.ndarray
    ) -> tuple[str | None, float | None, dict[str, object]]:
        """Try all engines and return the best result using intelligent selection."""
        all_results: list[dict[str, object]] = []

        for i, recognizer in enumerate(self._recognizers):
            try:
                text, conf, meta = recognizer.recognize(plate_bgr)
                engine_name = self._engines[i]
                result_info = {
                    "engine": engine_name,
                    "text": text,
                    "confidence": conf,
                    "meta": meta,
                    "raw_text": meta.get("raw_text") if meta else None,
                    "normalized_text": meta.get("normalized_text") if meta else None,
                }
                all_results.append(result_info)
            except Exception as e:
                # Log but continue with other engines
                import logging

                logging.warning(f"OCR engine {self._engines[i]} failed: {e}")
                all_results.append(
                    {
                        "engine": self._engines[i],
                        "text": None,
                        "confidence": None,
                        "error": str(e),
                    }
                )

        # Intelligent selection: score each result and pick the best
        scored_results: list[tuple[dict[str, object], float]] = []

        for result in all_results:
            text = cast("str | None", result.get("text"))
            if not text:
                continue

            score = self._score_result(result, all_results)
            scored_results.append((result, score))

        if not scored_results:
            return None, None, {"ensemble_results": all_results}

        # Sort by score (descending) and pick the best
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_result, best_score = scored_results[0]

        best_text: str | None = cast("str | None", best_result.get("text"))
        best_conf = cast(
            "float", best_result.get("confidence", 0.7)
        )  # Default confidence if missing
        best_meta_base = best_result.get("meta", {})
        best_meta = {
            **cast("dict[str, object]", best_meta_base),
            "ensemble_engine": best_result["engine"],
            "ensemble_score": best_score,
            "ensemble_results": all_results,
            "ensemble_rankings": [
                {"engine": r["engine"], "score": s, "text": r.get("text")}
                for r, s in scored_results[:3]  # Top 3 for debugging
            ],
        }

        return best_text, best_conf, best_meta

    def _score_result(
        self, result: dict[str, object], all_results: list[dict[str, object]]
    ) -> float:
        """Score a result based on multiple factors to determine the 'best bet'."""
        score = 0.0
        text = cast("str", result.get("text", ""))
        conf = cast("float | None", result.get("confidence"))
        engine = cast("str", result.get("engine", ""))

        # Factor 1: Confidence score (0-1, weighted 40%)
        if conf is not None:
            score += conf * 0.4
        else:
            # Default confidence for engines without scores
            if engine in ["dotsocr", "chandra"]:
                score += 0.7 * 0.4  # High-quality engines get default high confidence
            else:
                score += 0.5 * 0.4  # Other engines get medium default

        # Factor 2: Text quality (0-1, weighted 30%)
        # Prefer results that look like valid license plates
        text_quality = self._assess_text_quality(text)
        score += text_quality * 0.3

        # Factor 3: Consensus (0-1, weighted 20%)
        # If multiple engines agree, that's a good sign
        consensus = self._calculate_consensus(result, all_results)
        score += consensus * 0.2

        # Factor 4: Engine reliability (0-1, weighted 10%)
        # Prefer engines known to be more accurate
        engine_reliability = self._get_engine_reliability(engine)
        score += engine_reliability * 0.1

        return score

    def _assess_text_quality(self, text: str) -> float:
        """Assess text quality based on license plate characteristics."""
        if not text:
            return 0.0

        quality = 0.5  # Base score

        # Prefer reasonable length (German plates are typically 6-8 chars)
        length = len(text)
        if 5 <= length <= 10:
            quality += 0.2
        elif 3 <= length <= 12:
            quality += 0.1

        # Prefer alphanumeric content (license plates are alphanumeric)
        alnum_count = sum(1 for c in text if c.isalnum())
        alnum_ratio = alnum_count / len(text) if text else 0.0
        quality += alnum_ratio * 0.2

        # Prefer uppercase (German plates are uppercase)
        alpha_count = sum(1 for c in text if c.isalpha())
        upper_count = sum(1 for c in text if c.isupper())
        upper_ratio = (upper_count / alpha_count) if alpha_count else 0.0
        quality += upper_ratio * 0.1

        return min(1.0, quality)

    def _calculate_consensus(
        self, result: dict[str, object], all_results: list[dict[str, object]]
    ) -> float:
        """Calculate how much other engines agree with this result."""
        text = cast("str", result.get("text", ""))
        if not text:
            return 0.0
        normalized = _normalize_plate_text(text)
        matches = 0.0
        total = 0
        for other in all_results:
            other_text = other.get("text")
            if not other_text or other == result:
                continue
            total += 1
            other_normalized = _normalize_plate_text(cast("str", other_text))
            matches += _consensus_match_score(normalized, other_normalized)
        return matches / total if total > 0 else 0.0

    def _get_engine_reliability(self, engine: str) -> float:
        """Get reliability score for an engine based on known performance."""
        reliability_scores = {
            "dotsocr": 0.95,  # State-of-the-art, very reliable
            "chandra": 0.90,  # High-quality, reliable
            "paddleocr": 0.75,  # Good for license plates
            "easyocr": 0.70,  # General purpose, decent
            "tesseract": 0.65,  # Classic, reliable but lower accuracy
        }
        return reliability_scores.get(engine, 0.5)


def create_plate_recognizer_from_config(
    *,
    ocr_engine: str | list[str],
    languages: list[str],
    min_confidence: float = 0.5,
    model_storage_directory: Path | None = None,
    download_enabled: bool = True,
    preprocess: bool = False,
    allowlist: bool = False,
    allowlist_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    normalize: bool = False,
) -> PlateOcr:
    """Create a plate recognizer from config (supports single engine or ensemble)."""
    if isinstance(ocr_engine, list):
        if len(ocr_engine) == 1:
            # Single engine in list, unwrap it
            return create_plate_recognizer(
                engine=ocr_engine[0],
                languages=languages,
                min_confidence=min_confidence,
                model_storage_directory=model_storage_directory,
                download_enabled=download_enabled,
                preprocess=preprocess,
                allowlist=allowlist,
                allowlist_chars=allowlist_chars,
                normalize=normalize,
            )
        # Multiple engines, use ensemble
        return EnsemblePlateRecognizer(
            engines=ocr_engine,
            languages=languages,
            min_confidence=min_confidence,
            model_storage_directory=model_storage_directory,
            download_enabled=download_enabled,
            preprocess=preprocess,
            allowlist=allowlist,
            allowlist_chars=allowlist_chars,
            normalize=normalize,
        )
    # Single engine string
    return create_plate_recognizer(
        engine=ocr_engine,
        languages=languages,
        min_confidence=min_confidence,
        model_storage_directory=model_storage_directory,
        download_enabled=download_enabled,
        preprocess=preprocess,
        allowlist=allowlist,
        allowlist_chars=allowlist_chars,
        normalize=normalize,
    )
