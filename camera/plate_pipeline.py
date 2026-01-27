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
        """Try multiple preprocessing strategies and return best result.

        Optimized to use fewer strategies for normal images, all strategies only for challenging cases.
        """
        # Get original crop size for strategy selection
        h, w = plate_bgr.shape[:2]
        is_very_small = h < 30 or w < 100
        is_small = h < 50 or w < 150

        # Adaptive strategy selection:
        # - Normal images: try 2-3 fast strategies first, early exit if good result
        # - Small images: try 4-5 strategies
        # - Very small images: try all 8 strategies
        all_strategies = _preprocess_plate_strategies(plate_bgr)

        # Fast strategies to try first (good balance of speed/accuracy)
        # Note: "upscale_clahe_threshold" is the actual name, but we'll match by prefix
        fast_strategies = ["original", "upscale_clahe", "remove_vignette"]

        # Select which strategies to use
        if is_very_small:
            # Very small: use all strategies
            strategies_to_try = all_strategies
        elif is_small:
            # Small: use most strategies but skip slowest ones
            strategies_to_try = [
                s for s in all_strategies if s[0] not in ["flatten", "smooth_sharpen"]
            ]
        else:
            # Normal size: try fast strategies first, then others if needed
            strategies_to_try = []
            # Add fast strategies first
            for name, func in all_strategies:
                if name in fast_strategies or name.startswith("upscale_clahe"):
                    strategies_to_try.append((name, func))
            # Add remaining strategies (but limit to 2-3 more for normal images)
            remaining = [
                s
                for s in all_strategies
                if s[0] not in fast_strategies and not s[0].startswith("upscale_clahe")
            ]
            # For normal images, only add 2 more strategies to keep it fast
            strategies_to_try.extend(remaining[:2])

        all_results: list[tuple[str | None, float | None, dict[str, object]]] = []
        good_enough_conf = 0.7  # Early exit if we get a high-confidence result

        for strategy_name, processed in strategies_to_try:
            plate_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            text, conf, meta = self._recognize_single(
                plate_rgb, strategy_name=strategy_name
            )
            if text is not None:
                all_results.append((text, conf, meta))

                # Early exit optimization: if we get a high-confidence result from fast strategies,
                # and the image is not very small, we can skip remaining strategies
                if (
                    not is_very_small
                    and (
                        strategy_name in fast_strategies
                        or strategy_name.startswith("upscale_clahe")
                    )
                    and conf is not None
                    and conf >= good_enough_conf
                    and len(text) >= 4  # Reasonable length
                ):
                    # Good enough result from fast strategy - skip remaining strategies
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.debug(
                        "Early exit: got good result (conf=%.2f, text=%s) from strategy %s, skipping %d remaining strategies",
                        conf,
                        text,
                        strategy_name,
                        len(strategies_to_try) - len(all_results),
                    )
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

        # Select best result: prefer longer text IF confidence is reasonable, else prefer confidence
        # This helps when OCR only reads part of the plate, but avoids low-confidence garbage
        min_conf_for_length_preference = 0.3  # Only prefer length if confidence >= this

        def score_result(
            x: tuple[str | None, float | None, dict[str, object]],
        ) -> tuple[float, float, float]:
            text, conf, meta = x
            text_len = len(text or "")
            conf_val = conf or 0.0
            strategy = meta.get("strategy_name", "unknown") if meta else "unknown"

            # Boost score for "flatten" strategy on very small crops
            # This strategy is specifically designed for tiny crops with background flattening
            strategy_boost = 0.15 if strategy == "flatten" and is_very_small else 0.0
            conf_val_boosted = conf_val + strategy_boost

            # If confidence is too low, prioritize confidence
            if conf_val_boosted < min_conf_for_length_preference:
                return (conf_val_boosted, 0.0, 0.0)  # Only confidence matters

            # If confidence is reasonable, prefer longer text
            return (
                1.0,
                text_len,
                conf_val_boosted,
            )  # (high priority flag, length, confidence)

        best_result = max(all_results, key=score_result)

        best_text, best_conf, best_meta = best_result
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


def _best_easyocr_candidate(results: object) -> tuple[str, float] | None:
    """Select best OCR result, combining multiple detections if they're on the same line.

    For plates with spaces (e.g., "LIP VE 351"), EasyOCR may return multiple detections.
    We combine them if they're vertically aligned (same line).
    """
    if not isinstance(results, list) or not results:
        return None

    # If only one result, return it
    if len(results) == 1:
        item = results[0]
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            text_str = str(item[1]).strip()
            if text_str:
                return text_str, float(item[2])
        return None

    # Multiple results: try to combine them if they're on the same line
    valid_results: list[tuple[str, float, float, float]] = (
        []
    )  # (text, conf, y_center, x_center)

    for item in results:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text_str = str(item[1]).strip()
        if not text_str:
            continue
        conf_f = float(item[2])

        # Extract bounding box to check vertical alignment
        bbox = item[0] if len(item) > 0 else None
        y_center = 0.0
        x_center = 0.0
        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) >= 4
            and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in bbox)
        ):
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            y_coords = [pt[1] for pt in bbox]
            x_coords = [pt[0] for pt in bbox]
            y_center = sum(y_coords) / len(y_coords)
            x_center = sum(x_coords) / len(x_coords)

        valid_results.append((text_str, conf_f, y_center, x_center))

    if not valid_results:
        return None

    # Group results by vertical position (same line if y_center is within threshold)
    # Adaptive threshold: use 10% of average bbox height, minimum 10px, maximum 30px
    if valid_results:
        # Estimate image height from y coordinates (rough estimate)
        max_y = max(r[2] for r in valid_results)
        min_y = min(r[2] for r in valid_results)
        estimated_height = max_y - min_y if max_y > min_y else 100.0
        y_threshold = max(10.0, min(30.0, estimated_height * 0.1))
    else:
        y_threshold = 20.0
    groups: list[list[tuple[str, float, float, float]]] = []

    for result in valid_results:
        text, conf, y, _x = result
        # Find a group with similar y position
        matched = False
        for group in groups:
            if group and abs(group[0][2] - y) < y_threshold:
                group.append(result)
                matched = True
                break
        if not matched:
            groups.append([result])

    # For each group, combine text (sorted by x position) and use average confidence
    combined_results: list[tuple[str, float]] = []
    for group in groups:
        # Sort by x position (left to right)
        group.sort(key=lambda r: r[3])
        combined_text = " ".join(r[0] for r in group)
        avg_conf = sum(r[1] for r in group) / len(group)
        combined_results.append((combined_text, avg_conf))

    # Return the result with highest confidence
    if combined_results:
        return max(combined_results, key=lambda x: x[1])

    # Fallback: return single best result
    best_text: str | None = None
    best_conf: float | None = None
    for text, conf, _, _ in valid_results:
        if best_conf is None or conf > best_conf:
            best_text = text
            best_conf = conf

    if best_text is None or best_conf is None:
        return None
    return best_text, best_conf


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


def _remove_vignette_characters(text: str) -> str:
    """Remove spurious single characters that are likely vignettes (circular seals).

    German license plates have vignettes (circular seals) between letter groups
    that can be misread as single characters like 'E' or 'B'. This function
    removes isolated single characters in the middle of the text that are likely
    vignettes rather than actual plate characters.

    Pattern: Letters-Numbers or Letters-Letters-Numbers
    Vignette appears between groups, often as a single character.
    """
    if not text or len(text) < 4:
        return text

    import re

    # Characters commonly misread from vignettes (circular/oval shapes)
    # Note: 'I' and '1' are added because circular seals can appear as vertical lines
    vignette_chars = {"E", "B", "O", "0", "8", "6", "G", "Q", "D", "P", "I", "1"}

    # Step 1: Handle text with spaces - split by spaces and process
    # This handles both: "LIP E AS277" and "L1P BAS277" (B at start of word)
    parts = text.split()
    if len(parts) >= 2:
        result_parts = []
        i = 0
        while i < len(parts):
            # Case 1: Single vignette char between two longer parts: "LIP E AS277"
            if (
                i > 0
                and i < len(parts) - 1
                and len(parts[i]) == 1
                and parts[i].upper() in vignette_chars
                and len(parts[i - 1]) >= 2
                and len(parts[i + 1]) >= 2
            ):
                # This is a vignette - skip it and combine prev and next
                result_parts[-1] = result_parts[-1] + parts[i + 1]
                i += 2  # Skip both the vignette and the next part (already merged)
            # Case 2: Vignette char at start of word: "L1P BAS277" -> remove B
            elif (
                i > 0
                and len(parts[i]) >= 3  # At least 3 chars (vignette + 2+ rest)
                and parts[i][0].upper() in vignette_chars
                and len(parts[i - 1]) >= 2
            ):
                # Remove first char (vignette) and merge with previous
                result_parts[-1] = result_parts[-1] + parts[i][1:]
                i += 1
            else:
                result_parts.append(parts[i])
                i += 1
        text = " ".join(result_parts)

    # Step 2: Remove spaces and separators for further analysis
    text_clean = text.replace(" ", "").replace(".", "").replace("-", "")
    if len(text_clean) < 4:
        return text_clean

    # Step 3: Handle single vignette chars without spaces (more aggressive)
    # Use regex but be smarter - only remove if character is clearly in middle
    # e.g., "LIPBAS277" -> "LIPAS277" (remove B)
    def remove_vignette(match):
        prefix = match.group(1)
        single_char = match.group(2)
        suffix = match.group(3)

        # Only remove if it's a common vignette character
        if single_char.upper() not in vignette_chars:
            return match.group(0)  # Keep original

        # Additional check: vignettes are typically between letter groups
        # Don't remove if character is part of a valid city code (positions 0-2)
        # German city codes are typically 1-3 letters, so positions 0-2 are protected
        char_pos = len(prefix)

        # Be conservative: don't remove if character is at position 0, 1, or 2
        # These positions are typically part of the city code (1-3 letters)
        # Exception: if prefix is only 1-2 letters AND suffix starts with digits,
        # then the character might be a vignette (e.g., "AB" + "I" + "123")
        if char_pos <= 2:
            # Character is in city code range (positions 0-2)
            # Only remove if prefix is short (1-2 chars) AND suffix looks like digits
            if len(prefix) >= 3:
                # Prefix is 3+ chars - definitely part of city code, don't remove
                return match.group(0)
            # For 1-2 char prefix, check if suffix starts with digits
            # If suffix starts with letters, the character might be part of city code
            if suffix and suffix[0].isalpha():
                # Suffix starts with letter - might be part of city code, be conservative
                return match.group(0)

        return prefix + suffix

    # Try matching with explicit vignette chars first (more targeted)
    # Build regex pattern dynamically from vignette_chars set
    # Use greedy matching (default) - vignettes typically appear after letter groups
    vignette_pattern = "".join(sorted(vignette_chars))
    result = re.sub(
        f"([A-Z0-9]{{2,}})([{vignette_pattern}])([A-Z0-9]{{2,}})",
        remove_vignette,
        text_clean,
        flags=re.IGNORECASE,
    )

    # Only return modified result if it's still reasonable length
    if result != text_clean and len(result) >= 3:
        return result

    return text_clean


def _correct_plate_characters_by_position(text: str) -> str:
    """Apply position-based character correction for German plates.

    German plate format: [1-3 letters][1-2 letters][1-4 digits]
    Positions 0-2: letters (city code)
    Positions 3-4: optional letters (suffix)
    Positions 5+: digits

    Only corrects characters that are likely OCR errors (e.g., 0/O, 1/I confusion).
    Uses heuristics: if we see a pattern like "letter-digit-letter", assume digits section starts later.
    """
    if not text or len(text) < 3:
        return text

    # Heuristic: find where digits section likely starts
    # Look for pattern: letters, then digits (at least 2 consecutive digits)
    digit_start = len(text)
    for i in range(len(text) - 1):
        if text[i].isdigit() and text[i + 1].isdigit():
            # Found start of digit section
            digit_start = i
            break

    corrected = []
    for i, char in enumerate(text):
        if i < digit_start:  # Letter positions (before digit section)
            # In letter positions, digits are likely OCR errors
            if char.isdigit() and char in _INT_TO_CHAR:
                corrected.append(_INT_TO_CHAR[char])
            else:
                corrected.append(char)
        else:  # Digit positions
            # In digit positions, letters that look like digits are likely OCR errors
            if char.isalpha() and char in _CHAR_TO_INT:
                corrected.append(_CHAR_TO_INT[char])
            else:
                corrected.append(char)

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


def _preprocess_plate_strategy_flatten(plate_bgr: np.ndarray) -> np.ndarray:
    """Strategy 6: Aggressive upscaling + background flattening for very small crops.

    This strategy is optimized for very small crops (e.g., 20px tall) where simple
    upscaling isn't enough. It:
    1. Aggressively upscales (targets 500px+ width, 150px+ height)
    2. Flattens background using multiple techniques (makes plate background uniform)
    3. Enhances contrast with CLAHE
    4. Applies denoising
    5. Uses adaptive thresholding optimized for small text
    6. Additional sharpening for character clarity
    """
    if plate_bgr.size == 0:
        return plate_bgr

    h, w = plate_bgr.shape[:2]

    # Very aggressive upscaling for tiny crops
    # Target: at least 500px wide or 150px tall (even larger for better OCR)
    scale = max(6.0, 500.0 / w, 150.0 / h)
    up = cv2.resize(
        plate_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_LANCZOS4,  # LANCZOS4 for best quality on upscaling
    )

    # Convert to grayscale
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    # Denoise first (helps with small text artifacts)
    denoised = cv2.fastNlMeansDenoising(
        gray, h=10, templateWindowSize=7, searchWindowSize=21
    )

    # Background flattening: multiple techniques
    # Method 1: Morphological background estimation
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bg_morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel_bg)
    bg_morph = cv2.GaussianBlur(bg_morph, (31, 31), 0)

    # Method 2: Rolling ball algorithm (simplified - subtract blurred version)
    bg_blur = cv2.GaussianBlur(denoised, (51, 51), 0)

    # Combine both background estimates
    background = cv2.addWeighted(bg_morph, 0.5, bg_blur, 0.5, 0)

    # Subtract background to flatten (makes text stand out more)
    flattened = cv2.subtract(denoised, background)

    # Normalize to full dynamic range
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)

    # Apply CLAHE for contrast enhancement (more aggressive for small crops)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    eq = clahe.apply(flattened)

    # Stronger sharpening to enhance character edges (important for small text)
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

    # Adaptive threshold optimized for small text
    # Block size based on image size
    block_size = max(11, int(up.shape[1] / 8))
    if block_size % 2 == 0:
        block_size += 1
    thr = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        10,  # Higher C value for better contrast on flattened images
    )

    # Additional morphological cleanup (remove small noise)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel_clean)

    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


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
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

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


def _preprocess_plate_strategy_remove_vignette(plate_bgr: np.ndarray) -> np.ndarray:
    """Strategy 8: Remove German plate vignette (circular seal) that causes spurious OCR.

    German license plates have a circular/oval vignette (seal/sticker) between letter groups
    that has different contrast and can be misread as characters like 'E' or 'B'.

    This strategy:
    1. Upscales for better detection
    2. Detects circular/oval dark regions (vignettes)
    3. Masks them out (fills with background color)
    4. Applies standard preprocessing
    """
    if plate_bgr.size == 0:
        return plate_bgr

    h, w = plate_bgr.shape[:2]

    # Upscale for better vignette detection
    scale = max(4.0, 400.0 / w, 100.0 / h)
    up = cv2.resize(
        plate_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # Convert to grayscale
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    # Detect and mask vignettes (circular/oval dark regions)
    # Vignettes are typically darker than text and have circular/oval shape
    # Use adaptive threshold to find dark regions
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    # Find contours that could be vignettes
    # Vignettes are typically circular/oval, medium-sized, and positioned in the middle
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask for vignette removal
    mask = np.ones(gray.shape, dtype=np.uint8) * 255

    h_img, w_img = gray.shape
    center_y, center_x = h_img // 2, w_img // 2

    for contour in contours:
        # Check if contour is roughly circular/oval (vignette shape)
        area = cv2.contourArea(contour)
        if area < 50 or area > (h_img * w_img * 0.3):  # Too small or too large
            continue

        # Check circularity (vignettes are roughly circular)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Vignettes have circularity around 0.7-1.0 (circular/oval)
        if circularity < 0.5:  # Not circular enough
            continue

        # Check if it's positioned in the middle region (between letter groups)
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        # Vignettes are typically in the middle horizontally, middle-upper vertically
        x_center_dist = abs(cx - center_x) / w_img
        y_center_dist = abs(cy - center_y) / h_img

        # Should be roughly centered horizontally, can be slightly above center vertically
        if x_center_dist < 0.3 and y_center_dist < 0.4:
            # This looks like a vignette - mask it out
            cv2.drawContours(
                mask, [contour], -1, 0, -1
            )  # Fill with black (will be replaced)

    # Apply mask: replace vignette regions with background color (light gray/white)
    # Get background color (average of edge pixels)
    edge_pixels = np.concatenate(
        [
            gray[0, :],  # Top edge
            gray[-1, :],  # Bottom edge
            gray[:, 0],  # Left edge
            gray[:, -1],  # Right edge
        ]
    )
    background_color = int(np.median(edge_pixels))

    # Replace masked regions with background color
    gray_masked = gray.copy()
    gray_masked[mask == 0] = background_color

    # Apply standard preprocessing after vignette removal
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_masked)

    # Sharpening
    kernel_sharpen = np.array(
        [[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32
    )
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

    # Adaptive threshold
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

    # Time OCR (will be measured inside detect_plates_from_candidates)
    result = detect_plates_from_candidates(
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
    yolo_time: float | None = None,  # YOLO detection time (for logging)
) -> ImagePlateDetections:
    """Run OCR (and optional crop previews) for provided plate candidates.

    This is useful for dataset evaluation where bounding boxes come from labels
    rather than a detector model.
    """
    import time

    ts = captured_at or datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)

    # Time OCR processing
    ocr_start = time.perf_counter()
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
    ocr_time = time.perf_counter() - ocr_start

    # Log timing if YOLO time was provided (from detect_plates_in_image)
    if yolo_time is not None:
        import logging

        logger = logging.getLogger(__name__)
        total_time = yolo_time + ocr_time
        logger.info(
            "Detection timing: YOLO=%.3fs, OCR=%.3fs, total=%.3fs (candidates=%d)",
            yolo_time,
            ocr_time,
            total_time,
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
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
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

        # Parse PaddleOCR results (format varies by version)
        best_text: str | None = None
        best_conf: float = 0.0

        for line in results[0]:
            if not line or len(line) < 2:
                continue

            text: str | None = None
            conf: float = 0.0

            if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                text_val, conf_val = line[1][0], line[1][1]
                text = str(text_val).strip() if text_val else None
                try:
                    conf = float(conf_val)
                except (ValueError, TypeError):
                    conf = 0.0
            elif len(line) >= 3:
                text_val = line[1]
                conf_val = line[2] if len(line) > 2 else 0.0
                text = str(text_val).strip() if text_val else None
                try:
                    conf = float(conf_val)
                except (ValueError, TypeError):
                    conf = 0.0

            if text and conf > best_conf:
                best_text = text
                best_conf = conf

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

        if self._allowlist:
            best_text = "".join(
                c for c in best_text if c.upper() in self._allowlist_chars
            )

        # Remove vignette characters (spurious single chars from circular seals)
        text_no_vignette = _remove_vignette_characters(best_text)

        # Apply position-based character correction
        corrected_text = _correct_plate_characters_by_position(text_no_vignette)

        normalized_text = (
            _normalize_plate_text(corrected_text) if self._normalize else corrected_text
        )
        meta: dict[str, object] = {
            "candidates": len(results[0]),
            "raw_text": best_text,
            "corrected_text": corrected_text,
            "raw_confidence": best_conf,
            "normalized_text": normalized_text if self._normalize else None,
            "preprocess": self._preprocess,
            "allowlist": self._allowlist,
            "engine": "paddleocr",
        }

        if best_conf < self._min_confidence:
            return None, best_conf, meta

        return (normalized_text if self._normalize else corrected_text), best_conf, meta

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
            import pytesseract  # type: ignore[import-untyped]
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

        # Try multiple PSM modes
        psm_modes = ["7", "11", "6"]
        best_text: str | None = None
        best_conf: float = 0.0

        for psm in psm_modes:
            psm_config = f"--psm {psm}"
            if self._allowlist:
                psm_config += f" -c tessedit_char_whitelist={self._allowlist_chars}"

            try:
                data = self._pytesseract.image_to_data(
                    plate_rgb,
                    lang=self._lang,
                    config=psm_config,
                    output_type=self._pytesseract.Output.DICT,
                )

                texts: list[str] = []
                confidences: list[float] = []

                num_words = len(data.get("text", []))
                for i in range(num_words):
                    text = (
                        data.get("text", [""])[i]
                        if i < len(data.get("text", []))
                        else ""
                    )
                    if text and text.strip():
                        conf_val = (
                            int(data.get("conf", [0])[i])
                            if i < len(data.get("conf", []))
                            else 0
                        )
                        if conf_val > 0:
                            texts.append(text.strip())
                            confidences.append(float(conf_val) / 100.0)

                if texts:
                    combined = " ".join(texts).strip()
                    avg_conf = (
                        sum(confidences) / len(confidences) if confidences else 0.5
                    )
                    if avg_conf > best_conf:
                        best_text = combined
                        best_conf = avg_conf
            except Exception:
                continue

        if not best_text:
            try:
                simple_text = self._pytesseract.image_to_string(
                    plate_rgb, lang=self._lang, config=config
                ).strip()
                if simple_text:
                    best_text = simple_text
                    best_conf = 0.5
            except Exception:
                pass

        if not best_text:
            return (
                None,
                None,
                {"candidates": 0, "raw_text": None, "raw_confidence": None},
            )

        # Remove vignette characters (spurious single chars from circular seals)
        text_no_vignette = _remove_vignette_characters(best_text)

        # Apply position-based character correction
        corrected_text = _correct_plate_characters_by_position(text_no_vignette)

        normalized_text = (
            _normalize_plate_text(corrected_text) if self._normalize else corrected_text
        )
        meta: dict[str, object] = {
            "candidates": 1,
            "raw_text": best_text,
            "corrected_text": corrected_text,
            "raw_confidence": best_conf,
            "normalized_text": normalized_text if self._normalize else None,
            "preprocess": self._preprocess,
            "allowlist": self._allowlist,
            "engine": "tesseract",
        }

        if best_conf < self._min_confidence:
            return None, best_conf, meta

        return (normalized_text if self._normalize else corrected_text), best_conf, meta

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
            from dots_ocr import DotsOCRParser  # type: ignore[import-untyped]
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

        # Convert numpy array to PIL Image for dots.ocr
        from PIL import Image

        pil_image = Image.fromarray(plate_rgb)

        try:
            # Use OCR-only prompt for license plates
            result = self._parser.parse(
                pil_image,
                prompt_mode="prompt_ocr",  # Text extraction only
            )

            # Extract text from result
            text = result.markdown.strip() if hasattr(result, "markdown") else ""
            if not text and hasattr(result, "json"):
                # Fallback: extract from JSON structure
                import json

                data = (
                    json.loads(result.json)
                    if isinstance(result.json, str)
                    else result.json
                )
                # Extract text from layout elements
                texts = []
                for elem in data.get("layout", []):
                    if elem.get("category") == "Text" and elem.get("text"):
                        texts.append(elem["text"])
                text = " ".join(texts).strip()

            if not text:
                return (
                    None,
                    None,
                    {"candidates": 0, "raw_text": None, "raw_confidence": None},
                )

            # dots.ocr doesn't provide confidence scores, use default
            conf = 0.7  # Default confidence for dots.ocr

            if self._allowlist:
                text = "".join(c for c in text if c.upper() in self._allowlist_chars)

            # Remove vignette characters (spurious single chars from circular seals)
            text_no_vignette = _remove_vignette_characters(text)

            # Apply position-based character correction
            corrected_text = _correct_plate_characters_by_position(text_no_vignette)

            normalized_text = (
                _normalize_plate_text(corrected_text)
                if self._normalize
                else corrected_text
            )
            meta: dict[str, object] = {
                "candidates": 1,
                "raw_text": text,
                "corrected_text": corrected_text,
                "raw_confidence": conf,
                "normalized_text": normalized_text if self._normalize else None,
                "preprocess": self._preprocess,
                "allowlist": self._allowlist,
                "engine": "dotsocr",
            }

            if conf < self._min_confidence:
                return None, conf, meta

            return (normalized_text if self._normalize else corrected_text), conf, meta
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
            from chandra.model import InferenceManager  # type: ignore[import-untyped]
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

        # Convert numpy array to PIL Image for Chandra
        from PIL import Image

        pil_image = Image.fromarray(plate_rgb)

        try:
            # Chandra expects a list of images
            results = self._manager.generate([pil_image])

            if not results or not results[0]:
                return (
                    None,
                    None,
                    {"candidates": 0, "raw_text": None, "raw_confidence": None},
                )

            result = results[0]
            text = result.markdown.strip() if hasattr(result, "markdown") else ""

            if not text:
                return (
                    None,
                    None,
                    {"candidates": 0, "raw_text": None, "raw_confidence": None},
                )

            # Chandra doesn't provide confidence scores, use default
            conf = 0.7  # Default confidence for Chandra

            if self._allowlist:
                text = "".join(c for c in text if c.upper() in self._allowlist_chars)

            # Remove vignette characters (spurious single chars from circular seals)
            text_no_vignette = _remove_vignette_characters(text)

            # Apply position-based character correction
            corrected_text = _correct_plate_characters_by_position(text_no_vignette)

            normalized_text = (
                _normalize_plate_text(corrected_text)
                if self._normalize
                else corrected_text
            )
            meta: dict[str, object] = {
                "candidates": 1,
                "raw_text": text,
                "corrected_text": corrected_text,
                "raw_confidence": conf,
                "normalized_text": normalized_text if self._normalize else None,
                "preprocess": self._preprocess,
                "allowlist": self._allowlist,
                "engine": "chandra",
            }

            if conf < self._min_confidence:
                return None, conf, meta

            return (normalized_text if self._normalize else corrected_text), conf, meta
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
            text = result.get("text")
            if not text:
                continue

            score = self._score_result(result, all_results)
            scored_results.append((result, score))

        if not scored_results:
            return None, None, {"ensemble_results": all_results}

        # Sort by score (descending) and pick the best
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_result, best_score = scored_results[0]

        best_text = best_result["text"]
        best_conf = best_result.get("confidence", 0.7)  # Default confidence if missing
        best_meta = {
            **best_result.get("meta", {}),
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
        text = result.get("text", "")
        conf = result.get("confidence")
        engine = result.get("engine", "")

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
        alnum_ratio = sum(1 for c in text if c.isalnum()) / len(text) if text else 0
        quality += alnum_ratio * 0.2

        # Prefer uppercase (German plates are uppercase)
        upper_ratio = (
            sum(1 for c in text if c.isupper()) / sum(1 for c in text if c.isalpha())
            if any(c.isalpha() for c in text)
            else 0
        )
        quality += upper_ratio * 0.1

        return min(1.0, quality)

    def _calculate_consensus(
        self, result: dict[str, object], all_results: list[dict[str, object]]
    ) -> float:
        """Calculate how much other engines agree with this result."""
        text = result.get("text", "")
        if not text:
            return 0.0

        # Normalize text for comparison
        normalized = _normalize_plate_text(text)

        matches = 0
        total = 0

        for other in all_results:
            other_text = other.get("text")
            if not other_text or other == result:
                continue

            total += 1
            other_normalized = _normalize_plate_text(other_text)

            # Exact match
            if normalized == other_normalized:
                matches += 1.0
            # Partial match (75%+ characters match)
            elif len(normalized) > 0 and len(other_normalized) > 0:
                common_chars = sum(1 for c in normalized if c in other_normalized)
                similarity = common_chars / max(len(normalized), len(other_normalized))
                if similarity >= 0.75:
                    matches += 0.5

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
