"""Tests for ensemble OCR recognition."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from camera.plate_pipeline import (
    EnsemblePlateRecognizer,
    create_plate_recognizer_from_config,
)


@pytest.mark.integration
def test_ensemble_ocr_single_engine_fallback() -> None:
    """Test that ensemble with single engine works like regular recognizer."""
    # Create ensemble with just one engine
    ensemble = EnsemblePlateRecognizer(
        engines=["easyocr"],
        languages=["en"],
        min_confidence=0.1,
    )

    # Create regular recognizer
    regular = create_plate_recognizer(
        engine="easyocr",
        languages=["en"],
        min_confidence=0.1,
    )

    # Create a simple test image with text
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.putText(
        img, "ABC123", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    # Both should work similarly
    ensemble_text, ensemble_conf, ensemble_meta = ensemble.recognize(img)
    regular_text, regular_conf, regular_meta = regular.recognize(img)

    # Results should be similar (ensemble might have extra metadata)
    assert ensemble_text == regular_text or (
        ensemble_text is None and regular_text is None
    )
    assert "ensemble_engine" in ensemble_meta
    assert ensemble_meta["ensemble_engine"] == "easyocr"


@pytest.mark.integration
def test_ensemble_ocr_multiple_engines() -> None:
    """Test ensemble with multiple engines selects best result."""
    # Create ensemble with multiple engines (only easyocr and tesseract for testing)
    ensemble = EnsemblePlateRecognizer(
        engines=["easyocr", "tesseract"],
        languages=["en"],
        min_confidence=0.1,
    )

    # Create a simple test image
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.putText(
        img, "ABC123", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    text, conf, meta = ensemble.recognize(img)

    # Should have ensemble metadata
    assert "ensemble_engine" in meta
    assert "ensemble_results" in meta
    assert isinstance(meta["ensemble_results"], list)
    assert len(meta["ensemble_results"]) == 2  # Two engines

    # Check that results from both engines are stored
    engines_used = {r["engine"] for r in meta["ensemble_results"]}
    assert "easyocr" in engines_used
    assert "tesseract" in engines_used


@pytest.mark.integration
def test_ensemble_ocr_handles_failures() -> None:
    """Test that ensemble continues if one engine fails."""
    # Create ensemble with valid and invalid engines
    ensemble = EnsemblePlateRecognizer(
        engines=["easyocr", "nonexistent_engine"],
        languages=["en"],
        min_confidence=0.1,
    )

    # Should still work with the valid engine
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.putText(
        img, "ABC123", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    # Should not crash, but may return None if all engines fail
    text, conf, meta = ensemble.recognize(img)

    # Should have metadata about what happened
    assert "ensemble_results" in meta


def test_ensemble_scoring_logic() -> None:
    """Test the scoring logic for selecting best result."""
    ensemble = EnsemblePlateRecognizer(
        engines=["easyocr"],
        languages=["en"],
        min_confidence=0.1,
    )

    # Test text quality assessment
    assert ensemble._assess_text_quality("ABC123") > 0.8  # Good plate (6 chars)
    # AB12 is 4 chars - gets base 0.5 + 0.1 (length 3-12) + alnum bonus = ~0.8-0.9
    # But it's still lower quality than a proper 6-char plate
    assert ensemble._assess_text_quality("AB12") < ensemble._assess_text_quality(
        "ABC123"
    )
    # Long strings (12 chars) still get decent quality but not perfect
    assert ensemble._assess_text_quality("ABC123XYZ456") <= 0.95  # Too long (12 chars)

    # Test engine reliability
    assert ensemble._get_engine_reliability("dotsocr") > 0.9
    assert ensemble._get_engine_reliability("easyocr") > 0.6
    assert ensemble._get_engine_reliability("unknown") == 0.5

    # Test consensus calculation
    result1 = {"text": "ABC123", "engine": "easyocr"}
    result2 = {"text": "ABC123", "engine": "tesseract"}
    result3 = {"text": "ABC123", "engine": "paddleocr"}

    consensus = ensemble._calculate_consensus(result1, [result1, result2, result3])
    assert consensus > 0.5  # Should have good consensus

    # Test with different results
    result4 = {"text": "XYZ789", "engine": "other"}
    consensus2 = ensemble._calculate_consensus(result1, [result1, result2, result4])
    assert consensus2 < consensus  # Less consensus with different result


@pytest.mark.integration
def test_create_plate_recognizer_from_config_single() -> None:
    """Test config-based creation with single engine."""
    ocr = create_plate_recognizer_from_config(
        ocr_engine="easyocr",
        languages=["en"],
        min_confidence=0.1,
    )

    # Should create regular recognizer, not ensemble
    assert not isinstance(ocr, EnsemblePlateRecognizer)

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.putText(
        img, "ABC123", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    text, conf, meta = ocr.recognize(img)
    # Should work normally
    assert "ensemble" not in meta


@pytest.mark.integration
def test_create_plate_recognizer_from_config_ensemble() -> None:
    """Test config-based creation with ensemble."""
    ocr = create_plate_recognizer_from_config(
        ocr_engine=["easyocr", "tesseract"],
        languages=["en"],
        min_confidence=0.1,
    )

    # Should create ensemble
    assert isinstance(ocr, EnsemblePlateRecognizer)

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.putText(
        img, "ABC123", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    text, conf, meta = ocr.recognize(img)
    # Should have ensemble metadata
    assert "ensemble_engine" in meta
    assert "ensemble_results" in meta
