"""Tests for structured logging."""

import json
import tempfile
from pathlib import Path

from camera.logger import configure_from_settings, get_logger, setup_logging


def test_setup_logging_console() -> None:
    """Test setting up console logging."""
    setup_logging(log_level="INFO", log_to_console=True)
    logger = get_logger("test")
    logger.info("test message", key="value")


def test_setup_logging_file() -> None:
    """Test setting up file logging with JSON output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_file = Path(f.name)

    try:
        setup_logging(log_level="INFO", log_file=str(log_file), log_to_console=False)
        logger = get_logger("test")
        logger.info("test message", key="value", number=42)

        # Read and verify log file
        log_content = log_file.read_text()
        assert "test message" in log_content
        assert "key" in log_content
        assert "value" in log_content

        # Verify it's valid JSON (one line per log entry)
        lines = log_content.strip().split("\n")
        for line in lines:
            if line.strip():
                log_entry = json.loads(line)
                assert "timestamp" in log_entry
                assert "event" in log_entry or "message" in log_entry
                # Verify RFC 3339 format: YYYY-MM-DDTHH:MM:SS.mmmZ
                timestamp = log_entry["timestamp"]
                assert timestamp.endswith("Z")
                assert "T" in timestamp
                assert "." in timestamp  # Millisecond precision
    finally:
        if log_file.exists():
            log_file.unlink()


def test_rfc3339_timestamp_format() -> None:
    """Test that timestamps are in RFC 3339 format with milliseconds."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_file = Path(f.name)

    try:
        setup_logging(log_level="INFO", log_file=str(log_file), log_to_console=False)
        logger = get_logger("test")
        logger.info("timestamp test")

        # Read log file
        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())

        # Verify timestamp format
        timestamp = log_entry["timestamp"]
        # RFC 3339 format: 2024-01-15T10:30:45.123Z
        assert timestamp.endswith("Z")
        assert "T" in timestamp
        parts = timestamp.replace("Z", "").split("T")
        assert len(parts) == 2
        date_part, time_part = parts
        # Date: YYYY-MM-DD
        assert len(date_part.split("-")) == 3
        # Time: HH:MM:SS.mmm
        assert "." in time_part
        time_without_ms = time_part.split(".")[0]
        assert len(time_without_ms.split(":")) == 3
    finally:
        if log_file.exists():
            log_file.unlink()


def test_configure_from_settings() -> None:
    """Test configuring logging from Settings object."""

    # Mock settings object
    class MockLoggingSettings:
        log_file = "logs/test.log"
        log_level = "DEBUG"
        log_to_console = True

    class MockSettings:
        logging = MockLoggingSettings()

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        settings = MockSettings()
        settings.logging.log_file = str(log_file)

        configure_from_settings(settings)
        logger = get_logger("test")
        logger.info("configured from settings", test=True)

        # Verify log file was created
        assert log_file.exists()


def test_logger_context() -> None:
    """Test that logger can add context."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_file = Path(f.name)

    try:
        setup_logging(log_level="INFO", log_file=str(log_file), log_to_console=False)
        logger = get_logger("test")
        logger = logger.bind(component="camera", version="1.0")
        logger.info("contextual message", extra_field="value")

        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())
        assert log_entry.get("component") == "camera"
        assert log_entry.get("version") == "1.0"
        assert log_entry.get("extra_field") == "value"
    finally:
        if log_file.exists():
            log_file.unlink()
