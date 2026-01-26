"""Tests for configuration management."""

from pathlib import Path

import pytest

from camera.config import Settings


def test_load_config_from_project_root(tmp_path: Path) -> None:
    """Test loading configuration from project root."""
    # Create a config.toml file
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[camera]
device = "/dev/video1"
width = 1280
height = 720

[plate_detection]
model_size = "small"
confidence_threshold = 0.7
""")

    # Change to the temp directory
    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        settings = Settings.load_from_project_root()
        assert settings.camera.device == "/dev/video1"
        assert settings.camera.width == 1280
        assert settings.camera.height == 720
        assert settings.plate_detection.model_size == "small"
        assert settings.plate_detection.confidence_threshold == 0.7
    finally:
        os.chdir(original_cwd)


def test_auto_detect_camera_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test auto-detection of camera device on Linux uses /dev/video*."""
    settings = Settings()
    settings.camera.device = "auto"

    # Mock glob to return test devices
    def mock_glob(pattern: str) -> list[str]:
        if pattern == "/dev/video*":
            return ["/dev/video0", "/dev/video1", "/dev/video10"]
        return []

    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("camera.config.glob.glob", mock_glob)

    device = settings.get_camera_device()
    assert device == "/dev/video0"


def test_auto_detect_camera_device_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test auto-detection of camera device on macOS uses AVFoundation."""
    settings = Settings()
    settings.camera.device = "auto"

    # Ensure we do not accidentally rely on /dev/video* on macOS.
    def _should_not_be_called(_pattern: str) -> list[str]:
        raise AssertionError("glob.glob should not be used on macOS")

    import sys

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("camera.config.glob.glob", _should_not_be_called)

    assert settings.get_camera_device() == "avfoundation:0"


def test_specific_camera_device() -> None:
    """Test using a specific camera device."""
    settings = Settings()
    settings.camera.device = "/dev/video2"

    device = settings.get_camera_device()
    assert device == "/dev/video2"


def test_auto_resolution(tmp_path: Path) -> None:
    """Test auto resolution detection."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[camera]
device = "auto"
width = "auto"
height = "auto"
capture_fps = 30
""")

    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        settings = Settings.load_from_project_root()
        assert settings.camera.width == "auto"
        assert settings.camera.height == "auto"
    finally:
        os.chdir(original_cwd)


def test_mixed_resolution(tmp_path: Path) -> None:
    """Test mixed auto and specific resolution."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[camera]
device = "auto"
width = "auto"
height = 720
capture_fps = 30
""")

    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        settings = Settings.load_from_project_root()
        assert settings.camera.width == "auto"
        assert settings.camera.height == 720
    finally:
        os.chdir(original_cwd)


def test_default_settings() -> None:
    """Test that default settings are applied correctly."""
    settings = Settings()

    assert settings.camera.device == "auto"
    assert settings.camera.width == 1920
    assert settings.camera.height == 1080
    assert settings.camera.capture_fps == 30
    assert settings.plate_detection.model_size == "nano"
    assert settings.plate_detection.confidence_threshold == 0.5
    assert settings.debouncing.appearance_count == 3
    assert settings.debouncing.appearance_window == 2.0
    assert settings.debouncing.disappearance_timeout == 5.0
    assert settings.logging.log_plates == "file"


def test_legacy_fps_is_migrated_to_capture_fps(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[camera]
fps = 25
""")
    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        settings = Settings.load_from_project_root()
        assert settings.camera.capture_fps == 25
    finally:
        os.chdir(original_cwd)
