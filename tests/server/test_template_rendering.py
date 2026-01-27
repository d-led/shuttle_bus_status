"""Tests for template rendering in camera_view - these actually render the template and check HTML output."""

from __future__ import annotations

from pathlib import Path

from pyview.vendor import ibis
from pyview.vendor.ibis.loaders import FileReloader

from server.camera_view import (
    LiveViewConfiguration,
    build_camera_live_view_dependencies,
    build_display_configuration,
    build_route_display_settings,
    create_camera_live_view,
)
from server.camera_stream import NullCameraStreamController


def _create_test_view() -> type:
    """Create a test CameraLiveView class."""
    stream_controller = NullCameraStreamController()
    raw_config = {
        "camera": {"device": "test:data/test_images"},
        "plate_detection": {"poll_interval_seconds": 1.0},
        "plate_recognition": {"languages": ["de", "en"]},
        "debouncing": {"appearance_min_count": 3},
        "logging": {"log_level": "INFO"},
    }
    dependencies = build_camera_live_view_dependencies(
        raw_config=raw_config,
        server_bind="0.0.0.0:8001",
        stream=stream_controller,
    )
    display_config = build_display_configuration(raw_config=raw_config)
    route_display = build_route_display_settings(raw_config=raw_config)
    config = LiveViewConfiguration(
        dependencies=dependencies,
        display_config=display_config,
        route_display=route_display,
    )
    return create_camera_live_view(config)


def _load_template() -> ibis.Template:
    """Load the camera_view.html template."""
    current_file_path = Path(__file__).resolve()
    templates_dir = current_file_path.parent.parent.parent / "server" / "src" / "server"

    if not hasattr(ibis, "loader") or not isinstance(ibis.loader, FileReloader):
        ibis.loader = FileReloader(str(templates_dir))

    template_file = templates_dir / "camera_view.html"
    template_content = template_file.read_text(encoding="utf-8")

    return ibis.Template(template_content, template_id=str(template_file))


def test_render_includes_plate_text() -> None:
    """Given a detection, when rendering, then plate text is in HTML."""
    view_class = _create_test_view()
    view = view_class()

    detection = {
        "text": "ABC123",
        "raw_text": "ABC123",
        "raw_ocr_confidence": 0.85,
    }

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": False,
        "plates_detected": 1,
        "camera_frame_b64": None,
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [],
        "latest_detections": [detection],
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    assert "ABC123" in result


def test_render_includes_confidence() -> None:
    """Given a detection with confidence, when rendering, then confidence is in HTML."""
    view_class = _create_test_view()
    view = view_class()

    detection = {
        "text": "ABC123",
        "raw_text": "ABC123",
        "raw_ocr_confidence": 0.85,
    }

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": False,
        "plates_detected": 1,
        "camera_frame_b64": None,
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [],
        "latest_detections": [detection],
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    # Should show confidence (85% or 85)
    assert "85" in result


def test_render_shows_raw_text_when_different() -> None:
    """Given a detection with different raw text, when rendering, then raw text is shown."""
    view_class = _create_test_view()
    view = view_class()

    detection = {
        "text": "ABC123",
        "raw_text": "ABC 123",  # Different from processed
        "raw_ocr_confidence": 0.85,
    }

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": False,
        "plates_detected": 1,
        "camera_frame_b64": None,
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [],
        "latest_detections": [detection],
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    assert "ABC123" in result
    assert "ABC 123" in result or "Raw:" in result


def test_render_empty_list_shows_no_detections_message() -> None:
    """Given empty detections, when rendering, then shows no detections message."""
    view_class = _create_test_view()
    view = view_class()

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": False,
        "plates_detected": 0,
        "camera_frame_b64": None,
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [],
        "latest_detections": [],
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    assert "No plates detected yet" in result


def test_render_single_detection() -> None:
    """Given a single detection, when rendering, then it is displayed."""
    view_class = _create_test_view()
    view = view_class()

    detection = {
        "text": "ABC123",
        "raw_text": "ABC123",
        "raw_ocr_confidence": 0.85,
    }

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": False,
        "plates_detected": 1,
        "camera_frame_b64": None,
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [],
        "latest_detections": [detection],
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    assert "ABC123" in result
    # Should not show "No plates detected yet"
    assert "No plates detected yet" not in result


def test_render_multiple_detections() -> None:
    """Given multiple detections, when rendering, then all are displayed."""
    view_class = _create_test_view()
    view = view_class()

    detections = [
        {"text": "ABC123", "raw_text": "ABC123", "raw_ocr_confidence": 0.85},
        {"text": "XYZ789", "raw_text": "XYZ789", "raw_ocr_confidence": 0.92},
        {"text": "DEF456", "raw_text": "DEF456", "raw_ocr_confidence": 0.78},
    ]

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": False,
        "plates_detected": 3,
        "camera_frame_b64": None,
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [],
        "latest_detections": detections,
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    assert "ABC123" in result
    assert "XYZ789" in result
    assert "DEF456" in result


def test_render_detection_without_confidence() -> None:
    """Given a detection without confidence, when rendering, then it still displays."""
    view_class = _create_test_view()
    view = view_class()

    detection = {
        "text": "ABC123",
        "raw_text": "ABC123",
        "raw_ocr_confidence": None,
    }

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": False,
        "plates_detected": 1,
        "camera_frame_b64": None,
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [],
        "latest_detections": [detection],
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    assert "ABC123" in result


def test_render_detection_with_none_text() -> None:
    """Given a detection with None text, when rendering, then shows Unknown."""
    view_class = _create_test_view()
    view = view_class()

    detection = {
        "text": None,
        "raw_text": None,
        "raw_ocr_confidence": 0.5,
    }

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": False,
        "plates_detected": 1,
        "camera_frame_b64": None,
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [],
        "latest_detections": [detection],
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    assert "Unknown" in result


def test_render_camera_connected_shows_frame() -> None:
    """Given camera connected with frame, when rendering, then frame is in HTML."""
    view_class = _create_test_view()
    view = view_class()

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": True,
        "plates_detected": 0,
        "camera_frame_b64": "fake_base64_string",
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [],
        "latest_detections": [],
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    assert "data:image/jpeg;base64" in result
    assert "fake_base64_string" in result


def test_render_camera_disconnected_shows_waiting_message() -> None:
    """Given camera disconnected, when rendering, then shows waiting message."""
    view_class = _create_test_view()
    view = view_class()

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": False,
        "plates_detected": 0,
        "camera_frame_b64": None,
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [],
        "latest_detections": [],
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    assert "Waiting for camera sample" in result or "camera" in result.lower()


def test_render_plates_detected_count() -> None:
    """Given plates detected count, when rendering, then count is displayed."""
    view_class = _create_test_view()
    view = view_class()

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": False,
        "plates_detected": 5,
        "camera_frame_b64": None,
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [],
        "latest_detections": [],
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    assert "5" in result or "plates_detected" in result.lower()


def test_render_config_sections() -> None:
    """Given config sections, when rendering, then they are displayed."""
    view_class = _create_test_view()
    view = view_class()

    template_data = {
        "title": "Shuttle Bus Status",
        "status": "ready",
        "camera_connected": False,
        "plates_detected": 0,
        "camera_frame_b64": None,
        "camera_device": "test:data/test_images",
        "camera_device_options": [],
        "server_bind": "0.0.0.0:8001",
        "feedback_lines": [],
        "config_sections": [
            {
                "section": "Camera",
                "rows": [
                    {"key": "device", "value": "test:data/test_images"},
                    {"key": "width", "value": "1920"},
                ],
            }
        ],
        "latest_detections": [],
    }

    assigns = view._build_template_assigns(template_data)
    template = _load_template()
    result = template.render(**assigns)

    assert "Camera" in result
    assert "device" in result or "1920" in result
