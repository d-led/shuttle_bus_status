"""Unit tests for template parsing - catch template errors before server startup."""

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


class TestTemplateParsing:
    """Test that the camera_view.html template can be parsed and rendered."""

    def _load_template(self) -> ibis.Template:
        """Load the camera_view.html template using the same method as CameraLiveView.

        Returns the underlying ibis.Template directly for testing.
        """
        current_file_path = Path(__file__).resolve()
        templates_dir = (
            current_file_path.parent.parent.parent / "server" / "src" / "server"
        )

        if not hasattr(ibis, "loader") or not isinstance(ibis.loader, FileReloader):
            ibis.loader = FileReloader(str(templates_dir))

        template_file = templates_dir / "camera_view.html"
        template_content = template_file.read_text(encoding="utf-8")

        return ibis.Template(template_content, template_id=str(template_file))

    def _create_test_live_view(self) -> type:
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

    def _create_minimal_template_assigns(self) -> dict:
        """Create minimal template assigns for testing."""
        view_class = self._create_test_live_view()
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

        return view._build_template_assigns(template_data)

    def test_template_can_be_loaded(self) -> None:
        """Test that the template file can be loaded without errors."""
        template = self._load_template()
        assert template is not None

    def test_template_can_be_rendered_with_minimal_data(self) -> None:
        """Test that the template can be rendered with minimal data."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        # Render the template - this will raise an exception if there are syntax errors
        result = template.render(**assigns)

        # Verify we got some HTML output
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

        # Verify key elements are present
        assert "<html" in result.lower() or "<div" in result.lower()

    def test_template_renders_with_empty_detections(self) -> None:
        """Test that the template renders correctly with no detections."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        # Ensure empty state
        assigns["latest_detections"] = []
        assigns["plates_detected"] = 0

        # Should render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)
        assert "No plates detected yet" in result

    def test_template_renders_with_single_detection(self) -> None:
        """Test that the template renders correctly with a single detection."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        detection = {
            "text": "ABC123",
            "raw_text": "ABC123",
            "raw_ocr_confidence": 0.85,
        }

        assigns["latest_detections"] = [detection]
        assigns["plates_detected"] = 1

        # Should render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ABC123" in result

    def test_template_renders_with_multiple_detections(self) -> None:
        """Test that the template renders correctly with multiple detections."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        detections = [
            {"text": "ABC123", "raw_text": "ABC123", "raw_ocr_confidence": 0.85},
            {"text": "XYZ789", "raw_text": "XYZ789", "raw_ocr_confidence": 0.92},
            {"text": "DEF456", "raw_text": "DEF456", "raw_ocr_confidence": 0.78},
        ]

        assigns["latest_detections"] = detections
        assigns["plates_detected"] = 3

        # Should render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ABC123" in result
        assert "XYZ789" in result
        assert "DEF456" in result

    def test_template_renders_with_detection_confidence(self) -> None:
        """Test that the template correctly displays detection confidence."""
        view_class = self._create_test_live_view()
        view = view_class()
        template = self._load_template()

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

        # Should render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)
        # Should show confidence percentage (processed by _build_template_assigns)
        assert "85" in result or "85%" in result

    def test_template_handles_missing_optional_fields(self) -> None:
        """Test that the template handles missing optional fields gracefully."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        # Detection without raw_text or confidence
        detection = {"text": "ABC123"}

        assigns["latest_detections"] = [detection]

        # Should still render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)

    def test_template_with_camera_connected(self) -> None:
        """Test that the template renders with camera connected state."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        assigns["camera_connected"] = True
        assigns["camera_frame_b64"] = "fake_base64_string"

        # Should render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)
        assert "data:image/jpeg;base64" in result

    def test_template_with_camera_disconnected(self) -> None:
        """Test that the template renders with camera disconnected state."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        assigns["camera_connected"] = False
        assigns["camera_frame_b64"] = None

        # Should render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)
        assert "Waiting for camera sample" in result or "camera" in result.lower()

    def test_template_with_detection_raw_text_different(self) -> None:
        """Test that the template shows raw text when different from processed text."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        detection = {
            "text": "ABC123",
            "raw_text": "ABC 123",  # Different from processed
            "raw_ocr_confidence": 0.85,
        }

        assigns["latest_detections"] = [detection]

        # Should render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)
        assert "ABC123" in result
        assert "ABC 123" in result or "Raw:" in result

    def test_template_with_detection_no_text(self) -> None:
        """Test that the template handles detection with None text."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        detection = {
            "text": None,
            "raw_text": None,
            "raw_ocr_confidence": 0.5,
        }

        assigns["latest_detections"] = [detection]

        # Should render without errors (shows "Unknown")
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)
        assert "Unknown" in result

    def test_build_template_assigns_includes_required_keys(self) -> None:
        """Test that _build_template_assigns includes all required keys."""
        view_class = self._create_test_live_view()
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

        # Verify all required keys are present
        required_keys = [
            "title",
            "status",
            "camera_connected",
            "plates_detected",
            "camera_frame_b64",
            "camera_device",
            "camera_device_options",
            "server_bind",
            "feedback_lines",
            "config_sections",
            "latest_detections",
        ]

        for key in required_keys:
            assert key in assigns, f"Missing required key: {key}"

    def test_template_with_all_config_options(self) -> None:
        """Test that the template renders with various configuration options."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        # Add config sections
        assigns["config_sections"] = [
            {
                "section": "Camera",
                "rows": [
                    {"key": "device", "value": "test:data/test_images"},
                    {"key": "width", "value": "1920"},
                ],
            }
        ]

        # Should render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)

    def test_template_with_special_characters_in_plate_text(self) -> None:
        """Test that the template handles special characters in plate text."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        detection = {
            "text": "ABC-123",
            "raw_text": "ABC-123",
            "raw_ocr_confidence": 0.85,
        }

        assigns["latest_detections"] = [detection]

        # Should render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)
