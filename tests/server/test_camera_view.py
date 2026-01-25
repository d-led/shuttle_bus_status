"""Tests for camera LiveView."""

import pytest

from server.camera_view import CameraLiveView, app_frame


def test_app_frame_includes_title() -> None:
    """Test that app_frame includes the provided title."""
    title = "Test Title"
    content = "<p>Test content</p>"
    html = app_frame(title, content)

    assert title in html
    assert f"<title>{title}</title>" in html


def test_app_frame_includes_content() -> None:
    """Test that app_frame includes the provided content."""
    title = "Test"
    content = "<div>Custom content here</div>"
    html = app_frame(title, content)

    assert content in html


def test_app_frame_has_html_structure() -> None:
    """Test that app_frame returns valid HTML structure."""
    html = app_frame("Test", "<p>Content</p>")

    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "<head>" in html
    assert "<body>" in html
    assert "</html>" in html


def test_app_frame_includes_styles() -> None:
    """Test that app_frame includes CSS styles."""
    html = app_frame("Test", "<p>Content</p>")

    assert "<style>" in html
    assert "body" in html
    assert "font-family" in html


@pytest.mark.asyncio
async def test_camera_liveview_mount_returns_initial_state() -> None:
    """Test that mount returns correct initial state."""
    view = CameraLiveView()
    state = await view.mount(None, {})

    assert state["title"] == "Shuttle Bus Status"
    assert state["status"] == "ready"
    assert state["camera_connected"] is False
    assert state["plates_detected"] == 0
    assert state["last_update"] is None


@pytest.mark.asyncio
async def test_camera_liveview_render_with_default_state() -> None:
    """Test that render produces HTML with default state."""
    view = CameraLiveView()
    state = await view.mount(None, {})
    html = await view.render(state)

    assert "Shuttle Bus Status" in html
    assert "Ready" in html
    assert "Not Connected" in html
    assert "Plates Detected" in html
    assert "0" in html


@pytest.mark.asyncio
async def test_camera_liveview_render_with_camera_connected() -> None:
    """Test that render shows connected status when camera is connected."""
    view = CameraLiveView()
    state = await view.mount(None, {})
    state["camera_connected"] = True
    html = await view.render(state)

    assert "Connected" in html
    assert "status-indicator" in html


@pytest.mark.asyncio
async def test_camera_liveview_render_with_plates_detected() -> None:
    """Test that render shows correct plate count."""
    view = CameraLiveView()
    state = await view.mount(None, {})
    state["plates_detected"] = 5
    html = await view.render(state)

    assert "5" in html
    assert "Plates Detected" in html


@pytest.mark.asyncio
async def test_camera_liveview_render_with_initializing_status() -> None:
    """Test that render shows initializing status correctly."""
    view = CameraLiveView()
    state = await view.mount(None, {})
    state["status"] = "initializing"
    html = await view.render(state)

    assert "Initializing..." in html
    assert "Ready" not in html


@pytest.mark.asyncio
async def test_camera_liveview_render_includes_camera_feed_section() -> None:
    """Test that render includes camera feed placeholder."""
    view = CameraLiveView()
    state = await view.mount(None, {})
    html = await view.render(state)

    assert "Camera Feed" in html
    assert "Camera feed will appear here" in html
    assert "📹" in html


@pytest.mark.asyncio
async def test_camera_liveview_render_includes_activity_section() -> None:
    """Test that render includes recent activity section."""
    view = CameraLiveView()
    state = await view.mount(None, {})
    html = await view.render(state)

    assert "Recent Activity" in html
    assert "Plate detection events will appear here" in html
    assert "📋" in html


@pytest.mark.asyncio
async def test_camera_liveview_render_handles_missing_state_keys() -> None:
    """Test that render handles missing state keys gracefully."""
    view = CameraLiveView()
    # Use minimal state without all keys
    state = {"title": "Test"}
    html = await view.render(state)

    # Should not crash and should use defaults
    assert "Test" in html
    assert "0" in html  # Default plates_detected
