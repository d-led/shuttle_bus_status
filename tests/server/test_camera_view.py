"""Tests for camera LiveView."""

import pytest
from pyview.live_socket import UnconnectedSocket
from pyview.meta import PyViewMeta

from server.camera_view import (
    build_camera_live_view_dependencies,
    create_camera_live_view,
    load_camera_template_for_tests,
    LiveViewConfiguration,
    RouteDisplaySettings,
    DisplayConfiguration,
)


@pytest.mark.asyncio
async def test_camera_liveview_mount_sets_initial_context() -> None:
    deps = build_camera_live_view_dependencies(
        raw_config={
            "camera": {"device": "auto"},
            "plate_detection": {"poll_interval": 1.0},
            "plate_recognition": {"languages": ["de", "en"]},
            "debouncing": {"appearance_count": 3},
            "logging": {"log_level": "INFO"},
        },
        server_bind="0.0.0.0:8000",
    )
    lv_config = LiveViewConfiguration(
        dependencies=deps,
        route_display=RouteDisplaySettings(title="Shuttle Bus Status", theme="light"),
        display_config=DisplayConfiguration(),
    )
    view = create_camera_live_view(lv_config)()
    socket: UnconnectedSocket[dict[str, object]] = UnconnectedSocket()

    await view.mount(socket, {})

    assert socket.context["title"] == "Shuttle Bus Status"
    assert socket.context["status"] == "ready"
    assert socket.context["camera_connected"] is False
    assert socket.context["plates_detected"] == 0
    assert socket.live_title == "Shuttle Bus Status"
    assert socket.context["server_bind"] == "0.0.0.0:8000"
    assert isinstance(socket.context["config_sections"], list)


@pytest.mark.asyncio
async def test_camera_liveview_render_renders_html() -> None:
    deps = build_camera_live_view_dependencies(
        raw_config={
            "camera": {"width": 1920, "height": 1080},
            "plate_detection": {"model_size": "nano"},
        },
        server_bind="0.0.0.0:8000",
    )
    lv_config = LiveViewConfiguration(
        dependencies=deps,
        route_display=RouteDisplaySettings(title="Shuttle Bus Status", theme="light"),
        display_config=DisplayConfiguration(),
    )
    view = create_camera_live_view(lv_config)()
    socket: UnconnectedSocket[dict[str, object]] = UnconnectedSocket()
    await view.mount(socket, {})

    meta = PyViewMeta(socket=socket)
    rendered = await view.render(socket.context, meta)
    html = rendered.text(socket=socket)

    assert "Camera feed" in html
    assert "Recent activity" in html
    assert "Configuration" in html
    assert "<table" in html


@pytest.mark.asyncio
async def test_camera_liveview_render_shows_connected_state() -> None:
    deps = build_camera_live_view_dependencies(
        raw_config={"camera": {"device": "/dev/video0"}},
        server_bind="0.0.0.0:8000",
    )
    lv_config = LiveViewConfiguration(
        dependencies=deps,
        route_display=RouteDisplaySettings(title="Shuttle Bus Status", theme="light"),
        display_config=DisplayConfiguration(),
    )
    view = create_camera_live_view(lv_config)()
    socket: UnconnectedSocket[dict[str, object]] = UnconnectedSocket()
    await view.mount(socket, {})
    socket.context["camera_connected"] = True

    meta = PyViewMeta(socket=socket)
    rendered = await view.render(socket.context, meta)
    html = rendered.text(socket=socket)

    assert "Connected" in html


def test_camera_template_renders_with_minimal_assigns() -> None:
    template = load_camera_template_for_tests()
    html = template.render(
        {
            "title": "Shuttle Bus Status",
            "status": "ready",
            "camera_connected": False,
            "plates_detected": 0,
            "server_bind": "0.0.0.0:8000",
            "config_sections": [],
        },
        PyViewMeta(socket=UnconnectedSocket()),
    )
    assert "Configuration" in html
