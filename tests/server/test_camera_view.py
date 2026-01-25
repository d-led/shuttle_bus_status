"""Tests for camera LiveView."""

import pytest
from pyview.live_socket import UnconnectedSocket
from pyview.meta import PyViewMeta

from server.camera_view import CameraLiveView


@pytest.mark.asyncio
async def test_camera_liveview_mount_sets_initial_context() -> None:
    view = CameraLiveView()
    socket: UnconnectedSocket[dict[str, object]] = UnconnectedSocket()

    await view.mount(socket, {})

    assert socket.context["title"] == "Shuttle Bus Status"
    assert socket.context["status"] == "ready"
    assert socket.context["camera_connected"] is False
    assert socket.context["plates_detected"] == 0
    assert socket.live_title == "Shuttle Bus Status"


@pytest.mark.asyncio
async def test_camera_liveview_render_renders_html() -> None:
    view = CameraLiveView()
    socket: UnconnectedSocket[dict[str, object]] = UnconnectedSocket()
    await view.mount(socket, {})

    meta = PyViewMeta(socket=socket)
    rendered = await view.render(socket.context, meta)
    html = rendered.text(socket=socket)

    assert "System Status" in html
    assert "Camera Feed" in html
    assert "Recent Activity" in html


@pytest.mark.asyncio
async def test_camera_liveview_render_shows_connected_state() -> None:
    view = CameraLiveView()
    socket: UnconnectedSocket[dict[str, object]] = UnconnectedSocket()
    await view.mount(socket, {})
    socket.context["camera_connected"] = True

    meta = PyViewMeta(socket=socket)
    rendered = await view.render(socket.context, meta)
    html = rendered.text(socket=socket)

    assert "Connected" in html
