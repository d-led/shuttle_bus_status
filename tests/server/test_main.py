"""Tests for server main module."""

from unittest.mock import MagicMock, patch

import pytest
from pyview.live_socket import UnconnectedSocket
from pyview.meta import PyViewMeta

from server.main import (
    IndexLiveView,
    create_app,
    create_camera_app,
    main,
    main_camera,
)


@pytest.mark.asyncio
async def test_index_liveview_mount() -> None:
    """Test that IndexLiveView mount returns correct state."""
    view = IndexLiveView()
    socket: UnconnectedSocket[dict[str, object]] = UnconnectedSocket()
    await view.mount(socket, {})

    assert socket.context["title"] == "Shuttle Bus Status"
    assert socket.live_title == "Shuttle Bus Status"


@pytest.mark.asyncio
async def test_index_liveview_render() -> None:
    """Test that IndexLiveView render produces HTML."""
    view = IndexLiveView()
    socket: UnconnectedSocket[dict[str, object]] = UnconnectedSocket()
    await view.mount(socket, {})
    meta = PyViewMeta(socket=socket)
    rendered = await view.render(socket.context, meta)
    html = rendered.text(socket=socket)

    assert "Shuttle Bus Status" in html


def test_create_app_returns_starlette_app() -> None:
    """Test that create_app returns an application."""
    app = create_app()

    assert app is not None
    # Check that it has routes
    assert len(app.routes) > 0


def test_create_app_has_index_route() -> None:
    """Test that create_app includes index route."""
    app = create_app()

    # Check routes
    route_paths = [route.path for route in app.routes]
    assert "/" in route_paths


def test_create_camera_app_returns_starlette_app() -> None:
    """Test that create_camera_app returns an application."""
    app = create_camera_app()

    assert app is not None
    assert len(app.routes) > 0


def test_create_camera_app_has_root_route() -> None:
    """Test that create_camera_app includes root route."""
    app = create_camera_app()

    route_paths = [route.path for route in app.routes]
    assert "/" in route_paths


def test_create_camera_app_mounts_static_assets() -> None:
    """Camera app must serve PyView client JS under /static/assets/app.js."""
    app = create_camera_app()
    route_paths = [getattr(route, "path", None) for route in app.routes]
    assert "/static" in route_paths
    assert "/static/assets/app.js" in route_paths


def test_create_app_mounts_static_assets() -> None:
    """Main app must serve PyView client JS under /static/assets/app.js."""
    app = create_app()
    route_paths = [getattr(route, "path", None) for route in app.routes]
    assert "/static" in route_paths
    assert "/static/assets/app.js" in route_paths


@patch("server.main.uvicorn.run")
@patch("server.main.Settings.load_from_project_root")
def test_main_loads_server_config(
    mock_load_settings: MagicMock, mock_run: MagicMock
) -> None:
    """Test that main uses server config from settings."""
    # Create mock settings
    mock_settings = MagicMock()
    mock_settings.server.host = "127.0.0.1"
    mock_settings.server.port = 9000
    mock_settings.server.debug = True
    mock_load_settings.return_value = mock_settings

    main()

    # Verify settings were loaded
    mock_load_settings.assert_called_once()

    # Verify uvicorn.run was called with correct parameters
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[1]["host"] == "127.0.0.1"
    assert call_args[1]["port"] == 9000
    assert call_args[1]["log_level"] == "debug"


@patch("server.main.uvicorn.run")
@patch("server.main.load_raw_config_from_project_root")
@patch("server.main.Settings.load_from_project_root")
def test_main_camera_loads_server_config(
    mock_load_settings: MagicMock, mock_load_raw: MagicMock, mock_run: MagicMock
) -> None:
    """Test that main_camera uses server config from settings."""
    # Create mock settings
    mock_settings = MagicMock()
    mock_settings.server.host = "0.0.0.0"
    mock_settings.server.port = 8080
    mock_settings.server.debug = False
    mock_load_settings.return_value = mock_settings
    mock_load_raw.return_value = {}

    main_camera()

    # Verify settings were loaded
    mock_load_settings.assert_called_once()

    # Verify uvicorn.run was called with server config
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[1]["host"] == "0.0.0.0"
    assert call_args[1]["port"] == 8080
    assert call_args[1]["log_level"] == "info"


@patch("server.main.uvicorn.run")
@patch("server.main.load_raw_config_from_project_root")
@patch("server.main.Settings.load_from_project_root")
def test_main_camera_uses_debug_log_level_when_enabled(
    mock_load_settings: MagicMock, mock_load_raw: MagicMock, mock_run: MagicMock
) -> None:
    """Test that main_camera uses debug log level when debug is enabled."""
    mock_settings = MagicMock()
    mock_settings.server.host = "0.0.0.0"
    mock_settings.server.port = 8000
    mock_settings.server.debug = True
    mock_load_settings.return_value = mock_settings
    mock_load_raw.return_value = {}

    main_camera()

    call_args = mock_run.call_args
    assert call_args[1]["log_level"] == "debug"
