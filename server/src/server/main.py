"""Main entry point for the server."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
from pyview import LiveView
from pyview.live_view import LiveRender, LiveTemplate
from pyview.pyview import PyView
from pyview.template import Template
from starlette.responses import FileResponse, Response
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

from server.camera_stream import (
    CameraStreamConfig,
    CameraStreamController,
    format_camera_device_options_for_ui,
    resolve_device,
)
from server.camera_view import (
    LiveViewConfiguration,
    build_camera_live_view_dependencies,
    build_display_configuration,
    build_route_display_settings,
    create_camera_live_view,
)
from server.config import Settings, load_raw_config_from_project_root
from server.root_template import camera_root_template

if TYPE_CHECKING:
    from pyview.meta import PyViewMeta


def _register_pyview_static(app: PyView) -> None:
    """Register PyView's client assets like MVG does.

    PyView's default/root templates reference `/static/assets/app.js`.
    """
    import pyview

    pyview_path = Path(pyview.__file__).resolve().parent
    pyview_static = pyview_path / "static"
    if not pyview_static.exists():
        raise RuntimeError(f"pyview static directory not found at {pyview_static}")

    async def _serve_app_js(_request: Any) -> Response:
        client_js_path = pyview_path / "static" / "assets" / "app.js"
        if client_js_path.exists():
            return FileResponse(
                str(client_js_path),
                media_type="application/javascript",
            )

        alt_path = pyview_path / "assets" / "js" / "app.js"
        if alt_path.exists():
            return FileResponse(
                str(alt_path),
                media_type="application/javascript",
            )

        return Response(
            content="// PyView client not found",
            media_type="application/javascript",
            status_code=404,
        )

    # IMPORTANT: register specific routes before mounting `/static`
    app.routes.insert(0, Route("/static/assets/app.js", _serve_app_js))
    app.mount("/static", StaticFiles(directory=str(pyview_static)), name="static")


class IndexLiveView(LiveView):
    """Main index page live view."""

    async def mount(self, socket: Any, session: Any) -> None:
        """Initialize the live view."""
        del session
        socket.context = {"title": "Shuttle Bus Status"}
        socket.live_title = "Shuttle Bus Status"

    async def render(self, assigns: dict[str, Any], meta: PyViewMeta) -> LiveRender:
        template = LiveTemplate(
            Template(
                "<div><h1>{{ title }}</h1><p>Shuttle bus status monitoring system</p></div>",
                template_id="server.index",
            )
        )
        return LiveRender(template, assigns, meta)


def create_app() -> PyView:
    """Create and configure the main server application."""
    app = PyView()
    _register_pyview_static(app)
    app.add_live_view("/", IndexLiveView)
    return app


def create_camera_app(
    *, settings: Settings | None = None, raw_config: dict[str, Any] | None = None
) -> PyView:
    """Create and configure the camera server application."""
    if settings is None:
        settings = Settings.load_from_project_root()
    if raw_config is None:
        raw_config = load_raw_config_from_project_root()

    _log_camera_devices(raw_config)

    stream = _build_camera_stream(raw_config)
    deps = build_camera_live_view_dependencies(
        raw_config=raw_config,
        server_bind=f"{settings.server.host}:{settings.server.port}",
        stream=stream,
    )
    route_display = build_route_display_settings(raw_config=raw_config)
    display_config = build_display_configuration(raw_config=raw_config)
    lv_config = LiveViewConfiguration(
        dependencies=deps,
        route_display=route_display,
        display_config=display_config,
    )

    app = PyView()
    _register_pyview_static(app)
    app.rootTemplate = camera_root_template(
        theme=route_display.theme or "light",
        css_vars={
            "banner_color": display_config.banner_color,
            "font_size_route_number": display_config.font_size_route_number,
            "font_size_destination": display_config.font_size_destination,
            "font_size_platform": display_config.font_size_platform,
            "font_size_time": display_config.font_size_time,
            "font_size_no_departures": display_config.font_size_no_departures,
            "font_size_direction_header": display_config.font_size_direction_header,
            "font_size_stop_header": display_config.font_size_stop_header,
            "font_size_pagination_indicator": (
                display_config.font_size_pagination_indicator
            ),
            "font_size_countdown_text": display_config.font_size_countdown_text,
            "font_size_status_header": display_config.font_size_status_header,
            "font_size_delay_amount": display_config.font_size_delay_amount,
        },
    )
    app.add_live_view("/", create_camera_live_view(lv_config))
    return app


def _log_camera_devices(raw_config: dict[str, Any]) -> None:
    import logging

    camera_cfg = raw_config.get("camera", {}) if isinstance(raw_config, dict) else {}
    configured = (
        camera_cfg.get("device", "auto") if isinstance(camera_cfg, dict) else "auto"
    )
    configured_str = str(configured) if isinstance(configured, str | int) else "auto"
    effective = resolve_device(configured_str)

    opts = format_camera_device_options_for_ui()
    selected_label = next(
        (o["label"] for o in opts if o.get("value") == configured_str),
        None,
    )
    logging.getLogger("uvicorn.error").info(
        "Camera selection: configured=%r effective=%r%s",
        configured_str,
        effective,
        f" ({selected_label})" if selected_label else "",
    )

    lines = [f'{o["value"]}: {o["label"]}' for o in opts]
    logging.getLogger("uvicorn.error").debug("Camera devices:\n%s", "\n".join(lines))


def _section(raw_config: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw_config.get(key, {})
    return value if isinstance(value, dict) else {}


def _get_int(section: dict[str, Any], key: str) -> int | None:
    value = section.get(key)
    return int(value) if isinstance(value, int) else None


def _get_device(section: dict[str, Any]) -> str:
    value = section.get("device", "auto")
    return str(value) if isinstance(value, str | int) else "auto"


def _get_poll_interval_s(section: dict[str, Any]) -> float:
    value = section.get("poll_interval_seconds", 1.0)
    return float(value) if isinstance(value, float | int) else 1.0


def _get_float(section: dict[str, Any], key: str, default: float) -> float:
    value = section.get(key, default)
    return float(value) if isinstance(value, float | int) else default


def _build_camera_stream(raw_config: dict[str, Any]) -> CameraStreamController:
    camera_cfg = _section(raw_config, "camera")
    plate_detection_cfg = _section(raw_config, "plate_detection")
    test_camera_cfg = _section(raw_config, "test_camera")
    raw_device = _get_device(camera_cfg)
    capture_fps = _get_int(camera_cfg, "capture_fps")

    return CameraStreamController(
        CameraStreamConfig(
            device=resolve_device(str(raw_device)),
            width=_get_int(camera_cfg, "width"),
            height=_get_int(camera_cfg, "height"),
            capture_fps=capture_fps,
            poll_interval_s=_get_poll_interval_s(plate_detection_cfg),
            test_min_duration_s=_get_float(
                test_camera_cfg, "min_duration_seconds", 1.0
            ),
            test_max_duration_s=_get_float(
                test_camera_cfg, "max_duration_seconds", 15.0
            ),
        )
    )


def main() -> None:
    """Main entry point for the server."""
    import signal
    import sys

    settings = Settings.load_from_project_root()
    app = create_app()

    # Configure uvicorn for graceful shutdown
    config = uvicorn.Config(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level="debug" if settings.server.debug else "info",
    )
    server = uvicorn.Server(config)

    # Handle Ctrl-C gracefully
    def signal_handler(_sig: int, _frame: object) -> None:
        logger = logging.getLogger("uvicorn.error")
        logger.info("Received interrupt signal, shutting down gracefully...")
        server.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.run()
    except KeyboardInterrupt:
        logger = logging.getLogger("uvicorn.error")
        logger.info("Keyboard interrupt received, shutting down...")
        sys.exit(0)


def main_camera() -> None:
    """Main entry point for the camera server (Raspberry Pi)."""
    import signal
    import sys

    settings = Settings.load_from_project_root()
    raw_config = load_raw_config_from_project_root()
    app = create_camera_app(settings=settings, raw_config=raw_config)

    # Configure uvicorn for graceful shutdown
    config = uvicorn.Config(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level="debug" if settings.server.debug else "info",
    )
    server = uvicorn.Server(config)

    # Handle Ctrl-C gracefully
    def signal_handler(_sig: int, _frame: object) -> None:
        logger = logging.getLogger("uvicorn.error")
        logger.info("Received interrupt signal, shutting down gracefully...")
        server.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.run()
    except KeyboardInterrupt:
        logger = logging.getLogger("uvicorn.error")
        logger.info("Keyboard interrupt received, shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
