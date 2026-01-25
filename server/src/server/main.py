"""Main entry point for the server."""

from __future__ import annotations

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

    stream = _build_camera_stream(raw_config)
    deps = build_camera_live_view_dependencies(
        raw_config=raw_config,
        server_bind=f"{settings.public_server.host}:{settings.public_server.port}",
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


def _build_camera_stream(raw_config: dict[str, Any]) -> CameraStreamController:
    camera_cfg = raw_config.get("camera", {}) if isinstance(raw_config, dict) else {}
    raw_device = (
        camera_cfg.get("device", "auto") if isinstance(camera_cfg, dict) else "auto"
    )
    width = camera_cfg.get("width") if isinstance(camera_cfg, dict) else None
    height = camera_cfg.get("height") if isinstance(camera_cfg, dict) else None
    fps = camera_cfg.get("fps") if isinstance(camera_cfg, dict) else None

    return CameraStreamController(
        CameraStreamConfig(
            device=resolve_device(str(raw_device)),
            width=int(width) if isinstance(width, int) else None,
            height=int(height) if isinstance(height, int) else None,
            fps=int(fps) if isinstance(fps, int) else None,
            max_fps=2.0,  # keep Pi CPU low; increase later if needed
        )
    )


def main() -> None:
    """Main entry point for the server."""
    settings = Settings.load_from_project_root()
    app = create_app()
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level="debug" if settings.server.debug else "info",
    )


def main_camera() -> None:
    """Main entry point for the camera server (Raspberry Pi)."""
    settings = Settings.load_from_project_root()
    raw_config = load_raw_config_from_project_root()
    app = create_camera_app(settings=settings, raw_config=raw_config)
    uvicorn.run(
        app,
        host=settings.public_server.host,
        port=settings.public_server.port,
        log_level="debug" if settings.public_server.debug else "info",
    )


if __name__ == "__main__":
    main()
