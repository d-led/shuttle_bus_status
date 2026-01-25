"""Main entry point for the server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import uvicorn
from pyview import LiveView
from pyview.live_view import LiveRender, LiveTemplate
from pyview.pyview import PyView
from pyview.template import Template

from server.camera_view import CameraLiveView
from server.config import Settings

if TYPE_CHECKING:
    from pyview.meta import PyViewMeta


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
    app.add_live_view("/", IndexLiveView)
    return app


def create_camera_app() -> PyView:
    """Create and configure the camera server application."""
    app = PyView()
    app.add_live_view("/", CameraLiveView)
    return app


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
    app = create_camera_app()
    uvicorn.run(
        app,
        host=settings.public_server.host,
        port=settings.public_server.port,
        log_level="debug" if settings.public_server.debug else "info",
    )


if __name__ == "__main__":
    main()
