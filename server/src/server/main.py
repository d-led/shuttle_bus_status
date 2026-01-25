"""Main entry point for the server."""

import uvicorn
from pyview import LiveView
from starlette.applications import Starlette
from starlette.routing import Route

from server.config import Settings


class IndexLiveView(LiveView):
    """Main index page live view."""

    async def mount(self, _session, _params):
        """Initialize the live view."""
        return {"title": "Shuttle Bus Status"}

    async def render(self, state):
        """Render the page."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{state["title"]}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
            <h1>{state["title"]}</h1>
            <p>Shuttle bus status monitoring system</p>
        </body>
        </html>
        """


def create_app() -> Starlette:
    """Create and configure the Starlette application."""
    return Starlette(
        routes=[
            Route("/", IndexLiveView()),
        ]
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


if __name__ == "__main__":
    main()
