"""Camera server LiveView (PyView).

Patterns are intentionally aligned with:
- Monty: `liveview_experiment.py`
- MVG: `departures.py`

Key points:
- We register the LiveView via `PyView.add_live_view(...)` (a LiveView is not an ASGI app).
- We inject dependencies by returning a configured LiveView class (factory pattern).
- We load and render templates like MVG/Monty (ibis + FileReloader).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyview import LiveView, LiveViewSocket
from pyview.template.live_template import LiveRender, LiveTemplate
from pyview.vendor import ibis
from pyview.vendor.ibis.loaders import FileReloader


@dataclass(frozen=True)
class ConfigTableRow:
    section: str
    key: str
    value: str


@dataclass(frozen=True)
class CameraLiveViewDependencies:
    server_bind: str
    config_rows: list[ConfigTableRow]


@dataclass(frozen=True)
class RouteDisplaySettings:
    """MVG-style: per-route display overrides."""

    title: str | None = None
    theme: str | None = None


@dataclass(frozen=True)
class DisplayConfiguration:
    """MVG-style: visual configuration (defaults are compact)."""

    banner_color: str = "#667eea"
    font_size_route_number: str = "1.15rem"
    font_size_destination: str = "1rem"
    font_size_platform: str = "0.9rem"
    font_size_time: str = "1.1rem"
    font_size_no_departures: str = "1rem"
    font_size_direction_header: str = "1rem"
    font_size_stop_header: str = "1rem"
    font_size_pagination_indicator: str = "0.9rem"
    font_size_countdown_text: str = "0.9rem"
    font_size_status_header: str = "0.9rem"
    font_size_delay_amount: str = "0.9rem"


@dataclass(frozen=True)
class LiveViewConfiguration:
    """MVG-style: single configuration object passed into the LiveView factory."""

    dependencies: CameraLiveViewDependencies
    route_display: RouteDisplaySettings
    display_config: DisplayConfiguration


def _stringify_config_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _build_config_rows(raw_config: dict[str, Any]) -> list[ConfigTableRow]:
    sections: list[tuple[str, str]] = [
        ("camera", "Camera"),
        ("plate_detection", "Plate detection"),
        ("plate_recognition", "Plate recognition"),
        ("debouncing", "Debouncing"),
        ("logging", "Logging"),
    ]

    rows: list[ConfigTableRow] = []
    for section_key, section_label in sections:
        section_value = raw_config.get(section_key, {})
        if not isinstance(section_value, dict):
            continue

        for key, value in sorted(section_value.items(), key=lambda kv: str(kv[0])):
            rows.append(
                ConfigTableRow(
                    section=section_label,
                    key=str(key),
                    value=_stringify_config_value(value),
                )
            )

    return rows


def _group_config_rows(
    rows: list[ConfigTableRow],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.section, []).append({"key": row.key, "value": row.value})

    return [
        {"section": section, "rows": grouped[section]}
        for section in sorted(grouped.keys())
    ]


class CameraLiveView(LiveView[dict[str, Any]]):
    """Camera monitoring view.

    Works even when no camera is connected; the UI will just show "Not Connected".
    """

    def __init__(self, config: LiveViewConfiguration) -> None:
        super().__init__()
        self._config = config
        self._template = self._load_template()

    @staticmethod
    def _load_template() -> LiveTemplate:
        """Load template like MVG/Monty (ibis + FileReloader)."""
        current_file_path = Path(__file__).resolve()
        templates_dir = current_file_path.parent

        if not hasattr(ibis, "loader") or not isinstance(ibis.loader, FileReloader):
            ibis.loader = FileReloader(str(templates_dir))

        template_file = templates_dir / "camera_view.html"
        template_content = template_file.read_text(encoding="utf-8")
        template = ibis.Template(template_content, template_id=str(template_file))
        return LiveTemplate(template)

    def _build_template_assigns(self, context: dict[str, Any]) -> dict[str, Any]:
        title = self._config.route_display.title or "Shuttle Bus Status"
        status = str(context.get("status", "ready"))
        camera_connected = bool(context.get("camera_connected", False))

        return {
            "title": title,
            "status": status,
            "camera_connected": camera_connected,
            "plates_detected": int(context.get("plates_detected", 0)),
            "server_bind": self._config.dependencies.server_bind,
            "config_sections": _group_config_rows(
                self._config.dependencies.config_rows
            ),
        }

    async def mount(self, socket: LiveViewSocket[dict[str, Any]], session: Any) -> None:
        del session

        title = self._config.route_display.title or "Shuttle Bus Status"
        socket.context = {
            "title": "Shuttle Bus Status",
            "status": "ready",
            "camera_connected": False,
            "plates_detected": 0,
            "server_bind": self._config.dependencies.server_bind,
            "config_sections": _group_config_rows(
                self._config.dependencies.config_rows
            ),
        }
        socket.live_title = title

    async def render(self, assigns: dict[str, Any], meta: Any) -> str:
        template_assigns = self._build_template_assigns(assigns)
        return LiveRender(self._template, template_assigns, meta)  # type: ignore[no-any-return]


def create_camera_live_view(config: LiveViewConfiguration) -> type[CameraLiveView]:
    """MVG-style: return a configured LiveView class (PyView expects a class)."""

    class ConfiguredCameraLiveView(CameraLiveView):
        def __init__(self) -> None:
            super().__init__(config)

    return ConfiguredCameraLiveView


def build_camera_live_view_dependencies(
    *, raw_config: dict[str, Any], server_bind: str
) -> CameraLiveViewDependencies:
    return CameraLiveViewDependencies(
        server_bind=server_bind,
        config_rows=_build_config_rows(raw_config),
    )


def load_camera_template_for_tests() -> LiveTemplate:
    """Test helper: load the camera template via MVG-style loader."""
    return CameraLiveView._load_template()


def build_route_display_settings(*, raw_config: dict[str, Any]) -> RouteDisplaySettings:
    # Keep it simple: no server-side theme config yet.
    _ = raw_config
    return RouteDisplaySettings(title="Shuttle Bus Status", theme="light")


def build_display_configuration(*, raw_config: dict[str, Any]) -> DisplayConfiguration:
    # Future: read UI config from TOML if we add it.
    _ = raw_config
    return DisplayConfiguration()
