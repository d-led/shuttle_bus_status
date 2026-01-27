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

import glob
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyview import LiveView, LiveViewSocket, is_connected
from pyview.template.live_template import LiveRender, LiveTemplate
from pyview.vendor import ibis
from pyview.vendor.ibis.loaders import FileReloader

from server.camera_stream import (
    CameraStreamControllerProtocol,
    NullCameraStreamController,
    format_camera_device_options_for_ui,
)

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class ConfigTableRow:
    section: str
    key: str
    value: str


@dataclass(frozen=True)
class CameraLiveViewDependencies:
    server_bind: str
    config_rows: list[ConfigTableRow]
    feedback_lines: list[str]
    stream: CameraStreamControllerProtocol
    camera_device: str
    camera_device_options: list[dict[str, str]]


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
    rows: list[ConfigTableRow] = []
    for section_key, section_label in _config_sections():
        rows.extend(_rows_from_config_section(raw_config, section_key, section_label))
    return rows


def _config_sections() -> list[tuple[str, str]]:
    return [
        ("camera", "Camera"),
        ("plate_detection", "Plate detection"),
        ("plate_recognition", "Plate recognition"),
        ("debouncing", "Debouncing"),
        ("logging", "Logging"),
    ]


def _rows_from_config_section(
    raw_config: dict[str, Any], section_key: str, section_label: str
) -> list[ConfigTableRow]:
    section_value = raw_config.get(section_key)
    if not isinstance(section_value, dict):
        return []

    # Filter out misleading settings that don't apply to our static image sampling
    excluded_keys = {"capture_fps"} if section_key == "camera" else set()

    items = sorted(section_value.items(), key=lambda kv: str(kv[0]))
    return [
        ConfigTableRow(
            section=section_label,
            key=str(key),
            value=_stringify_config_value(value),
        )
        for key, value in items
        if key not in excluded_keys
    ]


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


def _build_feedback_lines(raw_config: dict[str, Any]) -> list[str]:
    camera_cfg = raw_config.get("camera", {})
    device = (
        camera_cfg.get("device", "auto") if isinstance(camera_cfg, dict) else "auto"
    )

    logging_cfg = raw_config.get("logging", {})
    log_plates = (
        logging_cfg.get("log_plates", "file")
        if isinstance(logging_cfg, dict)
        else "file"
    )

    lines: list[str] = [
        "Config loaded (config.toml).",
        f"camera.device = {device!r}",
        f"logging.log_plates = {log_plates!r}",
    ]

    lines.append(
        f"Camera device selected: {_selected_camera_device_label(raw_config)!r}"
    )

    if sys.platform == "darwin":
        return lines

    # Linux/RPi: V4L2
    video_devices = sorted(glob.glob("/dev/video*"))
    if not video_devices:
        lines.append("No /dev/video* devices found. Is the camera connected?")
        return lines

    lines.append(f"Detected video devices: {', '.join(video_devices[:5])}")
    if len(video_devices) > 5:
        lines.append(f"... and {len(video_devices) - 5} more")
    return lines


def _camera_device_from_raw_config(raw_config: dict[str, Any]) -> str:
    camera_cfg = raw_config.get("camera", {})
    if not isinstance(camera_cfg, dict):
        return "auto"
    device = camera_cfg.get("device", "auto")
    return str(device) if isinstance(device, str | int) else "auto"


def _build_camera_device_options(raw_config: dict[str, Any]) -> list[dict[str, str]]:
    options = format_camera_device_options_for_ui()
    selected = _camera_device_from_raw_config(raw_config)
    if (
        selected
        and selected != "auto"
        and not any(o["value"] == selected for o in options)
    ):
        options.append({"value": selected, "label": selected})
    return options


def _camera_device_candidates_as_labels() -> list[str]:
    opts = format_camera_device_options_for_ui()
    # Keep it short: don't repeat the expanded "auto (→ ...)" entry.
    labels = [o["label"] for o in opts if o["value"] != "auto"]
    return ["auto", *labels]


def _selected_camera_device_label(raw_config: dict[str, Any]) -> str:
    selected = _camera_device_from_raw_config(raw_config)
    options = _build_camera_device_options(raw_config)
    for opt in options:
        if opt.get("value") == selected:
            return opt.get("label", selected)
    if selected == "auto" and options:
        # Prefer the enriched auto label (e.g. "auto (→ avfoundation:0 — ...)")
        first = options[0]
        if first.get("value") == "auto":
            return first.get("label", "auto")
    return selected


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
        feedback_lines = context.get("feedback_lines")
        if not isinstance(feedback_lines, list):
            feedback_lines = list(self._config.dependencies.feedback_lines)
        config_sections = context.get("config_sections")
        if not isinstance(config_sections, list):
            config_sections = _group_config_rows(self._config.dependencies.config_rows)
        camera_device = context.get("camera_device")
        if not isinstance(camera_device, str):
            camera_device = self._config.dependencies.camera_device
        camera_device_options = context.get("camera_device_options")
        if not isinstance(camera_device_options, list):
            camera_device_options = self._config.dependencies.camera_device_options

        return {
            "title": title,
            "status": status,
            "camera_connected": camera_connected,
            "plates_detected": int(context.get("plates_detected", 0)),
            "camera_frame_b64": context.get("camera_frame_b64"),
            "camera_device": camera_device,
            "camera_device_options": camera_device_options,
            "server_bind": self._config.dependencies.server_bind,
            "feedback_lines": list(feedback_lines),
            "config_sections": config_sections,
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
            "feedback_lines": list(self._config.dependencies.feedback_lines),
            "camera_device": self._config.dependencies.camera_device,
            "camera_device_options": list(
                self._config.dependencies.camera_device_options
            ),
            "config_sections": _group_config_rows(
                self._config.dependencies.config_rows
            ),
        }
        socket.live_title = title

        if is_connected(socket):
            # Start streaming only for connected sessions (WebSocket).
            await self._config.dependencies.stream.register(socket)

    async def disconnect(self, socket: LiveViewSocket[dict[str, Any]]) -> None:
        if is_connected(socket):
            await self._config.dependencies.stream.unregister(socket)

    async def handle_event(
        self,
        event: str,
        payload: Any = None,
        socket: LiveViewSocket[dict[str, Any]] | None = None,
    ) -> None:
        if event != "set_camera_device" or socket is None:
            return

        raw_device = _extract_device_from_payload(payload)
        if not isinstance(raw_device, str):
            return

        logger.info("Camera device selection requested: %r", raw_device)
        logger.debug("Camera device selection payload: %r", payload)

        # Update UI state immediately.
        socket.context["camera_device"] = raw_device
        socket.context["camera_frame_b64"] = None
        socket.context["camera_connected"] = False
        socket.context = dict(socket.context)

        if is_connected(socket):
            await self._config.dependencies.stream.set_device(raw_device)

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
    *,
    raw_config: dict[str, Any],
    server_bind: str,
    stream: CameraStreamControllerProtocol | None = None,
) -> CameraLiveViewDependencies:
    rows = _build_config_rows(raw_config)
    # Put bind into config table (not in the page header).
    rows.append(ConfigTableRow(section="Server", key="bind", value=server_bind))
    camera_device = _camera_device_from_raw_config(raw_config)
    camera_device_options = _build_camera_device_options(raw_config)
    return CameraLiveViewDependencies(
        server_bind=server_bind,
        config_rows=rows,
        feedback_lines=_build_feedback_lines(raw_config),
        stream=stream or NullCameraStreamController(),
        camera_device=camera_device,
        camera_device_options=camera_device_options,
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


def _extract_device_from_payload(payload: Any) -> str | None:
    """Best-effort extraction of camera device from a LiveView change payload."""
    if not isinstance(payload, dict):
        return None

    for data in _payload_dict_candidates(payload):
        device = _device_from_dict(data)
        if device is not None:
            return device

    return _single_string_value(payload)


def _payload_dict_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [payload]
    for key in ("value", "values", "form", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
    return candidates


def _device_from_dict(data: dict[str, Any]) -> str | None:
    value = data.get("device")
    return value if isinstance(value, str) else None


def _single_string_value(data: dict[str, Any]) -> str | None:
    values = [v for v in data.values() if isinstance(v, str)]
    return values[0] if len(values) == 1 else None
