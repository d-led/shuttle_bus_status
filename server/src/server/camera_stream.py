"""Camera streaming service (server-side).

Goal:
- Start capturing frames only when at least one LiveView client is connected
- Stop capturing when the last client disconnects

This keeps Raspberry Pi resource usage low.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import glob
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import cv2
from pyview.events import InfoEvent

if TYPE_CHECKING:
    from pyview.live_socket import ConnectedLiveViewSocket

logger = logging.getLogger("uvicorn.error")


class CameraStreamControllerProtocol(Protocol):
    async def register(
        self, socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None: ...

    async def unregister(
        self, socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None: ...

    async def set_device(self, device: str) -> None: ...


@dataclass(frozen=True)
class CameraStreamConfig:
    device: str
    width: int | None
    height: int | None
    capture_fps: int | None
    poll_interval_s: float


def resolve_device(raw_camera_device: str) -> str:
    """Resolve a configured device string into an OpenCV-usable selector."""
    if raw_camera_device != "auto":
        return raw_camera_device

    if sys.platform == "darwin":
        return "avfoundation:0"

    devices = sorted(glob.glob("/dev/video*"))
    if devices:
        return devices[0]
    return "/dev/video0"


def list_camera_device_candidates() -> list[str]:
    """List camera device candidates without probing hardware.

    This is intentionally conservative to avoid triggering camera permission prompts.
    - macOS: if available, use ffmpeg's AVFoundation device listing (names + indices).
      Otherwise, show a small range of indices.
    - Linux/RPi: prefer V4L2 devices that look like actual cameras (via v4l2-ctl names).
    """
    if sys.platform == "darwin":
        return _darwin_camera_device_candidates()
    return _linux_camera_device_candidates()


def _darwin_camera_device_candidates() -> list[str]:
    names = _avfoundation_device_names_from_ffmpeg()
    if not names:
        return [f"avfoundation:{i}" for i in range(5)]
    return [f"avfoundation:{i}" for i in sorted(names)]


def _linux_camera_device_candidates() -> list[str]:
    names_by_path = _v4l2_device_names_from_v4l2ctl()
    camera_like = _camera_like_v4l2_devices(names_by_path)
    return camera_like or sorted(glob.glob("/dev/video*"))


def _camera_like_v4l2_devices(names_by_path: dict[str, str]) -> list[str]:
    if not names_by_path:
        return []
    return [
        dev
        for dev, name in sorted(names_by_path.items())
        if _looks_like_camera_device(name)
    ]


def _looks_like_camera_device(name: str) -> bool:
    lowered = name.lower()
    # Keep things a bit permissive; we mainly want to hide codec/isp pseudo-devices.
    if "codec" in lowered or "decode" in lowered:
        return False
    if "isp" in lowered:
        return False
    return (
        "camera" in lowered or "uvc" in lowered or "usb" in lowered or "hd" in lowered
    )


def list_camera_device_candidates_with_names() -> list[tuple[str, str | None]]:
    """Return (device_selector, friendly_name) pairs without probing devices.

    On macOS we try to get names via ffmpeg's AVFoundation device listing.
    On Linux we try to get names via v4l2-ctl.
    Falls back to just selectors if tools are missing.
    """
    candidates = list_camera_device_candidates()
    if not candidates:
        return []

    if sys.platform == "darwin":
        names = _avfoundation_device_names_from_ffmpeg()
        result: list[tuple[str, str | None]] = []
        for sel in candidates:
            idx = _parse_avfoundation_index(sel)
            result.append((sel, names.get(idx) if idx is not None else None))
        return result

    # Linux/RPi
    names_by_path = _v4l2_device_names_from_v4l2ctl()
    return [(sel, names_by_path.get(sel)) for sel in candidates]


def format_camera_device_options_for_ui() -> list[dict[str, str]]:
    """Format camera device options for templates (value + label)."""
    options: list[dict[str, str]] = []

    resolved_auto = resolve_device("auto")
    auto_label = f"auto (→ {resolved_auto})"
    if sys.platform == "darwin":
        names = dict(list_camera_device_candidates_with_names())
        name = names.get(resolved_auto)
        if name:
            auto_label = f"auto (→ {resolved_auto} — {name})"
    options.append({"value": "auto", "label": auto_label})

    for value, name in list_camera_device_candidates_with_names():
        label = f"{value} — {name}" if name else value
        options.append({"value": value, "label": label})
    return options


def _parse_avfoundation_index(selector: str) -> int | None:
    if not selector.startswith("avfoundation:"):
        return None
    raw = selector.split(":", 1)[1]
    return int(raw) if raw.isdigit() else None


def _avfoundation_device_names_from_ffmpeg() -> dict[int, str]:
    if shutil.which("ffmpeg") is None:
        return {}

    # ffmpeg prints device lists to stderr.
    proc = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        check=False,
        capture_output=True,
        text=True,
    )
    text_out = (proc.stderr or "") + "\n" + (proc.stdout or "")

    in_video_section = False
    names: dict[int, str] = {}
    for line in text_out.splitlines():
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices" in line:
            in_video_section = False
            continue
        if not in_video_section:
            continue

        m = re.search(r"\[\s*(\d+)\s*\]\s+(.+)$", line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        name = m.group(2).strip()
        if name:
            names[idx] = name
    return names


def _v4l2_device_names_from_v4l2ctl() -> dict[str, str]:
    if shutil.which("v4l2-ctl") is None:
        return {}

    proc = subprocess.run(
        ["v4l2-ctl", "--list-devices"],
        check=False,
        capture_output=True,
        text=True,
    )
    out = proc.stdout or ""
    current_name: str | None = None
    mapping: dict[str, str] = {}
    for raw_line in out.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            # Device header line (usually ends with ':')
            current_name = line.rstrip(":").strip()
            continue
        if current_name is None:
            continue
        dev = line.strip()
        if dev.startswith("/dev/video"):
            mapping[dev] = current_name
    return mapping


class NullCameraStreamController(CameraStreamControllerProtocol):
    async def register(
        self, _socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None:
        return

    async def unregister(
        self, _socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None:
        return

    async def set_device(self, _device: str) -> None:
        return


class CameraStreamController(CameraStreamControllerProtocol):
    def __init__(self, config: CameraStreamConfig) -> None:
        self._config = config
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._sockets: set[ConnectedLiveViewSocket[dict[str, object]]] = set()

        # OpenCV objects are created inside the task (lazy).
        # OpenCV VideoCapture is untyped; keep it as Any.
        self._cap: Any | None = None

    async def register(
        self, socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None:
        async with self._lock:
            self._sockets.add(socket)
            if self._task is None or self._task.done():
                logger.info(
                    "Starting camera sampler task (clients=%d)", len(self._sockets)
                )
                self._task = asyncio.create_task(self._run(), name="camera-stream")

    async def unregister(
        self, socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None:
        async with self._lock:
            self._sockets.discard(socket)
            if not self._sockets:
                logger.info("Stopping camera sampler task (no clients)")
                await self._stop_locked()

    async def set_device(self, device: str) -> None:
        """Switch the active camera device.

        This is intended for interactive selection when multiple cameras exist.
        """
        next_device = _normalize_device_selection(device)
        async with self._lock:
            if next_device == self._config.device:
                return
            logger.info(
                "Switching camera device from %r to %r",
                self._config.device,
                next_device,
            )
            self._config = CameraStreamConfig(
                device=next_device,
                width=self._config.width,
                height=self._config.height,
                capture_fps=self._config.capture_fps,
                poll_interval_s=self._config.poll_interval_s,
            )
            self._close_capture()

    async def _stop_locked(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._close_capture()

    async def _run(self) -> None:
        try:
            await self._stream_loop()
        except asyncio.CancelledError:
            self._close_capture()
            raise
        except Exception as e:
            await self._handle_stream_error(e)

    async def _handle_stream_error(self, error: Exception) -> None:
        # Never crash the server due to streaming errors.
        logger.error("Camera stream error", exc_info=error)
        await self._broadcast_feedback(f"Camera stream error: {error!r}")
        self._close_capture()
        # Try again later if someone is still connected.
        await asyncio.sleep(1.0)
        async with self._lock:
            if self._sockets:
                self._task = asyncio.create_task(self._run(), name="camera-stream")

    async def _broadcast_feedback(self, message: str) -> None:
        sockets = await self._snapshot_sockets()
        for sock in sockets:
            _append_feedback(sock, message)

    async def _stream_loop(self) -> None:
        await self._open_capture()
        interval = max(float(self._config.poll_interval_s), 0.1)
        while await self._tick(interval):
            pass

    async def _tick(self, interval: float) -> bool:
        sockets = await self._snapshot_sockets()
        if not sockets:
            self._close_capture()
            return False

        if self._cap is None:
            await self._open_capture()

        frame_b64 = self._try_read_frame_b64()
        for sock in sockets:
            self._apply_frame_to_socket(sock, frame_b64)
            await sock.send_info(InfoEvent("update", "update"))

        await asyncio.sleep(interval if frame_b64 is not None else 1.0)
        return True

    async def _snapshot_sockets(
        self,
    ) -> list[ConnectedLiveViewSocket[dict[str, object]]]:
        async with self._lock:
            return list(self._sockets)

    async def _open_capture(self) -> None:
        if self._cap is not None:
            return

        device = self._config.device
        logger.info("Opening camera device=%r", device)
        if device.startswith("avfoundation:"):
            idx = int(device.split(":", 1)[1])
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(device)

        # Fail fast with a helpful message (otherwise reads just return False/None).
        is_opened = getattr(cap, "isOpened", None)
        if callable(is_opened) and not cap.isOpened():
            hint = ""
            if sys.platform == "darwin":
                hint = (
                    " On macOS, ensure the running process (e.g. Terminal/Cursor) "
                    "has camera permission in System Settings > Privacy & Security > Camera."
                )
            message = f"OpenCV could not open camera device {device!r}.{hint}"
            logger.warning(message)
            raise RuntimeError(message)

        if self._config.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.width))
        if self._config.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.height))
        if self._config.capture_fps is not None:
            cap.set(cv2.CAP_PROP_FPS, float(self._config.capture_fps))

        # Best-effort: keep the capture buffer small to reduce "stale frames" when sampling.
        # Not all backends support this property, but it's safe to try.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)

        self._cap = cap

    def _close_capture(self) -> None:
        if self._cap is None:
            return
        try:
            self._cap.release()
        finally:
            self._cap = None

    def _try_read_frame_b64(self) -> str | None:
        if self._cap is None:
            return None

        frame = _read_latest_frame(self._cap, flush_count=5)
        if frame is None:
            logger.debug("Camera read() returned no frame")
            return None

        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok2:
            logger.debug("JPEG encode failed")
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _apply_frame_to_socket(
        socket: ConnectedLiveViewSocket[dict[str, object]], frame_b64: str | None
    ) -> None:
        # Keep state minimal: the UI can decide what to render.
        #
        # IMPORTANT: do not spam the UI feedback window with camera status.
        # Only errors are surfaced there (see `_handle_stream_error`).
        socket.context["camera_connected"] = frame_b64 is not None
        if frame_b64 is not None:
            socket.context["camera_frame_b64"] = frame_b64


def _read_latest_frame(cap: Any, *, flush_count: int) -> Any | None:
    """Read the most recent frame, discarding buffered frames.

    When the camera captures at a higher FPS than our polling interval, OpenCV may
    return an old buffered frame. We call `grab()` a few times to advance to the
    latest frame and then `retrieve()` it.
    """
    for _ in range(max(int(flush_count), 0)):
        ok = cap.grab()
        if not ok:
            break

    ok, frame = cap.retrieve()
    if ok and frame is not None:
        return frame

    ok2, frame2 = cap.read()
    return frame2 if ok2 and frame2 is not None else None


def _append_feedback(
    socket: ConnectedLiveViewSocket[dict[str, object]],
    message: str,
    *,
    max_lines: int = 30,
) -> None:
    raw = socket.context.get("feedback_lines")
    if not isinstance(raw, list):
        return
    raw.append(message)
    if len(raw) > max_lines:
        del raw[:-max_lines]


def _normalize_device_selection(raw: str) -> str:
    selection = str(raw).strip()
    if selection == "auto":
        return resolve_device("auto")
    return selection
