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
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pyview.events import InfoEvent

if TYPE_CHECKING:
    from pyview.live_socket import ConnectedLiveViewSocket


class CameraStreamControllerProtocol(Protocol):
    async def register(
        self, socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None: ...

    async def unregister(
        self, socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None: ...


@dataclass(frozen=True)
class CameraStreamConfig:
    device: str
    width: int | None
    height: int | None
    fps: int | None
    max_fps: float


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


class NullCameraStreamController(CameraStreamControllerProtocol):
    async def register(
        self, _socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None:
        return

    async def unregister(
        self, _socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None:
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
                self._task = asyncio.create_task(self._run(), name="camera-stream")

    async def unregister(
        self, socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None:
        async with self._lock:
            self._sockets.discard(socket)
            if not self._sockets:
                await self._stop_locked()

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
        except Exception:
            await self._handle_stream_error()

    async def _handle_stream_error(self) -> None:
        # Never crash the server due to streaming errors.
        self._close_capture()
        # Try again later if someone is still connected.
        await asyncio.sleep(1.0)
        async with self._lock:
            if self._sockets:
                self._task = asyncio.create_task(self._run(), name="camera-stream")

    async def _stream_loop(self) -> None:
        await self._open_capture()
        interval = max(1.0 / float(self._config.max_fps), 0.05)
        while await self._tick(interval):
            pass

    async def _tick(self, interval: float) -> bool:
        sockets = await self._snapshot_sockets()
        if not sockets:
            self._close_capture()
            return False

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
        import cv2  # imported lazily

        device = self._config.device
        if device.startswith("avfoundation:"):
            idx = int(device.split(":", 1)[1])
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(device)

        if self._config.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.width))
        if self._config.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.height))
        if self._config.fps is not None:
            cap.set(cv2.CAP_PROP_FPS, float(self._config.fps))

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

        import cv2  # imported lazily

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None

        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok2:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _apply_frame_to_socket(
        socket: ConnectedLiveViewSocket[dict[str, object]], frame_b64: str | None
    ) -> None:
        # Keep state minimal: the UI can decide what to render.
        if frame_b64 is None:
            socket.context["camera_connected"] = False
            _append_feedback(socket, "Camera stream: no frame available.")
            return

        socket.context["camera_connected"] = True
        socket.context["camera_frame_b64"] = frame_b64
        _append_feedback(socket, "Camera stream: frame updated.")


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
