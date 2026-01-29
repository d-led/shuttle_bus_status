"""Test camera device that cycles through images from a directory.

This device shows a random image for 1-15 seconds, then switches to the next.
Useful for testing plate detection without a real camera.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from pyview.events import InfoEvent

if TYPE_CHECKING:
    import numpy as np
    from pyview.live_socket import ConnectedLiveViewSocket

logger = logging.getLogger("uvicorn.error")


class TestCameraDevice:
    """Test camera device that cycles through images, lingering on each for 1-15 cycles."""

    def __init__(
        self,
        *,
        image_directory: Path,
        min_duration_s: float = 1.0,
        max_duration_s: float = 15.0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Initialize test camera device.

        Args:
            image_directory: Directory containing test images
            min_duration_s: Minimum cycles to show each image (interpreted as cycles, not seconds)
            max_duration_s: Maximum cycles to show each image (interpreted as cycles, not seconds)
            width: Target width for images (None = keep original)
            height: Target height for images (None = keep original)
        """
        self._image_dir = Path(image_directory)
        # Convert to int cycles (round to nearest integer)
        self._min_cycles = max(1, round(min_duration_s))
        self._max_cycles = max(self._min_cycles, round(max_duration_s))
        self._target_width = width
        self._target_height = height

        # Find all images
        self._images: list[Path] = []
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            self._images.extend(self._image_dir.rglob(f"*{ext}"))

        if not self._images:
            raise ValueError(f"No images found in {image_directory}")

        # Images will be selected randomly each time (no need to shuffle or track index)
        self._current_image: np.ndarray | None = None
        self._current_image_path: Path | None = None
        self._current_cycles_remaining: int = 0
        self._target_cycles: int = 0

        logger.info(
            f"Test camera device initialized with {len(self._images)} images from {image_directory} "
            f"(will linger on each image for {self._min_cycles}-{self._max_cycles} cycles)"
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read current frame (OpenCV-compatible interface).

        Returns:
            (success, frame) tuple. Frame is None if no image available.
        """
        # Check if we need to switch to next image (cycle-based, not time-based)
        if self._current_image is None or self._current_cycles_remaining <= 0:
            self._switch_to_next_image()

        if self._current_image is None:
            return False, None

        # Decrement cycle counter
        self._current_cycles_remaining -= 1

        # Return a copy to avoid issues with concurrent access
        return True, self._current_image.copy()

    def grab(self) -> bool:
        """Grab next frame (OpenCV-compatible interface)."""
        success, _ = self.read()
        return success

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        """Retrieve grabbed frame (OpenCV-compatible interface)."""
        return self.read()

    def isOpened(self) -> bool:  # noqa: N802
        """Check if device is opened (always True for test device)."""
        return True

    def release(self) -> None:
        """Release device (no-op for test device)."""

    def set(self, prop: int, value: float) -> bool:
        """Set property (OpenCV-compatible interface).

        Supports:
        - cv2.CAP_PROP_FRAME_WIDTH
        - cv2.CAP_PROP_FRAME_HEIGHT
        """
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self._target_width = int(value)
            # Reload current image with new size
            if self._current_image_path:
                self._load_image(self._current_image_path)
            return True
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self._target_height = int(value)
            # Reload current image with new size
            if self._current_image_path:
                self._load_image(self._current_image_path)
            return True
        return False

    def get(self, prop: int) -> float:
        """Get property (OpenCV-compatible interface)."""
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return (
                float(self._current_image.shape[1])
                if self._current_image is not None
                else 0.0
            )
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return (
                float(self._current_image.shape[0])
                if self._current_image is not None
                else 0.0
            )
        if prop == cv2.CAP_PROP_FPS:
            return 1.0  # 1 FPS (we show static images)
        return 0.0

    def _switch_to_next_image(self) -> None:
        """Switch to a randomly selected image."""
        if not self._images:
            self._current_image = None
            self._current_image_path = None
            return

        # Randomly select next image
        image_path = random.choice(self._images)

        # Load image
        self._load_image(image_path)

        # Set random number of cycles to linger on this image
        self._target_cycles = random.randint(self._min_cycles, self._max_cycles)
        self._current_cycles_remaining = self._target_cycles

        logger.debug(
            f"Test camera: switched to {image_path.name} "
            f"(will linger for {self._target_cycles} cycles)"
        )

    def _load_image(self, image_path: Path) -> None:
        """Load and optionally resize image."""
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            self._current_image = None
            self._current_image_path = None
            return

        # Resize if target dimensions specified
        if self._target_width is not None or self._target_height is not None:
            h, w = img.shape[:2]
            target_w = self._target_width if self._target_width else w
            target_h = self._target_height if self._target_height else h

            # Maintain aspect ratio if only one dimension specified
            if self._target_width is None:
                target_w = int(w * (target_h / h))
            elif self._target_height is None:
                target_h = int(h * (target_w / w))

            if target_w != w or target_h != h:
                img = cv2.resize(
                    img, (target_w, target_h), interpolation=cv2.INTER_AREA
                )

        self._current_image = img
        self._current_image_path = image_path


class TestCameraStreamController:
    """Camera stream controller that uses TestCameraDevice."""

    def __init__(
        self,
        *,
        image_directory: Path,
        min_duration_s: float = 1.0,
        max_duration_s: float = 15.0,
        width: int | None = None,
        height: int | None = None,
        poll_interval_s: float = 1.0,
    ) -> None:
        """Initialize test camera stream controller.

        Args:
            image_directory: Directory containing test images
            min_duration_s: Minimum cycles to show each image (interpreted as cycles, not seconds)
            max_duration_s: Maximum cycles to show each image (interpreted as cycles, not seconds)
            width: Target width for images
            height: Target height for images
            poll_interval_s: How often to check for new frames
        """
        self._device = TestCameraDevice(
            image_directory=image_directory,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            width=width,
            height=height,
        )
        self._poll_interval = poll_interval_s
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._sockets: set[ConnectedLiveViewSocket[dict[str, object]]] = set()

    async def register(
        self, socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None:
        """Register a client socket."""
        async with self._lock:
            self._sockets.add(socket)
            if self._task is None or self._task.done():
                logger.info(
                    "Starting test camera stream task (clients=%d)", len(self._sockets)
                )
                self._task = asyncio.create_task(self._run(), name="test-camera-stream")

    async def unregister(
        self, socket: ConnectedLiveViewSocket[dict[str, object]]
    ) -> None:
        """Unregister a client socket."""
        async with self._lock:
            self._sockets.discard(socket)
            if not self._sockets:
                logger.info("Stopping test camera stream task (no clients)")
                await self._stop_locked()

    async def set_device(self, device: str) -> None:
        """Set device (no-op for test device, but required by protocol)."""
        del device

    async def _stop_locked(self) -> None:
        """Stop the stream task."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def _run(self) -> None:
        """Main stream loop."""
        try:
            await self._stream_loop()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Test camera stream error", exc_info=e)
            await self._broadcast_feedback(f"Test camera stream error: {e!r}")
            await asyncio.sleep(1.0)
            async with self._lock:
                if self._sockets:
                    self._task = asyncio.create_task(
                        self._run(), name="test-camera-stream"
                    )

    async def _broadcast_feedback(self, message: str) -> None:
        """Broadcast feedback message to all sockets."""
        sockets = await self._snapshot_sockets()
        for sock in sockets:
            _append_feedback(sock, message)

    async def _stream_loop(self) -> None:
        """Stream loop that reads frames and sends to clients."""
        interval = max(float(self._poll_interval), 0.1)
        while await self._tick(interval):
            pass

    async def _tick(self, interval: float) -> bool:
        """Single tick of the stream loop."""
        sockets = await self._snapshot_sockets()
        if not sockets:
            return False

        # Read frame from test device
        success, frame = self._device.read()
        frame_b64: str | None = None

        if success and frame is not None:
            # Encode as JPEG
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        # Send to all sockets
        for sock in sockets:
            sock.context["camera_connected"] = frame_b64 is not None
            if frame_b64 is not None:
                sock.context["camera_frame_b64"] = frame_b64
            await sock.send_info(InfoEvent("update", "update"))

        await asyncio.sleep(interval)
        return True

    async def _snapshot_sockets(
        self,
    ) -> list[ConnectedLiveViewSocket[dict[str, object]]]:
        """Get snapshot of connected sockets."""
        async with self._lock:
            return list(self._sockets)


def _append_feedback(
    socket: ConnectedLiveViewSocket[dict[str, object]],
    message: str,
    *,
    max_lines: int = 30,
) -> None:
    """Append feedback message to socket context."""
    raw = socket.context.get("feedback_lines")
    if not isinstance(raw, list):
        return
    raw.append(message)
    if len(raw) > max_lines:
        del raw[:-max_lines]
