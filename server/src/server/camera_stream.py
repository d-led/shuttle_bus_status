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
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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
    # Test camera device settings (only used when device starts with "test:")
    test_min_duration_s: float = 1.0
    test_max_duration_s: float = 15.0


def _resolve_auto_device() -> str:
    """Resolve 'auto' to a real camera or test device. Prefers real hardware."""
    if sys.platform == "darwin":
        candidates = _darwin_camera_device_candidates()
        return candidates[0] if candidates else "avfoundation:0"

    candidates = _linux_camera_device_candidates()
    if candidates:
        return candidates[0]

    test_camera_dir = Path("data/test_camera")
    if test_camera_dir.exists() and _has_images(test_camera_dir):
        logger.warning(
            "No real camera found, falling back to test camera device: %s",
            test_camera_dir,
        )
        return f"test:{test_camera_dir}"

    return "/dev/video0"


def resolve_device(raw_camera_device: str) -> str:
    """Resolve a configured device string into an OpenCV-usable selector.

    Special handling:
    - "test:<directory>" -> Test camera device using images from directory
    - "auto" -> Auto-detect real camera (prefers real hardware over test device)
    - Otherwise -> Use as-is
    """
    if raw_camera_device.startswith("test:"):
        return raw_camera_device
    if raw_camera_device != "auto":
        return raw_camera_device
    return _resolve_auto_device()


def _has_images(dir_path: Path) -> bool:
    """Return True if directory contains jpg or png images."""
    return bool(
        any(dir_path.glob("*.[jJ][pP][gG]")) or any(dir_path.glob("*.[pP][nN][gG]"))
    )


def _test_device_candidates() -> list[str]:
    """Return test device candidates (test:path) for UI selection."""
    out: list[str] = []
    test_camera_dir = Path("data/test_camera")
    if test_camera_dir.exists() and _has_images(test_camera_dir):
        out.append(f"test:{test_camera_dir}")
    test_dataset = Path("data/test_images/german_plates")
    if test_dataset.exists():
        for subdir in ["kaggle", "roboflow"]:
            test_dir = test_dataset / subdir
            if test_dir.exists():
                out.append(f"test:{test_dir}")
                break
    return out


def list_camera_device_candidates() -> list[str]:
    """List camera device candidates without probing hardware.

    This is intentionally conservative to avoid triggering camera permission prompts.
    - macOS: if available, use ffmpeg's AVFoundation device listing (names + indices).
      Otherwise, show a small range of indices.
    - Linux/RPi: prefer V4L2 devices that look like actual cameras (via v4l2-ctl names).
    - Test device: "test:<directory>" format for testing with images.

    Note: Test devices are listed FIRST for easy UI selection, but "auto" will prefer
    real cameras when resolving.
    """
    candidates = _test_device_candidates()
    if sys.platform == "darwin":
        candidates.extend(_darwin_camera_device_candidates())
    else:
        candidates.extend(_linux_camera_device_candidates())
    return candidates


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
    Test devices get friendly names from directory path.
    """
    candidates = list_camera_device_candidates()
    if not candidates:
        return []

    result: list[tuple[str, str | None]] = []

    for sel in candidates:
        # Handle test devices
        if sel.startswith("test:"):
            test_dir = Path(sel.split(":", 1)[1])
            name = f"Test Camera ({test_dir.name})"
            result.append((sel, name))
            continue

        # Handle macOS devices
        if sys.platform == "darwin":
            names = _avfoundation_device_names_from_ffmpeg()
            idx = _parse_avfoundation_index(sel)
            result.append((sel, names.get(idx) if idx is not None else None))
        else:
            # Linux/RPi
            names_by_path = _v4l2_device_names_from_v4l2ctl()
            result.append((sel, names_by_path.get(sel)))

    return result


def format_camera_device_options_for_ui() -> list[dict[str, str]]:
    """Format camera device options for templates (value + label).

    Test devices are listed first for easy selection, but "auto" will prefer
    real cameras when resolving.
    """
    options: list[dict[str, str]] = []

    # Add "auto" option with resolved device info
    resolved_auto = resolve_device("auto")
    auto_label = f"auto (→ {resolved_auto})"
    if sys.platform == "darwin":
        names = dict(list_camera_device_candidates_with_names())
        name = names.get(resolved_auto)
        if name:
            auto_label = f"auto (→ {resolved_auto} — {name})"
    # Add note if auto resolved to test device
    if resolved_auto.startswith("test:"):
        auto_label += " [no real camera found]"
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


def _detect_slow_log_entry(
    detect_time: float,
) -> tuple[int, str, tuple[float, ...]] | None:
    """Return (log_level, message, args) if detect_time is above thresholds, else None."""
    if detect_time > 10.0:
        return (
            logging.ERROR,
            "Plate detection took %.2fs total - TOO SLOW! "
            "Check logs for YOLO/OCR breakdown. "
            "SOLUTION: Set preprocess=false in config.toml and/or use faster OCR engine (tesseract)",
            (detect_time,),
        )
    if detect_time > 5.0:
        return (
            logging.WARNING,
            "Plate detection took %.2fs total. "
            "Check debug logs for YOLO/OCR breakdown. "
            "Consider: reducing image resolution, disabling preprocessing, or using faster OCR engine",
            (detect_time,),
        )
    if detect_time > 2.0:
        return (
            logging.INFO,
            "Plate detection took %.2fs total (check debug logs for YOLO/OCR breakdown)",
            (detect_time,),
        )
    return None


def _create_capture_for_device(device: str, config: CameraStreamConfig) -> Any:
    """Create and return an open VideoCapture (or TestCameraDevice). Raises if open fails."""
    if device.startswith("test:"):
        from server.test_camera_device import TestCameraDevice

        image_dir = Path(device.split(":", 1)[1])
        if not image_dir.exists():
            raise RuntimeError(f"Test camera image directory not found: {image_dir}")
        return TestCameraDevice(
            image_directory=image_dir,
            min_duration_s=config.test_min_duration_s,
            max_duration_s=config.test_max_duration_s,
            width=config.width,
            height=config.height,
        )
    if device.startswith("avfoundation:"):
        idx = int(device.split(":", 1)[1])
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(device)

    is_opened = getattr(cap, "isOpened", None)
    if callable(is_opened) and not cap.isOpened():
        hint = ""
        if sys.platform == "darwin":
            hint = (
                " On macOS, ensure the running process (e.g. Terminal/Cursor) "
                "has camera permission in System Settings > Privacy & Security > Camera."
            )
        raise RuntimeError(f"OpenCV could not open camera device {device!r}.{hint}")
    return cap


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

        # Plate detection components (lazy initialization)
        self._detector: Any | None = None
        self._ocr: Any | None = None
        self._detection_running: bool = False  # Track if detection is currently running
        self._detection_tasks: set[asyncio.Task[None]] = (
            set()
        )  # Track all detection tasks
        self._latest_detections: list[dict[str, object]] = (
            []
        )  # Latest detection results for UI
        self._current_frame_bgr: Any | None = (
            None  # Current frame for detection (to avoid reading different frames)
        )
        import threading

        self._detections_lock = (
            threading.Lock()
        )  # Thread-safe lock for detections (used from thread pool)

        # Plate arrival/departure tracking (lazy initialization)
        self._plate_tracker: Any | None = None

        # Camera connection state (stable, not flickering)
        self._camera_connected: bool = False

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
        """Stop the stream task and all detection tasks."""
        # Cancel all detection tasks
        for task in list(self._detection_tasks):
            if not task.done():
                task.cancel()
        # Wait for detection tasks to finish cancelling
        if self._detection_tasks:
            await asyncio.gather(*self._detection_tasks, return_exceptions=True)
        self._detection_tasks.clear()

        # Cancel main stream task
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._close_capture()

        # Close plate tracker
        if self._plate_tracker:
            self._plate_tracker.close()
            self._plate_tracker = None

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

        # Read frame once and store it (so detection uses same frame as displayed)
        # This is critical for test camera which advances on each read()
        frame_bgr = self._get_current_frame()
        frame_b64 = None
        if frame_bgr is not None:
            # Store frame for detection (thread-safe)
            with self._detections_lock:
                self._current_frame_bgr = frame_bgr.copy()  # Copy to avoid modification

            # Convert to base64 for UI
            _, buffer = cv2.imencode(".jpg", frame_bgr)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

        # Get latest detections (thread-safe read)
        with self._detections_lock:
            latest_detections = list(self._latest_detections)  # Copy for thread safety
        logger.debug("_tick: passing %d detections to UI", len(latest_detections))
        for sock in sockets:
            self._apply_frame_to_socket(sock, frame_b64, latest_detections)
            # Only send update event if context actually changed (avoid spamming WebSocket)
            # The context update itself triggers a render, so we don't always need the event
            await sock.send_info(InfoEvent("update", "update"))

        # Run plate detection in background thread to avoid blocking UI updates
        # This allows the async loop to continue and update the camera feed
        if frame_bgr is not None and not self._detection_running:
            # Run detection in thread pool - don't await, let it run in background
            # This prevents blocking the async loop and allows UI to update
            # Only start new detection if one isn't already running
            task = asyncio.create_task(self._run_plate_detection_async())
            self._detection_tasks.add(task)
            # Remove task from set when it completes
            task.add_done_callback(self._detection_tasks.discard)

        # Always sleep the full interval to maintain consistent frame rate
        # Detection runs in background and doesn't block UI updates
        await asyncio.sleep(interval)

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
        self._camera_connected = False

        cap = _create_capture_for_device(device, self._config)
        if self._config.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.width))
        if self._config.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.height))
        if self._config.capture_fps is not None:
            cap.set(cv2.CAP_PROP_FPS, float(self._config.capture_fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
        self._cap = cap

    def _close_capture(self) -> None:
        if self._cap is None:
            return
        try:
            # Test device has release() method, standard VideoCapture too
            if hasattr(self._cap, "release"):
                self._cap.release()
        finally:
            self._cap = None
            # Update connection state when camera is closed
            if self._camera_connected:
                self._camera_connected = False
                logger.debug(
                    "Camera connection state changed: disconnected (camera closed)"
                )

    def _resolve_yolo_model_path(self, project_root: Path, size_suffix: str) -> Path:
        """Resolve YOLO model path from possible locations."""
        possible_paths = [
            project_root / "models" / f"yolov8{size_suffix}.pt",
            project_root / "models" / f"best_{size_suffix}.pt",
            project_root / "models" / "best.pt",
            project_root / "models" / "plate_detection.pt",
        ]
        for path in possible_paths:
            if path.exists():
                return path
        logger.warning(
            "Model file not found in expected locations, using: %s",
            possible_paths[0],
        )
        return possible_paths[0]

    def _load_detection_settings(self) -> tuple[Any, Path, str]:
        """Load camera config and return (settings, project_root, size_suffix)."""
        from camera.config import Settings

        settings = Settings.load_from_project_root()
        project_root = Settings._find_project_root()
        size_map = {
            "nano": "n",
            "small": "s",
            "medium": "m",
            "large": "l",
            "xlarge": "x",
        }
        size_suffix = size_map.get(settings.plate_detection.model_size, "n")
        return settings, project_root, size_suffix

    def _create_detector_and_ocr(
        self, settings: Any, project_root: Path, size_suffix: str
    ) -> tuple[Any, Any]:
        """Create and return (detector, ocr) from settings."""
        from camera.plate_pipeline import (
            UltralyticsYoloPlateDetector,
            create_plate_recognizer_from_config,
        )

        model_path = self._resolve_yolo_model_path(project_root, size_suffix)
        detector = UltralyticsYoloPlateDetector(
            model_path=model_path,
            confidence_threshold=float(settings.plate_detection.confidence_threshold),
        )
        rec = settings.plate_recognition
        ocr = create_plate_recognizer_from_config(
            ocr_engine=rec.ocr_engine,
            languages=list(rec.languages),
            min_confidence=float(rec.min_confidence),
            preprocess=bool(rec.preprocess),
            allowlist=bool(rec.allowlist),
            allowlist_chars=str(rec.allowlist_chars),
            normalize=bool(rec.normalize),
        )
        return detector, ocr

    def _ensure_detection_components(self) -> None:
        """Lazy initialization of plate detector and OCR."""
        if self._detector is not None and self._ocr is not None:
            return
        try:
            settings, project_root, size_suffix = self._load_detection_settings()
            self._detector, self._ocr = self._create_detector_and_ocr(
                settings, project_root, size_suffix
            )
            rec = settings.plate_recognition
            logger.info(
                "Plate detection components initialized (model=%s, ocr_engine=%s, preprocess=%s)",
                self._resolve_yolo_model_path(project_root, size_suffix),
                rec.ocr_engine,
                rec.preprocess,
            )
        except Exception as e:
            logger.error(
                "Failed to initialize plate detection components: %s", e, exc_info=True
            )

    async def _notify_sockets_with_detections(
        self, latest_detections: list[dict[str, object]]
    ) -> None:
        """Update all connected sockets with latest detections and send update event."""
        logger.debug(
            "After detection: updating UI with %d detections",
            len(latest_detections),
        )
        sockets = await self._snapshot_sockets()
        for sock in sockets:
            sock.context["latest_detections"] = latest_detections
            sock.context["plates_detected"] = len(latest_detections)
            await sock.send_info(InfoEvent("update", "update"))

    async def _run_plate_detection_async(self) -> None:
        """Run plate detection on the current frame in a background thread (non-blocking)."""
        if self._cap is None or self._detection_running:
            return

        self._detection_running = True
        try:
            await asyncio.to_thread(self._run_plate_detection_sync)
            with self._detections_lock:
                latest_detections = list(self._latest_detections)
            await self._notify_sockets_with_detections(latest_detections)
        except asyncio.CancelledError:
            logger.debug("Plate detection task cancelled")
            raise
        except Exception as e:
            logger.error("Plate detection error: %s", e, exc_info=True)
        finally:
            self._detection_running = False

    def _ensure_plate_tracker(self) -> None:
        """Ensure plate tracker is initialized (lazy initialization)."""
        if self._plate_tracker is None:
            # Import from same package (server package)
            from camera.config import Settings

            from server.plate_tracker import PlateTracker

            settings = Settings.load_from_project_root()
            self._plate_tracker = PlateTracker(
                debouncing=settings.debouncing,
                logging_settings=settings.logging,
                settings=settings,
            )

    def _log_init_time_if_slow(self, init_time: float) -> None:
        """Log detection component init time if above thresholds."""
        if init_time > 1.0:
            logger.warning(
                "Detection component initialization took %.2fs (this should only happen once on first detection)",
                init_time,
            )
        elif init_time > 0.1:
            logger.info(
                "Detection component initialization took %.2fs",
                init_time,
            )

    def _log_detect_time_if_slow(self, detect_time: float) -> None:
        """Log plate detection duration if above thresholds."""
        entry = _detect_slow_log_entry(detect_time)
        if entry is not None:
            level, msg, args = entry
            logger.log(level, msg, *args)

    def _store_detections_and_update_tracker(
        self,
        valid_detections: list[Any],
        result: Any,
    ) -> None:
        """Store detection dicts for UI and update plate tracker."""
        from camera.reporting import _detection_to_dict

        detection_dicts = [_detection_to_dict(det) for det in valid_detections]
        with self._detections_lock:
            self._latest_detections = detection_dicts
        logger.debug("Stored %d detections in _latest_detections", len(detection_dicts))
        detected_at = result.captured_at or datetime.now(UTC)
        if self._plate_tracker:
            self._plate_tracker.update(
                [det.text for det in valid_detections],
                detected_at=detected_at,
            )
        for det in valid_detections:
            logger.info(
                "Plate detected: %s (confidence: %.2f)",
                det.text,
                det.raw_ocr_confidence or 0.0,
            )

    def _clear_detections_and_update_tracker(self, result: Any) -> None:
        """Clear UI detections and update tracker with empty list."""
        with self._detections_lock:
            self._latest_detections = []
        logger.debug("No detections found, clearing _latest_detections")
        if self._plate_tracker:
            detected_at = result.captured_at or datetime.now(UTC)
            self._plate_tracker.update([], detected_at=detected_at)

    def _get_frame_for_detection(self) -> Any | None:
        """Get current frame for detection (thread-safe copy). Returns None if unavailable."""
        if self._detector is None or self._ocr is None:
            return None
        with self._detections_lock:
            if self._current_frame_bgr is None:
                return None
            return self._current_frame_bgr.copy()

    def _run_detection_on_frame(self, frame: Any) -> None:
        """Run plate detection on frame and update _latest_detections and tracker."""
        from camera.plate_pipeline import detect_plates_in_image

        assert self._detector is not None
        assert self._ocr is not None
        logger.debug("Running detection on frame: shape=%s", frame.shape)
        detect_start = time.perf_counter()
        result = detect_plates_in_image(
            image_bgr=frame,
            detector=self._detector,
            ocr=self._ocr,
            include_crops=False,
        )
        self._log_detect_time_if_slow(time.perf_counter() - detect_start)

        if result is None:
            logger.error(
                "detect_plates_in_image returned None - this should not happen"
            )
            with self._detections_lock:
                self._latest_detections = []
            return

        valid_detections = [det for det in result.detections if det.text is not None]
        if valid_detections:
            self._store_detections_and_update_tracker(valid_detections, result)
        else:
            self._clear_detections_and_update_tracker(result)

    def _run_plate_detection_sync(self) -> None:
        """Run plate detection on the current frame synchronously (blocking call).

        This is called from a thread pool to avoid blocking the async loop.
        """
        if self._cap is None:
            return
        try:
            init_start = time.perf_counter()
            self._ensure_detection_components()
            self._ensure_plate_tracker()
            self._log_init_time_if_slow(time.perf_counter() - init_start)

            frame = self._get_frame_for_detection()
            if frame is None:
                logger.warning("_run_plate_detection_sync: No frame available")
                return

            self._run_detection_on_frame(frame)
        except Exception as e:
            logger.error("Plate detection error: %s", e, exc_info=True)
            with self._detections_lock:
                self._latest_detections = []

    def _get_current_frame(self) -> Any | None:
        """Get the current frame as BGR numpy array."""
        if self._cap is None:
            return None

        # Test device has its own read() method
        if hasattr(self._cap, "read") and not isinstance(self._cap, cv2.VideoCapture):
            success, frame = self._cap.read()
            return frame if success else None

        # Standard OpenCV VideoCapture - get latest frame
        return _read_latest_frame(self._cap, flush_count=5)

    def _try_read_frame_b64(self) -> str | None:
        if self._cap is None:
            return None

        # Test camera device has its own read() method
        if hasattr(self._cap, "read") and not isinstance(self._cap, cv2.VideoCapture):
            # Custom test device
            success, frame = self._cap.read()
            if not success or frame is None:
                return None
        else:
            # Standard OpenCV VideoCapture
            frame = _read_latest_frame(self._cap, flush_count=5)
            if frame is None:
                logger.debug("Camera read() returned no frame")
                return None

        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok2:
            logger.debug("JPEG encode failed")
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _apply_frame_to_socket(
        self,
        socket: ConnectedLiveViewSocket[dict[str, object]],
        frame_b64: str | None,
        detections: list[dict[str, object]] | None = None,
    ) -> None:
        # Keep state minimal: the UI can decide what to render.
        #
        # IMPORTANT: do not spam the UI feedback window with camera status.
        # Only errors are surfaced there (see `_handle_stream_error`).

        # Update camera connection state (debounced - only change if sustained)
        # This prevents flickering when frames are temporarily unavailable
        if frame_b64 is not None:
            # We have a frame - mark as connected
            if not self._camera_connected:
                self._camera_connected = True
                logger.debug("Camera connection state changed: connected")
        elif self._cap is None and self._camera_connected:
            # Camera is closed - mark as disconnected
            self._camera_connected = False
            logger.debug("Camera connection state changed: disconnected (cap is None)")
        # If frame_b64 is None but cap exists, keep current state (don't flicker)

        socket.context["camera_connected"] = self._camera_connected
        if frame_b64 is not None:
            socket.context["camera_frame_b64"] = frame_b64

        # Apply latest detections to socket context
        if detections is not None:
            socket.context["latest_detections"] = detections
            socket.context["plates_detected"] = len(detections)


def _read_latest_frame(cap: Any, *, flush_count: int) -> Any | None:
    """Read the most recent frame, discarding buffered frames.

    When the camera captures at a higher FPS than our polling interval, OpenCV may
    return an old buffered frame. We call `grab()` a few times to advance to the
    latest frame and then `retrieve()` it.

    For test devices, just call read() directly.
    """
    # Test devices don't need flushing
    if hasattr(cap, "read") and not isinstance(cap, cv2.VideoCapture):
        success, frame = cap.read()
        return frame if success else None

    # Standard OpenCV VideoCapture
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
