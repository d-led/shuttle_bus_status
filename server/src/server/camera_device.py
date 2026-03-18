"""Camera device resolution and listing (no streaming).

Resolves "auto" and "test:..." device strings and lists device candidates
for UI selection. Kept separate from camera_stream to improve maintainability.
"""

from __future__ import annotations

import glob
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("uvicorn.error")


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


def _test_camera_dir_candidate() -> str | None:
    """Return 'test:data/test_camera' if dir exists and has images, else None."""
    p = Path("data/test_camera")
    return f"test:{p}" if p.exists() and _has_images(p) else None


def _test_dataset_subdir_candidate() -> str | None:
    """Return first existing test: path under data/test_images/german_plates (kaggle/roboflow), else None."""
    base = Path("data/test_images/german_plates")
    if not base.exists():
        return None
    for subdir in ["kaggle", "roboflow"]:
        candidate = base / subdir
        if candidate.exists():
            return f"test:{candidate}"
    return None


def _test_device_candidates() -> list[str]:
    """Return test device candidates (test:path) for UI selection."""
    out: list[str] = []
    c1 = _test_camera_dir_candidate()
    if c1 is not None:
        out.append(c1)
    c2 = _test_dataset_subdir_candidate()
    if c2 is not None:
        out.append(c2)
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
