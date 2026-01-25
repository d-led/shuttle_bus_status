"""Resolve camera device from `config.toml`.

This is used by scripts (e.g. `take-one-photo.sh`) to pick the right backend:
- Linux/RPi: `/dev/video0` (V4L2)
- macOS: `avfoundation:0`
"""

from __future__ import annotations

from camera.config import Settings


def main() -> None:
    settings = Settings.load_from_project_root()
    print(settings.get_camera_device())


if __name__ == "__main__":
    main()

