"""Configuration management for camera plate detection."""

import glob
from pathlib import Path
from typing import Literal

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]  # Python < 3.11

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CameraSettings(BaseSettings):
    """Camera configuration."""

    device: str = Field(
        default="auto",
        description="Camera device path. Use 'auto' for automatic detection or specify a path like '/dev/video0'",
    )
    width: int | Literal["auto"] = Field(
        default=1920,
        description="Camera resolution width in pixels, or 'auto' to use camera's default/native resolution",
    )
    height: int | Literal["auto"] = Field(
        default=1080,
        description="Camera resolution height in pixels, or 'auto' to use camera's default/native resolution",
    )
    fps: int = Field(default=30, ge=1, le=60, description="Camera frame rate")
    input_format: str = Field(
        default="",
        description="Input format (mjpeg, yuyv, etc.). Leave empty for auto-detection",
    )

    @field_validator("width", mode="before")
    @classmethod
    def validate_width(cls, v: int | str) -> int | str:
        """Validate width value - allow 'auto' or positive integer."""
        if isinstance(v, str) and v.lower() == "auto":
            return "auto"
        if isinstance(v, int) and v >= 1:
            return v
        raise ValueError("Width must be 'auto' or a positive integer")

    @field_validator("height", mode="before")
    @classmethod
    def validate_height(cls, v: int | str) -> int | str:
        """Validate height value - allow 'auto' or positive integer."""
        if isinstance(v, str) and v.lower() == "auto":
            return "auto"
        if isinstance(v, int) and v >= 1:
            return v
        raise ValueError("Height must be 'auto' or a positive integer")


class PlateDetectionSettings(BaseSettings):
    """License plate detection configuration."""

    model_size: Literal["nano", "small", "medium", "large", "xlarge"] = Field(
        default="nano",
        description="YOLOv8 model size (nano is fastest, xlarge is most accurate)",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for plate detection",
    )
    poll_interval: float = Field(
        default=1.0,
        ge=0.1,
        description="Detection polling interval in seconds",
    )


class PlateRecognitionSettings(BaseSettings):
    """License plate text recognition configuration."""

    languages: list[str] = Field(
        default_factory=lambda: ["de", "en"],
        description="Languages for OCR (German is required)",
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for OCR text recognition",
    )


class DebouncingSettings(BaseSettings):
    """Debouncing configuration for plate appearance/disappearance."""

    appearance_count: int = Field(
        default=3,
        ge=1,
        description="Number of detections required before a plate is considered 'appeared'",
    )
    appearance_window: float = Field(
        default=2.0,
        ge=0.1,
        description="Time window in seconds for appearance detection",
    )
    disappearance_timeout: float = Field(
        default=5.0,
        ge=0.1,
        description="Time in seconds a plate must be absent before it's considered 'disappeared'",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    log_file: str = Field(
        default="logs/plates.log",
        description="Log file path (relative to project root or absolute)",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level",
    )
    log_to_console: bool = Field(
        default=True,
        description="Whether to also log to stdout",
    )


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=None,  # We use TOML, not .env
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    camera: CameraSettings = Field(default_factory=CameraSettings)
    plate_detection: PlateDetectionSettings = Field(
        default_factory=PlateDetectionSettings
    )
    plate_recognition: PlateRecognitionSettings = Field(
        default_factory=PlateRecognitionSettings
    )
    debouncing: DebouncingSettings = Field(default_factory=DebouncingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @staticmethod
    def _find_project_root() -> Path:
        """Find project root by looking for config.toml."""
        current = Path.cwd()
        for parent in [current, *current.parents]:
            if (parent / "config.toml").exists():
                return parent
        return current

    @classmethod
    def load_from_project_root(cls, project_root: Path | None = None) -> "Settings":
        """Load settings from config.toml in project root."""
        if project_root is None:
            project_root = cls._find_project_root()

        config_file = project_root / "config.toml"
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}. "
                "Please create config.toml in the project root."
            )

        with config_file.open("rb") as f:
            toml_data = tomllib.load(f)

        return cls(**toml_data)

    def get_camera_device(self) -> str:
        """Get the camera device path, with auto-detection if needed."""
        if self.camera.device.lower() == "auto":
            return self._auto_detect_camera()
        return self.camera.device

    @staticmethod
    def _auto_detect_camera() -> str:
        """Auto-detect the first available video device."""
        video_devices = sorted(glob.glob("/dev/video*"))
        if not video_devices:
            raise RuntimeError(
                "No video devices found. Please connect a camera or specify device path in config.toml"
            )

        # Filter out video devices that are likely not cameras (like video10+ which are often codec devices)
        # Prefer video0, video1, etc.
        preferred_devices = [
            d for d in video_devices if "/dev/video" in d and len(d) <= 11
        ]
        if preferred_devices:
            return preferred_devices[0]

        return video_devices[0]
