"""Plate arrival/departure tracking with logging.

Wraps ArrivalsAndDepartures to add logging functionality.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from camera.config import DebouncingSettings, LoggingSettings, Settings

from server.arrivals_and_departures import ArrivalsAndDepartures, PlateState

logger = logging.getLogger(__name__)


class PlateTracker:
    """Tracks plate arrivals and departures with logging.

    Wraps ArrivalsAndDepartures to add logging functionality.
    Uses configurable thresholds to determine when a plate "arrives" (appears)
    and "departs" (disappears) to avoid logging false positives from brief
    detection errors.
    """

    def __init__(
        self,
        debouncing: DebouncingSettings,
        logging_settings: LoggingSettings,
        settings: Settings | None = None,
    ) -> None:
        self._debouncing = debouncing
        self._logging = logging_settings
        self._settings = settings or Settings.load_from_project_root()

        # Core tracking logic (no logging)
        self._tracker = ArrivalsAndDepartures(
            appearance_min_count=debouncing.appearance_min_count,
            appearance_window_seconds=debouncing.appearance_window_seconds,
            disappearance_timeout_seconds=debouncing.disappearance_timeout_seconds,
        )

        # Set up logging if enabled
        self._log_file: Any | None = None
        if self._logging.log_plates == "file":
            self._setup_file_logging()

    def _setup_file_logging(self) -> None:
        """Set up file logging for plate events."""
        log_path = Path(self._logging.log_file)
        if not log_path.is_absolute():
            # Relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            log_path = project_root / log_path

        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        try:
            self._log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
            logger.info("Plate event logging enabled: %s", log_path)
        except Exception as e:
            logger.warning("Failed to open log file %s: %s", log_path, e)
            self._log_file = None

    def _log_event(self, event_type: str, plate_text: str, **kwargs: Any) -> None:
        """Log a plate event according to logging settings."""
        timestamp = datetime.now(UTC).isoformat()
        message = f"[{timestamp}] {event_type}: {plate_text}"
        if kwargs:
            details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message += f" ({details})"

        # Log to file if enabled
        if self._logging.log_plates == "file" and self._log_file:
            try:
                self._log_file.write(message + "\n")
                self._log_file.flush()
            except Exception as e:
                logger.warning("Failed to write to log file: %s", e)

        # Log to console if enabled
        if self._logging.log_plates == "console" or (
            self._logging.log_plates == "file" and self._logging.log_to_console
        ):
            logger.info(message)

    def update(
        self, detected_plates: list[str], detected_at: datetime | None = None
    ) -> None:
        """Update tracker with new detections and log arrivals/departures.

        Args:
            detected_plates: List of plate texts detected in current frame
            detected_at: Timestamp of detection (defaults to now)
        """
        # Delegate to core tracking logic
        arrival_events, departure_events = self._tracker.update(
            detected_plates, detected_at
        )

        # Log arrival events
        for event in arrival_events:
            self._log_event("ARRIVED", event.plate_text)

        # Log departure events
        for dep_event in departure_events:
            duration = f"{dep_event.duration_seconds:.1f}s"
            self._log_event("DEPARTED", dep_event.plate_text, duration=duration)

    def get_present_plates(self) -> dict[str, PlateState]:
        """Get all plates that are currently present (have arrived)."""
        return self._tracker.get_present_plates()

    def close(self) -> None:
        """Close log file if open."""
        if self._log_file:
            with contextlib.suppress(Exception):
                self._log_file.close()
            self._log_file = None
