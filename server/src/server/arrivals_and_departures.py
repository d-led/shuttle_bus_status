"""Core arrival/departure tracking logic (no logging).

This module provides the pure tracking logic without any logging concerns.
Logging is handled by PlateTracker which wraps this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class PlateState:
    """State tracking for a single plate."""

    # Detection timestamps (most recent first)
    detections: list[datetime] = field(default_factory=list)

    # Whether this plate has been "arrived" (met appearance threshold)
    has_arrived: bool = False

    # When the plate arrived (first time it met threshold)
    arrived_at: datetime | None = None

    # Last time this plate was detected
    last_seen: datetime | None = None


@dataclass
class ArrivalEvent:
    """Event indicating a plate has arrived."""

    plate_text: str
    arrived_at: datetime


@dataclass
class DepartureEvent:
    """Event indicating a plate has departed."""

    plate_text: str
    departed_at: datetime
    duration_seconds: float


class ArrivalsAndDepartures:
    """Tracks plate arrivals and departures with debouncing.

    Pure tracking logic without logging. Returns events that can be
    handled by the caller (e.g., logged, stored, etc.).

    Uses configurable thresholds to determine when a plate "arrives" (appears)
    and "departs" (disappears) to avoid false positives from brief
    detection errors.
    """

    def __init__(
        self,
        appearance_min_count: int = 3,
        appearance_window_seconds: float = 2.0,
        disappearance_timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize tracker.

        Args:
            appearance_min_count: Number of detections required for arrival
            appearance_window_seconds: Time window for counting appearances
            disappearance_timeout_seconds: Time plate must be absent to depart
        """
        self._appearance_min_count = appearance_min_count
        self._appearance_window_seconds = appearance_window_seconds
        self._disappearance_timeout_seconds = disappearance_timeout_seconds

        # Map: plate_text -> PlateState
        self._plates: dict[str, PlateState] = {}

    def _record_detection(
        self, plate_text: str, detected_at: datetime
    ) -> ArrivalEvent | None:
        """Record a detection and return an arrival event if plate just arrived."""
        if plate_text not in self._plates:
            self._plates[plate_text] = PlateState()
        state = self._plates[plate_text]
        state.detections.append(detected_at)
        state.last_seen = detected_at
        window_start = detected_at.timestamp() - self._appearance_window_seconds
        state.detections = [
            dt for dt in state.detections if dt.timestamp() >= window_start
        ]
        if (
            not state.has_arrived
            and len(state.detections) >= self._appearance_min_count
        ):
            state.has_arrived = True
            state.arrived_at = detected_at
            return ArrivalEvent(plate_text=plate_text, arrived_at=detected_at)
        return None

    def _collect_departures(
        self, detected_set: set[str], detected_at: datetime
    ) -> list[DepartureEvent]:
        """Find plates that have been absent long enough and return departure events."""
        now = detected_at.timestamp()
        events: list[DepartureEvent] = []
        to_remove: list[str] = []
        for plate_text, state in self._plates.items():
            if not state.has_arrived or plate_text in detected_set:
                continue
            if state.last_seen is None:
                continue
            time_since_last_seen = now - state.last_seen.timestamp()
            if time_since_last_seen < self._disappearance_timeout_seconds:
                continue
            duration_seconds = (
                (detected_at - state.arrived_at).total_seconds()
                if state.arrived_at
                else 0.0
            )
            events.append(
                DepartureEvent(
                    plate_text=plate_text,
                    departed_at=detected_at,
                    duration_seconds=duration_seconds,
                )
            )
            to_remove.append(plate_text)
        for plate_text in to_remove:
            del self._plates[plate_text]
        return events

    def update(
        self,
        detected_plates: list[str],
        detected_at: datetime | None = None,
    ) -> tuple[list[ArrivalEvent], list[DepartureEvent]]:
        """Update tracker with new detections and return events.

        Args:
            detected_plates: List of plate texts detected in current frame
            detected_at: Timestamp of detection (defaults to now)

        Returns:
            Tuple of (arrival_events, departure_events) that occurred in this update
        """
        if detected_at is None:
            detected_at = datetime.now(UTC)
        detected_set = {p for p in detected_plates if p and p.strip()}

        arrival_events: list[ArrivalEvent] = []
        for plate_text in detected_set:
            event = self._record_detection(plate_text, detected_at)
            if event is not None:
                arrival_events.append(event)

        departure_events = self._collect_departures(detected_set, detected_at)
        return (arrival_events, departure_events)

    def get_present_plates(self) -> dict[str, PlateState]:
        """Get all plates that are currently present (have arrived)."""
        return {
            plate: state for plate, state in self._plates.items() if state.has_arrived
        }

    def get_plate_state(self, plate_text: str) -> PlateState | None:
        """Get state for a specific plate, or None if not tracked."""
        return self._plates.get(plate_text)
