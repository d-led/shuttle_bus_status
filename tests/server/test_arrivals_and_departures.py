"""Unit tests for ArrivalsAndDepartures core tracking logic."""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

import pytest

from server.arrivals_and_departures import (
    ArrivalsAndDepartures,
    ArrivalEvent,
    DepartureEvent,
    PlateState,
)


class TestArrivalsAndDepartures:
    """Test ArrivalsAndDepartures core tracking logic."""

    def test_initial_state(self) -> None:
        """Test tracker starts empty."""
        tracker = ArrivalsAndDepartures()
        assert len(tracker._plates) == 0
        assert tracker.get_present_plates() == {}

    def test_single_detection_not_enough_for_arrival(self) -> None:
        """Test single detection doesn't trigger arrival."""
        tracker = ArrivalsAndDepartures(
            appearance_min_count=3,
            appearance_window_seconds=2.0,
        )
        
        now = datetime.now(UTC)
        arrivals, departures = tracker.update(["ABC123"], now)
        
        assert arrivals == []
        assert departures == []
        assert "ABC123" in tracker._plates
        assert tracker._plates["ABC123"].has_arrived is False
        assert len(tracker.get_present_plates()) == 0

    def test_arrival_after_threshold(self) -> None:
        """Test arrival event after meeting threshold."""
        tracker = ArrivalsAndDepartures(
            appearance_min_count=3,
            appearance_window_seconds=2.0,
        )
        
        now = datetime.now(UTC)
        # First detection
        arrivals, departures = tracker.update(["ABC123"], now)
        assert arrivals == []
        assert not tracker._plates["ABC123"].has_arrived
        
        # Second detection
        arrivals, departures = tracker.update(["ABC123"], now + timedelta(seconds=0.5))
        assert arrivals == []
        assert not tracker._plates["ABC123"].has_arrived
        
        # Third detection - should trigger arrival
        arrivals, departures = tracker.update(["ABC123"], now + timedelta(seconds=1.0))
        assert len(arrivals) == 1
        assert arrivals[0].plate_text == "ABC123"
        assert arrivals[0].arrived_at == now + timedelta(seconds=1.0)
        assert departures == []
        
        assert tracker._plates["ABC123"].has_arrived is True
        assert tracker._plates["ABC123"].arrived_at == now + timedelta(seconds=1.0)
        assert len(tracker.get_present_plates()) == 1

    def test_departure_after_timeout(self) -> None:
        """Test departure event after timeout."""
        tracker = ArrivalsAndDepartures(
            appearance_min_count=1,
            appearance_window_seconds=2.0,
            disappearance_timeout_seconds=5.0,
        )
        
        now = datetime.now(UTC)
        # Arrive
        arrivals, departures = tracker.update(["ABC123"], now)
        assert len(arrivals) == 1
        assert "ABC123" in tracker._plates
        assert tracker._plates["ABC123"].has_arrived
        
        # Plate not detected for timeout period
        arrivals, departures = tracker.update([], now + timedelta(seconds=6.0))
        assert arrivals == []
        assert len(departures) == 1
        assert departures[0].plate_text == "ABC123"
        assert departures[0].departed_at == now + timedelta(seconds=6.0)
        assert departures[0].duration_seconds == pytest.approx(6.0, abs=0.1)
        
        # Plate should be removed from tracking
        assert "ABC123" not in tracker._plates
        assert len(tracker.get_present_plates()) == 0

    def test_multiple_plates_independent_tracking(self) -> None:
        """Test multiple plates tracked independently."""
        tracker = ArrivalsAndDepartures(
            appearance_min_count=2,
            appearance_window_seconds=2.0,
            disappearance_timeout_seconds=5.0,
        )
        
        now = datetime.now(UTC)
        
        # Plate 1 arrives
        arrivals, departures = tracker.update(["ABC123"], now)
        arrivals, departures = tracker.update(["ABC123"], now + timedelta(seconds=0.5))
        assert len(arrivals) == 1
        assert arrivals[0].plate_text == "ABC123"
        
        # Plate 2 arrives
        arrivals, departures = tracker.update(["XYZ789"], now + timedelta(seconds=1.0))
        arrivals, departures = tracker.update(["XYZ789"], now + timedelta(seconds=1.5))
        assert len(arrivals) == 1
        assert arrivals[0].plate_text == "XYZ789"
        
        # Both should be present
        present = tracker.get_present_plates()
        assert len(present) == 2
        assert "ABC123" in present
        assert "XYZ789" in present
        
        # Plate 1 departs
        arrivals, departures = tracker.update(["XYZ789"], now + timedelta(seconds=7.0))
        assert len(departures) == 1
        assert departures[0].plate_text == "ABC123"
        
        # Only Plate 2 should remain
        present = tracker.get_present_plates()
        assert len(present) == 1
        assert "XYZ789" in present

    def test_empty_plate_texts_ignored(self) -> None:
        """Test empty or None plate texts are ignored."""
        tracker = ArrivalsAndDepartures(
            appearance_min_count=1,
            appearance_window_seconds=2.0,
        )
        
        now = datetime.now(UTC)
        arrivals, departures = tracker.update(["", None, "   ", "ABC123"], now)
        
        # Only valid plate should be tracked
        assert "ABC123" in tracker._plates
        assert "" not in tracker._plates
        assert None not in tracker._plates
        assert len(arrivals) == 1
        assert arrivals[0].plate_text == "ABC123"

    def test_plate_reappears_after_departure(self) -> None:
        """Test plate can arrive again after departing."""
        tracker = ArrivalsAndDepartures(
            appearance_min_count=1,
            appearance_window_seconds=2.0,
            disappearance_timeout_seconds=5.0,
        )
        
        now = datetime.now(UTC)
        
        # First arrival
        arrivals, departures = tracker.update(["ABC123"], now)
        assert len(arrivals) == 1
        assert arrivals[0].plate_text == "ABC123"
        
        # Departure
        arrivals, departures = tracker.update([], now + timedelta(seconds=6.0))
        assert len(departures) == 1
        assert departures[0].plate_text == "ABC123"
        
        # Reappear
        arrivals, departures = tracker.update(["ABC123"], now + timedelta(seconds=7.0))
        assert len(arrivals) == 1
        assert arrivals[0].plate_text == "ABC123"
        
        assert "ABC123" in tracker._plates
        assert tracker._plates["ABC123"].has_arrived

    def test_get_plate_state(self) -> None:
        """Test get_plate_state returns correct state."""
        tracker = ArrivalsAndDepartures()
        
        now = datetime.now(UTC)
        tracker.update(["ABC123"], now)
        
        state = tracker.get_plate_state("ABC123")
        assert state is not None
        assert len(state.detections) == 1
        
        # Non-existent plate
        assert tracker.get_plate_state("XYZ789") is None
