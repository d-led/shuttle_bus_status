"""Unit tests for plate arrival/departure tracking."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, UTC
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from camera.config import DebouncingSettings, LoggingSettings, Settings
from server.arrivals_and_departures import (
    ArrivalsAndDepartures,
    ArrivalEvent,
    DepartureEvent,
    PlateState,
)
from server.plate_tracker import PlateTracker


class TestPlateState:
    """Test PlateState dataclass."""

    def test_initial_state(self) -> None:
        """Test initial state is empty."""
        state = PlateState()
        assert state.detections == []
        assert state.has_arrived is False
        assert state.arrived_at is None
        assert state.last_seen is None

    def test_state_with_detections(self) -> None:
        """Test state with detections."""
        now = datetime.now(UTC)
        state = PlateState(
            detections=[now],
            has_arrived=True,
            arrived_at=now,
            last_seen=now,
        )
        assert len(state.detections) == 1
        assert state.has_arrived is True
        assert state.arrived_at == now
        assert state.last_seen == now


class TestPlateTracker:
    """Test PlateTracker arrival/departure logic."""

    def test_initial_state(self) -> None:
        """Test tracker starts empty."""
        debouncing = DebouncingSettings()
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        assert len(tracker._tracker._plates) == 0
        assert tracker.get_present_plates() == {}
        tracker.close()

    def test_single_detection_not_enough_for_arrival(self) -> None:
        """Test single detection doesn't trigger arrival."""
        debouncing = DebouncingSettings(
            appearance_min_count=3,
            appearance_window_seconds=2.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        tracker.update(["ABC123"], now)
        
        assert "ABC123" in tracker._tracker._plates
        assert tracker._tracker._plates["ABC123"].has_arrived is False
        assert len(tracker.get_present_plates()) == 0
        tracker.close()

    def test_arrival_after_threshold(self) -> None:
        """Test arrival is logged after meeting threshold."""
        debouncing = DebouncingSettings(
            appearance_min_count=3,
            appearance_window_seconds=2.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        # First detection
        tracker.update(["ABC123"], now)
        assert not tracker._tracker._plates["ABC123"].has_arrived
        
        # Second detection
        tracker.update(["ABC123"], now + timedelta(seconds=0.5))
        assert not tracker._tracker._plates["ABC123"].has_arrived
        
        # Third detection - should trigger arrival
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update(["ABC123"], now + timedelta(seconds=1.0))
            mock_log.assert_called_once_with("ARRIVED", "ABC123")
        
        assert tracker._tracker._plates["ABC123"].has_arrived is True
        assert tracker._tracker._plates["ABC123"].arrived_at == now + timedelta(seconds=1.0)
        assert len(tracker.get_present_plates()) == 1
        tracker.close()

    def test_arrival_window_expires(self) -> None:
        """Test detections outside window don't count toward threshold."""
        debouncing = DebouncingSettings(
            appearance_min_count=3,
            appearance_window_seconds=2.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        # Two detections
        tracker.update(["ABC123"], now)
        tracker.update(["ABC123"], now + timedelta(seconds=0.5))
        
        # Wait for window to expire
        tracker.update(["ABC123"], now + timedelta(seconds=3.0))
        
        # Should only have 1 detection in window (the last one)
        state = tracker._tracker._plates["ABC123"]
        assert len(state.detections) == 1  # Old ones expired
        assert not state.has_arrived
        tracker.close()

    def test_departure_after_timeout(self) -> None:
        """Test departure is logged after timeout."""
        debouncing = DebouncingSettings(
            appearance_min_count=1,  # Easy threshold for testing
            appearance_window_seconds=2.0,
            disappearance_timeout_seconds=5.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        # Arrive
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update(["ABC123"], now)
            mock_log.assert_called_once_with("ARRIVED", "ABC123")
        
        assert "ABC123" in tracker._tracker._plates
        assert tracker._tracker._plates["ABC123"].has_arrived
        
        # Plate not detected for timeout period
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update([], now + timedelta(seconds=6.0))
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == "DEPARTED"
            assert call_args[0][1] == "ABC123"
            assert "duration" in call_args[1]
        
        # Plate should be removed from tracking
        assert "ABC123" not in tracker._tracker._plates
        assert len(tracker.get_present_plates()) == 0
        tracker.close()

    def test_departure_before_timeout_not_logged(self) -> None:
        """Test departure not logged if timeout not reached."""
        debouncing = DebouncingSettings(
            appearance_min_count=1,
            appearance_window_seconds=2.0,
            disappearance_timeout_seconds=5.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        # Arrive
        tracker.update(["ABC123"], now)
        
        # Plate not detected but timeout not reached
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update([], now + timedelta(seconds=3.0))
            # Should not log departure
            mock_log.assert_not_called()
        
        # Plate should still be tracked
        assert "ABC123" in tracker._tracker._plates
        assert len(tracker.get_present_plates()) == 1
        tracker.close()

    def test_multiple_plates_independent_tracking(self) -> None:
        """Test multiple plates tracked independently."""
        debouncing = DebouncingSettings(
            appearance_min_count=2,
            appearance_window_seconds=2.0,
            disappearance_timeout_seconds=5.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        
        # Plate 1 arrives
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update(["ABC123"], now)
            tracker.update(["ABC123"], now + timedelta(seconds=0.5))
            assert mock_log.call_count == 1
            assert mock_log.call_args[0][1] == "ABC123"
        
        # Plate 2 arrives
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update(["XYZ789"], now + timedelta(seconds=1.0))
            tracker.update(["XYZ789"], now + timedelta(seconds=1.5))
            assert mock_log.call_count == 1
            assert mock_log.call_args[0][1] == "XYZ789"
        
        # Both should be present
        present = tracker.get_present_plates()
        assert len(present) == 2
        assert "ABC123" in present
        assert "XYZ789" in present
        
        # Plate 1 departs
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update(["XYZ789"], now + timedelta(seconds=7.0))
            assert mock_log.call_count == 1
            assert mock_log.call_args[0][0] == "DEPARTED"
            assert mock_log.call_args[0][1] == "ABC123"
        
        # Only Plate 2 should remain
        present = tracker.get_present_plates()
        assert len(present) == 1
        assert "XYZ789" in present
        tracker.close()

    def test_empty_plate_texts_ignored(self) -> None:
        """Test empty or None plate texts are ignored."""
        debouncing = DebouncingSettings(
            appearance_min_count=1,
            appearance_window_seconds=2.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        tracker.update(["", None, "   ", "ABC123"], now)
        
        # Only valid plate should be tracked
        assert "ABC123" in tracker._tracker._plates
        assert "" not in tracker._tracker._plates
        assert None not in tracker._tracker._plates
        tracker.close()

    def test_plate_reappears_after_departure(self) -> None:
        """Test plate can arrive again after departing."""
        debouncing = DebouncingSettings(
            appearance_min_count=1,
            appearance_window_seconds=2.0,
            disappearance_timeout_seconds=5.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        
        # First arrival
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update(["ABC123"], now)
            assert mock_log.call_count == 1
            assert mock_log.call_args[0][0] == "ARRIVED"
        
        # Departure
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update([], now + timedelta(seconds=6.0))
            assert mock_log.call_count == 1
            assert mock_log.call_args[0][0] == "DEPARTED"
        
        # Reappear
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update(["ABC123"], now + timedelta(seconds=7.0))
            assert mock_log.call_count == 1
            assert mock_log.call_args[0][0] == "ARRIVED"
        
        assert "ABC123" in tracker._tracker._plates
        assert tracker._tracker._plates["ABC123"].has_arrived
        tracker.close()

    def test_file_logging(self) -> None:
        """Test file logging functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "plates.log"
            debouncing = DebouncingSettings(
                appearance_min_count=1,
                appearance_window_seconds=2.0,
            )
            logging_settings = LoggingSettings(
                log_plates="file",
                log_file=str(log_file),
                log_to_console=False,
            )
            tracker = PlateTracker(debouncing, logging_settings)
            
            now = datetime.now(UTC)
            tracker.update(["ABC123"], now)
            
            # Close to flush
            tracker.close()
            
            # Check log file was created and contains entry
            assert log_file.exists()
            content = log_file.read_text(encoding="utf-8")
            assert "ARRIVED" in content
            assert "ABC123" in content

    def test_console_logging(self) -> None:
        """Test console logging functionality."""
        debouncing = DebouncingSettings(
            appearance_min_count=1,
            appearance_window_seconds=2.0,
        )
        logging_settings = LoggingSettings(
            log_plates="console",
            log_to_console=False,  # Don't duplicate
        )
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        with patch("server.plate_tracker.logger") as mock_logger:
            tracker.update(["ABC123"], now)
            tracker.close()
            
            # Should have logged to console
            assert mock_logger.info.called
            call_args = mock_logger.info.call_args[0][0]
            assert "ARRIVED" in call_args
            assert "ABC123" in call_args

    def test_no_logging(self) -> None:
        """Test logging disabled."""
        debouncing = DebouncingSettings(
            appearance_min_count=1,
            appearance_window_seconds=2.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        with patch.object(tracker, "_log_file") as mock_file:
            tracker.update(["ABC123"], now)
            tracker.close()
            
            # File should not be opened
            assert tracker._log_file is None

    def test_departure_duration_calculation(self) -> None:
        """Test departure log includes correct duration."""
        debouncing = DebouncingSettings(
            appearance_min_count=1,
            appearance_window_seconds=2.0,
            disappearance_timeout_seconds=5.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        arrival_time = datetime.now(UTC)
        tracker.update(["ABC123"], arrival_time)
        
        departure_time = arrival_time + timedelta(seconds=10.0)
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update([], departure_time)
            
            # Check duration is included
            call_kwargs = mock_log.call_args[1]
            assert "duration" in call_kwargs
            duration = call_kwargs["duration"]
            assert duration is not None
            # Should be approximately 10 seconds
            assert "10.0" in duration or "9." in duration or "10." in duration
        tracker.close()

    def test_get_present_plates_only_arrived(self) -> None:
        """Test get_present_plates only returns arrived plates."""
        debouncing = DebouncingSettings(
            appearance_min_count=2,
            appearance_window_seconds=2.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        
        # Plate 1: not enough detections
        tracker.update(["ABC123"], now)
        assert len(tracker.get_present_plates()) == 0
        
        # Plate 2: arrives
        tracker.update(["XYZ789"], now)
        tracker.update(["XYZ789"], now + timedelta(seconds=0.5))
        assert len(tracker.get_present_plates()) == 1
        assert "XYZ789" in tracker.get_present_plates()
        assert "ABC123" not in tracker.get_present_plates()
        
        tracker.close()

    def test_concurrent_detections_same_plate(self) -> None:
        """Test multiple detections of same plate in same update."""
        debouncing = DebouncingSettings(
            appearance_min_count=2,
            appearance_window_seconds=2.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        # Same plate detected multiple times in one frame
        tracker.update(["ABC123", "ABC123", "ABC123"], now)
        
        # Should only count as one detection (set deduplication)
        state = tracker._tracker._plates["ABC123"]
        assert len(state.detections) == 1  # Set deduplication
        assert not state.has_arrived
        
        # Second update should trigger arrival
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update(["ABC123"], now + timedelta(seconds=0.5))
            mock_log.assert_called_once_with("ARRIVED", "ABC123")
        
        tracker.close()

    def test_detection_history_cleanup(self) -> None:
        """Test old detections are removed from history."""
        debouncing = DebouncingSettings(
            appearance_min_count=2,
            appearance_window_seconds=2.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        # Multiple detections
        tracker.update(["ABC123"], now)
        tracker.update(["ABC123"], now + timedelta(seconds=0.5))
        tracker.update(["ABC123"], now + timedelta(seconds=1.0))
        
        state = tracker._tracker._plates["ABC123"]
        assert len(state.detections) == 3
        
        # Detection well outside window should cause cleanup
        # Window is 2.0s, so detection at now+4.0s should only keep itself
        tracker.update(["ABC123"], now + timedelta(seconds=4.0))
        
        # Old detections should be removed (only new one within 2s window)
        state = tracker._tracker._plates["ABC123"]
        assert len(state.detections) == 1  # Only the new one
        tracker.close()

    def test_plate_seen_again_resets_departure_timer(self) -> None:
        """Test plate seen again resets departure timeout."""
        debouncing = DebouncingSettings(
            appearance_min_count=1,
            appearance_window_seconds=2.0,
            disappearance_timeout_seconds=5.0,
        )
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        now = datetime.now(UTC)
        tracker.update(["ABC123"], now)
        
        # Not seen for 4 seconds (close to timeout)
        tracker.update([], now + timedelta(seconds=4.0))
        
        # Seen again - should reset timer (last_seen becomes 4.0)
        tracker.update(["ABC123"], now + timedelta(seconds=4.0))
        
        # Not seen for 6 seconds (more than 5s timeout since last_seen at 4.0)
        with patch.object(tracker, "_log_event") as mock_log:
            tracker.update([], now + timedelta(seconds=10.0))
            # Should log departure (6 seconds since last seen, > 5s timeout)
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "DEPARTED"
        
        tracker.close()

    def test_close_without_file_logging(self) -> None:
        """Test close() works when file logging not enabled."""
        debouncing = DebouncingSettings()
        logging_settings = LoggingSettings(log_plates="no")
        tracker = PlateTracker(debouncing, logging_settings)
        
        # Should not raise
        tracker.close()
        tracker.close()  # Idempotent

    def test_file_logging_error_handling(self) -> None:
        """Test file logging handles errors gracefully."""
        debouncing = DebouncingSettings(
            appearance_min_count=1,
            appearance_window_seconds=2.0,
        )
        # Use a path that exists but is a directory (will fail to open as file)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "subdir" / "plates.log"  # Parent doesn't exist yet
            logging_settings = LoggingSettings(
                log_plates="file",
                log_file=str(log_file),
                log_to_console=False,
            )
            
            # Should not raise, but log warning if parent dir doesn't exist
            # Actually, mkdir(parents=True) should create it, so this might not fail
            # Let's test with a path that definitely fails (read-only or invalid)
            # Actually, let's just test that errors during write are handled
            tracker = PlateTracker(debouncing, logging_settings)
            
            # If file was opened, test write error handling
            if tracker._log_file:
                # Close it and make it invalid
                tracker._log_file.close()
                tracker._log_file = None
            
            # Try to log - should handle missing file gracefully
            now = datetime.now(UTC)
            tracker.update(["ABC123"], now)  # Should not raise
            
            tracker.close()

    def test_log_event_with_kwargs(self) -> None:
        """Test _log_event includes kwargs in message."""
        debouncing = DebouncingSettings()
        logging_settings = LoggingSettings(log_plates="file", log_to_console=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "plates.log"
            logging_settings.log_file = str(log_file)
            tracker = PlateTracker(debouncing, logging_settings)
            
            tracker._log_event("TEST", "ABC123", duration="5.0s", confidence=0.95)
            tracker.close()
            
            # Check message includes kwargs
            content = log_file.read_text(encoding="utf-8")
            assert "TEST" in content
            assert "ABC123" in content
            assert "duration=5.0s" in content
            assert "confidence=0.95" in content
