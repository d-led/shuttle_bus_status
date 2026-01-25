"""Structured logging setup with RFC 3339 timestamps."""

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, Processor


def add_rfc3339_timestamp(
    _logger: logging.Logger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Add RFC 3339 timestamp with millisecond precision and timezone."""
    # Get current time in UTC with millisecond precision
    now = datetime.now(UTC)
    # Format as RFC 3339 with milliseconds: 2024-01-15T10:30:45.123Z
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    event_dict["timestamp"] = timestamp
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_file: str | Path | None = None,
    log_to_console: bool = True,
) -> None:
    """Configure structured logging with RFC 3339 timestamps.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional). If None, no file logging.
        log_to_console: Whether to log to console/stdout
    """
    # Convert log level string to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard logging handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    handlers: list[logging.Handler] = []

    # Console handler
    console_handler: logging.Handler | None = None
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        handlers.append(console_handler)

    # File handler
    file_handler: logging.Handler | None = None
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        handlers.append(file_handler)

    # Base processors (same for all outputs)
    base_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_rfc3339_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure formatters based on output type
    if log_file and log_to_console:
        # Both file and console: JSON for file, console-friendly for stdout
        # File gets JSON
        assert file_handler is not None
        assert console_handler is not None
        file_processors = [*base_processors, structlog.processors.JSONRenderer()]
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processor=file_processors[-1],  # type: ignore[arg-type]
            foreign_pre_chain=file_processors[:-1],  # type: ignore[arg-type]
        )
        file_handler.setFormatter(file_formatter)

        # Console gets pretty output
        console_processors = [*base_processors, structlog.dev.ConsoleRenderer()]
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processor=console_processors[-1],  # type: ignore[arg-type]
            foreign_pre_chain=console_processors[:-1],  # type: ignore[arg-type]
        )
        console_handler.setFormatter(console_formatter)

        # Use ProcessorFormatter wrapper for structlog
        processors = [
            *base_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    elif log_file:
        # File only: JSON output
        assert file_handler is not None
        file_processors = [*base_processors, structlog.processors.JSONRenderer()]
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processor=file_processors[-1],  # type: ignore[arg-type]
            foreign_pre_chain=file_processors[:-1],  # type: ignore[arg-type]
        )
        file_handler.setFormatter(file_formatter)
        processors = [
            *base_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    else:
        # Console only: console-friendly output
        processors = [*base_processors, structlog.dev.ConsoleRenderer()]

    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def configure_from_settings(settings: Any) -> None:
    """Configure logging from Settings object.

    Args:
        settings: Settings object with logging configuration
    """
    log_file = None
    if hasattr(settings, "logging") and hasattr(settings.logging, "log_file"):
        log_file = settings.logging.log_file

    log_level = "INFO"
    if hasattr(settings, "logging") and hasattr(settings.logging, "log_level"):
        log_level = settings.logging.log_level

    log_to_console = True
    if hasattr(settings, "logging") and hasattr(settings.logging, "log_to_console"):
        log_to_console = settings.logging.log_to_console

    setup_logging(
        log_level=log_level,
        log_file=log_file,
        log_to_console=log_to_console,
    )
