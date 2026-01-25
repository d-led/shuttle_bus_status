"""Structured logging setup with RFC 3339 timestamps."""

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import structlog
from structlog.types import EventDict, Processor
from structlog.typing import FilteringBoundLogger


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


def _get_base_processors() -> list[Processor]:
    """Get base processors for structlog."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_rfc3339_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]


def _create_file_handler(log_file: str | Path, level: int) -> logging.FileHandler:
    """Create and configure a file handler."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    return handler


def _create_console_handler(level: int) -> logging.StreamHandler:
    """Create and configure a console handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    return handler


def _create_file_formatter(
    base_processors: list[Processor],
) -> structlog.stdlib.ProcessorFormatter:
    """Create formatter for file output (JSON)."""
    file_processors = [*base_processors, structlog.processors.JSONRenderer()]
    return structlog.stdlib.ProcessorFormatter(
        processor=file_processors[-1],  # type: ignore[arg-type]
        foreign_pre_chain=file_processors[:-1],  # type: ignore[arg-type]
    )


def _create_console_formatter(
    base_processors: list[Processor],
) -> structlog.stdlib.ProcessorFormatter:
    """Create formatter for console output (pretty)."""
    console_processors = [*base_processors, structlog.dev.ConsoleRenderer()]
    return structlog.stdlib.ProcessorFormatter(
        processor=console_processors[-1],  # type: ignore[arg-type]
        foreign_pre_chain=console_processors[:-1],  # type: ignore[arg-type]
    )


def _configure_handlers(
    log_file: str | Path | None,
    log_to_console: bool,
    level: int,
) -> tuple[list[logging.Handler], list[Processor]]:
    """Configure logging handlers and return handlers and processors."""
    handlers: list[logging.Handler] = []
    base_processors = _get_base_processors()

    if log_file and log_to_console:
        # Both file and console
        file_handler = _create_file_handler(log_file, level)
        console_handler = _create_console_handler(level)
        file_handler.setFormatter(_create_file_formatter(base_processors))
        console_handler.setFormatter(_create_console_formatter(base_processors))
        handlers.extend([file_handler, console_handler])
        processors = [
            *base_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    elif log_file:
        # File only
        file_handler = _create_file_handler(log_file, level)
        file_handler.setFormatter(_create_file_formatter(base_processors))
        handlers.append(file_handler)
        processors = [
            *base_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    else:
        # Console only
        console_handler = _create_console_handler(level)
        handlers.append(console_handler)
        processors = [*base_processors, structlog.dev.ConsoleRenderer()]

    return handlers, processors


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
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Configure handlers and processors
    handlers, processors = _configure_handlers(log_file, log_to_console, level)

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


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured structlog logger
    """
    # structlog.get_logger is typed as Any; cast to the public protocol type.
    return cast("FilteringBoundLogger", structlog.get_logger(name))


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
