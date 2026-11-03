"""Cost-capped ring-buffer logging and immediate breakage alert sink."""

from collections import deque
import logging
import sys
import time
from typing import Callable, Deque, Dict, List, Optional

from src.constants import (
    DEFAULT_LOG_DETAIL_LIMIT,
    DEFAULT_LOG_RING_CAPACITY,
    LOGGER_NAME,
    MAX_ALERT_SINK_CAPACITY,
    MAX_LOG_RING_CAPACITY,
    ONE_INT,
    ROUND_DECIMALS_SHAPES,
    ZERO_INT,
    verify_condition,
)
from src.gdpr import scrub_pii_text


class CostCappedRingBufferHandler(logging.Handler):
    """Fixed-capacity ring-buffer handler with PII scrubbing and alert sink."""

    def __init__(
        self,
        capacity: int = DEFAULT_LOG_RING_CAPACITY,
        max_detail_length: int = DEFAULT_LOG_DETAIL_LIMIT,
        alert_callback: Optional[Callable[[Dict[str, str]], None]] = None,
    ) -> None:
        """Initializes the bounded ring buffer and alert sink.

        Args:
            capacity (int, optional): Maximum log entries retained in memory.
            max_detail_length (int, optional): Maximum message character
                length before truncation.
            alert_callback (Optional[Callable[[Dict[str, str]], None]]):
                Optional callback invoked on ERROR or CRITICAL events.
        """
        super().__init__()
        verify_condition(
            ONE_INT <= int(capacity) <= MAX_LOG_RING_CAPACITY,
            "Log ring buffer capacity is outside allowed bounds.",
        )
        verify_condition(
            ONE_INT <= int(max_detail_length) <= MAX_LOG_RING_CAPACITY,
            "Log detail truncation limit is outside allowed bounds.",
        )
        self._capacity = int(capacity)
        self._max_detail_length = int(max_detail_length)
        self._records: Deque[Dict[str, str]] = deque(maxlen=self._capacity)
        self._alerts: Deque[Dict[str, str]] = deque(
            maxlen=MAX_ALERT_SINK_CAPACITY
        )
        self._alert_callback = alert_callback

    def emit(self, record: logging.LogRecord) -> None:
        """Formats, scrubs, truncates, and stores a log record without traces.

        Args:
            record (logging.LogRecord): Incoming Python logging record.
        """
        raw_msg = str(record.getMessage())
        scrubbed_msg = scrub_pii_text(raw_msg)
        if len(scrubbed_msg) > self._max_detail_length:
            scrubbed_msg = scrubbed_msg[: self._max_detail_length] + "..."

        entry = {
            "timestamp": str(round(time.time(), ROUND_DECIMALS_SHAPES)),
            "level": record.levelname,
            "logger": record.name,
            "detail": scrubbed_msg,
        }
        self._records.append(entry)

        if record.levelno >= logging.ERROR:
            alert_payload = {
                "timestamp": entry["timestamp"],
                "severity": record.levelname,
                "service": LOGGER_NAME,
                "summary": scrubbed_msg,
            }
            self._alerts.append(alert_payload)
            if self._alert_callback is not None:
                self._alert_callback(alert_payload)

    def get_recent_logs(self) -> List[Dict[str, str]]:
        """Returns a snapshot list of retained ring-buffer log entries.

        Returns:
            List[Dict[str, str]]: Ordered list of bounded log dictionaries.
        """
        return list(self._records)

    def get_dispatched_alerts(self) -> List[Dict[str, str]]:
        """Returns all immediate breakage alerts captured by the sink.

        Returns:
            List[Dict[str, str]]: List of ERROR/CRITICAL alert payloads.
        """
        return list(self._alerts)


_ACTIVE_RING_HANDLER: Optional[CostCappedRingBufferHandler] = None


def configure_logging(
    level: str = "INFO",
    enabled: bool = True,
    ring_capacity: int = DEFAULT_LOG_RING_CAPACITY,
    max_detail_length: int = DEFAULT_LOG_DETAIL_LIMIT,
    alert_callback: Optional[Callable[[Dict[str, str]], None]] = None,
) -> logging.Logger:
    """Configures and returns the cost-capped, PII-scrubbing package logger.

    Args:
        level (str, optional): Severity level ('DEBUG', 'INFO', 'WARNING',
            'ERROR', 'CRITICAL'). Defaults to "INFO".
        enabled (bool, optional): Toggle logging state. Defaults to True.
        ring_capacity (int, optional): Ring buffer max entries. Defaults to
            DEFAULT_LOG_RING_CAPACITY.
        max_detail_length (int, optional): Max character length per log entry.
        alert_callback (Optional[Callable[[Dict[str, str]], None]]): Optional
            hook triggered immediately on ERROR or CRITICAL logs.

    Returns:
        logging.Logger: Configured logger instance.
    """
    global _ACTIVE_RING_HANDLER
    verify_condition(
        isinstance(enabled, bool),
        "Logging enabled flag must be a boolean.",
    )
    verify_condition(
        isinstance(level, str) and len(level) > ZERO_INT,
        "Logging level must be a non-empty string.",
    )

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(numeric_level)
    logger.propagate = False
    logger.disabled = not enabled

    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    ring_handler = CostCappedRingBufferHandler(
        capacity=ring_capacity,
        max_detail_length=max_detail_length,
        alert_callback=alert_callback,
    )
    ring_handler.setFormatter(formatter)
    logger.addHandler(ring_handler)
    _ACTIVE_RING_HANDLER = ring_handler

    return logger


def get_logger() -> logging.Logger:
    """Retrieves the central package logger instance.

    Returns:
        logging.Logger: Central package logger.
    """
    return logging.getLogger(LOGGER_NAME)


def get_ring_buffer_handler() -> Optional[CostCappedRingBufferHandler]:
    """Retrieves the active cost-capped ring-buffer handler for diagnostics.

    Returns:
        Optional[CostCappedRingBufferHandler]: Active ring handler or None.
    """
    return _ACTIVE_RING_HANDLER