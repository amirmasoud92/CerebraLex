"""Logging setup shared by library code and command-line entry points."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_NOISY_LOGGERS = ("mne", "matplotlib", "numba", "h5py")


class JsonLinesFormatter(logging.Formatter):
    """Serialise each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ConsoleHandler(logging.StreamHandler):
    """Marker subclass so repeated configuration does not duplicate console output."""


def configure_logging(
    name: str,
    log_dir: Path | None = None,
    level: int = logging.INFO,
    run_id: str | None = None,
) -> logging.Logger:
    """Configure the root logger once and return a named child logger.

    Console output is human-readable; when ``log_dir`` is given a JSON-lines file
    named ``{name}_{run_id}.jsonl`` is also written for machine parsing.
    """
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, _ConsoleHandler) for h in root.handlers):
        console = _ConsoleHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(console)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"{name}_{stamp}.jsonl", encoding="utf-8")
        file_handler.setFormatter(JsonLinesFormatter())
        root.addHandler(file_handler)
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields: object) -> None:
    """Log ``msg`` with structured ``fields`` attached to the record."""
    logger.log(level, msg, extra={"fields": fields})
