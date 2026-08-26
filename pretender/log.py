"""JSONL logging: one JSON object per line, with chat/cycle context.

- ``chat_key_var`` / ``cycle_id_var`` contextvars are attached to every
  record emitted inside a ``chat_context`` block, so a log line always says
  which chat and which cycle it belongs to.
- Rotation is bounded: a RotatingFileHandler with max_bytes/backup_count.
- The formatter merges any ``extra={...}`` kwargs passed to the logger call
  (keys that are not stdlib LogRecord attributes) into the JSON object.
"""

from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

import orjson

chat_key_var: ContextVar[str | None] = ContextVar("chat_key", default=None)
cycle_id_var: ContextVar[str | None] = ContextVar("cycle_id", default=None)

# Attributes every LogRecord carries; anything else in record.__dict__ is a
# caller-supplied extra and belongs in the JSON payload.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        chat_key = chat_key_var.get()
        if chat_key is not None:
            payload["chat_key"] = chat_key
        cycle_id = cycle_id_var.get()
        if cycle_id is not None:
            payload["cycle_id"] = cycle_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = value
        return orjson.dumps(payload, default=str).decode("utf-8")


def setup_logging(
    directory: str | Path = "logs",
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
    console: bool = True,
) -> logging.Logger:
    """Configure the ``pretender`` logger with a bounded JSONL file handler.

    Idempotent: re-running clears the previous handlers on the logger.
    Returns the logger so callers can attach more handlers if they want.
    """
    logger = logging.getLogger("pretender")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    log_dir = Path(directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "pretender.jsonl",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JsonFormatter())
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """A namespaced child logger, e.g. ``get_logger("gate")``."""
    return logging.getLogger(f"pretender.{name}")


@contextlib.contextmanager
def chat_context(chat_key: str | None, cycle_id: str | None = None) -> Iterator[None]:
    """Attach chat/cycle context to every log line emitted inside the block."""
    tokens: list[Any] = []
    if chat_key is not None:
        tokens.append(chat_key_var.set(chat_key))
    if cycle_id is not None:
        tokens.append(cycle_id_var.set(cycle_id))
    try:
        yield
    finally:
        for token in reversed(tokens):
            token.var.reset(token)