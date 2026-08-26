"""Log: JSONL format, chat/cycle contextvars, bounded rotation, extras."""

from __future__ import annotations

import json
import logging

from pretender.log import JsonFormatter, chat_context, get_logger, setup_logging


def _read_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_setup_logging_writes_jsonl(tmp_path):
    logger = setup_logging(directory=tmp_path, console=False)
    logger.info("hello %s", "world")
    lines = _read_lines(tmp_path / "pretender.jsonl")
    assert len(lines) == 1
    record = lines[0]
    assert record["msg"] == "hello world"
    assert record["level"] == "INFO"
    assert record["logger"] == "pretender"
    assert isinstance(record["ts"], float)


def test_chat_context_attaches_chat_key_and_cycle_id(tmp_path):
    logger = setup_logging(directory=tmp_path, console=False)
    with chat_context("qq:group:1", "cycle-42"):
        logger.info("inside")
    logger.info("outside")
    lines = _read_lines(tmp_path / "pretender.jsonl")
    assert lines[0]["chat_key"] == "qq:group:1"
    assert lines[0]["cycle_id"] == "cycle-42"
    assert "chat_key" not in lines[1]
    assert "cycle_id" not in lines[1]


def test_chat_context_restores_previous_value(tmp_path):
    logger = setup_logging(directory=tmp_path, console=False)
    with chat_context("outer"):
        with chat_context("inner"):
            logger.info("nested")
        logger.info("back")
    lines = _read_lines(tmp_path / "pretender.jsonl")
    assert lines[0]["chat_key"] == "inner"
    assert lines[1]["chat_key"] == "outer"


def test_extra_kwargs_land_in_json(tmp_path):
    logger = setup_logging(directory=tmp_path, console=False)
    logger.info("with extra", extra={"score": 12.5, "tags": ["a", "b"]})
    record = _read_lines(tmp_path / "pretender.jsonl")[0]
    assert record["score"] == 12.5
    assert record["tags"] == ["a", "b"]


def test_exception_info_is_serialized(tmp_path):
    logger = setup_logging(directory=tmp_path, console=False)
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")
    record = _read_lines(tmp_path / "pretender.jsonl")[0]
    assert "ValueError: boom" in record["exc"]


def test_rotation_is_bounded(tmp_path):
    logger = setup_logging(
        directory=tmp_path, console=False, max_bytes=1024, backup_count=2
    )
    for i in range(200):
        logger.info("line %d with some padding to grow the file", i)
    files = sorted(p.name for p in tmp_path.iterdir())
    # current file + up to backup_count rotated files
    assert "pretender.jsonl" in files
    rotated = [f for f in files if f != "pretender.jsonl"]
    assert len(rotated) <= 2


def test_setup_logging_is_idempotent(tmp_path):
    first = setup_logging(directory=tmp_path, console=False)
    second = setup_logging(directory=tmp_path, console=False)
    assert first is second
    assert len(first.handlers) == 1  # re-run cleared the previous handler


def test_get_logger_namespaced():
    logger = get_logger("gate")
    assert logger.name == "pretender.gate"


def test_json_formatter_handles_non_serializable_values(tmp_path):
    logger = setup_logging(directory=tmp_path, console=False)
    logger.info("obj", extra={"obj": object()})  # default=str fallback
    record = _read_lines(tmp_path / "pretender.jsonl")[0]
    assert isinstance(record["obj"], str)


def test_contextvars_exposed():
    from pretender.log import chat_key_var, cycle_id_var

    assert chat_key_var.get() is None
    assert cycle_id_var.get() is None


def test_formatter_used_by_handler(tmp_path):
    logger = setup_logging(directory=tmp_path, console=False)
    handler = logger.handlers[0]
    assert isinstance(handler.formatter, JsonFormatter)