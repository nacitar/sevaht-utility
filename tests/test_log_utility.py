from __future__ import annotations

import logging

import pytest

from sevaht_utility.log_utility import log_exceptions


def test_log_exceptions_logs_and_reraises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    test_logger = logging.getLogger("tests.log_exceptions")

    @log_exceptions(logger=test_logger)
    def raises() -> None:
        message = "boom"
        raise ValueError(message)

    with (
        caplog.at_level(logging.ERROR, logger=test_logger.name),
        pytest.raises(ValueError, match="boom"),
    ):
        raises()

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert record.message == "uncaught exception"
    assert getattr(record, "file_only", False) is True


def test_log_exceptions_returns_normal_result() -> None:
    @log_exceptions()
    def returns_value(value: int) -> int:
        return value + 1

    assert returns_value(4) == 5
