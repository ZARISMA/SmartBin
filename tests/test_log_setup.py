import logging
from smartwaste.log_setup import get_logger

def test_get_logger_returns_logger_with_expected_name():
    """Verify that calling get_logger returns a valid logging.Logger instance with the expected name."""
    logger = get_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "smartwaste"

def test_get_logger_initializes_handlers():
    """Verify that get_logger initializes handlers if they are missing."""
    logger = get_logger()
    assert len(logger.handlers) >= 1

def test_get_logger_does_not_duplicate_handlers():
    """Verify that calling get_logger multiple times does not add duplicate handlers."""
    logger1 = get_logger()
    initial_count = len(logger1.handlers)

    logger2 = get_logger()
    assert len(logger2.handlers) == initial_count
    assert logger1 is logger2
