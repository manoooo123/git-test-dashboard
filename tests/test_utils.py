"""
Unit tests for utility functions, logger, and custom exceptions.
"""

import pytest
from utils import (
    setup_logger,
    log_execution_time,
    AQIException,
    DataPipelineError,
    ModelNotFoundError,
    APIIntegrationError,
    InvalidInputError,
)


def test_custom_exceptions():
    err = AQIException("General error", {"code": 500})
    assert "General error" in str(err)

    d_err = DataPipelineError("Data missing")
    assert isinstance(d_err, AQIException)

    m_err = ModelNotFoundError("Model missing")
    assert isinstance(m_err, AQIException)

    a_err = APIIntegrationError("API timeout")
    assert isinstance(a_err, AQIException)

    i_err = InvalidInputError("Bad input")
    assert isinstance(i_err, AQIException)


def test_logger():
    logger = setup_logger("test_logger")
    assert logger is not None


def test_log_execution_time_decorator():
    @log_execution_time
    def sample_func(a, b):
        return a + b

    res = sample_func(5, 10)
    assert res == 15
