"""Tests for the logging configuration module."""

import logging
import os
from unittest import mock

from app.logging_config import configure_logging


def test_configure_logging_sets_info_by_default():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LOG_LEVEL", None)
        configure_logging()
        root = logging.getLogger("fakenews")
        assert root.level == logging.INFO


def test_configure_logging_respects_env():
    with mock.patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
        configure_logging()
        root = logging.getLogger("fakenews")
        assert root.level == logging.DEBUG


def test_configure_logging_handles_invalid_level():
    with mock.patch.dict(os.environ, {"LOG_LEVEL": "NOT_A_REAL_LEVEL"}):
        configure_logging()
