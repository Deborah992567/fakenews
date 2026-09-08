"""Shared pytest fixtures.

The URL-extraction cache lives on the module-level app state and outlives
individual tests; without an explicit clear, a URL cached by one test could
short-circuit the fetch mock of a later test using the same URL. This autouse
fixture clears the bounded cache before and after every test so tests never see
each other's cached extractions.
"""

from __future__ import annotations

import pytest

import app.main as main_module


@pytest.fixture(autouse=True)
def _isolate_url_cache():
    cache = getattr(main_module.state, "url_cache", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()