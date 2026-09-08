"""Tests for request correlation and structured access logging (audit B7)."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.observability import (
    RequestIdFilter,
    request_id_var,
)


@pytest.fixture(autouse=True)
def _fresh_context():
    token = request_id_var.set("-")
    yield
    request_id_var.reset(token)


class TestRequestIdHeader:
    def test_response_carries_request_id(self):
        client = TestClient(app)
        resp = client.get("/health/live")
        assert resp.status_code == 200
        forwarded = resp.headers.get("x-request-id")
        assert forwarded and len(forwarded) == 12

    def test_well_formed_incoming_id_is_echoed(self):
        client = TestClient(app)
        resp = client.get(
            "/health/live", headers={"X-Request-ID": "client-trace-42"}
        )
        assert resp.headers["x-request-id"] == "client-trace-42"

    def test_invalid_incoming_id_is_replaced(self):
        client = TestClient(app)
        resp = client.get(
            "/health/live", headers={"X-Request-ID": "bad id with spaces!!"}
        )
        echoed = resp.headers["x-request-id"]
        assert echoed != "bad id with spaces!!"
        assert len(echoed) == 12  # freshly generated

    def test_request_context_is_reset_after_request(self):
        client = TestClient(app)
        client.get("/health/live")
        assert request_id_var.get() == "-"


class TestAccessLog:
    def test_access_record_is_structured_with_request_id(self, caplog):
        client = TestClient(app)
        with caplog.at_level(logging.INFO, logger="fakenews.access"):
            client.get("/health/live")
        lines = [r for r in caplog.records if r.name == "fakenews.access"]
        assert lines
        latest = lines[-1]
        text = latest.getMessage()
        assert "method=GET" in text
        assert "path=/health/live" in text
        assert "status=200" in text
        assert "duration_ms=" in text
        assert "request_id=" in text

    def test_rejected_requests_are_still_correlated(self, caplog, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 1)
        from app.main import create_app

        real = create_app()
        client = TestClient(real)
        with caplog.at_level(logging.INFO, logger="fakenews.access"):
            client.get("/health/live")
            second = client.get("/health/live")
        assert second.status_code == 429
        assert second.headers["x-request-id"]
        lines = [r for r in caplog.records if r.name == "fakenews.access"]
        assert any("status=429" in r.getMessage() for r in lines)
        assert any("status=200" in r.getMessage() for r in lines)


class TestRequestIdFilter:
    def test_filter_attaches_active_request_id(self):
        record = logging.LogRecord(
            name="fakenews", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        filt = RequestIdFilter()
        token = request_id_var.set("abc123")
        try:
            assert filt.filter(record) is True
            assert record.request_id == "abc123"
        finally:
            request_id_var.reset(token)

    def test_filter_defaults_to_dash(self):
        record = logging.LogRecord(
            name="fakenews", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        RequestIdFilter().filter(record)
        assert record.request_id == "-"