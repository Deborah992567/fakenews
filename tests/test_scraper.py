"""Tests for URL validation and scraping."""

from unittest import mock

import pytest

from app.scraper import UrlFetcher, ScrapeError


class TestUrlValidation:
    def test_rejects_ftp(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError, match="Only http and https"):
            fetcher._validate_url("ftp://example.com/article")

    def test_rejects_file_scheme(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError, match="Only http and https"):
            fetcher._validate_url("file:///etc/passwd")

    def test_rejects_localhost(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError, match="not allowed"):
            fetcher._validate_url("http://localhost/admin")

    def test_rejects_local_domain(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError, match="not allowed"):
            fetcher._validate_url("http://myserver.local/api")

    def test_rejects_no_hostname(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError, match="missing a valid hostname"):
            fetcher._validate_url("http:///")

    def test_rejects_empty_string(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError):
            fetcher._validate_url("")

    def test_accepts_valid_http(self):
        """Validation passes for a valid HTTP URL (network check is separate)."""
        fetcher = UrlFetcher()
        # Mock DNS resolution to avoid actual network calls
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("93.184.216.34", 0))
        ]):
            fetcher._validate_url("http://example.com/article")

    def test_rejects_private_ip(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("192.168.1.1", 0))
        ]):
            with pytest.raises(ScrapeError, match="private network"):
                fetcher._validate_url("http://internal.company.com")

    def test_rejects_127_x(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("127.0.0.1", 0))
        ]):
            with pytest.raises(ScrapeError, match="private network"):
                fetcher._validate_url("http://myserver.com")

    def test_rejects_link_local(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("169.254.0.1", 0))
        ]):
            with pytest.raises(ScrapeError, match="private network"):
                fetcher._validate_url("http://link-local.dev")

    def test_rejects_unresolvable_host(self):
        fetcher = UrlFetcher()
        import socket
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
            with pytest.raises(ScrapeError, match="Unable to resolve"):
                fetcher._validate_url("http://definitely-not-a-real-host-12345.invalid")

    def test_rejects_too_many_redirects(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("93.184.216.34", 0))
        ]):
            with mock.patch("app.scraper.requests.Session.get") as mock_get:
                import requests
                mock_get.side_effect = requests.exceptions.TooManyRedirects()
                with pytest.raises(ScrapeError, match="Too many redirects"):
                    fetcher.fetch_text("http://example.com/redirect-loop")

    def test_rejects_timeout(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("93.184.216.34", 0))
        ]):
            with mock.patch("app.scraper.requests.Session.get") as mock_get:
                import requests
                mock_get.side_effect = requests.exceptions.Timeout()
                with pytest.raises(ScrapeError, match="timed out"):
                    fetcher.fetch_text("http://example.com/slow")

    def test_rejects_connection_error(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("93.184.216.34", 0))
        ]):
            with mock.patch("app.scraper.requests.Session.get") as mock_get:
                import requests
                mock_get.side_effect = requests.exceptions.ConnectionError("refused")
                with pytest.raises(ScrapeError, match="Unable to connect"):
                    fetcher.fetch_text("http://example.com")