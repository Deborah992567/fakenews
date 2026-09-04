"""Safe article URL fetching and text extraction.

Treats URLs as fully untrusted input. Enforces HTTPS/HTTP only, blocks
private/reserved network ranges (SSRF guard), caps response size, applies
timeouts and redirect limits, and extracts readable text via BeautifulSoup.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.config import settings


class ScrapeError(Exception):
    """Raised when a URL cannot be fetched or its article text extracted."""


_UNSUPPORTED_PATTERNS = ("localhost", ".local")
_PRIVATE_PREFIXES = ("10.", "127.", "169.254.", "172.", "192.168.")


class UrlFetcher:
    """Fetches and extracts article text from a validated URL."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.max_redirects = settings.MAX_REDIRECTS

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ScrapeError("Only http and https URLs are supported.")
        if not parsed.hostname:
            raise ScrapeError("The URL is missing a valid hostname.")

        hostname = parsed.hostname.lower()
        if any(pattern in hostname for pattern in _UNSUPPORTED_PATTERNS):
            raise ScrapeError("This URL type is not allowed for analysis.")

        # Prevent SSRF-style access to private / link-local / loopback ranges.
        self._check_resolved_addresses(hostname)

    def _check_resolved_addresses(self, hostname: str) -> None:
        try:
            infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise ScrapeError("Unable to resolve the URL host.") from exc

        for info in infos:
            try:
                address = info[4][0]
                ip = ipaddress.ip_address(address)
            except ValueError:
                continue
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ScrapeError("This URL points to a private network and is blocked.")

    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            element.decompose()
        text = soup.get_text(separator=" ")
        return " ".join(text.split())

    def _fetch_with_redirect_guard(self, url: str) -> requests.Response:
        """Fetch a URL following redirects manually, validating each hop.

        This prevents an SSRF attack that could smuggle the client onto a
        private network via a redirect from a public URL.
        """
        current = url
        for _ in range(settings.MAX_REDIRECTS + 1):
            self._validate_url(current)
            response = self.session.get(
                current,
                timeout=settings.REQUEST_TIMEOUT,
                headers={"User-Agent": "FakeNewsDetector/1.0"},
                allow_redirects=False,
                stream=True,
            )
            response.raise_for_status()
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ScrapeError("The URL returned an invalid redirect.")
                current = requests.compat.urljoin(current, location)
                response.close()
                continue
            return response
        raise ScrapeError("Too many redirects while fetching the URL.")

    def fetch_text(self, url: str) -> str:
        """Validate, fetch and extract readable article text from ``url``."""
        self._validate_url(url)

        try:
            response = self._fetch_with_redirect_guard(url)
        except ScrapeError:
            raise
        except requests.exceptions.Timeout as exc:
            raise ScrapeError("The request timed out while fetching the URL.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise ScrapeError("Unable to connect to the URL.") from exc
        except requests.exceptions.TooManyRedirects as exc:
            raise ScrapeError("Too many redirects while fetching the URL.") from exc
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            raise ScrapeError(
                f"The URL returned an HTTP error ({status or 'unknown'})."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ScrapeError("Failed to fetch the URL.") from exc

        headers = getattr(response, "headers", None)
        content_type = ""
        if headers is not None:
            content_type = headers.get("content-type", "") if hasattr(headers, "get") else ""
        if content_type and "html" not in content_type.lower() and "xml" not in content_type.lower():
            raise ScrapeError("The URL does not return HTML content to analyse.")

        if len(response.content) > settings.MAX_URL_RESPONSE_SIZE:
            raise ScrapeError("The page content is too large to analyse.")

        text = self._extract_text(response.content)
        if len(text.strip()) < 20:
            raise ScrapeError("No readable article text was found at the URL.")
        return text.strip()


def fetch_article_text(url: str) -> str:
    """Convenience wrapper returning the extracted article text for ``url``."""
    return UrlFetcher().fetch_text(url)
