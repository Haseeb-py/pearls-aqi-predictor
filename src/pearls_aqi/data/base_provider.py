"""Base provider protocol and HTTP client infrastructure."""

from abc import ABC
from typing import Any, Dict

import httpx

from pearls_aqi.domain.exceptions import ProviderAPIError


class BaseProvider(ABC):
    """Abstract base class for external data providers with retry and timeout logic."""

    def __init__(self, timeout_seconds: float = 30.0, max_retries: int = 3):
        self.timeout = timeout_seconds
        self.max_retries = max_retries

    def fetch_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch JSON data from a URL with retries and timeout error handling."""
        attempt = 0
        last_exception = None
        while attempt < self.max_retries:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_exception = exc
                attempt += 1

        raise ProviderAPIError(
            f"Failed to fetch data from {url} after {self.max_retries} attempts: {last_exception}"
        )
