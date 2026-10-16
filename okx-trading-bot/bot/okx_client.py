"""Public, GET-only OKX V5 REST client."""

from __future__ import annotations

from collections.abc import Callable
from http.client import HTTPException
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import Config


class OkxError(Exception):
    """Base error for public OKX data requests."""


class OkxTransportError(OkxError):
    """A public request failed at the HTTP or network layer."""


class OkxResponseError(OkxError):
    """OKX rejected the request or returned malformed data."""


def _http_get_json(url: str, timeout: float) -> Any:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "okx-trading-bot/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


class OkxClient:
    """Supports only the three public endpoints needed for the check."""

    def __init__(
        self,
        config: Config,
        transport: Callable[[str, float], Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._transport = transport or _http_get_json
        self._sleep = sleeper

    def _get(self, path: str, params: dict[str, str], *, allow_empty: bool = False) -> list[Any]:
        url = f"{self.config.base_url}{path}?{urlencode(params)}"
        for attempt in range(self.config.max_attempts):
            try:
                payload = self._transport(url, self.config.timeout_seconds)
            except HTTPError as exc:
                status = exc.code
                # urlopen raises before its context manager can close an error response.
                exc.close()
                if status != 429 and not 500 <= status <= 599:
                    raise OkxTransportError(f"OKX HTTP error {status}") from exc
                problem = f"OKX HTTP error {status}"
            except (URLError, TimeoutError, ConnectionError, HTTPException) as exc:
                problem = f"OKX network error: {exc.reason if isinstance(exc, URLError) else type(exc).__name__}"
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise OkxResponseError("Invalid JSON in OKX response") from exc
            else:
                if not isinstance(payload, dict) or not isinstance(payload.get("code"), str):
                    raise OkxResponseError("Malformed OKX response envelope")
                code = payload["code"]
                if code == "0":
                    data = payload.get("data")
                    if not isinstance(data, list) or (not data and not allow_empty):
                        raise OkxResponseError("OKX response has no data list")
                    return data
                if code != "50011":
                    message = payload.get("msg", "")
                    raise OkxResponseError(f"OKX error {code}: {message}")
                problem = f"OKX rate limit error {code}"
            if attempt + 1 == self.config.max_attempts:
                raise OkxTransportError(f"{problem}; attempts exhausted")
            self._sleep(min(0.5 * (2**attempt), 2.0))
        raise AssertionError("Unreachable retry state")

    def get_candles(self, limit: int = 101) -> list[Any]:
        return self._get(
            "/api/v5/market/candles",
            {"instId": "BTC-USDT", "bar": "15m", "limit": str(limit)},
        )

    def get_history_candles(self, *, after: int | None = None, limit: int = 100) -> list[Any]:
        """Public GET; OKX `after` pages toward older candle timestamps."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("history-candles limit must be 1..100")
        if after is not None and (not isinstance(after, int) or isinstance(after, bool) or after <= 0):
            raise ValueError("history-candles cursor must be a positive millisecond timestamp")
        params = {"instId": "BTC-USDT", "bar": "15m", "limit": str(limit)}
        if after is not None:
            params["after"] = str(after)
        return self._get("/api/v5/market/history-candles", params, allow_empty=True)

    def get_daily_candles(self, limit: int = 100) -> list[Any]:
        """UTC-aligned daily candles (newest first, the first one may be unconfirmed)."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 300:
            raise ValueError("daily candles limit must be 1..300")
        return self._get("/api/v5/market/candles", {"instId": "BTC-USDT", "bar": "1Dutc", "limit": str(limit)})

    def get_ticker(self) -> list[Any]:
        return self._get("/api/v5/market/ticker", {"instId": "BTC-USDT"})

    def get_spot_instrument(self) -> list[Any]:
        return self._get(
            "/api/v5/public/instruments", {"instType": "SPOT", "instId": "BTC-USDT"}
        )
