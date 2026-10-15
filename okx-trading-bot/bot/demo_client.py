"""GET-only OKX V5 Demo client. No trading or transfer endpoints are exposed."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from http.client import HTTPException
import hmac
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import Config, ConfigError, _read_env_file, demo_request_headers


class DemoError(Exception):
    """Safe, credential-free error for a Demo account check."""


@dataclass(frozen=True, repr=False)
class DemoCredentials:
    api_key: str = field(repr=False)
    secret: str = field(repr=False)
    passphrase: str = field(repr=False)

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.api_key, self.secret, self.passphrase)):
            raise ConfigError("В локальном .env нужны Demo API key, secret и passphrase")


@dataclass(frozen=True)
class CurrencyBalance:
    total: Decimal
    available: Decimal


@dataclass(frozen=True)
class DemoSummary:
    btc: CurrencyBalance
    usdt: CurrencyBalance
    open_order_count: int
    recent_fill_count: int
    checked_at_utc: datetime | None = None


@dataclass(frozen=True)
class SpotFeeRates:
    """Signed OKX maker/taker rates: negative means a charged fee."""

    maker: Decimal
    taker: Decimal


def _http_get_json(request: Request, timeout: float) -> Any:
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _amount(value: Any, label: str) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise DemoError(f"Некорректный demo-баланс: отсутствует {label}")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise DemoError(f"Некорректный demo-баланс: {label}") from exc
    if not number.is_finite() or number < 0:
        raise DemoError(f"Некорректный demo-баланс: {label}")
    return number


def _signed_rate(value: Any, label: str) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise DemoError(f"Некорректная ставка Demo-комиссии: {label}")
    try:
        rate = Decimal(value)
    except InvalidOperation as exc:
        raise DemoError(f"Некорректная ставка Demo-комиссии: {label}") from exc
    if not rate.is_finite() or not -1 < rate < 1:
        raise DemoError(f"Некорректная ставка Demo-комиссии: {label}")
    return rate


class DemoClient:
    """Only allowlisted private GET endpoints, always in Demo mode."""

    _BALANCE = "/api/v5/account/balance"
    _PENDING = "/api/v5/trade/orders-pending"
    _FILLS = "/api/v5/trade/fills-history"
    _TRADE_FEE = "/api/v5/account/trade-fee"
    _ALLOWED_PATHS = frozenset({_BALANCE, _PENDING, _FILLS, _TRADE_FEE})

    def __init__(
        self,
        config: Config,
        credentials: DemoCredentials,
        transport: Callable[[Request, float], Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if config.mode != "DEMO":
            raise ConfigError("Private requests require DEMO mode; LIVE is forbidden")
        if not isinstance(credentials, DemoCredentials):
            raise ConfigError("Demo credentials are required")
        self._config = config
        self._credentials = credentials
        self._transport = transport or _http_get_json
        self._sleep = sleeper
        self._clock = clock

    @classmethod
    def from_env_file(cls, env_file: Path | None = None) -> DemoClient:
        """Read private credentials only from this project's local .env file."""
        path = env_file or Path(__file__).resolve().parents[1] / ".env"
        if not path.is_file():
            raise ConfigError("Нет локального .env; настройте отдельный OKX Demo-аккаунт")
        try:
            values = _read_env_file(path)
        except (OSError, UnicodeError) as exc:
            raise ConfigError("Не удалось прочитать локальный .env") from exc
        if "OKX_REGION" not in values:
            raise ConfigError("Укажите OKX_REGION в локальном .env для Demo-аккаунта")
        if values.get("OKX_MODE", "").upper() != "DEMO":
            raise ConfigError("Укажите OKX_MODE=DEMO в локальном .env; LIVE запрещён")
        try:
            config = Config.from_environment(path, environ={})
        except (OSError, UnicodeError) as exc:
            raise ConfigError("Не удалось прочитать локальный .env") from exc
        credentials = DemoCredentials(
            values.get("OKX_DEMO_API_KEY", ""),
            values.get("OKX_DEMO_API_SECRET", ""),
            values.get("OKX_DEMO_PASSPHRASE", ""),
        )
        return cls(config, credentials)

    @property
    def config(self) -> Config:
        """Non-secret regional configuration for matching public requests."""
        return self._config

    def _timestamp(self) -> str:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise DemoError("Не удалось получить время UTC для подписи Demo-запроса")
        return now.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _get(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        if path not in self._ALLOWED_PATHS:
            raise DemoError("Запрещённый приватный endpoint")
        query = urlencode(params)
        request_path = path + (f"?{query}" if query else "")
        url = self._config.base_url + request_path
        for attempt in range(self._config.max_attempts):
            timestamp = self._timestamp()
            prehash = timestamp + "GET" + request_path  # GET body is empty.
            signature = base64.b64encode(
                hmac.new(self._credentials.secret.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256).digest()
            ).decode("ascii")
            headers = {
                "Accept": "application/json",
                "User-Agent": "okx-trading-bot/0.1",
                "OK-ACCESS-KEY": self._credentials.api_key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self._credentials.passphrase,
                **demo_request_headers(self._config),
            }
            request = Request(url, headers=headers, method="GET")
            try:
                payload = self._transport(request, self._config.timeout_seconds)
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in (429,) and not 500 <= status <= 599:
                    raise DemoError(f"Demo API отклонил запрос (HTTP {status}); проверьте Demo-ключи и права") from None
                problem = f"Demo API HTTP {status}"
            except (URLError, TimeoutError, ConnectionError, HTTPException, OSError):
                problem = "Demo API недоступен или истёк тайм-аут"
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise DemoError("Demo API вернул некорректный JSON") from exc
            else:
                if not isinstance(payload, dict) or not isinstance(payload.get("code"), str):
                    raise DemoError("Demo API вернул некорректную структуру ответа")
                code = payload["code"]
                if not re.fullmatch(r"\d{1,6}", code):
                    raise DemoError("Demo API вернул некорректный код ответа")
                if code == "0":
                    data = payload.get("data")
                    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
                        raise DemoError("Demo API вернул некорректный список данных")
                    return data
                if code == "50102":
                    raise DemoError("Неверное время запроса Demo API; синхронизируйте часы Windows")
                if code != "50011":
                    raise DemoError(f"Demo API отклонил запрос (код {code}); проверьте Demo-ключи и права")
                problem = "Demo API ограничил частоту запросов (50011)"
            if attempt + 1 == self._config.max_attempts:
                raise DemoError(f"{problem}; попытки исчерпаны")
            self._sleep(min(0.5 * (2**attempt), 2.0))
        raise AssertionError("Unreachable retry state")

    def get_balances(self) -> tuple[CurrencyBalance, CurrencyBalance]:
        rows = self._get(self._BALANCE, {"ccy": "BTC,USDT"})
        if len(rows) != 1 or not isinstance(rows[0].get("details"), list):
            raise DemoError("Demo API вернул некорректный снимок баланса")
        balances: dict[str, CurrencyBalance] = {}
        for detail in rows[0]["details"]:
            if not isinstance(detail, dict) or not isinstance(detail.get("ccy"), str):
                raise DemoError("Demo API вернул некорректную валюту баланса")
            currency = detail["ccy"]
            if currency not in ("BTC", "USDT"):
                raise DemoError("Demo API вернул неожиданную валюту баланса")
            if currency in balances:
                raise DemoError("Demo API вернул дублирующийся баланс")
            total = _amount(detail.get("cashBal"), f"{currency} cashBal")
            available = _amount(detail.get("availBal"), f"{currency} availBal")
            if available > total:
                raise DemoError(f"Demo API вернул противоречивый баланс {currency}")
            balances[currency] = CurrencyBalance(total, available)
        zero = CurrencyBalance(Decimal("0"), Decimal("0"))
        return balances.get("BTC", zero), balances.get("USDT", zero)

    def get_open_orders(self) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        ids: set[str] = set()
        for _ in range(10):
            params = {"instType": "SPOT", "limit": "100"}
            if orders:
                params["after"] = orders[-1]["ordId"]
            page = self._get(self._PENDING, params)
            if len(page) > 100:
                raise DemoError("Demo API вернул слишком много открытых ордеров")
            for row in page:
                order_id = row.get("ordId")
                if (row.get("instType") != "SPOT" or not isinstance(row.get("instId"), str)
                        or not row["instId"] or not isinstance(order_id, str) or not order_id
                        or order_id in ids or row.get("state") not in ("live", "partially_filled")):
                    raise DemoError("Demo API вернул противоречивый открытый ордер")
                ids.add(order_id)
                orders.append(row)
            if len(page) < 100:
                return orders
        raise DemoError("Не удалось полностью прочитать открытые Demo-ордера")

    def get_fills_history(self) -> list[dict[str, Any]]:
        """Return at most 100 recent SPOT fills, subject to OKX history retention."""
        rows = self._get(self._FILLS, {"instType": "SPOT", "limit": "100"})
        if len(rows) > 100:
            raise DemoError("Demo API вернул слишком много записей исполнений")
        for row in rows:
            if (row.get("instType") != "SPOT" or not isinstance(row.get("instId"), str)
                    or not row["instId"] or not isinstance(row.get("tradeId"), str)
                    or not row["tradeId"]):
                raise DemoError("Demo API вернул некорректную историю исполнений")
        return rows

    def get_spot_fee_rates(self) -> SpotFeeRates:
        """Read the account's BTC-USDT SPOT fee schedule, without estimating slippage."""
        rows = self._get(self._TRADE_FEE, {"instType": "SPOT", "instId": "BTC-USDT"})
        if len(rows) != 1:
            raise DemoError("Demo API вернул некорректные ставки комиссий")
        row = rows[0]
        return SpotFeeRates(_signed_rate(row.get("maker"), "maker"),
                            _signed_rate(row.get("taker"), "taker"))

    def check_account(self) -> DemoSummary:
        btc, usdt = self.get_balances()
        balance_checked_at = datetime.now(timezone.utc)
        orders = self.get_open_orders()
        fills = self.get_fills_history()
        return DemoSummary(btc, usdt, len(orders), len(fills), balance_checked_at)
