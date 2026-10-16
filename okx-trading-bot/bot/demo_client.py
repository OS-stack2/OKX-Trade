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


class DemoApiCodeError(DemoError):
    """OKX answered with a definite non-zero business code."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


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


@dataclass(frozen=True)
class DemoOrder:
    client_order_id: str
    order_id: str
    state: str
    size_btc: Decimal
    filled_btc: Decimal
    limit_price_usdt: Decimal
    checked_at_utc: datetime


@dataclass(frozen=True)
class DemoFill:
    trade_id: str
    order_id: str
    size_btc: Decimal
    price_usdt: Decimal
    fee: Decimal
    fee_currency: str
    observed_at_utc: datetime


@dataclass(frozen=True)
class DemoAlgoOrder:
    """Read-only Algo observation; no parent/OCO proof is inferred from it."""

    algo_id: str
    order_type: str
    state: str
    size_btc: Decimal | None
    side: str


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
    _ORDER_INFO = "/api/v5/trade/order"
    _ORDER_FILLS = "/api/v5/trade/fills"
    _ALGO_PENDING = "/api/v5/trade/orders-algo-pending"
    _ALGO_INFO = "/api/v5/trade/order-algo"
    _ALLOWED_PATHS = frozenset({_BALANCE, _PENDING, _FILLS, _TRADE_FEE,
                                _ORDER_INFO, _ORDER_FILLS, _ALGO_PENDING, _ALGO_INFO})
    _CREDENTIAL_NAMES = ("OKX_DEMO_API_KEY", "OKX_DEMO_API_SECRET", "OKX_DEMO_PASSPHRASE")

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
        key_name, secret_name, passphrase_name = cls._credential_names(values)
        credentials = DemoCredentials(
            values.get(key_name, ""),
            values.get(secret_name, ""),
            values.get(passphrase_name, ""),
        )
        return cls(config, credentials)

    @classmethod
    def _credential_names(cls, values: dict[str, str]) -> tuple[str, str, str]:
        return cls._CREDENTIAL_NAMES

    @property
    def config(self) -> Config:
        """Non-secret regional configuration for matching public requests."""
        return self._config

    def _timestamp(self) -> str:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise DemoError("Не удалось получить время UTC для подписи Demo-запроса")
        return now.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _signed_headers(self, method: str, request_path: str, body: str = "") -> dict[str, str]:
        timestamp = self._timestamp()
        prehash = timestamp + method + request_path + body
        signature = base64.b64encode(
            hmac.new(self._credentials.secret.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256).digest()
        ).decode("ascii")
        return {
            "Accept": "application/json",
            "User-Agent": "okx-trading-bot/0.1",
            "OK-ACCESS-KEY": self._credentials.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self._credentials.passphrase,
            **demo_request_headers(self._config),
        }

    def _get(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        if path not in self._ALLOWED_PATHS:
            raise DemoError("Запрещённый приватный endpoint")
        query = urlencode(params)
        request_path = path + (f"?{query}" if query else "")
        url = self._config.base_url + request_path
        for attempt in range(self._config.max_attempts):
            request = Request(url, headers=self._signed_headers("GET", request_path), method="GET")
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
                    raise DemoApiCodeError(f"Demo API отклонил запрос (код {code}); проверьте Demo-ключи и права", code)
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

    def get_order_by_client_id(self, client_order_id: str) -> DemoOrder:
        """Read exactly one Demo BTC-USDT cash LIMIT BUY by its persisted ID."""
        if not isinstance(client_order_id, str) or not re.fullmatch(r"TB[a-f0-9]{30}", client_order_id):
            raise DemoError("Некорректный clientOrderId для Demo-сверки")
        rows = self._get(self._ORDER_INFO, {"instId": "BTC-USDT", "clOrdId": client_order_id})
        if len(rows) != 1:
            raise DemoError("Demo-ордер по clientOrderId не найден однозначно")
        row = rows[0]
        order_id = row.get("ordId")
        if (row.get("instType") != "SPOT" or row.get("instId") != "BTC-USDT"
                or row.get("tdMode") != "cash" or row.get("side") != "buy"
                or row.get("ordType") != "limit" or row.get("clOrdId") != client_order_id
                or not isinstance(order_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", order_id)
                or row.get("state") not in ("live", "partially_filled", "filled", "canceled")):
            raise DemoError("Demo-ордер противоречит ожидаемому BTC-USDT LIMIT BUY")
        size = _amount(row.get("sz"), "размер ордера")
        filled = _amount(row.get("accFillSz"), "исполненный размер")
        price = _amount(row.get("px"), "цена ордера")
        if (size <= 0 or price <= 0 or filled > size
                or (row["state"] == "live" and filled != 0)
                or (row["state"] == "partially_filled" and not 0 < filled < size)
                or (row["state"] == "filled" and filled != size)):
            raise DemoError("Размер, цена или состояние Demo-ордера противоречивы")
        return DemoOrder(client_order_id, order_id, row["state"], size, filled,
                         price, self._clock())

    def get_order_fills(self, order_id: str) -> tuple[DemoFill, ...]:
        """Read recent per-order fills; a full page is treated as incomplete."""
        if not isinstance(order_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", order_id):
            raise DemoError("Некорректный ordId для Demo-сверки")
        rows = self._get(self._ORDER_FILLS, {"instType": "SPOT", "instId": "BTC-USDT",
                                             "ordId": order_id, "limit": "100"})
        if len(rows) >= 100:
            raise DemoError("История исполнений ордера достигла лимита страницы; полнота не доказана")
        fills: list[DemoFill] = []
        seen: set[str] = set()
        for row in rows:
            trade_id = row.get("tradeId")
            fee_ccy = row.get("feeCcy")
            if (row.get("instType") != "SPOT" or row.get("instId") != "BTC-USDT"
                    or row.get("ordId") != order_id or row.get("side") != "buy"
                    or not isinstance(trade_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", trade_id)
                    or trade_id in seen or fee_ccy not in ("BTC", "USDT")):
                raise DemoError("История исполнений Demo-ордера противоречива")
            seen.add(trade_id)
            size = _amount(row.get("fillSz"), "размер исполнения")
            price = _amount(row.get("fillPx"), "цена исполнения")
            try:
                fee = Decimal(row.get("fee"))
                observed = datetime.fromtimestamp(int(row["ts"]) / 1000, tz=timezone.utc)
            except (InvalidOperation, TypeError, ValueError, KeyError, OverflowError, OSError) as exc:
                raise DemoError("Комиссия исполнения Demo-ордера неизвестна") from exc
            if (size <= 0 or price <= 0 or not fee.is_finite() or fee > 0
                    or observed > self._clock()):
                raise DemoError("Размер, цена или комиссия исполнения Demo-ордера некорректны")
            fills.append(DemoFill(trade_id, order_id, size, price, fee, fee_ccy, observed))
        return tuple(fills)

    @staticmethod
    def _algo_row(row: dict[str, Any], *, expected_id: str | None = None,
                  expected_type: str | None = None) -> DemoAlgoOrder:
        algo_id = row.get("algoId")
        order_type = row.get("ordType")
        state = row.get("state")
        side = row.get("side")
        if (row.get("instType") != "SPOT" or row.get("instId") != "BTC-USDT"
                or not isinstance(algo_id, str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", algo_id) is None
                or (expected_id is not None and algo_id != expected_id)
                or not isinstance(order_type, str) or not order_type
                or (expected_type is not None and order_type != expected_type)
                or not isinstance(state, str) or not state
                or side not in ("buy", "sell")):
            raise DemoError("Demo Algo-ордер противоречит BTC-USDT SPOT или его идентификатору")
        raw_size = row.get("sz")
        size = None if raw_size in (None, "") else _amount(raw_size, "размер Algo-ордера")
        return DemoAlgoOrder(algo_id, order_type, state, size, side)

    def get_pending_algo_orders(self) -> tuple[DemoAlgoOrder, ...]:
        """Read conditional and OCO pending Algo pages, without inferring coverage.

        These two order types do not establish that all possible attached types
        were returned. A full page always requires a further GET page.
        """
        orders: list[DemoAlgoOrder] = []
        seen: set[str] = set()
        for order_type in ("conditional", "oco"):
            after = ""
            for _ in range(10):
                params = {"ordType": order_type, "instType": "SPOT",
                          "instId": "BTC-USDT", "limit": "100"}
                if after:
                    params["after"] = after
                page = self._get(self._ALGO_PENDING, params)
                if len(page) > 100:
                    raise DemoError("Demo API вернул слишком много Algo-ордеров")
                for row in page:
                    order = self._algo_row(row, expected_type=order_type)
                    if order.algo_id in seen:
                        raise DemoError("Demo API вернул повторный Algo ID")
                    seen.add(order.algo_id)
                    orders.append(order)
                if len(page) < 100:
                    break
                after = orders[-1].algo_id
            else:
                raise DemoError("Полнота списка Demo Algo-ордеров не доказана")
        return tuple(orders)

    def get_algo_order_by_id(self, algo_id: str) -> DemoAlgoOrder:
        """Read one saved exchange Algo ID; never search by guessed IDs."""
        if not isinstance(algo_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", algo_id) is None:
            raise DemoError("Некорректный Algo ID для Demo-сверки")
        rows = self._get(self._ALGO_INFO, {"algoId": algo_id})
        if len(rows) != 1:
            raise DemoError("Demo Algo ID не найден однозначно")
        return self._algo_row(rows[0], expected_id=algo_id)

    def check_account(self) -> DemoSummary:
        btc, usdt = self.get_balances()
        balance_checked_at = datetime.now(timezone.utc)
        orders = self.get_open_orders()
        fills = self.get_fills_history()
        return DemoSummary(btc, usdt, len(orders), len(fills), balance_checked_at)
