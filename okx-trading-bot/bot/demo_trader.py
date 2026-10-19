"""Continuous OKX Demo BTC-USDT SPOT trader, one cycle per process run.

Each `run_once` call (driven by a systemd timer every 15 minutes):
1. refuses to act while a durable HALT is stored;
2. reconciles the single tracked position with the exchange (OCO hit, manual
   close, in-flight orders after a crash);
3. evaluates the EMA signal on the newest confirmed candle exactly once;
4. on BUY places a market buy sized by risk, then protects the *filled* net BTC
   with an exchange-side OCO (SL/TP). If protection cannot be placed, the BTC is
   sold at market and the trader halts;
5. on SELL cancels the OCO and sells the position at market.

Only Demo mode with the `x-simulated-trading` header is possible: the client
inherits DemoClient, which refuses any non-DEMO configuration. The bot only ever
sells BTC it bought itself; other balances on the Demo account are untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request
from uuid import uuid4

from bot.demo_client import DemoApiCodeError, DemoClient, DemoError
from bot.market_data import MarketSnapshot, SpotInstrument
from bot.risk_manager import RiskSettings
from bot.strategy import Signal, StrategyError, StrategyResult, evaluate_signal
from config import ConfigError, _read_env_file


INST_ID = "BTC-USDT"
DEFAULT_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_trader.sqlite3"
RESUME_PHRASE = "RESUME OKX DEMO TRADER"
DEFAULT_BUDGET_USDT = Decimal("5000")
DEFAULT_SLIPPAGE = Decimal("0.0005")
MAX_SIGNAL_DELAY = timedelta(minutes=20)
ORDER_NOT_FOUND_CODES = frozenset({"51603"})
_CLIENT_ID = re.compile(r"DT[a-f0-9]{30}\Z")
_EXCHANGE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
ZERO = Decimal(0)


class TraderError(Exception):
    """A contradiction or unsafe condition; the caller halts the trader."""


class OrderUncertain(DemoError):
    """A POST may or may not have reached OKX; never resend blindly."""


def _dec(value: Any, label: str, *, allow_zero: bool = True, signed: bool = False) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise TraderError(f"OKX вернул некорректное поле {label}") from exc
    if not number.is_finite() or (not signed and number < 0) or (not allow_zero and number == 0):
        raise TraderError(f"OKX вернул некорректное поле {label}")
    return number


def _floor(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _round_tick(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    return (value / step).to_integral_value(rounding=rounding) * step


@dataclass(frozen=True)
class OrderStatus:
    order_id: str
    client_order_id: str
    side: str
    state: str
    filled_btc: Decimal
    avg_price: Decimal
    fee: Decimal          # signed, negative = charged
    fee_currency: str

    @property
    def terminal(self) -> bool:
        return self.state in ("filled", "canceled", "mmp_canceled")


@dataclass(frozen=True)
class AlgoStatus:
    algo_id: str
    state: str
    order_ids: tuple[str, ...]
    actual_side: str


class DemoTradeClient(DemoClient):
    """DemoClient plus a minimal allowlist of trading POST endpoints."""

    _PLACE_ORDER = "/api/v5/trade/order"
    _PLACE_ALGO = "/api/v5/trade/order-algo"
    _CANCEL_ALGOS = "/api/v5/trade/cancel-algos"
    _POST_PATHS = frozenset({_PLACE_ORDER, _PLACE_ALGO, _CANCEL_ALGOS})
    _CREDENTIAL_NAMES = ("OKX_DEMO_TRADE_API_KEY", "OKX_DEMO_TRADE_API_SECRET",
                         "OKX_DEMO_TRADE_PASSPHRASE")

    @classmethod
    def _credential_names(cls, values: dict[str, str]) -> tuple[str, str, str]:
        """Prefer a dedicated Trade key; otherwise reuse the existing Demo key."""
        if any(values.get(name) for name in cls._CREDENTIAL_NAMES):
            return cls._CREDENTIAL_NAMES
        return DemoClient._CREDENTIAL_NAMES

    def _post(self, path: str, body: Any) -> dict[str, Any]:
        """Single attempt: a lost response is reported as OrderUncertain."""
        if path not in self._POST_PATHS:
            raise DemoError("Запрещённый торговый endpoint")
        payload_text = json.dumps(body, separators=(",", ":"))
        headers = {**self._signed_headers("POST", path, payload_text), "Content-Type": "application/json"}
        request = Request(self._config.base_url + path, data=payload_text.encode("utf-8"),
                          headers=headers, method="POST")
        try:
            payload = self._transport(request, self._config.timeout_seconds)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if status == 429 or status >= 500:
                raise OrderUncertain(f"Demo API HTTP {status} на торговый запрос") from None
            raise DemoError(f"Demo API отклонил торговый запрос (HTTP {status})") from None
        except (URLError, TimeoutError, ConnectionError, HTTPException, OSError):
            raise OrderUncertain("Ответ на торговый запрос не получен") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise OrderUncertain("Некорректный JSON в ответе на торговый запрос") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("code"), str):
            raise OrderUncertain("Некорректная структура ответа на торговый запрос")
        data = payload.get("data")
        row = data[0] if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) else None
        if payload["code"] == "0" and row is not None and row.get("sCode") == "0":
            return row
        s_code = row.get("sCode") if row is not None else None
        code = s_code if isinstance(s_code, str) and s_code else payload["code"]
        raise DemoApiCodeError(f"OKX отклонил торговый запрос (код {code})", str(code))

    # --- orders -----------------------------------------------------------------
    def place_market_order(self, *, client_order_id: str, side: str, size_btc: Decimal) -> str:
        if not _CLIENT_ID.fullmatch(client_order_id) or side not in ("buy", "sell") or size_btc <= 0:
            raise DemoError("Некорректные параметры рыночного ордера")
        row = self._post(self._PLACE_ORDER, {
            "instId": INST_ID, "tdMode": "cash", "clOrdId": client_order_id, "side": side,
            "ordType": "market", "sz": format(size_btc, "f"), "tgtCcy": "base_ccy",
        })
        order_id = row.get("ordId")
        if not isinstance(order_id, str) or not _EXCHANGE_ID.fullmatch(order_id):
            raise OrderUncertain("OKX не вернул ordId")
        return order_id

    def get_order(self, client_order_id: str) -> OrderStatus | None:
        """Exact lookup by our clOrdId; None only when OKX says it does not exist."""
        try:
            rows = self._get(self._ORDER_INFO, {"instId": INST_ID, "clOrdId": client_order_id})
        except DemoApiCodeError as exc:
            if exc.code in ORDER_NOT_FOUND_CODES:
                return None
            raise
        if len(rows) != 1:
            raise TraderError("Ордер по clOrdId не найден однозначно")
        return self._order(rows[0], client_order_id=client_order_id)

    def get_order_by_id(self, order_id: str) -> OrderStatus:
        rows = self._get(self._ORDER_INFO, {"instId": INST_ID, "ordId": order_id})
        if len(rows) != 1:
            raise TraderError("Ордер по ordId не найден однозначно")
        return self._order(rows[0], order_id=order_id)

    @staticmethod
    def _order(row: dict[str, Any], *, client_order_id: str | None = None,
               order_id: str | None = None) -> OrderStatus:
        oid = row.get("ordId")
        if (row.get("instId") != INST_ID or row.get("side") not in ("buy", "sell")
                or not isinstance(oid, str) or not _EXCHANGE_ID.fullmatch(oid)
                or (order_id is not None and oid != order_id)
                or (client_order_id is not None and row.get("clOrdId") != client_order_id)
                or not isinstance(row.get("state"), str)):
            raise TraderError("Ордер OKX противоречит ожидаемому BTC-USDT")
        filled = _dec(row.get("accFillSz") or "0", "accFillSz")
        avg = _dec(row.get("avgPx") or "0", "avgPx")
        fee = _dec(row.get("fee") or "0", "fee", signed=True)
        fee_ccy = row.get("feeCcy") or ""
        if filled > 0 and (avg <= 0 or fee_ccy not in ("BTC", "USDT")):
            raise TraderError("Цена или валюта комиссии исполнения неизвестны")
        return OrderStatus(oid, row.get("clOrdId") or "", row["side"], row["state"],
                           filled, avg, fee, fee_ccy)

    # --- OCO protection ---------------------------------------------------------
    def place_oco(self, *, algo_client_id: str, size_btc: Decimal,
                  take_profit: Decimal, stop_loss: Decimal) -> str:
        row = self._post(self._PLACE_ALGO, {
            "instId": INST_ID, "tdMode": "cash", "side": "sell", "ordType": "oco",
            "sz": format(size_btc, "f"), "algoClOrdId": algo_client_id,
            "tpTriggerPx": format(take_profit, "f"), "tpOrdPx": "-1",
            "slTriggerPx": format(stop_loss, "f"), "slOrdPx": "-1",
        })
        algo_id = row.get("algoId")
        if not isinstance(algo_id, str) or not _EXCHANGE_ID.fullmatch(algo_id):
            raise OrderUncertain("OKX не вернул algoId")
        return algo_id

    def get_algo(self, algo_id: str) -> AlgoStatus:
        rows = self._get(self._ALGO_INFO, {"algoId": algo_id})
        if len(rows) != 1:
            raise TraderError("OCO не найден однозначно")
        row = rows[0]
        if row.get("algoId") != algo_id or row.get("instId") != INST_ID or not isinstance(row.get("state"), str):
            raise TraderError("OCO противоречит сохранённому algoId")
        ids = row.get("ordIdList") or ([row["ordId"]] if row.get("ordId") else [])
        if not isinstance(ids, list) or any(not isinstance(i, str) or not _EXCHANGE_ID.fullmatch(i) for i in ids):
            raise TraderError("OCO вернул некорректные ордера")
        return AlgoStatus(algo_id, row["state"], tuple(ids), str(row.get("actualSide") or ""))

    def cancel_algo(self, algo_id: str) -> None:
        self._post(self._CANCEL_ALGOS, [{"algoId": algo_id, "instId": INST_ID}])


# --- durable state ------------------------------------------------------------

@dataclass(frozen=True)
class Position:
    id: int
    state: str                    # OPENING, OPEN, CLOSING, CLOSED, VOID
    candle_closed_at: str
    buy_client_id: str
    planned_btc: Decimal
    net_btc: Decimal | None
    entry_price: Decimal | None
    entry_cost_usdt: Decimal | None
    stop_price: Decimal | None
    take_price: Decimal | None
    algo_id: str | None
    sell_client_id: str | None
    opened_at: str
    closed_at: str | None
    pnl_usdt: Decimal | None
    exit_reason: str | None


class TraderStore:
    def __init__(self, path: Path = DEFAULT_STATE_PATH) -> None:
        self.path = path

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5, isolation_level=None)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute("""CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
                db.execute("""CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, state TEXT NOT NULL,
                    candle_closed_at TEXT NOT NULL UNIQUE, buy_client_id TEXT NOT NULL UNIQUE,
                    planned_btc TEXT NOT NULL, net_btc TEXT, entry_price TEXT, entry_cost_usdt TEXT,
                    stop_price TEXT, take_price TEXT, algo_id TEXT, sell_client_id TEXT,
                    opened_at TEXT NOT NULL, closed_at TEXT, pnl_usdt TEXT, exit_reason TEXT)""")
                db.execute("""CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, kind TEXT NOT NULL,
                    detail TEXT NOT NULL)""")
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _position(row: sqlite3.Row | None) -> Position | None:
        if row is None:
            return None
        def d(key: str) -> Decimal | None:
            return None if row[key] is None else Decimal(row[key])
        return Position(row["id"], row["state"], row["candle_closed_at"], row["buy_client_id"],
                        Decimal(row["planned_btc"]), d("net_btc"), d("entry_price"),
                        d("entry_cost_usdt"), d("stop_price"), d("take_price"), row["algo_id"],
                        row["sell_client_id"], row["opened_at"], row["closed_at"], d("pnl_usdt"),
                        row["exit_reason"])

    def get_meta(self, key: str) -> str | None:
        with self._db() as db:
            row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return None if row is None else row["value"]

    def set_meta(self, key: str, value: str | None) -> None:
        with self._db() as db:
            if value is None:
                db.execute("DELETE FROM meta WHERE key=?", (key,))
            else:
                db.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                           "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def event(self, now: datetime, kind: str, detail: str) -> None:
        with self._db() as db:
            db.execute("INSERT INTO events(at, kind, detail) VALUES(?, ?, ?)", (now.isoformat(), kind, detail))

    def active_position(self) -> Position | None:
        with self._db() as db:
            rows = db.execute("SELECT * FROM positions WHERE state IN ('OPENING','OPEN','CLOSING') "
                              "ORDER BY id").fetchall()
            if len(rows) > 1:
                raise TraderError("В журнале больше одной активной позиции")
            return self._position(rows[0] if rows else None)

    def closed_positions(self) -> list[Position]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM positions WHERE state='CLOSED' ORDER BY id").fetchall()
            return [p for p in map(self._position, rows) if p is not None]

    def entries_on(self, day: str) -> int:
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM positions WHERE state!='VOID' AND substr(opened_at,1,10)=?",
                              (day,)).fetchone()[0]

    def create_opening(self, *, candle_closed_at: datetime, client_id: str, planned_btc: Decimal,
                       now: datetime) -> Position:
        with self._db() as db:
            db.execute("INSERT INTO positions(state, candle_closed_at, buy_client_id, planned_btc, opened_at) "
                       "VALUES('OPENING', ?, ?, ?, ?)",
                       (candle_closed_at.isoformat(), client_id, format(planned_btc, "f"), now.isoformat()))
            row = db.execute("SELECT * FROM positions WHERE buy_client_id=?", (client_id,)).fetchone()
        position = self._position(row)
        assert position is not None
        return position

    def update(self, position_id: int, **fields: Any) -> None:
        allowed = {"state", "net_btc", "entry_price", "entry_cost_usdt", "stop_price", "take_price",
                   "algo_id", "sell_client_id", "closed_at", "pnl_usdt", "exit_reason"}
        if not fields or set(fields) - allowed:
            raise ValueError("Unsupported position update")
        values = [format(v, "f") if isinstance(v, Decimal) else v for v in fields.values()]
        with self._db() as db:
            db.execute(f"UPDATE positions SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                       (*values, position_id))


# --- trader -------------------------------------------------------------------

@dataclass(frozen=True)
class TraderSettings:
    budget_usdt: Decimal = DEFAULT_BUDGET_USDT
    slippage: Decimal = DEFAULT_SLIPPAGE
    risk: RiskSettings = RiskSettings()
    order_poll_attempts: int = 10
    order_poll_delay: float = 0.5

    @classmethod
    def from_env_file(cls, path: Path | None = None) -> TraderSettings:
        values = _read_env_file(path or Path(__file__).resolve().parents[1] / ".env")
        if values.get("OKX_TRADER_ENABLED") != "1":
            raise ConfigError("Demo-трейдер выключен: установите OKX_TRADER_ENABLED=1 в .env")
        try:
            budget = Decimal(values.get("OKX_TRADER_BUDGET_USDT", str(DEFAULT_BUDGET_USDT)))
            slippage = Decimal(values.get("OKX_TRADER_SLIPPAGE", str(DEFAULT_SLIPPAGE)))
        except InvalidOperation as exc:
            raise ConfigError("Некорректный OKX_TRADER_BUDGET_USDT или OKX_TRADER_SLIPPAGE") from exc
        if not budget.is_finite() or not 10 <= budget <= 1_000_000:
            raise ConfigError("OKX_TRADER_BUDGET_USDT должен быть от 10 до 1000000")
        if not slippage.is_finite() or not 0 <= slippage <= Decimal("0.01"):
            raise ConfigError("OKX_TRADER_SLIPPAGE должен быть от 0 до 0.01")
        return cls(budget_usdt=budget, slippage=slippage)


@dataclass(frozen=True)
class CycleReport:
    status: str               # HALTED, HOLD, OPENED, CLOSED, SKIPPED, ERROR
    lines: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return 1 if self.status in ("HALTED", "ERROR") else 0


def trader_state_path_from_env(path: Path | None = None) -> Path:
    values = _read_env_file(path or Path(__file__).resolve().parents[1] / ".env")
    raw = values.get("OKX_TRADER_STATE_PATH")
    return Path(raw) if raw else DEFAULT_STATE_PATH


class DemoTrader:
    CLOSE_REASON = "SIGNAL"    # exit reason for bot-initiated closes
    RESUME_COMMAND = "--demo-trade-resume"
    def __init__(self, *, client: DemoTradeClient, store: TraderStore,
                 market_provider: Callable[[], MarketSnapshot], settings: TraderSettings = TraderSettings(),
                 notifier: Callable[[str], object] | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 sleeper: Callable[[float], None] | None = None,
                 signal_provider: Callable[[tuple], StrategyResult] = evaluate_signal,
                 brake: Callable[[datetime], str | None] | None = None) -> None:
        self.client = client
        self.store = store
        self.market_provider = market_provider
        self.settings = settings
        self.notifier = notifier
        self.clock = clock
        self.sleep = sleeper or time.sleep
        self.signal_provider = signal_provider
        self.brake = brake
        self._lines: list[str] = []

    # helpers
    def _say(self, line: str) -> None:
        self._lines.append(line)

    def _notify(self, text: str) -> None:
        if self.notifier is not None:
            try:
                self.notifier("OKX Demo BTC-USDT: " + text)
            except Exception:
                pass

    def _halt(self, reason: str) -> CycleReport:
        now = self.clock()
        self.store.set_meta("halted", f"{now.isoformat()} {reason}")
        self.store.event(now, "HALT", reason)
        self._notify(f"АВАРИЙНАЯ ОСТАНОВКА. {reason}. Новые сделки запрещены до ручной проверки "
                     f"и команды {self.RESUME_COMMAND}.")
        self._say(f"HALTED: {reason}")
        return CycleReport("HALTED", tuple(self._lines))

    def equity(self) -> Decimal:
        return self.settings.budget_usdt + sum((p.pnl_usdt or ZERO) for p in self.store.closed_positions())

    def _day_start_equity(self, day: str) -> Decimal:
        return self.settings.budget_usdt + sum(
            (p.pnl_usdt or ZERO) for p in self.store.closed_positions() if (p.closed_at or "")[:10] < day)

    def _realized_today(self, day: str) -> Decimal:
        return sum(((p.pnl_usdt or ZERO) for p in self.store.closed_positions()
                    if (p.closed_at or "")[:10] == day), ZERO)

    def _wait_terminal(self, client_id: str) -> OrderStatus | None:
        status = None
        for _ in range(self.settings.order_poll_attempts):
            status = self.client.get_order(client_id)
            if status is not None and status.terminal:
                return status
            self.sleep(self.settings.order_poll_delay)
        return status

    # main entry
    def run_once(self) -> CycleReport:
        self._lines = []
        halted = self.store.get_meta("halted")
        if halted:
            self._say(f"HALTED с {halted}")
            return CycleReport("HALTED", tuple(self._lines))
        try:
            closed = self._reconcile()
            position = self.store.active_position()
            if position is not None and position.state != "OPEN":
                self._say(f"Позиция в состоянии {position.state}; ожидание сверки на следующем цикле")
                return CycleReport("SKIPPED", tuple(self._lines))
            market = self.market_provider()
            result = self.signal_provider(market.candles)
            now = self.clock()
            if now - result.candle_closed_at > MAX_SIGNAL_DELAY:
                self._say("Последняя закрытая свеча устарела; действий нет")
                return CycleReport("SKIPPED", tuple(self._lines))
            candle_key = result.candle_closed_at.isoformat()
            if self.store.get_meta("last_candle") == candle_key:
                self._say("Эта свеча уже обработана")
                return CycleReport("CLOSED" if closed else "SKIPPED", tuple(self._lines))
            self._say(f"Сигнал {result.signal} на свече {candle_key}: EMA9 {result.ema9:.2f}, "
                      f"EMA21 {result.ema21:.2f}; цена {market.last_price}")
            status = "CLOSED" if closed else "HOLD"
            if result.signal == Signal.SELL and position is not None:
                self._close_by_signal(position)
                status = "CLOSED"
            elif result.signal == Signal.BUY and position is None:
                status = self._open(market, result)
            self.store.set_meta("last_candle", candle_key)
            self._say(f"Капитал стратегии: {self.equity():.2f} USDT")
            return CycleReport(status, tuple(self._lines))
        except TraderError as exc:
            return self._halt(str(exc))
        except OrderUncertain as exc:
            return self._halt(f"Неопределённый результат торгового запроса: {exc}")
        except (DemoError, StrategyError, ConfigError, ValueError, OSError) as exc:
            # Read failures are retried on the next timer run; nothing was sent.
            self._say(f"Цикл пропущен: {exc}")
            return CycleReport("ERROR", tuple(self._lines))

    def close_now(self) -> CycleReport:
        """Owner-requested exit: reconcile, then cancel the OCO and sell the position at market."""
        self._lines = []
        if self.store.get_meta("halted"):
            return CycleReport("HALTED", ("Сначала снимите HALT после ручной проверки",))
        try:
            self._reconcile()
            position = self.store.active_position()
            if position is None:
                return CycleReport("HOLD", tuple(self._lines) + ("Открытой позиции нет",))
            if position.state != "OPEN":
                raise TraderError(f"Позиция в состоянии {position.state}; закрыть вручную нельзя")
            reason = self.CLOSE_REASON
            self.CLOSE_REASON = "MANUAL"
            try:
                self._close_by_signal(position)
            finally:
                self.CLOSE_REASON = reason
            return CycleReport("CLOSED", tuple(self._lines))
        except TraderError as exc:
            return self._halt(str(exc))
        except OrderUncertain as exc:
            return self._halt(f"Неопределённый результат торгового запроса: {exc}")
        except DemoError as exc:
            return CycleReport("ERROR", (f"Закрытие не выполнено: {exc}",))

    # --- reconciliation -----------------------------------------------------------
    def _reconcile(self) -> bool:
        position = self.store.active_position()
        if position is None:
            return False
        if position.state == "OPENING":
            order = self.client.get_order(position.buy_client_id)
            if order is None:
                opened = datetime.fromisoformat(position.opened_at)
                if self.clock() - opened < timedelta(minutes=2):
                    raise TraderError("Покупка не найдена на бирже, но ещё может появиться")
                self.store.update(position.id, state="VOID", exit_reason="NOT_PLACED")
                self._say("Незавершённая покупка не дошла до OKX; запись аннулирована")
                return False
            if not order.terminal:
                raise TraderError("Рыночная покупка не завершилась")
            self._protect(position, order, self.market_provider().instrument)
            return False
        if position.state == "CLOSING":
            assert position.sell_client_id is not None
            order = self.client.get_order(position.sell_client_id)
            if order is None or not order.terminal:
                raise TraderError("Состояние продажи позиции неизвестно")
            self._finish(position, order, self.CLOSE_REASON)
            return True
        # OPEN: check whether the OCO triggered.
        assert position.algo_id is not None
        algo = self.client.get_algo(position.algo_id)
        if algo.state in ("live", "pause"):
            self._say(f"Позиция открыта: {position.net_btc} BTC от {position.entry_price}, "
                      f"SL {position.stop_price}, TP {position.take_price}")
            return False
        if algo.state == "effective" and algo.order_ids:
            order = self.client.get_order_by_id(algo.order_ids[-1])
            if not order.terminal:
                self._say("OCO сработал, продажа исполняется")
                return False
            reason = {"tp": "TAKE_PROFIT", "sl": "STOP_LOSS"}.get(algo.actual_side, "OCO")
            self._finish(position, order, reason)
            return True
        raise TraderError(f"Защитный OCO в состоянии {algo.state}; позиция может быть без защиты")

    # --- entry --------------------------------------------------------------------
    def _open(self, market: MarketSnapshot, result: StrategyResult) -> str:
        now = self.clock()
        day = now.date().isoformat()
        risk = self.settings.risk
        reason = self.brake(now) if self.brake is not None else None
        if reason:
            self._say(f"BUY пропущен: тормоз по новостям — {reason}")
            return "HOLD"
        if self.store.entries_on(day) >= risk.max_new_entries_per_day:
            self._say("BUY пропущен: лимит входов за сутки")
            return "HOLD"
        day_start = self._day_start_equity(day)
        realized = self._realized_today(day)
        loss_limit = day_start * risk.daily_loss_limit_pct
        if realized <= -loss_limit:
            self._say(f"BUY пропущен: дневной убыток {realized:.2f} USDT достиг лимита")
            return "HOLD"
        instrument = market.instrument
        if instrument.state not in (None, "live"):
            self._say(f"BUY пропущен: инструмент в состоянии {instrument.state}")
            return "HOLD"
        fee = abs(self.client.get_spot_fee_rates().taker)
        btc, usdt = self.client.get_balances()
        price = market.last_price
        entry_unit = price * (1 + fee + self.settings.slippage)
        equity = self.equity()
        quantity = self._size(price=price, fee=fee, equity=equity,
                              risk_left=loss_limit + realized)
        spendable = min(equity, usdt.available * Decimal("0.98"))
        quantity = min(quantity, spendable / entry_unit)
        if instrument.max_market_size is not None:
            quantity = min(quantity, instrument.max_market_size / entry_unit)
        quantity = _floor(quantity, instrument.lot_size)
        if quantity < instrument.min_size:
            self._say(f"BUY пропущен: размер {quantity} BTC меньше минимального {instrument.min_size}")
            return "HOLD"
        client_id = "DT" + uuid4().hex[:30]
        position = self.store.create_opening(candle_closed_at=result.candle_closed_at, client_id=client_id,
                                             planned_btc=quantity, now=now)
        self.store.event(now, "BUY_SENT", f"{client_id} {quantity}")
        try:
            self.client.place_market_order(client_order_id=client_id, side="buy", size_btc=quantity)
        except DemoApiCodeError as exc:
            self.store.update(position.id, state="VOID", exit_reason=f"REJECTED {exc.code}")
            self._say(f"BUY отклонён OKX (код {exc.code}); средства не потрачены")
            return "HOLD"
        order = self._wait_terminal(client_id)
        if order is None or not order.terminal:
            raise TraderError("Рыночная покупка не подтвердилась вовремя")
        return "OPENED" if self._protect(position, order, instrument) else "HOLD"

    def _size(self, *, price: Decimal, fee: Decimal, equity: Decimal, risk_left: Decimal) -> Decimal:
        """BTC quantity before balance and lot limits: risk-per-trade at the stop."""
        risk = self.settings.risk
        cost_rate = fee + self.settings.slippage
        entry_unit = price * (1 + cost_rate)
        stop_unit = price * (1 - risk.stop_loss_pct) * (1 - cost_rate)
        return min(equity * risk.risk_per_trade, risk_left) / (entry_unit - stop_unit)

    def _protect(self, position: Position, order: OrderStatus, instrument: SpotInstrument) -> bool:
        if order.side != "buy":
            raise TraderError("Ордер входа оказался не покупкой")
        if order.filled_btc == 0:
            self.store.update(position.id, state="VOID", exit_reason="NOT_FILLED")
            self._say("Покупка не исполнилась")
            return False
        fee_btc = -order.fee if order.fee_currency == "BTC" else ZERO
        fee_usdt = -order.fee if order.fee_currency == "USDT" else ZERO
        net = order.filled_btc - fee_btc
        size = _floor(net, instrument.lot_size)
        cost = order.filled_btc * order.avg_price + fee_usdt
        risk = self.settings.risk
        stop = _round_tick(order.avg_price * (1 - risk.stop_loss_pct), instrument.tick_size, ROUND_DOWN)
        take = _round_tick(order.avg_price * (1 + risk.take_profit_pct), instrument.tick_size, ROUND_UP)
        self.store.update(position.id, net_btc=size, entry_price=order.avg_price, entry_cost_usdt=cost,
                          stop_price=stop, take_price=take)
        if size < instrument.min_size:
            raise TraderError(f"Купленный объём {size} BTC меньше минимума для защиты")
        try:
            algo_id = self.client.place_oco(algo_client_id="DA" + position.buy_client_id[2:],
                                            size_btc=size, take_profit=take, stop_loss=stop)
            algo = self.client.get_algo(algo_id)
            if algo.state not in ("live", "pause"):
                raise TraderError(f"OCO не активен ({algo.state})")
        except (DemoError, TraderError) as exc:
            self._emergency_sell(position, size)
            raise TraderError(f"Защита SL/TP не установлена ({exc}); позиция закрыта рыночной продажей") from None
        self.store.update(position.id, state="OPEN", algo_id=algo_id)
        self.store.event(self.clock(), "OPEN", f"{size} BTC @ {order.avg_price} SL {stop} TP {take} algo {algo_id}")
        self._say(f"ПОКУПКА: {size} BTC по {order.avg_price} USDT; SL {stop}, TP {take}")
        self._notify(f"ПОКУПКА {size} BTC по {order.avg_price} USDT. SL {stop}, TP {take}.")
        return True

    def _emergency_sell(self, position: Position, size: Decimal) -> None:
        sell_id = "DT" + uuid4().hex[:30]
        self.store.update(position.id, state="CLOSING", sell_client_id=sell_id)
        try:
            self.client.place_market_order(client_order_id=sell_id, side="sell", size_btc=size)
            order = self._wait_terminal(sell_id)
        except DemoError:
            return  # stays CLOSING; HALT follows and the owner checks manually
        if order is not None and order.terminal:
            refreshed = self.store.active_position()
            if refreshed is not None:
                self._finish(refreshed, order, "EMERGENCY")

    # --- exit ---------------------------------------------------------------------
    def _close_by_signal(self, position: Position) -> None:
        assert position.algo_id is not None and position.net_btc is not None
        try:
            self.client.cancel_algo(position.algo_id)
        except DemoApiCodeError:
            pass  # may have just triggered; the state check below decides
        algo = self.client.get_algo(position.algo_id)
        if algo.state == "effective":
            self._say("OCO сработал одновременно с сигналом SELL; сверка на следующем цикле")
            return
        if algo.state != "canceled":
            raise TraderError(f"Не удалось снять OCO перед продажей ({algo.state})")
        btc, _ = self.client.get_balances()
        if btc.available < position.net_btc:
            raise TraderError("Доступного BTC меньше, чем куплено ботом")
        sell_id = "DT" + uuid4().hex[:30]
        self.store.update(position.id, state="CLOSING", sell_client_id=sell_id)
        self.store.event(self.clock(), "SELL_SENT", f"{sell_id} {position.net_btc}")
        self.client.place_market_order(client_order_id=sell_id, side="sell", size_btc=position.net_btc)
        order = self._wait_terminal(sell_id)
        if order is None or not order.terminal:
            raise TraderError("Рыночная продажа не подтвердилась вовремя")
        refreshed = self.store.active_position()
        assert refreshed is not None
        self._finish(refreshed, order, self.CLOSE_REASON)

    def _finish(self, position: Position, order: OrderStatus, reason: str) -> None:
        if order.side != "sell" or position.net_btc is None or position.entry_cost_usdt is None:
            raise TraderError("Закрывающий ордер противоречит позиции")
        if order.filled_btc != position.net_btc:
            raise TraderError(f"Продано {order.filled_btc} BTC вместо {position.net_btc}")
        fee_usdt = -order.fee if order.fee_currency == "USDT" else -order.fee * order.avg_price
        proceeds = order.filled_btc * order.avg_price - fee_usdt
        pnl = (proceeds - position.entry_cost_usdt).quantize(Decimal("0.00000001"))
        now = self.clock()
        self.store.update(position.id, state="CLOSED", closed_at=now.isoformat(), pnl_usdt=pnl, exit_reason=reason)
        self.store.event(now, "CLOSE", f"{reason} {order.filled_btc} @ {order.avg_price} pnl {pnl}")
        label = {"TAKE_PROFIT": "тейк-профит", "STOP_LOSS": "стоп-лосс", "SIGNAL": "сигнал SELL",
                 "EMERGENCY": "аварийное закрытие", "TIME": "истекло время удержания",
                 "MANUAL": "закрыто по команде владельца"}.get(reason, reason)
        self._say(f"ПРОДАЖА ({label}): {order.filled_btc} BTC по {order.avg_price}; PnL {pnl:.2f} USDT")
        self._notify(f"ПРОДАЖА ({label}) {order.filled_btc} BTC по {order.avg_price} USDT. "
                     f"PnL {pnl:+.2f} USDT. Капитал стратегии {self.equity():.2f} USDT.")


ABANDON_PHRASE = "ABANDON OKX DEMO POSITION"


CLOSE_PHRASE = "CLOSE OKX DEMO POSITION"


def resume(store: TraderStore, phrase: str, now: datetime) -> bool:
    """Clear a HALT after the owner checked the Demo account manually.

    An active position stays tracked; the next cycle reconciles it with OKX and
    halts again if the contradiction persists.
    """
    if phrase != RESUME_PHRASE or store.get_meta("halted") is None:
        return False
    store.set_meta("halted", None)
    store.event(now, "RESUME", "manual")
    return True


def abandon_position(store: TraderStore, phrase: str, now: datetime) -> bool:
    """Stop tracking the active position after the owner closed it on OKX by hand."""
    position = store.active_position()
    if phrase != ABANDON_PHRASE or position is None:
        return False
    store.update(position.id, state="VOID", closed_at=now.isoformat(), exit_reason="ABANDONED_BY_OWNER")
    store.event(now, "ABANDON", str(position.id))
    return True
