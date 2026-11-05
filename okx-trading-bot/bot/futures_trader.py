"""Aggressive discretionary Demo trader on the BTC-USDT perpetual swap.

An experiment, agreed with the owner on 2026-11-05: what happens when a model is
allowed to trade "like a person" — long or short, up to 3x leverage, its own
entries, stops and targets — instead of a fixed rule. It is judged by
pre-registered criteria on 2027-02-16 (docs/CRITERIA_2026-12-16.md, section E).

How it works:
* once an hour (a few minutes after the hourly candle closes) and on every strong,
  confirmed bullish or bearish radar headline, DeepSeek in reasoning mode sees
  the market context, its open position, its book and its recent decisions, and
  answers OPEN_LONG / OPEN_SHORT / CLOSE / HOLD with size, leverage, stop and target;
* hard rules the model cannot override: leverage cap, a stop on every trade, a
  maximum loss at the stop per trade, a daily entry limit and a book stop — when
  the book (realized + unrealized) falls to half of its start, the position is
  closed and the bot halts until the owner reviews it;
* execution: isolated-margin market order, then an exchange-side OCO (SL/TP) on
  the filled contracts. If the OCO cannot be placed, the position is closed at
  market and the bot halts. PnL comes from OKX positions history, so fees,
  funding and liquidation are all included.

Demo only: the client inherits DemoClient, which refuses any non-DEMO config.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterator
from urllib.request import Request, urlopen
from uuid import uuid4

from bot.demo_client import DemoApiCodeError, DemoError
from bot.demo_trader import DemoTradeClient, OrderUncertain, TraderError, ORDER_NOT_FOUND_CODES
from bot.news_radar import DEEPSEEK_URL, NewsError
from config import ConfigError, _read_env_file


INST = "BTC-USDT-SWAP"
DEFAULT_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "futures_trader.sqlite3"
RESUME_PHRASE = "RESUME OKX DEMO FUTURES TRADER"
CLOSE_PHRASE = "CLOSE OKX DEMO FUTURES POSITION"
OPEN_LONG, OPEN_SHORT, CLOSE, HOLD = "OPEN_LONG", "OPEN_SHORT", "CLOSE", "HOLD"
ACTIONS = (OPEN_LONG, OPEN_SHORT, CLOSE, HOLD)
TAKER_FEE = Decimal("0.0005")          # OKX regular tier, swap taker, per side
RECONCILE_EVERY = timedelta(seconds=10)
SETTLE_TIMEOUT = timedelta(minutes=10)
PENDING_GRACE = timedelta(minutes=2)
_CLIENT_ID = re.compile(r"FT[a-f0-9]{30}\Z")
_EXCHANGE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
ZERO = Decimal(0)


def _dec(value: Any, label: str, *, signed: bool = False) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise TraderError(f"OKX вернул некорректное поле {label}") from exc
    if not number.is_finite() or (not signed and number < 0):
        raise TraderError(f"OKX вернул некорректное поле {label}")
    return number


def _floor(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _tick(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    return (value / step).to_integral_value(rounding=rounding) * step


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


# --- market data ------------------------------------------------------------------

@dataclass(frozen=True)
class SwapInstrument:
    ct_val: Decimal        # BTC per contract
    lot: Decimal           # contract step
    min_size: Decimal      # contracts
    tick: Decimal
    max_lever: Decimal
    state: str


def read_swap_instrument(public: Any) -> SwapInstrument:
    rows = public._get("/api/v5/public/instruments", {"instType": "SWAP", "instId": INST})
    if len(rows) != 1 or rows[0].get("instId") != INST or rows[0].get("ctValCcy") != "BTC":
        raise DemoError("Параметры BTC-USDT-SWAP недоступны")
    row = rows[0]
    try:
        return SwapInstrument(Decimal(row["ctVal"]), Decimal(row["lotSz"]), Decimal(row["minSz"]),
                              Decimal(row["tickSz"]), Decimal(row["lever"]), str(row.get("state")))
    except (KeyError, InvalidOperation) as exc:
        raise DemoError("Параметры BTC-USDT-SWAP некорректны") from exc


def read_swap_price(public: Any) -> Decimal:
    rows = public._get("/api/v5/market/ticker", {"instId": INST})
    try:
        price = Decimal(rows[0]["last"])
    except (IndexError, KeyError, InvalidOperation) as exc:
        raise DemoError("Цена BTC-USDT-SWAP недоступна") from exc
    if not price.is_finite() or price <= 0:
        raise DemoError("Цена BTC-USDT-SWAP некорректна")
    return price


# --- exchange client --------------------------------------------------------------

@dataclass(frozen=True)
class SwapOrder:
    order_id: str
    client_id: str
    side: str
    pos_side: str
    state: str
    filled: Decimal        # contracts
    avg_price: Decimal

    @property
    def terminal(self) -> bool:
        return self.state in ("filled", "canceled", "mmp_canceled")


@dataclass(frozen=True)
class SwapPosition:
    pos_side: str
    contracts: Decimal
    avg_price: Decimal
    upl: Decimal
    liq_price: Decimal | None


@dataclass(frozen=True)
class SwapAlgo:
    algo_id: str
    state: str
    actual_side: str


class FuturesClient(DemoTradeClient):
    """Demo client for one instrument, BTC-USDT-SWAP, isolated margin, long/short mode."""

    _POSITIONS = "/api/v5/account/positions"
    _POS_HISTORY = "/api/v5/account/positions-history"
    _SET_LEVER = "/api/v5/account/set-leverage"
    _ALLOWED_PATHS = DemoTradeClient._ALLOWED_PATHS | frozenset({_POSITIONS, _POS_HISTORY})
    _POST_PATHS = DemoTradeClient._POST_PATHS | frozenset({_SET_LEVER})

    def set_leverage(self, lever: int, pos_side: str) -> None:
        if not 1 <= lever <= 10 or pos_side not in ("long", "short"):
            raise DemoError("Некорректное плечо")
        try:
            self._post(self._SET_LEVER, {"instId": INST, "lever": str(lever),
                                         "mgnMode": "isolated", "posSide": pos_side})
        except DemoApiCodeError as exc:
            if exc.code != "0":    # this endpoint answers without a per-row sCode
                raise

    def place_market(self, *, client_id: str, side: str, pos_side: str, contracts: Decimal) -> None:
        if (not _CLIENT_ID.fullmatch(client_id) or side not in ("buy", "sell")
                or pos_side not in ("long", "short") or contracts <= 0):
            raise DemoError("Некорректные параметры ордера")
        row = self._post(self._PLACE_ORDER, {
            "instId": INST, "tdMode": "isolated", "clOrdId": client_id, "side": side,
            "posSide": pos_side, "ordType": "market", "sz": format(contracts, "f")})
        if not isinstance(row.get("ordId"), str) or not _EXCHANGE_ID.fullmatch(row["ordId"]):
            raise OrderUncertain("OKX не вернул ordId")

    def get_swap_order(self, client_id: str) -> SwapOrder | None:
        try:
            rows = self._get(self._ORDER_INFO, {"instId": INST, "clOrdId": client_id})
        except DemoApiCodeError as exc:
            if exc.code in ORDER_NOT_FOUND_CODES:
                return None
            raise
        if len(rows) != 1:
            raise TraderError("Ордер по clOrdId не найден однозначно")
        row = rows[0]
        if (row.get("instId") != INST or row.get("clOrdId") != client_id
                or row.get("side") not in ("buy", "sell") or row.get("posSide") not in ("long", "short")
                or not isinstance(row.get("state"), str)):
            raise TraderError("Ордер OKX противоречит ожидаемому BTC-USDT-SWAP")
        filled = _dec(row.get("accFillSz") or "0", "accFillSz")
        avg = _dec(row.get("avgPx") or "0", "avgPx")
        if filled > 0 and avg <= 0:
            raise TraderError("Цена исполнения неизвестна")
        return SwapOrder(str(row.get("ordId")), client_id, row["side"], row["posSide"], row["state"], filled, avg)

    def place_oco(self, *, algo_client_id: str, side: str, pos_side: str, contracts: Decimal,
                  take_profit: Decimal, stop_loss: Decimal) -> str:
        row = self._post(self._PLACE_ALGO, {
            "instId": INST, "tdMode": "isolated", "side": side, "posSide": pos_side, "ordType": "oco",
            "sz": format(contracts, "f"), "algoClOrdId": algo_client_id,
            "tpTriggerPx": format(take_profit, "f"), "tpOrdPx": "-1",
            "slTriggerPx": format(stop_loss, "f"), "slOrdPx": "-1"})
        algo_id = row.get("algoId")
        if not isinstance(algo_id, str) or not _EXCHANGE_ID.fullmatch(algo_id):
            raise OrderUncertain("OKX не вернул algoId")
        return algo_id

    def get_swap_algo(self, algo_id: str) -> SwapAlgo:
        rows = self._get(self._ALGO_INFO, {"algoId": algo_id})
        if len(rows) != 1 or rows[0].get("algoId") != algo_id or rows[0].get("instId") != INST:
            raise TraderError("OCO фьючерса не найден однозначно")
        return SwapAlgo(algo_id, str(rows[0].get("state")), str(rows[0].get("actualSide") or ""))

    def cancel_swap_algo(self, algo_id: str) -> None:
        self._post(self._CANCEL_ALGOS, [{"algoId": algo_id, "instId": INST}])

    def get_position(self, pos_side: str) -> SwapPosition | None:
        rows = self._get(self._POSITIONS, {"instType": "SWAP", "instId": INST})
        found = None
        for row in rows:
            if row.get("instId") != INST or row.get("posSide") != pos_side or row.get("mgnMode") != "isolated":
                continue
            contracts = _dec(row.get("pos") or "0", "pos", signed=True)
            if contracts == 0:
                continue
            if found is not None or contracts < 0:
                raise TraderError("Позиция фьючерса противоречива")
            liq = row.get("liqPx")
            found = SwapPosition(pos_side, contracts, _dec(row.get("avgPx") or "0", "avgPx"),
                                 _dec(row.get("upl") or "0", "upl", signed=True),
                                 _dec(liq, "liqPx") if liq else None)
        return found

    def closed_position(self, pos_side: str, since: datetime) -> dict[str, Any] | None:
        """The first full close of this side after `since` (fees and funding are inside realizedPnl)."""
        rows = self._get(self._POS_HISTORY, {"instType": "SWAP", "instId": INST, "limit": "20"})
        matches = [r for r in rows if r.get("posSide") == pos_side and r.get("mgnMode") == "isolated"
                   and str(r.get("type")) in ("2", "3", "5") and int(r.get("uTime") or 0) >= _ms(since)]
        return min(matches, key=lambda r: int(r["uTime"])) if matches else None


# --- journal ----------------------------------------------------------------------

@dataclass(frozen=True)
class FuturesPosition:
    id: int
    state: str              # OPENING, OPEN, CLOSING, SETTLING, CLOSED, VOID
    side: str               # long / short
    open_client_id: str
    close_client_id: str | None
    contracts: Decimal
    lever: int
    entry_price: Decimal | None
    stop_price: Decimal | None
    take_price: Decimal | None
    algo_id: str | None
    opened_at: str
    closed_at: str | None
    pnl_usdt: Decimal | None
    exit_reason: str | None
    decision_id: int | None


class FuturesStore:
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
                db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                db.execute("""CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, state TEXT NOT NULL, side TEXT NOT NULL,
                    open_client_id TEXT NOT NULL UNIQUE, close_client_id TEXT, contracts TEXT NOT NULL,
                    lever INTEGER NOT NULL, entry_price TEXT, stop_price TEXT, take_price TEXT, algo_id TEXT,
                    opened_at TEXT NOT NULL, closed_at TEXT, pnl_usdt TEXT, exit_reason TEXT,
                    decision_id INTEGER)""")
                db.execute("""CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, trigger TEXT NOT NULL,
                    action TEXT NOT NULL, confidence INTEGER NOT NULL, price TEXT NOT NULL,
                    detail TEXT NOT NULL)""")
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

    # meta (also the NewsFeed cursor interface)
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

    def count_events(self, kind: str) -> int:
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM events WHERE kind=?", (kind,)).fetchone()[0]

    @staticmethod
    def _position(row: sqlite3.Row | None) -> FuturesPosition | None:
        if row is None:
            return None
        def d(key: str) -> Decimal | None:
            return None if row[key] is None else Decimal(row[key])
        return FuturesPosition(row["id"], row["state"], row["side"], row["open_client_id"],
                               row["close_client_id"], Decimal(row["contracts"]), row["lever"],
                               d("entry_price"), d("stop_price"), d("take_price"), row["algo_id"],
                               row["opened_at"], row["closed_at"], d("pnl_usdt"), row["exit_reason"],
                               row["decision_id"])

    def active_position(self) -> FuturesPosition | None:
        with self._db() as db:
            rows = db.execute("SELECT * FROM positions WHERE state IN ('OPENING','OPEN','CLOSING','SETTLING') "
                              "ORDER BY id").fetchall()
        if len(rows) > 1:
            raise TraderError("В журнале больше одной активной позиции фьючерса")
        return self._position(rows[0] if rows else None)

    def closed_positions(self) -> list[FuturesPosition]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM positions WHERE state='CLOSED' ORDER BY id").fetchall()
        return [p for p in map(self._position, rows) if p is not None]

    def entries_on(self, day: str) -> int:
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM positions WHERE state!='VOID' AND substr(opened_at,1,10)=?",
                              (day,)).fetchone()[0]

    def create_opening(self, *, side: str, client_id: str, contracts: Decimal, lever: int,
                       now: datetime, decision_id: int | None) -> FuturesPosition:
        with self._db() as db:
            db.execute("INSERT INTO positions(state, side, open_client_id, contracts, lever, opened_at, decision_id) "
                       "VALUES('OPENING', ?, ?, ?, ?, ?, ?)",
                       (side, client_id, format(contracts, "f"), lever, now.isoformat(), decision_id))
            row = db.execute("SELECT * FROM positions WHERE open_client_id=?", (client_id,)).fetchone()
        position = self._position(row)
        assert position is not None
        return position

    def update(self, position_id: int, **fields: Any) -> None:
        allowed = {"state", "close_client_id", "contracts", "entry_price", "stop_price", "take_price",
                   "algo_id", "closed_at", "pnl_usdt", "exit_reason"}
        if not fields or set(fields) - allowed:
            raise ValueError("Unsupported futures position update")
        values = [format(v, "f") if isinstance(v, Decimal) else v for v in fields.values()]
        with self._db() as db:
            db.execute(f"UPDATE positions SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                       (*values, position_id))

    def add_decision(self, *, now: datetime, trigger: str, action: str, confidence: int,
                     price: Decimal, detail: dict[str, Any]) -> int:
        with self._db() as db:
            cursor = db.execute("INSERT INTO decisions(at, trigger, action, confidence, price, detail) "
                                "VALUES(?, ?, ?, ?, ?, ?)", (now.isoformat(), trigger, action, confidence,
                                                             format(price, "f"),
                                                             json.dumps(detail, ensure_ascii=False)))
            return int(cursor.lastrowid)

    def recent_decisions(self, limit: int = 8) -> list[sqlite3.Row]:
        with self._db() as db:
            return db.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()[::-1]

    def decision_counts(self) -> dict[str, int]:
        with self._db() as db:
            rows = db.execute("SELECT action, COUNT(*) AS n FROM decisions GROUP BY action").fetchall()
        return {r["action"]: r["n"] for r in rows}


def futures_state_path_from_env(path: Path | None = None) -> Path:
    values = _read_env_file(path or Path(__file__).resolve().parents[1] / ".env")
    raw = values.get("OKX_FUTURES_STATE_PATH")
    return Path(raw) if raw else DEFAULT_STATE_PATH


# --- settings -----------------------------------------------------------------------

@dataclass(frozen=True)
class FuturesSettings:
    budget_usdt: Decimal = Decimal("1000")
    max_lever: int = 3
    book_stop_pct: Decimal = Decimal("0.5")            # halt when the book falls by this share
    max_loss_at_stop_pct: Decimal = Decimal("0.25")    # of the book, per trade, incl. fees
    max_entries_per_day: int = 6
    min_confidence: int = 50
    decision_minute: int = 2                           # hourly review at HH:02 UTC
    min_impact: int = 2
    stop_range: tuple[Decimal, Decimal] = (Decimal("0.3"), Decimal("10"))    # % from entry
    take_range: tuple[Decimal, Decimal] = (Decimal("0.3"), Decimal("30"))
    order_poll_attempts: int = 10
    order_poll_delay: float = 0.5

    @classmethod
    def from_env_file(cls, path: Path | None = None) -> FuturesSettings:
        values = _read_env_file(path or Path(__file__).resolve().parents[1] / ".env")
        if values.get("OKX_FUTURES_ENABLED") != "1":
            raise ConfigError("Фьючерс-трейдер выключен: установите OKX_FUTURES_ENABLED=1 в .env")
        try:
            budget = Decimal(values.get("OKX_FUTURES_BUDGET_USDT", "1000"))
            lever = int(values.get("OKX_FUTURES_MAX_LEVER", "3"))
            stop = Decimal(values.get("OKX_FUTURES_BOOK_STOP_PCT", "0.5"))
        except (InvalidOperation, ValueError) as exc:
            raise ConfigError("Некорректные настройки OKX_FUTURES_*") from exc
        if not (100 <= budget <= 100_000 and 1 <= lever <= 5 and Decimal("0.1") <= stop <= Decimal("0.9")):
            raise ConfigError("OKX_FUTURES_*: бюджет 100..100000, плечо 1..5, стоп книги 0.1..0.9")
        return cls(budget_usdt=budget, max_lever=lever, book_stop_pct=stop)


# --- decision -----------------------------------------------------------------------

TRADER_PROMPT = """You are an experienced, aggressive discretionary crypto trader running a small book on the
BTC-USDT perpetual swap (OKX). You may go long or short, with 1x to {max_lever}x leverage, isolated margin.
You trade like a skilled human: you read the market, form a view, take risk when you see an edge,
cut losers, let winners run, and stay flat when there is nothing to do. You also remember your
recent decisions and trades, and learn from them.

Costs: taker fee {fee}% of notional per side; funding is paid every 8h (shorts receive it when it is positive).
Every trade must have a stop-loss and a take-profit; they are placed on the exchange.
Hard rules the system enforces whatever you answer: loss at the stop is capped at {max_loss}% of the
book (the size is cut to fit); at most {max_entries} entries per UTC day; if the book falls to
{book_floor} USDT the position is closed and you are stopped for good.

Decide now. Answer ONLY with JSON:
{{"action": "OPEN_LONG" | "OPEN_SHORT" | "CLOSE" | "HOLD",
 "confidence": 0-100,
 "leverage": 1-{max_lever} (for OPEN_*),
 "margin_pct": 5-100 (for OPEN_*: share of the book used as margin; notional = margin x leverage),
 "stop_loss_pct": 0.3-10 (for OPEN_*: distance from entry, %),
 "take_profit_pct": 0.3-30 (for OPEN_*: distance from entry, %),
 "reasons_ru": ["2-4 short reasons in Russian"],
 "lesson_ru": "optional: one sentence in Russian on what your recent results teach you"}}
OPEN_* only when flat. CLOSE / HOLD when a position is open (HOLD keeps it with its stop and target).
To reverse a position, CLOSE now; you will be asked again at the next review."""


@dataclass(frozen=True)
class FuturesDecision:
    action: str
    confidence: int
    leverage: int = 1
    margin_pct: Decimal = Decimal("0")
    stop_loss_pct: Decimal = Decimal("0")
    take_profit_pct: Decimal = Decimal("0")
    reasons_ru: tuple[str, ...] = ()
    lesson_ru: str = ""
    notes: tuple[str, ...] = ()       # hard-rule adjustments

    def detail(self) -> dict[str, Any]:
        return {"leverage": self.leverage, "margin_pct": str(self.margin_pct),
                "stop_loss_pct": str(self.stop_loss_pct), "take_profit_pct": str(self.take_profit_pct),
                "reasons": list(self.reasons_ru), "lesson": self.lesson_ru, "notes": list(self.notes)}


def parse_decision(text: str) -> FuturesDecision:
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or data.get("action") not in ACTIONS:
        raise NewsError("Трейдер вернул некорректное решение")
    try:
        confidence = max(0, min(100, int(data.get("confidence", 0))))
        leverage = int(data.get("leverage") or 1)
        margin = Decimal(str(data.get("margin_pct") or 0))
        stop = Decimal(str(data.get("stop_loss_pct") or 0))
        take = Decimal(str(data.get("take_profit_pct") or 0))
    except (TypeError, ValueError, InvalidOperation):
        raise NewsError("Трейдер вернул некорректные числа") from None
    if not all(x.is_finite() for x in (margin, stop, take)):
        raise NewsError("Трейдер вернул некорректные числа")
    reasons = tuple(str(r)[:200] for r in (data.get("reasons_ru") or ())[:4])
    return FuturesDecision(data["action"], confidence, leverage, margin, stop, take, reasons,
                           str(data.get("lesson_ru") or "")[:300])


def apply_hard_rules(decision: FuturesDecision, *, settings: FuturesSettings, has_position: bool,
                     entries_today: int) -> FuturesDecision:
    """Clamp what the model may ask for; turn impossible actions into HOLD."""
    notes: list[str] = []
    action = decision.action
    if action in (OPEN_LONG, OPEN_SHORT) and has_position:
        action, _ = HOLD, notes.append("позиция уже открыта — вход невозможен")
    if action == CLOSE and not has_position:
        action, _ = HOLD, notes.append("позиции нет — закрывать нечего")
    if action in (OPEN_LONG, OPEN_SHORT) and entries_today >= settings.max_entries_per_day:
        action, _ = HOLD, notes.append(f"лимит {settings.max_entries_per_day} входов за сутки")
    if action in (OPEN_LONG, OPEN_SHORT) and decision.confidence < settings.min_confidence:
        action, _ = HOLD, notes.append(f"уверенность {decision.confidence}% ниже {settings.min_confidence}%")
    lever = max(1, min(settings.max_lever, decision.leverage))
    margin = max(Decimal("5"), min(Decimal("100"), decision.margin_pct))
    stop = max(settings.stop_range[0], min(settings.stop_range[1], decision.stop_loss_pct))
    take = max(settings.take_range[0], min(settings.take_range[1], decision.take_profit_pct))
    if action in (OPEN_LONG, OPEN_SHORT):
        if lever != decision.leverage:
            notes.append(f"плечо {decision.leverage}x → {lever}x")
        if stop != decision.stop_loss_pct or take != decision.take_profit_pct:
            notes.append(f"стоп/цель приведены к {stop}% / {take}%")
    return FuturesDecision(action, decision.confidence, lever, margin, stop, take,
                           decision.reasons_ru, decision.lesson_ru, decision.notes + tuple(notes))


class FuturesAnalyst:
    def __init__(self, api_key: str, transport: Callable[[Request, float], Any] | None = None,
                 model: str = "deepseek-reasoner") -> None:
        self._key = api_key
        self._model = model
        self._transport = transport or (lambda req, t: json.load(urlopen(req, timeout=t)))

    def decide(self, *, settings: FuturesSettings, situation: str) -> FuturesDecision:
        system = TRADER_PROMPT.format(max_lever=settings.max_lever, fee=TAKER_FEE * 100,
                                      max_loss=settings.max_loss_at_stop_pct * 100,
                                      max_entries=settings.max_entries_per_day,
                                      book_floor=settings.budget_usdt * (1 - settings.book_stop_pct))
        body = json.dumps({"model": self._model, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": situation}]}).encode("utf-8")
        request = Request(DEEPSEEK_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        try:
            reply = self._transport(request, 180)
            text = reply["choices"][0]["message"]["content"]
        except Exception as exc:
            raise NewsError(f"DeepSeek не ответил: {type(exc).__name__}") from None
        return parse_decision(text)


# --- trader -------------------------------------------------------------------------

@dataclass(frozen=True)
class Trigger:
    kind: str               # HOURLY, NEWS
    text: str               # what the model is told about the trigger
    label: str              # short Russian label for the journal and Telegram


@dataclass(frozen=True)
class FuturesReport:
    status: str             # HALTED, HOLD, OPENED, CLOSED, ERROR
    lines: tuple[str, ...]


class FuturesTrader:
    def __init__(self, *, client: FuturesClient, store: FuturesStore, settings: FuturesSettings,
                 analyst: FuturesAnalyst, instrument: Callable[[], SwapInstrument],
                 price: Callable[[], Decimal], market_text: Callable[[datetime], str],
                 feed: Any | None = None, notifier: Callable[[str], object] | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 sleeper: Callable[[float], None] | None = None) -> None:
        self.client = client
        self.store = store
        self.settings = settings
        self.analyst = analyst
        self.instrument = instrument
        self.price = price
        self.market_text = market_text
        self.feed = feed
        self.notifier = notifier
        self.clock = clock
        self.sleep = sleeper or time.sleep
        self._lines: list[str] = []
        self._next_reconcile = datetime.min.replace(tzinfo=timezone.utc)

    # --- helpers ----------------------------------------------------------------
    def _say(self, line: str) -> None:
        self._lines.append(line)

    def _notify(self, text: str) -> None:
        if self.notifier is not None:
            try:
                self.notifier("OKX Demo фьючерс-трейдер: " + text)
            except Exception:
                pass

    def _report(self, status: str) -> FuturesReport:
        return FuturesReport(status, tuple(self._lines))

    def _halt(self, reason: str) -> FuturesReport:
        now = self.clock()
        self.store.set_meta("halted", f"{now.isoformat()} {reason}")
        self.store.event(now, "HALT", reason)
        self._notify(f"ОСТАНОВКА. {reason}. Новые сделки запрещены до ручной проверки и команды "
                     f"--futures-resume.")
        self._say(f"HALTED: {reason}")
        return self._report("HALTED")

    def realized(self) -> Decimal:
        return sum(((p.pnl_usdt or ZERO) for p in self.store.closed_positions()), ZERO)

    def equity(self) -> Decimal:
        return self.settings.budget_usdt + self.realized()

    def book_floor(self) -> Decimal:
        return self.settings.budget_usdt * (1 - self.settings.book_stop_pct)

    def _wait(self, client_id: str) -> SwapOrder | None:
        order = None
        for _ in range(self.settings.order_poll_attempts):
            order = self.client.get_swap_order(client_id)
            if order is not None and order.terminal:
                return order
            self.sleep(self.settings.order_poll_delay)
        return order

    @staticmethod
    def _close_side(side: str) -> str:
        return "sell" if side == "long" else "buy"

    # --- loop -------------------------------------------------------------------
    def step(self) -> FuturesReport | None:
        """One loop tick; a report only when something happened."""
        self._lines = []
        if self.store.get_meta("halted"):
            return self._settle_after_halt()
        try:
            now = self.clock()
            if self.store.get_meta("start_price") is None:
                self.store.set_meta("start_price", format(self.price(), "f"))
                self.store.set_meta("started_at", now.isoformat())
            acted = False
            if now >= self._next_reconcile:
                self._next_reconcile = now + RECONCILE_EVERY
                acted = self._reconcile(now)
                if self.store.get_meta("halted"):
                    return self._report("HALTED")
            trigger = self._news_trigger(now) or self._hourly_trigger(now)
            if trigger is not None:
                return self._decide(trigger, now)
            return self._report("CLOSED") if acted else None
        except TraderError as exc:
            return self._halt(str(exc))
        except OrderUncertain as exc:
            return self._halt(f"Неопределённый результат торгового запроса: {exc}")
        except (DemoError, NewsError, ConfigError, ValueError, OSError) as exc:
            self._say(f"Тик пропущен: {exc}")
            return self._report("ERROR")

    def _settle_after_halt(self) -> FuturesReport | None:
        """A stopped bot still books a position the exchange has already closed."""
        try:
            position = self.store.active_position()
            now = self.clock()
            if position is None or position.state != "SETTLING" or now < self._next_reconcile:
                return None
            self._next_reconcile = now + RECONCILE_EVERY
            return self._report("CLOSED") if self._settle(position, now) else None
        except (TraderError, DemoError, OSError):
            return None

    def _hourly_trigger(self, now: datetime) -> Trigger | None:
        key = now.strftime("%Y-%m-%dT%H")
        if now.minute < self.settings.decision_minute or self.store.get_meta("last_hour") == key:
            return None
        self.store.set_meta("last_hour", key)
        return Trigger("HOURLY", "Scheduled hourly review (the 1H candle just closed).", "ежечасный обзор")

    def _news_trigger(self, now: datetime) -> Trigger | None:
        if self.feed is None:
            return None
        signal = self.feed.next_signal(now, self.settings.min_impact)
        if signal is None:
            return None
        published = signal.published_at or signal.seen_at
        text = (f"Breaking headline ({signal.source}, classified {signal.direction}, impact {signal.impact}/3, "
                f"published {(now - published).total_seconds() / 60:.0f} min ago): {signal.text[:800]}\n"
                f"Summary: {signal.summary}")
        return Trigger("NEWS", text, f"новость: {signal.summary[:120]}")

    # --- decision ---------------------------------------------------------------
    def _situation(self, trigger: Trigger, now: datetime, price: Decimal,
                   position: FuturesPosition | None, live: SwapPosition | None) -> str:
        equity = self.equity()
        upl = live.upl if live is not None else ZERO
        lines = [f"Trigger: {trigger.text}", "",
                 f"Book: start {self.settings.budget_usdt} USDT, realized now {equity:.2f} USDT, "
                 f"unrealized {upl:+.2f}; stop-out level {self.book_floor():.0f} USDT.",
                 f"Entries today: {self.store.entries_on(now.date().isoformat())} of "
                 f"{self.settings.max_entries_per_day}.", f"BTC-USDT-SWAP price: {price}"]
        if position is not None and position.entry_price is not None:
            held = (now - datetime.fromisoformat(position.opened_at)).total_seconds() / 3600
            lines.append(f"Open position: {position.side.upper()} {position.contracts} contracts "
                         f"({position.contracts * self.instrument().ct_val} BTC) at {position.lever}x from "
                         f"{position.entry_price}, stop {position.stop_price}, target {position.take_price}, "
                         f"liquidation {live.liq_price if live else 'n/a'}, held {held:.1f} h, "
                         f"unrealized {upl:+.2f} USDT.")
        else:
            lines.append("Open position: none (flat).")
        closed = self.store.closed_positions()[-5:]
        if closed:
            lines.append("Your last trades:")
            lines += [f"- {p.opened_at[:16]} {p.side} {p.lever}x from {p.entry_price}: {p.exit_reason}, "
                      f"PnL {p.pnl_usdt:+.2f} USDT" for p in closed]
        decisions = self.store.recent_decisions(8)
        if decisions:
            lines.append("Your recent decisions:")
            lines += [f"- {d['at'][5:16]} {d['trigger']}: {d['action']} ({d['confidence']}%) at {d['price']}"
                      for d in decisions]
        lines += ["", "Market context:", self.market_text(now)]
        return "\n".join(lines)

    def _decide(self, trigger: Trigger, now: datetime) -> FuturesReport:
        position = self.store.active_position()
        if position is not None and position.state != "OPEN":
            self._say(f"Решение отложено: позиция в состоянии {position.state}")
            return self._report("HOLD")
        price = self.price()
        live = self.client.get_position(position.side) if position is not None else None
        raw = self.analyst.decide(settings=self.settings,
                                  situation=self._situation(trigger, now, price, position, live))
        decision = apply_hard_rules(raw, settings=self.settings, has_position=position is not None,
                                    entries_today=self.store.entries_on(now.date().isoformat()))
        decision_id = self.store.add_decision(now=now, trigger=trigger.kind, action=decision.action,
                                              confidence=decision.confidence, price=price,
                                              detail={**decision.detail(), "asked": raw.action,
                                                      "trigger": trigger.label})
        self._say(f"{trigger.label}: {decision.action} ({decision.confidence}%)"
                  + (f" — {'; '.join(decision.reasons_ru)}" if decision.reasons_ru else ""))
        header = self._describe(trigger, raw, decision, price)
        if decision.action in (OPEN_LONG, OPEN_SHORT):
            status = self._open(decision, decision_id, price, header)
            return self._report(status)
        if decision.action == CLOSE and position is not None:
            self._notify(header + "\nЗакрываю позицию.")
            self._close(position, "SIGNAL")
            return self._report("CLOSED")
        if trigger.kind == "NEWS" or raw.action != decision.action:
            self._notify(header)
        return self._report("HOLD")

    @staticmethod
    def _describe(trigger: Trigger, raw: FuturesDecision, decision: FuturesDecision, price: Decimal) -> str:
        label = {OPEN_LONG: "ЛОНГ", OPEN_SHORT: "ШОРТ", CLOSE: "ЗАКРЫТЬ", HOLD: "ЖДАТЬ"}
        lines = [f"🧠 {trigger.label} → {label[decision.action]} (уверенность {decision.confidence}%), "
                 f"цена {price:.1f}"]
        if raw.action != decision.action:
            lines.append(f"Модель хотела {label[raw.action]}, правило: {'; '.join(decision.notes)}")
        lines += [f"• {r}" for r in decision.reasons_ru]
        if decision.lesson_ru:
            lines.append(f"Урок: {decision.lesson_ru}")
        return "\n".join(lines)

    # --- entry ------------------------------------------------------------------
    def _open(self, decision: FuturesDecision, decision_id: int, price: Decimal, header: str) -> str:
        now = self.clock()
        side = "long" if decision.action == OPEN_LONG else "short"
        instrument = self.instrument()
        if instrument.state not in ("live", "None"):
            self._say(f"Вход пропущен: инструмент в состоянии {instrument.state}")
            return "HOLD"
        lever = min(decision.leverage, int(instrument.max_lever))
        equity = self.equity()
        notional = equity * decision.margin_pct / 100 * lever
        loss_rate = decision.stop_loss_pct / 100 + 2 * TAKER_FEE
        cap = equity * self.settings.max_loss_at_stop_pct
        notes = []
        if notional * loss_rate > cap:
            notional = cap / loss_rate
            notes.append(f"размер урезан: убыток на стопе ≤ {cap:.0f} USDT")
        _, usdt = self.client.get_balances()
        if notional / lever > usdt.available * Decimal("0.9"):
            notional = usdt.available * Decimal("0.9") * lever
            notes.append("размер урезан: свободный USDT на счёте")
        contracts = _floor(notional / (price * instrument.ct_val), instrument.lot)
        if contracts < instrument.min_size:
            self._say(f"Вход пропущен: {contracts} контрактов меньше минимума {instrument.min_size}")
            return "HOLD"
        self.client.set_leverage(lever, side)
        client_id = "FT" + uuid4().hex[:30]
        position = self.store.create_opening(side=side, client_id=client_id, contracts=contracts, lever=lever,
                                             now=now, decision_id=decision_id)
        self.store.event(now, "OPEN_SENT", f"{client_id} {side} {contracts} {lever}x")
        try:
            self.client.place_market(client_id=client_id, side="buy" if side == "long" else "sell",
                                     pos_side=side, contracts=contracts)
        except DemoApiCodeError as exc:
            self.store.update(position.id, state="VOID", exit_reason=f"REJECTED {exc.code}")
            self._say(f"Вход отклонён OKX (код {exc.code})")
            self._notify(header + f"\nOKX отклонил ордер (код {exc.code}).")
            return "HOLD"
        order = self._wait(client_id)
        if order is None or not order.terminal:
            raise TraderError("Рыночный вход не подтвердился вовремя")
        opened = self._protect(position, order, decision.stop_loss_pct, decision.take_profit_pct)
        if opened:
            fresh = self.store.active_position()
            assert fresh is not None
            self._notify(header + ("\n" + "; ".join(notes) if notes else "") +
                         f"\nОткрыт {side.upper()} {fresh.contracts} контр. "
                         f"({fresh.contracts * instrument.ct_val} BTC) {lever}x по {fresh.entry_price}; "
                         f"стоп {fresh.stop_price}, цель {fresh.take_price}.")
        return "OPENED" if opened else "HOLD"

    def _protect(self, position: FuturesPosition, order: SwapOrder,
                 stop_pct: Decimal, take_pct: Decimal) -> bool:
        if order.pos_side != position.side or order.side != ("buy" if position.side == "long" else "sell"):
            raise TraderError("Ордер входа противоречит позиции")
        if order.filled == 0:
            self.store.update(position.id, state="VOID", exit_reason="NOT_FILLED")
            self._say("Вход не исполнился")
            return False
        tick = self.instrument().tick
        sign = 1 if position.side == "long" else -1
        stop = _tick(order.avg_price * (1 - sign * stop_pct / 100), tick, ROUND_DOWN if sign > 0 else ROUND_UP)
        take = _tick(order.avg_price * (1 + sign * take_pct / 100), tick, ROUND_UP if sign > 0 else ROUND_DOWN)
        self.store.update(position.id, contracts=order.filled, entry_price=order.avg_price,
                          stop_price=stop, take_price=take)
        try:
            algo_id = self.client.place_oco(algo_client_id="FA" + position.open_client_id[2:],
                                            side=self._close_side(position.side), pos_side=position.side,
                                            contracts=order.filled, take_profit=take, stop_loss=stop)
            algo = self.client.get_swap_algo(algo_id)
            if algo.state not in ("live", "pause"):
                raise TraderError(f"OCO не активен ({algo.state})")
        except (DemoError, TraderError) as exc:
            self._emergency_close(position.id)
            raise TraderError(f"Защита SL/TP не установлена ({exc}); позиция закрыта по рынку") from None
        self.store.update(position.id, state="OPEN", algo_id=algo_id)
        self.store.event(self.clock(), "OPEN", f"{position.side} {order.filled} @ {order.avg_price} "
                                               f"SL {stop} TP {take} algo {algo_id}")
        self._say(f"ОТКРЫТ {position.side.upper()} {order.filled} контр. по {order.avg_price}; SL {stop}, TP {take}")
        return True

    def _emergency_close(self, position_id: int) -> None:
        position = self.store.active_position()
        if position is None or position.id != position_id:
            return
        close_id = "FT" + uuid4().hex[:30]
        self.store.update(position.id, state="CLOSING", close_client_id=close_id, exit_reason="EMERGENCY")
        try:
            self.client.place_market(client_id=close_id, side=self._close_side(position.side),
                                     pos_side=position.side, contracts=position.contracts)
            self._wait(close_id)
        except DemoError:
            pass  # stays CLOSING; the HALT that follows makes the owner check

    # --- exit ---------------------------------------------------------------------
    def _close(self, position: FuturesPosition, reason: str) -> None:
        assert position.algo_id is not None
        try:
            self.client.cancel_swap_algo(position.algo_id)
        except DemoApiCodeError:
            pass  # may have just triggered; the state below decides
        algo = self.client.get_swap_algo(position.algo_id)
        if algo.state == "effective":
            self._say("OCO сработал одновременно с решением закрыть; сверка на следующем тике")
            self.store.update(position.id, state="SETTLING", closed_at=self.clock().isoformat())
            return
        if algo.state != "canceled":
            raise TraderError(f"Не удалось снять OCO перед закрытием ({algo.state})")
        close_id = "FT" + uuid4().hex[:30]
        self.store.update(position.id, state="CLOSING", close_client_id=close_id, exit_reason=reason)
        self.store.event(self.clock(), "CLOSE_SENT", f"{close_id} {reason}")
        self.client.place_market(client_id=close_id, side=self._close_side(position.side),
                                 pos_side=position.side, contracts=position.contracts)
        order = self._wait(close_id)
        if order is None or not order.terminal or order.filled != position.contracts:
            raise TraderError("Рыночное закрытие не подтвердилось полностью")
        self.store.update(position.id, state="SETTLING", closed_at=self.clock().isoformat())
        fresh = self.store.active_position()
        if fresh is not None:
            self._settle(fresh, self.clock())

    def close_now(self) -> FuturesReport:
        """Owner-requested exit at market."""
        self._lines = []
        try:
            now = self.clock()
            self._reconcile(now)
            position = self.store.active_position()
            if position is None:
                self._say("Открытой позиции нет")
                return self._report("HOLD")
            if position.state != "OPEN":
                raise TraderError(f"Позиция в состоянии {position.state}; закрыть вручную нельзя")
            self._close(position, "MANUAL")
            return self._report("CLOSED")
        except TraderError as exc:
            return self._halt(str(exc))
        except OrderUncertain as exc:
            return self._halt(f"Неопределённый результат торгового запроса: {exc}")
        except DemoError as exc:
            self._say(f"Закрытие не выполнено: {exc}")
            return self._report("ERROR")

    # --- reconciliation -------------------------------------------------------------
    def _reconcile(self, now: datetime) -> bool:
        """Bring the journal in line with OKX; True when a position was closed."""
        position = self.store.active_position()
        if position is None:
            return False
        age = now - datetime.fromisoformat(position.opened_at)
        if position.state == "OPENING":
            order = self.client.get_swap_order(position.open_client_id)
            if order is None or not order.terminal:
                if age < PENDING_GRACE:
                    return False
                if order is None:
                    self.store.update(position.id, state="VOID", exit_reason="NOT_PLACED")
                    self._say("Незавершённый вход не дошёл до OKX; запись аннулирована")
                    return False
                raise TraderError("Рыночный вход не завершился")
            if order.filled == 0:
                self.store.update(position.id, state="VOID", exit_reason="NOT_FILLED")
                return False
            raise TraderError("Вход исполнен, но защита не подтверждена после перезапуска")
        if position.state == "CLOSING":
            assert position.close_client_id is not None
            order = self.client.get_swap_order(position.close_client_id)
            if order is None or not order.terminal:
                raise TraderError("Состояние закрытия позиции неизвестно")
            self.store.update(position.id, state="SETTLING", closed_at=now.isoformat())
            position = self.store.active_position()
            assert position is not None
        if position.state == "SETTLING":
            return self._settle(position, now)
        # OPEN
        live = self.client.get_position(position.side)
        if live is None:
            self.store.update(position.id, state="SETTLING", closed_at=now.isoformat())
            fresh = self.store.active_position()
            assert fresh is not None
            return self._settle(fresh, now)
        if live.contracts != position.contracts:
            raise TraderError(f"На бирже {live.contracts} контрактов вместо {position.contracts}")
        assert position.algo_id is not None
        algo = self.client.get_swap_algo(position.algo_id)
        if algo.state not in ("live", "pause", "effective"):
            raise TraderError(f"Защитный OCO в состоянии {algo.state}; позиция может быть без защиты")
        if algo.state in ("live", "pause") and self.equity() + live.upl <= self.book_floor():
            self._close(position, "BOOK_STOP")
            self.store.set_meta("halted", f"{now.isoformat()} BOOK_STOP")
            self.store.event(now, "BOOK_STOP", f"equity {self.equity():.2f}")
            self._notify(f"СТОП КНИГИ: капитал упал до {self.equity():.2f} USDT (порог {self.book_floor():.0f}). "
                         f"Позиция закрыта, эксперимент остановлен.")
            self._say("СТОП КНИГИ: позиция закрыта, торговля остановлена")
            return True
        return False

    def _settle(self, position: FuturesPosition, now: datetime) -> bool:
        row = self.client.closed_position(position.side, datetime.fromisoformat(position.opened_at))
        if row is None:
            started = datetime.fromisoformat(position.closed_at or now.isoformat())
            if now - started > SETTLE_TIMEOUT:
                raise TraderError("Закрытая позиция не появилась в истории OKX")
            return False
        pnl = _dec(row.get("realizedPnl") or row.get("pnl") or "0", "realizedPnl", signed=True)
        close_px = _dec(row.get("closeAvgPx") or "0", "closeAvgPx")
        reason = position.exit_reason
        if str(row.get("type")) in ("3", "5"):
            reason = "LIQUIDATION" if str(row.get("type")) == "3" else "ADL"
        elif reason is None and position.algo_id:
            algo = self.client.get_swap_algo(position.algo_id)
            reason = {"tp": "TAKE_PROFIT", "sl": "STOP_LOSS"}.get(algo.actual_side, "EXCHANGE")
            if algo.state in ("live", "pause"):
                self.client.cancel_swap_algo(position.algo_id)   # closed by hand on OKX: drop the orphan
                reason = "EXTERNAL"
        reason = reason or "EXCHANGE"
        self.store.update(position.id, state="CLOSED", closed_at=now.isoformat(),
                          pnl_usdt=pnl.quantize(Decimal("0.00000001")), exit_reason=reason)
        self.store.event(now, "CLOSE", f"{reason} {position.side} @ {close_px} pnl {pnl}")
        label = {"TAKE_PROFIT": "цель", "STOP_LOSS": "стоп-лосс", "SIGNAL": "решение трейдера",
                 "BOOK_STOP": "стоп книги", "MANUAL": "команда владельца", "LIQUIDATION": "ЛИКВИДАЦИЯ",
                 "EMERGENCY": "аварийное закрытие", "EXTERNAL": "закрыта вне бота"}.get(reason, reason)
        self._say(f"ЗАКРЫТ {position.side.upper()} ({label}) по {close_px}; PnL {pnl:+.2f} USDT")
        self._notify(f"Закрыт {position.side.upper()} ({label}) по {close_px}. PnL {pnl:+.2f} USDT "
                     f"(с комиссиями и фандингом). Книга {self.equity():.2f} USDT.")
        return True


def resume(store: FuturesStore, phrase: str, now: datetime) -> bool:
    halted = store.get_meta("halted")
    if phrase != RESUME_PHRASE or halted is None or "BOOK_STOP" in halted:
        return False     # the book stop ends the experiment; it is not resumable
    store.set_meta("halted", None)
    store.event(now, "RESUME", "manual")
    return True


def run_forever(trader: FuturesTrader, *, on_report: Callable[[FuturesReport], None],
                on_heartbeat: Callable[[bool], None], tick: float = 2.0,
                sleeper: Callable[[float], None] = time.sleep) -> None:
    last_beat = 0.0
    errors = 0
    while True:
        report = trader.step()
        if report is not None:
            on_report(report)
            errors = errors + 1 if report.status == "ERROR" else 0
        if time.monotonic() - last_beat >= 60:
            on_heartbeat(errors < 3)
            last_beat = time.monotonic()
        sleeper(tick)


def summary_lines(store: FuturesStore, settings: FuturesSettings, price: Decimal | None) -> list[str]:
    """Book, trades and the buy-and-hold comparison for status and reports."""
    closed = store.closed_positions()
    pnl = sum(((p.pnl_usdt or ZERO) for p in closed), ZERO)
    equity = settings.budget_usdt + pnl
    halted = store.get_meta("halted")
    active = store.active_position()
    wins = sum(1 for p in closed if (p.pnl_usdt or 0) > 0)
    lines = [f"  статус: {'HALTED — ' + halted if halted else 'работает'}",
             f"  книга {equity:.2f} USDT из {settings.budget_usdt} ({(equity / settings.budget_usdt - 1) * 100:+.1f}%), "
             f"сделок {len(closed)} (прибыльных {wins})",
             "  позиция: " + ("нет" if active is None else
                              f"{active.state} {active.side.upper()} {active.contracts} контр. {active.lever}x "
                              f"от {active.entry_price}, SL {active.stop_price}, TP {active.take_price}")]
    start = store.get_meta("start_price")
    if start and price:
        hold = (price / Decimal(start) - 1) * 100
        lines.append(f"  для сравнения: BTC с начала эксперимента {hold:+.1f}% "
                     f"(с {Decimal(start):.0f} до {price:.0f})")
    reasons: dict[str, int] = {}
    for p in closed:
        reasons[p.exit_reason or "?"] = reasons.get(p.exit_reason or "?", 0) + 1
    if reasons:
        lines.append("  выходы: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())))
    counts = store.decision_counts()
    if counts:
        lines.append("  решения: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return lines
