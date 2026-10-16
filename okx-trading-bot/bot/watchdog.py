"""One-shot, observation-only OKX Demo BTC-USDT SPOT watchdog.

Its SQLite belongs solely to the watchdog. It never reads the Windows decision
journal, Risk Manager state or position ownership evidence. Successful GETs
therefore never establish bot-owned BTC or confirmed stop-loss protection.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Callable, Iterator

from bot.demo_client import CurrencyBalance, DemoAlgoOrder, DemoClient
from bot.okx_client import OkxClient
from bot.telegram_alerts import TelegramAlerter
from config import Config, _read_env_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WATCHDOG_PATH = PROJECT_ROOT / "watchdog_data" / "watchdog.sqlite3"
WINDOWS_JOURNAL_PATH = PROJECT_ROOT / "data" / "dry_run_journal.sqlite3"
MAX_OBSERVATIONS = 576  # Two days at five-minute intervals.
MAX_ALERTS = 200
MAX_LIST_ITEMS = 200


class WatchdogError(ValueError):
    """Unsafe watchdog configuration or unavailable separate observation log."""


def watchdog_path_from_env(path: Path | None = None) -> Path:
    """Read only the local project .env; never accept a shell override."""
    env_path = path or PROJECT_ROOT / ".env"
    try:
        if env_path.is_symlink():
            raise WatchdogError("Ссылка вместо локального .env Watchdog запрещена")
        if os.name == "posix" and env_path.is_file() and env_path.stat().st_mode & 0o077:
            raise WatchdogError("Права локального .env Watchdog должны быть 0600")
        values = _read_env_file(env_path)
    except WatchdogError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise WatchdogError("Локальный .env для Watchdog недоступен или некорректен") from exc
    raw = values.get("OKX_WATCHDOG_STATE_PATH", "")
    if not raw:
        return DEFAULT_WATCHDOG_PATH
    selected = Path(raw)
    if not selected.is_absolute():
        raise WatchdogError("OKX_WATCHDOG_STATE_PATH должен быть абсолютным путём")
    return selected


def _amount(value: object) -> Decimal:
    try:
        number = value if isinstance(value, Decimal) else Decimal(value)
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise WatchdogError("Некорректный виртуальный баланс") from exc
    if not number.is_finite() or number < 0:
        raise WatchdogError("Некорректный виртуальный баланс")
    return number


def _utc(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise WatchdogError("Watchdog требует время UTC")
    return value.isoformat().replace("+00:00", "Z")


def _signature(parts: tuple[object, ...]) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WatchdogSnapshot:
    btc_total: Decimal
    btc_available: Decimal
    usdt_total: Decimal
    usdt_available: Decimal
    spot_order_signatures: tuple[str, ...]
    algo_signatures: tuple[str, ...]
    other_spot_orders: int = 0
    unknown_algo_sizes: bool = False

    @property
    def spot_order_count(self) -> int:
        return len(self.spot_order_signatures)

    @property
    def algo_order_count(self) -> int:
        return len(self.algo_signatures)

    def to_json(self) -> str:
        return json.dumps({
            "btc_total": str(self.btc_total), "btc_available": str(self.btc_available),
            "usdt_total": str(self.usdt_total), "usdt_available": str(self.usdt_available),
            "spot_order_signatures": self.spot_order_signatures,
            "algo_signatures": self.algo_signatures,
            "other_spot_orders": self.other_spot_orders,
            "unknown_algo_sizes": self.unknown_algo_sizes,
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_json(cls, raw: str) -> WatchdogSnapshot:
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or set(data) != {
                "btc_total", "btc_available", "usdt_total", "usdt_available",
                "spot_order_signatures", "algo_signatures", "other_spot_orders",
                "unknown_algo_sizes",
            }:
                raise WatchdogError("Снимок Watchdog повреждён")
            orders, algos = data["spot_order_signatures"], data["algo_signatures"]
            if (not isinstance(orders, list) or not isinstance(algos, list)
                    or len(orders) > MAX_LIST_ITEMS or len(algos) > MAX_LIST_ITEMS
                    or any(not isinstance(x, str) or len(x) != 64 for x in orders + algos)
                    or type(data["other_spot_orders"]) is not int
                    or not 0 <= data["other_spot_orders"] <= len(orders)
                    or type(data["unknown_algo_sizes"]) is not bool):
                raise WatchdogError("Снимок Watchdog повреждён")
            snapshot = cls(_amount(data["btc_total"]), _amount(data["btc_available"]),
                           _amount(data["usdt_total"]), _amount(data["usdt_available"]),
                           tuple(orders), tuple(algos), data["other_spot_orders"],
                           data["unknown_algo_sizes"])
            if snapshot.btc_available > snapshot.btc_total or snapshot.usdt_available > snapshot.usdt_total:
                raise WatchdogError("Снимок Watchdog противоречив")
            return snapshot
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise WatchdogError("Снимок Watchdog повреждён") from exc


@dataclass(frozen=True)
class WatchdogReport:
    status: str
    snapshot: WatchdogSnapshot | None
    changed: bool
    reasons: tuple[str, ...]
    notifications: tuple[str, ...]
    journaled: bool
    alert_failed: bool = False

    @property
    def exit_code(self) -> int:
        return 0 if (self.status == "OBSERVED_UNVERIFIED" and self.journaled
                     and not self.alert_failed) else 1


class WatchdogStore:
    """Separate bounded SQLite observation log and durable alert deduplication."""

    def __init__(self, path: Path = DEFAULT_WATCHDOG_PATH) -> None:
        self.path = Path(path)
        self._guard_path()

    def _guard_path(self) -> None:
        if not self.path.is_absolute():
            raise WatchdogError("Для отдельного журнала нужен абсолютный путь")
        resolved = self.path.resolve()
        if (resolved == WINDOWS_JOURNAL_PATH.absolute()
                or resolved.is_relative_to((PROJECT_ROOT / "data").absolute())
                or self.path.name != "watchdog.sqlite3"):
            raise WatchdogError("Журнал Watchdog не может указывать на рабочую SQLite")
        if self.path.exists() and self.path.stat().st_nlink != 1:
            raise WatchdogError("Жёсткая ссылка на журнал Watchdog запрещена")

    def _prepare(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._guard_path()
            if self.path.is_symlink():
                raise WatchdogError("Символическая ссылка на журнал Watchdog запрещена")
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
            try:
                if os.fstat(descriptor).st_nlink != 1:
                    raise WatchdogError("Жёсткая ссылка на журнал Watchdog запрещена")
                if os.name == "posix":
                    os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise WatchdogError("Не удалось безопасно открыть отдельный журнал Watchdog") from exc

    @staticmethod
    def _schema(db: sqlite3.Connection) -> None:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"watchdog_state", "watchdog_observations", "watchdog_alerts"}
        if names & required and names & required != required:
            raise WatchdogError("Схема журнала Watchdog неполна")
        db.execute("""CREATE TABLE IF NOT EXISTS watchdog_state (
            id INTEGER PRIMARY KEY CHECK(id=1), health TEXT NOT NULL,
            snapshot_json TEXT, updated_at_utc TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS watchdog_observations (
            id INTEGER PRIMARY KEY, checked_at_utc TEXT NOT NULL,
            status TEXT NOT NULL, snapshot_json TEXT, reasons_json TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS watchdog_alerts (
            id INTEGER PRIMARY KEY, created_at_utc TEXT NOT NULL, event TEXT NOT NULL,
            attempted INTEGER NOT NULL CHECK(attempted=1), delivered INTEGER)""")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._prepare()
        try:
            with closing(sqlite3.connect(self.path, timeout=3)) as db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("PRAGMA busy_timeout=3000")
                db.execute("PRAGMA journal_size_limit=1048576")
                db.execute("PRAGMA cache_size=-2048")
                try:
                    db.execute("BEGIN IMMEDIATE")
                    self._schema(db)
                    yield db
                    db.commit()
                except BaseException:
                    db.rollback()
                    raise
        except (sqlite3.Error, OSError) as exc:
            raise WatchdogError("Отдельный журнал Watchdog недоступен") from exc

    def record(self, *, at_utc: datetime, status: str,
               snapshot: WatchdogSnapshot | None, reasons: tuple[str, ...]
               ) -> tuple[str, bool, tuple[tuple[int, str], ...]]:
        """Atomically save the observation and claim one attempt per new alert."""
        if status not in {"API_UNAVAILABLE", "SUSPICIOUS", "OBSERVED_UNVERIFIED"}:
            raise WatchdogError("Недопустимый статус Watchdog")
        if (status == "API_UNAVAILABLE") != (snapshot is None):
            raise WatchdogError("Снимок Watchdog противоречит статусу")
        stamp = _utc(at_utc)
        if not isinstance(reasons, tuple) or any(not isinstance(x, str) or len(x) > 80 for x in reasons):
            raise WatchdogError("Причины Watchdog некорректны")
        with self._transaction() as db:
            row = db.execute("SELECT health,snapshot_json FROM watchdog_state WHERE id=1").fetchone()
            if row is None and db.execute("SELECT 1 FROM watchdog_observations LIMIT 1").fetchone():
                raise WatchdogError("Состояние журнала Watchdog отсутствует")
            previous_health, previous_json = (row if row is not None else (None, None))
            if previous_health not in (None, "UP", "DOWN"):
                raise WatchdogError("Состояние журнала Watchdog повреждено")
            if previous_health == "UP" and previous_json is None:
                raise WatchdogError("Последний успешный снимок Watchdog отсутствует")
            previous = WatchdogSnapshot.from_json(previous_json) if previous_json is not None else None
            changed = snapshot is not None and previous is not None and snapshot != previous
            if changed:
                status = "SUSPICIOUS"
                reasons += ("Снимок BTC/USDT или ордеров изменился после предыдущей проверки",)
            events: list[str] = []
            if snapshot is None:
                if previous_health != "DOWN":
                    events.append("WATCHDOG_API_DOWN")
                new_health, new_json = "DOWN", previous_json
            else:
                if previous_health == "DOWN":
                    events.append("WATCHDOG_RESTORED")
                if (changed or previous is None) and status == "SUSPICIOUS":
                    events.append("WATCHDOG_SUSPICIOUS")
                elif changed:
                    events.append("WATCHDOG_CHANGED")
                new_health, new_json = "UP", snapshot.to_json()
            db.execute("""INSERT INTO watchdog_state(id,health,snapshot_json,updated_at_utc)
                VALUES(1,?,?,?) ON CONFLICT(id) DO UPDATE SET health=excluded.health,
                snapshot_json=excluded.snapshot_json,updated_at_utc=excluded.updated_at_utc""",
                (new_health, new_json, stamp))
            db.execute("INSERT INTO watchdog_observations(checked_at_utc,status,snapshot_json,reasons_json) "
                       "VALUES(?,?,?,?)", (stamp, status, snapshot.to_json() if snapshot else None,
                                           json.dumps(reasons, ensure_ascii=True)))
            alerts: list[tuple[int, str]] = []
            for event in events:
                cursor = db.execute("INSERT INTO watchdog_alerts(created_at_utc,event,attempted,delivered) "
                                    "VALUES(?,?,1,NULL)", (stamp, event))
                alerts.append((cursor.lastrowid, event))
            db.execute("DELETE FROM watchdog_observations WHERE id <= "
                       "(SELECT COALESCE(MAX(id),0)-? FROM watchdog_observations)", (MAX_OBSERVATIONS,))
            db.execute("DELETE FROM watchdog_alerts WHERE id <= "
                       "(SELECT COALESCE(MAX(id),0)-? FROM watchdog_alerts)", (MAX_ALERTS,))
            return status, bool(changed), tuple(alerts)

    def mark_delivery(self, alert_id: int, delivered: bool) -> None:
        with self._transaction() as db:
            db.execute("UPDATE watchdog_alerts SET delivered=? WHERE id=? AND delivered IS NULL",
                       (1 if delivered else 0, alert_id))


class Watchdog:
    """Exactly one GET cycle; no loop, position ownership or trade capability."""

    def __init__(self, *, store: WatchdogStore,
                 demo_loader: Callable[[], DemoClient] = DemoClient.from_env_file,
                 public_factory: Callable[[Config], OkxClient] = OkxClient,
                 alerter_loader: Callable[[], TelegramAlerter | None] = TelegramAlerter.from_env_file,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.store = store
        self.demo_loader = demo_loader
        self.public_factory = public_factory
        self.alerter_loader = alerter_loader
        self.clock = clock

    def _snapshot(self) -> WatchdogSnapshot:
        demo = self.demo_loader()
        if not isinstance(demo, DemoClient) or demo.config.mode != "DEMO":
            raise WatchdogError("Только отдельный Demo-клиент допустим")
        public = self.public_factory(demo.config)
        ticker = public.get_ticker()
        if not isinstance(ticker, list) or len(ticker) != 1 or not isinstance(ticker[0], dict):
            raise WatchdogError("Публичный BTC-USDT ticker неполон")
        row = ticker[0]
        now = self.clock()
        try:
            stamped = datetime.fromtimestamp(int(row["ts"]) / 1000, timezone.utc)
            price = Decimal(row["last"])
        except (KeyError, TypeError, ValueError, InvalidOperation, OSError, OverflowError) as exc:
            raise WatchdogError("Публичный BTC-USDT ticker некорректен") from exc
        if (row.get("instId") != "BTC-USDT" or not price.is_finite() or price <= 0
                or now.tzinfo is None or now.utcoffset() != timedelta(0)
                or stamped > now + timedelta(seconds=30) or now - stamped > timedelta(minutes=5)):
            raise WatchdogError("Публичный BTC-USDT ticker устарел или противоречив")
        btc, usdt = demo.get_balances()
        for balance in (btc, usdt):
            if (not isinstance(balance, CurrencyBalance)
                    or _amount(balance.total) < _amount(balance.available)):
                raise WatchdogError("Виртуальный баланс противоречив")
        orders = demo.get_open_orders()
        algos = demo.get_pending_algo_orders()
        if (not isinstance(orders, list) or not isinstance(algos, tuple)
                or len(orders) > MAX_LIST_ITEMS or len(algos) > MAX_LIST_ITEMS):
            raise WatchdogError("Полнота открытых ордеров неизвестна")
        order_tokens: list[str] = []
        other = 0
        for order in orders:
            if (not isinstance(order, dict) or order.get("instType") != "SPOT"
                    or not isinstance(order.get("ordId"), str) or not order["ordId"]
                    or len(order["ordId"]) > 64
                    or not isinstance(order.get("instId"), str) or not order["instId"]
                    or len(order["instId"]) > 64
                    or order.get("state") not in {"live", "partially_filled"}):
                raise WatchdogError("Открытый SPOT-ордер противоречив")
            other += order["instId"] != "BTC-USDT"
            order_tokens.append(_signature((order["ordId"], order["instId"], order["state"],
                                            order.get("side"), order.get("sz"),
                                            order.get("accFillSz"))))
        algo_tokens: list[str] = []
        unknown_algo_sizes = False
        for algo in algos:
            if not isinstance(algo, DemoAlgoOrder):
                raise WatchdogError("Защитный Algo-ордер противоречив")
            unknown_algo_sizes |= algo.size_btc is None
            algo_tokens.append(_signature((algo.algo_id, algo.order_type, algo.state,
                                           str(algo.size_btc), algo.side)))
        if len(set(order_tokens)) != len(order_tokens) or len(set(algo_tokens)) != len(algo_tokens):
            raise WatchdogError("Биржа вернула дублирующийся ордер")
        return WatchdogSnapshot(btc.total, btc.available, usdt.total, usdt.available,
                                tuple(sorted(order_tokens)), tuple(sorted(algo_tokens)),
                                other, unknown_algo_sizes)

    def run_once(self) -> WatchdogReport:
        now = self.clock()
        try:
            _utc(now)
        except WatchdogError:
            return WatchdogReport("JOURNAL_ERROR", None, False,
                                  ("Время UTC недоступно",), (), False)
        try:
            snapshot = self._snapshot()
            reasons = ["Принадлежность BTC и фактическое покрытие SL неизвестны без журнала стратегии"]
            if snapshot.spot_order_count:
                reasons.append("Есть открытые SPOT-ордера неизвестного происхождения")
            if snapshot.algo_order_count:
                reasons.append("Есть Algo-ордера без доказанной связи с BTC бота")
            if snapshot.other_spot_orders:
                reasons.append("Есть SPOT-ордера по другому инструменту")
            if snapshot.unknown_algo_sizes:
                reasons.append("Размер одного или нескольких Algo-ордеров неизвестен")
            if snapshot.btc_available < snapshot.btc_total and not snapshot.spot_order_count:
                reasons.append("Часть BTC зарезервирована без понятного SPOT-ордера")
            if snapshot.usdt_available < snapshot.usdt_total and not snapshot.spot_order_count:
                reasons.append("Часть USDT зарезервирована без понятного SPOT-ордера")
            status = "SUSPICIOUS" if len(reasons) > 1 else "OBSERVED_UNVERIFIED"
        except Exception:
            # Never print transport errors: URLs and headers can contain secrets.
            snapshot, status = None, "API_UNAVAILABLE"
            reasons = ["Публичный или приватный Demo GET недоступен либо ответ противоречив"]
        try:
            status, changed, alerts = self.store.record(at_utc=now, status=status,
                                                        snapshot=snapshot, reasons=tuple(reasons))
        except (WatchdogError, OSError, sqlite3.Error):
            return WatchdogReport("JOURNAL_ERROR", None, False,
                                  ("Отдельный журнал Watchdog недоступен; состояние неизвестно",), (), False)
        if changed:
            reasons.append("Снимок BTC/USDT или ордеров изменился после предыдущей проверки")
        notifications: list[str] = []
        alert_failed = False
        try:
            alerter = self.alerter_loader()
        except Exception:
            alerter = None
        for alert_id, event in alerts:
            delivered = False
            if alerter is not None:
                try:
                    delivered = alerter.notify(event) is True
                except Exception:
                    pass
            alert_failed |= not delivered
            try:
                self.store.mark_delivery(alert_id, delivered)
            except (WatchdogError, OSError, sqlite3.Error):
                notifications.append("Результат Telegram не удалось записать")
                alert_failed = True
            notifications.append(f"{event}: {'принято Telegram' if delivered else 'доставка не подтверждена'}")
        return WatchdogReport(status, snapshot, changed, tuple(reasons),
                              tuple(notifications), True, alert_failed)
