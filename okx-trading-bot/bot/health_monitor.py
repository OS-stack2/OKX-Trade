"""One-shot freshness monitor for the separate, GET-only Demo Watchdog journal.

This module never contacts OKX or opens the strategy/trading SQLite. HEALTHY
means a recent successful Watchdog observation, not a safe trading position.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from typing import Callable

from bot.telegram_alerts import TelegramAlerter
from bot.watchdog import WatchdogError, WatchdogSnapshot, WatchdogStore, watchdog_path_from_env
from config import _read_env_file


class HealthError(ValueError):
    """Invalid health configuration or inaccessible dedicated state."""


@dataclass(frozen=True)
class HealthSettings:
    watchdog_path: Path
    health_path: Path
    stale_after: timedelta


def health_settings_from_env(path: Path | None = None) -> HealthSettings:
    """Use the existing local Watchdog path; health state is its sibling file."""
    try:
        watchdog_path = watchdog_path_from_env(path)
        values = _read_env_file(path or Path(__file__).resolve().parents[1] / ".env")
        if values.get("OKX_MODE") != "DEMO":
            raise HealthError("Health Monitor требует локальный OKX_MODE=DEMO")
        raw = values.get("OKX_HEALTH_STALE_MINUTES", "15")
        if not raw.isascii() or not raw.isdecimal() or not 1 <= int(raw) <= 1440:
            raise HealthError("OKX_HEALTH_STALE_MINUTES должен быть целым числом от 1 до 1440")
        return HealthSettings(watchdog_path, watchdog_path.with_name("health.sqlite3"),
                              timedelta(minutes=int(raw)))
    except HealthError:
        raise
    except (WatchdogError, OSError, UnicodeError, ValueError) as exc:
        raise HealthError("Локальная конфигурация Health Monitor недоступна") from exc


def _stamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise HealthError("Требуется UTC-время")
    return value.isoformat().replace("+00:00", "Z")


def _parse_stamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise HealthError("Время наблюдения Watchdog неизвестно")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        _stamp(result)
    except (ValueError, OverflowError) as exc:
        raise HealthError("Время наблюдения Watchdog некорректно") from exc
    return result


@dataclass(frozen=True)
class SourceObservation:
    latest_id: int
    latest_status: str
    latest_at: datetime
    success_id: int | None
    success_at: datetime | None


@dataclass(frozen=True)
class HealthReport:
    status: str
    reason: str
    last_success_utc: datetime | None
    notification: str | None
    state_saved: bool

    @property
    def exit_code(self) -> int:
        return 0 if self.status == "HEALTHY" and self.state_saved else 1


class HealthStore:
    """Read Watchdog SQLite in read-only mode; write only a different 0600 file."""

    def __init__(self, settings: HealthSettings) -> None:
        self.settings = settings
        self._guard_paths()

    def _guard_paths(self) -> None:
        source, state = self.settings.watchdog_path, self.settings.health_path
        try:
            WatchdogStore(source)._guard_path()
            if (not state.is_absolute() or state.name != "health.sqlite3"
                    or state.parent != source.parent or state.resolve() == source.resolve()
                    or state.is_symlink() or (state.exists() and state.stat().st_nlink != 1)):
                raise HealthError("Health SQLite должна быть отдельным обычным файлом рядом с Watchdog")
            if source.is_symlink() or (source.exists() and source.stat().st_nlink != 1):
                raise HealthError("Ссылка на журнал Watchdog запрещена")
            if self.settings.stale_after <= timedelta(0) or self.settings.stale_after > timedelta(days=1):
                raise HealthError("Некорректный порог свежести")
        except (WatchdogError, OSError) as exc:
            raise HealthError("Недопустимые пути Health Monitor") from exc

    def read_source(self) -> SourceObservation:
        """Never create or mutate the Watchdog journal, even on missing/corrupt data."""
        self._guard_paths()
        source = self.settings.watchdog_path
        if not source.is_file():
            raise HealthError("Журнал Watchdog отсутствует")
        try:
            with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
                db.execute("PRAGMA query_only=ON")
                if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                    raise HealthError("Журнал Watchdog повреждён")
                latest = db.execute("SELECT id,checked_at_utc,status,snapshot_json "
                                    "FROM watchdog_observations ORDER BY id DESC LIMIT 1").fetchone()
                state = db.execute("SELECT health,updated_at_utc FROM watchdog_state WHERE id=1").fetchone()
                if latest is None or state is None:
                    raise HealthError("Наблюдения Watchdog отсутствуют")
                ident, raw_at, status, snapshot = latest
                if type(ident) is not int or ident < 1 or status not in (
                        "OBSERVED_UNVERIFIED", "SUSPICIOUS", "API_UNAVAILABLE"):
                    raise HealthError("Последнее наблюдение Watchdog противоречиво")
                at = _parse_stamp(raw_at)
                prior_row = db.execute("SELECT checked_at_utc FROM watchdog_observations "
                                       "WHERE id < ? ORDER BY id DESC LIMIT 1", (ident,)).fetchone()
                if prior_row is not None and _parse_stamp(prior_row[0]) > at:
                    raise HealthError("Часы Watchdog переведены назад")
                if _parse_stamp(state[1]) != at or state[0] != ("DOWN" if status == "API_UNAVAILABLE" else "UP"):
                    raise HealthError("Состояние Watchdog противоречит последнему наблюдению")
                if (status == "API_UNAVAILABLE") != (snapshot is None):
                    raise HealthError("Неполное наблюдение Watchdog")
                if snapshot is not None:
                    WatchdogSnapshot.from_json(snapshot)
                success = db.execute("SELECT id,checked_at_utc,snapshot_json FROM watchdog_observations "
                                     "WHERE status IN ('OBSERVED_UNVERIFIED','SUSPICIOUS') "
                                     "ORDER BY id DESC LIMIT 1").fetchone()
                if success is not None:
                    if type(success[0]) is not int or success[0] > ident or success[2] is None:
                        raise HealthError("Последнее успешное наблюдение противоречиво")
                    WatchdogSnapshot.from_json(success[2])
                    if _parse_stamp(success[1]) > at:
                        raise HealthError("Время успешного наблюдения противоречиво")
                return SourceObservation(ident, status, at,
                                         success[0] if success else None,
                                         _parse_stamp(success[1]) if success else None)
        except (sqlite3.Error, OSError, WatchdogError) as exc:
            raise HealthError("Журнал Watchdog недоступен или повреждён") from exc

    def record(self, *, now: datetime, status: str, success_id: int | None,
               latest_id: int | None) -> tuple[str, tuple[int, str] | None]:
        """Claim a Telegram attempt in SQLite before sending; never retry ambiguity."""
        self._guard_paths()
        if status not in {"HEALTHY", "STALE", "API_UNAVAILABLE", "UNKNOWN"}:
            raise HealthError("Некорректный статус Health Monitor")
        stamp = _stamp(now)
        path = self.settings.health_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._guard_paths()
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags, 0o600)
            try:
                if os.fstat(fd).st_nlink != 1:
                    raise HealthError("Жёсткая ссылка на Health SQLite запрещена")
                if os.name == "posix":
                    os.fchmod(fd, 0o600)
            finally:
                os.close(fd)
            with closing(sqlite3.connect(path, timeout=3)) as db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("PRAGMA busy_timeout=3000")
                if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                    raise HealthError("Health SQLite повреждена")
                try:
                    db.execute("BEGIN IMMEDIATE")
                    names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    required = {"health_state", "health_alerts"}
                    if names & required and names & required != required:
                        raise HealthError("Схема Health SQLite неполна")
                    db.execute("""CREATE TABLE IF NOT EXISTS health_state (
                        id INTEGER PRIMARY KEY CHECK(id=1), status TEXT NOT NULL,
                        highwater_utc TEXT NOT NULL, failure_success_id INTEGER,
                        latest_id INTEGER)""")
                    db.execute("""CREATE TABLE IF NOT EXISTS health_alerts (
                        id INTEGER PRIMARY KEY, created_at_utc TEXT NOT NULL,
                        event TEXT NOT NULL, delivered INTEGER)""")
                    prior = db.execute("SELECT status,highwater_utc,failure_success_id,latest_id "
                                       "FROM health_state WHERE id=1").fetchone()
                    if prior is None and db.execute("SELECT 1 FROM health_alerts LIMIT 1").fetchone():
                        raise HealthError("Health SQLite потеряла состояние дедупликации")
                    previous, highwater, failure_id, previous_latest = prior if prior else (None, None, None, None)
                    if (previous not in (None, "HEALTHY", "STALE", "API_UNAVAILABLE", "UNKNOWN")
                            or (failure_id is not None and (type(failure_id) is not int or failure_id < 0))
                            or (previous_latest is not None and (type(previous_latest) is not int or previous_latest < 1))):
                        raise HealthError("Состояние Health SQLite повреждено")
                    highwater_at = _parse_stamp(highwater) if highwater is not None else None
                    had_failure = failure_id is not None
                    if highwater_at is not None and now < highwater_at:
                        status = "UNKNOWN"  # Wall clock moved backwards: do not claim recovery.
                    if previous_latest is not None and latest_id is not None and latest_id < previous_latest:
                        status = "UNKNOWN"  # Watchdog history reset/replaced.
                    if status in {"STALE", "API_UNAVAILABLE"} and failure_id is None:
                        failure_id = success_id or 0
                    if status == "HEALTHY" and failure_id is not None:
                        if success_id is None or success_id <= failure_id:
                            status = "UNKNOWN"  # A new successful GET must prove recovery.
                    event = None
                    if status == "STALE" and previous != "STALE" and not (previous == "UNKNOWN" and had_failure):
                        event = "HEALTH_STALE"
                    elif status == "API_UNAVAILABLE" and previous != "API_UNAVAILABLE":
                        event = "HEALTH_API_UNAVAILABLE"
                    elif status == "HEALTHY" and failure_id is not None:
                        event = "HEALTH_RESTORED"
                        failure_id = None
                    db.execute("""INSERT INTO health_state(id,status,highwater_utc,failure_success_id,latest_id)
                        VALUES(1,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,
                        highwater_utc=excluded.highwater_utc,
                        failure_success_id=excluded.failure_success_id,latest_id=excluded.latest_id""",
                               (status, _stamp(max(now, highwater_at) if highwater_at else now),
                                failure_id, max(x for x in (latest_id, previous_latest) if x is not None)
                                if latest_id is not None or previous_latest is not None else None))
                    alert = None
                    if event:
                        cursor = db.execute("INSERT INTO health_alerts(created_at_utc,event,delivered) "
                                            "VALUES(?,?,NULL)", (stamp, event))
                        alert = (cursor.lastrowid, event)
                    db.execute("DELETE FROM health_alerts WHERE id <= "
                               "(SELECT COALESCE(MAX(id),0)-200 FROM health_alerts)")
                    db.commit()
                    return status, alert
                except BaseException:
                    db.rollback()
                    raise
        except (sqlite3.Error, OSError) as exc:
            raise HealthError("Отдельная Health SQLite недоступна") from exc

    def mark_delivery(self, alert_id: int, delivered: bool) -> None:
        self._guard_paths()
        try:
            with closing(sqlite3.connect(self.settings.health_path.as_uri() + "?mode=rw",
                                         uri=True, timeout=3)) as db:
                db.execute("UPDATE health_alerts SET delivered=? WHERE id=? AND delivered IS NULL",
                           (1 if delivered else 0, alert_id))
                db.commit()
        except (sqlite3.Error, OSError) as exc:
            raise HealthError("Результат уведомления не удалось записать") from exc


class HealthMonitor:
    """Offline observation freshness check; Telegram is optional and best effort."""

    def __init__(self, store: HealthStore, *,
                 alerter_loader: Callable[[], TelegramAlerter | None] = TelegramAlerter.from_env_file,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.store = store
        self.alerter_loader = alerter_loader
        self.clock = clock

    def run_once(self) -> HealthReport:
        try:
            now = self.clock()
            _stamp(now)
        except (HealthError, ValueError, TypeError):
            return HealthReport("UNKNOWN", "Системное время UTC неизвестно", None, None, False)
        source = None
        try:
            source = self.store.read_source()
            if source.latest_at > now + timedelta(seconds=30) or (source.success_at and source.success_at > now + timedelta(seconds=30)):
                status, reason = "UNKNOWN", "Время Watchdog находится в будущем; проверьте системные часы"
            elif source.latest_status == "API_UNAVAILABLE":
                status, reason = "API_UNAVAILABLE", "Последний цикл Watchdog сообщил об ошибке публичного или Demo GET"
            elif source.success_at is None:
                status, reason = "UNKNOWN", "Нет подтверждённого успешного GET Watchdog"
            elif now - source.success_at > self.store.settings.stale_after:
                status, reason = "STALE", "Последнее успешное наблюдение Watchdog просрочено"
            else:
                status, reason = "HEALTHY", "Успешный GET Watchdog свежий; принадлежность BTC и защита неизвестны"
        except HealthError:
            status, reason = "UNKNOWN", "Журнал Watchdog отсутствует, недоступен или повреждён"
        try:
            recorded_status, alert = self.store.record(now=now, status=status,
                                                       success_id=source.success_id if source else None,
                                                       latest_id=source.latest_id if source else None)
        except HealthError:
            return HealthReport("UNKNOWN", "Отдельная Health SQLite или конфигурация недоступна",
                                source.success_at if source else None, None, False)
        if recorded_status != status:
            status, reason = "UNKNOWN", "Системное время или история Watchdog противоречивы; новый GET не подтвердил восстановление"
        notification = None
        state_saved = True
        if alert:
            alert_id, event = alert
            try:
                alerter = self.alerter_loader()
                delivered = alerter is not None and alerter.notify(event) is True
            except Exception:
                delivered = False
            notification = f"{event}: {'принято Telegram' if delivered else 'доставка не подтверждена'}"
            try:
                self.store.mark_delivery(alert_id, delivered)
            except HealthError:
                notification += "; результат доставки не сохранён"
                state_saved = False
                status, reason = "UNKNOWN", "Не удалось сохранить результат уведомления в Health SQLite"
        return HealthReport(status, reason, source.success_at if source else None,
                            notification, state_saved)
