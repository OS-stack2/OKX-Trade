"""Explicit, one-time creation of an empty Demo journal on a separate host.

The preview is read-only. Initialization creates a private temporary journal
using DecisionJournal, then publishes it with a no-overwrite hard link.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import sqlite3
import stat
import tempfile

from bot.execution_diagnostics import inspect_execution
from bot.execution_engine import ExecutionEngine
from bot.journal import CycleDraft, CycleStatus, DecisionJournal, JournalError
from bot.risk_manager import RiskManager, RiskSettings, RiskStateError
from bot.strategy_budget import BudgetError, StrategyBudgetStore


SQLITE_CONFIRMATION_PHRASE = "INITIALIZE NEW OKX DEMO SQLITE"


class SqliteSetupError(ValueError):
    """The file could not be initialized without risk of overwriting state."""


@dataclass(frozen=True)
class SqliteSetupReport:
    status: str  # MISSING, EXISTS, BLOCKED
    reason: str
    path: Path
    directory_mode: str | None
    file_mode: str | None
    schema: str
    intent_count: int | None
    halted: bool | None
    budget: str
    risk: str
    token: str | None = None


class ServerSqliteSetup:
    """Never opens the Demo API, changes risk state, or initializes a budget."""

    def __init__(self, *, journal: DecisionJournal | None = None,
                 risk_manager: RiskManager | None = None,
                 clock=lambda: datetime.now(timezone.utc)) -> None:
        self.journal = journal or DecisionJournal()
        self.risk_manager = risk_manager or RiskManager()
        self.path = Path(self.journal.path)
        self.clock = clock

    def _risk_status(self) -> str:
        try:
            self.risk_manager.validate_settings()
            if self.risk_manager.settings != RiskSettings():
                return "INVALID_SETTINGS"
        except RiskStateError:
            return "INVALID_SETTINGS"
        state = self.risk_manager.state_path
        lock = Path(f"{state}.lock")
        if not state.exists() and not state.is_symlink() and not lock.exists() and not lock.is_symlink():
            return "UNINITIALIZED"
        try:
            snapshot = self.risk_manager.inspect_daily_state(self.clock())
            return "LOCKED" if snapshot.loss_locked or snapshot.pending_entry else "EXISTING"
        except (RiskStateError, OSError, ValueError):
            return "UNKNOWN_OR_LOCKED"

    def preview(self) -> SqliteSetupReport:
        """Inspect path, permissions, schema, intentions and separate safety locks."""
        parent = self.path.parent
        if not parent.is_dir() or parent.is_symlink():
            return SqliteSetupReport("BLOCKED", "Каталог data отсутствует или недопустим", self.path,
                                     None, None, "UNKNOWN", None, None, "UNKNOWN", "UNKNOWN")
        try:
            parent_stat = parent.stat()
            directory_mode = f"{stat.S_IMODE(parent_stat.st_mode):04o}"
            if (not os.access(parent, os.R_OK | os.X_OK)
                    or not (parent_stat.st_mode & 0o444)
                    or not (parent_stat.st_mode & 0o111)):
                raise PermissionError("Каталог недоступен для чтения")
            risk = self._risk_status()
            if self.path.is_symlink():
                return SqliteSetupReport("BLOCKED", "SQLite не может быть символической ссылкой", self.path,
                                         directory_mode, None, "UNKNOWN", None, None, "UNKNOWN", risk)
            if any(parent.glob(".sqlite-bootstrap-*.sqlite3")):
                return SqliteSetupReport("BLOCKED", "Найден незавершённый временный журнал; нужна ручная проверка", self.path,
                                         directory_mode, None, "UNKNOWN", None, None, "UNKNOWN", risk)
            if not self.path.exists():
                writable = (os.access(parent, os.W_OK | os.X_OK)
                            and bool(parent_stat.st_mode & 0o222)
                            and bool(parent_stat.st_mode & 0o111))
                if not writable:
                    return SqliteSetupReport("BLOCKED", "Нет прав создать SQLite в каталоге data", self.path,
                                             directory_mode, None, "MISSING", 0, False, "UNCONFIRMED", risk)
                if risk != "UNINITIALIZED":
                    return SqliteSetupReport("BLOCKED", "SQLite отсутствует, но состояние Risk Manager уже существует или неизвестно", self.path,
                                             directory_mode, None, "MISSING", 0, False, "UNCONFIRMED", risk)
                now = self.clock()
                if (not isinstance(now, datetime) or now.tzinfo is None
                        or now.utcoffset() != timedelta(0)):
                    raise ValueError("Требуется UTC-время")
                challenge = (f"{self.path.absolute()}|{parent_stat.st_dev}|{parent_stat.st_ino}|"
                             f"{int(now.timestamp()) // 300}")
                token = hashlib.sha256(challenge.encode("utf-8")).hexdigest()[:16]
                return SqliteSetupReport("MISSING", "Новый локальный журнал можно подготовить после отдельного подтверждения", self.path,
                                         directory_mode, None, "MISSING", 0, False, "UNCONFIRMED", risk, token)
            file_stat = self.path.lstat()
            file_mode = f"{stat.S_IMODE(file_stat.st_mode):04o}"
            if (not stat.S_ISREG(file_stat.st_mode) or not os.access(self.path, os.R_OK)
                    or not (file_stat.st_mode & 0o444)):
                return SqliteSetupReport("BLOCKED", "SQLite не является читаемым обычным файлом", self.path,
                                         directory_mode, file_mode, "UNKNOWN", None, None, "UNKNOWN", risk)
            if os.name == "posix" and stat.S_IMODE(file_stat.st_mode) & 0o077:
                return SqliteSetupReport("BLOCKED", "Права SQLite шире 0600; файл не изменялся", self.path,
                                         directory_mode, file_mode, "UNKNOWN", None, None, "UNKNOWN", risk)
            try:
                audit = self.journal.audit_read_only()
                inspection = inspect_execution(ExecutionEngine(journal=self.journal))
                budget_record = StrategyBudgetStore(self.path).read()
            except (JournalError, BudgetError, sqlite3.Error, OSError):
                return SqliteSetupReport("BLOCKED", "Схема или бюджет SQLite повреждены; файл не изменялся", self.path,
                                         directory_mode, file_mode, "UNKNOWN", None, None, "UNKNOWN", risk)
            if inspection.sqlite_status == "UNKNOWN":
                return SqliteSetupReport("BLOCKED", "Состояние исполнителя SQLite не подтверждено", self.path,
                                         directory_mode, file_mode, "UNKNOWN", None, None, "UNKNOWN", risk)
            budget = "CONFIRMED_5000_DEMO_USDT" if budget_record is not None else "UNCONFIRMED"
            if inspection.halted or inspection.intent_count:
                return SqliteSetupReport("BLOCKED", "Сохранён HALTED или прежнее намерение; инициализация запрещена", self.path,
                                         directory_mode, file_mode, f"VALID_DECISIONS_{audit.decision_count}",
                                         inspection.intent_count, inspection.halted, budget, risk)
            return SqliteSetupReport("EXISTS", "Существующая SQLite читается; повторное создание запрещено", self.path,
                                     directory_mode, file_mode, f"VALID_DECISIONS_{audit.decision_count}",
                                     0, False, budget, risk)
        except (OSError, ValueError, OverflowError):
            return SqliteSetupReport("BLOCKED", "Права, время или состояние пути неизвестны", self.path,
                                     None, None, "UNKNOWN", None, None, "UNKNOWN", "UNKNOWN")

    def initialize(self, phrase: str, token: str) -> SqliteSetupReport:
        """Publish a verified new journal atomically; existing target is never opened for write."""
        if phrase != SQLITE_CONFIRMATION_PHRASE or not isinstance(token, str) or len(token) != 16:
            raise SqliteSetupError("Нужны точная фраза и токен текущей диагностики")
        report = self.preview()
        if report.status != "MISSING" or report.token != token:
            raise SqliteSetupError("Подготовка заблокирована: путь, права или токен изменились")
        temporary: str | None = None
        try:
            fd, temporary = tempfile.mkstemp(prefix=".sqlite-bootstrap-", suffix=".sqlite3",
                                            dir=self.path.parent)
            os.close(fd)
            draft = CycleDraft(self.clock(), None, None, CycleStatus.BLOCK,
                               "Служебная подготовка нового Demo-журнала; торговля отключена")
            temporary_journal = DecisionJournal(Path(temporary))
            temporary_journal.append(draft)
            audit = temporary_journal.audit_read_only()
            if audit.decision_count != 1 or audit.executed_count != 0:
                raise SqliteSetupError("Новый журнал не прошёл проверку")
            if StrategyBudgetStore(Path(temporary)).read() is not None:
                raise SqliteSetupError("Бюджет появился до отдельного подтверждения")
            inspection = inspect_execution(ExecutionEngine(journal=temporary_journal))
            if inspection.sqlite_status != "CLEAR" or inspection.intent_count or inspection.halted:
                raise SqliteSetupError("Новый журнал содержит блокировку или намерение")
            # Hard-link creation fails with EEXIST; unlike rename/replace it cannot
            # overwrite a database created by a concurrent process.
            os.link(temporary, self.path)
            if os.name == "posix":
                directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            os.unlink(temporary)
            temporary = None
            result = self.preview()
            if result.status != "EXISTS" or result.budget != "UNCONFIRMED":
                raise SqliteSetupError("Новый журнал опубликован, но состояние неизвестно; проверьте вручную")
            return result
        except (OSError, JournalError, BudgetError, sqlite3.Error) as exc:
            raise SqliteSetupError("Новый журнал не удалось безопасно опубликовать; существующая SQLite не перезаписана") from exc
        finally:
            if temporary is not None:
                os.unlink(temporary)
