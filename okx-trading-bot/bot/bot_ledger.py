"""Read-only view of what the Demo bots believe they hold, for Watchdog reconciliation."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sqlite3

from bot.demo_trader import trader_state_path_from_env
from bot.trend_trader import trend_state_path_from_env
from config import _read_env_file


DEFAULT_EXTERNAL_BTC = Decimal("1")
BTC_TOLERANCE = Decimal("0.00001")  # sub-lot fee dust left by market orders


@dataclass(frozen=True)
class BotHoldings:
    btc: Decimal
    algo_ids: frozenset[str]
    external_btc: Decimal


def _rows(path: Path, query: str) -> list[sqlite3.Row] | None:
    """None when the journal does not exist (bot never ran)."""
    if not path.is_file():
        return None
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)) as db:
        db.row_factory = sqlite3.Row
        try:
            return db.execute(query).fetchall()
        except sqlite3.OperationalError:
            return []  # created but schema not written yet


def read_bot_holdings(env_path: Path | None = None) -> BotHoldings | None:
    """Return the bots' expected BTC and protective Algo IDs.

    None means the change cannot be attributed to the bots: no bot journal
    exists, or a bot is mid-transaction (an order in flight).
    """
    values = _read_env_file(env_path or Path(__file__).resolve().parents[1] / ".env")
    try:
        external = Decimal(values.get("OKX_WATCHDOG_EXTERNAL_BTC", str(DEFAULT_EXTERNAL_BTC)))
    except InvalidOperation:
        return None
    trader = _rows(trader_state_path_from_env(env_path),
                   "SELECT state, net_btc, algo_id FROM positions WHERE state IN ('OPENING','OPEN','CLOSING')")
    trend = _rows(trend_state_path_from_env(env_path), "SELECT key, value FROM meta")
    if trader is None and trend is None:
        return None
    btc = Decimal(0)
    algo_ids: set[str] = set()
    for row in trader or []:
        if row["state"] != "OPEN" or row["net_btc"] is None or not row["algo_id"]:
            return None
        btc += Decimal(row["net_btc"])
        algo_ids.add(row["algo_id"])
    meta = {row["key"]: row["value"] for row in trend or []}
    if "pending" in meta:
        return None
    btc += Decimal(meta.get("btc", "0"))
    return BotHoldings(btc, frozenset(algo_ids), external)
