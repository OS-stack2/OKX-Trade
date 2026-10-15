"""Offline protection evidence; all observations keep execution halted."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest
from unittest.mock import patch

from bot.execution_engine import ExecutionEngine, ExecutionError, ExecutionState, RecoveryProof, RECOVERY_PHRASE
from bot.journal import CycleDraft, CycleStatus, DecisionJournal
from bot.protection_model import (AlgoLeg, BuyObservation, BuyState,
                                  ProtectionJournal, assess_protection)
from bot.risk_manager import RiskManager
from bot.strategy_budget import StrategyBudgetStore


NOW = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)
CLIENT_ID = "TB" + "a" * 30


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.journal = DecisionJournal(root / "journal.sqlite3")
        self.journal.append(CycleDraft(NOW, NOW - timedelta(minutes=15),
                                       "HOLD", CycleStatus.HOLD, "offline fixture"))
        self.engine = ExecutionEngine(journal=self.journal,
            risk_manager=RiskManager(root / "risk.json"),
            budget_store=StrategyBudgetStore(self.journal.path), clock=lambda: NOW)
        with self.engine._transaction() as db:
            self.engine._schema(db)
            db.execute("""INSERT INTO execution_intents (
                client_order_id,decision_id,candle_closed_at,mode,inst_id,side,state,
                quantity,entry_price,stop_price,take_profit_price,estimated_risk_usdt,
                min_size,lot_size,tick_size,created_at_utc,updated_at_utc,
                exchange_order_id,filled_quantity,fee_btc,fee_usdt,average_fill_price,reason_code
            ) VALUES (?,? ,?,'DEMO','BTC-USDT','BUY','PARTIALLY_FILLED',
                '0.03','80000','79200','81600','25','0.0001','0.0001','0.1',
                ?,?,'buy-order','0.02','0','-1','80000','FILL_OBSERVED')""",
                (CLIENT_ID, "decision-1", NOW.isoformat(), NOW.isoformat(), NOW.isoformat()))
        self.intent = self.engine.diagnose().intents[0]
        self.ledger = ProtectionJournal(self.engine)

    def legs(self, *, size=D("0.02"), group="attached-1"):
        common = dict(parent_order_id="buy-order", parent_client_order_id=CLIENT_ID,
                      group_id=group, size_btc=size, state="live")
        return (AlgoLeg("stop-1", "STOP", trigger_price=D("79200"), **common),
                AlgoLeg("take-1", "TAKE", trigger_price=D("81600"), **common))

    def observation(self, state=BuyState.PARTIAL_ACTIVE, **changes):
        stop, take = self.legs()
        base = BuyObservation(CLIENT_ID, "buy-order", state, D("0.02"), NOW,
                              stop, take, True)
        return replace(base, **changes)

    def test_partial_active_even_with_exact_protection_stays_halted(self):
        result = self.ledger.record(self.observation())
        self.assertEqual(result.confirmed_covered_btc, D("0.02"))
        self.assertEqual(result.uncovered_btc, D(0))
        self.assertTrue(any("остаток BUY" in reason for reason in result.reasons))
        self.assertTrue(self.engine.diagnose().halted)
        self.assertEqual(self.engine.diagnose().intents[0].state, ExecutionState.HALTED)

    def test_partial_canceled_and_full_fill_are_distinct_and_never_ready(self):
        canceled = assess_protection(self.intent, self.observation(BuyState.PARTIAL_CANCELED),
                                     now_utc=NOW)
        self.assertTrue(any("Отмена" in reason for reason in canceled.reasons))
        full_intent = replace(self.intent, filled_quantity=D("0.03"), fee_btc=D(0),
                              state=ExecutionState.FILLED)
        stop, take = self.legs(size=D("0.03"))
        full = assess_protection(full_intent,
            self.observation(BuyState.FILLED, filled_btc=D("0.03"), stop=stop, take=take),
            now_utc=NOW)
        self.assertEqual(full.confirmed_covered_btc, D("0.03"))
        self.assertTrue(full.halted)
        self.assertTrue(any("Атомарность" in reason for reason in full.reasons))

    def test_timeout_cannot_reuse_protection_as_permission(self):
        result = self.ledger.record(self.observation(BuyState.TIMEOUT))
        self.assertTrue(any("Таймаут" in reason for reason in result.reasons))
        self.assertTrue(self.engine.diagnose().halted)

    def test_missing_or_short_stop_halts_and_records_uncovered_owned_btc(self):
        missing = self.ledger.record(self.observation(stop=None))
        self.assertEqual(missing.uncovered_btc, D("0.02"))
        stop, take = self.legs(size=D("0.01"))
        short = self.ledger.record(self.observation(stop=stop, take=take))
        self.assertEqual(short.uncovered_btc, D("0.02"))
        self.assertTrue(any("фактический размер" in reason for reason in short.reasons))
        self.assertTrue(self.engine.diagnose().halted)

    def test_delayed_algo_ids_never_retroactively_protect_earlier_fill(self):
        first = self.ledger.record(self.observation(stop=None, take=None))
        self.assertEqual(first.uncovered_btc, D("0.02"))
        later = self.ledger.record(self.observation())
        self.assertEqual(later.confirmed_covered_btc, D("0.02"))
        self.assertTrue(any("Атомарность" in reason for reason in later.reasons))
        self.assertTrue(self.engine.diagnose().halted)
        with closing(sqlite3.connect(self.journal.path)) as db:
            with closing(db.execute("SELECT uncovered_btc FROM protection_observations ORDER BY rowid")) as cursor:
                self.assertEqual([D(row[0]) for row in cursor.fetchall()], [D("0.02"), D(0)])

    def test_algo_ids_parent_and_demo_identity_must_match(self):
        stop, take = self.legs()
        for bad_stop in (replace(stop, algo_id=take.algo_id),
                         replace(stop, parent_order_id="foreign-order"),
                         replace(stop, mode="LIVE"),
                         replace(stop, side="BUY"),
                         replace(stop, state="canceled"),
                         replace(stop, trigger_price=D("79000"))):
            with self.subTest(bad_stop=bad_stop):
                result = assess_protection(self.intent, self.observation(stop=bad_stop),
                                           now_utc=NOW)
                self.assertEqual(result.uncovered_btc, D("0.02"))

    def test_independent_sell_legs_could_double_sell_and_are_not_coverage(self):
        stop, take = self.legs()
        for observation in (self.observation(exclusive_group_confirmed=False),
                            self.observation(take=replace(take, group_id="other"))):
            result = assess_protection(self.intent, observation, now_utc=NOW)
            self.assertEqual(result.confirmed_covered_btc, D(0))
            self.assertTrue(any("двойная продажа" in x for x in result.reasons))

    def test_external_one_btc_is_not_owned_or_eligible_for_protection(self):
        # Demo account may hold 1 external BTC, but only journal-confirmed net
        # BUY fills are available to this model.
        result = assess_protection(self.intent, self.observation(), now_utc=NOW)
        self.assertEqual(result.owned_btc, D("0.02"))
        stop, take = self.legs(size=D("1.02"))
        over = assess_protection(self.intent, self.observation(stop=stop, take=take), now_utc=NOW)
        self.assertEqual(over.confirmed_covered_btc, D(0))
        self.assertEqual(over.uncovered_btc, D("0.02"))

    def test_stale_and_contradictory_buy_status_fail_closed(self):
        stale = assess_protection(self.intent,
            self.observation(checked_at_utc=NOW - timedelta(minutes=2)), now_utc=NOW)
        self.assertTrue(any("устарело" in reason for reason in stale.reasons))
        self.assertEqual(stale.confirmed_covered_btc, D(0))
        wrong = assess_protection(self.intent, self.observation(BuyState.FILLED), now_utc=NOW)
        self.assertTrue(any("Полное исполнение" in reason for reason in wrong.reasons))
        with self.assertRaises(ExecutionError):
            assess_protection(self.intent, self.observation(state="UNKNOWN"), now_utc=NOW)

    def test_btc_fee_changes_owned_size_and_untradable_dust_blocks(self):
        charged = replace(self.intent, fee_btc=D("-0.00005"))
        stop, take = self.legs(size=D("0.01995"))
        result = assess_protection(charged, self.observation(stop=stop, take=take), now_utc=NOW)
        self.assertEqual(result.owned_btc, D("0.01995"))
        self.assertEqual(result.confirmed_covered_btc, D(0))
        self.assertTrue(any("minSz/lotSz" in x for x in result.reasons))

    def test_step_aligned_btc_fee_uses_net_owned_btc(self):
        charged = replace(self.intent, fee_btc=D("-0.0001"))
        stop, take = self.legs(size=D("0.0199"))
        result = assess_protection(charged, self.observation(stop=stop, take=take), now_utc=NOW)
        self.assertEqual(result.owned_btc, D("0.0199"))
        self.assertEqual(result.confirmed_covered_btc, D("0.0199"))
        self.assertTrue(result.halted)

    def test_append_survives_restart_and_never_changes_risk_or_budget(self):
        before = self.journal.audit_read_only()
        result = self.ledger.record(self.observation(stop=None))
        restarted = ExecutionEngine(journal=self.journal,
            risk_manager=self.engine.risk_manager, budget_store=self.engine.budget_store,
            clock=lambda: NOW)
        self.assertTrue(restarted.diagnose().halted)
        self.assertEqual(ProtectionJournal(restarted).latest(CLIENT_ID), result)
        self.assertEqual(self.journal.audit_read_only(), before)
        self.assertFalse(self.engine.risk_manager.state_path.exists())
        self.assertIsNone(self.engine.budget_store.read())
        with self.assertRaises(ExecutionError):
            restarted.manual_recover(phrase=RECOVERY_PHRASE,
                proof=RecoveryProof(CLIENT_ID, NOW, True, True, True, "OKX_DEMO_READ_ONLY"))

    def test_repeated_observation_is_append_only_and_no_secret_columns(self):
        self.ledger.record(self.observation(stop=None))
        self.ledger.record(self.observation(stop=None))
        with closing(sqlite3.connect(self.journal.path)) as db:
            with closing(db.execute("SELECT COUNT(*) FROM protection_observations")) as cursor:
                self.assertEqual(cursor.fetchone()[0], 2)
            with closing(db.execute("PRAGMA table_info(protection_observations)")) as cursor:
                names = {row[1] for row in cursor.fetchall()}
        self.assertFalse(names & {"api_key", "secret", "passphrase", "http_headers"})

    def test_failed_observation_write_leaves_durable_halt(self):
        with patch("bot.protection_model.uuid4", side_effect=OSError("disk unavailable")):
            with self.assertRaises(ExecutionError):
                self.ledger.record(self.observation(stop=None))
        restarted = ExecutionEngine(journal=self.journal,
            risk_manager=self.engine.risk_manager, budget_store=self.engine.budget_store,
            clock=lambda: NOW)
        self.assertTrue(restarted.diagnose().halted)


if __name__ == "__main__":
    unittest.main()
