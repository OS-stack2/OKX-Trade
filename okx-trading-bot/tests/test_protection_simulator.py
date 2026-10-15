"""Offline lifecycle, loss, restart and Windows-friendly SQLite closure tests."""

from __future__ import annotations

from contextlib import closing, redirect_stdout
from decimal import Decimal as D
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest
from unittest.mock import patch

import main
from bot.journal import DEFAULT_JOURNAL_PATH
from bot.protection_simulator import (SimEvent, SimulationConfig, SimulationError,
                                      SimulationLedger, parse_price_path,
                                      run_sample_scenarios)


class ProtectionSimulatorTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "simulation.sqlite3"
        self.config = SimulationConfig()
        self.ledger = SimulationLedger(self.path, self.config)

    def add(self, event_id, second, kind, **fields):
        return self.ledger.append(SimEvent(event_id, second, kind, **fields))

    def buy(self, *, quantity=D("0.02"), fee_btc=D("-0.00002")):
        self.add("buy-ack", 0, "BUY_ACK")
        return self.add("buy-fill", 1, "BUY_FILL", quantity_btc=quantity,
                        price_usdt=D("80000"), fee_btc=fee_btc, order_id="fill-1")

    def protect(self, *, second=2, size=D("0.01998"), stop=D("79200"),
                take=D("81600"), stop_id="stop-1", take_id="take-1",
                group="group-1", exclusive=True, stop_size=None, take_size=None):
        return self.add(f"protect-{second}", second, "PROTECTION_ACK",
                        quantity_btc=size, stop_size_btc=stop_size or size,
                        take_size_btc=take_size or size,
                        stop_price_usdt=stop, take_price_usdt=take,
                        stop_id=stop_id, take_id=take_id, group_id=group,
                        exclusive_oco=exclusive)

    def test_full_buy_net_btc_one_oco_reservation_and_sticky_halt(self):
        bought = self.buy()
        self.assertEqual(bought.bot_owned_btc, D("0.01998"))
        self.assertEqual(bought.uncovered_btc, D("0.01998"))
        result = self.protect(second=1)
        self.assertEqual(result.stop_usdt, D("79200"))
        self.assertEqual(result.take_usdt, D("81600"))
        self.assertEqual(result.simulated_reserved_btc, D("0.01998"))
        self.assertEqual(result.uncovered_btc, D(0))
        self.assertEqual(result.unprotected_seconds, 0)
        self.assertEqual(result.external_btc, D("1"))
        self.assertEqual(result.decision, "BLOCK")
        self.assertTrue(result.halted)  # delayed evidence cannot clear the gap

    def test_partial_active_then_cancel_residual_and_delayed_protection(self):
        reports = dict(run_sample_scenarios((D("80000"), D("78400"))))
        active = next(value for name, value in reports.items() if name.startswith("Частичный BUY: SL"))
        canceled = next(value for name, value in reports.items() if name.startswith("Частичный BUY, отмена"))
        self.assertEqual((active.buy_state, active.uncovered_btc, active.unprotected_seconds),
                         ("FILLED", D("0.00999"), 79))
        self.assertEqual((canceled.buy_state, canceled.uncovered_btc, canceled.unprotected_seconds),
                         ("PARTIAL_CANCELED", D(0), 39))
        self.assertTrue(active.halted and canceled.halted)

    def test_rejected_buy_is_zero_position_while_timeout_loss_is_unknown(self):
        reports = dict(run_sample_scenarios((D("78400"),)))
        rejected = next(value for name, value in reports.items() if name.startswith("Отказ BUY"))
        timeout = next(value for name, value in reports.items() if name.startswith("Таймаут BUY"))
        self.assertEqual((rejected.bot_owned_btc, rejected.price_losses_usdt[0][1]), (D(0), D(0)))
        self.assertEqual(timeout.buy_state, "UNKNOWN")
        self.assertIsNone(timeout.unprotected_seconds)
        self.assertIsNone(timeout.price_losses_usdt[0][1])
        self.assertTrue(timeout.halted)

    def test_rejection_after_fill_and_fill_after_timeout_remain_uncertain(self):
        self.buy()
        rejected = self.add("late-reject", 2, "BUY_REJECT")
        self.assertEqual(rejected.bot_owned_btc, D("0.01998"))
        self.assertTrue(rejected.halted)
        self.assertIsNone(self.add("price", 60, "PRICE", price_usdt=D("78400"))
                          .price_losses_usdt[0][1])
        with TemporaryDirectory() as directory:
            ledger = SimulationLedger(Path(directory) / "simulation.sqlite3")
            ledger.append(SimEvent("ack", 0, "BUY_ACK"))
            ledger.append(SimEvent("timeout", 1, "BUY_TIMEOUT"))
            observed = ledger.append(SimEvent("late-fill", 2, "BUY_FILL",
                quantity_btc=D("0.01"), price_usdt=D("80000"),
                fee_btc=D("-0.00001"), order_id="fill-1"))
            self.assertEqual(observed.bot_owned_btc, D("0.00999"))
            self.assertIsNone(observed.unprotected_seconds)
            self.assertTrue(observed.halted)

    def test_rejected_protection_keeps_all_owned_btc_uncovered(self):
        self.buy()
        rejected = self.add("sl-reject", 15, "PROTECTION_REJECT")
        self.assertEqual(rejected.uncovered_btc, D("0.01998"))
        self.assertEqual(rejected.simulated_reserved_btc, D(0))
        self.assertTrue(rejected.halted)

    def test_price_path_loss_includes_btc_fee_exit_fee_and_slippage(self):
        self.buy()
        result = self.add("price", 60, "PRICE", price_usdt=D("78400"))
        # 0.02*80000 - 0.01998*78400*(1 - 0.001 - 0.001).
        self.assertEqual(result.price_losses_usdt[0], (D("78400"), D("36.700864")))
        self.assertEqual(result.unprotected_seconds, 59)
        self.assertEqual(result.btc_seconds_unprotected, D("0.01998") * 59)

    def test_wrong_sl_tp_price_size_or_oco_never_counts_as_coverage(self):
        for change in ({"stop": D("79300")}, {"take": D("81500")},
                       {"size": D("0.03")}, {"stop_size": D("0.01")},
                       {"take_size": D("0.01")}, {"exclusive": False},
                       {"stop_id": "same", "take_id": "same"}):
            with self.subTest(change=change):
                with TemporaryDirectory() as directory:
                    ledger = SimulationLedger(Path(directory) / "simulation.sqlite3")
                    ledger.append(SimEvent("ack", 0, "BUY_ACK"))
                    ledger.append(SimEvent("fill", 1, "BUY_FILL", quantity_btc=D("0.02"),
                                           price_usdt=D("80000"), fee_btc=D("-0.00002"), order_id="fill-1"))
                    params = {"quantity_btc": D("0.01998"), "stop_size_btc": D("0.01998"),
                              "take_size_btc": D("0.01998"),
                              "stop_price_usdt": D("79200"), "take_price_usdt": D("81600"), "stop_id": "stop-1",
                              "take_id": "take-1", "group_id": "group-1", "exclusive_oco": True}
                    mapping = {"stop": "stop_price_usdt", "take": "take_price_usdt",
                               "size": "quantity_btc", "stop_size": "stop_size_btc",
                               "take_size": "take_size_btc", "exclusive": "exclusive_oco"}
                    for key, value in change.items():
                        params[mapping.get(key, key)] = value
                    result = ledger.append(SimEvent("protect", 2, "PROTECTION_ACK", **params))
                    self.assertEqual(result.simulated_reserved_btc, D(0))
                    self.assertEqual(result.uncovered_btc, D("0.01998"))
                    self.assertTrue(result.halted)

    def test_rounding_down_after_btc_fee_leaves_dust_unprotected(self):
        with TemporaryDirectory() as directory:
            config = SimulationConfig(min_size_btc=D("0.0001"), lot_size_btc=D("0.0001"))
            ledger = SimulationLedger(Path(directory) / "simulation.sqlite3", config)
            ledger.append(SimEvent("ack", 0, "BUY_ACK"))
            ledger.append(SimEvent("fill", 1, "BUY_FILL", quantity_btc=D("0.02"),
                                   price_usdt=D("80000"), fee_btc=D("-0.00005"), order_id="fill-1"))
            result = ledger.append(SimEvent("protect", 2, "PROTECTION_ACK",
                quantity_btc=D("0.0199"), stop_size_btc=D("0.0199"),
                take_size_btc=D("0.0199"), stop_price_usdt=config.stop_usdt,
                take_price_usdt=config.take_usdt, stop_id="stop-1", take_id="take-1",
                group_id="oco-1", exclusive_oco=True))
            self.assertEqual(result.bot_owned_btc, D("0.01995"))
            self.assertEqual(result.simulated_reserved_btc, D("0.0199"))
            self.assertEqual(result.uncovered_btc, D("0.00005"))
            self.assertTrue(result.halted)

    def test_partial_protection_fill_and_double_sell_cannot_consume_external_btc(self):
        self.buy()
        self.protect(second=1)
        partial = self.add("partial-sl", 45, "PROTECTION_FILL",
                           quantity_btc=D("0.005"), price_usdt=D("79200"),
                           fee_usdt=D("-0.396"), order_id="stop-1")
        self.assertEqual(partial.bot_remaining_btc, D("0.01498"))
        self.assertEqual(partial.uncovered_btc, D("0.01498"))
        self.assertTrue(partial.halted)
        repeated = self.add("double-sell", 46, "PROTECTION_FILL",
                            quantity_btc=D("1"), price_usdt=D("79200"), order_id="take-1")
        self.assertEqual(repeated.bot_remaining_btc, D("0.01498"))
        self.assertEqual(repeated.external_btc, D("1"))
        self.assertIsNone(self.add("price", 60, "PRICE", price_usdt=D("78400"))
                          .price_losses_usdt[0][1])

    def test_full_take_profit_uses_only_bot_btc_and_other_leg_cannot_sell_again(self):
        self.buy()
        self.protect(second=1)
        exited = self.add("take-fill", 30, "PROTECTION_FILL",
                          quantity_btc=D("0.01998"), price_usdt=D("81600"),
                          fee_usdt=D("-1.630368"), order_id="take-1")
        self.assertEqual(exited.bot_remaining_btc, D(0))
        self.assertEqual(exited.external_btc, D("1"))
        self.assertEqual(exited.simulated_reserved_btc, D(0))
        blocked_second = self.add("stop-after-take", 31, "PROTECTION_FILL",
                                  quantity_btc=D("0.01998"), price_usdt=D("79200"),
                                  order_id="stop-1")
        self.assertEqual(blocked_second.bot_remaining_btc, D(0))
        self.assertTrue(blocked_second.halted)

    def test_duplicate_algo_group_invalidates_protection_without_second_reserve(self):
        self.buy()
        self.protect(second=1)
        duplicate = self.protect(second=20, stop_id="stop-2", take_id="take-2", group="oco-2")
        self.assertEqual(duplicate.simulated_reserved_btc, D(0))
        self.assertEqual(duplicate.uncovered_btc, D("0.01998"))
        self.assertTrue(any("дубликат" in reason for reason in duplicate.reasons))

    def test_disconnect_restart_replay_keeps_halted_and_unknown_loss(self):
        self.buy()
        self.add("disconnect", 10, "DISCONNECT")
        before = self.add("restart", 20, "RESTART")
        restarted = SimulationLedger(self.path, self.config)
        self.assertEqual(restarted.replay(), before)
        after = restarted.append(SimEvent("price", 60, "PRICE", price_usdt=D("78400")))
        self.assertTrue(after.halted)
        self.assertIsNone(after.unprotected_seconds)
        self.assertIsNone(after.price_losses_usdt[0][1])

    def test_same_event_id_is_idempotent_but_conflicting_payload_rejected(self):
        event = SimEvent("ack", 0, "BUY_ACK")
        first = self.ledger.append(event)
        self.assertEqual(self.ledger.append(event), first)
        with self.assertRaises(SimulationError):
            self.ledger.append(SimEvent("ack", 1, "BUY_ACK"))
        self.assertEqual(self.ledger.replay(), first)

    def test_working_database_path_is_rejected_without_opening_it(self):
        original = DEFAULT_JOURNAL_PATH.read_bytes() if DEFAULT_JOURNAL_PATH.is_file() else None
        with self.assertRaises(SimulationError):
            SimulationLedger(DEFAULT_JOURNAL_PATH)
        current = DEFAULT_JOURNAL_PATH.read_bytes() if DEFAULT_JOURNAL_PATH.is_file() else None
        self.assertEqual(current, original)

    def test_invalid_instrument_or_cost_assumptions_fail_before_simulation(self):
        for config in ({"lot_size_btc": D(0)}, {"planned_btc": D("2")},
                       {"entry_usdt": D("80000.05")},
                       {"exit_fee_rate": D("0.8"), "exit_slippage_rate": D("0.3")},
                       {"planned_btc": "0.02"}):
            with self.subTest(config=config), self.assertRaises(SimulationError):
                SimulationConfig(**config)

    def test_temporary_sqlite_connections_are_closed_before_windows_cleanup(self):
        temporary = TemporaryDirectory()
        path = Path(temporary.name) / "simulation.sqlite3"
        try:
            ledger = SimulationLedger(path)
            ledger.append(SimEvent("ack", 0, "BUY_ACK"))
            self.assertEqual(SimulationLedger(path).replay().event_count, 1)
        finally:
            temporary.cleanup()  # WinError 32 here would fail the test on Windows.
        self.assertFalse(path.exists())

    def test_corrupt_event_log_is_blocked_after_connections_close(self):
        self.add("ack", 0, "BUY_ACK")
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE simulation_events SET payload='not-json' WHERE event_id='ack'")
            db.commit()
        with self.assertRaises(SimulationError):
            SimulationLedger(self.path).replay()

    def test_price_input_and_cli_never_access_api_or_working_journal(self):
        self.assertEqual(parse_price_path("80000, 79200,78400"),
                         (D("80000"), D("79200"), D("78400")))
        for bad in ("", "NaN", "0", "-1", "80000,", "80000,$100"):
            with self.subTest(bad=bad), self.assertRaises(SimulationError):
                parse_price_path(bad)
        output = StringIO()
        with (patch.object(main.DemoClient, "from_env_file", side_effect=AssertionError("API forbidden")),
              patch.object(main, "OkxClient", side_effect=AssertionError("API forbidden")),
              patch.object(main, "DecisionJournal", side_effect=AssertionError("working DB forbidden")),
              patch.object(main, "RiskManager", side_effect=AssertionError("risk state forbidden")),
              redirect_stdout(output)):
            self.assertEqual(main.main(["--protection-simulate", "--sim-price-path", "80000,78400"]), 1)
        result = output.getvalue()
        self.assertIn("BLOCK", result)
        self.assertIn("неизвестен", result)
        self.assertIn("Время известной незащищённой позиции", result)
        self.assertIn("внешний BTC 1 не использовался", result)


if __name__ == "__main__":
    unittest.main()
