from __future__ import annotations

import unittest
from datetime import datetime, timezone

import config
from trading.auto_trader import AutoTrader, PaperPosition


class ConfigPatchMixin:
    def setUp(self) -> None:
        super().setUp()
        self._cfg_old: dict[str, object] = {}

    def patch_cfg(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            if key not in self._cfg_old:
                self._cfg_old[key] = getattr(config, key, None)
            setattr(config, key, value)

    def tearDown(self) -> None:
        for key, value in self._cfg_old.items():
            setattr(config, key, value)
        super().tearDown()


class AutoTraderExitEngineTests(ConfigPatchMixin, unittest.TestCase):
    def _blank_trader(self) -> AutoTrader:
        trader = AutoTrader.__new__(AutoTrader)
        trader.paper_balance_usd = 0.0
        trader.realized_pnl_usd = 0.0
        trader.day_realized_pnl_usd = 0.0
        trader._update_stair_floor = lambda: None
        trader._save_state = lambda: None
        trader._write_trade_decision = lambda _event: None
        return trader

    @staticmethod
    def _mk_position() -> PaperPosition:
        return PaperPosition(
            token_address="0x1111111111111111111111111111111111111111",
            symbol="TEST",
            entry_price_usd=1.0,
            current_price_usd=1.0,
            position_size_usd=1.0,
            score=95,
            liquidity_usd=10000.0,
            risk_level="LOW",
            opened_at=datetime.now(timezone.utc),
            max_hold_seconds=120,
            take_profit_percent=99,
            stop_loss_percent=5.0,
            expected_edge_percent=2.0,
            buy_cost_percent=0.0,
            sell_cost_percent=0.0,
            gas_cost_usd=0.0,
        )

    def test_partial_take_profit_two_stages(self) -> None:
        self.patch_cfg(
            PAPER_PARTIAL_TP_ENABLED=True,
            PAPER_PARTIAL_TP_TRIGGER_PERCENT=1.0,
            PAPER_PARTIAL_TP_SELL_FRACTION=0.40,
            PAPER_PARTIAL_TP_STAGE2_ENABLED=True,
            PAPER_PARTIAL_TP_STAGE2_TRIGGER_MULT=2.0,
            PAPER_PARTIAL_TP_STAGE2_SELL_FRACTION=0.50,
            PAPER_PARTIAL_TP_MIN_REMAINING_USD=0.05,
            PAPER_PARTIAL_TP_MOVE_SL_TO_BREAK_EVEN=True,
            PAPER_PARTIAL_TP_BREAK_EVEN_BUFFER_PERCENT=0.1,
            PAPER_PARTIAL_TP_BREAK_EVEN_STAGE2_BONUS_PERCENT=0.1,
            PAPER_REALISM_ENABLED=False,
            PAPER_REALISM_CAP_ENABLED=False,
        )
        trader = self._blank_trader()
        pos = self._mk_position()

        trader._maybe_partial_take_profit(pos, raw_price_pnl_percent=1.2)
        self.assertEqual(AutoTrader._partial_tp_stage(pos), 1)
        self.assertTrue(pos.partial_tp_done)
        self.assertAlmostEqual(pos.position_size_usd, 0.6, places=4)

        trader._maybe_partial_take_profit(pos, raw_price_pnl_percent=2.6)
        self.assertEqual(AutoTrader._partial_tp_stage(pos), 2)
        self.assertLess(pos.position_size_usd, 0.6)
        self.assertAlmostEqual(pos.position_size_usd, 0.3, places=4)
        self.assertGreater(trader.realized_pnl_usd, 0.0)

    def test_effective_timeout_extension_is_capped(self) -> None:
        self.patch_cfg(
            EXIT_TIMEOUT_EXTENSION_ENABLED=True,
            EXIT_TIMEOUT_EXTENSION_MAX_SECONDS=50,
            EXIT_TIMEOUT_EXTENSION_EDGE_SECONDS=20,
            EXIT_TIMEOUT_EXTENSION_MIN_EDGE_PERCENT=1.5,
            EXIT_TIMEOUT_EXTENSION_MOMENTUM_SECONDS=30,
            EXIT_TIMEOUT_EXTENSION_MIN_PEAK_PERCENT=1.2,
            EXIT_TIMEOUT_EXTENSION_MIN_PNL_PERCENT=0.2,
            EXIT_TIMEOUT_EXTENSION_POST_PARTIAL_SECONDS=10,
        )
        trader = self._blank_trader()
        pos = self._mk_position()
        pos.max_hold_seconds = 100
        pos.expected_edge_percent = 2.0
        pos.peak_pnl_percent = 2.2
        pos.pnl_percent = 0.8
        pos.partial_tp_stage = 1

        effective_hold = trader._effective_timeout_seconds(pos)
        self.assertEqual(pos.timeout_extension_seconds, 50)
        self.assertEqual(effective_hold, 150)

    def test_trailing_floor_calculation(self) -> None:
        self.patch_cfg(
            PAPER_TRAILING_EXIT_ENABLED=True,
            PAPER_TRAILING_ACTIVATE_PEAK_PERCENT=1.5,
            PAPER_TRAILING_GIVEBACK_PERCENT=1.0,
            PAPER_TRAILING_GIVEBACK_RATIO=0.50,
            PAPER_TRAILING_MIN_PNL_PERCENT=0.2,
        )
        trader = self._blank_trader()
        pos = self._mk_position()
        pos.peak_pnl_percent = 4.0
        floor = trader._trailing_floor_percent(pos)
        self.assertIsNotNone(floor)
        self.assertAlmostEqual(float(floor), 2.0, places=4)

        pos.peak_pnl_percent = 1.2
        self.assertIsNone(trader._trailing_floor_percent(pos))

    def test_momentum_decay_exit_rule(self) -> None:
        self.patch_cfg(
            MOMENTUM_DECAY_EXIT_ENABLED=True,
            MOMENTUM_DECAY_MIN_AGE_PERCENT=20.0,
            MOMENTUM_DECAY_MIN_HOLD_SECONDS=40,
            MOMENTUM_DECAY_MIN_PEAK_PERCENT=2.0,
            MOMENTUM_DECAY_RETAIN_RATIO=0.35,
            MOMENTUM_DECAY_MIN_DROP_PERCENT=1.0,
            MOMENTUM_DECAY_MAX_PNL_PERCENT=0.8,
        )
        trader = self._blank_trader()
        pos = self._mk_position()
        pos.max_hold_seconds = 200
        pos.peak_pnl_percent = 3.0
        pos.pnl_percent = 0.6
        self.assertTrue(trader._should_exit_momentum_decay(pos, age_seconds=55))

        pos.pnl_percent = 1.1
        self.assertFalse(trader._should_exit_momentum_decay(pos, age_seconds=55))


if __name__ == "__main__":
    unittest.main()
