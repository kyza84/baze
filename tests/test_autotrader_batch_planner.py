from __future__ import annotations

import unittest

import config
from trading.auto_trader import AutoTrader
from utils.addressing import normalize_address


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


class AutoTraderBatchPlannerTests(ConfigPatchMixin, unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _blank_trader() -> AutoTrader:
        trader = AutoTrader.__new__(AutoTrader)
        trader.open_positions = {}
        trader._recent_batch_open_timestamps = []
        trader._active_trade_decision_context = None

        async def _no_profit_stop() -> bool:
            return False

        trader._check_live_profit_stop = _no_profit_stop  # type: ignore[method-assign]
        trader._profit_engine_batch_limit = (  # type: ignore[method-assign]
            lambda *, mode, selected_cap: (int(selected_cap), False, "off")
        )
        trader._rebalance_plan_batch_sources = (  # type: ignore[method-assign]
            lambda *, selected, eligible: list(selected)
        )
        trader._bump_skip_reason = lambda reason: None  # type: ignore[method-assign]
        return trader

    @staticmethod
    def _candidate(symbol: str, address: str, score: int, *, risk_level: str = "MEDIUM") -> tuple[dict, dict]:
        token = {
            "symbol": symbol,
            "address": address,
            "source": "watchlist",
            "liquidity": 100_000.0,
            "volume_5m": 9_000.0,
            "price_change_5m": 2.0,
            "risk_level": risk_level,
        }
        score_data = {
            "recommendation": "BUY",
            "score": int(score),
        }
        return token, score_data

    async def test_prefilter_skips_open_duplicate_before_plan_trade(self) -> None:
        self.patch_cfg(
            MIN_TOKEN_SCORE=0,
            AUTO_TRADE_ENTRY_MODE="top_n",
            AUTO_TRADE_TOP_N=3,
            PROFIT_ENGINE_ENABLED=False,
            BURST_GOVERNOR_ENABLED=False,
        )
        trader = self._blank_trader()
        open_addr = normalize_address("0x1111111111111111111111111111111111111111")
        trader.open_positions = {str(open_addr): object()}

        attempted: list[str] = []

        async def _plan_trade(token_data: dict, score_data: dict) -> object | None:
            attempted.append(str(token_data.get("symbol", "")))
            if str(token_data.get("symbol", "")) == "NEW":
                return object()
            return None

        trader.plan_trade = _plan_trade  # type: ignore[method-assign]
        candidates = [
            self._candidate("OPENED", "0x1111111111111111111111111111111111111111", 95),
            self._candidate("NEW", "0x2222222222222222222222222222222222222222", 90),
        ]

        opened = await trader.plan_batch(candidates)
        self.assertEqual(opened, 1)
        self.assertEqual(attempted, ["NEW"])

    async def test_burst_attempts_can_fallback_after_first_skip(self) -> None:
        self.patch_cfg(
            MIN_TOKEN_SCORE=0,
            AUTO_TRADE_ENTRY_MODE="top_n",
            AUTO_TRADE_TOP_N=6,
            PROFIT_ENGINE_ENABLED=False,
            BURST_GOVERNOR_ENABLED=True,
            BURST_GOVERNOR_MAX_OPENS_PER_BATCH=1,
            BURST_GOVERNOR_MAX_PER_SYMBOL_PER_BATCH=1,
            BURST_GOVERNOR_MAX_PER_CLUSTER_PER_BATCH=1,
            BURST_GOVERNOR_WINDOW_SECONDS=120,
            BURST_GOVERNOR_MAX_OPENS_PER_WINDOW=1,
            BURST_GOVERNOR_ATTEMPT_MULTIPLIER=3,
        )
        trader = self._blank_trader()
        trader._token_cluster_key = lambda **kwargs: str(kwargs.get("risk_level", "MEDIUM"))  # type: ignore[method-assign]

        attempted: list[str] = []

        async def _plan_trade(token_data: dict, score_data: dict) -> object | None:
            symbol = str(token_data.get("symbol", ""))
            attempted.append(symbol)
            if symbol == "CAND2":
                return object()
            return None

        trader.plan_trade = _plan_trade  # type: ignore[method-assign]
        candidates = [
            self._candidate("CAND1", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 100, risk_level="LOW"),
            self._candidate("CAND2", "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 99, risk_level="MEDIUM"),
            self._candidate("CAND3", "0xcccccccccccccccccccccccccccccccccccccccc", 98, risk_level="HIGH"),
        ]

        opened = await trader.plan_batch(candidates)
        self.assertEqual(opened, 1)
        self.assertGreaterEqual(len(attempted), 2)
        self.assertEqual(attempted[:2], ["CAND1", "CAND2"])


if __name__ == "__main__":
    unittest.main()
