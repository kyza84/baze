from __future__ import annotations

import unittest
from datetime import datetime, timezone

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
        trader.token_cooldowns = {}
        trader.token_cooldown_reasons = {}
        trader.symbol_cooldowns = {}
        trader._blacklist = {}
        trader._recent_batch_open_timestamps = []
        trader._active_trade_decision_context = None

        async def _no_profit_stop() -> bool:
            return False

        trader._check_live_profit_stop = _no_profit_stop  # type: ignore[method-assign]
        trader._anti_choke_active = lambda: False  # type: ignore[method-assign]
        trader._anti_choke_symbol_dominant = (  # type: ignore[method-assign]
            lambda *, symbol, window_seconds=None: (False, "")
        )
        trader._profit_engine_batch_limit = (  # type: ignore[method-assign]
            lambda *, mode, selected_cap: (int(selected_cap), False, "off")
        )
        trader._rebalance_plan_batch_sources = (  # type: ignore[method-assign]
            lambda *, selected, eligible: list(selected)
        )
        trader._bump_skip_reason = lambda reason: None  # type: ignore[method-assign]
        return trader

    @staticmethod
    def _candidate(
        symbol: str,
        address: str,
        score: int,
        *,
        risk_level: str = "MEDIUM",
        source: str = "watchlist",
        recommendation: str = "BUY",
    ) -> tuple[dict, dict]:
        token = {
            "symbol": symbol,
            "address": address,
            "source": source,
            "liquidity": 100_000.0,
            "volume_5m": 9_000.0,
            "price_change_5m": 2.0,
            "risk_level": risk_level,
        }
        score_data = {
            "recommendation": recommendation,
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

    async def test_prefilter_skips_symbol_cooldown_before_plan_trade(self) -> None:
        self.patch_cfg(
            MIN_TOKEN_SCORE=0,
            AUTO_TRADE_ENTRY_MODE="top_n",
            AUTO_TRADE_TOP_N=3,
            PROFIT_ENGINE_ENABLED=False,
            BURST_GOVERNOR_ENABLED=False,
            ANTI_CHOKE_ALLOW_SYMBOL_COOLDOWN_BYPASS=False,
        )
        trader = self._blank_trader()
        now_ts = datetime.now(timezone.utc).timestamp()
        trader.symbol_cooldowns = {"COOLED": now_ts + 600.0}

        attempted: list[str] = []

        async def _plan_trade(token_data: dict, score_data: dict) -> object | None:
            attempted.append(str(token_data.get("symbol", "")))
            if str(token_data.get("symbol", "")) == "OPEN":
                return object()
            return None

        trader.plan_trade = _plan_trade  # type: ignore[method-assign]
        candidates = [
            self._candidate("COOLED", "0x3333333333333333333333333333333333333333", 95),
            self._candidate("OPEN", "0x4444444444444444444444444444444444444444", 90),
        ]

        opened = await trader.plan_batch(candidates)
        self.assertEqual(opened, 1)
        self.assertEqual(attempted, ["OPEN"])

    def test_watchlist_symbol_cooldown_shape_limits_watch_rows(self) -> None:
        self.patch_cfg(
            PLAN_WATCHLIST_SYMBOL_COOLDOWN_SHAPE_ENABLED=True,
            PLAN_WATCHLIST_SYMBOL_COOLDOWN_MAX_SHARE=0.25,
            PLAN_WATCHLIST_SYMBOL_COOLDOWN_MIN_KEEP=1,
        )
        trader = self._blank_trader()
        now_ts = datetime.now(timezone.utc).timestamp()
        trader.symbol_cooldowns = {
            "SAIRI": now_ts + 600.0,
            "MOLT": now_ts + 600.0,
            "COG": now_ts + 600.0,
        }

        rows = [
            self._candidate("SAIRI", "0x1000000000000000000000000000000000000001", 95, source="watchlist"),
            self._candidate("MOLT", "0x1000000000000000000000000000000000000002", 90, source="watchlist"),
            self._candidate("COG", "0x1000000000000000000000000000000000000003", 88, source="watchlist"),
            self._candidate("NW_A", "0x2000000000000000000000000000000000000001", 70, source="onchain"),
            self._candidate("NW_B", "0x2000000000000000000000000000000000000002", 72, source="geckoterminal"),
        ]

        shaped = trader._shape_watchlist_symbol_cooldown_rows(rows, stage="unit")
        shaped_symbols = [str(token.get("symbol", "")) for token, _ in shaped]
        watch_count = sum(1 for token, _ in shaped if str(token.get("source", "")) == "watchlist")
        self.assertIn("NW_A", shaped_symbols)
        self.assertIn("NW_B", shaped_symbols)
        self.assertEqual(watch_count, 1)
        self.assertEqual(len(shaped), 3)

    def test_watchlist_symbol_cooldown_shape_keeps_min_when_only_watchlist(self) -> None:
        self.patch_cfg(
            PLAN_WATCHLIST_SYMBOL_COOLDOWN_SHAPE_ENABLED=True,
            PLAN_WATCHLIST_SYMBOL_COOLDOWN_MAX_SHARE=0.25,
            PLAN_WATCHLIST_SYMBOL_COOLDOWN_MIN_KEEP=1,
        )
        trader = self._blank_trader()
        now_ts = datetime.now(timezone.utc).timestamp()
        trader.symbol_cooldowns = {
            "SAIRI": now_ts + 600.0,
            "MOLT": now_ts + 600.0,
            "COG": now_ts + 600.0,
        }
        rows = [
            self._candidate("SAIRI", "0x3000000000000000000000000000000000000001", 95, source="watchlist"),
            self._candidate("MOLT", "0x3000000000000000000000000000000000000002", 90, source="watchlist"),
            self._candidate("COG", "0x3000000000000000000000000000000000000003", 88, source="watchlist"),
        ]
        shaped = trader._shape_watchlist_symbol_cooldown_rows(rows, stage="unit")
        self.assertEqual(len(shaped), 1)
        self.assertEqual(str(shaped[0][0].get("source", "")), "watchlist")

    def test_prefilter_source_shape_keeps_non_watch_rows(self) -> None:
        self.patch_cfg(
            PLAN_PREFILTER_SOURCE_SHAPE_ENABLED=True,
            PLAN_PREFILTER_MAX_WATCHLIST_SHARE=0.50,
            PLAN_PREFILTER_MIN_NON_WATCH_ROWS=2,
            PLAN_PREFILTER_MAX_WATCH_PER_NON_WATCH=5,
        )
        trader = self._blank_trader()
        rows = [
            self._candidate("WL1", "0x4000000000000000000000000000000000000001", 95, source="watchlist"),
            self._candidate("WL2", "0x4000000000000000000000000000000000000002", 94, source="watchlist"),
            self._candidate("WL3", "0x4000000000000000000000000000000000000003", 93, source="watchlist"),
            self._candidate("WL4", "0x4000000000000000000000000000000000000004", 92, source="watchlist"),
            self._candidate("NW1", "0x5000000000000000000000000000000000000001", 80, source="onchain"),
            self._candidate("NW2", "0x5000000000000000000000000000000000000002", 79, source="geckoterminal"),
        ]
        shaped = trader._shape_prefilter_source_mix_rows(rows, stage="unit")
        watch_count = sum(1 for token, _ in shaped if str(token.get("source", "")).startswith("watchlist"))
        non_watch_count = len(shaped) - watch_count
        self.assertEqual(len(shaped), 5)
        self.assertEqual(watch_count, 3)
        self.assertEqual(non_watch_count, 2)

    def test_prefilter_source_shape_respects_watch_per_non_watch_cap(self) -> None:
        self.patch_cfg(
            PLAN_PREFILTER_SOURCE_SHAPE_ENABLED=True,
            PLAN_PREFILTER_MAX_WATCHLIST_SHARE=0.95,
            PLAN_PREFILTER_MIN_NON_WATCH_ROWS=1,
            PLAN_PREFILTER_MAX_WATCH_PER_NON_WATCH=2,
        )
        trader = self._blank_trader()
        rows = [
            self._candidate("WL1", "0x7000000000000000000000000000000000000001", 99, source="watchlist"),
            self._candidate("WL2", "0x7000000000000000000000000000000000000002", 98, source="watchlist"),
            self._candidate("WL3", "0x7000000000000000000000000000000000000003", 97, source="watchlist"),
            self._candidate("WL4", "0x7000000000000000000000000000000000000004", 96, source="watchlist"),
            self._candidate("WL5", "0x7000000000000000000000000000000000000005", 95, source="watchlist"),
            self._candidate("NW1", "0x7100000000000000000000000000000000000001", 80, source="onchain"),
        ]
        shaped = trader._shape_prefilter_source_mix_rows(rows, stage="unit")
        watch_count = sum(1 for token, _ in shaped if str(token.get("source", "")).startswith("watchlist"))
        self.assertEqual(watch_count, 2)
        self.assertEqual(len(shaped), 3)

    def test_prefilter_source_shape_noop_without_non_watch(self) -> None:
        self.patch_cfg(
            PLAN_PREFILTER_SOURCE_SHAPE_ENABLED=True,
            PLAN_PREFILTER_MAX_WATCHLIST_SHARE=0.20,
            PLAN_PREFILTER_MIN_NON_WATCH_ROWS=2,
        )
        trader = self._blank_trader()
        rows = [
            self._candidate("WL1", "0x6000000000000000000000000000000000000001", 95, source="watchlist"),
            self._candidate("WL2", "0x6000000000000000000000000000000000000002", 94, source="watchlist"),
            self._candidate("WL3", "0x6000000000000000000000000000000000000003", 93, source="watchlist"),
        ]
        shaped = trader._shape_prefilter_source_mix_rows(rows, stage="unit")
        self.assertEqual(len(shaped), 3)

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

    async def test_burst_fallback_can_expand_beyond_selected_cap(self) -> None:
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
        trader._profit_engine_batch_limit = (  # type: ignore[method-assign]
            lambda *, mode, selected_cap: (1, False, "off")
        )

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

    async def test_profit_priority_prefers_higher_net_edge_over_raw_score(self) -> None:
        self.patch_cfg(
            MIN_TOKEN_SCORE=0,
            AUTO_TRADE_ENTRY_MODE="single",
            AUTO_TRADE_TOP_N=1,
            PROFIT_ENGINE_ENABLED=False,
            BURST_GOVERNOR_ENABLED=False,
            PLAN_PROFIT_PRIORITY_ENABLED=True,
        )
        trader = self._blank_trader()
        attempted: list[str] = []

        async def _plan_trade(token_data: dict, score_data: dict) -> object | None:
            attempted.append(str(token_data.get("symbol", "")))
            return object()

        trader.plan_trade = _plan_trade  # type: ignore[method-assign]
        weak_high_score = self._candidate(
            "HI_SCORE_WEAK",
            "0x1111111111111111111111111111111111111112",
            100,
            risk_level="HIGH",
        )
        weak_high_score[0]["liquidity"] = 8_000.0
        weak_high_score[0]["volume_5m"] = 800.0
        strong_lower_score = self._candidate(
            "LOWER_SCORE_STRONG",
            "0x2222222222222222222222222222222222222223",
            92,
            risk_level="LOW",
        )
        strong_lower_score[0]["liquidity"] = 450_000.0
        strong_lower_score[0]["volume_5m"] = 80_000.0

        opened = await trader.plan_batch([weak_high_score, strong_lower_score])
        self.assertEqual(opened, 1)
        self.assertEqual(attempted, ["LOWER_SCORE_STRONG"])

    async def test_profit_tier_mix_keeps_mid_tier_in_topn(self) -> None:
        self.patch_cfg(
            MIN_TOKEN_SCORE=0,
            AUTO_TRADE_ENTRY_MODE="top_n",
            AUTO_TRADE_TOP_N=3,
            PROFIT_ENGINE_ENABLED=False,
            BURST_GOVERNOR_ENABLED=False,
            PLAN_PROFIT_PRIORITY_ENABLED=True,
            PLAN_PROFIT_PRIORITY_HIGH_EDGE_USD=0.02,
            PLAN_PROFIT_PRIORITY_MID_EDGE_USD=0.008,
            PLAN_PROFIT_PRIORITY_MINOR_SHARE=0.34,
            PLAN_PROFIT_PRIORITY_MINOR_MIN_SLOTS=1,
        )
        trader = self._blank_trader()

        hint_map = {
            "A_HI": (0.060, 5.0, 1.0),
            "B_HI": (0.050, 4.5, 1.0),
            "C_HI": (0.040, 4.0, 1.0),
            "D_MID": (0.009, 2.0, 1.0),
        }

        def _hint(token_data: dict, score_data: dict) -> tuple[float, float, float]:
            symbol = str(token_data.get("symbol", ""))
            return hint_map.get(symbol, (0.0, 0.0, 0.1))

        trader._batch_candidate_profit_hint = _hint  # type: ignore[method-assign]
        attempted: list[str] = []

        async def _plan_trade(token_data: dict, score_data: dict) -> object | None:
            attempted.append(str(token_data.get("symbol", "")))
            return None

        trader.plan_trade = _plan_trade  # type: ignore[method-assign]
        candidates = [
            self._candidate("A_HI", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 100, risk_level="LOW"),
            self._candidate("B_HI", "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 99, risk_level="LOW"),
            self._candidate("C_HI", "0xcccccccccccccccccccccccccccccccccccccccc", 98, risk_level="LOW"),
            self._candidate("D_MID", "0xdddddddddddddddddddddddddddddddddddddddd", 97, risk_level="LOW"),
        ]

        await trader.plan_batch(candidates)
        self.assertEqual(len(attempted), 3)
        self.assertIn("D_MID", attempted)

    async def test_plan_batch_non_watch_bridge_injects_when_non_watch_buy_missing(self) -> None:
        self.patch_cfg(
            MIN_TOKEN_SCORE=50,
            PLAN_NON_WATCH_MIN_SCORE=50,
            PLAN_NON_WATCH_BRIDGE_ENABLED=True,
            PLAN_NON_WATCH_BRIDGE_ALLOW_HOLD=True,
            PLAN_NON_WATCH_BRIDGE_MIN_SCORE=35,
            PLAN_NON_WATCH_BRIDGE_MIN_PER_BATCH=1,
            PLAN_NON_WATCH_BRIDGE_MAX_PER_BATCH=2,
            AUTO_TRADE_ENTRY_MODE="top_n",
            AUTO_TRADE_TOP_N=2,
            PROFIT_ENGINE_ENABLED=False,
            BURST_GOVERNOR_ENABLED=False,
            PLAN_NON_WATCH_BRIDGE_POST_PREFILTER_ENABLED=True,
            PLAN_NON_WATCH_BRIDGE_POST_PREFILTER_MAX_PER_BATCH=2,
        )
        trader = self._blank_trader()
        trader._batch_candidate_profit_hint = (  # type: ignore[method-assign]
            lambda token_data, score_data: (float(score_data.get("score", 0)) / 1000.0, float(score_data.get("score", 0)), 0.3)
        )
        attempted: list[str] = []

        async def _plan_trade(token_data: dict, score_data: dict) -> object | None:
            attempted.append(str(token_data.get("symbol", "")))
            return None

        trader.plan_trade = _plan_trade  # type: ignore[method-assign]
        candidates = [
            self._candidate("WL_BUY", "0x1111111111111111111111111111111111111111", 90, source="watchlist", recommendation="BUY"),
            self._candidate("NW_HOLD", "0x2222222222222222222222222222222222222222", 40, source="onchain", recommendation="HOLD"),
        ]

        await trader.plan_batch(candidates)
        self.assertIn("NW_HOLD", attempted)

    async def test_plan_batch_post_prefilter_bridge_recovers_non_watch_after_cooldown_drop(self) -> None:
        self.patch_cfg(
            MIN_TOKEN_SCORE=50,
            PLAN_NON_WATCH_MIN_SCORE=50,
            PLAN_NON_WATCH_BRIDGE_ENABLED=True,
            PLAN_NON_WATCH_BRIDGE_ALLOW_HOLD=True,
            PLAN_NON_WATCH_BRIDGE_MIN_SCORE=35,
            PLAN_NON_WATCH_BRIDGE_MIN_PER_BATCH=1,
            PLAN_NON_WATCH_BRIDGE_MAX_PER_BATCH=1,
            PLAN_NON_WATCH_BRIDGE_POST_PREFILTER_ENABLED=True,
            PLAN_NON_WATCH_BRIDGE_POST_PREFILTER_MAX_PER_BATCH=1,
            AUTO_TRADE_ENTRY_MODE="top_n",
            AUTO_TRADE_TOP_N=2,
            PROFIT_ENGINE_ENABLED=False,
            BURST_GOVERNOR_ENABLED=False,
            ANTI_CHOKE_ALLOW_SYMBOL_COOLDOWN_BYPASS=False,
        )
        trader = self._blank_trader()
        now_ts = datetime.now(timezone.utc).timestamp()
        trader.symbol_cooldowns = {"NW_BUY_COOLED": now_ts + 600.0}
        trader._batch_candidate_profit_hint = (  # type: ignore[method-assign]
            lambda token_data, score_data: (float(score_data.get("score", 0)) / 1000.0, float(score_data.get("score", 0)), 0.3)
        )
        attempted: list[str] = []

        async def _plan_trade(token_data: dict, score_data: dict) -> object | None:
            attempted.append(str(token_data.get("symbol", "")))
            return None

        trader.plan_trade = _plan_trade  # type: ignore[method-assign]
        candidates = [
            self._candidate("WL_BUY", "0x8111111111111111111111111111111111111111", 90, source="watchlist", recommendation="BUY"),
            self._candidate("NW_BUY_COOLED", "0x8222222222222222222222222222222222222222", 80, source="onchain", recommendation="BUY"),
            self._candidate("NW_HOLD_RECOVER", "0x8333333333333333333333333333333333333333", 40, source="geckoterminal", recommendation="HOLD"),
        ]

        await trader.plan_batch(candidates)
        self.assertIn("NW_HOLD_RECOVER", attempted)

    async def test_plan_batch_bridge_fills_non_watch_deficit_when_partial_non_watch_exists(self) -> None:
        self.patch_cfg(
            MIN_TOKEN_SCORE=50,
            PLAN_NON_WATCH_MIN_SCORE=50,
            PLAN_NON_WATCH_BRIDGE_ENABLED=True,
            PLAN_NON_WATCH_BRIDGE_ALLOW_HOLD=True,
            PLAN_NON_WATCH_BRIDGE_MIN_SCORE=48,
            PLAN_NON_WATCH_BRIDGE_PROBE_MIN_SCORE=30,
            PLAN_NON_WATCH_BRIDGE_MIN_PER_BATCH=2,
            PLAN_NON_WATCH_BRIDGE_MAX_PER_BATCH=2,
            PLAN_NON_WATCH_BRIDGE_POST_PREFILTER_ENABLED=True,
            PLAN_NON_WATCH_BRIDGE_POST_PREFILTER_MAX_PER_BATCH=2,
            AUTO_TRADE_ENTRY_MODE="all",
            PROFIT_ENGINE_ENABLED=False,
            BURST_GOVERNOR_ENABLED=False,
        )
        trader = self._blank_trader()
        trader._batch_candidate_profit_hint = (  # type: ignore[method-assign]
            lambda token_data, score_data: (float(score_data.get("score", 0)) / 1000.0, float(score_data.get("score", 0)), 0.3)
        )
        attempted: list[str] = []

        async def _plan_trade(token_data: dict, score_data: dict) -> object | None:
            attempted.append(str(token_data.get("symbol", "")))
            return None

        trader.plan_trade = _plan_trade  # type: ignore[method-assign]
        wl_buy = self._candidate("WL_BUY", "0x9111111111111111111111111111111111111111", 90, source="watchlist", recommendation="BUY")
        nw_buy = self._candidate("NW_BUY", "0x9222222222222222222222222222222222222222", 80, source="onchain", recommendation="BUY")
        nw_hold_probe = self._candidate(
            "NW_HOLD_PROBE",
            "0x9333333333333333333333333333333333333333",
            34,
            source="geckoterminal",
            recommendation="HOLD",
        )
        nw_hold_probe[0]["_non_watch_score_probe_pass"] = True
        candidates = [wl_buy, nw_buy, nw_hold_probe]

        await trader.plan_batch(candidates)
        self.assertIn("NW_BUY", attempted)
        self.assertIn("NW_HOLD_PROBE", attempted)

    def test_signal_gate_allows_non_watch_bridge_hold(self) -> None:
        self.patch_cfg(
            MIN_TOKEN_SCORE=50,
            PLAN_NON_WATCH_BRIDGE_ENABLED=True,
            PLAN_NON_WATCH_BRIDGE_ALLOW_HOLD=True,
            PLAN_NON_WATCH_BRIDGE_MIN_SCORE=35,
        )
        trader = self._blank_trader()
        ok, recommendation, score, path = trader._signal_allows_plan(
            {"source": "onchain", "_plan_bridge_non_watch": True},
            {"recommendation": "HOLD", "score": 40},
        )
        self.assertTrue(ok)
        self.assertEqual(recommendation, "HOLD")
        self.assertEqual(score, 40)
        self.assertEqual(path, "non_watch_bridge_signal")

    def test_signal_gate_respects_row_specific_bridge_min_score(self) -> None:
        self.patch_cfg(
            MIN_TOKEN_SCORE=50,
            PLAN_NON_WATCH_BRIDGE_ENABLED=True,
            PLAN_NON_WATCH_BRIDGE_ALLOW_HOLD=True,
            PLAN_NON_WATCH_BRIDGE_MIN_SCORE=45,
        )
        trader = self._blank_trader()
        ok, recommendation, score, path = trader._signal_allows_plan(
            {
                "source": "onchain",
                "_plan_bridge_non_watch": True,
                "_plan_bridge_min_score": 30,
            },
            {"recommendation": "HOLD", "score": 35},
        )
        self.assertTrue(ok)
        self.assertEqual(recommendation, "HOLD")
        self.assertEqual(score, 35)
        self.assertEqual(path, "non_watch_bridge_signal")

    def test_signal_gate_rejects_watchlist_bridge_hold(self) -> None:
        self.patch_cfg(
            MIN_TOKEN_SCORE=50,
            PLAN_NON_WATCH_BRIDGE_ENABLED=True,
            PLAN_NON_WATCH_BRIDGE_ALLOW_HOLD=True,
            PLAN_NON_WATCH_BRIDGE_MIN_SCORE=35,
        )
        trader = self._blank_trader()
        ok, _recommendation, _score, path = trader._signal_allows_plan(
            {"source": "watchlist", "_plan_bridge_non_watch": True},
            {"recommendation": "HOLD", "score": 40},
        )
        self.assertFalse(ok)
        self.assertEqual(path, "signal")


if __name__ == "__main__":
    unittest.main()
