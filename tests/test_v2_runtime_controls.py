from __future__ import annotations

import unittest
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import config
from trading.v2_runtime import (
    DualEntryController,
    PolicyEntryRouter,
    RollingEdgeGovernor,
    RuntimeKpiLoop,
    SourceQosController,
    UniverseQualityGateController,
)


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


def _cand(idx: int, tier: str, score: int = 70) -> tuple[dict, dict]:
    return (
        {
            "_entry_tier": tier,
            "symbol": f"T{idx}",
            "liquidity": 10_000 + idx,
            "volume_5m": 5_000 + idx,
        },
        {"score": score},
    )


@dataclass
class _ClosedPos:
    pnl_usd: float


@dataclass
class _ClosedRichPos:
    symbol: str
    token_cluster_key: str
    pnl_usd: float
    candidate_id: str
    closed_at: datetime


class PolicyRouterTests(ConfigPatchMixin, unittest.TestCase):
    def test_limited_mode_keeps_only_strict_when_configured(self) -> None:
        self.patch_cfg(
            V2_POLICY_ROUTER_ENABLED=True,
            V2_POLICY_FAIL_CLOSED_ACTION="limited",
            V2_POLICY_DEGRADED_ACTION="limited",
            V2_POLICY_LIMITED_ENTRY_RATIO=0.50,
            V2_POLICY_LIMITED_MIN_PER_CYCLE=1,
            V2_POLICY_LIMITED_ONLY_STRICT=True,
            V2_POLICY_LIMITED_ALLOW_EXPLORE_IN_RED=False,
            DATA_POLICY_HARD_BLOCK_ENABLED=False,
        )
        router = PolicyEntryRouter()
        rows = [_cand(1, "A", 95), _cand(2, "B", 90), _cand(3, "A", 88), _cand(4, "B", 82)]
        out, meta = router.route(candidates=rows, policy_state="FAIL_CLOSED", policy_reason="x", market_mode="YELLOW")
        self.assertEqual(meta.get("effective_mode"), "LIMITED")
        self.assertEqual(len(out), 2)
        self.assertTrue(all(str(row[0].get("_entry_tier", "")).upper() == "A" for row in out))

    def test_hard_block_wins_over_limited(self) -> None:
        self.patch_cfg(
            V2_POLICY_ROUTER_ENABLED=True,
            V2_POLICY_FAIL_CLOSED_ACTION="limited",
            V2_POLICY_DEGRADED_ACTION="limited",
            DATA_POLICY_HARD_BLOCK_ENABLED=True,
        )
        router = PolicyEntryRouter()
        out, meta = router.route(candidates=[_cand(1, "A", 90)], policy_state="DEGRADED", policy_reason="x", market_mode="YELLOW")
        self.assertEqual(out, [])
        self.assertEqual(meta.get("action"), "hard_block")
        self.assertEqual(meta.get("effective_mode"), "BLOCKED")

    def test_router_disabled_keeps_legacy_block_for_non_ok(self) -> None:
        self.patch_cfg(V2_POLICY_ROUTER_ENABLED=False, DATA_POLICY_HARD_BLOCK_ENABLED=False)
        router = PolicyEntryRouter()
        rows = [_cand(1, "A", 90), _cand(2, "B", 80)]
        ok_out, _ = router.route(candidates=rows, policy_state="OK", policy_reason="x", market_mode="GREEN")
        blocked_out, _ = router.route(candidates=rows, policy_state="FAIL_CLOSED", policy_reason="x", market_mode="RED")
        self.assertEqual(len(ok_out), len(rows))
        self.assertEqual(blocked_out, [])


class DualEntryTests(ConfigPatchMixin, unittest.TestCase):
    def test_disabled_dual_entry_tags_all_core(self) -> None:
        self.patch_cfg(V2_ENTRY_DUAL_CHANNEL_ENABLED=False)
        dual = DualEntryController()
        out, meta = dual.allocate(candidates=[_cand(1, "A", 90), _cand(2, "B", 80)], market_mode="GREEN")
        self.assertEqual(len(out), 2)
        self.assertEqual(meta.get("explore_out"), 0)
        self.assertTrue(all(str(row[0].get("_entry_channel", "")) == "core" for row in out))

    def test_explore_quota_not_exceeded(self) -> None:
        self.patch_cfg(
            V2_ENTRY_DUAL_CHANNEL_ENABLED=True,
            V2_ENTRY_EXPLORE_MAX_SHARE=0.25,
            V2_ENTRY_EXPLORE_MAX_PER_CYCLE=2,
            V2_ENTRY_EXPLORE_ALLOW_IN_RED=True,
            V2_ENTRY_CORE_MIN_PER_CYCLE=1,
        )
        dual = DualEntryController()
        rows = [_cand(1, "A", 92), _cand(2, "A", 90), _cand(3, "B", 88), _cand(4, "B", 86), _cand(5, "B", 84), _cand(6, "B", 82)]
        out, meta = dual.allocate(candidates=rows, market_mode="YELLOW")
        explore_out = int(meta.get("explore_out", 0) or 0)
        self.assertLessEqual(explore_out, 2)
        self.assertEqual(len(out), int(meta.get("core_out", 0) or 0) + explore_out)

    def test_red_mode_can_disable_explore(self) -> None:
        self.patch_cfg(
            V2_ENTRY_DUAL_CHANNEL_ENABLED=True,
            V2_ENTRY_EXPLORE_MAX_SHARE=0.50,
            V2_ENTRY_EXPLORE_MAX_PER_CYCLE=3,
            V2_ENTRY_EXPLORE_ALLOW_IN_RED=False,
        )
        dual = DualEntryController()
        out, meta = dual.allocate(candidates=[_cand(1, "A", 90), _cand(2, "B", 80), _cand(3, "B", 70)], market_mode="RED")
        self.assertEqual(int(meta.get("explore_out", 0) or 0), 0)
        self.assertTrue(all(str(row[0].get("_entry_channel", "")) == "core" for row in out))


class SourceQosTests(ConfigPatchMixin, unittest.TestCase):
    def test_filter_enforces_topk_and_symbol_cap(self) -> None:
        self.patch_cfg(
            V2_SOURCE_QOS_ENABLED=True,
            V2_SOURCE_QOS_TOPK_ENABLED=True,
            V2_SOURCE_QOS_TOPK_PER_CYCLE=3,
            V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE=1,
            V2_SOURCE_QOS_SOURCE_CAPS="onchain:4,watchlist:4",
        )
        qos = SourceQosController()
        tokens = [
            {"symbol": "AAA", "source": "onchain", "liquidity": 100_000, "volume_5m": 8_000, "price_change_5m": 2.5},
            {"symbol": "AAA", "source": "onchain", "liquidity": 90_000, "volume_5m": 7_500, "price_change_5m": 2.0},
            {"symbol": "BBB", "source": "onchain", "liquidity": 80_000, "volume_5m": 7_000, "price_change_5m": 1.8},
            {"symbol": "CCC", "source": "watchlist", "liquidity": 75_000, "volume_5m": 6_800, "price_change_5m": 1.5},
            {"symbol": "DDD", "source": "watchlist", "liquidity": 70_000, "volume_5m": 6_500, "price_change_5m": 1.2},
        ]
        accepted, dropped, meta = qos.filter_tokens(tokens)
        self.assertEqual(int(meta.get("in_total", 0) or 0), 5)
        self.assertEqual(int(meta.get("out_total", 0) or 0), 3)
        self.assertEqual(len(accepted), 3)
        self.assertEqual(len(dropped), 2)
        symbols = [str(row.get("symbol", "")).upper() for row in accepted]
        self.assertEqual(len(symbols), len(set(symbols)))

    def test_single_source_mode_applies_cap_floor(self) -> None:
        self.patch_cfg(
            V2_SOURCE_QOS_ENABLED=True,
            V2_SOURCE_QOS_TOPK_ENABLED=False,
            V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE=1,
            V2_SOURCE_QOS_SOURCE_CAPS="watchlist:1",
        )
        qos = SourceQosController()
        tokens = [
            {"symbol": f"S{i}", "source": "watchlist", "liquidity": 40_000 + i, "volume_5m": 6_000 + i, "price_change_5m": 1.2}
            for i in range(6)
        ]
        accepted, _dropped, meta = qos.filter_tokens(tokens)
        self.assertTrue(bool(meta.get("single_source_mode")))
        self.assertGreaterEqual(int((meta.get("single_source_cap_floor", {}) or {}).get("watchlist", 0) or 0), 3)
        self.assertGreaterEqual(len(accepted), 3)

    def test_record_cycle_can_trigger_source_cooldown(self) -> None:
        self.patch_cfg(
            V2_SOURCE_QOS_ENABLED=True,
            V2_SOURCE_QOS_WINDOW_CYCLES=3,
            V2_SOURCE_QOS_MIN_EVAL_CYCLES=3,
            V2_SOURCE_QOS_MIN_SEEN_PER_WINDOW=30,
            V2_SOURCE_QOS_MIN_PLAN_PER_WINDOW=6,
            V2_SOURCE_QOS_MIN_PASS_RATE=0.20,
            V2_SOURCE_QOS_MIN_EV_POSITIVE_RATE=0.50,
            V2_SOURCE_QOS_MIN_OPEN_RATE=0.40,
            V2_SOURCE_QOS_REQUIRED_FAIL_SIGNALS=2,
            V2_SOURCE_QOS_COOLDOWN_SECONDS=600,
            V2_SOURCE_QOS_SOURCE_CAPS="onchain:10",
        )
        qos = SourceQosController()
        event = None
        for _ in range(3):
            event = qos.record_cycle(
                seen_by_source={"onchain": 20},
                passed_by_source={"onchain": 1},
                planned_by_source={"onchain": 5},
                source_flow={"onchain": {"plan_attempts": 5, "opens": 0, "ev_net_low_skips": 5, "ev_positive": 0}},
            )
        self.assertIsNotNone(event)
        cooled = [str(x.get("source", "")) for x in list((event or {}).get("cooled_sources", []) or [])]
        self.assertIn("onchain", cooled)
        accepted, dropped, _ = qos.filter_tokens(
            [{"symbol": "AAA", "source": "onchain", "liquidity": 50_000, "volume_5m": 5_000, "price_change_5m": 1.0}]
        )
        self.assertEqual(len(accepted), 0)
        self.assertEqual(str((dropped[0] if dropped else {}).get("reason", "")), "source_qos_cooldown")


class RollingEdgeTests(ConfigPatchMixin, unittest.TestCase):
    def test_relax_path_respects_min_bounds(self) -> None:
        self.patch_cfg(
            V2_ROLLING_EDGE_ENABLED=True,
            V2_ROLLING_EDGE_INTERVAL_SECONDS=60,
            V2_ROLLING_EDGE_MIN_CLOSED=10,
            V2_ROLLING_EDGE_WINDOW_CLOSED=30,
            V2_ROLLING_EDGE_RELAX_STEP_USD=0.002,
            V2_ROLLING_EDGE_RELAX_STEP_PERCENT=0.10,
            V2_ROLLING_EDGE_EDGE_LOW_SHARE_RELAX=0.60,
            V2_ROLLING_EDGE_LOSS_SHARE_TIGHTEN=0.95,
            V2_ROLLING_EDGE_MIN_USD=0.008,
            V2_ROLLING_EDGE_MIN_PERCENT=0.35,
            MIN_EXPECTED_EDGE_USD=0.020,
            MIN_EXPECTED_EDGE_PERCENT=1.20,
        )
        gov = RollingEdgeGovernor()
        gov._next_eval_ts = 0.0
        gov.record_cycle(candidates=100, opened=3, skip_reasons_cycle={"edge_low": 60, "negative_edge": 20})
        auto_stats = {"closed": 30}
        trader = type("T", (), {"closed_positions": [_ClosedPos(0.01) for _ in range(30)]})()
        payload = gov.maybe_apply(auto_trader=trader, auto_stats=auto_stats)
        self.assertIsNotNone(payload)
        self.assertEqual(payload.get("action"), "relax")
        self.assertLess(float(getattr(config, "MIN_EXPECTED_EDGE_USD")), 0.0201)
        self.assertGreaterEqual(float(getattr(config, "MIN_EXPECTED_EDGE_USD")), 0.008)

    def test_tighten_path_on_high_losses(self) -> None:
        self.patch_cfg(
            V2_ROLLING_EDGE_ENABLED=True,
            V2_ROLLING_EDGE_INTERVAL_SECONDS=60,
            V2_ROLLING_EDGE_MIN_CLOSED=10,
            V2_ROLLING_EDGE_WINDOW_CLOSED=30,
            V2_ROLLING_EDGE_TIGHTEN_STEP_USD=0.003,
            V2_ROLLING_EDGE_TIGHTEN_STEP_PERCENT=0.20,
            V2_ROLLING_EDGE_LOSS_SHARE_TIGHTEN=0.50,
            MIN_EXPECTED_EDGE_USD=0.010,
            MIN_EXPECTED_EDGE_PERCENT=0.80,
            V2_ANTI_SELF_TIGHTEN_ENABLED=False,
            V2_RUNTIME_EDGE_RELAX_UNTIL_TS=0.0,
        )
        gov = RollingEdgeGovernor()
        gov._next_eval_ts = 0.0
        gov.record_cycle(candidates=50, opened=8, skip_reasons_cycle={})
        losses = [_ClosedPos(-0.02) for _ in range(20)] + [_ClosedPos(0.01) for _ in range(10)]
        payload = gov.maybe_apply(auto_trader=type("T", (), {"closed_positions": losses})(), auto_stats={"closed": 30})
        self.assertIsNotNone(payload)
        self.assertEqual(payload.get("action"), "tighten")
        self.assertGreater(float(getattr(config, "MIN_EXPECTED_EDGE_USD")), 0.010)

    def test_no_change_when_not_enough_closed_trades(self) -> None:
        self.patch_cfg(V2_ROLLING_EDGE_ENABLED=True, V2_ROLLING_EDGE_MIN_CLOSED=20, MIN_EXPECTED_EDGE_USD=0.020, MIN_EXPECTED_EDGE_PERCENT=1.00)
        gov = RollingEdgeGovernor()
        gov._next_eval_ts = 0.0
        payload = gov.maybe_apply(auto_trader=type("T", (), {"closed_positions": [_ClosedPos(0.01)] * 5})(), auto_stats={"closed": 5})
        self.assertIsNone(payload)
        self.assertAlmostEqual(float(getattr(config, "MIN_EXPECTED_EDGE_USD")), 0.020, places=6)

    def test_anti_self_tighten_blocks_tighten_on_low_flow(self) -> None:
        self.patch_cfg(
            V2_ROLLING_EDGE_ENABLED=True,
            V2_ROLLING_EDGE_INTERVAL_SECONDS=60,
            V2_ROLLING_EDGE_MIN_CLOSED=10,
            V2_ROLLING_EDGE_WINDOW_CLOSED=30,
            V2_ROLLING_EDGE_TIGHTEN_STEP_USD=0.003,
            V2_ROLLING_EDGE_TIGHTEN_STEP_PERCENT=0.20,
            V2_ROLLING_EDGE_RELAX_STEP_USD=0.002,
            V2_ROLLING_EDGE_RELAX_STEP_PERCENT=0.10,
            V2_ROLLING_EDGE_EDGE_LOW_SHARE_RELAX=0.60,
            V2_ROLLING_EDGE_LOSS_SHARE_TIGHTEN=0.50,
            V2_ANTI_SELF_TIGHTEN_ENABLED=True,
            V2_ANTI_SELF_TIGHTEN_OPEN_RATE_THRESHOLD=0.04,
            V2_ANTI_SELF_TIGHTEN_EDGE_LOW_SHARE_THRESHOLD=0.60,
            V2_ANTI_SELF_TIGHTEN_COOLDOWN_SECONDS=900,
            MIN_EXPECTED_EDGE_USD=0.020,
            MIN_EXPECTED_EDGE_PERCENT=1.20,
            V2_RUNTIME_EDGE_RELAX_UNTIL_TS=0.0,
        )
        gov = RollingEdgeGovernor()
        gov._next_eval_ts = 0.0
        gov.record_cycle(candidates=100, opened=1, skip_reasons_cycle={"edge_low": 70, "negative_edge": 20})
        losses = [_ClosedPos(-0.02) for _ in range(20)] + [_ClosedPos(0.01) for _ in range(10)]
        payload = gov.maybe_apply(auto_trader=type("T", (), {"closed_positions": losses})(), auto_stats={"closed": 30})
        self.assertIsNotNone(payload)
        self.assertNotEqual(payload.get("action"), "tighten")
        self.assertLessEqual(float(getattr(config, "MIN_EXPECTED_EDGE_USD")), 0.020)


class KpiLoopTests(ConfigPatchMixin, unittest.TestCase):
    def test_low_flow_boosts_throughput_and_diversity(self) -> None:
        self.patch_cfg(
            V2_KPI_LOOP_ENABLED=True,
            V2_KPI_QUALITY_REBALANCE_ENABLED=False,
            V2_KPI_LOOP_INTERVAL_SECONDS=60,
            V2_KPI_LOOP_WINDOW_CYCLES=3,
            V2_KPI_EDGE_LOW_RELAX_TRIGGER=0.50,
            V2_KPI_OPEN_RATE_LOW_TRIGGER=0.05,
            V2_KPI_POLICY_BLOCK_TRIGGER=0.34,
            V2_KPI_UNIQUE_SYMBOLS_MIN=4,
            V2_KPI_MAX_BUYS_BOOST_STEP=4,
            V2_KPI_MAX_BUYS_CAP=40,
            V2_KPI_TOPN_BOOST_STEP=2,
            V2_KPI_TOPN_CAP=20,
            V2_KPI_EXPLORE_SHARE_STEP=0.05,
            V2_KPI_EXPLORE_SHARE_MAX=0.45,
            V2_KPI_NOVELTY_SHARE_STEP=0.05,
            V2_KPI_NOVELTY_SHARE_MAX=0.55,
            MAX_BUYS_PER_HOUR=24,
            AUTO_TRADE_TOP_N=10,
            V2_ENTRY_EXPLORE_MAX_SHARE=0.20,
            V2_UNIVERSE_NOVELTY_MIN_SHARE=0.20,
        )
        loop = RuntimeKpiLoop()
        loop._next_eval_ts = 0.0
        for _ in range(3):
            loop.record_cycle(
                candidates=100,
                opened=2,
                policy_state="FAIL_CLOSED",
                symbols=["AAA"],
                skip_reasons_cycle={"edge_low": 70, "negative_edge": 20},
            )
        payload = loop.maybe_apply()
        self.assertIsNotNone(payload)
        self.assertGreater(int(getattr(config, "MAX_BUYS_PER_HOUR")), 24)
        self.assertGreater(int(getattr(config, "AUTO_TRADE_TOP_N")), 10)
        self.assertGreater(float(getattr(config, "V2_UNIVERSE_NOVELTY_MIN_SHARE")), 0.20)

    def test_no_apply_when_window_not_full(self) -> None:
        self.patch_cfg(
            V2_KPI_LOOP_ENABLED=True,
            V2_KPI_QUALITY_REBALANCE_ENABLED=False,
            V2_KPI_LOOP_WINDOW_CYCLES=4,
            V2_KPI_LOOP_INTERVAL_SECONDS=60,
        )
        loop = RuntimeKpiLoop()
        loop._next_eval_ts = 0.0
        for _ in range(3):
            loop.record_cycle(candidates=20, opened=2, policy_state="OK", symbols=["A", "B"], skip_reasons_cycle={})
        self.assertIsNone(loop.maybe_apply())

    def test_caps_are_respected(self) -> None:
        self.patch_cfg(
            V2_KPI_LOOP_ENABLED=True,
            V2_KPI_QUALITY_REBALANCE_ENABLED=False,
            V2_KPI_LOOP_WINDOW_CYCLES=3,
            V2_KPI_LOOP_INTERVAL_SECONDS=60,
            V2_KPI_MAX_BUYS_BOOST_STEP=10,
            V2_KPI_MAX_BUYS_CAP=30,
            V2_KPI_TOPN_BOOST_STEP=10,
            V2_KPI_TOPN_CAP=14,
            V2_KPI_EXPLORE_SHARE_STEP=0.10,
            V2_KPI_EXPLORE_SHARE_MAX=0.40,
            V2_KPI_NOVELTY_SHARE_STEP=0.10,
            V2_KPI_NOVELTY_SHARE_MAX=0.50,
            MAX_BUYS_PER_HOUR=29,
            AUTO_TRADE_TOP_N=13,
            V2_ENTRY_EXPLORE_MAX_SHARE=0.39,
            V2_UNIVERSE_NOVELTY_MIN_SHARE=0.49,
        )
        loop = RuntimeKpiLoop()
        loop._next_eval_ts = 0.0
        for _ in range(3):
            loop.record_cycle(candidates=100, opened=1, policy_state="DEGRADED", symbols=["ONE"], skip_reasons_cycle={"edge_usd_low": 40})
        payload = loop.maybe_apply()
        self.assertIsNotNone(payload)
        self.assertLessEqual(int(getattr(config, "MAX_BUYS_PER_HOUR")), 30)
        self.assertLessEqual(int(getattr(config, "AUTO_TRADE_TOP_N")), 14)
        self.assertLessEqual(float(getattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE")), 0.40)
        self.assertLessEqual(float(getattr(config, "V2_UNIVERSE_NOVELTY_MIN_SHARE")), 0.50)

    def test_quality_rebalance_cuts_explore_when_edge_low_is_extreme(self) -> None:
        self.patch_cfg(
            V2_KPI_LOOP_ENABLED=True,
            V2_KPI_QUALITY_REBALANCE_ENABLED=True,
            V2_KPI_LOOP_WINDOW_CYCLES=3,
            V2_KPI_LOOP_INTERVAL_SECONDS=60,
            V2_KPI_EDGE_LOW_HARD_TRIGGER=0.75,
            V2_KPI_QUALITY_REBALANCE_EXPLORE_STEP=0.05,
            V2_KPI_QUALITY_REBALANCE_NOVELTY_STEP=0.04,
            V2_KPI_QUALITY_REBALANCE_TOPN_STEP=1,
            V2_KPI_QUALITY_REBALANCE_MAX_BUYS_STEP=4,
            MAX_BUYS_PER_HOUR=70,
            AUTO_TRADE_TOP_N=20,
            V2_ENTRY_EXPLORE_MAX_SHARE=0.42,
            V2_UNIVERSE_NOVELTY_MIN_SHARE=0.48,
        )
        loop = RuntimeKpiLoop()
        loop._next_eval_ts = 0.0
        for _ in range(3):
            loop.record_cycle(
                candidates=100,
                opened=1,
                policy_state="OK",
                symbols=["ONE", "TWO"],
                skip_reasons_cycle={"edge_low": 80, "negative_edge": 10},
            )
        payload = loop.maybe_apply()
        self.assertIsNotNone(payload)
        self.assertIn("quality_rebalance", str(payload.get("action", "")))
        self.assertLess(int(getattr(config, "AUTO_TRADE_TOP_N")), 20)
        self.assertLess(float(getattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE")), 0.42)

    def test_quality_rebalance_respects_source_budget_toggle(self) -> None:
        self.patch_cfg(
            V2_KPI_LOOP_ENABLED=True,
            V2_KPI_QUALITY_REBALANCE_ENABLED=True,
            V2_KPI_LOOP_WINDOW_CYCLES=3,
            V2_KPI_LOOP_INTERVAL_SECONDS=60,
            V2_KPI_EDGE_LOW_HARD_TRIGGER=0.75,
            V2_KPI_QUALITY_REBALANCE_EXPLORE_STEP=0.05,
            V2_KPI_QUALITY_REBALANCE_NOVELTY_STEP=0.04,
            V2_KPI_QUALITY_REBALANCE_TOPN_STEP=1,
            V2_KPI_QUALITY_REBALANCE_MAX_BUYS_STEP=4,
            MAX_BUYS_PER_HOUR=70,
            AUTO_TRADE_TOP_N=20,
            V2_ENTRY_EXPLORE_MAX_SHARE=0.42,
            V2_UNIVERSE_NOVELTY_MIN_SHARE=0.48,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=False,
        )
        loop = RuntimeKpiLoop()
        loop._next_eval_ts = 0.0
        for _ in range(3):
            loop.record_cycle(
                candidates=100,
                opened=2,
                policy_state="OK",
                symbols=["ONE", "TWO"],
                skip_reasons_cycle={"edge_low": 80, "negative_edge": 10},
            )
        payload = loop.maybe_apply()
        self.assertIsNotNone(payload)
        self.assertIn("quality_rebalance", str(payload.get("action", "")))
        self.assertFalse(bool(getattr(config, "V2_QUALITY_SOURCE_BUDGET_ENABLED")))

    def test_policy_lock_zero_open_blocks_quality_tighten(self) -> None:
        self.patch_cfg(
            V2_KPI_LOOP_ENABLED=True,
            V2_KPI_POLICY_MODE="lock_zero_open",
            V2_KPI_QUALITY_REBALANCE_ENABLED=True,
            V2_KPI_LOOP_WINDOW_CYCLES=3,
            V2_KPI_LOOP_INTERVAL_SECONDS=60,
            V2_KPI_EDGE_LOW_HARD_TRIGGER=0.75,
            V2_KPI_OPEN_RATE_LOW_TRIGGER=0.05,
            V2_KPI_EDGE_LOW_RELAX_TRIGGER=0.60,
            V2_KPI_QUALITY_REBALANCE_EXPLORE_STEP=0.05,
            V2_KPI_QUALITY_REBALANCE_NOVELTY_STEP=0.04,
            V2_KPI_QUALITY_REBALANCE_TOPN_STEP=2,
            V2_KPI_QUALITY_REBALANCE_MAX_BUYS_STEP=8,
            V2_KPI_MAX_BUYS_BOOST_STEP=4,
            V2_KPI_TOPN_BOOST_STEP=2,
            V2_KPI_MAX_BUYS_CAP=90,
            V2_KPI_TOPN_CAP=30,
            MAX_BUYS_PER_HOUR=70,
            AUTO_TRADE_TOP_N=20,
            V2_ENTRY_EXPLORE_MAX_SHARE=0.42,
            V2_UNIVERSE_NOVELTY_MIN_SHARE=0.48,
        )
        loop = RuntimeKpiLoop()
        loop._next_eval_ts = 0.0
        for _ in range(3):
            loop.record_cycle(
                candidates=120,
                opened=0,
                policy_state="OK",
                symbols=["ONE", "TWO"],
                skip_reasons_cycle={"edge_low": 90, "negative_edge": 20},
            )
        payload = loop.maybe_apply()
        self.assertIsNotNone(payload)
        self.assertIn("quality_rebalance_lock_zero_open", str(payload.get("action", "")))
        self.assertGreaterEqual(int(getattr(config, "MAX_BUYS_PER_HOUR")), 70)
        self.assertGreaterEqual(int(getattr(config, "AUTO_TRADE_TOP_N")), 20)
        self.assertGreaterEqual(float(getattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE")), 0.42)

    def test_explore_failsafe_shrinks_share_on_bad_explore_window(self) -> None:
        class _ExplorePos:
            def __init__(self, pnl_usd: float) -> None:
                self.entry_channel = "explore"
                self.pnl_usd = float(pnl_usd)

        self.patch_cfg(
            V2_KPI_LOOP_ENABLED=True,
            V2_KPI_QUALITY_REBALANCE_ENABLED=False,
            V2_KPI_FAST_ANTISTALL_ENABLED=False,
            V2_KPI_LOOP_WINDOW_CYCLES=3,
            V2_KPI_LOOP_INTERVAL_SECONDS=60,
            V2_KPI_EXPLORE_FAILSAFE_ENABLED=True,
            V2_KPI_EXPLORE_FAILSAFE_WINDOW_CLOSED=20,
            V2_KPI_EXPLORE_FAILSAFE_MIN_CLOSED=10,
            V2_KPI_EXPLORE_FAILSAFE_LOSS_SHARE_TRIGGER=0.60,
            V2_KPI_EXPLORE_FAILSAFE_AVG_PNL_TRIGGER_USD=-0.0010,
            V2_KPI_EXPLORE_FAILSAFE_SHRINK_STEP=0.06,
            V2_KPI_EXPLORE_FAILSAFE_MIN_SHARE=0.08,
            V2_KPI_EXPLORE_FAILSAFE_COOLDOWN_SECONDS=0,
            V2_ENTRY_EXPLORE_MAX_SHARE=0.30,
            V2_KPI_EXPLORE_SHARE_MAX=0.55,
            MAX_BUYS_PER_HOUR=30,
            AUTO_TRADE_TOP_N=12,
        )
        loop = RuntimeKpiLoop()
        loop._next_eval_ts = 0.0
        for _ in range(3):
            loop.record_cycle(
                candidates=100,
                opened=8,
                policy_state="OK",
                symbols=["A", "B", "C", "D"],
                skip_reasons_cycle={"cooldown": 4},
            )
        trader = type("T", (), {"closed_positions": [_ExplorePos(-0.010) for _ in range(12)]})()
        payload = loop.maybe_apply(auto_trader=trader)
        self.assertIsNotNone(payload)
        self.assertIn("explore_failsafe_shrink", str(payload.get("action", "")))
        self.assertLess(float(getattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE")), 0.30)

    def test_explore_failsafe_can_recover_share_after_shrink(self) -> None:
        class _ExplorePos:
            def __init__(self, pnl_usd: float) -> None:
                self.entry_channel = "explore"
                self.pnl_usd = float(pnl_usd)

        self.patch_cfg(
            V2_KPI_LOOP_ENABLED=True,
            V2_KPI_QUALITY_REBALANCE_ENABLED=False,
            V2_KPI_FAST_ANTISTALL_ENABLED=False,
            V2_KPI_LOOP_WINDOW_CYCLES=3,
            V2_KPI_LOOP_INTERVAL_SECONDS=60,
            V2_KPI_EXPLORE_FAILSAFE_ENABLED=True,
            V2_KPI_EXPLORE_FAILSAFE_WINDOW_CLOSED=20,
            V2_KPI_EXPLORE_FAILSAFE_MIN_CLOSED=10,
            V2_KPI_EXPLORE_FAILSAFE_LOSS_SHARE_TRIGGER=0.60,
            V2_KPI_EXPLORE_FAILSAFE_AVG_PNL_TRIGGER_USD=-0.0010,
            V2_KPI_EXPLORE_FAILSAFE_SHRINK_STEP=0.06,
            V2_KPI_EXPLORE_FAILSAFE_MIN_SHARE=0.08,
            V2_KPI_EXPLORE_FAILSAFE_RECOVERY_WINRATE_MIN=0.55,
            V2_KPI_EXPLORE_FAILSAFE_RECOVERY_AVG_PNL_MIN_USD=0.0005,
            V2_KPI_EXPLORE_FAILSAFE_RECOVERY_STEP=0.03,
            V2_KPI_EXPLORE_FAILSAFE_COOLDOWN_SECONDS=0,
            V2_ENTRY_EXPLORE_MAX_SHARE=0.30,
            V2_KPI_EXPLORE_SHARE_MAX=0.55,
            MAX_BUYS_PER_HOUR=30,
            AUTO_TRADE_TOP_N=12,
        )
        loop = RuntimeKpiLoop()
        loop._next_eval_ts = 0.0
        # First apply shrink.
        for _ in range(3):
            loop.record_cycle(
                candidates=100,
                opened=8,
                policy_state="OK",
                symbols=["A", "B", "C", "D"],
                skip_reasons_cycle={},
            )
        bad_trader = type("T", (), {"closed_positions": [_ExplorePos(-0.010) for _ in range(12)]})()
        p1 = loop.maybe_apply(auto_trader=bad_trader)
        self.assertIsNotNone(p1)
        shrunk = float(getattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE"))
        self.assertLess(shrunk, 0.30)

        # Then recover on strong explore window.
        loop._next_eval_ts = 0.0
        for _ in range(3):
            loop.record_cycle(
                candidates=100,
                opened=8,
                policy_state="OK",
                symbols=["A", "B", "C", "D"],
                skip_reasons_cycle={},
            )
        good_rows = [_ExplorePos(0.010) for _ in range(10)] + [_ExplorePos(-0.001) for _ in range(2)]
        good_trader = type("T", (), {"closed_positions": good_rows})()
        p2 = loop.maybe_apply(auto_trader=good_trader)
        self.assertIsNotNone(p2)
        self.assertIn("explore_failsafe_recover", str(p2.get("action", "")))
        self.assertGreater(float(getattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE")), shrunk)


class QualityGateTests(ConfigPatchMixin, unittest.TestCase):
    def test_quality_gate_core_and_cooldown_split(self) -> None:
        self.patch_cfg(
            V2_QUALITY_GATE_ENABLED=True,
            V2_QUALITY_REFRESH_SECONDS=60,
            V2_QUALITY_WINDOW_SECONDS=3600,
            V2_QUALITY_MIN_SYMBOL_TRADES=4,
            V2_QUALITY_MIN_CLUSTER_TRADES=4,
            V2_QUALITY_MIN_AVG_PNL_USD=0.0,
            V2_QUALITY_MAX_LOSS_SHARE=0.60,
            V2_QUALITY_BAD_AVG_PNL_USD=-0.0005,
            V2_QUALITY_BAD_LOSS_SHARE=0.65,
            V2_QUALITY_EXPLORE_MAX_SHARE=0.50,
            V2_QUALITY_EXPLORE_MIN_ABS=1,
            V2_QUALITY_COOLDOWN_PROBE_PROBABILITY=0.0,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=False,
            V2_QUALITY_SYMBOL_MAX_SHARE=0.80,
        )
        q = UniverseQualityGateController()
        now = datetime.now(timezone.utc)
        closed = []
        for i in range(5):
            closed.append(
                _ClosedRichPos(
                    symbol="GOOD",
                    token_cluster_key="s3|l2|v2|m1|r0",
                    pnl_usd=0.01,
                    candidate_id=f"cid_good_{i}",
                    closed_at=now,
                )
            )
            closed.append(
                _ClosedRichPos(
                    symbol="BAD",
                    token_cluster_key="s1|l1|v1|m1|r1",
                    pnl_usd=-0.01,
                    candidate_id=f"cid_bad_{i}",
                    closed_at=now,
                )
            )
        trader = type("T", (), {"closed_positions": closed})()
        rows = [
            (
                {"_candidate_id": "new_good", "symbol": "GOOD", "source": "onchain", "liquidity": 20000, "volume_5m": 6000},
                {"score": 88},
            ),
            (
                {"_candidate_id": "new_bad", "symbol": "BAD", "source": "onchain", "liquidity": 15000, "volume_5m": 5000},
                {"score": 86},
            ),
            (
                {"_candidate_id": "new_unknown", "symbol": "NEW", "source": "dexscreener", "liquidity": 14000, "volume_5m": 4000},
                {"score": 78},
            ),
        ]
        out, meta = q.filter_candidates(candidates=rows, auto_trader=trader, market_mode="YELLOW")
        self.assertTrue(bool(meta.get("enabled")))
        out_syms = {str((row[0] or {}).get("symbol", "")) for row in out}
        self.assertIn("GOOD", out_syms)
        self.assertNotIn("BAD", out_syms)
        self.assertGreaterEqual(int(meta.get("core_out", 0) or 0), 1)
        self.assertGreaterEqual(int(meta.get("explore_out", 0) or 0), 1)
        self.assertGreater(int(meta.get("drop_counts", {}).get("cooldown_bucket", 0) or 0), 0)

    def test_quality_gate_forces_single_cooldown_probe_when_empty(self) -> None:
        self.patch_cfg(
            V2_QUALITY_GATE_ENABLED=True,
            V2_QUALITY_REFRESH_SECONDS=60,
            V2_QUALITY_WINDOW_SECONDS=3600,
            V2_QUALITY_MIN_SYMBOL_TRADES=3,
            V2_QUALITY_MIN_CLUSTER_TRADES=3,
            V2_QUALITY_MIN_AVG_PNL_USD=0.0,
            V2_QUALITY_MAX_LOSS_SHARE=0.60,
            V2_QUALITY_BAD_AVG_PNL_USD=-0.0005,
            V2_QUALITY_BAD_LOSS_SHARE=0.65,
            V2_QUALITY_EXPLORE_MAX_SHARE=0.0,
            V2_QUALITY_EXPLORE_MIN_ABS=0,
            V2_QUALITY_COOLDOWN_PROBE_PROBABILITY=0.0,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=False,
            V2_QUALITY_SYMBOL_MAX_SHARE=1.0,
        )
        q = UniverseQualityGateController()
        now = datetime.now(timezone.utc)
        closed = [
            _ClosedRichPos(
                symbol="BAD",
                token_cluster_key="s1|l1|v1|m1|r1",
                pnl_usd=-0.01,
                candidate_id=f"cid_bad_{i}",
                closed_at=now,
            )
            for i in range(4)
        ]
        trader = type("T", (), {"closed_positions": closed})()
        rows = [
            (
                {"_candidate_id": "bad_probe", "symbol": "BAD", "source": "watchlist", "liquidity": 18000, "volume_5m": 4200},
                {"score": 80},
            )
        ]
        out, meta = q.filter_candidates(candidates=rows, auto_trader=trader, market_mode="YELLOW")
        self.assertEqual(len(out), 1)
        self.assertEqual(str((out[0][0] or {}).get("_quality_bucket", "")), "cooldown_forced")
        self.assertEqual(int(meta.get("forced_cooldown_probe_out", 0) or 0), 1)

    def test_quality_gate_limits_symbol_concentration(self) -> None:
        self.patch_cfg(
            V2_QUALITY_GATE_ENABLED=True,
            V2_QUALITY_REFRESH_SECONDS=60,
            V2_QUALITY_WINDOW_SECONDS=3600,
            V2_QUALITY_MIN_SYMBOL_TRADES=99,
            V2_QUALITY_MIN_CLUSTER_TRADES=99,
            V2_QUALITY_EXPLORE_MAX_SHARE=1.0,
            V2_QUALITY_EXPLORE_MIN_ABS=0,
            V2_QUALITY_COOLDOWN_PROBE_PROBABILITY=0.0,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=False,
            V2_QUALITY_SYMBOL_CONCENTRATION_WINDOW_SECONDS=3600,
            V2_QUALITY_SYMBOL_MAX_SHARE=0.34,
            V2_QUALITY_SYMBOL_MIN_ABS_CAP=1,
        )
        q = UniverseQualityGateController()
        trader = type("T", (), {"closed_positions": []})()
        rows = [
            (
                {"_candidate_id": f"cid_{i}", "symbol": "AAA", "source": "onchain", "liquidity": 12000 + i, "volume_5m": 3500 + i},
                {"score": 80 - i},
            )
            for i in range(6)
        ]
        out, meta = q.filter_candidates(candidates=rows, auto_trader=trader, market_mode="YELLOW")
        aa_count = sum(1 for token, _ in out if str((token or {}).get("symbol", "")).upper() == "AAA")
        self.assertLessEqual(aa_count, 2)
        self.assertGreater(int(meta.get("drop_counts", {}).get("symbol_concentration", 0) or 0), 0)

    def test_quality_gate_reserves_explore_even_with_core_overflow(self) -> None:
        self.patch_cfg(
            V2_QUALITY_GATE_ENABLED=True,
            V2_QUALITY_REFRESH_SECONDS=60,
            V2_QUALITY_WINDOW_SECONDS=3600,
            V2_QUALITY_MIN_SYMBOL_TRADES=3,
            V2_QUALITY_MIN_CLUSTER_TRADES=99,
            V2_QUALITY_MIN_AVG_PNL_USD=0.0,
            V2_QUALITY_MAX_LOSS_SHARE=0.90,
            V2_QUALITY_BAD_AVG_PNL_USD=-1.0,
            V2_QUALITY_BAD_LOSS_SHARE=1.0,
            V2_QUALITY_EXPLORE_MAX_SHARE=0.10,
            V2_QUALITY_EXPLORE_RESERVE_SHARE=0.30,
            V2_QUALITY_EXPLORE_MIN_ABS=1,
            V2_QUALITY_COOLDOWN_PROBE_PROBABILITY=0.0,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=False,
            V2_QUALITY_SYMBOL_MAX_SHARE=1.0,
        )
        q = UniverseQualityGateController()
        now = datetime.now(timezone.utc)
        closed = [
            _ClosedRichPos(
                symbol="CORE",
                token_cluster_key="",
                pnl_usd=0.01,
                candidate_id=f"core_{i}",
                closed_at=now,
            )
            for i in range(4)
        ]
        trader = type("T", (), {"closed_positions": closed})()
        rows = [
            (
                {"_candidate_id": f"c_core_{i}", "symbol": "CORE", "source": "watchlist", "liquidity": 20000, "volume_5m": 5000},
                {"score": 90 - i},
            )
            for i in range(4)
        ] + [
            (
                {"_candidate_id": "c_new", "symbol": "NEW", "source": "dexscreener", "liquidity": 21000, "volume_5m": 5200},
                {"score": 86},
            )
        ]
        out, meta = q.filter_candidates(candidates=rows, auto_trader=trader, market_mode="GREEN")
        out_symbols = [str((token or {}).get("symbol", "")).upper() for token, _ in out]
        self.assertIn("NEW", out_symbols)
        self.assertGreaterEqual(int(meta.get("explore_out", 0) or 0), 1)
        self.assertGreaterEqual(int(meta.get("explore_target", 0) or 0), 1)

    def test_quality_gate_provisional_core_promotes_tier_a(self) -> None:
        self.patch_cfg(
            V2_QUALITY_GATE_ENABLED=True,
            V2_QUALITY_REFRESH_SECONDS=60,
            V2_QUALITY_WINDOW_SECONDS=3600,
            V2_QUALITY_MIN_SYMBOL_TRADES=99,
            V2_QUALITY_MIN_CLUSTER_TRADES=99,
            V2_QUALITY_EXPLORE_MAX_SHARE=0.0,
            V2_QUALITY_EXPLORE_RESERVE_SHARE=0.0,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=False,
            V2_QUALITY_SYMBOL_MAX_SHARE=1.0,
            V2_QUALITY_PROVISIONAL_CORE_ENABLED=True,
            V2_QUALITY_PROVISIONAL_CORE_MIN_SCORE=92,
            V2_QUALITY_PROVISIONAL_CORE_MIN_LIQUIDITY_USD=50000,
            V2_QUALITY_PROVISIONAL_CORE_MIN_VOLUME_5M_USD=3000,
            V2_QUALITY_PROVISIONAL_CORE_MIN_ABS_CHANGE_5M=2.0,
            V2_QUALITY_PROVISIONAL_CORE_MAX_SHARE=1.0,
        )
        q = UniverseQualityGateController()
        trader = type("T", (), {"closed_positions": []})()
        rows = [
            (
                {
                    "_candidate_id": "cid_prov",
                    "symbol": "IMP",
                    "source": "onchain",
                    "liquidity": 100000,
                    "volume_5m": 4500,
                    "price_change_5m": 4.5,
                },
                {"score": 95},
            )
        ]
        out, meta = q.filter_candidates(candidates=rows, auto_trader=trader, market_mode="GREEN")
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out[0][0].get("_entry_tier", "")), "A")
        self.assertEqual(str(out[0][0].get("_quality_bucket", "")), "core_provisional")
        self.assertGreaterEqual(int(meta.get("core_provisional_out", 0) or 0), 1)

    def test_quality_gate_core_fallback_promotes_high_score_explore(self) -> None:
        self.patch_cfg(
            V2_QUALITY_GATE_ENABLED=True,
            V2_QUALITY_REFRESH_SECONDS=60,
            V2_QUALITY_WINDOW_SECONDS=3600,
            V2_QUALITY_MIN_SYMBOL_TRADES=99,
            V2_QUALITY_MIN_CLUSTER_TRADES=99,
            V2_QUALITY_EXPLORE_MAX_SHARE=1.0,
            V2_QUALITY_EXPLORE_MIN_ABS=0,
            V2_QUALITY_COOLDOWN_PROBE_PROBABILITY=0.0,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=False,
            V2_QUALITY_SYMBOL_MAX_SHARE=1.0,
            V2_QUALITY_CORE_MIN_ABS=1,
            V2_QUALITY_CORE_FALLBACK_FROM_EXPLORE_ENABLED=True,
            V2_QUALITY_CORE_FALLBACK_SCORE_MIN=90,
            V2_QUALITY_CORE_FALLBACK_MAX_SHARE=0.5,
        )
        q = UniverseQualityGateController()
        trader = type("T", (), {"closed_positions": []})()
        rows = [
            (
                {"_candidate_id": "cid_hi", "symbol": "HIGHS", "source": "onchain", "liquidity": 90000, "volume_5m": 6000},
                {"score": 95},
            ),
            (
                {"_candidate_id": "cid_lo", "symbol": "LOWS", "source": "onchain", "liquidity": 50000, "volume_5m": 2500},
                {"score": 78},
            ),
        ]
        out, meta = q.filter_candidates(candidates=rows, auto_trader=trader, market_mode="GREEN")
        self.assertTrue(out)
        self.assertGreaterEqual(int(meta.get("core_fallback_in", 0) or 0), 1)
        self.assertGreaterEqual(int(meta.get("core_fallback_out", 0) or 0), 1)
        high_row = next((row for row in out if str((row[0] or {}).get("symbol", "")).upper() == "HIGHS"), None)
        self.assertIsNotNone(high_row)
        self.assertEqual(str(high_row[0].get("_entry_tier", "")), "A")
        self.assertEqual(str(high_row[0].get("_quality_bucket", "")), "core_fallback")

    def test_quality_gate_source_ev_cooldown_drops_bad_source(self) -> None:
        self.patch_cfg(
            V2_QUALITY_GATE_ENABLED=True,
            V2_QUALITY_REFRESH_SECONDS=60,
            V2_QUALITY_WINDOW_SECONDS=7200,
            V2_QUALITY_MIN_SYMBOL_TRADES=99,
            V2_QUALITY_MIN_CLUSTER_TRADES=99,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=True,
            V2_QUALITY_SOURCE_MIN_TRADES=5,
            V2_QUALITY_SOURCE_GOOD_AVG_PNL_USD=0.0020,
            V2_QUALITY_SOURCE_BAD_AVG_PNL_USD=-0.0010,
            V2_QUALITY_SOURCE_BAD_LOSS_SHARE=0.60,
            V2_QUALITY_SOURCE_SEVERE_AVG_PNL_USD=-0.0020,
            V2_QUALITY_SOURCE_SEVERE_LOSS_SHARE=0.70,
            V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY=0.0,
            V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY=0.0,
            V2_QUALITY_SOURCE_COOLDOWN_IN_RED_MULT=1.0,
            V2_QUALITY_EXPLORE_MAX_SHARE=1.0,
            V2_QUALITY_EXPLORE_MIN_ABS=0,
            V2_QUALITY_COOLDOWN_PROBE_PROBABILITY=0.0,
            V2_QUALITY_SYMBOL_MAX_SHARE=1.0,
        )
        q = UniverseQualityGateController()
        now = datetime.now(timezone.utc)
        closed = []
        for i in range(12):
            closed.append(
                _ClosedRichPos(
                    symbol="BADSRC",
                    token_cluster_key="s1|l1|v1|m1|r1",
                    pnl_usd=-0.02,
                    candidate_id="cid_bad_src",
                    closed_at=now,
                )
            )
            closed.append(
                _ClosedRichPos(
                    symbol="GOODSRC",
                    token_cluster_key="s3|l3|v2|m1|r0",
                    pnl_usd=0.02,
                    candidate_id="cid_good_src",
                    closed_at=now,
                )
            )
        trader = type("T", (), {"closed_positions": closed})()
        rows = [
            (
                {"_candidate_id": "cid_bad_src", "symbol": "BADSRC", "source": "dexscreener", "liquidity": 60000, "volume_5m": 7000},
                {"score": 92},
            ),
            (
                {"_candidate_id": "cid_good_src", "symbol": "GOODSRC", "source": "watchlist", "liquidity": 65000, "volume_5m": 7200},
                {"score": 90},
            ),
        ]
        out, meta = q.filter_candidates(candidates=rows, auto_trader=trader, market_mode="YELLOW")
        out_syms = {str((row[0] or {}).get("symbol", "")).upper() for row in out}
        self.assertIn("GOODSRC", out_syms)
        self.assertNotIn("BADSRC", out_syms)
        self.assertGreater(int(meta.get("drop_counts", {}).get("source_ev_cooldown", 0) or 0), 0)

    def test_quality_gate_source_cap_soft_fill_reaches_target(self) -> None:
        self.patch_cfg(
            V2_QUALITY_GATE_ENABLED=True,
            V2_QUALITY_REFRESH_SECONDS=60,
            V2_QUALITY_WINDOW_SECONDS=3600,
            V2_QUALITY_MIN_SYMBOL_TRADES=99,
            V2_QUALITY_MIN_CLUSTER_TRADES=99,
            V2_QUALITY_EXPLORE_MAX_SHARE=1.0,
            V2_QUALITY_EXPLORE_MIN_ABS=0,
            V2_QUALITY_COOLDOWN_PROBE_PROBABILITY=0.0,
            V2_QUALITY_SYMBOL_MAX_SHARE=1.0,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=True,
            V2_QUALITY_SOURCE_MIN_TRADES=99,
            V2_QUALITY_SOURCE_MAX_SHARE=0.34,
            V2_QUALITY_SOURCE_CAP_SOFT_FILL_ENABLED=True,
        )
        q = UniverseQualityGateController()
        trader = type("T", (), {"closed_positions": [], "_recent_open_symbols": []})()
        rows = [
            (
                {
                    "_candidate_id": f"cid_{i}",
                    "symbol": f"S{i}",
                    "source": "watchlist",
                    "liquidity": 30000 + i,
                    "volume_5m": 5000 + i,
                },
                {"score": 85 - i},
            )
            for i in range(5)
        ]
        out, _meta = q.filter_candidates(candidates=rows, auto_trader=trader, market_mode="YELLOW")
        self.assertEqual(len(out), 5)

    def test_quality_gate_drop_counts_are_deduped_per_candidate_reason(self) -> None:
        self.patch_cfg(
            V2_QUALITY_GATE_ENABLED=True,
            V2_QUALITY_REFRESH_SECONDS=60,
            V2_QUALITY_WINDOW_SECONDS=3600,
            V2_QUALITY_MIN_SYMBOL_TRADES=99,
            V2_QUALITY_MIN_CLUSTER_TRADES=99,
            V2_QUALITY_EXPLORE_MAX_SHARE=1.0,
            V2_QUALITY_EXPLORE_MIN_ABS=0,
            V2_QUALITY_COOLDOWN_PROBE_PROBABILITY=0.0,
            V2_QUALITY_SYMBOL_MAX_SHARE=1.0,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=True,
            V2_QUALITY_SOURCE_MIN_TRADES=99,
            V2_QUALITY_SOURCE_MAX_SHARE=0.34,
            V2_QUALITY_SOURCE_CAP_SOFT_FILL_ENABLED=False,
        )
        q = UniverseQualityGateController()
        trader = type("T", (), {"closed_positions": [], "_recent_open_symbols": []})()
        rows = [
            (
                {
                    "_candidate_id": f"cid_{i}",
                    "symbol": f"S{i}",
                    "source": "watchlist",
                    "liquidity": 22000 + i,
                    "volume_5m": 4200 + i,
                },
                {"score": 82 - i},
            )
            for i in range(3)
        ]
        out, meta = q.filter_candidates(candidates=rows, auto_trader=trader, market_mode="YELLOW")
        self.assertEqual(len(out), 2)
        self.assertEqual(int(meta.get("drop_counts", {}).get("source_budget", 0) or 0), 1)
        dropped_source_budget = [
            row
            for row in list(meta.get("dropped", []) or [])
            if str((row or {}).get("reason", "")) == "source_budget"
        ]
        self.assertEqual(len(dropped_source_budget), 1)

    def test_quality_gate_does_not_report_duplicate_for_soft_fill_revisit(self) -> None:
        self.patch_cfg(
            V2_QUALITY_GATE_ENABLED=True,
            V2_QUALITY_REFRESH_SECONDS=60,
            V2_QUALITY_WINDOW_SECONDS=3600,
            V2_QUALITY_MIN_SYMBOL_TRADES=99,
            V2_QUALITY_MIN_CLUSTER_TRADES=99,
            V2_QUALITY_EXPLORE_MAX_SHARE=1.0,
            V2_QUALITY_EXPLORE_MIN_ABS=0,
            V2_QUALITY_COOLDOWN_PROBE_PROBABILITY=0.0,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=False,
            V2_QUALITY_SOURCE_CAP_SOFT_FILL_ENABLED=True,
            V2_QUALITY_SYMBOL_MAX_SHARE=1.0,
            V2_QUALITY_SYMBOL_MIN_ABS_CAP=8,
            V2_QUALITY_SYMBOL_CONCENTRATION_WINDOW_SECONDS=300,
        )
        q = UniverseQualityGateController()
        trader = type("T", (), {"closed_positions": [], "_recent_open_symbols": []})()
        rows = [
            (
                {
                    "_candidate_id": f"cid_{i}",
                    "address": f"0x{i+1:040x}",
                    "symbol": f"S{i}",
                    "source": "watchlist",
                    "liquidity": 20000 + i,
                    "volume_5m": 3000 + i,
                },
                {"score": 88 - i},
            )
            for i in range(3)
        ]
        out, meta = q.filter_candidates(candidates=rows, auto_trader=trader, market_mode="YELLOW")
        self.assertEqual(len(out), 3)
        self.assertEqual(int(meta.get("drop_counts", {}).get("duplicate_address", 0) or 0), 0)

    def test_quality_gate_uses_open_history_without_mutating_selected_history(self) -> None:
        self.patch_cfg(
            V2_QUALITY_GATE_ENABLED=True,
            V2_QUALITY_REFRESH_SECONDS=60,
            V2_QUALITY_WINDOW_SECONDS=3600,
            V2_QUALITY_MIN_SYMBOL_TRADES=99,
            V2_QUALITY_MIN_CLUSTER_TRADES=99,
            V2_QUALITY_EXPLORE_MAX_SHARE=1.0,
            V2_QUALITY_EXPLORE_MIN_ABS=0,
            V2_QUALITY_COOLDOWN_PROBE_PROBABILITY=0.0,
            V2_QUALITY_SOURCE_BUDGET_ENABLED=False,
            V2_QUALITY_SYMBOL_MAX_SHARE=1.0,
            V2_QUALITY_SYMBOL_CONCENTRATION_USE_OPEN_HISTORY=True,
        )
        q = UniverseQualityGateController()
        trader = type("T", (), {"closed_positions": [], "_recent_open_symbols": [(time.time(), "AAA")]})()
        rows = [
            (
                {
                    "_candidate_id": "cid_a",
                    "symbol": "AAA",
                    "source": "onchain",
                    "liquidity": 40000,
                    "volume_5m": 6000,
                },
                {"score": 88},
            )
        ]
        q.filter_candidates(candidates=rows, auto_trader=trader, market_mode="GREEN")
        self.assertEqual(len(q._symbol_history), 0)


if __name__ == "__main__":
    unittest.main()
