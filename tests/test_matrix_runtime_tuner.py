from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import matrix_runtime_tuner as mrt  # noqa: E402


def _actions_by_key(actions: list[mrt.Action]) -> dict[str, mrt.Action]:
    return {a.key: a for a in actions}


class MatrixRuntimeTunerTests(unittest.TestCase):
    def test_active_override_subset_prefers_env_over_stale_active_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs_dir = root / "data" / "matrix" / "runs"
            env_dir = root / "data" / "matrix" / "env"
            runs_dir.mkdir(parents=True, exist_ok=True)
            env_dir.mkdir(parents=True, exist_ok=True)

            env_file = env_dir / "u_case.env"
            env_file.write_text(
                "PLAN_MAX_WATCHLIST_SHARE=0.35\nMIN_EXPECTED_EDGE_PERCENT=0.03\n",
                encoding="utf-8",
            )

            active_payload = {
                "items": [
                    {
                        "id": "u_case",
                        "env_file": str(env_file),
                        "overrides": {
                            "PLAN_MAX_WATCHLIST_SHARE": "0.20",
                            "MIN_EXPECTED_EDGE_PERCENT": "0.05",
                        },
                    }
                ]
            }
            (runs_dir / "active_matrix.json").write_text(
                json.dumps(active_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            subset = mrt._active_override_subset(
                root,
                "u_case",
                {"PLAN_MAX_WATCHLIST_SHARE", "MIN_EXPECTED_EDGE_PERCENT"},
            )
            self.assertEqual(subset.get("PLAN_MAX_WATCHLIST_SHARE"), "0.35")
            self.assertEqual(subset.get("MIN_EXPECTED_EDGE_PERCENT"), "0.03")

    def test_relax_score_can_go_below_static_mode_floor_when_flow_is_starved(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=30,
            opened_from_batch=0,
            filter_fail_reasons=Counter({"score_min": 10}),
        )
        overrides = {
            "MARKET_MODE_STRICT_SCORE": "45",
            "MARKET_MODE_SOFT_SCORE": "40",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        self.assertIn("MARKET_MODE_STRICT_SCORE", by_key)
        self.assertIn("MARKET_MODE_SOFT_SCORE", by_key)
        self.assertLess(int(by_key["MARKET_MODE_STRICT_SCORE"].new_value), 45)
        self.assertLess(int(by_key["MARKET_MODE_SOFT_SCORE"].new_value), 40)

    def test_relax_score_when_score_min_blocks_and_no_opens(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=5,
            opened_from_batch=0,
            filter_fail_reasons=Counter({"score_min": 12}),
        )
        overrides = {
            "MARKET_MODE_STRICT_SCORE": "60",
            "MARKET_MODE_SOFT_SCORE": "52",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        self.assertIn("MARKET_MODE_STRICT_SCORE", by_key)
        self.assertIn("MARKET_MODE_SOFT_SCORE", by_key)
        self.assertLess(int(by_key["MARKET_MODE_STRICT_SCORE"].new_value), 60)
        self.assertLess(int(by_key["MARKET_MODE_SOFT_SCORE"].new_value), 52)

    def test_enforce_soft_leq_strict_relation(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=2,
            opened_from_batch=0,
            filter_fail_reasons=Counter(),
        )
        overrides = {
            "MARKET_MODE_STRICT_SCORE": "45",
            "MARKET_MODE_SOFT_SCORE": "60",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        self.assertIn("MARKET_MODE_SOFT_SCORE", by_key)
        self.assertEqual(by_key["MARKET_MODE_SOFT_SCORE"].new_value, "45")

    def test_disable_profit_engine_on_cold_start_fast_mode(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=3,
            opened_from_batch=0,
            pe_reasons=Counter({"cold_start": 3}),
        )
        overrides = {"PROFIT_ENGINE_ENABLED": "true"}
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["fast"])
        by_key = _actions_by_key(actions)
        self.assertIn("PROFIT_ENGINE_ENABLED", by_key)
        self.assertEqual(by_key["PROFIT_ENGINE_ENABLED"].new_value, "false")

    def test_build_actions_do_not_touch_safety_guard_keys_in_any_mode(self) -> None:
        locked = {
            "TOKEN_SAFETY_FAIL_CLOSED",
            "HONEYPOT_API_FAIL_CLOSED",
            "SAFE_REQUIRE_CONTRACT_SAFE",
            "SAFE_REQUIRE_RISK_LEVEL",
            "ENTRY_FAIL_CLOSED_ON_SAFETY_GAP",
            "ENTRY_ALLOWED_SAFETY_SOURCES",
            "PAPER_WATCHLIST_STRICT_GUARD_ENABLED",
        }
        metrics = mrt.WindowMetrics(
            scanned=300,
            selected_from_batch=60,
            opened_from_batch=0,
            filter_fail_reasons=Counter({"score_min": 20, "safe_volume": 15, "heavy_dedup_ttl": 12}),
            autotrade_skip_reasons=Counter({"min_trade_size": 10, "ev_net_low": 10, "blacklist": 8}),
            pe_reasons=Counter({"cold_start": 2}),
        )
        overrides = {
            "MARKET_MODE_STRICT_SCORE": "60",
            "MARKET_MODE_SOFT_SCORE": "52",
            "SAFE_MIN_VOLUME_5M_USD": "20",
            "MIN_EXPECTED_EDGE_PERCENT": "0.8",
            "MIN_EXPECTED_EDGE_USD": "0.01",
            "MIN_TRADE_USD": "0.3",
        }
        for mode in mrt.MODE_SPECS.values():
            actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mode)
            self.assertTrue(all(a.key not in locked for a in actions))

    def test_anti_concentration_actions_when_single_symbol_dominates(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=8,
            opened_from_batch=4,
            buy_symbol_counts=Counter({"BOTCOIN": 4, "PEON": 1}),
        )
        overrides = {
            "V2_UNIVERSE_NOVELTY_MIN_SHARE": "0.40",
            "SYMBOL_CONCENTRATION_MAX_SHARE": "0.35",
            "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE": "8",
            "MAX_TOKEN_COOLDOWN_SECONDS": "30",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        # conveyor limits actions per tick; top anti-concentration deltas must appear first.
        self.assertIn("V2_UNIVERSE_NOVELTY_MIN_SHARE", by_key)
        self.assertIn("SYMBOL_CONCENTRATION_MAX_SHARE", by_key)
        self.assertGreater(float(by_key["V2_UNIVERSE_NOVELTY_MIN_SHARE"].new_value), 0.40)
        self.assertLess(float(by_key["SYMBOL_CONCENTRATION_MAX_SHARE"].new_value), 0.35)

    def test_symbol_churn_detector_flags_flat_single_symbol_loop(self) -> None:
        rows = [
            {"decision_stage": "trade_open", "decision": "open", "symbol": "cbBTC"},
            {"decision_stage": "trade_open", "decision": "open", "symbol": "cbBTC"},
            {"decision_stage": "trade_open", "decision": "open", "symbol": "cbBTC"},
            {"decision_stage": "trade_close", "decision": "close", "symbol": "cbBTC", "reason": "NO_MOMENTUM", "pnl_usd": 0.0002},
            {"decision_stage": "trade_close", "decision": "close", "symbol": "cbBTC", "reason": "TIMEOUT", "pnl_usd": -0.0003},
            {"decision_stage": "trade_close", "decision": "close", "symbol": "cbBTC", "reason": "NO_MOMENTUM", "pnl_usd": 0.0001},
        ]
        out = mrt._symbol_churn_15m(rows)
        self.assertTrue(bool(out.get("detected")))
        self.assertEqual(str(out.get("symbol")), "CBBTC")
        self.assertGreater(int(out.get("ttl_seconds", 0) or 0), 0)

    def test_churn_lock_adds_excluded_symbol_and_tightens(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=42,
            opened_from_batch=2,
            buy_symbol_counts=Counter({"cbBTC": 5, "PEON": 1}),
        )
        overrides = {
            "AUTO_TRADE_EXCLUDED_SYMBOLS": "ZORA",
            "V2_UNIVERSE_NOVELTY_MIN_SHARE": "0.20",
            "SYMBOL_CONCENTRATION_MAX_SHARE": "0.35",
            "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE": "4",
            "V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS": "180",
            "TOKEN_EV_MEMORY_MIN_TRADES": "4",
            "TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY": "0.72",
            "TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY": "0.42",
            "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY": "0.55",
            "SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY": "0.35",
        }
        telemetry = {
            "symbol_churn_15m": {
                "detected": True,
                "symbol": "cbBTC",
                "open_share": 0.80,
                "flat_close_share": 0.75,
                "ttl_seconds": 1200,
            },
            "funnel_15m": {"raw": 300, "pre": 180, "buy": 2},
            "top_reasons_15m": {"quality_skip": [], "plan_skip": []},
        }
        state = mrt.RuntimeState()
        actions, _trace, meta = mrt._build_action_plan(
            metrics=metrics,
            overrides=overrides,
            mode=mrt.MODE_SPECS["conveyor"],
            telemetry=telemetry,
            runtime_state=state,
            now_ts=1000.0,
        )
        by_key = _actions_by_key(actions)
        self.assertIn("AUTO_TRADE_EXCLUDED_SYMBOLS", by_key)
        self.assertIn("CBBTC", by_key["AUTO_TRADE_EXCLUDED_SYMBOLS"].new_value)
        self.assertIn("TOKEN_EV_MEMORY_MIN_TRADES", by_key)
        self.assertIn("SOURCE_ROUTER_BAD_ENTRY_PROBABILITY", by_key)
        self.assertEqual(state.churn_lock_symbol, "CBBTC")
        self.assertGreater(float(state.churn_lock_until_ts), 1000.0)
        churn_meta = (meta or {}).get("churn_lock", {}) or {}
        self.assertTrue(bool(churn_meta.get("active", False)))

    def test_churn_lock_releases_on_ttl_expiry(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=30, opened_from_batch=0)
        overrides = {
            "AUTO_TRADE_EXCLUDED_SYMBOLS": "CBBTC,ZORA",
        }
        telemetry = {
            "symbol_churn_15m": {"detected": False, "symbol": "", "open_share": 0.0, "flat_close_share": 0.0, "ttl_seconds": 0},
            "funnel_15m": {"raw": 120, "pre": 70, "buy": 0},
            "top_reasons_15m": {"quality_skip": [], "plan_skip": []},
        }
        state = mrt.RuntimeState(churn_lock_symbol="CBBTC", churn_lock_until_ts=900.0)
        actions, _trace, _meta = mrt._build_action_plan(
            metrics=metrics,
            overrides=overrides,
            mode=mrt.MODE_SPECS["conveyor"],
            telemetry=telemetry,
            runtime_state=state,
            now_ts=1200.0,
        )
        by_key = _actions_by_key(actions)
        self.assertIn("AUTO_TRADE_EXCLUDED_SYMBOLS", by_key)
        self.assertNotIn("CBBTC", by_key["AUTO_TRADE_EXCLUDED_SYMBOLS"].new_value)
        self.assertEqual(state.churn_lock_symbol, "")
        self.assertEqual(float(state.churn_lock_until_ts), 0.0)

    def test_recover_tighten_when_over_relaxed_signals_high(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=40,
            opened_from_batch=6,
            total_closed=12,
            winrate_total=48.0,
            realized_total=-0.03,
            buy_symbol_counts=Counter({"BOTCOIN": 4, "PEON": 2}),
            autotrade_skip_reasons=Counter({"symbol_concentration": 10, "min_trade_size": 9}),
        )
        overrides = {
            "MARKET_MODE_STRICT_SCORE": "45",
            "MARKET_MODE_SOFT_SCORE": "40",
            "SAFE_MIN_VOLUME_5M_USD": "8",
            "MIN_EXPECTED_EDGE_PERCENT": "0.20",
            "MIN_EXPECTED_EDGE_USD": "0.002",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        self.assertIn("MARKET_MODE_STRICT_SCORE", by_key)
        self.assertIn("MARKET_MODE_SOFT_SCORE", by_key)
        self.assertGreater(int(by_key["MARKET_MODE_STRICT_SCORE"].new_value), 45)
        self.assertGreater(int(by_key["MARKET_MODE_SOFT_SCORE"].new_value), 40)

    def test_recover_tighten_blocked_when_open_rate_is_too_low(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=120,
            opened_from_batch=1,
            total_closed=14,
            winrate_total=45.0,
            realized_total=-0.02,
            autotrade_skip_reasons=Counter({"symbol_concentration": 12, "min_trade_size": 16}),
        )
        actions, trace, meta = mrt._build_action_plan(
            metrics=metrics,
            overrides={
                "MARKET_MODE_STRICT_SCORE": "45",
                "MARKET_MODE_SOFT_SCORE": "40",
            },
            mode=mrt.MODE_SPECS["conveyor"],
        )
        by_key = _actions_by_key(actions)
        self.assertNotIn("MARKET_MODE_STRICT_SCORE", by_key)
        self.assertNotIn("MARKET_MODE_SOFT_SCORE", by_key)
        self.assertTrue(any(("recover_tighten_blocked" in t) or ("recover_tighten_not_ready" in t) for t in trace))
        self.assertIn("rule_hits", meta)

    def test_duplicate_choke_relaxes_source_diversity(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=42,
            opened_from_batch=1,
            autotrade_skip_reasons=Counter({"address_or_duplicate": 18, "min_trade_size": 6}),
        )
        overrides = {
            "V2_UNIVERSE_NOVELTY_MIN_SHARE": "0.40",
            "PLAN_MAX_WATCHLIST_SHARE": "0.30",
            "PLAN_MIN_NON_WATCHLIST_PER_BATCH": "1",
            "PLAN_MAX_SINGLE_SOURCE_SHARE": "0.50",
            "V2_SOURCE_QOS_SOURCE_CAPS": "onchain:300,onchain+market:300,dexscreener:260,geckoterminal:260,watchlist:140,dex_boosts:100",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["fast"])
        by_key = _actions_by_key(actions)
        self.assertIn("AUTO_TRADE_TOP_N", by_key)
        self.assertTrue(
            any(
                key in by_key
                for key in (
                    "PLAN_MAX_WATCHLIST_SHARE",
                    "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
                    "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY",
                    "SOURCE_ROUTER_MIN_TRADES",
                )
            )
        )
        self.assertGreater(int(by_key["AUTO_TRADE_TOP_N"].new_value), 40)
        if "PLAN_MAX_WATCHLIST_SHARE" in by_key:
            self.assertLess(float(by_key["PLAN_MAX_WATCHLIST_SHARE"].new_value), 0.30)

    def test_rebalance_source_caps_respects_watchlist_floor(self) -> None:
        caps = {
            "onchain": 320,
            "onchain+market": 320,
            "dexscreener": 280,
            "geckoterminal": 280,
            "watchlist": 1,
            "dex_boosts": 120,
        }
        out = mrt._rebalance_source_caps(
            caps,
            watchlist_scale=0.60,
            min_value=1,
            max_value=600,
            watchlist_min=6,
        )
        self.assertGreaterEqual(int(out.get("watchlist", 0) or 0), 6)

    def test_route_floor_normalizes_watchlist_source_caps(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=20,
            opened_from_batch=0,
        )
        overrides = {
            "V2_SOURCE_QOS_SOURCE_CAPS": "onchain:320,onchain+market:320,dexscreener:280,geckoterminal:280,watchlist:1,dex_boosts:120",
            "V2_UNIVERSE_SOURCE_CAPS": "onchain:220,onchain+market:220,dexscreener:180,geckoterminal:180,watchlist:1,dex_boosts:80",
            "PLAN_MAX_WATCHLIST_SHARE": "0.20",
            "PLAN_MIN_NON_WATCHLIST_PER_BATCH": "2",
            "PLAN_MAX_SINGLE_SOURCE_SHARE": "0.40",
        }
        actions, _trace, _meta = mrt._build_action_plan(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        self.assertIn("V2_SOURCE_QOS_SOURCE_CAPS", by_key)
        self.assertIn("V2_UNIVERSE_SOURCE_CAPS", by_key)
        self.assertIn("watchlist:8", by_key["V2_SOURCE_QOS_SOURCE_CAPS"].new_value)
        self.assertIn("watchlist:4", by_key["V2_UNIVERSE_SOURCE_CAPS"].new_value)

    def test_selected_zero_still_allows_relax_when_filters_exist(self) -> None:
        metrics = mrt.WindowMetrics(
            scanned=200,
            selected_from_batch=0,
            opened_from_batch=0,
            filter_fail_reasons=Counter({"score_min": 20, "safe_volume": 10}),
        )
        overrides = {
            "MARKET_MODE_STRICT_SCORE": "55",
            "MARKET_MODE_SOFT_SCORE": "50",
            "SAFE_MIN_VOLUME_5M_USD": "20",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["fast"])
        by_key = _actions_by_key(actions)
        self.assertIn("MARKET_MODE_STRICT_SCORE", by_key)
        self.assertIn("SAFE_MIN_VOLUME_5M_USD", by_key)
        self.assertLess(int(by_key["MARKET_MODE_STRICT_SCORE"].new_value), 55)
        self.assertLess(float(by_key["SAFE_MIN_VOLUME_5M_USD"].new_value), 20.0)

    def test_route_pressure_rebalances_plan_and_source_caps(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=48,
            opened_from_batch=1,
            autotrade_skip_reasons=Counter({"source_route_prob": 11}),
        )
        overrides = {
            "PLAN_MAX_WATCHLIST_SHARE": "0.30",
            "PLAN_MIN_NON_WATCHLIST_PER_BATCH": "1",
            "PLAN_MAX_SINGLE_SOURCE_SHARE": "0.50",
            "V2_SOURCE_QOS_SOURCE_CAPS": "onchain:300,onchain+market:300,dexscreener:260,geckoterminal:260,watchlist:140,dex_boosts:100",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        self.assertIn("PLAN_MAX_WATCHLIST_SHARE", by_key)
        self.assertIn("PLAN_MIN_NON_WATCHLIST_PER_BATCH", by_key)
        self.assertIn("PLAN_MAX_SINGLE_SOURCE_SHARE", by_key)
        self.assertIn("SOURCE_ROUTER_MIN_TRADES", by_key)
        self.assertIn("SOURCE_ROUTER_BAD_ENTRY_PROBABILITY", by_key)
        self.assertLess(float(by_key["PLAN_MAX_WATCHLIST_SHARE"].new_value), 0.30)
        self.assertGreater(int(by_key["PLAN_MIN_NON_WATCHLIST_PER_BATCH"].new_value), 1)

    def test_source_qos_cap_choke_rebalances_caps_and_topk(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=120,
            opened_from_batch=0,
            filter_fail_reasons=Counter({"source_qos_cap": 44}),
        )
        overrides = {
            "V2_SOURCE_QOS_SOURCE_CAPS": "onchain:120,onchain+market:120,dexscreener:100,geckoterminal:100,watchlist:2,dex_boosts:40",
            "V2_SOURCE_QOS_TOPK_PER_CYCLE": "120",
            "PLAN_MAX_WATCHLIST_SHARE": "0.02",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        self.assertIn("V2_SOURCE_QOS_SOURCE_CAPS", by_key)
        self.assertIn("V2_SOURCE_QOS_TOPK_PER_CYCLE", by_key)
        self.assertIn("PLAN_MAX_WATCHLIST_SHARE", by_key)
        caps_text = by_key["V2_SOURCE_QOS_SOURCE_CAPS"].new_value
        watch_cap = 0
        for part in str(caps_text).split(","):
            name, _, raw = part.partition(":")
            if str(name).strip().lower() == "watchlist":
                try:
                    watch_cap = int(str(raw).strip())
                except Exception:
                    watch_cap = 0
                break
        self.assertGreaterEqual(watch_cap, 8)
        self.assertGreater(int(by_key["V2_SOURCE_QOS_TOPK_PER_CYCLE"].new_value), 120)
        self.assertGreater(float(by_key["PLAN_MAX_WATCHLIST_SHARE"].new_value), 0.02)

    def test_coalesce_actions_keeps_last_value_and_merges_reasons(self) -> None:
        actions = [
            mrt.Action("AUTO_TRADE_TOP_N", "12", "16", "flow_expand"),
            mrt.Action("AUTO_TRADE_TOP_N", "16", "18", "source_qos_cap_rebalance"),
            mrt.Action("MARKET_MODE_STRICT_SCORE", "50", "48", "score_min_dominant"),
            mrt.Action("AUTO_TRADE_TOP_N", "18", "20", "duplicate_choke_topn"),
        ]
        coalesced, collapsed = mrt._coalesce_actions(actions)
        by_key = _actions_by_key(coalesced)
        self.assertEqual(int(collapsed), 2)
        self.assertIn("AUTO_TRADE_TOP_N", by_key)
        self.assertEqual(by_key["AUTO_TRADE_TOP_N"].old_value, "12")
        self.assertEqual(by_key["AUTO_TRADE_TOP_N"].new_value, "20")
        self.assertIn("flow_expand", by_key["AUTO_TRADE_TOP_N"].reason)
        self.assertIn("duplicate_choke_topn", by_key["AUTO_TRADE_TOP_N"].reason)

    def test_blacklist_forensics_exposes_detail_classes(self) -> None:
        trade_rows_15m = [
            {
                "decision_stage": "plan_trade",
                "decision": "skip",
                "reason": "blacklist",
                "detail": "honeypot_guard:is_honeypot",
                "token_address": "0x1",
            },
            {
                "decision_stage": "plan_trade",
                "decision": "skip",
                "reason": "blacklist",
                "detail": "",
                "token_address": "0x2",
            },
        ]
        out = mrt._blacklist_forensics(
            candidate_rows_15m=[],
            trade_rows_15m=trade_rows_15m,
            trade_rows_60m=[],
            metrics=None,
        )
        self.assertEqual(int(out.get("plan_skip_blacklist_honeypot_15m", 0)), 1)
        self.assertEqual(int(out.get("plan_skip_blacklist_unknown_15m", 0)), 1)
        self.assertGreater(float(out.get("plan_skip_blacklist_honeypot_share_15m", 0.0)), 0.0)
        self.assertGreater(float(out.get("plan_skip_blacklist_unknown_share_15m", 0.0)), 0.0)

    def test_edge_deadlock_recovery_relaxes_runtime_edge_floors(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=180,
            opened_from_batch=0,
            autotrade_skip_reasons=Counter(
                {
                    "edge_low": 120,
                    "edge_usd_low": 14,
                    "negative_edge": 12,
                }
            ),
        )
        overrides = {
            "V2_ROLLING_EDGE_MIN_PERCENT": "0.35",
            "V2_ROLLING_EDGE_MIN_USD": "0.008",
            "V2_CALIBRATION_EDGE_USD_MIN": "0.012",
            "V2_CALIBRATION_VOLUME_MIN": "80",
            "V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW": "false",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        self.assertIn("V2_ROLLING_EDGE_MIN_PERCENT", by_key)
        self.assertIn("V2_ROLLING_EDGE_MIN_USD", by_key)
        self.assertIn("V2_CALIBRATION_EDGE_USD_MIN", by_key)
        self.assertIn("V2_CALIBRATION_VOLUME_MIN", by_key)
        self.assertIn("V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW", by_key)
        self.assertLess(float(by_key["V2_ROLLING_EDGE_MIN_PERCENT"].new_value), 0.35)
        self.assertLess(float(by_key["V2_ROLLING_EDGE_MIN_USD"].new_value), 0.008)
        self.assertLess(float(by_key["V2_CALIBRATION_EDGE_USD_MIN"].new_value), 0.012)
        self.assertLess(float(by_key["V2_CALIBRATION_VOLUME_MIN"].new_value), 80.0)

    def test_excluded_symbol_rotation_adjusts_watchlist_controls(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=64,
            opened_from_batch=0,
            autotrade_skip_reasons=Counter({"excluded_symbol": 28}),
        )
        overrides = {
            "PLAN_MAX_WATCHLIST_SHARE": "0.20",
            "PLAN_MIN_NON_WATCHLIST_PER_BATCH": "2",
            "WATCHLIST_REFRESH_SECONDS": "600",
            "WATCHLIST_MAX_TOKENS": "20",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        self.assertIn("PLAN_MAX_WATCHLIST_SHARE", by_key)
        self.assertIn("WATCHLIST_REFRESH_SECONDS", by_key)
        self.assertIn("WATCHLIST_MAX_TOKENS", by_key)
        self.assertLess(float(by_key["PLAN_MAX_WATCHLIST_SHARE"].new_value), 0.20)
        self.assertLess(int(by_key["WATCHLIST_REFRESH_SECONDS"].new_value), 600)
        self.assertGreater(int(by_key["WATCHLIST_MAX_TOKENS"].new_value), 20)

    def test_feed_starvation_relaxes_token_age_and_seen_ttl(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=40,
            opened_from_batch=0,
        )
        telemetry = {
            "funnel_15m": {"raw": 120, "pre": 95, "buy": 0},
            "top_reasons_15m": {"quality_skip": [], "plan_skip": []},
        }
        overrides = {
            "TOKEN_AGE_MAX": "3600",
            "SEEN_TOKEN_TTL": "10800",
            "WATCHLIST_REFRESH_SECONDS": "900",
            "WATCHLIST_MAX_TOKENS": "25",
            "WATCHLIST_MIN_LIQUIDITY_USD": "200000",
            "WATCHLIST_MIN_VOLUME_24H_USD": "500000",
            "PAPER_WATCHLIST_MIN_SCORE": "85",
            "PAPER_WATCHLIST_MIN_LIQUIDITY_USD": "150000",
            "PAPER_WATCHLIST_MIN_VOLUME_5M_USD": "500",
        }
        actions, _trace, _meta = mrt._build_action_plan(
            metrics=metrics,
            overrides=overrides,
            mode=mrt.MODE_SPECS["conveyor"],
            telemetry=telemetry,
        )
        by_key = _actions_by_key(actions)
        self.assertIn("TOKEN_AGE_MAX", by_key)
        self.assertIn("SEEN_TOKEN_TTL", by_key)
        self.assertGreater(int(by_key["TOKEN_AGE_MAX"].new_value), 3600)
        self.assertLess(int(by_key["SEEN_TOKEN_TTL"].new_value), 10800)

    def test_quality_duplicate_telemetry_triggers_duplicate_choke(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=120,
            opened_from_batch=1,
            autotrade_skip_reasons=Counter(),
        )
        telemetry = {
            "funnel_15m": {"raw": 420, "pre": 360, "buy": 0},
            "top_reasons_15m": {
                "quality_skip": [
                    {"reason_code": "duplicate_address", "count": 180},
                    {"reason_code": "source_budget", "count": 40},
                ],
                "plan_skip": [],
            },
        }
        overrides = {
            "AUTO_TRADE_TOP_N": "30",
            "PLAN_MAX_WATCHLIST_SHARE": "0.25",
            "PLAN_MIN_NON_WATCHLIST_PER_BATCH": "1",
            "PLAN_MAX_SINGLE_SOURCE_SHARE": "0.50",
            "V2_SOURCE_QOS_SOURCE_CAPS": "onchain:300,onchain+market:300,dexscreener:260,geckoterminal:260,watchlist:140,dex_boosts:100",
        }
        actions, _trace, _meta = mrt._build_action_plan(
            metrics=metrics,
            overrides=overrides,
            mode=mrt.MODE_SPECS["fast"],
            telemetry=telemetry,
        )
        by_key = _actions_by_key(actions)
        self.assertIn("AUTO_TRADE_TOP_N", by_key)
        self.assertGreater(int(by_key["AUTO_TRADE_TOP_N"].new_value), 30)

    def test_relax_cooldown_uses_plain_cooldown_reason(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=32,
            opened_from_batch=0,
            autotrade_skip_reasons=Counter({"cooldown": 12}),
        )
        overrides = {
            "MAX_TOKEN_COOLDOWN_SECONDS": "120",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        self.assertIn("MAX_TOKEN_COOLDOWN_SECONDS", by_key)
        self.assertLess(int(by_key["MAX_TOKEN_COOLDOWN_SECONDS"].new_value), 120)

    def test_relax_cooldown_respects_mode_floor_when_zero_not_allowed(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=40,
            opened_from_batch=0,
            autotrade_skip_reasons=Counter({"cooldown": 20}),
        )
        overrides = {
            "MAX_TOKEN_COOLDOWN_SECONDS": "10",
        }
        actions = mrt._build_actions(
            metrics=metrics,
            overrides=overrides,
            mode=mrt.MODE_SPECS["conveyor"],
            allow_zero_cooldown=False,
        )
        for action in actions:
            if str(action.key) == "MAX_TOKEN_COOLDOWN_SECONDS":
                self.assertGreaterEqual(int(action.new_value), 10)

    def test_adaptive_bounds_only_drop_cooldown_floor_when_explicitly_allowed(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=40,
            opened_from_batch=0,
        )
        bounds_default = mrt._adaptive_bounds(metrics, mrt.MODE_SPECS["conveyor"], allow_zero_cooldown=False)
        bounds_zero = mrt._adaptive_bounds(metrics, mrt.MODE_SPECS["conveyor"], allow_zero_cooldown=True)
        self.assertEqual(int(bounds_default.cooldown_floor), 10)
        self.assertEqual(int(bounds_zero.cooldown_floor), 0)

    def test_min_trade_relax_updates_a_core_min_trade_floor(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=40,
            opened_from_batch=0,
            autotrade_skip_reasons=Counter({"min_trade_size": 14}),
        )
        overrides = {
            "MIN_TRADE_USD": "0.30",
            "PAPER_TRADE_SIZE_MIN_USD": "0.30",
            "PAPER_TRADE_SIZE_MAX_USD": "0.80",
            "ENTRY_A_CORE_MIN_TRADE_USD": "0.55",
        }
        actions = mrt._build_actions(metrics=metrics, overrides=overrides, mode=mrt.MODE_SPECS["conveyor"])
        by_key = _actions_by_key(actions)
        self.assertIn("ENTRY_A_CORE_MIN_TRADE_USD", by_key)
        self.assertLess(float(by_key["ENTRY_A_CORE_MIN_TRADE_USD"].new_value), 0.55)

    def test_enforce_mutable_action_keys_blocks_sensitive_and_unknown_keys(self) -> None:
        actions = [
            mrt.Action("MARKET_MODE_STRICT_SCORE", "50", "48", "score_min_dominant"),
            mrt.Action("SAFE_REQUIRE_CONTRACT_SAFE", "true", "false", "bad"),
            mrt.Action("UNKNOWN_RUNTIME_KEY", "1", "2", "bad"),
        ]
        allowed, blocked = mrt._enforce_mutable_action_keys(
            actions=actions,
            protected_keys={"SAFE_REQUIRE_CONTRACT_SAFE"},
        )
        by_key = _actions_by_key(allowed)
        self.assertIn("MARKET_MODE_STRICT_SCORE", by_key)
        self.assertNotIn("SAFE_REQUIRE_CONTRACT_SAFE", by_key)
        self.assertEqual(len(blocked), 2)
        blocked_keys = {str(x.get("key")) for x in blocked}
        self.assertIn("SAFE_REQUIRE_CONTRACT_SAFE", blocked_keys)
        self.assertIn("UNKNOWN_RUNTIME_KEY", blocked_keys)

    def test_filter_actions_by_phase_hold_keeps_only_neutral_or_force(self) -> None:
        actions = [
            mrt.Action("MARKET_MODE_STRICT_SCORE", "50", "48", "score_min_dominant"),
            mrt.Action("MARKET_MODE_STRICT_SCORE", "50", "51", "recover_tighten safe_volume"),
            mrt.Action("MARKET_MODE_SOFT_SCORE", "52", "50", "soft_must_be_leq_strict"),
            mrt.Action("PROFIT_ENGINE_ENABLED", "true", "false", "misc"),
        ]
        filtered, blocked = mrt._filter_actions_by_phase(actions=actions, phase="hold")
        by_key = _actions_by_key(filtered)
        self.assertIn("MARKET_MODE_SOFT_SCORE", by_key)
        self.assertIn("PROFIT_ENGINE_ENABLED", by_key)
        self.assertEqual(len(blocked), 2)

    def test_filter_actions_by_phase_tighten_allows_flow_escape_signals(self) -> None:
        actions = [
            mrt.Action(
                "V2_SOURCE_QOS_SOURCE_CAPS",
                "onchain:120,onchain+market:120,dexscreener:100,geckoterminal:100,watchlist:2,dex_boosts:40",
                "onchain:150,onchain+market:150,dexscreener:130,geckoterminal:130,watchlist:8,dex_boosts:55",
                "source_qos_cap_rebalance caps",
            ),
            mrt.Action("MARKET_MODE_STRICT_SCORE", "45", "47", "recover_tighten safe_volume"),
        ]
        filtered, blocked = mrt._filter_actions_by_phase(actions=actions, phase="tighten")
        by_key = _actions_by_key(filtered)
        self.assertIn("V2_SOURCE_QOS_SOURCE_CAPS", by_key)
        self.assertIn("MARKET_MODE_STRICT_SCORE", by_key)
        self.assertEqual(len(blocked), 0)

    def test_apply_action_delta_caps_limits_large_jumps(self) -> None:
        actions = [
            mrt.Action("AUTO_TRADE_TOP_N", "20", "80", "low_throughput_expand_topn"),
            mrt.Action(
                "V2_SOURCE_QOS_SOURCE_CAPS",
                "onchain:300,onchain+market:300,dexscreener:260,geckoterminal:260,watchlist:20,dex_boosts:100",
                "onchain:300,onchain+market:300,dexscreener:260,geckoterminal:260,watchlist:220,dex_boosts:100",
                "route_pressure source_qos_caps",
            ),
        ]
        capped, capped_rows = mrt._apply_action_delta_caps(actions=actions, phase="expand")
        by_key = _actions_by_key(capped)
        self.assertIn("AUTO_TRADE_TOP_N", by_key)
        self.assertLessEqual(int(by_key["AUTO_TRADE_TOP_N"].new_value), 26)
        self.assertIn("V2_SOURCE_QOS_SOURCE_CAPS", by_key)
        self.assertIn("watchlist:80", by_key["V2_SOURCE_QOS_SOURCE_CAPS"].new_value)
        self.assertGreaterEqual(len(capped_rows), 1)

    def test_apply_contract_bounds_clamps_to_safe_limits(self) -> None:
        contract = {
            "allowed_keys": {
                "AUTO_TRADE_TOP_N": {"type": "int", "min": 10, "max": 80},
                "MIN_EXPECTED_EDGE_PERCENT": {"type": "float", "min": 0.05, "max": 1.0},
                "V2_SOURCE_QOS_SOURCE_CAPS": {
                    "type": "source_cap_map",
                    "max_items": 6,
                    "min_value": 1,
                    "max_value": 500,
                },
            }
        }
        actions = [
            mrt.Action("AUTO_TRADE_TOP_N", "80", "90", "low_throughput_expand_topn"),
            mrt.Action("MIN_EXPECTED_EDGE_PERCENT", "0.05", "0.03", "relax_ev_low"),
            mrt.Action(
                "V2_SOURCE_QOS_SOURCE_CAPS",
                "onchain:300,onchain+market:300,dexscreener:260,geckoterminal:260,watchlist:20,dex_boosts:100",
                "onchain:700,onchain+market:700,dexscreener:700,geckoterminal:700,watchlist:0,dex_boosts:700",
                "route_pressure source_qos_caps",
            ),
        ]
        bounded, rows = mrt._apply_contract_bounds(actions=actions, contract=contract)
        by_key = _actions_by_key(bounded)
        self.assertEqual(by_key["AUTO_TRADE_TOP_N"].new_value, "80")
        self.assertEqual(by_key["MIN_EXPECTED_EDGE_PERCENT"].new_value, "0.05")
        self.assertIn("onchain:500", by_key["V2_SOURCE_QOS_SOURCE_CAPS"].new_value)
        self.assertIn("watchlist:1", by_key["V2_SOURCE_QOS_SOURCE_CAPS"].new_value)
        self.assertGreaterEqual(len(rows), 3)

    def test_resolve_policy_phase_tighten_on_blacklist_pressure(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=40, opened_from_batch=0)
        telemetry = {
            "funnel_15m": {"buy": 0},
            "exec_health_15m": {"open_rate": 0.01, "closes": 8, "winrate_closed": 0.4},
            "exit_mix_60m": {"pnl_usd_sum": -0.01},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 25,
                "plan_skip_blacklist_share_15m": 0.8,
                "unique_blacklist_tokens_15m": 8,
            },
        }
        policy = mrt.TargetPolicy(max_blacklist_added_15m=10, max_blacklist_share_15m=0.5)
        state = mrt.RuntimeState(last_phase="expand")
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=policy,
            state=state,
            forced_phase="auto",
        )
        self.assertEqual(decision.phase, "tighten")
        self.assertTrue(decision.blacklist_fail)

    def test_resolve_policy_phase_ignores_single_token_blacklist_storm(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=40, opened_from_batch=0)
        telemetry = {
            "funnel_15m": {"buy": 0},
            "exec_health_15m": {"open_rate": 0.02, "closes": 0, "winrate_closed": 0.0},
            "exit_mix_60m": {"pnl_usd_sum": 0.0},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 40,
                "plan_skip_blacklist_share_15m": 0.9,
                "unique_blacklist_tokens_15m": 1,
            },
        }
        policy = mrt.TargetPolicy(max_blacklist_added_15m=10, max_blacklist_share_15m=0.5)
        state = mrt.RuntimeState(last_phase="expand")
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=policy,
            state=state,
            forced_phase="auto",
        )
        self.assertFalse(decision.blacklist_fail)
        self.assertTrue(any("blacklist_concentrated_ignore" in r for r in decision.reasons))

    def test_resolve_policy_phase_ignores_two_token_blacklist_storm(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=40, opened_from_batch=0)
        telemetry = {
            "funnel_15m": {"buy": 0},
            "exec_health_15m": {"open_rate": 0.02, "closes": 0, "winrate_closed": 0.0},
            "exit_mix_60m": {"pnl_usd_sum": 0.0},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 100,
                "plan_skip_blacklist_share_15m": 0.75,
                "unique_blacklist_tokens_15m": 2,
                "top_blacklist_tokens_15m": [
                    {"token": "0xabc", "count": 58},
                    {"token": "0xdef", "count": 36},
                ],
            },
        }
        policy = mrt.TargetPolicy(max_blacklist_added_15m=40, max_blacklist_share_15m=0.45)
        state = mrt.RuntimeState(last_phase="expand")
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=policy,
            state=state,
            forced_phase="auto",
        )
        self.assertFalse(decision.blacklist_fail)
        self.assertTrue(any("blacklist_concentrated_ignore" in r for r in decision.reasons))

    def test_resolve_policy_phase_ignores_two_token_share_storm_even_below_added_cap(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=22, opened_from_batch=1)
        telemetry = {
            "funnel_15m": {"buy": 1},
            "exec_health_15m": {"open_rate": 0.05, "closes": 2, "winrate_closed": 0.5, "plan_attempts": 40},
            "exit_mix_60m": {"pnl_usd_sum": 0.0},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 23,
                "plan_skip_blacklist_share_15m": 0.575,
                "unique_blacklist_tokens_15m": 2,
                "top_blacklist_tokens_15m": [
                    {"token": "0xabc", "count": 12},
                    {"token": "0xdef", "count": 11},
                ],
            },
        }
        policy = mrt.TargetPolicy(max_blacklist_added_15m=80, max_blacklist_share_15m=0.45)
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=policy,
            state=mrt.RuntimeState(last_phase="tighten"),
            forced_phase="auto",
            effective_target_trades_per_hour=4.0,
        )
        self.assertFalse(decision.blacklist_fail)
        self.assertTrue(any("blacklist_concentrated_ignore" in r for r in decision.reasons))

    def test_resolve_policy_phase_does_not_trigger_blacklist_share_on_micro_sample(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=2, opened_from_batch=1)
        telemetry = {
            "funnel_15m": {"buy": 1},
            "exec_health_15m": {
                "open_rate": 1.0,
                "plan_attempts": 1,
                "closes": 0,
                "winrate_closed": 0.0,
            },
            "exit_mix_60m": {"pnl_usd_sum": 0.0},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 1,
                "plan_skip_blacklist_share_15m": 1.0,
                "unique_blacklist_tokens_15m": 1,
                "top_blacklist_tokens_15m": [{"token": "0xaaa", "count": 1}],
            },
        }
        policy = mrt.TargetPolicy(max_blacklist_added_15m=80, max_blacklist_share_15m=0.45)
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=policy,
            state=mrt.RuntimeState(last_phase="hold"),
            forced_phase="auto",
            effective_target_trades_per_hour=4.0,
        )
        self.assertFalse(decision.blacklist_fail)

    def test_resolve_policy_phase_ignores_unknown_dominated_blacklist_storm(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=40, opened_from_batch=0)
        telemetry = {
            "funnel_15m": {"buy": 0},
            "exec_health_15m": {"open_rate": 0.02, "closes": 0, "winrate_closed": 0.0},
            "exit_mix_60m": {"pnl_usd_sum": 0.0},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 28,
                "plan_skip_blacklist_share_15m": 0.70,
                "unique_blacklist_tokens_15m": 14,
                "plan_skip_blacklist_unknown_share_15m": 0.89,
                "plan_skip_blacklist_honeypot_share_15m": 0.02,
            },
        }
        policy = mrt.TargetPolicy(max_blacklist_added_15m=20, max_blacklist_share_15m=0.5)
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=policy,
            state=mrt.RuntimeState(last_phase="expand"),
            forced_phase="auto",
        )
        self.assertFalse(decision.blacklist_fail)
        self.assertTrue(any("blacklist_unknown_dominated_ignore" in r for r in decision.reasons))

    def test_resolve_policy_phase_keeps_honeypot_dominated_blacklist_fail(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=40, opened_from_batch=0)
        telemetry = {
            "funnel_15m": {"buy": 0},
            "exec_health_15m": {"open_rate": 0.01, "closes": 0, "winrate_closed": 0.0},
            "exit_mix_60m": {"pnl_usd_sum": 0.0},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 24,
                "plan_skip_blacklist_share_15m": 0.72,
                "unique_blacklist_tokens_15m": 12,
                "plan_skip_blacklist_unknown_share_15m": 0.08,
                "plan_skip_blacklist_honeypot_share_15m": 0.83,
            },
        }
        policy = mrt.TargetPolicy(max_blacklist_added_15m=20, max_blacklist_share_15m=0.5)
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=policy,
            state=mrt.RuntimeState(last_phase="expand"),
            forced_phase="auto",
        )
        self.assertTrue(decision.blacklist_fail)
        self.assertEqual(decision.phase, "tighten")

    def test_resolve_policy_phase_tighten_on_pre_risk_before_closes(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=42, opened_from_batch=0)
        telemetry = {
            "funnel_15m": {"buy": 0},
            "exec_health_15m": {
                "open_rate": 0.01,
                "plan_attempts": 22,
                "opens": 1,
                "closes": 0,
                "buy_fail": [{"reason_code": "buy_fail", "count": 9, "share": 0.409}],
                "sell_fail": [],
                "route_fail": [{"reason_code": "no_route", "count": 8, "share": 0.364}],
            },
            "exit_mix_60m": {"pnl_usd_sum": 0.0, "total": 0, "tail_loss_ratio": "N/A"},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 0,
                "plan_skip_blacklist_share_15m": 0.0,
            },
        }
        policy = mrt.TargetPolicy(
            min_selected_15m=16,
            pre_risk_min_plan_attempts_15m=8,
            pre_risk_route_fail_rate_15m=0.30,
            pre_risk_buy_fail_rate_15m=0.30,
        )
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=policy,
            state=mrt.RuntimeState(last_phase="expand"),
            forced_phase="auto",
            effective_target_trades_per_hour=8.0,
        )
        self.assertEqual(decision.phase, "tighten")
        self.assertTrue(bool(decision.pre_risk_fail))
        self.assertTrue(any("pre_risk_fail" in r for r in decision.reasons))

    def test_resolve_policy_phase_tighten_on_tail_loss_ratio(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=30, opened_from_batch=1)
        telemetry = {
            "funnel_15m": {"buy": 1},
            "exec_health_15m": {"open_rate": 0.08, "closes": 8, "winrate_closed": 0.70},
            "exit_mix_60m": {
                "pnl_usd_sum": 0.01,
                "total": 9,
                "largest_loss_usd": -0.09,
                "median_win_usd": 0.009,
                "tail_loss_ratio": 10.0,
            },
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 0,
                "plan_skip_blacklist_share_15m": 0.0,
            },
        }
        policy = mrt.TargetPolicy(
            tail_loss_min_closes_60m=6,
            tail_loss_ratio_max=8.0,
            min_closed_for_risk_checks=6,
            min_winrate_closed_15m=0.35,
        )
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=policy,
            state=mrt.RuntimeState(last_phase="hold"),
            forced_phase="auto",
            effective_target_trades_per_hour=6.0,
        )
        self.assertEqual(decision.phase, "tighten")
        self.assertTrue(bool(decision.risk_fail))
        self.assertTrue(any("tail_loss_fail" in r for r in decision.reasons))

    def test_resolve_policy_phase_unlocks_expand_when_tighten_stalls(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=48,
            opened_from_batch=0,
            open_positions=0,
        )
        telemetry = {
            "funnel_15m": {"buy": 0},
            "exec_health_15m": {"open_rate": 0.02, "closes": 12, "winrate_closed": 0.2},
            "exit_mix_60m": {"pnl_usd_sum": -0.12},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 0,
                "plan_skip_blacklist_share_15m": 0.0,
            },
        }
        policy = mrt.TargetPolicy(
            min_selected_15m=16,
            min_closed_for_risk_checks=6,
            min_winrate_closed_15m=0.35,
            rollback_degrade_streak=3,
        )
        state = mrt.RuntimeState(last_phase="tighten", degrade_streak=4)
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=policy,
            state=state,
            forced_phase="auto",
        )
        self.assertEqual(decision.phase, "expand")
        self.assertTrue(decision.risk_fail)
        self.assertTrue(any("risk_fail_recovery_unlock" in r for r in decision.reasons))

    def test_adaptive_target_bootstrap_caps_unrealistic_goal(self) -> None:
        target = mrt.TargetPolicy(
            target_trades_per_hour=30.0,
            adaptive_target_enabled=True,
            adaptive_target_floor_trades_per_hour=4.0,
            adaptive_target_headroom_mult=1.35,
            adaptive_target_headroom_add_trades_per_hour=2.0,
        )
        state = mrt.RuntimeState()
        effective, meta = mrt._resolve_effective_target_trades_per_hour(
            requested_tph=30.0,
            target=target,
            state=state,
            throughput_est=4.0,
        )
        self.assertGreaterEqual(float(effective), 4.0)
        self.assertLess(float(effective), 30.0)
        self.assertTrue(bool(meta.get("enabled")))

    def test_adaptive_target_state_steps_up_then_down(self) -> None:
        target = mrt.TargetPolicy(
            target_trades_per_hour=20.0,
            adaptive_target_enabled=True,
            adaptive_target_floor_trades_per_hour=4.0,
            adaptive_target_step_up_trades_per_hour=2.0,
            adaptive_target_step_down_trades_per_hour=3.0,
            adaptive_target_stable_ticks_for_step_up=2,
            adaptive_target_fail_ticks_for_step_down=2,
        )
        state = mrt.RuntimeState(effective_target_trades_per_hour=6.0)
        stable_decision = mrt.PolicyDecision(
            phase="hold",
            reasons=[],
            target_trades_per_hour_effective=6.0,
            target_trades_per_hour_requested=20.0,
            throughput_est_trades_h=8.0,
            pnl_hour_usd=0.01,
            blacklist_added_15m=0,
            blacklist_share_15m=0.0,
            open_rate_15m=0.12,
            risk_fail=False,
            flow_fail=False,
            blacklist_fail=False,
        )
        stable_metrics = mrt.WindowMetrics(selected_from_batch=40, opened_from_batch=2)
        mrt._update_effective_target_state(
            state=state,
            target=target,
            decision=stable_decision,
            metrics=stable_metrics,
        )
        post_up = mrt._update_effective_target_state(
            state=state,
            target=target,
            decision=stable_decision,
            metrics=stable_metrics,
        )
        self.assertEqual(str(post_up.get("adjustment")), "step_up")
        self.assertGreater(float(state.effective_target_trades_per_hour), 6.0)

        fail_decision = mrt.PolicyDecision(
            phase="expand",
            reasons=[],
            target_trades_per_hour_effective=float(state.effective_target_trades_per_hour),
            target_trades_per_hour_requested=20.0,
            throughput_est_trades_h=2.0,
            pnl_hour_usd=-0.01,
            blacklist_added_15m=0,
            blacklist_share_15m=0.0,
            open_rate_15m=0.01,
            risk_fail=False,
            flow_fail=True,
            blacklist_fail=False,
        )
        fail_metrics = mrt.WindowMetrics(selected_from_batch=50, opened_from_batch=0)
        mrt._update_effective_target_state(
            state=state,
            target=target,
            decision=fail_decision,
            metrics=fail_metrics,
        )
        post_down = mrt._update_effective_target_state(
            state=state,
            target=target,
            decision=fail_decision,
            metrics=fail_metrics,
        )
        self.assertEqual(str(post_down.get("adjustment")), "step_down")
        self.assertLessEqual(float(state.effective_target_trades_per_hour), float(post_up.get("effective_tph_after", 99.0)))

    def test_resolve_policy_phase_uses_effective_target(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=40, opened_from_batch=1)
        telemetry = {
            "funnel_15m": {"buy": 1},
            "exec_health_15m": {"open_rate": 0.08, "closes": 1, "winrate_closed": 1.0},
            "exit_mix_60m": {"pnl_usd_sum": 0.0},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 0,
                "plan_skip_blacklist_share_15m": 0.0,
            },
        }
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=mrt.TargetPolicy(target_trades_per_hour=30.0),
            state=mrt.RuntimeState(last_phase="hold"),
            forced_phase="auto",
            effective_target_trades_per_hour=6.0,
        )
        self.assertFalse(bool(decision.flow_fail))
        self.assertAlmostEqual(float(decision.target_trades_per_hour_effective), 6.0, places=6)

    def test_resolve_policy_phase_marks_diversity_fail_and_expands(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=28,
            opened_from_batch=4,
            buy_symbol_counts=Counter({"CBBTC": 4}),
        )
        telemetry = {
            "funnel_15m": {"buy": 4},
            "exec_health_15m": {"open_rate": 0.20, "closes": 0, "winrate_closed": 0.0},
            "exit_mix_60m": {"pnl_usd_sum": 0.0},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 0,
                "plan_skip_blacklist_share_15m": 0.0,
            },
        }
        decision = mrt._resolve_policy_phase(
            metrics=metrics,
            telemetry=telemetry,
            target=mrt.TargetPolicy(
                diversity_min_buys_15m=4,
                diversity_min_unique_symbols_15m=2,
                diversity_max_top1_open_share_15m=0.72,
            ),
            state=mrt.RuntimeState(last_phase="hold"),
            forced_phase="auto",
            effective_target_trades_per_hour=8.0,
        )
        self.assertTrue(bool(decision.diversity_fail))
        self.assertEqual(decision.phase, "expand")
        self.assertTrue(any("diversity_fail" in r for r in decision.reasons))

    def test_idle_relax_state_activates_after_min_ticks(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=26,
            opened_from_batch=0,
            open_positions=0,
        )
        target = mrt.TargetPolicy(
            idle_relax_enabled=True,
            idle_relax_min_no_open_ticks=2,
            idle_relax_min_selected_15m=12,
            idle_relax_min_opportunity_per_hour=6.0,
        )
        state = mrt.RuntimeState(idle_no_open_ticks=1, idle_relax_ticks=0)
        decision = mrt.PolicyDecision(
            phase="expand",
            reasons=[],
            target_trades_per_hour_effective=8.0,
            target_trades_per_hour_requested=12.0,
            throughput_est_trades_h=0.0,
            pnl_hour_usd=0.0,
            blacklist_added_15m=0,
            blacklist_share_15m=0.0,
            open_rate_15m=0.0,
            risk_fail=False,
            flow_fail=True,
            blacklist_fail=False,
            pre_risk_fail=False,
            diversity_fail=False,
        )
        snap = mrt._update_idle_relax_state(
            metrics=metrics,
            target=target,
            state=state,
            policy_decision=decision,
            silence_diagnostics={
                "opportunity_rate_per_hour": 10.0,
                "action_rate_per_hour": 0.0,
                "execution_open_rate_per_hour": 0.0,
                "bottleneck_stage": "action",
                "bottleneck_hint": "plan_or_policy",
            },
        )
        self.assertTrue(bool(snap.get("active")))
        self.assertGreaterEqual(int(state.idle_no_open_ticks), 2)
        self.assertGreaterEqual(int(state.idle_relax_ticks), 1)

    def test_idle_relax_guard_allows_only_safe_expand_keys_with_cap(self) -> None:
        actions = [
            mrt.Action("MIN_EXPECTED_EDGE_PERCENT", "0.10", "0.05", "ev_net_low"),
            mrt.Action("MIN_EXPECTED_EDGE_USD", "0.003", "0.002", "ev_net_low"),
            mrt.Action("TOKEN_AGE_MAX", "3600", "4500", "feed_starvation token_age_max"),
            mrt.Action("WATCHLIST_MIN_LIQUIDITY_USD", "200000", "180000", "feed_starvation watch_min_liq"),
            mrt.Action("MARKET_MODE_SOFT_SCORE", "52", "50", "score_min_dominant"),
        ]
        target = mrt.TargetPolicy(idle_relax_max_expand_actions_per_tick=2)
        filtered, blocked = mrt._apply_idle_relax_guard(
            actions=actions,
            idle_relax_snapshot={"active": True},
            target=target,
        )
        kept_keys = [str(a.key) for a in filtered]
        self.assertIn("MIN_EXPECTED_EDGE_PERCENT", kept_keys)
        self.assertIn("MIN_EXPECTED_EDGE_USD", kept_keys)
        self.assertNotIn("TOKEN_AGE_MAX", kept_keys)
        self.assertNotIn("WATCHLIST_MIN_LIQUIDITY_USD", kept_keys)
        self.assertNotIn("MARKET_MODE_SOFT_SCORE", kept_keys)
        self.assertGreaterEqual(len(blocked), 1)

    def test_idle_relax_fast_rollback_requires_tighten_and_relax_history(self) -> None:
        decision = mrt.PolicyDecision(
            phase="tighten",
            reasons=[],
            target_trades_per_hour_effective=8.0,
            target_trades_per_hour_requested=12.0,
            throughput_est_trades_h=0.0,
            pnl_hour_usd=-0.02,
            blacklist_added_15m=0,
            blacklist_share_15m=0.0,
            open_rate_15m=0.0,
            risk_fail=True,
            flow_fail=False,
            blacklist_fail=False,
            pre_risk_fail=False,
            diversity_fail=False,
        )
        target = mrt.TargetPolicy(idle_relax_fast_rollback_enabled=True)
        self.assertTrue(
            mrt._idle_relax_fast_rollback_required(
                target=target,
                decision=decision,
                state=mrt.RuntimeState(idle_relax_ticks=2),
            )
        )
        self.assertFalse(
            mrt._idle_relax_fast_rollback_required(
                target=target,
                decision=decision,
                state=mrt.RuntimeState(idle_relax_ticks=0),
            )
        )

    def test_target_policy_parses_idle_relax_defaults(self) -> None:
        parser = mrt.build_parser()
        args = parser.parse_args(["once", "--profile-id", "u_case", "--dry-run"])
        policy = mrt._target_policy_from_args(args)
        self.assertTrue(bool(policy.idle_relax_enabled))
        self.assertEqual(int(policy.idle_relax_min_no_open_ticks), 3)
        self.assertEqual(int(policy.idle_relax_min_selected_15m), 12)
        self.assertAlmostEqual(float(policy.idle_relax_min_opportunity_per_hour), 6.0, places=6)

    def test_collect_telemetry_v2_builds_funnel_and_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_id = "u_test_profile"
            cand_dir = root / "logs" / "matrix" / profile_id
            cand_dir.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc)

            cand_rows = [
                {
                    "timestamp": (now - timedelta(minutes=5)).isoformat(),
                    "candidate_id": "c1",
                    "decision_stage": "filter_fail",
                    "decision": "skip",
                    "reason": "safe_source",
                    "run_tag": profile_id,
                },
                {
                    "timestamp": (now - timedelta(minutes=4)).isoformat(),
                    "candidate_id": "c2",
                    "decision_stage": "post_filters",
                    "decision": "candidate_pass",
                    "reason": "passed_all_filters",
                    "run_tag": profile_id,
                },
                {
                    "timestamp": (now - timedelta(minutes=3)).isoformat(),
                    "candidate_id": "c2",
                    "decision_stage": "quality_gate",
                    "decision": "skip",
                    "reason": "symbol_concentration",
                    "run_tag": profile_id,
                },
                {
                    "timestamp": (now - timedelta(minutes=2)).isoformat(),
                    "candidate_id": "c3",
                    "decision_stage": "post_filters",
                    "decision": "candidate_pass",
                    "reason": "passed_all_filters",
                    "run_tag": profile_id,
                },
            ]
            trade_rows = [
                {
                    "timestamp": (now - timedelta(minutes=2)).isoformat(),
                    "decision_stage": "plan_trade",
                    "decision": "skip",
                    "reason": "ev_net_low",
                    "candidate_id": "c2",
                    "run_tag": profile_id,
                },
                {
                    "timestamp": (now - timedelta(minutes=1)).isoformat(),
                    "decision_stage": "trade_open",
                    "decision": "open",
                    "reason": "buy_paper",
                    "candidate_id": "c3",
                    "run_tag": profile_id,
                },
                {
                    "timestamp": (now - timedelta(minutes=1)).isoformat(),
                    "decision_stage": "trade_close",
                    "decision": "close",
                    "reason": "TIMEOUT",
                    "candidate_id": "c3",
                    "pnl_percent": 1.2,
                    "pnl_usd": 0.01,
                    "run_tag": profile_id,
                },
            ]
            cand_path = cand_dir / "candidates.jsonl"
            trade_path = cand_dir / "trade_decisions.jsonl"
            cand_path.write_text("\n".join(json.dumps(x) for x in cand_rows) + "\n", encoding="utf-8")
            trade_path.write_text("\n".join(json.dumps(x) for x in trade_rows) + "\n", encoding="utf-8")

            telemetry = mrt._collect_telemetry_v2(
                root=root,
                profile_id=profile_id,
                config_before={"A": "1"},
                config_after={"A": "2"},
                actions=[mrt.Action(key="A", old_value="1", new_value="2", reason="test")],
            )
            funnel = telemetry["funnel_15m"]
            self.assertEqual(int(funnel["raw"]), 3)
            self.assertEqual(int(funnel["buy"]), 1)
            self.assertEqual(int(funnel["trade_close"]), 1)
            top = telemetry["top_reasons_15m"]
            plan_skip = top["plan_skip"]
            self.assertTrue(any(str(x.get("reason_code")) == "ev_net_low" for x in plan_skip))
            exit_rows = telemetry["exit_mix_60m"]["distribution"]
            self.assertTrue(any(str(x.get("reason")) == "TIMEOUT" for x in exit_rows))
            silence = telemetry["silence_diagnostics_15m"]
            self.assertAlmostEqual(float(silence["opportunity_rate_per_hour"]), 8.0, places=6)
            self.assertAlmostEqual(float(silence["action_rate_per_hour"]), 0.0, places=6)
            self.assertAlmostEqual(float(silence["execution_open_rate_per_hour"]), 4.0, places=6)
            self.assertEqual(str(silence["bottleneck_stage"]), "action")
            self.assertEqual(str(silence["bottleneck_hint"]), "plan_or_policy")

    def test_overrides_hash_is_order_independent(self) -> None:
        h1 = mrt._overrides_hash({"B": "2", "A": "1"})
        h2 = mrt._overrides_hash({"A": "1", "B": "2"})
        self.assertEqual(h1, h2)

    def test_diff_override_keys_detects_mismatch(self) -> None:
        diff = mrt._diff_override_keys(
            {"A": "1", "B": "2"},
            {"A": "1", "B": "3", "C": "x"},
        )
        self.assertIn("B", diff)
        self.assertIn("C", diff)
        self.assertNotIn("A", diff)

    def test_build_action_plan_exposes_trace_and_rule_hits(self) -> None:
        metrics = mrt.WindowMetrics(
            selected_from_batch=5,
            opened_from_batch=0,
            filter_fail_reasons=Counter({"score_min": 6}),
        )
        actions, trace, meta = mrt._build_action_plan(
            metrics=metrics,
            overrides={"MARKET_MODE_STRICT_SCORE": "60", "MARKET_MODE_SOFT_SCORE": "52"},
            mode=mrt.MODE_SPECS["conveyor"],
        )
        self.assertTrue(actions)
        self.assertTrue(trace)
        self.assertIn("rule_hits", meta)

    def test_summarize_runtime_rows_counts_actions_and_states(self) -> None:
        rows = [
            {
                "apply_state": "dry_run",
                "actions": [{"key": "A", "reason": "r1"}],
                "decision_meta": {"rule_hits": {"rule_x": 1}},
                "telemetry_v2": {"funnel_15m": {"raw": 10, "buy": 1}, "top_reasons_15m": {"plan_skip": [{"reason_code": "ev_net_low", "count": 3}]}},
            },
            {
                "apply_state": "noop",
                "actions": [],
                "decision_meta": {"rule_hits": {"rule_x": 2}},
                "telemetry_v2": {"funnel_15m": {"raw": 20, "buy": 0}, "top_reasons_15m": {"plan_skip": [{"reason_code": "ev_net_low", "count": 2}]}},
            },
        ]
        summary = mrt._summarize_runtime_rows(rows)
        self.assertEqual(int(summary["ticks"]), 2)
        self.assertEqual(int(summary["action_ticks"]), 1)
        self.assertTrue(any(k == "A" for k, _ in summary["top_action_keys"]))
        self.assertTrue(any(k == "rule_x" for k, _ in summary["rule_hits"]))

    def test_runtime_lock_blocks_second_owner_when_pid_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = "u_lock_test"
            lock_path = mrt._runtime_lock_path(root, profile)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps(
                    {
                        "pid": 77777,
                        "owner": "existing",
                        "profile_id": profile,
                        "started_at": "2026-01-01T00:00:00+00:00",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(mrt, "_pid_is_running", return_value=True):
                ok, msg = mrt._acquire_runtime_lock(root, profile, "new_owner")
            self.assertFalse(ok)
            self.assertIn("lock busy", msg)

    def test_runtime_lock_allows_replace_stale_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = "u_lock_test"
            lock_path = mrt._runtime_lock_path(root, profile)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps({"pid": 77777, "owner": "stale", "profile_id": profile}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(mrt, "_pid_is_running", return_value=False):
                ok, msg = mrt._acquire_runtime_lock(root, profile, "fresh_owner")
            self.assertTrue(ok, msg)
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(str(payload.get("owner")), "fresh_owner")
            self.assertEqual(int(payload.get("pid", 0)), int(mrt.os.getpid()))

    def test_runtime_lock_release_keeps_file_for_alive_foreign_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = "u_lock_test"
            lock_path = mrt._runtime_lock_path(root, profile)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps({"pid": 77777, "owner": "active_foreign", "profile_id": profile}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(mrt, "_pid_is_running", return_value=True):
                mrt._release_runtime_lock(root, profile)
            self.assertTrue(lock_path.exists())

    def test_profile_running_false_when_pid_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "data" / "matrix" / "runs" / "active_matrix.json"
            active.parent.mkdir(parents=True, exist_ok=True)
            active.write_text(
                json.dumps(
                    {
                        "running": True,
                        "items": [
                            {"id": "u1", "status": "running", "pid": 12345},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(mrt, "_pid_is_running", return_value=False):
                self.assertFalse(mrt._profile_running(root, "u1"))

    def test_profile_running_true_when_pid_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "data" / "matrix" / "runs" / "active_matrix.json"
            active.parent.mkdir(parents=True, exist_ok=True)
            active.write_text(
                json.dumps(
                    {
                        "running": True,
                        "items": [
                            {"id": "u1", "status": "running", "pid": 12345},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(mrt, "_pid_is_running", return_value=True):
                self.assertTrue(mrt._profile_running(root, "u1"))

    def test_prune_restart_history_keeps_recent_sorted_values(self) -> None:
        now_ts = 10000.0
        history = [9800.0, 5000.0, 9990.0, 12000.0, 9950.0]
        pruned = mrt._prune_restart_history(history, now_ts=now_ts, window_seconds=3600.0)
        self.assertEqual(pruned, [9800.0, 9950.0, 9990.0])

    def test_restart_gate_blocks_when_budget_exhausted(self) -> None:
        now_ts = 10000.0
        history = [9800.0, 9900.0, 9950.0]
        can_restart, cooldown_left, budget_left = mrt._restart_gate(
            now_ts=now_ts,
            history=history,
            restart_cooldown_seconds=10,
            restart_max_per_hour=3,
        )
        self.assertFalse(can_restart)
        self.assertEqual(int(budget_left), 0)
        self.assertEqual(float(cooldown_left), 0.0)

    def test_restart_gate_blocks_on_cooldown(self) -> None:
        now_ts = 10000.0
        history = [9995.0]
        can_restart, cooldown_left, budget_left = mrt._restart_gate(
            now_ts=now_ts,
            history=history,
            restart_cooldown_seconds=30,
            restart_max_per_hour=6,
        )
        self.assertFalse(can_restart)
        self.assertGreater(float(cooldown_left), 0.0)
        self.assertGreaterEqual(int(budget_left), 1)


if __name__ == "__main__":
    unittest.main()
