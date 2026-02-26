from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import matrix_runtime_tuner as mrt  # noqa: E402


def _actions_by_key(actions: list[mrt.Action]) -> dict[str, mrt.Action]:
    return {a.key: a for a in actions}


class MatrixRuntimeTunerTests(unittest.TestCase):
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
        self.assertIn("watchlist:6", by_key["V2_SOURCE_QOS_SOURCE_CAPS"].new_value)
        self.assertIn("watchlist:6", by_key["V2_UNIVERSE_SOURCE_CAPS"].new_value)

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

    def test_resolve_policy_phase_tighten_on_blacklist_pressure(self) -> None:
        metrics = mrt.WindowMetrics(selected_from_batch=40, opened_from_batch=0)
        telemetry = {
            "funnel_15m": {"buy": 0},
            "exec_health_15m": {"open_rate": 0.01, "closes": 8, "winrate_closed": 0.4},
            "exit_mix_60m": {"pnl_usd_sum": -0.01},
            "blacklist_forensics_15m": {
                "plan_skip_blacklist_15m": 25,
                "plan_skip_blacklist_share_15m": 0.8,
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


if __name__ == "__main__":
    unittest.main()
