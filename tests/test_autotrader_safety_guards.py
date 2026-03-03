from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import tempfile
import unittest

import config
from trading.auto_trader import AutoTrader


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


class AutoTraderSafetyGuardTests(ConfigPatchMixin, unittest.TestCase):
    @staticmethod
    def _blank_trader() -> AutoTrader:
        trader = AutoTrader.__new__(AutoTrader)
        trader._blacklist = {}
        trader.token_cooldowns = {}
        trader._recent_open_symbols = []
        trader._recent_open_sources = []
        return trader

    def test_transient_safety_source_is_blocked_when_fail_closed(self) -> None:
        self.patch_cfg(
            ENTRY_FAIL_CLOSED_ON_SAFETY_GAP=True,
            ENTRY_ALLOW_TRANSIENT_SAFETY_SOURCE=False,
            ENTRY_ALLOWED_SAFETY_SOURCES=["goplus", "cache_transient", "transient_fallback", "fallback"],
            SAFE_TEST_MODE=False,
            SAFE_MIN_HOLDERS=0,
            ENTRY_BLOCK_RISKY_CONTRACT_FLAGS=True,
        )
        trader = self._blank_trader()
        ok, reason = trader._token_guard_result(
            {
                "price_change_5m": 0.0,
                "safety": {"source": "cache_transient", "fail_reason": ""},
                "risk_level": "LOW",
            }
        )
        self.assertFalse(ok)
        self.assertIn("safety_source_transient", reason)

    def test_transient_fail_reason_is_allowed_if_transient_source_is_allowed(self) -> None:
        self.patch_cfg(
            ENTRY_FAIL_CLOSED_ON_SAFETY_GAP=True,
            ENTRY_ALLOW_TRANSIENT_SAFETY_SOURCE=True,
            ENTRY_ALLOWED_SAFETY_SOURCES=["goplus", "cache_transient", "transient_fallback", "fallback"],
            SAFE_TEST_MODE=False,
            SAFE_MIN_HOLDERS=0,
        )
        trader = self._blank_trader()
        ok, reason = trader._token_guard_result(
            {
                "price_change_5m": 0.0,
                "safety": {"source": "cache_transient", "fail_reason": "rpc_timeout"},
                "risk_level": "LOW",
            }
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_non_watch_soft_age_pass_is_respected_by_safety_guards(self) -> None:
        self.patch_cfg(
            ENTRY_FAIL_CLOSED_ON_SAFETY_GAP=False,
            SAFE_TEST_MODE=True,
            SAFE_MIN_LIQUIDITY_USD=5000.0,
            SAFE_MIN_VOLUME_5M_USD=100.0,
            SAFE_MIN_AGE_SECONDS=240,
            SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT=18.0,
            SAFE_REQUIRE_CONTRACT_SAFE=False,
            SAFE_MAX_WARNING_FLAGS=10,
            SAFE_MIN_HOLDERS=0,
            SAFE_AGE_NON_WATCH_SOFT_ENABLED=True,
        )
        trader = self._blank_trader()
        ok, reason = trader._token_guard_result(
            {
                "source": "onchain",
                "liquidity": 25000.0,
                "volume_5m": 350.0,
                "age_seconds": 120,
                "price_change_5m": 1.2,
                "_safe_age_soft_pass": True,
                "risk_level": "LOW",
            }
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_watchlist_does_not_get_non_watch_soft_age_bypass(self) -> None:
        self.patch_cfg(
            ENTRY_FAIL_CLOSED_ON_SAFETY_GAP=False,
            SAFE_TEST_MODE=True,
            SAFE_MIN_LIQUIDITY_USD=5000.0,
            SAFE_MIN_VOLUME_5M_USD=100.0,
            SAFE_MIN_AGE_SECONDS=240,
            SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT=18.0,
            SAFE_REQUIRE_CONTRACT_SAFE=False,
            SAFE_MAX_WARNING_FLAGS=10,
            SAFE_MIN_HOLDERS=0,
            SAFE_AGE_NON_WATCH_SOFT_ENABLED=True,
        )
        trader = self._blank_trader()
        ok, reason = trader._token_guard_result(
            {
                "source": "watchlist",
                "liquidity": 25000.0,
                "volume_5m": 350.0,
                "age_seconds": 120,
                "price_change_5m": 1.2,
                "_safe_age_soft_pass": True,
                "risk_level": "LOW",
            }
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "safe_min_age")

    def test_non_watch_soft_volume_and_change_pass_are_respected(self) -> None:
        self.patch_cfg(
            ENTRY_FAIL_CLOSED_ON_SAFETY_GAP=False,
            SAFE_TEST_MODE=True,
            SAFE_MIN_LIQUIDITY_USD=5000.0,
            SAFE_MIN_VOLUME_5M_USD=100.0,
            SAFE_MIN_AGE_SECONDS=240,
            SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT=18.0,
            SAFE_REQUIRE_CONTRACT_SAFE=False,
            SAFE_MAX_WARNING_FLAGS=10,
            SAFE_MIN_HOLDERS=0,
            SAFE_VOLUME_TWO_TIER_NON_WATCH_ENABLED=True,
            SAFE_CHANGE_5M_NON_WATCH_SOFT_ENABLED=True,
        )
        trader = self._blank_trader()
        ok, reason = trader._token_guard_result(
            {
                "source": "geckoterminal",
                "liquidity": 25000.0,
                "volume_5m": 10.0,
                "age_seconds": 360,
                "price_change_5m": 32.0,
                "_safe_volume_soft_pass": True,
                "_safe_change_soft_pass": True,
                "risk_level": "LOW",
            }
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_hard_blocked_address_matches_case_insensitive(self) -> None:
        self.patch_cfg(
            AUTO_TRADE_HARD_BLOCKED_ADDRESSES=["0xf30bf00edd0c22db54c9274b90d2a4c21fc09b07"],
            AUTO_TRADE_EXCLUDED_SYMBOLS=[],
        )
        trader = self._blank_trader()
        self.assertTrue(trader._is_hard_blocked_token("0xF30BF00EDD0C22DB54C9274B90D2A4C21FC09B07", symbol="FELIX"))
        self.assertFalse(trader._is_hard_blocked_token("0x1111111111111111111111111111111111111111", symbol="OTHER"))

    def test_hard_blocked_symbol_works_when_config_is_csv_string(self) -> None:
        self.patch_cfg(
            AUTO_TRADE_HARD_BLOCKED_ADDRESSES=[],
            AUTO_TRADE_EXCLUDED_SYMBOLS="NOOK,FELIX",
        )
        trader = self._blank_trader()
        self.assertTrue(trader._is_hard_blocked_token("0x1111111111111111111111111111111111111111", symbol="NOOK"))
        self.assertFalse(trader._is_hard_blocked_token("0x1111111111111111111111111111111111111111", symbol="OTHER"))

    def test_blacklist_lookup_accepts_legacy_address_key(self) -> None:
        self.patch_cfg(AUTOTRADE_BLACKLIST_ENABLED=True)
        trader = self._blank_trader()
        until = datetime.now(timezone.utc).timestamp() + 3600
        legacy = "0xf30bf00edd0c22db54c9274b90d2a4c21fc09b07"
        trader._blacklist = {legacy: {"reason": "legacy", "until_ts": until}}
        blocked, reason = trader._blacklist_is_blocked("0xF30BF00EDD0C22DB54C9274B90D2A4C21FC09B07")
        self.assertTrue(blocked)
        self.assertEqual(reason, "legacy")

    def test_blacklist_ignores_placeholder_zero_address(self) -> None:
        trader = self._blank_trader()
        zero = "0x0000000000000000000000000000000000000000"
        self.assertEqual(trader._blacklist_key(zero), "")
        self.assertEqual(trader._blacklist_lookup_keys(zero), [])

    def test_transient_safety_blacklist_ttl_uses_short_tuned_value(self) -> None:
        self.patch_cfg(
            AUTOTRADE_BLACKLIST_TTL_SECONDS=10800,
            AUTOTRADE_BLACKLIST_TRANSIENT_SAFETY_TTL_SECONDS=900,
        )
        ttl = AutoTrader._blacklist_default_ttl_seconds("safety_guard:safety_source_transient:transient_fallback")
        self.assertEqual(ttl, 900)

    def test_load_blacklist_prunes_transient_safety_entries_when_disabled(self) -> None:
        self.patch_cfg(
            AUTOTRADE_BLACKLIST_ENABLED=True,
            ENTRY_TRANSIENT_SAFETY_TO_BLACKLIST=False,
            AUTOTRADE_BLACKLIST_MAX_ENTRIES=5000,
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "blacklist.json")
            now = datetime.now(timezone.utc).timestamp()
            payload = {
                "0x1111111111111111111111111111111111111111": {
                    "reason": "safety_guard:safety_source_transient:transient_fallback",
                    "added_ts": now,
                    "until_ts": now + 3600,
                },
                "0x2222222222222222222222222222222222222222": {
                    "reason": "hard_blocklist:config",
                    "added_ts": now,
                    "until_ts": now + 3600,
                },
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            trader = self._blank_trader()
            trader._blacklist_file = path
            trader._load_blacklist()
            self.assertEqual(len(trader._blacklist), 1)
            only_reason = next(iter(trader._blacklist.values())).get("reason")
            self.assertEqual(only_reason, "hard_blocklist:config")

    def test_load_blacklist_prunes_transient_honeypot_and_zero_address_rows(self) -> None:
        self.patch_cfg(
            AUTOTRADE_BLACKLIST_ENABLED=True,
            HONEYPOT_TRANSIENT_TO_BLACKLIST=False,
            AUTOTRADE_BLACKLIST_MAX_ENTRIES=5000,
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "blacklist.json")
            now = datetime.now(timezone.utc).timestamp()
            payload = {
                "0x1111111111111111111111111111111111111111": {
                    "reason": "honeypot_guard:honeypot_api_status_404",
                    "added_ts": now,
                    "until_ts": now + 3600,
                },
                "0x0000000000000000000000000000000000000000": {
                    "reason": "honeypot_guard:is_honeypot",
                    "added_ts": now,
                    "until_ts": now + 3600,
                },
                "0x2222222222222222222222222222222222222222": {
                    "reason": "honeypot_guard:is_honeypot",
                    "added_ts": now,
                    "until_ts": now + 3600,
                },
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            trader = self._blank_trader()
            trader._blacklist_file = path
            trader._load_blacklist()
            self.assertEqual(len(trader._blacklist), 1)
            only_reason = next(iter(trader._blacklist.values())).get("reason")
            self.assertEqual(only_reason, "honeypot_guard:is_honeypot")

    def test_blacklist_soft_reason_does_not_block_in_paper_hard_only_mode(self) -> None:
        self.patch_cfg(
            AUTOTRADE_BLACKLIST_ENABLED=True,
            AUTOTRADE_BLACKLIST_PAPER_HARD_ONLY=True,
            AUTO_TRADE_ENABLED=True,
            AUTO_TRADE_PAPER=True,
        )
        trader = self._blank_trader()
        until = datetime.now(timezone.utc).timestamp() + 3600
        legacy = "0xf30bf00edd0c22db54c9274b90d2a4c21fc09b07"
        trader._blacklist = {legacy: {"reason": "unsupported_buy_route:quote_failed", "until_ts": until}}
        blocked, _ = trader._blacklist_is_blocked("0xF30BF00EDD0C22DB54C9274B90D2A4C21FC09B07")
        self.assertFalse(blocked)

    def test_blacklist_hard_reason_blocks_in_paper_hard_only_mode(self) -> None:
        self.patch_cfg(
            AUTOTRADE_BLACKLIST_ENABLED=True,
            AUTOTRADE_BLACKLIST_PAPER_HARD_ONLY=True,
            AUTO_TRADE_ENABLED=True,
            AUTO_TRADE_PAPER=True,
        )
        trader = self._blank_trader()
        until = datetime.now(timezone.utc).timestamp() + 3600
        legacy = "0xf30bf00edd0c22db54c9274b90d2a4c21fc09b07"
        trader._blacklist = {legacy: {"reason": "honeypot_guard:is_honeypot", "until_ts": until}}
        blocked, reason = trader._blacklist_is_blocked("0xF30BF00EDD0C22DB54C9274B90D2A4C21FC09B07")
        self.assertTrue(blocked)
        self.assertIn("honeypot_guard", reason)

    def test_blacklist_transient_honeypot_reason_does_not_block_in_paper_hard_only_mode(self) -> None:
        self.patch_cfg(
            AUTOTRADE_BLACKLIST_ENABLED=True,
            AUTOTRADE_BLACKLIST_PAPER_HARD_ONLY=True,
            AUTO_TRADE_ENABLED=True,
            AUTO_TRADE_PAPER=True,
        )
        trader = self._blank_trader()
        until = datetime.now(timezone.utc).timestamp() + 3600
        legacy = "0xf30bf00edd0c22db54c9274b90d2a4c21fc09b07"
        trader._blacklist = {legacy: {"reason": "honeypot_guard:honeypot_api_status_400", "until_ts": until}}
        blocked, _ = trader._blacklist_is_blocked("0xF30BF00EDD0C22DB54C9274B90D2A4C21FC09B07")
        self.assertFalse(blocked)

    def test_transient_honeypot_blacklist_ttl_uses_unknown_ttl(self) -> None:
        self.patch_cfg(
            AUTOTRADE_BLACKLIST_TTL_SECONDS=10800,
            AUTOTRADE_BLACKLIST_UNKNOWN_TTL_SECONDS=900,
        )
        ttl = AutoTrader._blacklist_default_ttl_seconds("honeypot_guard:honeypot_api_status_400")
        self.assertEqual(ttl, 900)

    def test_max_loss_cap_uses_non_gas_cost_component(self) -> None:
        self.patch_cfg(MAX_LOSS_PER_TRADE_PERCENT_BALANCE=1.0)
        trader = self._blank_trader()
        trader._equity_usd = lambda: 7.0  # type: ignore[method-assign]

        capped = trader._apply_max_loss_per_trade_cap(
            position_size_usd=1.0,
            stop_loss_percent=4,
            total_cost_percent=4.98,  # includes gas @ size=1.0
            gas_usd=0.0276,
        )
        legacy_double_count = max(0.0, (0.07 - 0.0276) / ((4.0 + 4.98) / 100.0))
        expected_without_double_count = max(0.0, (0.07 - 0.0276) / ((4.0 + 2.22) / 100.0))

        self.assertGreater(capped, legacy_double_count)
        self.assertAlmostEqual(capped, expected_without_double_count, places=6)

    def test_cost_dominant_non_watch_explore_fast_guard_sets_symbol_cooldown(self) -> None:
        self.patch_cfg(
            EDGE_COST_DOMINANT_GUARD_ENABLED=True,
            EDGE_COST_DOMINANT_NON_WATCH_EXPLORE_FAST_GUARD_ENABLED=True,
            EDGE_COST_DOMINANT_NON_WATCH_EXPLORE_MAX_SIZE_USD=0.30,
            EDGE_COST_DOMINANT_NON_WATCH_EXPLORE_MIN_GROSS_PERCENT=0.5,
            EDGE_COST_DOMINANT_NON_WATCH_EXPLORE_MIN_COST_PERCENT=10.0,
            EDGE_COST_DOMINANT_NON_WATCH_EXPLORE_MIN_DELTA_PERCENT=6.0,
            EDGE_COST_DOMINANT_NON_WATCH_EXPLORE_HIT_WINDOW_SECONDS=900,
            EDGE_COST_DOMINANT_NON_WATCH_EXPLORE_HITS_TO_COOLDOWN=2,
            EDGE_COST_DOMINANT_NON_WATCH_EXPLORE_SYMBOL_COOLDOWN_SECONDS=600,
        )
        trader = self._blank_trader()
        trader._edge_cost_dominant_hits_by_symbol = {}
        trader.symbol_cooldowns = {}
        now_ts = datetime.now(timezone.utc).timestamp()

        trader._cost_dominant_skip_guard(
            symbol="SERV",
            skip_reason="edge_low",
            gross_percent=1.2,
            cost_total_percent=22.0,
            source_name="onchain+market",
            entry_channel="explore",
            position_size_usd=0.10,
        )
        trader._cost_dominant_skip_guard(
            symbol="SERV",
            skip_reason="edge_low",
            gross_percent=1.1,
            cost_total_percent=21.5,
            source_name="onchain+market",
            entry_channel="explore",
            position_size_usd=0.10,
        )
        until = float(trader.symbol_cooldowns.get("SERV", 0.0) or 0.0)
        self.assertGreater(until, now_ts)

    def test_anti_choke_dominant_symbol_detection(self) -> None:
        self.patch_cfg(
            ANTI_CHOKE_BYPASS_SYMBOL_MIN_OPENS=4,
            ANTI_CHOKE_BYPASS_SYMBOL_SHARE_LIMIT=0.70,
        )
        trader = self._blank_trader()
        now_ts = datetime.now(timezone.utc).timestamp()
        trader._recent_open_symbols = [
            (now_ts - 10, "CBBTC"),
            (now_ts - 20, "CBBTC"),
            (now_ts - 30, "CBBTC"),
            (now_ts - 40, "CBBTC"),
            (now_ts - 50, "AERO"),
        ]
        trader._anti_choke_active = lambda: True  # type: ignore[method-assign]
        blocked, detail = trader._anti_choke_symbol_dominant(symbol="cbBTC", window_seconds=3600)
        self.assertTrue(blocked)
        self.assertIn("anti_choke_dominant_symbol", detail)

    def test_symbol_concentration_guard_still_blocks_dominant_symbol_in_anti_choke(self) -> None:
        self.patch_cfg(
            SYMBOL_CONCENTRATION_GUARD_ENABLED=True,
            SYMBOL_CONCENTRATION_WINDOW_SECONDS=3600,
            SYMBOL_CONCENTRATION_MIN_OPENS=4,
            SYMBOL_CONCENTRATION_MAX_SHARE=0.12,
            SYMBOL_CONCENTRATION_TIER_A_SHARE_MULT=1.0,
            ANTI_CHOKE_ALLOW_SYMBOL_CONCENTRATION_BYPASS=True,
            ANTI_CHOKE_BYPASS_SYMBOL_MIN_OPENS=4,
            ANTI_CHOKE_BYPASS_SYMBOL_SHARE_LIMIT=0.70,
            ANTI_CHOKE_CONCENTRATION_MIN_SHARE_CAP=0.35,
        )
        trader = self._blank_trader()
        now_ts = datetime.now(timezone.utc).timestamp()
        trader._recent_open_symbols = [
            (now_ts - 10, "CBBTC"),
            (now_ts - 20, "CBBTC"),
            (now_ts - 30, "CBBTC"),
            (now_ts - 40, "CBBTC"),
            (now_ts - 50, "CBBTC"),
        ]
        trader._anti_choke_active = lambda: True  # type: ignore[method-assign]
        blocked, detail = trader._symbol_concentration_blocked(symbol="cbBTC", entry_tier="A")
        self.assertTrue(blocked)
        self.assertIn("symbol_concentration", detail)

    def test_hard_top1_open_share_15m_blocks_dominant_symbol(self) -> None:
        self.patch_cfg(
            TOP1_OPEN_SHARE_15M_GUARD_ENABLED=True,
            TOP1_OPEN_SHARE_15M_WINDOW_SECONDS=900,
            TOP1_OPEN_SHARE_15M_MIN_OPENS=4,
            TOP1_OPEN_SHARE_15M_MAX_SHARE=0.70,
        )
        trader = self._blank_trader()
        now_ts = datetime.now(timezone.utc).timestamp()
        trader._recent_open_symbols = [
            (now_ts - 10, "CBBTC"),
            (now_ts - 20, "CBBTC"),
            (now_ts - 30, "CBBTC"),
            (now_ts - 40, "AERO"),
        ]
        blocked, detail = trader._hard_top1_open_share_blocked(symbol="cbBTC")
        self.assertTrue(blocked)
        self.assertIn("top1_open_share_15m", detail)

    def test_plan_non_watchlist_quota_blocks_watchlist_when_batch_has_alternative(self) -> None:
        self.patch_cfg(
            PLAN_NON_WATCHLIST_QUOTA_ENABLED=True,
            PLAN_NON_WATCHLIST_QUOTA_WINDOW_SECONDS=900,
            PLAN_NON_WATCHLIST_QUOTA_MIN_OPENS=4,
            PLAN_MIN_NON_WATCHLIST_PER_BATCH=2,
            PLAN_MAX_WATCHLIST_SHARE=0.30,
        )
        trader = self._blank_trader()
        now_ts = datetime.now(timezone.utc).timestamp()
        trader._recent_open_sources = [
            (now_ts - 10, "watchlist"),
            (now_ts - 20, "watchlist"),
            (now_ts - 30, "watchlist"),
            (now_ts - 40, "watchlist"),
        ]
        blocked, detail = trader._plan_non_watchlist_quota_blocked(
            source_name="watchlist",
            batch_has_non_watchlist=True,
        )
        self.assertTrue(blocked)
        self.assertIn("plan_non_watchlist_quota", detail)

    def test_plan_non_watchlist_quota_blocks_watchlist_on_empty_batch_when_enforced(self) -> None:
        self.patch_cfg(
            PLAN_NON_WATCHLIST_QUOTA_ENABLED=True,
            PLAN_NON_WATCHLIST_QUOTA_ENFORCE_EMPTY_BATCH=True,
            PLAN_NON_WATCHLIST_QUOTA_EMPTY_BATCH_MIN_OPENS=4,
            PLAN_NON_WATCHLIST_QUOTA_WINDOW_SECONDS=900,
            PLAN_NON_WATCHLIST_QUOTA_MIN_OPENS=2,
            PLAN_MIN_NON_WATCHLIST_PER_BATCH=2,
            PLAN_MAX_WATCHLIST_SHARE=0.30,
        )
        trader = self._blank_trader()
        now_ts = datetime.now(timezone.utc).timestamp()
        trader._recent_open_sources = [
            (now_ts - 10, "watchlist"),
            (now_ts - 20, "watchlist"),
            (now_ts - 30, "watchlist"),
            (now_ts - 40, "watchlist"),
        ]
        blocked, detail = trader._plan_non_watchlist_quota_blocked(
            source_name="watchlist",
            batch_has_non_watchlist=False,
        )
        self.assertTrue(blocked)
        self.assertIn("enforce_empty_batch=True", detail)

    def test_rebalance_respects_watchlist_cap_when_bypass_disabled(self) -> None:
        self.patch_cfg(
            PLAN_SOURCE_DIVERSITY_ENABLED=True,
            PLAN_MAX_SINGLE_SOURCE_SHARE=1.0,
            PLAN_MAX_WATCHLIST_SHARE=0.02,
            PLAN_MIN_NON_WATCHLIST_PER_BATCH=0,
            PLAN_SOURCE_DIVERSITY_ALLOW_WATCHLIST_CAP_BYPASS=False,
        )
        trader = self._blank_trader()
        selected = [
            ({"address": "0x1000000000000000000000000000000000000001", "symbol": "A1", "source": "watchlist"}, {}),
            ({"address": "0x1000000000000000000000000000000000000002", "symbol": "A2", "source": "watchlist"}, {}),
            ({"address": "0x1000000000000000000000000000000000000003", "symbol": "A3", "source": "watchlist"}, {}),
        ]
        out = trader._rebalance_plan_batch_sources(selected=selected, eligible=list(selected))
        watch_count = sum(1 for token, _ in out if str((token or {}).get("source", "")).startswith("watchlist"))
        self.assertLessEqual(int(watch_count), 1)


if __name__ == "__main__":
    unittest.main()
