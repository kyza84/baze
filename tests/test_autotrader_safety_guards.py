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

    def test_hard_blocked_address_matches_case_insensitive(self) -> None:
        self.patch_cfg(
            AUTO_TRADE_HARD_BLOCKED_ADDRESSES=["0xf30bf00edd0c22db54c9274b90d2a4c21fc09b07"],
            AUTO_TRADE_EXCLUDED_SYMBOLS=[],
        )
        trader = self._blank_trader()
        self.assertTrue(trader._is_hard_blocked_token("0xF30BF00EDD0C22DB54C9274B90D2A4C21FC09B07", symbol="FELIX"))
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


if __name__ == "__main__":
    unittest.main()
