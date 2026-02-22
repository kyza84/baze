from __future__ import annotations

from datetime import datetime, timezone
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


class AutoTraderPolicyTests(ConfigPatchMixin, unittest.TestCase):
    def _blank_trader(self) -> AutoTrader:
        trader = AutoTrader.__new__(AutoTrader)
        trader.data_policy_mode = "OK"
        trader.data_policy_reason = ""
        return trader

    def test_limited_mode_not_blocked_when_router_enabled(self) -> None:
        self.patch_cfg(V2_POLICY_ROUTER_ENABLED=True, DATA_POLICY_HARD_BLOCK_ENABLED=False)
        trader = self._blank_trader()
        trader.data_policy_mode = "LIMITED"
        trader.data_policy_reason = "router_limited"
        self.assertIsNone(trader._policy_block_detail())

    def test_hard_block_rejects_non_ok_modes(self) -> None:
        self.patch_cfg(V2_POLICY_ROUTER_ENABLED=True, DATA_POLICY_HARD_BLOCK_ENABLED=True)
        trader = self._blank_trader()
        trader.data_policy_mode = "LIMITED"
        trader.data_policy_reason = "hard"
        detail = trader._policy_block_detail()
        self.assertIsNotNone(detail)
        self.assertIn("data_policy_hard_block", str(detail))

    def test_legacy_block_when_router_disabled(self) -> None:
        self.patch_cfg(V2_POLICY_ROUTER_ENABLED=False, DATA_POLICY_HARD_BLOCK_ENABLED=False)
        trader = self._blank_trader()
        trader.data_policy_mode = "FAIL_CLOSED"
        trader.data_policy_reason = "legacy"
        detail = trader._policy_block_detail()
        self.assertIsNotNone(detail)
        self.assertIn("data_policy_legacy_block", str(detail))


class AutoTraderCoreProbeTests(ConfigPatchMixin, unittest.TestCase):
    def _blank_trader(self) -> AutoTrader:
        trader = AutoTrader.__new__(AutoTrader)
        trader._core_probe_open_timestamps = []
        return trader

    def test_core_probe_budget_disabled(self) -> None:
        self.patch_cfg(
            EV_FIRST_ENTRY_CORE_PROBE_ENABLED=False,
            EV_FIRST_ENTRY_CORE_PROBE_MAX_OPENS=2,
            EV_FIRST_ENTRY_CORE_PROBE_WINDOW_SECONDS=600,
        )
        trader = self._blank_trader()
        ok, used, cap = trader._core_probe_budget_available()
        self.assertFalse(ok)
        self.assertEqual(used, 0)
        self.assertEqual(cap, 0)

    def test_core_probe_budget_prunes_old_entries(self) -> None:
        self.patch_cfg(
            EV_FIRST_ENTRY_CORE_PROBE_ENABLED=True,
            EV_FIRST_ENTRY_CORE_PROBE_MAX_OPENS=2,
            EV_FIRST_ENTRY_CORE_PROBE_WINDOW_SECONDS=600,
        )
        trader = self._blank_trader()
        now_ts = datetime.now(timezone.utc).timestamp()
        trader._core_probe_open_timestamps = [now_ts - 700, now_ts - 120, now_ts - 30]
        ok, used, cap = trader._core_probe_budget_available()
        self.assertFalse(ok)
        self.assertEqual(cap, 2)
        self.assertEqual(used, 2)
        trader._core_probe_open_timestamps = [now_ts - 700, now_ts - 120]
        ok2, used2, cap2 = trader._core_probe_budget_available()
        self.assertTrue(ok2)
        self.assertEqual(cap2, 2)
        self.assertEqual(used2, 1)


if __name__ == "__main__":
    unittest.main()
