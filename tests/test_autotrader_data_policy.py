from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
