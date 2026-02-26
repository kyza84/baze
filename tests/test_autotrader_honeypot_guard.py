from __future__ import annotations

import asyncio
import unittest

import config
from trading.auto_trader import AutoTrader
from utils.http_client import HttpResult


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


class _StubHttp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def get_json(self, *args, **kwargs) -> HttpResult:  # type: ignore[no-untyped-def]
        return HttpResult(ok=True, status=200, data=self._payload, error="")


class AutoTraderHoneypotGuardTests(ConfigPatchMixin, unittest.TestCase):
    @staticmethod
    def _blank_trader(payload: dict) -> AutoTrader:
        trader = AutoTrader.__new__(AutoTrader)
        trader._honeypot_cache = {}
        trader._http = _StubHttp(payload)
        return trader

    def test_blocks_when_simulation_failed_and_high_fail_rate(self) -> None:
        self.patch_cfg(
            HONEYPOT_API_ENABLED=True,
            HONEYPOT_API_URL="https://api.honeypot.is/v2/IsHoneypot",
            HONEYPOT_API_FAIL_CLOSED=True,
            HONEYPOT_MAX_BUY_TAX_PERCENT=10.0,
            HONEYPOT_MAX_SELL_TAX_PERCENT=10.0,
        )
        payload = {
            "simulationSuccess": False,
            "simulationError": "execution reverted: HP: BUY_FAILED",
            "summary": {
                "flags": [
                    {
                        "flag": "high_fail_rate",
                        "severity": "critical",
                    }
                ]
            },
        }
        trader = self._blank_trader(payload)
        ok, detail = asyncio.run(trader._honeypot_guard_passes("0x1111111111111111111111111111111111111111"))
        self.assertFalse(ok)
        self.assertIn("simulation_failed", detail)
        self.assertIn("flag:high_fail_rate", detail)

    def test_blocks_when_can_sell_is_false_string(self) -> None:
        self.patch_cfg(
            HONEYPOT_API_ENABLED=True,
            HONEYPOT_API_URL="https://api.honeypot.is/v2/IsHoneypot",
            HONEYPOT_API_FAIL_CLOSED=True,
            HONEYPOT_MAX_BUY_TAX_PERCENT=10.0,
            HONEYPOT_MAX_SELL_TAX_PERCENT=10.0,
        )
        payload = {
            "simulationResult": {
                "canSell": "false",
                "buyTax": "1.2",
                "sellTax": "1.1",
            }
        }
        trader = self._blank_trader(payload)
        ok, detail = asyncio.run(trader._honeypot_guard_passes("0x2222222222222222222222222222222222222222"))
        self.assertFalse(ok)
        self.assertIn("cannot_sell", detail)

    def test_passes_on_low_taxes_and_sellable(self) -> None:
        self.patch_cfg(
            HONEYPOT_API_ENABLED=True,
            HONEYPOT_API_URL="https://api.honeypot.is/v2/IsHoneypot",
            HONEYPOT_API_FAIL_CLOSED=True,
            HONEYPOT_MAX_BUY_TAX_PERCENT=10.0,
            HONEYPOT_MAX_SELL_TAX_PERCENT=10.0,
        )
        payload = {
            "honeypotResult": {"isHoneypot": False},
            "simulationResult": {
                "canSell": True,
                "buyTax": 2.0,
                "sellTax": 2.0,
            },
        }
        trader = self._blank_trader(payload)
        ok, detail = asyncio.run(trader._honeypot_guard_passes("0x3333333333333333333333333333333333333333"))
        self.assertTrue(ok)
        self.assertIn("ok buyTax=", detail)


if __name__ == "__main__":
    unittest.main()
