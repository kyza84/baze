from __future__ import annotations

import unittest
from unittest.mock import patch

import config
from monitor.watchlist import WatchlistMonitor


class WatchlistMonitorTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_pair_created_at_uses_fallback_age(self) -> None:
        monitor = WatchlistMonitor()
        fallback_age = 7200

        async def fake_fetch_json(url: str, source: str, retries: int | None = None):  # noqa: ARG001
            return {
                "pairs": [
                    {
                        "chainId": "base",
                        "dexId": "uniswap",
                        "quoteToken": {"address": "0x4200000000000000000000000000000000000006"},
                        "baseToken": {"address": "0x1234567890abcdef1234567890abcdef12345678", "name": "Test", "symbol": "TEST"},
                        "liquidity": {"usd": 250000.0},
                        "volume": {"h24": 900000.0, "m5": 1200.0},
                        "priceChange": {"m5": 1.2},
                        "priceUsd": "0.0012",
                        "url": "https://dex.example/pair",
                    }
                ]
            }

        monitor._fetch_json = fake_fetch_json  # type: ignore[assignment]
        with patch.object(config, "CHAIN_ID", "base"), patch.object(config, "DEX_SEARCH_QUERIES", ["base"]), patch.object(
            config, "WATCHLIST_REQUIRE_WETH_QUOTE", False
        ), patch.object(config, "WATCHLIST_MIN_LIQUIDITY_USD", 1000.0), patch.object(
            config, "WATCHLIST_MIN_VOLUME_24H_USD", 1000.0
        ), patch.object(
            config, "WATCHLIST_DEX_ALLOWLIST", []
        ), patch.object(
            config, "WATCHLIST_MISSING_PAIR_AGE_FALLBACK_SECONDS", fallback_age
        ):
            rows = await monitor._fetch_dexscreener_watchlist()

        await monitor.close()
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(int(rows[0].get("age_seconds") or 0), fallback_age)


if __name__ == "__main__":
    unittest.main()
