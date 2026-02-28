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

    async def test_refresh_skips_excluded_and_hard_blocked_tokens(self) -> None:
        monitor = WatchlistMonitor()

        async def fake_fetch_dex(**kwargs) -> list[dict]:  # noqa: ARG001
            return [
                {
                    "symbol": "NOOK",
                    "address": "0x1111111111111111111111111111111111111111",
                    "liquidity": 300000.0,
                    "volume_5m": 2500.0,
                    "source": "watchlist",
                },
                {
                    "symbol": "OKX",
                    "address": "0x2222222222222222222222222222222222222222",
                    "liquidity": 350000.0,
                    "volume_5m": 3200.0,
                    "source": "watchlist",
                },
                {
                    "symbol": "HARD",
                    "address": "0x3333333333333333333333333333333333333333",
                    "liquidity": 450000.0,
                    "volume_5m": 5000.0,
                    "source": "watchlist",
                },
            ]

        async def fake_fetch_gecko_page(kind: str, page: int = 1, **kwargs) -> list[dict]:  # noqa: ARG001
            return []

        monitor._fetch_dexscreener_watchlist = fake_fetch_dex  # type: ignore[assignment]
        monitor._fetch_gecko_pool_page = fake_fetch_gecko_page  # type: ignore[assignment]
        with patch.object(config, "AUTO_TRADE_EXCLUDED_SYMBOLS", ["NOOK"]), patch.object(
            config, "AUTO_TRADE_EXCLUDED_SYMBOL_KEYWORDS", []
        ), patch.object(
            config, "AUTO_TRADE_HARD_BLOCKED_ADDRESSES", ["0x3333333333333333333333333333333333333333"]
        ), patch.object(
            config, "WATCHLIST_GECKO_TRENDING_PAGES", 1
        ), patch.object(
            config, "WATCHLIST_GECKO_POOLS_PAGES", 0
        ):
            rows = await monitor._refresh()

        await monitor.close()
        symbols = {str(x.get("symbol", "")).upper() for x in rows}
        self.assertEqual(symbols, {"OKX"})

    async def test_refresh_skips_excluded_when_symbol_list_is_csv_string(self) -> None:
        monitor = WatchlistMonitor()

        async def fake_fetch_dex(**kwargs) -> list[dict]:  # noqa: ARG001
            return [
                {
                    "symbol": "NOOK",
                    "address": "0x1111111111111111111111111111111111111111",
                    "liquidity": 300000.0,
                    "volume_5m": 2500.0,
                    "source": "watchlist",
                },
                {
                    "symbol": "OKX",
                    "address": "0x2222222222222222222222222222222222222222",
                    "liquidity": 350000.0,
                    "volume_5m": 3200.0,
                    "source": "watchlist",
                },
            ]

        async def fake_fetch_gecko_page(kind: str, page: int = 1, **kwargs) -> list[dict]:  # noqa: ARG001
            return []

        monitor._fetch_dexscreener_watchlist = fake_fetch_dex  # type: ignore[assignment]
        monitor._fetch_gecko_pool_page = fake_fetch_gecko_page  # type: ignore[assignment]
        with patch.object(config, "AUTO_TRADE_EXCLUDED_SYMBOLS", "NOOK,FELIX"), patch.object(
            config, "AUTO_TRADE_EXCLUDED_SYMBOL_KEYWORDS", ""
        ), patch.object(
            config, "AUTO_TRADE_HARD_BLOCKED_ADDRESSES", ""
        ), patch.object(
            config, "WATCHLIST_GECKO_TRENDING_PAGES", 1
        ), patch.object(
            config, "WATCHLIST_GECKO_POOLS_PAGES", 0
        ):
            rows = await monitor._refresh()

        await monitor.close()
        symbols = {str(x.get("symbol", "")).upper() for x in rows}
        self.assertEqual(symbols, {"OKX"})


if __name__ == "__main__":
    unittest.main()
