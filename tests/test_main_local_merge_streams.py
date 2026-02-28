from __future__ import annotations

import unittest

import config
import main_local


class MergeTokenStreamsTests(unittest.TestCase):
    def test_watchlist_enriches_but_does_not_override_primary_source(self) -> None:
        address = "0x1111111111111111111111111111111111111111"
        primary = {
            "address": address,
            "source": "onchain+market",
            "liquidity": 12000.0,
            "volume_5m": 150.0,
            "price_usd": 0.01,
        }
        watchlist = {
            "address": address,
            "source": "watchlist",
            "liquidity": 25000.0,
            "volume_5m": 500.0,
            "price_usd": 0.012,
        }

        merged = main_local._merge_token_streams([primary], [watchlist])
        self.assertEqual(len(merged), 1)
        row = merged[0]
        self.assertEqual(str(row.get("source")), "onchain+market")
        self.assertGreater(float(row.get("liquidity") or 0.0), 12000.0)
        self.assertGreater(float(row.get("volume_5m") or 0.0), 150.0)

    def test_non_watchlist_replaces_watchlist_for_same_token(self) -> None:
        address = "0x2222222222222222222222222222222222222222"
        watchlist = {
            "address": address,
            "source": "watchlist",
            "liquidity": 10000.0,
            "volume_5m": 200.0,
            "price_usd": 0.015,
        }
        primary = {
            "address": address,
            "source": "onchain+market",
            "liquidity": 9000.0,
            "volume_5m": 180.0,
            "price_usd": 0.014,
        }

        merged = main_local._merge_token_streams([watchlist], [primary])
        self.assertEqual(len(merged), 1)
        row = merged[0]
        self.assertEqual(str(row.get("source")), "onchain+market")

    def test_runtime_override_coerces_list_values_as_csv(self) -> None:
        old = getattr(config, "AUTO_TRADE_EXCLUDED_SYMBOLS", None)
        try:
            setattr(config, "AUTO_TRADE_EXCLUDED_SYMBOLS", ["NOOK"])
            coerced = main_local._coerce_runtime_override_value(
                key="AUTO_TRADE_EXCLUDED_SYMBOLS",
                raw="NOOK,FELIX,CBBTC",
            )
            self.assertEqual(coerced, ["NOOK", "FELIX", "CBBTC"])
        finally:
            setattr(config, "AUTO_TRADE_EXCLUDED_SYMBOLS", old)

    def test_runtime_tuner_reload_targets_maps_v2_prefixes(self) -> None:
        targets = main_local._runtime_tuner_reload_targets(
            [
                "V2_SOURCE_QOS_SOURCE_CAPS",
                "V2_UNIVERSE_SOURCE_CAPS",
                "V2_ROLLING_EDGE_MIN_USD",
                "V2_KPI_POLICY_MODE",
            ]
        )
        self.assertIn("source_qos", targets)
        self.assertIn("universe", targets)
        self.assertIn("rolling_edge", targets)
        self.assertIn("kpi_loop", targets)

    def test_runtime_tuner_applied_keys_normalizes_state(self) -> None:
        old = main_local._RUNTIME_TUNER_PATCH_STATE.get("applied_keys")
        try:
            main_local._RUNTIME_TUNER_PATCH_STATE["applied_keys"] = ["v2_source_qos_topk_per_cycle", "", None, "  V2_UNIVERSE_SOURCE_CAPS  "]
            keys = main_local._runtime_tuner_applied_keys()
            self.assertIn("V2_SOURCE_QOS_TOPK_PER_CYCLE", keys)
            self.assertIn("V2_UNIVERSE_SOURCE_CAPS", keys)
            self.assertEqual(len(keys), 2)
        finally:
            main_local._RUNTIME_TUNER_PATCH_STATE["applied_keys"] = old


if __name__ == "__main__":
    unittest.main()
