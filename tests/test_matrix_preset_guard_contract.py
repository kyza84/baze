from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from matrix_preset_guard import load_contract, validate_overrides  # noqa: E402


class MatrixPresetGuardContractTests(unittest.TestCase):
    def test_universe_source_caps_is_allowed_in_safe_contract(self) -> None:
        contract = load_contract(ROOT)
        issues = validate_overrides(
            {
                "V2_UNIVERSE_SOURCE_CAPS": "onchain:120,onchain+market:120,dexscreener:90,geckoterminal:90,watchlist:20",
            },
            contract,
        )
        self.assertEqual(issues, [])

    def test_watchlist_source_toggle_is_allowed(self) -> None:
        contract = load_contract(ROOT)
        issues = validate_overrides({"PAPER_ALLOW_WATCHLIST_SOURCE": "true"}, contract)
        self.assertEqual(issues, [])

    def test_ev_first_runtime_keys_are_allowed(self) -> None:
        contract = load_contract(ROOT)
        issues = validate_overrides(
            {
                "EV_FIRST_ENTRY_MIN_NET_USD": "0.0008",
                "EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD": "0.0040",
            },
            contract,
        )
        self.assertEqual(issues, [])

    def test_a_core_min_trade_floor_is_allowed(self) -> None:
        contract = load_contract(ROOT)
        issues = validate_overrides({"ENTRY_A_CORE_MIN_TRADE_USD": "0.25"}, contract)
        self.assertEqual(issues, [])

    def test_source_feed_window_keys_are_allowed(self) -> None:
        contract = load_contract(ROOT)
        issues = validate_overrides(
            {
                "TOKEN_AGE_MAX": "5400",
                "SEEN_TOKEN_TTL": "3600",
            },
            contract,
        )
        self.assertEqual(issues, [])

    def test_runtime_edge_floor_keys_are_allowed(self) -> None:
        contract = load_contract(ROOT)
        issues = validate_overrides(
            {
                "V2_ROLLING_EDGE_MIN_USD": "0.0010",
                "V2_ROLLING_EDGE_MIN_PERCENT": "0.20",
                "V2_CALIBRATION_ENABLED": "true",
                "V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW": "true",
                "V2_CALIBRATION_EDGE_USD_MIN": "0.0010",
                "V2_CALIBRATION_VOLUME_MIN": "20",
            },
            contract,
        )
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
