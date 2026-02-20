from __future__ import annotations

import unittest

import config
from trading import runtime_policy


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


class RuntimePolicyTests(ConfigPatchMixin, unittest.TestCase):
    def test_policy_state_fail_closed_when_safety_api_bad(self) -> None:
        self.patch_cfg(
            TOKEN_SAFETY_FAIL_CLOSED=True,
            DATA_POLICY_FAIL_CLOSED_FAIL_CLOSED_RATIO=40.0,
            DATA_POLICY_FAIL_CLOSED_API_ERROR_PERCENT=95.0,
            DATA_POLICY_DEGRADED_ERROR_PERCENT=35.0,
        )
        mode, reason = runtime_policy.policy_state(
            source_stats={"dex": {"error_percent": 10.0}},
            safety_stats={"checks_total": 10, "fail_closed": 6, "api_error_percent": 10.0},
        )
        self.assertEqual(mode, "FAIL_CLOSED")
        self.assertIn("fail_closed_ratio", reason)

    def test_policy_state_degraded_on_source_errors(self) -> None:
        self.patch_cfg(TOKEN_SAFETY_FAIL_CLOSED=False, DATA_POLICY_DEGRADED_ERROR_PERCENT=30.0)
        mode, _ = runtime_policy.policy_state(
            source_stats={"dex": {"error_percent": 45.0}},
            safety_stats={"checks_total": 10, "fail_closed": 0, "api_error_percent": 0.0},
        )
        self.assertEqual(mode, "DEGRADED")

    def test_detect_market_regime_green_from_candidates(self) -> None:
        self.patch_cfg(
            MARKET_REGIME_FAIL_CLOSED_RATIO=25.0,
            MARKET_REGIME_SOURCE_ERROR_PERCENT=20.0,
            MARKET_REGIME_MOMENTUM_CANDIDATES=2.0,
            MARKET_REGIME_THIN_CANDIDATES=0.8,
        )
        mode, _ = runtime_policy.detect_market_regime(
            policy_state_now="OK",
            source_stats={"dex": {"error_percent": 0.0}},
            safety_stats={"checks_total": 0, "fail_closed": 0},
            avg_candidates_recent=2.8,
        )
        self.assertEqual(mode, "GREEN")

    def test_market_mode_hysteresis_requires_streaks(self) -> None:
        self.patch_cfg(MARKET_MODE_ENTER_STREAK=2, MARKET_MODE_EXIT_STREAK=3)
        mode1, _, risk1, rec1 = runtime_policy.apply_market_mode_hysteresis(
            raw_mode="RED",
            raw_reason="x",
            current_mode="YELLOW",
            risk_streak=0,
            recover_streak=0,
        )
        self.assertEqual(mode1, "YELLOW")
        self.assertEqual(risk1, 1)
        mode2, _, risk2, rec2 = runtime_policy.apply_market_mode_hysteresis(
            raw_mode="RED",
            raw_reason="x",
            current_mode=mode1,
            risk_streak=risk1,
            recover_streak=rec1,
        )
        self.assertEqual(mode2, "RED")
        self.assertEqual(risk2, 2)
        self.assertEqual(rec2, 0)

    def test_market_mode_profile_reanchors_when_edge_drift_is_high(self) -> None:
        self.patch_cfg(
            V2_REGIME_REANCHOR_ENABLED=True,
            V2_REGIME_REANCHOR_EDGE_REFERENCE_PERCENT=0.60,
            V2_REGIME_REANCHOR_EDGE_REFERENCE_USD=0.010,
            V2_REGIME_REANCHOR_SENSITIVITY=1.0,
            V2_REGIME_REANCHOR_MAX_RELAX=0.50,
            MIN_EXPECTED_EDGE_PERCENT=1.20,
            MIN_EXPECTED_EDGE_USD=0.020,
            MARKET_MODE_RED_SCORE_DELTA=4,
            MARKET_MODE_RED_VOLUME_MULT=1.40,
            MARKET_MODE_RED_EDGE_MULT=1.30,
        )
        profile = runtime_policy.market_mode_entry_profile("RED")
        self.assertLessEqual(int(profile.get("score_delta", 0) or 0), 2)
        self.assertLess(float(profile.get("volume_mult", 1.0) or 1.0), 1.40)
        self.assertLess(float(profile.get("edge_mult", 1.0) or 1.0), 1.30)
        self.assertGreater(float(profile.get("reanchor_relax", 0.0) or 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
