from __future__ import annotations

import unittest

from utils import log_contracts


class LogContractsTests(unittest.TestCase):
    def test_candidate_event_adds_trace_and_reason_code(self) -> None:
        row = log_contracts.candidate_decision_event(
            {
                "candidate_id": "cand-1",
                "decision_stage": "filter_fail",
                "decision": "skip",
                "reason": "safe_volume",
                "symbol": "AAA",
                "address": "0x1111111111111111111111111111111111111111",
            },
            run_tag="mx_a",
        )
        self.assertEqual(row["trace_id"], "cand-1")
        self.assertTrue(str(row.get("decision_id", "")).startswith("dec_"))
        self.assertEqual(row["reason_code"], "FILTER_SAFE_VOLUME")
        self.assertEqual(row["reason_category"], "filter")

    def test_trade_event_generates_position_id_for_trade_stages(self) -> None:
        row = log_contracts.trade_decision_event(
            {
                "candidate_id": "cand-2",
                "decision_stage": "trade_open",
                "decision": "open",
                "reason": "buy_paper",
                "symbol": "BBB",
                "token_address": "0x2222222222222222222222222222222222222222",
            },
            run_tag="mx_b",
        )
        self.assertEqual(row["trace_id"], "cand-2")
        self.assertTrue(str(row.get("position_id", "")).startswith("pos_"))
        self.assertEqual(row["reason_code"], "EXEC_BUY_PAPER")
        self.assertEqual(row["reason_category"], "execute")

    def test_local_alert_event_gets_reason_code(self) -> None:
        row = log_contracts.local_alert_event(
            {
                "symbol": "CCC",
                "recommendation": "BUY",
                "risk_level": "MEDIUM",
            },
            run_tag="mx_c",
        )
        self.assertTrue(str(row.get("reason_code", "")).startswith("ALERT_"))
        self.assertEqual(row["decision_stage"], "alert")
        self.assertEqual(row["decision"], "emit")


if __name__ == "__main__":
    unittest.main()
