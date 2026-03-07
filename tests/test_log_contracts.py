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
                "source_mode": "onchain+market",
                "symbol": "AAA",
                "address": "0x1111111111111111111111111111111111111111",
            },
            run_tag="mx_a",
        )
        self.assertEqual(row["trace_id"], "cand-1")
        self.assertTrue(str(row.get("decision_id", "")).startswith("dec_"))
        self.assertEqual(row["reason_code"], "FILTER_SAFE_VOLUME")
        self.assertEqual(row["reason_category"], "filter")
        self.assertEqual(str(row.get("source", "")), "onchain+market")

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
        self.assertEqual(row.get("type"), "open")

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

    def test_trade_event_maps_cost_dominant_edge_to_plan_reason(self) -> None:
        row = log_contracts.trade_decision_event(
            {
                "candidate_id": "cand-3",
                "decision_stage": "plan_trade",
                "decision": "skip",
                "reason": "cost_dominant_edge",
                "symbol": "DDD",
            },
            run_tag="mx_d",
        )
        self.assertEqual(row["reason_code"], "PLAN_COST_DOMINANT_EDGE")
        self.assertEqual(row["reason_category"], "plan")
        self.assertEqual(row["reason_severity"], "INFO")

    def test_trade_event_rewrites_unknown_reason_code_when_reason_is_known(self) -> None:
        row = log_contracts.trade_decision_event(
            {
                "candidate_id": "cand-4",
                "decision_stage": "plan_select",
                "decision": "summary",
                "reason": "selection_summary",
                "reason_code": "UNKNOWN_SELECTION_SUMMARY",
                "symbol": "EEE",
            },
            run_tag="mx_e",
        )
        self.assertEqual(row["reason_code"], "PLAN_SELECTION_SUMMARY")
        self.assertEqual(row["reason_category"], "plan")

    def test_candidate_flow_metrics_reason_code_is_mapped(self) -> None:
        row = log_contracts.candidate_decision_event(
            {
                "candidate_id": "cand-flow-1",
                "decision_stage": "flow_metrics",
                "decision": "summary",
                "reason": "lane_split_summary",
                "symbol": "FLOW",
            },
            run_tag="mx_flow",
        )
        self.assertEqual(row["reason_code"], "FLOW_LANE_SPLIT_SUMMARY")
        self.assertEqual(row["reason_category"], "flow")
        self.assertEqual(row["reason_severity"], "INFO")

    def test_pre_rug_guard_reason_is_mapped(self) -> None:
        row = log_contracts.candidate_decision_event(
            {
                "candidate_id": "cand-rug-1",
                "decision_stage": "plan_trade",
                "decision": "skip",
                "reason": "pre_rug_guard",
                "symbol": "RUG",
            },
            run_tag="mx_rug",
        )
        self.assertEqual(row["reason_code"], "PRE_RUG_GUARD")
        self.assertEqual(row["reason_category"], "precheck")

    def test_safe_pump_history_reason_is_mapped(self) -> None:
        row = log_contracts.candidate_decision_event(
            {
                "candidate_id": "cand-pump-1",
                "decision_stage": "filter_fail",
                "decision": "skip",
                "reason": "safe_pump_history",
                "symbol": "PUMP",
            },
            run_tag="mx_pump",
        )
        self.assertEqual(row["reason_code"], "PRE_PUMP_HISTORY_BLOCK")
        self.assertEqual(row["reason_category"], "precheck")

    def test_proof_sell_fail_exit_reason_is_mapped(self) -> None:
        row = log_contracts.trade_decision_event(
            {
                "candidate_id": "cand-proof-1",
                "decision_stage": "trade_close",
                "decision": "close",
                "reason": "PROOF_SELL_FAIL",
                "symbol": "PSF",
            },
            run_tag="mx_proof",
        )
        self.assertEqual(row["reason_code"], "EXIT_PROOF_SELL_FAIL")
        self.assertEqual(row["reason_category"], "exit")


if __name__ == "__main__":
    unittest.main()
