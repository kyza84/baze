"""Policy and market regime helpers for main runtime loop."""

from __future__ import annotations

from typing import Any

import config


def policy_state(
    *,
    source_stats: dict[str, dict[str, int | float]],
    safety_stats: dict[str, int | float],
) -> tuple[str, str]:
    checks_total = int(safety_stats.get("checks_total", 0) or 0)
    fail_closed = int(safety_stats.get("fail_closed", 0) or 0)
    api_error_percent = float(safety_stats.get("api_error_percent", 0.0) or 0.0)
    fail_closed_ratio = (float(fail_closed) / checks_total * 100.0) if checks_total > 0 else 0.0
    fail_reason_top = str(safety_stats.get("fail_reason_top", "none") or "none").strip().lower() or "none"
    fail_reason_top_count = int(safety_stats.get("fail_reason_top_count", 0) or 0)

    if bool(getattr(config, "TOKEN_SAFETY_FAIL_CLOSED", False)) and (
        fail_closed_ratio >= float(getattr(config, "DATA_POLICY_FAIL_CLOSED_FAIL_CLOSED_RATIO", 60.0))
        or api_error_percent >= float(getattr(config, "DATA_POLICY_FAIL_CLOSED_API_ERROR_PERCENT", 90.0))
    ):
        return (
            "FAIL_CLOSED",
            (
                f"safety_api_unreliable fail_closed_ratio={fail_closed_ratio:.1f}% "
                f"api_err={api_error_percent:.1f}% top={fail_reason_top}:{fail_reason_top_count}"
            ),
        )

    totals = [float((row or {}).get("error_percent", 0.0) or 0.0) for row in source_stats.values()]
    max_err = max(totals) if totals else 0.0
    if max_err >= float(getattr(config, "DATA_POLICY_DEGRADED_ERROR_PERCENT", 35.0)):
        return "DEGRADED", f"source_errors_high max_err={max_err:.1f}%"

    return "OK", "healthy"


def apply_policy_hysteresis(
    *,
    raw_mode: str,
    raw_reason: str,
    current_mode: str,
    bad_streak: int,
    good_streak: int,
) -> tuple[str, str, int, int]:
    enter_req = max(1, int(getattr(config, "DATA_POLICY_ENTER_STREAK", 1) or 1))
    exit_req = max(1, int(getattr(config, "DATA_POLICY_EXIT_STREAK", 2) or 2))

    if raw_mode in {"FAIL_CLOSED", "DEGRADED"}:
        bad_streak += 1
        good_streak = 0
        if bad_streak >= enter_req:
            return raw_mode, raw_reason, bad_streak, good_streak
        return current_mode, f"hysteresis_enter_wait {bad_streak}/{enter_req} raw={raw_mode}", bad_streak, good_streak

    good_streak += 1
    bad_streak = 0
    if current_mode in {"FAIL_CLOSED", "DEGRADED"} and good_streak < exit_req:
        return current_mode, f"hysteresis_exit_wait {good_streak}/{exit_req} raw=OK", bad_streak, good_streak
    return "OK", raw_reason, bad_streak, good_streak


def detect_market_regime(
    *,
    policy_state_now: str,
    source_stats: dict[str, dict[str, int | float]],
    safety_stats: dict[str, int | float],
    avg_candidates_recent: float,
) -> tuple[str, str]:
    mode = str(policy_state_now).upper()
    if mode == "FAIL_CLOSED":
        return "RED", "policy_fail_closed"
    if mode == "DEGRADED":
        return "RED", "policy_degraded"

    checks_total = int(safety_stats.get("checks_total", 0) or 0)
    fail_closed = int(safety_stats.get("fail_closed", 0) or 0)
    fail_closed_ratio = (float(fail_closed) / checks_total * 100.0) if checks_total > 0 else 0.0
    max_err = 0.0
    for row in (source_stats or {}).values():
        max_err = max(max_err, float((row or {}).get("error_percent", 0.0) or 0.0))

    if fail_closed_ratio >= float(getattr(config, "MARKET_REGIME_FAIL_CLOSED_RATIO", 20.0)):
        return "RED", f"fail_closed={fail_closed_ratio:.1f}%"
    if max_err >= float(getattr(config, "MARKET_REGIME_SOURCE_ERROR_PERCENT", 20.0)):
        return "RED", f"max_err={max_err:.1f}%"

    if avg_candidates_recent >= float(getattr(config, "MARKET_REGIME_MOMENTUM_CANDIDATES", 2.5)):
        return "GREEN", f"avg_cand={avg_candidates_recent:.2f}"

    if avg_candidates_recent <= float(getattr(config, "MARKET_REGIME_THIN_CANDIDATES", 0.8)):
        return "YELLOW", f"avg_cand={avg_candidates_recent:.2f}"

    return "YELLOW", f"avg_cand={avg_candidates_recent:.2f}"


def market_mode_entry_profile(market_mode: str) -> dict[str, float | int | bool]:
    mode = str(market_mode or "YELLOW").strip().upper()
    if mode == "GREEN":
        return {
            "score_delta": int(getattr(config, "MARKET_MODE_GREEN_SCORE_DELTA", 0)),
            "volume_mult": float(getattr(config, "MARKET_MODE_GREEN_VOLUME_MULT", 1.0)),
            "edge_mult": float(getattr(config, "MARKET_MODE_GREEN_EDGE_MULT", 1.0)),
            "size_mult": float(getattr(config, "MARKET_MODE_GREEN_SIZE_MULT", 1.0)),
            "hold_mult": float(getattr(config, "MARKET_MODE_GREEN_HOLD_MULT", 1.0)),
            "partial_tp_trigger_mult": float(getattr(config, "MARKET_MODE_GREEN_PARTIAL_TP_TRIGGER_MULT", 1.0)),
            "partial_tp_sell_mult": float(getattr(config, "MARKET_MODE_GREEN_PARTIAL_TP_SELL_MULT", 1.0)),
            "allow_soft": True,
            "soft_cap": 0,
        }
    if mode == "RED":
        return {
            "score_delta": int(getattr(config, "MARKET_MODE_RED_SCORE_DELTA", 3)),
            "volume_mult": float(getattr(config, "MARKET_MODE_RED_VOLUME_MULT", 1.3)),
            "edge_mult": float(getattr(config, "MARKET_MODE_RED_EDGE_MULT", 1.2)),
            "size_mult": float(getattr(config, "MARKET_MODE_RED_SIZE_MULT", 0.55)),
            "hold_mult": float(getattr(config, "MARKET_MODE_RED_HOLD_MULT", 0.6)),
            "partial_tp_trigger_mult": float(getattr(config, "MARKET_MODE_RED_PARTIAL_TP_TRIGGER_MULT", 0.75)),
            "partial_tp_sell_mult": float(getattr(config, "MARKET_MODE_RED_PARTIAL_TP_SELL_MULT", 1.4)),
            "allow_soft": False,
            "soft_cap": 0,
        }
    return {
        "score_delta": int(getattr(config, "MARKET_MODE_YELLOW_SCORE_DELTA", 1)),
        "volume_mult": float(getattr(config, "MARKET_MODE_YELLOW_VOLUME_MULT", 1.1)),
        "edge_mult": float(getattr(config, "MARKET_MODE_YELLOW_EDGE_MULT", 1.08)),
        "size_mult": float(getattr(config, "MARKET_MODE_YELLOW_SIZE_MULT", 0.8)),
        "hold_mult": float(getattr(config, "MARKET_MODE_YELLOW_HOLD_MULT", 0.8)),
        "partial_tp_trigger_mult": float(getattr(config, "MARKET_MODE_YELLOW_PARTIAL_TP_TRIGGER_MULT", 0.9)),
        "partial_tp_sell_mult": float(getattr(config, "MARKET_MODE_YELLOW_PARTIAL_TP_SELL_MULT", 1.2)),
        "allow_soft": True,
        "soft_cap": int(getattr(config, "MARKET_MODE_YELLOW_SOFT_CAP_PER_CYCLE", 2)),
    }


def apply_market_mode_hysteresis(
    *,
    raw_mode: str,
    raw_reason: str,
    current_mode: str,
    risk_streak: int,
    recover_streak: int,
) -> tuple[str, str, int, int]:
    rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    raw = str(raw_mode or "YELLOW").strip().upper() or "YELLOW"
    current = str(current_mode or "YELLOW").strip().upper() or "YELLOW"
    raw_rank = rank.get(raw, 1)
    cur_rank = rank.get(current, 1)
    enter_req = max(1, int(getattr(config, "MARKET_MODE_ENTER_STREAK", 2) or 2))
    exit_req = max(1, int(getattr(config, "MARKET_MODE_EXIT_STREAK", 3) or 3))

    if raw_rank > cur_rank:
        risk_streak += 1
        recover_streak = 0
        if risk_streak >= enter_req:
            return raw, raw_reason, risk_streak, recover_streak
        return current, f"hysteresis_risk_wait {risk_streak}/{enter_req} raw={raw}", risk_streak, recover_streak

    if raw_rank < cur_rank:
        recover_streak += 1
        risk_streak = 0
        if recover_streak >= exit_req:
            return raw, raw_reason, risk_streak, recover_streak
        return current, f"hysteresis_recover_wait {recover_streak}/{exit_req} raw={raw}", risk_streak, recover_streak

    return current, raw_reason, 0, 0
