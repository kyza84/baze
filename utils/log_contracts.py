"""Stable log contracts shared across runtime writers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

LOG_SCHEMA_VERSION = "2026-02-19.v1"

SCHEMA_CANDIDATE_DECISION = "candidate_decision.v1"
SCHEMA_TRADE_DECISION = "trade_decision.v1"
SCHEMA_LOCAL_ALERT = "local_alert.v1"

_STAGE_PREFIX: dict[str, str] = {
    "filter_fail": "FILTER",
    "post_filters": "FILTER",
    "quality_gate": "QUALITY",
    "plan_trade": "PLAN",
    "trade_open": "EXEC",
    "trade_partial": "EXIT",
    "trade_close": "EXIT",
    "alert": "ALERT",
    "unknown": "UNKNOWN",
}

_REASON_CODE_OVERRIDES: dict[str, str] = {
    "ev_net_low": "PLAN_EV_NET_LOW",
    "edge_low": "PLAN_EDGE_LOW",
    "cooldown": "PLAN_COOLDOWN",
    "cooldown_left": "PLAN_COOLDOWN",
    "top1_open_share_15m": "PLAN_TOP1_OPEN_SHARE_15M",
    "symbol_concentration": "PLAN_SYMBOL_CONCENTRATION",
    "source_budget": "QUALITY_SOURCE_BUDGET",
    "duplicate_address": "QUALITY_DUPLICATE_ADDRESS",
    "blacklist": "PRE_BLACKLIST",
    "honeypot_guard": "PRE_HONEYPOT_GUARD",
    "safe_risky_flags": "PRE_RISKY_FLAGS",
    "safe_source": "PRE_SAFE_SOURCE",
    "source_disabled": "PRE_SOURCE_DISABLED",
    "watchlist_strict_guard": "PRE_WATCHLIST_STRICT_GUARD",
    "safe_age": "FILTER_SAFE_AGE",
    "safe_liquidity": "FILTER_SAFE_LIQUIDITY",
    "safe_volume": "FILTER_SAFE_VOLUME",
    "safe_change_5m": "FILTER_SAFE_CHANGE_5M",
    "score_min": "FILTER_SCORE_MIN",
    "safety_budget": "FILTER_SAFETY_BUDGET",
    "source_qos_symbol_cap": "FILTER_SOURCE_QOS_SYMBOL_CAP",
    "excluded_base_token": "FILTER_EXCLUDED_BASE_TOKEN",
    "address_or_duplicate": "FILTER_ADDRESS_OR_DUPLICATE",
    "buy_paper": "EXEC_BUY_PAPER",
    "buy_live": "EXEC_BUY_LIVE",
    "sell_fail": "EXEC_SELL_FAIL",
    "buy_fail": "EXEC_BUY_FAIL",
    "unsupported_live_route": "EXEC_UNSUPPORTED_LIVE_ROUTE",
    "roundtrip_ratio": "EXEC_ROUNDTRIP_RATIO",
    "kill_switch_active": "POLICY_KILL_SWITCH_ACTIVE",
    "no_momentum": "EXIT_NO_MOMENTUM",
    "timeout": "EXIT_TIMEOUT",
    "sl": "EXIT_STOP_LOSS",
    "weakness": "EXIT_WEAKNESS",
}

REASON_CODE_TAXONOMY: dict[str, dict[str, str]] = {
    "PLAN_EV_NET_LOW": {"severity": "INFO", "category": "plan", "title": "Expected net edge below threshold"},
    "PLAN_EDGE_LOW": {"severity": "INFO", "category": "plan", "title": "Expected edge below threshold"},
    "PLAN_COOLDOWN": {"severity": "INFO", "category": "plan", "title": "Cooldown active"},
    "PRE_BLACKLIST": {"severity": "WARN", "category": "precheck", "title": "Token blocked by blacklist"},
    "PRE_HONEYPOT_GUARD": {"severity": "WARN", "category": "precheck", "title": "Honeypot guard blocked token"},
    "FILTER_SAFE_VOLUME": {"severity": "INFO", "category": "filter", "title": "Volume below safety floor"},
    "FILTER_SAFE_AGE": {"severity": "INFO", "category": "filter", "title": "Age below safety floor"},
    "FILTER_SCORE_MIN": {"severity": "INFO", "category": "filter", "title": "Score below minimum"},
    "EXEC_BUY_PAPER": {"severity": "INFO", "category": "execute", "title": "Paper buy opened"},
    "EXEC_BUY_LIVE": {"severity": "INFO", "category": "execute", "title": "Live buy opened"},
    "EXIT_TIMEOUT": {"severity": "INFO", "category": "exit", "title": "Closed by timeout"},
    "EXIT_STOP_LOSS": {"severity": "WARN", "category": "exit", "title": "Closed by stop loss"},
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_ts(value: Any) -> float:
    if value is None or value == "":
        return datetime.now(timezone.utc).timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return datetime.now(timezone.utc).timestamp()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return datetime.now(timezone.utc).timestamp()


def _iso_from_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _normalize_address(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if len(raw) == 42 and raw.startswith("0x"):
        return raw
    return ""


def _normalize_reason_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text.startswith("cooldown_left_"):
        return "cooldown_left"
    if text.startswith("api_code_"):
        return "api_code"
    return text


def _sanitize_code_token(value: str) -> str:
    text = re.sub(r"[^A-Z0-9]+", "_", str(value or "").strip().upper())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "UNKNOWN"


def _stage_prefix(value: Any) -> str:
    stage = _normalize_reason_text(value) or "unknown"
    return _STAGE_PREFIX.get(stage, "UNKNOWN")


def reason_code_for_event(
    *,
    reason: Any,
    decision_stage: Any = "",
    decision: Any = "",
) -> str:
    normalized_reason = _normalize_reason_text(reason)
    if not normalized_reason:
        normalized_decision = _normalize_reason_text(decision)
        if normalized_decision:
            return f"{_stage_prefix(decision_stage)}_{_sanitize_code_token(normalized_decision)}"
        return "UNKNOWN"
    override = _REASON_CODE_OVERRIDES.get(normalized_reason)
    if override:
        return override
    if normalized_reason == "api_code":
        return "SAFETY_API_CODE"
    return f"{_stage_prefix(decision_stage)}_{_sanitize_code_token(normalized_reason)}"


def reason_code_meta(code: str) -> dict[str, str]:
    key = _sanitize_code_token(code)
    if key in REASON_CODE_TAXONOMY:
        return dict(REASON_CODE_TAXONOMY[key])
    return {
        "severity": "INFO",
        "category": "unknown",
        "title": key.replace("_", " ").title(),
    }


def _digest_seed(*parts: Any) -> str:
    seed = "|".join(str(p or "").strip() for p in parts)
    return hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()


def _trace_id(payload: dict[str, Any]) -> str:
    raw = str(payload.get("trace_id", "") or "").strip()
    if raw:
        return raw
    candidate_id = str(payload.get("candidate_id", "") or "").strip()
    if candidate_id:
        return candidate_id
    token_address = _normalize_address(payload.get("token_address", payload.get("address", "")))
    symbol = str(payload.get("symbol", "") or "").strip().upper()
    ts = _as_ts(payload.get("ts", payload.get("timestamp")))
    return f"tr_{_digest_seed(token_address, symbol, f'{ts:.6f}')[:20]}"


def _decision_id(payload: dict[str, Any], *, run_tag: str) -> str:
    raw = str(payload.get("decision_id", "") or "").strip()
    if raw:
        return raw
    ts = _as_ts(payload.get("ts", payload.get("timestamp")))
    token_address = _normalize_address(payload.get("token_address", payload.get("address", "")))
    return (
        "dec_"
        + _digest_seed(
            run_tag,
            payload.get("trace_id", ""),
            payload.get("decision_stage", ""),
            payload.get("decision", ""),
            payload.get("reason", ""),
            token_address,
            payload.get("symbol", ""),
            f"{ts:.6f}",
        )[:20]
    )


def _position_id(payload: dict[str, Any]) -> str:
    raw = str(payload.get("position_id", "") or "").strip()
    if raw:
        return raw
    stage = _normalize_reason_text(payload.get("decision_stage", ""))
    if stage not in {"trade_open", "trade_partial", "trade_close"}:
        return ""
    token_address = _normalize_address(payload.get("token_address", payload.get("address", "")))
    trace_id = str(payload.get("trace_id", "") or "").strip()
    candidate_id = str(payload.get("candidate_id", "") or "").strip()
    if not (trace_id or candidate_id or token_address):
        return ""
    return f"pos_{_digest_seed(trace_id, candidate_id, token_address)[:20]}"


def stamp_event(
    event: dict[str, Any],
    *,
    schema_name: str,
    event_type: str,
    run_tag: str = "",
) -> dict[str, Any]:
    payload = dict(event or {})
    ts = _as_ts(payload.get("ts", payload.get("timestamp")))
    payload["ts"] = float(ts)
    payload["timestamp"] = str(payload.get("timestamp", "") or _iso_from_ts(ts))
    payload.setdefault("schema_version", LOG_SCHEMA_VERSION)
    payload.setdefault("schema_name", schema_name)
    payload.setdefault("event_type", str(event_type or "event"))
    if run_tag:
        payload.setdefault("run_tag", str(run_tag))
    payload["trace_id"] = _trace_id(payload)
    payload["decision_id"] = _decision_id(payload, run_tag=str(payload.get("run_tag", run_tag or "")))
    payload["parent_decision_id"] = str(payload.get("parent_decision_id", "") or "")
    payload["position_id"] = _position_id(payload)
    return payload


def candidate_decision_event(event: dict[str, Any], *, run_tag: str = "") -> dict[str, Any]:
    payload = stamp_event(
        event,
        schema_name=SCHEMA_CANDIDATE_DECISION,
        event_type=str((event or {}).get("event_type", "candidate_decision")),
        run_tag=run_tag,
    )
    payload.setdefault("candidate_id", "")
    payload.setdefault("decision_stage", "unknown")
    payload.setdefault("decision", "unknown")
    payload["reason"] = str(payload.get("reason", "") or "")
    payload["symbol"] = str(payload.get("symbol", "N/A") or "N/A")
    payload["score"] = _safe_int(payload.get("score", 0), 0)
    payload["market_regime"] = str(payload.get("market_regime", payload.get("market_mode", "")) or "")
    payload["market_mode"] = str(payload.get("market_mode", payload.get("market_regime", "")) or "")
    payload["reason_code"] = str(
        payload.get("reason_code", "")
        or reason_code_for_event(
            reason=payload.get("reason", ""),
            decision_stage=payload.get("decision_stage", ""),
            decision=payload.get("decision", ""),
        )
    ).strip().upper()
    meta = reason_code_meta(payload["reason_code"])
    payload["reason_severity"] = str(payload.get("reason_severity", meta.get("severity", "INFO")) or "INFO")
    payload["reason_category"] = str(payload.get("reason_category", meta.get("category", "unknown")) or "unknown")

    token_address = _normalize_address(payload.get("token_address", payload.get("address", "")))
    if token_address:
        payload["token_address"] = token_address
        payload["address"] = token_address
    else:
        payload.setdefault("token_address", "")
        payload.setdefault("address", "")
    payload.setdefault("source_mode", str(payload.get("source", "") or ""))
    return payload


def trade_decision_event(event: dict[str, Any], *, run_tag: str = "") -> dict[str, Any]:
    payload = stamp_event(
        event,
        schema_name=SCHEMA_TRADE_DECISION,
        event_type=str((event or {}).get("event_type", "trade_decision")),
        run_tag=run_tag,
    )
    payload.setdefault("candidate_id", "")
    payload.setdefault("decision_stage", "unknown")
    payload.setdefault("decision", "unknown")
    payload["reason"] = str(payload.get("reason", "") or "")
    payload["symbol"] = str(payload.get("symbol", "N/A") or "N/A")
    payload["score"] = _safe_int(payload.get("score", 0), 0)
    payload["market_mode"] = str(payload.get("market_mode", payload.get("market_regime", "")) or "")
    payload["entry_tier"] = str(payload.get("entry_tier", "") or "")
    payload["entry_channel"] = str(payload.get("entry_channel", "") or "")
    payload["position_size_usd"] = _safe_float(payload.get("position_size_usd", 0.0), 0.0)
    payload["expected_edge_percent"] = _safe_float(payload.get("expected_edge_percent", 0.0), 0.0)
    payload["reason_code"] = str(
        payload.get("reason_code", "")
        or reason_code_for_event(
            reason=payload.get("reason", ""),
            decision_stage=payload.get("decision_stage", ""),
            decision=payload.get("decision", ""),
        )
    ).strip().upper()
    meta = reason_code_meta(payload["reason_code"])
    payload["reason_severity"] = str(payload.get("reason_severity", meta.get("severity", "INFO")) or "INFO")
    payload["reason_category"] = str(payload.get("reason_category", meta.get("category", "unknown")) or "unknown")
    token_address = _normalize_address(payload.get("token_address", payload.get("address", "")))
    if token_address:
        payload["token_address"] = token_address
    else:
        payload.setdefault("token_address", "")
    return payload


def local_alert_event(event: dict[str, Any], *, run_tag: str = "") -> dict[str, Any]:
    payload = stamp_event(
        event,
        schema_name=SCHEMA_LOCAL_ALERT,
        event_type=str((event or {}).get("event_type", "local_alert")),
        run_tag=run_tag,
    )
    payload["name"] = str(payload.get("name", "Unknown") or "Unknown")
    payload["symbol"] = str(payload.get("symbol", "N/A") or "N/A")
    payload["address"] = _normalize_address(payload.get("address", ""))
    payload["score"] = _safe_int(payload.get("score", 0), 0)
    payload["recommendation"] = str(payload.get("recommendation", "INFO") or "INFO")
    payload["risk_level"] = str(payload.get("risk_level", "INFO") or "INFO")
    payload["warning_flags"] = max(0, _safe_int(payload.get("warning_flags", 0), 0))
    payload["liquidity"] = _safe_float(payload.get("liquidity", 0.0), 0.0)
    payload["volume_5m"] = _safe_float(payload.get("volume_5m", 0.0), 0.0)
    payload["price_change_5m"] = _safe_float(payload.get("price_change_5m", 0.0), 0.0)
    payload["age_minutes"] = max(0, _safe_int(payload.get("age_minutes", 0), 0))
    payload["decision_stage"] = str(payload.get("decision_stage", "alert") or "alert")
    payload["decision"] = str(payload.get("decision", "emit") or "emit")
    payload["reason"] = str(payload.get("reason", "") or "").strip()
    if not payload["reason"]:
        payload["reason"] = f"recommendation_{str(payload.get('recommendation', 'info')).strip().lower() or 'info'}"
    payload["reason_code"] = str(
        payload.get("reason_code", "")
        or reason_code_for_event(
            reason=payload.get("reason", ""),
            decision_stage=payload.get("decision_stage", ""),
            decision=payload.get("decision", ""),
        )
    ).strip().upper()
    meta = reason_code_meta(payload["reason_code"])
    payload["reason_severity"] = str(payload.get("reason_severity", meta.get("severity", "INFO")) or "INFO")
    payload["reason_category"] = str(payload.get("reason_category", meta.get("category", "unknown")) or "unknown")
    payload.setdefault("breakdown", {})
    return payload
