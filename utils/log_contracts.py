"""Stable log contracts shared across runtime writers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

LOG_SCHEMA_VERSION = "2026-02-19.v1"

SCHEMA_CANDIDATE_DECISION = "candidate_decision.v1"
SCHEMA_TRADE_DECISION = "trade_decision.v1"
SCHEMA_LOCAL_ALERT = "local_alert.v1"


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
    payload.setdefault("reason", "")
    payload["symbol"] = str(payload.get("symbol", "N/A") or "N/A")
    payload["score"] = _safe_int(payload.get("score", 0), 0)
    payload["market_regime"] = str(payload.get("market_regime", payload.get("market_mode", "")) or "")
    payload["market_mode"] = str(payload.get("market_mode", payload.get("market_regime", "")) or "")

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
    payload.setdefault("reason", "")
    payload["symbol"] = str(payload.get("symbol", "N/A") or "N/A")
    payload["score"] = _safe_int(payload.get("score", 0), 0)
    payload["market_mode"] = str(payload.get("market_mode", payload.get("market_regime", "")) or "")
    payload["entry_tier"] = str(payload.get("entry_tier", "") or "")
    payload["entry_channel"] = str(payload.get("entry_channel", "") or "")
    payload["position_size_usd"] = _safe_float(payload.get("position_size_usd", 0.0), 0.0)
    payload["expected_edge_percent"] = _safe_float(payload.get("expected_edge_percent", 0.0), 0.0)
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
    payload.setdefault("breakdown", {})
    return payload
