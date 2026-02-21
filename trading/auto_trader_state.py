"""Persistence helpers extracted from AutoTrader runtime logic."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import config
from utils.addressing import normalize_address

logger = logging.getLogger(__name__)


def save_state(trader: Any) -> None:
    tmp_path = ""
    try:
        trader._prune_closed_positions()
        payload = {
            "initial_balance_usd": trader.initial_balance_usd,
            "paper_balance_usd": trader.paper_balance_usd,
            "realized_pnl_usd": trader.realized_pnl_usd,
            "session_peak_realized_pnl_usd": trader._session_peak_realized_pnl_usd,
            "session_profit_lock_last_trigger_ts": trader._session_profit_lock_last_trigger_ts,
            "session_profit_lock_armed": bool(getattr(trader, "_session_profit_lock_armed", True)),
            "session_profit_lock_rearm_ready_ts": float(getattr(trader, "_session_profit_lock_rearm_ready_ts", 0.0)),
            "session_profit_lock_rearm_floor_usd": float(getattr(trader, "_session_profit_lock_rearm_floor_usd", 0.0)),
            "session_profit_lock_last_floor_usd": float(getattr(trader, "_session_profit_lock_last_floor_usd", 0.0)),
            "session_profit_lock_last_metric_usd": float(getattr(trader, "_session_profit_lock_last_metric_usd", 0.0)),
            "total_plans": trader.total_plans,
            "total_executed": trader.total_executed,
            "total_closed": trader.total_closed,
            "total_wins": trader.total_wins,
            "total_losses": trader.total_losses,
            "current_loss_streak": trader.current_loss_streak,
            "trading_pause_until_ts": trader.trading_pause_until_ts,
            "last_pause_reason": str(getattr(trader, "_last_pause_reason", "") or ""),
            "last_pause_detail": str(getattr(trader, "_last_pause_detail", "") or ""),
            "last_pause_trigger_ts": float(getattr(trader, "_last_pause_trigger_ts", 0.0) or 0.0),
            "day_id": trader.day_id,
            "day_start_equity_usd": trader.day_start_equity_usd,
            "day_realized_pnl_usd": trader.day_realized_pnl_usd,
            "token_cooldowns": trader.token_cooldowns,
            "token_cooldown_strikes": trader.token_cooldown_strikes,
            "symbol_cooldowns": trader.symbol_cooldowns,
            "trade_open_timestamps": trader.trade_open_timestamps[-500:],
            "tx_event_timestamps": trader.tx_event_timestamps[-1000:],
            "price_guard_pending": trader.price_guard_pending,
            "recovery_queue": trader.recovery_queue[-200:],
            "recovery_untracked": trader.recovery_untracked,
            "recovery_attempts": trader._recovery_attempts,
            "recovery_last_attempt_ts": trader._recovery_last_attempt_ts,
            "data_policy_mode": trader.data_policy_mode,
            "data_policy_reason": trader.data_policy_reason,
            "emergency_halt_reason": trader.emergency_halt_reason,
            "emergency_halt_ts": trader.emergency_halt_ts,
            "stair_floor_usd": trader.stair_floor_usd,
            "stair_peak_balance_usd": trader.stair_peak_balance_usd,
            "last_reinvest_mult": trader._last_reinvest_mult,
            "live_start_ts": trader.live_start_ts,
            "live_start_balance_eth": trader.live_start_balance_eth,
            "live_start_balance_usd": trader.live_start_balance_usd,
            "open_positions": [trader._serialize_pos(p) for p in trader.open_positions.values()],
            "closed_positions": [trader._serialize_pos(p) for p in trader.closed_positions[-500:]],
        }
        state_dir = os.path.dirname(trader.state_file) or "."
        os.makedirs(state_dir, exist_ok=True)
        tmp_path = f"{trader.state_file}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, trader.state_file)
        trader._last_state_flush_ts = time.time()
    except Exception as exc:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        logger.warning("AutoTrade state save failed: %s", exc)


def load_state(trader: Any) -> None:
    if not os.path.exists(trader.state_file):
        return
    try:
        with open(trader.state_file, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        trader.initial_balance_usd = float(payload.get("initial_balance_usd", trader.initial_balance_usd))
        trader.paper_balance_usd = float(payload.get("paper_balance_usd", trader.paper_balance_usd))
        trader.realized_pnl_usd = float(payload.get("realized_pnl_usd", 0.0))
        trader._session_peak_realized_pnl_usd = float(
            payload.get("session_peak_realized_pnl_usd", trader.realized_pnl_usd) or 0.0
        )
        trader._session_profit_lock_last_trigger_ts = float(
            payload.get("session_profit_lock_last_trigger_ts", 0.0) or 0.0
        )
        trader._session_profit_lock_armed = bool(payload.get("session_profit_lock_armed", True))
        trader._session_profit_lock_rearm_ready_ts = float(
            payload.get("session_profit_lock_rearm_ready_ts", 0.0) or 0.0
        )
        trader._session_profit_lock_rearm_floor_usd = float(
            payload.get("session_profit_lock_rearm_floor_usd", 0.0) or 0.0
        )
        trader._session_profit_lock_last_floor_usd = float(
            payload.get("session_profit_lock_last_floor_usd", 0.0) or 0.0
        )
        trader._session_profit_lock_last_metric_usd = float(
            payload.get("session_profit_lock_last_metric_usd", 0.0) or 0.0
        )
        trader.total_plans = int(payload.get("total_plans", 0))
        trader.total_executed = int(payload.get("total_executed", 0))
        trader.total_closed = int(payload.get("total_closed", 0))
        trader.total_wins = int(payload.get("total_wins", 0))
        trader.total_losses = int(payload.get("total_losses", 0))
        trader.current_loss_streak = int(payload.get("current_loss_streak", 0))
        if bool(getattr(config, "PERSIST_TRADING_PAUSE_ON_RESTART", True)):
            trader.trading_pause_until_ts = float(payload.get("trading_pause_until_ts", 0.0) or 0.0)
        else:
            trader.trading_pause_until_ts = 0.0
        trader._last_pause_reason = str(payload.get("last_pause_reason", "") or "")
        trader._last_pause_detail = str(payload.get("last_pause_detail", "") or "")
        trader._last_pause_trigger_ts = float(payload.get("last_pause_trigger_ts", 0.0) or 0.0)
        trader.day_id = str(payload.get("day_id", trader._current_day_id()))
        trader.day_start_equity_usd = float(payload.get("day_start_equity_usd", trader.paper_balance_usd))
        trader.day_realized_pnl_usd = float(payload.get("day_realized_pnl_usd", 0.0))
        raw_cooldowns = payload.get("token_cooldowns", {}) or {}
        trader.token_cooldowns = {normalize_address(str(k)): float(v) for k, v in raw_cooldowns.items() if normalize_address(str(k))}
        raw_cooldown_strikes = payload.get("token_cooldown_strikes", {}) or {}
        trader.token_cooldown_strikes = {
            normalize_address(str(k)): max(0, int(v))
            for k, v in raw_cooldown_strikes.items()
            if normalize_address(str(k))
        }
        raw_symbol_cooldowns = payload.get("symbol_cooldowns", {}) or {}
        trader.symbol_cooldowns = {
            trader._symbol_key(str(k)): float(v)
            for k, v in raw_symbol_cooldowns.items()
            if trader._symbol_key(str(k))
        }
        raw_trade_ts = payload.get("trade_open_timestamps", []) or []
        trader.trade_open_timestamps = []
        for value in raw_trade_ts:
            try:
                trader.trade_open_timestamps.append(float(value))
            except (TypeError, ValueError):
                continue
        raw_tx_ts = payload.get("tx_event_timestamps", []) or []
        trader.tx_event_timestamps = []
        for value in raw_tx_ts:
            try:
                trader.tx_event_timestamps.append(float(value))
            except (TypeError, ValueError):
                continue
        pending_raw = payload.get("price_guard_pending", {}) or {}
        trader.price_guard_pending = {}
        if isinstance(pending_raw, dict):
            for address, record in pending_raw.items():
                if not isinstance(record, dict):
                    continue
                normalized = normalize_address(str(address))
                if not normalized:
                    continue
                trader.price_guard_pending[normalized] = {
                    "price": float(record.get("price", 0.0)),
                    "count": int(record.get("count", 0)),
                    "first_ts": float(record.get("first_ts", 0.0)),
                }
        trader.recovery_queue = []
        raw_recovery = payload.get("recovery_queue", []) or []
        if isinstance(raw_recovery, list):
            for value in raw_recovery:
                addr = normalize_address(value)
                if addr:
                    trader.recovery_queue.append(addr)
        trader.recovery_untracked = {}
        raw_untracked = payload.get("recovery_untracked", {}) or {}
        if isinstance(raw_untracked, dict):
            for k, v in raw_untracked.items():
                addr = normalize_address(k)
                if not addr:
                    continue
                try:
                    amount = int(v)
                except Exception:
                    amount = 0
                if amount > 0:
                    trader.recovery_untracked[addr] = amount
        trader._recovery_attempts = {}
        raw_attempts = payload.get("recovery_attempts", {}) or {}
        if isinstance(raw_attempts, dict):
            for k, v in raw_attempts.items():
                addr = normalize_address(k)
                if not addr:
                    continue
                try:
                    trader._recovery_attempts[addr] = int(v)
                except Exception:
                    continue
        trader._recovery_last_attempt_ts = {}
        raw_attempt_ts = payload.get("recovery_last_attempt_ts", {}) or {}
        if isinstance(raw_attempt_ts, dict):
            for k, v in raw_attempt_ts.items():
                addr = normalize_address(k)
                if not addr:
                    continue
                try:
                    trader._recovery_last_attempt_ts[addr] = float(v)
                except Exception:
                    continue
        trader.data_policy_mode = str(payload.get("data_policy_mode", trader.data_policy_mode) or "OK").upper()
        if trader.data_policy_mode not in {"OK", "DEGRADED", "FAIL_CLOSED", "LIMITED", "BLOCKED"}:
            trader.data_policy_mode = "OK"
        trader.data_policy_reason = str(payload.get("data_policy_reason", trader.data_policy_reason) or "")
        trader.emergency_halt_reason = str(payload.get("emergency_halt_reason", "") or "")
        trader.emergency_halt_ts = float(payload.get("emergency_halt_ts", 0.0) or 0.0)
        trader.stair_floor_usd = float(payload.get("stair_floor_usd", trader.stair_floor_usd))
        trader.stair_peak_balance_usd = float(payload.get("stair_peak_balance_usd", trader.paper_balance_usd))
        trader._last_reinvest_mult = float(payload.get("last_reinvest_mult", trader._last_reinvest_mult) or 1.0)
        trader.live_start_ts = float(payload.get("live_start_ts", trader.live_start_ts) or 0.0)
        trader.live_start_balance_eth = float(payload.get("live_start_balance_eth", trader.live_start_balance_eth) or 0.0)
        trader.live_start_balance_usd = float(payload.get("live_start_balance_usd", trader.live_start_balance_usd) or 0.0)
        trader._prune_hourly_window()

        trader.open_positions.clear()
        for row in payload.get("open_positions", []):
            pos = trader._deserialize_pos(row)
            if pos and pos.status == "OPEN":
                trader._set_open_position(pos)

        trader.closed_positions.clear()
        for row in payload.get("closed_positions", []):
            pos = trader._deserialize_pos(row)
            if pos:
                trader.closed_positions.append(pos)
        trader._prune_closed_positions()
        trader._sync_stair_state()

        logger.info(
            "AutoTrade state loaded open=%s closed=%s balance=$%.2f",
            len(trader.open_positions),
            len(trader.closed_positions),
            trader.paper_balance_usd,
        )
    except Exception as exc:
        logger.warning("AutoTrade state load failed: %s", exc)
