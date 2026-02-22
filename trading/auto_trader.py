"""Auto-trading engine with full paper buy/sell cycle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import config
from trading import auto_trader_state
from trading.live_executor import LiveExecutor
from utils.addressing import normalize_address
from utils.http_client import ResilientHttpClient
from utils.log_contracts import trade_decision_event

logger = logging.getLogger(__name__)
ADDRESS_RE = re.compile(r"^0x[a-f0-9]{40}$")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass
class PaperPosition:
    token_address: str
    symbol: str
    entry_price_usd: float
    current_price_usd: float
    position_size_usd: float
    score: int
    liquidity_usd: float
    risk_level: str
    opened_at: datetime
    max_hold_seconds: int
    take_profit_percent: int
    stop_loss_percent: float
    expected_edge_percent: float = 0.0
    buy_cost_percent: float = 0.0
    sell_cost_percent: float = 0.0
    gas_cost_usd: float = 0.0
    status: str = "OPEN"
    close_reason: str = ""
    closed_at: datetime | None = None
    pnl_percent: float = 0.0
    pnl_usd: float = 0.0
    peak_pnl_percent: float = 0.0
    token_amount_raw: int = 0
    buy_tx_hash: str = ""
    sell_tx_hash: str = ""
    buy_tx_status: str = "none"
    sell_tx_status: str = "none"
    spent_eth: float = 0.0
    original_position_size_usd: float = 0.0
    partial_tp_done: bool = False
    partial_tp_stage: int = 0
    partial_realized_pnl_usd: float = 0.0
    last_partial_tp_trigger_percent: float = 0.0
    timeout_extension_seconds: int = 0
    market_mode: str = ""
    entry_tier: str = ""
    entry_channel: str = ""
    partial_tp_trigger_mult: float = 1.0
    partial_tp_sell_mult: float = 1.0
    candidate_id: str = ""
    token_cluster_key: str = ""
    ev_expected_net_usd: float = 0.0
    ev_confidence: float = 0.0
    kelly_mult: float = 1.0
    execution_lane: str = ""
    execution_phase: str = ""
    cluster_ev_avg_net_usd: float = 0.0
    cluster_ev_loss_share: float = 0.0
    cluster_ev_samples: float = 0.0


class AutoTrader:
    def __init__(self) -> None:
        self.state_file = str(
            getattr(
                config,
                "PAPER_STATE_FILE",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_state.json"),
            )
        )
        self._blacklist_file = str(getattr(config, "AUTOTRADE_BLACKLIST_FILE", os.path.join("data", "autotrade_blacklist.json")))
        self._blacklist: dict[str, dict[str, Any]] = {}
        self.open_positions: dict[str, PaperPosition] = {}
        self.closed_positions: list[PaperPosition] = []
        self.total_plans = 0
        self.total_executed = 0
        self.total_closed = 0
        self.total_wins = 0
        self.total_losses = 0
        self.current_loss_streak = 0
        self.token_cooldowns: dict[str, float] = {}
        self.token_cooldown_strikes: dict[str, int] = {}
        self.symbol_cooldowns: dict[str, float] = {}
        self.trade_open_timestamps: list[float] = []
        self._core_probe_open_timestamps: list[float] = []
        self.tx_event_timestamps: list[float] = []
        self.price_guard_pending: dict[str, dict[str, float | int]] = {}
        self.trading_pause_until_ts = 0.0
        self._last_pause_reason = ""
        self._last_pause_detail = ""
        self._last_pause_trigger_ts = 0.0
        self._session_peak_realized_pnl_usd = 0.0
        self._session_profit_lock_last_trigger_ts = 0.0
        self._session_profit_lock_armed = True
        self._session_profit_lock_rearm_ready_ts = 0.0
        self._session_profit_lock_rearm_floor_usd = 0.0
        self._session_profit_lock_last_floor_usd = 0.0
        self._session_profit_lock_last_metric_usd = 0.0
        self.day_id = self._current_day_id()
        self.day_start_equity_usd = float(config.WALLET_BALANCE_USD)
        self.day_realized_pnl_usd = 0.0
        self.last_weth_price_usd = max(0.0, float(config.WETH_PRICE_FALLBACK_USD))
        self.stair_floor_usd = 0.0
        self.stair_peak_balance_usd = float(config.WALLET_BALANCE_USD)
        self.emergency_halt_reason = ""
        self.emergency_halt_ts = 0.0
        self._last_guard_log_ts = 0.0
        self._live_sell_failures = 0
        self._honeypot_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.recovery_queue: list[str] = []
        self.recovery_untracked: dict[str, int] = {}
        self._recovery_attempts: dict[str, int] = {}
        self._recovery_last_attempt_ts: dict[str, float] = {}
        self._skip_reason_counts_window: dict[str, int] = {}
        self._recent_open_symbols: list[tuple[float, str]] = []
        self._last_market_mode_seen = "YELLOW"
        self._last_reinvest_mult = 1.0
        self._ev_db_cache: dict[str, tuple[float, dict[str, float]]] = {}
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        ev_db_default = os.path.join(root, "data", "unified_dataset", "unified.db")
        raw_ev_db = str(getattr(config, "EV_FIRST_ENTRY_DB_PATH", ev_db_default) or ev_db_default).strip()
        self._ev_db_path = raw_ev_db if os.path.isabs(raw_ev_db) else os.path.join(root, raw_ev_db)
        self.trade_decisions_log_enabled = bool(getattr(config, "TRADE_DECISIONS_LOG_ENABLED", True))
        raw_decisions_log = str(
            getattr(config, "TRADE_DECISIONS_LOG_FILE", os.path.join("logs", "trade_decisions.jsonl")) or ""
        ).strip()
        if not raw_decisions_log:
            raw_decisions_log = os.path.join("logs", "trade_decisions.jsonl")
        self.trade_decisions_log_file = (
            raw_decisions_log if os.path.isabs(raw_decisions_log) else os.path.join(root, raw_decisions_log)
        )
        self._active_trade_decision_context: dict[str, Any] | None = None
        self.data_policy_mode = "OK"
        self.data_policy_reason = ""
        self._http = ResilientHttpClient(
            timeout_seconds=float(max(config.DEX_TIMEOUT, int(config.HONEYPOT_API_TIMEOUT_SECONDS))),
            source_limits={
                "dex_price": 6,
                "honeypot": 3,
            },
        )

        # Live session baseline tracking (used for "stop after profit" guard).
        self.live_start_ts = 0.0
        self.live_start_balance_eth = 0.0
        self.live_start_balance_usd = 0.0

        self.initial_balance_usd = float(config.WALLET_BALANCE_USD)
        self.paper_balance_usd = float(config.WALLET_BALANCE_USD)
        self.realized_pnl_usd = 0.0
        self.live_executor: LiveExecutor | None = None
        self._last_paper_summary_ts = 0.0
        self._paper_summary_prev_closed = 0
        self._paper_summary_prev_wins = 0
        self._paper_summary_prev_losses = 0
        self._paper_summary_prev_realized_usd = 0.0
        self._last_state_flush_ts = 0.0
        self._last_untracked_discovery_ts = 0.0
        self._state_flush_interval_seconds = max(
            0.2,
            float(getattr(config, "PAPER_STATE_FLUSH_INTERVAL_SECONDS", 1.0) or 1.0),
        )
        self._load_state()
        now_ts = datetime.now(timezone.utc).timestamp()
        self._last_paper_summary_ts = now_ts
        self._paper_summary_prev_closed = int(self.total_closed)
        self._paper_summary_prev_wins = int(self.total_wins)
        self._paper_summary_prev_losses = int(self.total_losses)
        self._paper_summary_prev_realized_usd = float(self.realized_pnl_usd)
        if bool(getattr(config, "LIVE_SESSION_RESET_ON_START", True)):
            # Reset baseline on each process start to make overnight runs predictable.
            self.live_start_ts = 0.0
            self.live_start_balance_eth = 0.0
            self.live_start_balance_usd = 0.0
        self._load_blacklist()
        if config.PAPER_RESET_ON_START:
            self._apply_startup_reset()
        self._sync_stair_state()
        self._prune_closed_positions()
        if self._is_live_mode():
            try:
                self.live_executor = LiveExecutor()
                logger.info("Live executor ready wallet=%s", config.LIVE_WALLET_ADDRESS)
                self._reconcile_live_state_on_startup()
                self._prune_recovery_untracked_live()
            except Exception as exc:
                self.live_executor = None
                logger.error("Live executor init failed: %s", exc)

    def _load_blacklist(self) -> None:
        if not bool(getattr(config, "AUTOTRADE_BLACKLIST_ENABLED", True)):
            self._blacklist = {}
            return
        try:
            path = self._blacklist_file
            if not path:
                self._blacklist = {}
                return
            if not os.path.exists(path):
                self._blacklist = {}
                return
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                self._blacklist = payload  # type: ignore[assignment]
            else:
                self._blacklist = {}
        except Exception:
            self._blacklist = {}

    def _save_blacklist(self) -> None:
        if not bool(getattr(config, "AUTOTRADE_BLACKLIST_ENABLED", True)):
            return
        try:
            path = self._blacklist_file
            if not path:
                return
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._blacklist, f, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            pass

    def _blacklist_prune(self) -> None:
        now_ts = datetime.now(timezone.utc).timestamp()
        # Drop expired
        self._blacklist = {
            k: v
            for k, v in self._blacklist.items()
            if float((v or {}).get("until_ts") or 0.0) > now_ts
        }
        max_entries = int(getattr(config, "AUTOTRADE_BLACKLIST_MAX_ENTRIES", 5000) or 5000)
        if len(self._blacklist) <= max_entries:
            return
        # Remove oldest entries (by added_ts)
        rows = sorted(
            self._blacklist.items(),
            key=lambda kv: float((kv[1] or {}).get("added_ts") or 0.0),
        )
        for addr, _ in rows[: max(0, len(rows) - max_entries)]:
            self._blacklist.pop(addr, None)

    def _blacklist_is_blocked(self, token_address: str) -> tuple[bool, str]:
        if not bool(getattr(config, "AUTOTRADE_BLACKLIST_ENABLED", True)):
            return False, ""
        key = normalize_address(token_address)
        if not key:
            return False, ""
        row = self._blacklist.get(key)
        if not isinstance(row, dict):
            return False, ""
        until_ts = float(row.get("until_ts") or 0.0)
        now_ts = datetime.now(timezone.utc).timestamp()
        if until_ts <= now_ts:
            return False, ""
        return True, str(row.get("reason") or "blacklisted")

    def _blacklist_add(self, token_address: str, reason: str, ttl_seconds: int | None = None) -> None:
        if not bool(getattr(config, "AUTOTRADE_BLACKLIST_ENABLED", True)):
            return
        key = normalize_address(token_address)
        if not key:
            return
        now_ts = datetime.now(timezone.utc).timestamp()
        ttl = int(ttl_seconds) if ttl_seconds is not None else self._blacklist_default_ttl_seconds(reason)
        ttl = max(300, ttl)
        self._blacklist[key] = {
            "reason": str(reason or "blacklisted"),
            "added_ts": now_ts,
            "until_ts": now_ts + ttl,
        }
        self._blacklist_prune()
        self._save_blacklist()

    @staticmethod
    def _blacklist_default_ttl_seconds(reason: str) -> int:
        """Shorten TTL for transient routing/quote failures to keep throughput healthy."""
        base_ttl = int(getattr(config, "AUTOTRADE_BLACKLIST_TTL_SECONDS", 86400) or 86400)
        key = str(reason or "").strip().lower()
        if key.startswith("unsupported_buy_route:") or key.startswith("unsupported_sell_route:"):
            tuned = int(getattr(config, "LIVE_BLACKLIST_UNSUPPORTED_ROUTE_TTL_SECONDS", 7200) or 7200)
            return min(base_ttl, tuned)
        if key.startswith("roundtrip_quote_failed:"):
            tuned = int(getattr(config, "LIVE_BLACKLIST_ROUNDTRIP_FAIL_TTL_SECONDS", 21600) or 21600)
            return min(base_ttl, tuned)
        if key.startswith("roundtrip_ratio:"):
            tuned = int(getattr(config, "LIVE_BLACKLIST_ROUNDTRIP_RATIO_TTL_SECONDS", 3600) or 3600)
            return min(base_ttl, tuned)
        if key.startswith("live_buy_zero_amount") or key.startswith("live_buy_failed:"):
            tuned = int(getattr(config, "LIVE_BLACKLIST_ZERO_AMOUNT_TTL_SECONDS", 1800) or 1800)
            return min(base_ttl, tuned)
        if key.startswith("honeypot_guard:") or key.startswith("abandoned:"):
            return max(base_ttl, 24 * 3600)
        return base_ttl

    def _apply_startup_reset(self) -> None:
        force_reset = bool(getattr(config, "PAPER_RESET_FORCE_ON_START", True))
        is_paper_mode = bool(getattr(config, "AUTO_TRADE_PAPER", False))
        if self.open_positions and (not (is_paper_mode and force_reset)):
            logger.info(
                "PAPER_RESET_ON_START skipped: keeping %s open position(s).",
                len(self.open_positions),
            )
            return
        if self.open_positions and is_paper_mode and force_reset:
            logger.warning(
                "PAPER_RESET_ON_START forced: dropping %s stale open paper position(s).",
                len(self.open_positions),
            )
        self.reset_paper_state(keep_closed=True)

    def _risk_drawdown_percent(self) -> float:
        self._refresh_daily_window()
        start = max(0.01, float(self.day_start_equity_usd or 0.01))
        equity = float(self._equity_usd())
        return ((equity - start) / start) * 100.0

    def _adaptive_pause_seconds(self, *, reason: str, base_seconds: int) -> int:
        sec = max(0, int(base_seconds))
        if sec <= 0:
            return 0
        if not bool(getattr(config, "RISK_GOVERNOR_ADAPTIVE_PAUSE_ENABLED", True)):
            return sec

        mode = str(self._last_market_mode_seen or "YELLOW").strip().upper() or "YELLOW"
        if mode == "GREEN":
            sec = int(round(sec * float(getattr(config, "RISK_GOVERNOR_PAUSE_GREEN_MULT", 0.65) or 0.65)))
        elif mode == "YELLOW":
            sec = int(round(sec * float(getattr(config, "RISK_GOVERNOR_PAUSE_YELLOW_MULT", 0.82) or 0.82)))
        else:
            sec = int(round(sec * float(getattr(config, "RISK_GOVERNOR_PAUSE_RED_MULT", 1.0) or 1.0)))

        min_sec = max(0, int(getattr(config, "RISK_GOVERNOR_PAUSE_MIN_SECONDS", 60) or 60))
        max_sec = max(min_sec, int(getattr(config, "RISK_GOVERNOR_PAUSE_MAX_SECONDS", 1800) or 1800))
        sec = max(min_sec, min(max_sec, sec))

        if str(reason or "").strip().lower() in {"session_profit_lock", "loss_streak"} and self._anti_choke_active():
            anti_choke_cap = max(min_sec, int(getattr(config, "RISK_GOVERNOR_ANTI_CHOKE_MAX_PAUSE_SECONDS", 300) or 300))
            sec = min(sec, anti_choke_cap)
        if str(reason or "").strip().lower() == "daily_drawdown":
            dd_min = max(min_sec, int(getattr(config, "RISK_GOVERNOR_DRAWDOWN_MIN_PAUSE_SECONDS", 240) or 240))
            sec = max(sec, dd_min)
        return max(0, int(sec))

    def _risk_trigger_pause(self, *, reason: str, detail: str, pause_seconds: int) -> None:
        pause_seconds = self._adaptive_pause_seconds(reason=str(reason or ""), base_seconds=int(pause_seconds))
        if pause_seconds <= 0:
            return
        now_ts = datetime.now(timezone.utc).timestamp()
        until_ts = now_ts + pause_seconds
        if until_ts <= float(self.trading_pause_until_ts or 0.0):
            return
        self.trading_pause_until_ts = until_ts
        self._last_pause_reason = str(reason or "")
        self._last_pause_detail = str(detail or "")
        self._last_pause_trigger_ts = now_ts
        logger.warning(
            "RISK_GOVERNOR pause reason=%s detail=%s pause=%ss until_ts=%s",
            reason,
            detail,
            pause_seconds,
            int(until_ts),
        )

    def _prune_symbol_open_window(self, window_seconds: int) -> None:
        win = max(60, int(window_seconds))
        now_ts = datetime.now(timezone.utc).timestamp()
        self._recent_open_symbols = [
            row
            for row in self._recent_open_symbols
            if (now_ts - float(row[0])) <= float(win)
        ]

    def _record_open_symbol(self, symbol: str) -> None:
        key = self._symbol_key(symbol)
        if not key:
            return
        now_ts = datetime.now(timezone.utc).timestamp()
        self._recent_open_symbols.append((now_ts, key))
        window_seconds = max(300, int(getattr(config, "SYMBOL_CONCENTRATION_WINDOW_SECONDS", 3600) or 3600))
        self._prune_symbol_open_window(window_seconds)

    def _symbol_concentration_blocked(self, *, symbol: str, entry_tier: str) -> tuple[bool, str]:
        if not bool(getattr(config, "SYMBOL_CONCENTRATION_GUARD_ENABLED", True)):
            return False, ""
        key = self._symbol_key(symbol)
        if not key:
            return False, ""
        if self._anti_choke_active():
            return False, ""

        window_seconds = max(300, int(getattr(config, "SYMBOL_CONCENTRATION_WINDOW_SECONDS", 3600) or 3600))
        min_opens = max(1, int(getattr(config, "SYMBOL_CONCENTRATION_MIN_OPENS", 12) or 12))
        max_share = max(0.01, min(1.0, float(getattr(config, "SYMBOL_CONCENTRATION_MAX_SHARE", 0.10) or 0.10)))
        tier_a_mult = max(1.0, float(getattr(config, "SYMBOL_CONCENTRATION_TIER_A_SHARE_MULT", 1.15) or 1.15))

        self._prune_symbol_open_window(window_seconds)
        total = len(self._recent_open_symbols)
        if total < min_opens:
            return False, ""
        same = sum(1 for _, s in self._recent_open_symbols if s == key)
        projected_share = float(same + 1) / float(total + 1)

        cap = max_share
        if str(entry_tier or "").strip().upper() == "A":
            cap = min(1.0, cap * tier_a_mult)
        if projected_share <= cap:
            return False, ""
        detail = (
            f"symbol_concentration symbol={key} projected_share={projected_share:.3f} "
            f"cap={cap:.3f} opens={same}/{total} window={window_seconds}s"
        )
        return True, detail

    def _risk_governor_after_close(self, *, close_reason: str, pnl_usd: float) -> None:
        if not bool(getattr(config, "RISK_GOVERNOR_ENABLED", True)):
            return
        max_streak = int(getattr(config, "RISK_GOVERNOR_MAX_LOSS_STREAK", config.MAX_CONSECUTIVE_LOSSES) or 0)
        streak_pause = int(getattr(config, "RISK_GOVERNOR_STREAK_PAUSE_SECONDS", config.LOSS_STREAK_COOLDOWN_SECONDS) or 0)
        if max_streak > 0 and int(self.current_loss_streak) >= max_streak:
            self._risk_trigger_pause(
                reason="loss_streak",
                detail=f"loss_streak={self.current_loss_streak} close_reason={close_reason}",
                pause_seconds=streak_pause,
            )

        dd_limit = float(getattr(config, "RISK_GOVERNOR_DRAWDOWN_LIMIT_PERCENT", config.DAILY_MAX_DRAWDOWN_PERCENT) or 0.0)
        dd_pause = int(getattr(config, "RISK_GOVERNOR_DRAWDOWN_PAUSE_SECONDS", config.LOSS_STREAK_COOLDOWN_SECONDS) or 0)
        if dd_limit > 0:
            drawdown_pct = self._risk_drawdown_percent()
            if drawdown_pct <= -abs(dd_limit):
                self._risk_trigger_pause(
                    reason="daily_drawdown",
                    detail=f"drawdown={drawdown_pct:.2f}% limit={-abs(dd_limit):.2f}% pnl_usd={pnl_usd:.2f}",
                    pause_seconds=dd_pause,
                )

    @staticmethod
    def _close_reason_matches(reason: str, patterns: list[str]) -> bool:
        key = str(reason or "").strip().upper()
        if not key:
            return False
        for raw in patterns:
            p = str(raw or "").strip().upper()
            if not p:
                continue
            if key == p or key.startswith(f"{p}:"):
                return True
        return False

    def _apply_token_cooldown_after_close(
        self,
        *,
        token_address: str,
        symbol: str,
        close_reason: str,
        pnl_usd: float,
        pnl_percent: float,
    ) -> None:
        normalized = normalize_address(token_address)
        if not normalized:
            return
        now_ts = datetime.now(timezone.utc).timestamp()
        base = max(0, int(getattr(config, "MAX_TOKEN_COOLDOWN_SECONDS", 0) or 0))
        dynamic_enabled = bool(getattr(config, "AUTO_TRADE_TOKEN_COOLDOWN_DYNAMIC_ENABLED", True))
        step_seconds = max(0, int(getattr(config, "AUTO_TRADE_TOKEN_COOLDOWN_STEP_SECONDS", 0) or 0))
        max_strikes = max(1, int(getattr(config, "AUTO_TRADE_TOKEN_COOLDOWN_MAX_STRIKES", 1) or 1))
        recovery_step = max(1, int(getattr(config, "AUTO_TRADE_TOKEN_COOLDOWN_RECOVERY_STEP", 1) or 1))
        cap_seconds = max(base, int(getattr(config, "AUTO_TRADE_TOKEN_COOLDOWN_MAX_SECONDS", base) or base))
        patterns = list(getattr(config, "AUTO_TRADE_TOKEN_COOLDOWN_ESCALATE_REASONS", []) or [])

        strikes_prev = int(self.token_cooldown_strikes.get(normalized, 0) or 0)
        strikes_next = strikes_prev
        should_escalate = bool(pnl_usd < 0.0) and self._close_reason_matches(close_reason, patterns)
        if dynamic_enabled and should_escalate:
            strikes_next = min(max_strikes, strikes_prev + 1)
        elif dynamic_enabled:
            strikes_next = max(0, strikes_prev - recovery_step)
        if strikes_next > 0:
            self.token_cooldown_strikes[normalized] = strikes_next
        else:
            self.token_cooldown_strikes.pop(normalized, None)

        cooldown_seconds = base
        if dynamic_enabled and base > 0 and step_seconds > 0:
            cooldown_seconds = min(cap_seconds, base + (strikes_next * step_seconds))
        if cooldown_seconds > 0:
            self.token_cooldowns[normalized] = now_ts + cooldown_seconds
            logger.info(
                "TOKEN_COOLDOWN token=%s reason=%s pnl=$%.4f cooldown=%ss strikes=%s->%s dynamic=%s",
                normalized,
                close_reason,
                float(pnl_usd),
                int(cooldown_seconds),
                strikes_prev,
                strikes_next,
                dynamic_enabled,
            )

        if bool(getattr(config, "EXTREME_SL_GUARD_ENABLED", True)):
            reason_key = str(close_reason or "").strip().upper()
            extreme_reasons = tuple(
                str(x or "").strip().upper()
                for x in (getattr(config, "EXTREME_SL_GUARD_REASONS", ["SL"]) or ["SL"])
                if str(x or "").strip()
            )
            extreme_threshold = float(getattr(config, "EXTREME_SL_PNL_PERCENT_THRESHOLD", -20.0) or -20.0)
            if any(reason_key == r or reason_key.startswith(f"{r}:") for r in extreme_reasons) and float(
                pnl_percent
            ) <= float(extreme_threshold):
                token_block_seconds = max(
                    0,
                    int(getattr(config, "EXTREME_SL_TOKEN_COOLDOWN_SECONDS", 5400) or 5400),
                )
                symbol_block_seconds = max(
                    0,
                    int(getattr(config, "EXTREME_SL_SYMBOL_COOLDOWN_SECONDS", 5400) or 5400),
                )
                if token_block_seconds > 0:
                    prev_until = float(self.token_cooldowns.get(normalized, 0.0) or 0.0)
                    self.token_cooldowns[normalized] = max(prev_until, now_ts + float(token_block_seconds))
                if symbol_block_seconds > 0:
                    self._set_symbol_cooldown(symbol, symbol_block_seconds, "extreme_sl")
                logger.warning(
                    "EXTREME_SL_GUARD symbol=%s token=%s reason=%s pnl=%.2f%% token_cd=%ss symbol_cd=%ss",
                    symbol,
                    normalized,
                    reason_key,
                    float(pnl_percent),
                    int(token_block_seconds),
                    int(symbol_block_seconds),
                )

    @staticmethod
    def _symbol_key(symbol: str) -> str:
        return str(symbol or "").strip().upper()

    def _symbol_cooldown_left_seconds(self, symbol: str) -> int:
        key = self._symbol_key(symbol)
        if not key:
            return 0
        until = float(self.symbol_cooldowns.get(key, 0.0) or 0.0)
        now_ts = datetime.now(timezone.utc).timestamp()
        if until <= now_ts:
            self.symbol_cooldowns.pop(key, None)
            return 0
        return int(max(1, until - now_ts))

    def _set_symbol_cooldown(self, symbol: str, seconds: int, reason: str) -> None:
        key = self._symbol_key(symbol)
        sec = max(0, int(seconds))
        if not key or sec <= 0:
            return
        now_ts = datetime.now(timezone.utc).timestamp()
        until = now_ts + sec
        prev_until = float(self.symbol_cooldowns.get(key, 0.0) or 0.0)
        self.symbol_cooldowns[key] = max(prev_until, until)
        logger.info(
            "SYMBOL_COOLDOWN symbol=%s seconds=%s reason=%s",
            key,
            sec,
            reason,
        )

    def _symbol_window_metrics(self, symbol: str) -> dict[str, float | int]:
        key = self._symbol_key(symbol)
        if not key:
            return {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "loss_share": 0.0,
                "sum_pnl": 0.0,
                "avg_pnl": 0.0,
                "loss_streak": 0,
            }
        window_minutes = max(1, int(getattr(config, "SYMBOL_EV_WINDOW_MINUTES", 120) or 120))
        cutoff = datetime.now(timezone.utc).timestamp() - (window_minutes * 60)
        rows = [
            p
            for p in self.closed_positions
            if self._symbol_key(p.symbol) == key and p.closed_at is not None and p.closed_at.timestamp() >= cutoff
        ]
        trades = len(rows)
        if trades <= 0:
            return {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "loss_share": 0.0,
                "sum_pnl": 0.0,
                "avg_pnl": 0.0,
                "loss_streak": 0,
            }
        sum_pnl = sum(float(r.pnl_usd) for r in rows)
        wins = sum(1 for r in rows if self._pnl_outcome(float(r.pnl_usd)) == "win")
        losses = sum(1 for r in rows if self._pnl_outcome(float(r.pnl_usd)) == "loss")
        sorted_rows = sorted(rows, key=lambda r: r.closed_at or datetime.now(timezone.utc), reverse=True)
        loss_streak = 0
        for r in sorted_rows:
            if self._pnl_outcome(float(r.pnl_usd)) == "loss":
                loss_streak += 1
            else:
                break
        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "loss_share": float(losses) / float(max(1, trades)),
            "sum_pnl": float(sum_pnl),
            "avg_pnl": float(sum_pnl) / float(max(1, trades)),
            "loss_streak": int(loss_streak),
        }

    @staticmethod
    def _token_cluster_key(
        *,
        score: int,
        liquidity_usd: float,
        volume_5m: float,
        price_change_5m: float,
        risk_level: str,
    ) -> str:
        s = int(score)
        if s >= 90:
            score_band = "s3"
        elif s >= 80:
            score_band = "s2"
        elif s >= 70:
            score_band = "s1"
        else:
            score_band = "s0"

        liq = float(liquidity_usd)
        if liq >= 500_000:
            liq_band = "l3"
        elif liq >= 120_000:
            liq_band = "l2"
        elif liq >= 35_000:
            liq_band = "l1"
        else:
            liq_band = "l0"

        vol = float(volume_5m)
        if vol >= 12_000:
            vol_band = "v3"
        elif vol >= 3_500:
            vol_band = "v2"
        elif vol >= 800:
            vol_band = "v1"
        else:
            vol_band = "v0"

        mom_abs = abs(float(price_change_5m))
        if mom_abs >= 8.0:
            mom_band = "m3"
        elif mom_abs >= 3.0:
            mom_band = "m2"
        elif mom_abs >= 1.0:
            mom_band = "m1"
        else:
            mom_band = "m0"

        risk = str(risk_level or "MEDIUM").strip().upper()
        risk_band = {
            "LOW": "r0",
            "MEDIUM": "r1",
            "HIGH": "r2",
        }.get(risk, "r1")
        return f"{score_band}|{liq_band}|{vol_band}|{mom_band}|{risk_band}"

    @staticmethod
    def _ev_stats_from_pnls(pnls: list[float]) -> dict[str, float]:
        rows = [float(x or 0.0) for x in (pnls or [])]
        if not rows:
            return {
                "samples": 0.0,
                "win_rate": 0.0,
                "avg_win_usd": 0.0,
                "avg_loss_usd": 0.0,
                "avg_net_usd": 0.0,
                "loss_share": 0.0,
            }
        wins = [x for x in rows if x > 0.0]
        losses = [abs(x) for x in rows if x <= 0.0]
        avg_win = float(sum(wins) / max(1, len(wins)))
        avg_loss = float(sum(losses) / max(1, len(losses)))
        win_rate = float(len(wins)) / float(max(1, len(rows)))
        avg_net = float(sum(rows) / float(max(1, len(rows))))
        loss_share = float(len(losses)) / float(max(1, len(rows)))
        return {
            "samples": float(len(rows)),
            "win_rate": float(win_rate),
            "avg_win_usd": float(avg_win),
            "avg_loss_usd": float(avg_loss),
            "avg_net_usd": float(avg_net),
            "loss_share": float(loss_share),
        }

    def _closed_rows_window(self, *, window_minutes: int) -> list[PaperPosition]:
        mins = max(5, int(window_minutes))
        cutoff_ts = datetime.now(timezone.utc).timestamp() - (mins * 60)
        return [
            p
            for p in self.closed_positions
            if p.closed_at is not None and p.closed_at.timestamp() >= cutoff_ts
        ]

    def _cluster_window_metrics(self, cluster_key: str) -> dict[str, float]:
        key = str(cluster_key or "").strip().lower()
        if not key:
            return self._ev_stats_from_pnls([])
        rows = [
            p
            for p in self._closed_rows_window(window_minutes=int(getattr(config, "EV_FIRST_ENTRY_LOCAL_WINDOW_MINUTES", 360) or 360))
            if str(getattr(p, "token_cluster_key", "") or "").strip().lower() == key
        ]
        return self._ev_stats_from_pnls([float(getattr(p, "pnl_usd", 0.0) or 0.0) for p in rows])

    def _symbol_ev_stats(self, symbol: str) -> dict[str, float]:
        key = self._symbol_key(symbol)
        if not key:
            return self._ev_stats_from_pnls([])
        rows = [
            p
            for p in self._closed_rows_window(window_minutes=int(getattr(config, "SYMBOL_EV_WINDOW_MINUTES", 120) or 120))
            if self._symbol_key(p.symbol) == key
        ]
        return self._ev_stats_from_pnls([float(getattr(p, "pnl_usd", 0.0) or 0.0) for p in rows])

    @staticmethod
    def _blend_ev_stats(stats_rows: list[dict[str, float]]) -> dict[str, float]:
        total_w = 0.0
        out = {
            "samples": 0.0,
            "win_rate": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "avg_net_usd": 0.0,
            "loss_share": 0.0,
        }
        for row in stats_rows:
            samples = max(0.0, float((row or {}).get("samples", 0.0) or 0.0))
            if samples <= 0.0:
                continue
            weight_cap = max(1.0, float(getattr(config, "EV_FIRST_ENTRY_DB_LOOKBACK_ROWS", 2500) or 2500) / 20.0)
            w = min(weight_cap, samples)
            total_w += w
            out["samples"] += samples
            out["win_rate"] += w * float((row or {}).get("win_rate", 0.0) or 0.0)
            out["avg_win_usd"] += w * float((row or {}).get("avg_win_usd", 0.0) or 0.0)
            out["avg_loss_usd"] += w * float((row or {}).get("avg_loss_usd", 0.0) or 0.0)
            out["avg_net_usd"] += w * float((row or {}).get("avg_net_usd", 0.0) or 0.0)
            out["loss_share"] += w * float((row or {}).get("loss_share", 0.0) or 0.0)
        if total_w <= 0.0:
            return out
        out["win_rate"] /= total_w
        out["avg_win_usd"] /= total_w
        out["avg_loss_usd"] /= total_w
        out["avg_net_usd"] /= total_w
        out["loss_share"] /= total_w
        return out

    def _ev_db_stats(self, *, market_mode: str, entry_tier: str, score: int) -> dict[str, float]:
        path = str(self._ev_db_path or "").strip()
        if (not path) or (not os.path.exists(path)):
            return self._ev_stats_from_pnls([])
        width = max(5, int(getattr(config, "EV_FIRST_ENTRY_DB_BUCKET_SCORE_WIDTH", 10) or 10))
        score_mid = int(score)
        lo = max(0, score_mid - width)
        hi = min(100, score_mid + width)
        key = f"{str(market_mode).upper()}|{str(entry_tier).upper()}|{lo}|{hi}"
        now_ts = time.time()
        cached = self._ev_db_cache.get(key)
        if cached and float(cached[0]) > now_ts:
            return dict(cached[1])

        rows_limit = max(100, int(getattr(config, "EV_FIRST_ENTRY_DB_LOOKBACK_ROWS", 2500) or 2500))
        p = str(market_mode or "").strip().upper()
        t = str(entry_tier or "").strip().upper()
        rows: list[tuple[float]] = []
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(path, timeout=2.0)
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT pnl_usd
                FROM closed_trades
                WHERE market_mode = ?
                  AND entry_tier = ?
                  AND score BETWEEN ? AND ?
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (p, t, int(lo), int(hi), int(rows_limit)),
            ).fetchall()
            if len(rows) < max(8, int(getattr(config, "EV_FIRST_ENTRY_MIN_SAMPLES", 24) or 24) // 2):
                rows = cur.execute(
                    """
                    SELECT pnl_usd
                    FROM closed_trades
                    WHERE market_mode = ?
                      AND entry_tier = ?
                    ORDER BY rowid DESC
                    LIMIT ?
                    """,
                    (p, t, int(rows_limit)),
                ).fetchall()
        except Exception:
            rows = []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        stats = self._ev_stats_from_pnls([_safe_float(r[0], 0.0) for r in rows])
        ttl = max(10, int(getattr(config, "EV_FIRST_ENTRY_DB_CACHE_TTL_SECONDS", 90) or 90))
        self._ev_db_cache[key] = (now_ts + float(ttl), dict(stats))
        return stats

    @staticmethod
    def _deterministic_probability(seed: str) -> float:
        digest = hashlib.sha1(str(seed or "").encode("utf-8")).hexdigest()
        top = int(digest[:8], 16)
        return float(top % 10_000) / 10_000.0

    def _passes_probability_gate(self, *, seed: str, probability: float) -> bool:
        p = max(0.0, min(1.0, float(probability)))
        if p >= 1.0:
            return True
        if p <= 0.0:
            return False
        return self._deterministic_probability(seed) <= p

    def _entry_idle_seconds(self) -> float:
        if not self.trade_open_timestamps:
            return 10**9
        now_ts = datetime.now(timezone.utc).timestamp()
        last_ts = max(float(ts) for ts in self.trade_open_timestamps)
        return max(0.0, now_ts - last_ts)

    def _prune_core_probe_window(self, window_seconds: int | None = None) -> None:
        ts = list(getattr(self, "_core_probe_open_timestamps", []) or [])
        window = max(
            60,
            int(
                window_seconds
                or getattr(config, "EV_FIRST_ENTRY_CORE_PROBE_WINDOW_SECONDS", 900)
                or 900
            ),
        )
        now_ts = datetime.now(timezone.utc).timestamp()
        self._core_probe_open_timestamps = [float(v) for v in ts if (now_ts - float(v)) <= float(window)]

    def _core_probe_budget_available(self) -> tuple[bool, int, int]:
        if not bool(getattr(config, "EV_FIRST_ENTRY_CORE_PROBE_ENABLED", True)):
            return False, 0, 0
        max_opens = max(0, int(getattr(config, "EV_FIRST_ENTRY_CORE_PROBE_MAX_OPENS", 1) or 1))
        if max_opens <= 0:
            return False, 0, max_opens
        window = max(60, int(getattr(config, "EV_FIRST_ENTRY_CORE_PROBE_WINDOW_SECONDS", 900) or 900))
        self._prune_core_probe_window(window)
        used = len(self._core_probe_open_timestamps)
        return used < max_opens, used, max_opens

    def _record_core_probe_open(self) -> None:
        ts = list(getattr(self, "_core_probe_open_timestamps", []) or [])
        ts.append(datetime.now(timezone.utc).timestamp())
        self._core_probe_open_timestamps = ts
        self._prune_core_probe_window()

    def _anti_choke_active(self) -> bool:
        if not bool(getattr(config, "ANTI_CHOKE_ENABLED", True)):
            return False
        idle_threshold = max(30, int(getattr(config, "ANTI_CHOKE_IDLE_SECONDS", 1800) or 1800))
        return self._entry_idle_seconds() >= float(idle_threshold)

    def _session_profit_lock_after_close(self, *, close_reason: str, pnl_usd: float) -> None:
        if not bool(getattr(config, "SESSION_PROFIT_LOCK_ENABLED", False)):
            return
        now_ts = datetime.now(timezone.utc).timestamp()
        realized_now = float(self.realized_pnl_usd or 0.0)
        equity_now = float(self._equity_usd())
        equity_pnl_now = float(equity_now - float(self.initial_balance_usd or 0.0))
        use_equity_guard = bool(getattr(config, "SESSION_PROFIT_LOCK_USE_EQUITY_GUARD", True))
        metric_now = max(realized_now, equity_pnl_now) if use_equity_guard else realized_now
        self._session_profit_lock_last_metric_usd = metric_now

        prev_peak = float(self._session_peak_realized_pnl_usd or 0.0)
        use_equity_peak = bool(getattr(config, "SESSION_PROFIT_LOCK_USE_EQUITY_FOR_PEAK", False))
        peak_source = metric_now if use_equity_peak else realized_now
        peak = max(prev_peak, peak_source)
        self._session_peak_realized_pnl_usd = peak

        activate = max(0.0, float(getattr(config, "SESSION_PROFIT_LOCK_ACTIVATE_USD", 0.35) or 0.35))
        if peak < activate:
            self._session_profit_lock_armed = True
            self._session_profit_lock_rearm_ready_ts = 0.0
            self._session_profit_lock_rearm_floor_usd = 0.0
            self._session_profit_lock_last_floor_usd = 0.0
            return
        keep_ratio = max(0.0, min(1.0, float(getattr(config, "SESSION_PROFIT_LOCK_KEEP_RATIO", 0.55) or 0.55)))
        min_floor = max(0.0, float(getattr(config, "SESSION_PROFIT_LOCK_MIN_FLOOR_USD", 0.10) or 0.10))
        floor = max(min_floor, peak * keep_ratio)
        self._session_profit_lock_last_floor_usd = floor
        rearm_buffer = max(0.0, float(getattr(config, "SESSION_PROFIT_LOCK_REARM_BUFFER_USD", 0.03) or 0.03))
        rearm_floor = max(floor + rearm_buffer, float(self._session_profit_lock_rearm_floor_usd or 0.0))
        rearm_cooldown = max(0, int(getattr(config, "SESSION_PROFIT_LOCK_REARM_COOLDOWN_SECONDS", 300) or 300))

        if not bool(self._session_profit_lock_armed):
            ready_by_time = now_ts >= float(self._session_profit_lock_rearm_ready_ts or 0.0)
            if ready_by_time and metric_now >= rearm_floor:
                self._session_profit_lock_armed = True
                self._session_profit_lock_rearm_ready_ts = 0.0
                self._session_profit_lock_rearm_floor_usd = 0.0
            else:
                return

        if metric_now >= floor:
            return
        min_retrigger_seconds = max(
            0,
            int(getattr(config, "SESSION_PROFIT_LOCK_MIN_RETRIGGER_SECONDS", 300) or 300),
        )
        if (now_ts - float(self._session_profit_lock_last_trigger_ts or 0.0)) < float(min_retrigger_seconds):
            return

        pause_seconds = max(0, int(getattr(config, "SESSION_PROFIT_LOCK_PAUSE_SECONDS", 900) or 900))
        self._risk_trigger_pause(
            reason="session_profit_lock",
            detail=(
                f"reason={close_reason} pnl_usd={pnl_usd:.4f} "
                f"realized={realized_now:.4f} equity_pnl={equity_pnl_now:.4f} metric={metric_now:.4f} "
                f"peak={peak:.4f} floor={floor:.4f} rearm_floor={rearm_floor:.4f}"
            ),
            pause_seconds=pause_seconds,
        )
        self._session_profit_lock_last_trigger_ts = now_ts
        self._session_profit_lock_armed = False
        self._session_profit_lock_rearm_ready_ts = now_ts + float(rearm_cooldown)
        self._session_profit_lock_rearm_floor_usd = rearm_floor

    def _risk_governor_block_reason(self) -> str | None:
        if not bool(getattr(config, "RISK_GOVERNOR_ENABLED", True)):
            return None
        now_ts = datetime.now(timezone.utc).timestamp()
        pause_until = float(self.trading_pause_until_ts or 0.0)
        if pause_until > now_ts:
            rem = int(max(1, pause_until - now_ts))
            pause_reason = str(self._last_pause_reason or "").strip()
            if pause_reason:
                return f"governor_pause {rem}s reason={pause_reason}"
            return f"governor_pause {rem}s"

        max_streak = int(getattr(config, "RISK_GOVERNOR_MAX_LOSS_STREAK", config.MAX_CONSECUTIVE_LOSSES) or 0)
        hard_block_on_streak = bool(getattr(config, "RISK_GOVERNOR_HARD_BLOCK_ON_STREAK", False))
        if hard_block_on_streak and max_streak > 0 and int(self.current_loss_streak) >= max_streak:
            return f"governor_loss_streak {self.current_loss_streak}/{max_streak}"

        dd_limit = float(getattr(config, "RISK_GOVERNOR_DRAWDOWN_LIMIT_PERCENT", config.DAILY_MAX_DRAWDOWN_PERCENT) or 0.0)
        if dd_limit > 0:
            drawdown_pct = self._risk_drawdown_percent()
            if drawdown_pct <= -abs(dd_limit):
                return f"governor_drawdown {drawdown_pct:.2f}% <= {-abs(dd_limit):.2f}%"
        return None

    def _policy_block_detail(self) -> str | None:
        mode = str(self.data_policy_mode or "OK").strip().upper() or "OK"
        reason = str(self.data_policy_reason or "").strip()
        if mode == "BLOCKED":
            return f"data_policy:{mode}:{reason}"
        if bool(getattr(config, "DATA_POLICY_HARD_BLOCK_ENABLED", False)) and mode != "OK":
            return f"data_policy_hard_block:{mode}:{reason}"
        if (not bool(getattr(config, "V2_POLICY_ROUTER_ENABLED", False))) and mode in {"DEGRADED", "FAIL_CLOSED"}:
            return f"data_policy_legacy_block:{mode}:{reason}"
        return None

    def can_open_trade(self) -> bool:
        policy_block = self._policy_block_detail()
        if policy_block is not None:
            return False
        self._refresh_daily_window()
        block_reason = self._risk_governor_block_reason()
        if block_reason is not None:
            now_ts = datetime.now(timezone.utc).timestamp()
            interval = int(getattr(config, "RISK_GOVERNOR_LOG_INTERVAL_SECONDS", 30) or 30)
            if (now_ts - self._last_guard_log_ts) >= interval:
                self._last_guard_log_ts = now_ts
                logger.warning("RISK_GOVERNOR block reason=%s", block_reason)
            return False
        if self.emergency_halt_reason:
            return False
        if self._kill_switch_active():
            logger.warning("KILL_SWITCH active path=%s", config.KILL_SWITCH_FILE)
            return False
        self._prune_hourly_window()
        max_buys_per_hour = int(getattr(config, "MAX_BUYS_PER_HOUR", config.MAX_TRADES_PER_HOUR))
        if max_buys_per_hour > 0 and len(self.trade_open_timestamps) >= max_buys_per_hour:
            return False
        self._prune_daily_tx_window()
        max_tx_per_day = int(getattr(config, "MAX_TX_PER_DAY", 0) or 0)
        if max_tx_per_day > 0 and len(self.tx_event_timestamps) >= max_tx_per_day:
            return False
        max_open = int(config.MAX_OPEN_TRADES)
        if max_open > 0 and len(self.open_positions) >= max_open:
            return False
        if not config.AUTO_TRADE_ENABLED:
            return False
        if self._is_live_mode() and self.live_executor is None:
            return False
        if bool(getattr(config, "LOW_BALANCE_GUARD_ENABLED", True)):
            min_available = max(0.1, float(config.AUTO_STOP_MIN_AVAILABLE_USD))
            available = self._available_balance_usd()
            if available < min_available:
                now_ts = datetime.now(timezone.utc).timestamp()
                if now_ts - self._last_guard_log_ts >= 30:
                    self._last_guard_log_ts = now_ts
                    logger.warning(
                        "KILL_SWITCH reason=LOW_BALANCE_GUARD available=$%.2f min=$%.2f open=%s",
                        available,
                        min_available,
                        len(self.open_positions),
                    )
                return False
        return True

    def _cannot_open_trade_detail(self) -> str:
        # Keep this in sync with can_open_trade() so "disabled_or_limits" logs
        # tell the operator exactly which guard/limit is active.
        policy_block = self._policy_block_detail()
        if policy_block is not None:
            return policy_block
        self._refresh_daily_window()
        block_reason = self._risk_governor_block_reason()
        if block_reason is not None:
            return block_reason
        if self.emergency_halt_reason:
            return f"emergency_halt:{self.emergency_halt_reason}"
        if self._kill_switch_active():
            return "kill_switch_file"

        self._prune_hourly_window()
        max_buys_per_hour = int(getattr(config, "MAX_BUYS_PER_HOUR", config.MAX_TRADES_PER_HOUR))
        if max_buys_per_hour > 0 and len(self.trade_open_timestamps) >= max_buys_per_hour:
            return f"max_buys_per_hour {len(self.trade_open_timestamps)}/{max_buys_per_hour}"

        self._prune_daily_tx_window()
        max_tx_per_day = int(getattr(config, "MAX_TX_PER_DAY", 0) or 0)
        if max_tx_per_day > 0 and len(self.tx_event_timestamps) >= max_tx_per_day:
            return f"max_tx_per_day {len(self.tx_event_timestamps)}/{max_tx_per_day}"

        max_open = int(config.MAX_OPEN_TRADES)
        if max_open > 0 and len(self.open_positions) >= max_open:
            return f"max_open_trades {len(self.open_positions)}/{max_open}"

        if not config.AUTO_TRADE_ENABLED:
            return "AUTO_TRADE_ENABLED=false"

        if self._is_live_mode() and self.live_executor is None:
            return "live_executor_unavailable"

        if bool(getattr(config, "LOW_BALANCE_GUARD_ENABLED", True)):
            min_available = max(0.1, float(config.AUTO_STOP_MIN_AVAILABLE_USD))
            available = self._available_balance_usd()
            if available < min_available:
                return f"low_balance_guard available=${available:.2f} min=${min_available:.2f}"

        return "unknown"

    def set_data_policy(self, mode: str, reason: str = "") -> None:
        normalized = str(mode or "OK").strip().upper()
        if normalized not in {"OK", "DEGRADED", "FAIL_CLOSED", "LIMITED", "BLOCKED"}:
            normalized = "DEGRADED"
        if normalized == self.data_policy_mode and str(reason or "") == self.data_policy_reason:
            return
        logger.warning(
            "DATA_POLICY mode=%s reason=%s prev_mode=%s prev_reason=%s",
            normalized,
            reason,
            self.data_policy_mode,
            self.data_policy_reason,
        )
        self.data_policy_mode = normalized
        self.data_policy_reason = str(reason or "")

    def _recovery_record_attempt(self, address: str) -> int:
        key = normalize_address(address)
        now_ts = datetime.now(timezone.utc).timestamp()
        self._recovery_last_attempt_ts[key] = now_ts
        self._recovery_attempts[key] = int(self._recovery_attempts.get(key, 0)) + 1
        return int(self._recovery_attempts.get(key, 0))

    def _recovery_attempt_allowed(self, address: str) -> bool:
        key = normalize_address(address)
        now_ts = datetime.now(timezone.utc).timestamp()
        min_interval = int(getattr(config, "RECOVERY_ATTEMPT_INTERVAL_SECONDS", 30) or 30)
        max_attempts = int(getattr(config, "RECOVERY_MAX_ATTEMPTS", 8) or 8)
        attempts = int(self._recovery_attempts.get(key, 0))
        if attempts >= max_attempts:
            return False
        last_ts = float(self._recovery_last_attempt_ts.get(key, 0.0))
        if last_ts > 0 and (now_ts - last_ts) < min_interval:
            return False
        return True

    def _recovery_clear_tracking(self, address: str) -> None:
        key = normalize_address(address)
        self._recovery_attempts.pop(key, None)
        self._recovery_last_attempt_ts.pop(key, None)

    def _recovery_forget_address(self, address: str) -> None:
        key = normalize_address(address)
        if not key:
            return
        self.recovery_queue = [a for a in self.recovery_queue if normalize_address(a) != key]

    @staticmethod
    def _position_key(token_address: str) -> str:
        return normalize_address(token_address)

    def _set_open_position(self, position: PaperPosition) -> None:
        key = self._position_key(position.token_address)
        if not key:
            return
        # Keep canonical (normalized) key format across runtime and persisted state.
        self.open_positions[key] = position

    def _pop_open_position(self, token_address: str) -> None:
        key = self._position_key(token_address)
        if key:
            self.open_positions.pop(key, None)
            self.price_guard_pending.pop(key, None)
        # Backward-compatible cleanup for any stale pre-fix raw keys.
        raw = str(token_address or "").strip()
        if raw and raw != key:
            self.open_positions.pop(raw, None)
            self.price_guard_pending.pop(raw, None)

    def _bump_skip_reason(self, reason: str) -> None:
        key = str(reason or "unknown").strip().lower() or "unknown"
        self._skip_reason_counts_window[key] = int(self._skip_reason_counts_window.get(key, 0)) + 1
        ctx = dict(self._active_trade_decision_context or {})
        if ctx:
            payload = {
                "ts": time.time(),
                "decision_stage": "plan_trade",
                "decision": "skip",
                "reason": key,
                **ctx,
            }
            self._write_trade_decision(payload)
            self._active_trade_decision_context = None

    def pop_skip_reason_counts_window(self) -> dict[str, int]:
        out = dict(self._skip_reason_counts_window)
        self._skip_reason_counts_window = {}
        return out

    @staticmethod
    def _decision_run_tag() -> str:
        run_tag = str(os.getenv("RUN_TAG", "") or "").strip()
        if run_tag:
            return run_tag
        inst = str(os.getenv("BOT_INSTANCE_ID", "") or "").strip()
        if inst:
            return inst
        return "single_or_other"

    def _write_trade_decision(self, event: dict[str, Any]) -> None:
        if not self.trade_decisions_log_enabled:
            return
        try:
            payload = trade_decision_event(dict(event or {}), run_tag=self._decision_run_tag())
            os.makedirs(os.path.dirname(self.trade_decisions_log_file) or ".", exist_ok=True)
            with open(self.trade_decisions_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=False) + "\n")
        except Exception:
            logger.exception("TRADE_DECISION log write failed")

    @staticmethod
    def _is_live_mode() -> bool:
        return bool(config.AUTO_TRADE_ENABLED) and not bool(config.AUTO_TRADE_PAPER)

    @staticmethod
    def _short_error_text(value: Any, limit: int = 140) -> str:
        text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    @staticmethod
    def _pnl_outcome(pnl_usd: float) -> str:
        """Classify trade outcome with a configurable break-even epsilon in USD."""
        try:
            eps = max(0.0, float(getattr(config, "PNL_BREAKEVEN_EPSILON_USD", 0.0) or 0.0))
        except Exception:
            eps = 0.0
        p = float(pnl_usd or 0.0)
        if p > eps:
            return "win"
        if p < -eps:
            return "loss"
        return "be"

    @staticmethod
    def _quality_size_multiplier(
        score: int,
        liquidity_usd: float,
        volume_5m: float,
        price_change_5m: float,
    ) -> tuple[float, str]:
        """
        Conservative sizing multiplier to reduce exposure on weak/unstable tokens.
        Returns (multiplier, detail_string).
        """
        # Score factor: reward high score slightly, punish low score more.
        s = int(score)
        if s >= 90:
            score_mult = 1.15
        elif s >= 80:
            score_mult = 1.08
        elif s >= 70:
            score_mult = 1.00
        elif s >= 60:
            score_mult = 0.85
        else:
            score_mult = 0.70

        liq = float(liquidity_usd or 0.0)
        if liq >= 250_000:
            liq_mult = 1.15
        elif liq >= 100_000:
            liq_mult = 1.08
        elif liq >= 25_000:
            liq_mult = 1.00
        elif liq >= 15_000:
            liq_mult = 0.90
        else:
            liq_mult = 0.75

        vol = float(volume_5m or 0.0)
        if vol >= 50_000:
            vol_mult = 1.10
        elif vol >= 10_000:
            vol_mult = 1.05
        elif vol >= 3_000:
            vol_mult = 1.00
        elif vol >= 1_500:
            vol_mult = 0.90
        else:
            vol_mult = 0.80

        chg = abs(float(price_change_5m or 0.0))
        if chg >= 35:
            volat_mult = 0.75
        elif chg >= 25:
            volat_mult = 0.85
        elif chg >= 15:
            volat_mult = 0.95
        elif chg <= 5:
            volat_mult = 1.05
        else:
            volat_mult = 1.00

        mult = float(score_mult * liq_mult * vol_mult * volat_mult)
        mult = max(0.25, min(1.35, mult))
        detail = f"score={score_mult:.2f} liq={liq_mult:.2f} vol={vol_mult:.2f} volat={volat_mult:.2f}"
        return mult, detail

    @staticmethod
    def _quality_entry_adaptation(
        token_data: dict[str, Any],
        *,
        entry_tier: str,
        entry_channel: str,
    ) -> tuple[float, float, str]:
        """
        Convert quality-gate output into direct entry adjustments.
        Returns (edge_threshold_mult, size_mult, detail).
        edge_threshold_mult < 1 relaxes thresholds; >1 tightens thresholds.
        """
        if not bool(getattr(config, "ENTRY_QUALITY_ADAPT_ENABLED", True)):
            return 1.0, 1.0, "off"

        src_mult = max(0.50, min(1.80, _safe_float((token_data or {}).get("_quality_source_mult"), 1.0)))
        ev_signal = _safe_float((token_data or {}).get("_quality_ev_signal"), 0.0)
        ev_signal = max(-0.02, min(0.02, ev_signal))

        edge_mult = 1.0
        size_mult = 1.0
        src_edge_relax = 0.0
        src_edge_tight = 0.0
        src_size_relax = 0.0
        src_size_cut = 0.0
        ev_edge_adj = 0.0
        ev_size_adj = 0.0

        if bool(getattr(config, "ENTRY_QUALITY_SOURCE_EDGE_ADAPT_ENABLED", True)):
            relax_max = max(0.0, min(0.60, _safe_float(getattr(config, "ENTRY_QUALITY_SOURCE_EDGE_RELAX_MAX", 0.16), 0.16)))
            tighten_max = max(0.0, min(0.80, _safe_float(getattr(config, "ENTRY_QUALITY_SOURCE_EDGE_TIGHTEN_MAX", 0.24), 0.24)))
            if src_mult > 1.0:
                src_edge_relax = min(relax_max, (src_mult - 1.0) * 0.45)
                edge_mult *= max(0.40, 1.0 - src_edge_relax)
            elif src_mult < 1.0:
                src_edge_tight = min(tighten_max, (1.0 - src_mult) * 0.65)
                edge_mult *= 1.0 + src_edge_tight
            scale = max(0.0, _safe_float(getattr(config, "ENTRY_QUALITY_EV_SIGNAL_EDGE_SCALE", 18.0), 18.0))
            if scale > 0.0:
                ev_edge_adj = max(-relax_max, min(tighten_max, -ev_signal * scale))
                edge_mult *= max(0.40, 1.0 + ev_edge_adj)

        if bool(getattr(config, "ENTRY_QUALITY_SOURCE_SIZE_ADAPT_ENABLED", True)):
            relax_max = max(0.0, min(0.80, _safe_float(getattr(config, "ENTRY_QUALITY_SOURCE_SIZE_RELAX_MAX", 0.18), 0.18)))
            cut_max = max(0.0, min(0.80, _safe_float(getattr(config, "ENTRY_QUALITY_SOURCE_SIZE_CUT_MAX", 0.24), 0.24)))
            if src_mult > 1.0:
                src_size_relax = min(relax_max, (src_mult - 1.0) * 0.40)
                size_mult *= 1.0 + src_size_relax
            elif src_mult < 1.0:
                src_size_cut = min(cut_max, (1.0 - src_mult) * 0.75)
                size_mult *= max(0.10, 1.0 - src_size_cut)
            scale = max(0.0, _safe_float(getattr(config, "ENTRY_QUALITY_EV_SIGNAL_SIZE_SCALE", 22.0), 22.0))
            if scale > 0.0:
                ev_size_adj = max(-cut_max, min(relax_max, ev_signal * scale))
                size_mult *= max(0.10, 1.0 + ev_size_adj)

        tier = str(entry_tier or "").strip().upper()
        channel = str(entry_channel or "").strip().lower()
        if tier == "A" and channel == "core":
            size_mult *= max(0.10, _safe_float(getattr(config, "ENTRY_QUALITY_CORE_BONUS_SIZE_MULT", 1.04), 1.04))
            edge_mult *= 0.98
        if channel == "explore":
            size_mult *= max(0.05, _safe_float(getattr(config, "ENTRY_QUALITY_EXPLORE_CUT_SIZE_MULT", 0.92), 0.92))
            edge_mult *= 1.04

        edge_mult = max(0.60, min(1.45, float(edge_mult)))
        size_mult = max(0.20, min(1.50, float(size_mult)))
        detail = (
            f"src_mult={src_mult:.2f} ev={ev_signal:.5f} "
            f"edge(src_relax={src_edge_relax:.3f} src_tight={src_edge_tight:.3f} ev_adj={ev_edge_adj:.3f}) "
            f"size(src_relax={src_size_relax:.3f} src_cut={src_size_cut:.3f} ev_adj={ev_size_adj:.3f})"
        )
        return float(edge_mult), float(size_mult), detail

    @staticmethod
    def _current_day_id() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _refresh_daily_window(self) -> None:
        now_ts = datetime.now(timezone.utc).timestamp()
        self.token_cooldowns = {k: v for k, v in self.token_cooldowns.items() if v > now_ts}
        self.symbol_cooldowns = {k: v for k, v in self.symbol_cooldowns.items() if v > now_ts}
        self._prune_symbol_open_window(max(300, int(getattr(config, "SYMBOL_CONCENTRATION_WINDOW_SECONDS", 3600) or 3600)))
        self._prune_daily_tx_window()
        current_day = self._current_day_id()
        if self.day_id == current_day:
            return
        self.day_id = current_day
        self.day_realized_pnl_usd = 0.0
        self.day_start_equity_usd = self._equity_usd()
        self.current_loss_streak = 0
        if bool(getattr(config, "SESSION_PROFIT_LOCK_RESET_ON_NEW_DAY", True)):
            self._session_peak_realized_pnl_usd = max(0.0, float(self.day_realized_pnl_usd or 0.0))
            self._session_profit_lock_armed = True
            self._session_profit_lock_rearm_ready_ts = 0.0
            self._session_profit_lock_rearm_floor_usd = 0.0
            self._session_profit_lock_last_floor_usd = 0.0
            self._session_profit_lock_last_metric_usd = 0.0

    @staticmethod
    def _kill_switch_active() -> bool:
        try:
            return bool(config.KILL_SWITCH_FILE) and os.path.exists(config.KILL_SWITCH_FILE)
        except Exception:
            return False

    def _prune_hourly_window(self) -> None:
        now_ts = datetime.now(timezone.utc).timestamp()
        self.trade_open_timestamps = [ts for ts in self.trade_open_timestamps if (now_ts - ts) <= 3600]
        self._prune_core_probe_window()

    def _prune_daily_tx_window(self) -> None:
        now_ts = datetime.now(timezone.utc).timestamp()
        self.tx_event_timestamps = [ts for ts in self.tx_event_timestamps if (now_ts - ts) <= 86400]

    def _record_tx_event(self) -> None:
        self.tx_event_timestamps.append(datetime.now(timezone.utc).timestamp())

    def _equity_usd(self) -> float:
        # In live mode, sizing/risk caps should be based on real on-chain wallet balance,
        # not the persisted paper_state.json balance (which can be stale and cause tiny caps).
        if self._is_live_mode() and self.live_executor is not None:
            try:
                balance_eth = float(self.live_executor.native_balance_eth())
                price_usd = float(self.last_weth_price_usd or 0.0)
                if price_usd <= 100.0:
                    price_usd = float(getattr(config, "WETH_PRICE_FALLBACK_USD", 0.0) or 0.0)
                if price_usd <= 0.0:
                    price_usd = 1.0
                native_usd = max(0.0, balance_eth * price_usd)
                open_positions_usd = 0.0
                for pos in self.open_positions.values():
                    open_positions_usd += max(0.0, float(pos.position_size_usd) + float(pos.pnl_usd))
                return max(0.0, native_usd + open_positions_usd)
            except Exception:
                # Fall back to paper equity calculation below.
                pass

        return self.paper_balance_usd + sum(pos.position_size_usd + pos.pnl_usd for pos in self.open_positions.values())

    def _available_balance_usd(self) -> float:
        if self._is_live_mode():
            if self.live_executor is None:
                return 0.0
            try:
                balance_eth = float(self.live_executor.native_balance_eth())
                # On startup we might not have market data yet. Using 1.0 here causes "low_balance" skips forever.
                price_usd = float(self.last_weth_price_usd or 0.0)
                if price_usd <= 100.0:
                    price_usd = float(getattr(config, "WETH_PRICE_FALLBACK_USD", 0.0) or 0.0)
                if price_usd <= 0:
                    price_usd = 1.0
                return max(0.0, (balance_eth * price_usd) - self._stair_reserved_usd())
            except Exception:
                return 0.0
        reserved = self._stair_reserved_usd()
        available = self.paper_balance_usd - reserved
        if not config.STAIR_STEP_ENABLED:
            gas_buffer = float(config.PAPER_GAS_PER_TX_USD) * 3.0
            available -= gas_buffer
        return max(0.0, available)

    def _honeypot_cache_get(self, token_address: str) -> dict[str, Any] | None:
        if not token_address:
            return None
        key = normalize_address(token_address)
        now_ts = datetime.now(timezone.utc).timestamp()
        ttl = int(config.HONEYPOT_API_CACHE_TTL_SECONDS)
        row = self._honeypot_cache.get(key)
        if not row:
            return None
        ts, payload = row
        if (now_ts - float(ts)) > ttl:
            self._honeypot_cache.pop(key, None)
            return None
        return payload

    def _honeypot_cache_put(self, token_address: str, payload: dict[str, Any]) -> None:
        if not token_address:
            return
        key = normalize_address(token_address)
        self._honeypot_cache[key] = (datetime.now(timezone.utc).timestamp(), payload)

    async def _honeypot_guard_passes(self, token_address: str) -> tuple[bool, str]:
        if not config.HONEYPOT_API_ENABLED:
            return True, "disabled"
        if not config.HONEYPOT_API_URL:
            return True, "no_url"

        cached = self._honeypot_cache_get(token_address)
        if cached is not None:
            is_ok = bool(cached.get("ok", False))
            return is_ok, str(cached.get("detail", "cached"))

        url = config.HONEYPOT_API_URL
        params = {
            "address": token_address,
            "chainID": int(config.LIVE_CHAIN_ID),
        }
        try:
            result = await self._http.get_json(
                url,
                source="honeypot",
                params=params,
                max_attempts=int(getattr(config, "HTTP_RETRY_ATTEMPTS", 3)),
            )
            if not result.ok or not isinstance(result.data, dict):
                detail = f"honeypot_api_status_{result.status}" if int(result.status or 0) > 0 else str(result.error or "honeypot_api_error")
                ok = not bool(config.HONEYPOT_API_FAIL_CLOSED)
                self._honeypot_cache_put(token_address, {"ok": ok, "detail": detail})
                return ok, detail
            payload = result.data
        except Exception as exc:
            detail = f"honeypot_api_error:{exc}"
            ok = not bool(config.HONEYPOT_API_FAIL_CLOSED)
            self._honeypot_cache_put(token_address, {"ok": ok, "detail": detail})
            return ok, detail

        # Parse common honeypot.is fields. We keep it defensive because the response can change.
        hp = payload if isinstance(payload, dict) else {}
        is_honeypot = bool((hp.get("honeypotResult") or {}).get("isHoneypot", False))
        simulation = hp.get("simulationResult") or {}
        buy_tax = float(simulation.get("buyTax") or 0)
        sell_tax = float(simulation.get("sellTax") or 0)
        can_sell = simulation.get("canSell")
        if can_sell is None:
            # Some versions expose it under a different key.
            can_sell = simulation.get("sellSuccess")
        can_sell_bool = True if can_sell is None else bool(can_sell)

        max_buy_tax = float(config.HONEYPOT_MAX_BUY_TAX_PERCENT)
        max_sell_tax = float(config.HONEYPOT_MAX_SELL_TAX_PERCENT)

        ok = True
        reasons: list[str] = []
        if is_honeypot:
            ok = False
            reasons.append("is_honeypot")
        if not can_sell_bool:
            ok = False
            reasons.append("cannot_sell")
        if max_buy_tax > 0 and buy_tax > max_buy_tax:
            ok = False
            reasons.append(f"buy_tax>{max_buy_tax:.1f}%({buy_tax:.1f}%)")
        if max_sell_tax > 0 and sell_tax > max_sell_tax:
            ok = False
            reasons.append(f"sell_tax>{max_sell_tax:.1f}%({sell_tax:.1f}%)")

        detail = ",".join(reasons) if reasons else f"ok buyTax={buy_tax:.1f}% sellTax={sell_tax:.1f}%"
        self._honeypot_cache_put(token_address, {"ok": ok, "detail": detail, "buy_tax": buy_tax, "sell_tax": sell_tax})
        return ok, detail

    @staticmethod
    def _stair_tradable_buffer_usd() -> float:
        configured = max(0.0, float(config.STAIR_STEP_TRADABLE_BUFFER_USD))
        min_trade_with_gas = max(0.0, float(config.PAPER_TRADE_SIZE_MIN_USD)) + (float(config.PAPER_GAS_PER_TX_USD) * 3.0)
        return max(configured, min_trade_with_gas)

    def _stair_reserved_usd(self) -> float:
        # Stair-step reserve is a paper-account concept; never apply it to live wallet balance.
        if self._is_live_mode():
            return 0.0
        if not config.STAIR_STEP_ENABLED:
            return 0.0
        reserve_target = max(0.0, min(self.paper_balance_usd, self.stair_floor_usd))
        max_reservable = max(0.0, self.paper_balance_usd - self._stair_tradable_buffer_usd())
        return max(0.0, min(reserve_target, max_reservable))

    def _sync_stair_state(self) -> None:
        if not config.STAIR_STEP_ENABLED:
            self.stair_floor_usd = 0.0
            self.stair_peak_balance_usd = max(self.paper_balance_usd, float(config.WALLET_BALANCE_USD))
            return

        if self.stair_floor_usd <= 0:
            self.stair_floor_usd = 0.0
        self.stair_floor_usd = min(self.stair_floor_usd, self.paper_balance_usd)
        self.stair_peak_balance_usd = max(self.stair_peak_balance_usd, self.paper_balance_usd)
        self._update_stair_floor()

    def _update_stair_floor(self) -> None:
        if not config.STAIR_STEP_ENABLED:
            return
        step_size = float(config.STAIR_STEP_SIZE_USD)
        if step_size <= 0:
            return
        start_balance = float(config.STAIR_STEP_START_BALANCE_USD)
        if start_balance <= 0:
            start_balance = 0.0
        start_balance = max(0.0, start_balance)

        self.stair_peak_balance_usd = max(self.stair_peak_balance_usd, self.paper_balance_usd)
        if self.stair_peak_balance_usd < start_balance:
            return
        levels = math.floor((self.stair_peak_balance_usd - start_balance) / step_size)
        milestone_floor = start_balance + (levels * step_size)
        new_floor = max(self.stair_floor_usd, milestone_floor)
        max_reservable = max(0.0, self.paper_balance_usd - self._stair_tradable_buffer_usd())
        new_floor = min(new_floor, max_reservable)
        if new_floor > self.stair_floor_usd:
            self.stair_floor_usd = min(new_floor, self.paper_balance_usd)
            logger.info(
                "STAIR_STEP floor_up new_floor=$%.2f peak=$%.2f step=$%.2f buffer=$%.2f",
                self.stair_floor_usd,
                self.stair_peak_balance_usd,
                step_size,
                self._stair_tradable_buffer_usd(),
            )

    async def plan_batch(self, candidates: list[tuple[dict[str, Any], dict[str, Any]]]) -> int:
        if not candidates:
            return 0

        if await self._check_live_profit_stop():
            return 0

        # only BUY recommendations above configured threshold
        eligible = [
            (token, score)
            for token, score in candidates
            if str(score.get("recommendation", "SKIP")) == "BUY"
            and int(score.get("score", 0)) >= int(config.MIN_TOKEN_SCORE)
        ]
        if not eligible:
            return 0

        mode = str(config.AUTO_TRADE_ENTRY_MODE or "single").lower()
        if mode not in {"single", "all", "top_n"}:
            mode = "single"

        # Prefer the highest score, but break ties by liquidity/volume to avoid selecting thin pools
        # when multiple candidates have similar scores.
        eligible.sort(
            key=lambda item: (
                int(item[1].get("score", 0)),
                float(item[0].get("liquidity") or 0),
                float(item[0].get("volume_5m") or 0),
            ),
            reverse=True,
        )
        if mode == "single":
            selected = eligible[:1]
        elif mode == "top_n":
            n = max(1, int(config.AUTO_TRADE_TOP_N or 1))
            selected = eligible[:n]
        else:
            selected = eligible

        opened = 0
        for token_data, score_data in selected:
            pos = await self.plan_trade(token_data, score_data)
            if pos:
                opened += 1
        logger.info(
            "AutoTrade batch mode=%s candidates=%s eligible=%s opened=%s",
            mode,
            len(candidates),
            len(eligible),
            opened,
        )
        return opened

    async def _check_live_profit_stop(self) -> bool:
        target = float(getattr(config, "LIVE_STOP_AFTER_PROFIT_USD", 0.0) or 0.0)
        if target <= 0:
            return False
        if not self._is_live_mode() or self.live_executor is None:
            return False

        # Initialize baseline when missing.
        if self.live_start_ts <= 0 or self.live_start_balance_usd <= 0:
            try:
                bal_eth = float(self.live_executor.native_balance_eth())
            except Exception:
                bal_eth = 0.0
            weth_price = await self._resolve_weth_price_usd({})
            bal_usd = bal_eth * float(weth_price or 0.0)
            self.live_start_ts = datetime.now(timezone.utc).timestamp()
            self.live_start_balance_eth = float(bal_eth)
            self.live_start_balance_usd = float(bal_usd)
            self._save_state()
            logger.info(
                "LIVE_SESSION baseline set eth=%.8f usd=%.2f target_profit=$%.2f",
                self.live_start_balance_eth,
                self.live_start_balance_usd,
                target,
            )

        if self.live_start_balance_usd <= 0:
            return False

        try:
            cur_eth = float(self.live_executor.native_balance_eth())
        except Exception:
            return False
        weth_price = await self._resolve_weth_price_usd({})
        cur_usd = cur_eth * float(weth_price or 0.0)
        profit = cur_usd - float(self.live_start_balance_usd)
        if profit >= target:
            self._activate_emergency_halt(
                "LIVE_PROFIT_TARGET",
                f"profit_usd=${profit:.2f} target=${target:.2f} start=${self.live_start_balance_usd:.2f} cur=${cur_usd:.2f}",
            )
            logger.warning(
                "LIVE_PROFIT_TARGET hit profit=$%.2f target=$%.2f start=$%.2f cur=$%.2f",
                profit,
                target,
                self.live_start_balance_usd,
                cur_usd,
            )
            return True
        return False

    def _pre_buy_invariants(self, token_address: str, spend_eth: float) -> tuple[bool, str]:
        addr = normalize_address(token_address)
        if not addr or not bool(ADDRESS_RE.match(addr)):
            return False, "invalid_normalized_address"

        policy_block = self._policy_block_detail()
        if policy_block is not None:
            return False, policy_block

        try:
            slippage_bps = int(config.LIVE_SLIPPAGE_BPS)
        except Exception:
            slippage_bps = 0
        if slippage_bps < 1 or slippage_bps > 5000:
            return False, f"slippage_out_of_range:{slippage_bps}"

        try:
            gas_cap_gwei = float(config.LIVE_MAX_GAS_GWEI)
        except Exception:
            gas_cap_gwei = 0.0
        if gas_cap_gwei <= 0 or gas_cap_gwei > 500:
            return False, f"gas_cap_out_of_range:{gas_cap_gwei}"

        if not self.can_open_trade():
            return False, self._cannot_open_trade_detail()

        if float(spend_eth) <= 0:
            return False, "invalid_spend_eth"

        return True, "ok"

    def _pre_sell_invariants(self, position: PaperPosition) -> tuple[bool, str]:
        addr = normalize_address(position.token_address)
        if not addr or not bool(ADDRESS_RE.match(addr)):
            return False, "invalid_normalized_address"
        if int(position.token_amount_raw or 0) <= 0:
            return False, "invalid_token_amount_raw"
        try:
            gas_cap_gwei = float(config.LIVE_MAX_GAS_GWEI)
        except Exception:
            gas_cap_gwei = 0.0
        if gas_cap_gwei <= 0 or gas_cap_gwei > 500:
            return False, f"gas_cap_out_of_range:{gas_cap_gwei}"
        return True, "ok"

    async def plan_trade(self, token_data: dict[str, Any], score_data: dict[str, Any]) -> PaperPosition | None:
        symbol = str(token_data.get("symbol", "N/A"))
        candidate_id = str(token_data.get("_candidate_id", "") or "").strip()
        ctx_token_address = normalize_address(str(token_data.get("address", "") or ""))
        ctx_entry_tier = str(token_data.get("_entry_tier", "") or "").strip().upper()
        ctx_market_mode = str(token_data.get("_regime_name", "") or "").strip().upper()
        ctx_entry_channel = str(token_data.get("_entry_channel", "") or "").strip().lower()
        if ctx_market_mode in {"GREEN", "YELLOW", "RED"}:
            self._last_market_mode_seen = ctx_market_mode
        self._active_trade_decision_context = {
            "candidate_id": candidate_id,
            "token_address": ctx_token_address,
            "symbol": symbol,
            "score": int(score_data.get("score", 0) or 0),
            "entry_tier": ctx_entry_tier,
            "entry_channel": ctx_entry_channel,
            "market_mode": ctx_market_mode,
            "source": str(token_data.get("source", "") or ""),
        }
        symbol_upper = symbol.strip().upper()
        excluded_symbols = set(getattr(config, "AUTO_TRADE_EXCLUDED_SYMBOLS", []) or [])
        if symbol_upper and symbol_upper in excluded_symbols:
            self._bump_skip_reason("excluded_symbol")
            logger.info("AutoTrade skip token=%s reason=excluded_symbol", symbol)
            return None
        if not self.can_open_trade():
            self._bump_skip_reason("disabled_or_limits")
            logger.info("AutoTrade skip token=%s reason=disabled_or_limits detail=%s", symbol, self._cannot_open_trade_detail())
            return None
        if self._kill_switch_active():
            self._write_trade_decision(
                {
                    "ts": time.time(),
                    "decision_stage": "plan_trade",
                    "decision": "skip",
                    "reason": "kill_switch_active",
                    **dict(self._active_trade_decision_context or {}),
                }
            )
            self._active_trade_decision_context = None
            logger.warning("KILL_SWITCH active path=%s", config.KILL_SWITCH_FILE)
            return None

        recommendation = str(score_data.get("recommendation", "SKIP"))
        score = int(score_data.get("score", 0))
        if recommendation != "BUY" or score < int(config.MIN_TOKEN_SCORE):
            self._bump_skip_reason("signal")
            logger.info(
                "AutoTrade skip token=%s reason=signal recommendation=%s score=%s",
                symbol,
                recommendation,
                score,
            )
            return None

        raw_token_address = str(token_data.get("address", "") or "")
        token_address = normalize_address(raw_token_address)
        entry_tier = str(token_data.get("_entry_tier", "") or "").strip().upper()
        if self._active_trade_decision_context is not None:
            self._active_trade_decision_context["token_address"] = token_address
            self._active_trade_decision_context["entry_tier"] = entry_tier
            self._active_trade_decision_context["entry_channel"] = str(
                token_data.get("_entry_channel", "") or ""
            ).strip().lower()
        market_mode = str(token_data.get("_regime_name", "") or "").strip().upper()
        if market_mode in {"GREEN", "YELLOW", "RED"}:
            self._last_market_mode_seen = market_mode
        entry_channel = str(token_data.get("_entry_channel", "core") or "core").strip().lower()
        if entry_channel not in {"core", "explore"}:
            entry_channel = "core"
        concentration_blocked, concentration_detail = self._symbol_concentration_blocked(
            symbol=symbol,
            entry_tier=entry_tier,
        )
        if concentration_blocked:
            self._bump_skip_reason("symbol_concentration")
            logger.info("AutoTrade skip token=%s reason=symbol_concentration detail=%s", symbol, concentration_detail)
            return None
        blocked, blk_reason = self._blacklist_is_blocked(token_address)
        if blocked:
            self._bump_skip_reason("blacklist")
            logger.info("AutoTrade skip token=%s reason=blacklist detail=%s", symbol, self._short_error_text(blk_reason))
            return None
        if not token_address or token_address in self.open_positions:
            self._bump_skip_reason("address_or_duplicate")
            logger.info(
                "AutoTrade skip token=%s reason=address_or_duplicate raw_address=%s normalized=%s source=%s",
                symbol,
                raw_token_address,
                token_address,
                str(token_data.get("source", "-")),
            )
            return None
        cooldown_until = float(self.token_cooldowns.get(token_address, 0.0))
        now_ts = datetime.now(timezone.utc).timestamp()
        if cooldown_until > now_ts:
            self._bump_skip_reason("cooldown")
            logger.info("AutoTrade skip token=%s reason=cooldown_left_%ss", symbol, int(cooldown_until - now_ts))
            return None
        anti_choke = self._anti_choke_active()
        symbol_cd_left = self._symbol_cooldown_left_seconds(symbol_upper)
        if symbol_cd_left > 0:
            bypass_max_seconds = max(
                0,
                int(getattr(config, "ANTI_CHOKE_BYPASS_SYMBOL_COOLDOWN_MAX_SECONDS", 600) or 600),
            )
            allow_symbol_cooldown_bypass = (
                anti_choke
                and bool(getattr(config, "ANTI_CHOKE_ALLOW_SYMBOL_COOLDOWN_BYPASS", True))
                and int(symbol_cd_left) <= int(bypass_max_seconds)
            )
            if allow_symbol_cooldown_bypass:
                logger.info(
                    "AutoTrade bypass token=%s reason=symbol_cooldown anti_choke=true left=%ss",
                    symbol,
                    symbol_cd_left,
                )
            else:
                self._bump_skip_reason("symbol_cooldown")
                logger.info("AutoTrade skip token=%s reason=symbol_cooldown_left_%ss", symbol, symbol_cd_left)
                return None

        symbol_metrics = self._symbol_window_metrics(symbol_upper)
        token_ev_memory = {
            "size_mult": 1.0,
            "edge_mult": 1.0,
            "entry_probability": 1.0,
            "detail": "neutral",
            "bad_ev": False,
            "severe_ev": False,
            "seed": "",
        }
        if bool(getattr(config, "SYMBOL_EV_GUARD_ENABLED", True)):
            trades_w = int(symbol_metrics.get("trades", 0) or 0)
            avg_pnl_w = float(symbol_metrics.get("avg_pnl", 0.0) or 0.0)
            loss_share_w = float(symbol_metrics.get("loss_share", 0.0) or 0.0)
            loss_streak_w = int(symbol_metrics.get("loss_streak", 0) or 0)

            fatigue_cap = int(getattr(config, "SYMBOL_FATIGUE_MAX_TRADES_PER_WINDOW", 0) or 0)
            if fatigue_cap > 0 and trades_w >= fatigue_cap:
                if anti_choke and bool(getattr(config, "ANTI_CHOKE_ALLOW_SYMBOL_FATIGUE_BYPASS", True)):
                    logger.info(
                        "AutoTrade bypass token=%s reason=symbol_fatigue_cap anti_choke=true trades=%s cap=%s",
                        symbol,
                        trades_w,
                        fatigue_cap,
                    )
                else:
                    self._bump_skip_reason("symbol_fatigue_cap")
                    cooldown = int(getattr(config, "SYMBOL_FATIGUE_COOLDOWN_SECONDS", 0) or 0)
                    self._set_symbol_cooldown(symbol_upper, cooldown, "fatigue_cap")
                    logger.info(
                        "AutoTrade skip token=%s reason=symbol_fatigue_cap trades=%s cap=%s",
                        symbol,
                        trades_w,
                        fatigue_cap,
                    )
                    return None

            max_loss_streak = int(getattr(config, "SYMBOL_FATIGUE_MAX_LOSS_STREAK", 0) or 0)
            if max_loss_streak > 0 and loss_streak_w >= max_loss_streak:
                self._bump_skip_reason("symbol_loss_streak")
                cooldown = int(getattr(config, "SYMBOL_FATIGUE_COOLDOWN_SECONDS", 0) or 0)
                self._set_symbol_cooldown(symbol_upper, cooldown, "loss_streak")
                logger.info(
                    "AutoTrade skip token=%s reason=symbol_loss_streak streak=%s max=%s",
                    symbol,
                    loss_streak_w,
                    max_loss_streak,
                )
                return None

            min_trades = int(getattr(config, "SYMBOL_EV_MIN_TRADES", 3) or 3)
            if trades_w >= min_trades:
                bad_ev = (
                    avg_pnl_w < float(getattr(config, "SYMBOL_EV_MIN_AVG_PNL_USD", 0.0005) or 0.0005)
                    or loss_share_w > float(getattr(config, "SYMBOL_EV_MAX_LOSS_SHARE", 0.60) or 0.60)
                )
                if bad_ev:
                    bad_action = str(getattr(config, "SYMBOL_EV_BAD_ACTION", "skip") or "skip").strip().lower()
                    if bool(getattr(config, "SYMBOL_EV_BAD_TO_STRICT_ONLY", True)) and entry_tier != "A":
                        if anti_choke and bool(getattr(config, "ANTI_CHOKE_ALLOW_SYMBOL_STRICT_BYPASS", True)):
                            logger.info(
                                "AutoTrade bypass token=%s reason=symbol_ev_strict_only anti_choke=true tier=%s trades=%s avg=%.5f loss_share=%.2f",
                                symbol,
                                entry_tier or "-",
                                trades_w,
                                avg_pnl_w,
                                loss_share_w,
                            )
                        else:
                            self._bump_skip_reason("symbol_ev_strict_only")
                            logger.info(
                                "AutoTrade skip token=%s reason=symbol_ev_strict_only tier=%s trades=%s avg=%.5f loss_share=%.2f",
                                symbol,
                                entry_tier or "-",
                                trades_w,
                                avg_pnl_w,
                                loss_share_w,
                            )
                            return None
                    if bad_action == "skip":
                        self._bump_skip_reason("symbol_ev_bad")
                        self._set_symbol_cooldown(
                            symbol_upper,
                            int(getattr(config, "SYMBOL_EV_BAD_COOLDOWN_SECONDS", 0) or 0),
                            "bad_ev",
                        )
                        logger.info(
                            "AutoTrade skip token=%s reason=symbol_ev_bad trades=%s avg=%.5f loss_share=%.2f",
                            symbol,
                            trades_w,
                            avg_pnl_w,
                            loss_share_w,
                        )
                        return None

        if bool(getattr(config, "TOKEN_EV_MEMORY_ENABLED", True)):
            token_ev_memory = self._token_ev_memory_adjustments(
                symbol=symbol_upper,
                market_mode=str(token_data.get("_regime_name", "") or ""),
                entry_tier=str(entry_tier or ""),
                candidate_id=candidate_id,
            )
            if self._active_trade_decision_context is not None:
                self._active_trade_decision_context.update(
                    {
                        "token_ev_memory_detail": str(token_ev_memory.get("detail", "")),
                        "token_ev_memory_size_mult": float(token_ev_memory.get("size_mult", 1.0) or 1.0),
                        "token_ev_memory_edge_mult": float(token_ev_memory.get("edge_mult", 1.0) or 1.0),
                        "token_ev_memory_probability": float(token_ev_memory.get("entry_probability", 1.0) or 1.0),
                    }
                )
            prob = float(token_ev_memory.get("entry_probability", 1.0) or 1.0)
            seed = str(token_ev_memory.get("seed", "") or "")
            if prob < 1.0 and (not self._passes_probability_gate(seed=seed, probability=prob)):
                self._bump_skip_reason("token_ev_memory_prob")
                logger.info(
                    "AutoTrade skip token=%s reason=token_ev_memory_prob p=%.2f detail=%s",
                    symbol,
                    prob,
                    str(token_ev_memory.get("detail", "")),
                )
                return None

        entry_price_usd = float(token_data.get("price_usd") or 0)
        if entry_price_usd <= 0:
            fetched = await self._fetch_current_price(token_address)
            if fetched:
                entry_price_usd = float(fetched[1])
        if entry_price_usd <= 0:
            self._bump_skip_reason("no_price")
            logger.info("AutoTrade skip token=%s reason=no_price", symbol)
            return None

        tp = int(config.PROFIT_TARGET_PERCENT)
        sl = int(config.STOP_LOSS_PERCENT)
        liquidity_usd = float(token_data.get("liquidity") or 0)
        volume_5m = float(token_data.get("volume_5m") or 0)
        price_change_5m = float(token_data.get("price_change_5m") or 0)
        risk_level = str(token_data.get("risk_level", "MEDIUM")).upper()
        cluster_key = self._token_cluster_key(
            score=int(score),
            liquidity_usd=float(liquidity_usd),
            volume_5m=float(volume_5m),
            price_change_5m=float(price_change_5m),
            risk_level=str(risk_level),
        )
        if self._active_trade_decision_context is not None:
            self._active_trade_decision_context.update(
                {
                    "token_cluster_key": cluster_key,
                    "symbol_window_trades": int(symbol_metrics.get("trades", 0) or 0),
                    "symbol_window_avg_pnl_usd": float(symbol_metrics.get("avg_pnl", 0.0) or 0.0),
                    "symbol_window_loss_share": float(symbol_metrics.get("loss_share", 0.0) or 0.0),
                    "token_ev_memory_detail": str(token_ev_memory.get("detail", "")),
                    "token_ev_memory_size_mult": float(token_ev_memory.get("size_mult", 1.0) or 1.0),
                    "token_ev_memory_edge_mult": float(token_ev_memory.get("edge_mult", 1.0) or 1.0),
                    "token_ev_memory_probability": float(token_ev_memory.get("entry_probability", 1.0) or 1.0),
                }
            )
        if (
            entry_channel == "core"
            and bool(getattr(config, "CORE_BLOCK_L1_FOR_CORE_ENABLED", True))
            and "|l1|" in str(cluster_key).lower()
        ):
            self._bump_skip_reason("core_l1_block")
            logger.info(
                "AutoTrade skip token=%s reason=core_l1_block cluster=%s entry_channel=%s",
                symbol,
                cluster_key,
                entry_channel,
            )
            return None
        if bool(getattr(config, "ENTRY_REQUIRE_POSITIVE_CHANGE_5M", False)):
            min_change_5m = float(getattr(config, "ENTRY_MIN_PRICE_CHANGE_5M_PERCENT", 0.0) or 0.0)
            if price_change_5m <= min_change_5m:
                self._bump_skip_reason("entry_momentum")
                logger.info(
                    "AutoTrade skip token=%s reason=entry_momentum detail=price_change_5m %.2f%% <= %.2f%%",
                    symbol,
                    price_change_5m,
                    min_change_5m,
                )
                return None
        if bool(getattr(config, "ENTRY_REQUIRE_VOLUME_BUFFER", False)):
            min_volume_abs = float(getattr(config, "ENTRY_MIN_VOLUME_5M_USD", 0.0) or 0.0)
            min_volume_mult = float(getattr(config, "ENTRY_MIN_VOLUME_5M_MULT", 1.0) or 1.0)
            safe_volume_floor = float(getattr(config, "SAFE_MIN_VOLUME_5M_USD", 0.0) or 0.0)
            required_volume = max(min_volume_abs, safe_volume_floor * min_volume_mult)
            if volume_5m < required_volume:
                self._bump_skip_reason("entry_volume_buffer")
                logger.info(
                    "AutoTrade skip token=%s reason=entry_volume_buffer detail=vol5m %.0f < req %.0f",
                    symbol,
                    volume_5m,
                    required_volume,
                )
                return None
        if not self._passes_token_guards(token_data):
            self._bump_skip_reason("safety_guards")
            logger.info("AutoTrade skip token=%s reason=safety_guards", symbol)
            return None

        # First pass cost model (neutral size), then re-price using actual position size below.
        cost_profile = self._estimate_cost_profile(liquidity_usd, risk_level, score)
        expected_edge_percent = self._estimate_edge_percent(score, tp, sl, cost_profile["total_percent"], risk_level)
        regime_edge_mult = _safe_float(token_data.get("_regime_edge_mult"), 1.0)
        regime_edge_mult = max(0.75, min(1.35, regime_edge_mult))
        regime_size_mult = _safe_float(token_data.get("_regime_size_mult"), 1.0)
        regime_size_mult = max(0.25, min(1.50, regime_size_mult))
        regime_hold_mult = _safe_float(token_data.get("_regime_hold_mult"), 1.0)
        regime_hold_mult = max(0.35, min(1.40, regime_hold_mult))
        regime_partial_tp_trigger_mult = _safe_float(token_data.get("_regime_partial_tp_trigger_mult"), 1.0)
        regime_partial_tp_trigger_mult = max(0.20, min(2.00, regime_partial_tp_trigger_mult))
        regime_partial_tp_sell_mult = _safe_float(token_data.get("_regime_partial_tp_sell_mult"), 1.0)
        regime_partial_tp_sell_mult = max(0.20, min(2.00, regime_partial_tp_sell_mult))
        entry_tier = str(token_data.get("_entry_tier", "") or "").strip().upper()
        market_mode = str(token_data.get("_regime_name", "") or "").strip().upper()
        entry_channel = str(token_data.get("_entry_channel", "core") or "core").strip().lower()
        if entry_channel not in {"core", "explore"}:
            entry_channel = "core"
        channel_size_mult = _safe_float(token_data.get("_entry_channel_size_mult"), 1.0)
        channel_size_mult = max(0.10, min(1.40, channel_size_mult))
        channel_hold_mult = _safe_float(token_data.get("_entry_channel_hold_mult"), 1.0)
        channel_hold_mult = max(0.20, min(1.40, channel_hold_mult))
        channel_edge_usd_mult = _safe_float(token_data.get("_entry_channel_edge_usd_mult"), 1.0)
        channel_edge_usd_mult = max(0.05, min(1.50, channel_edge_usd_mult))
        channel_edge_pct_mult = _safe_float(token_data.get("_entry_channel_edge_pct_mult"), 1.0)
        channel_edge_pct_mult = max(0.05, min(1.50, channel_edge_pct_mult))
        quality_edge_mult, quality_size_mult, quality_detail = self._quality_entry_adaptation(
            token_data,
            entry_tier=entry_tier,
            entry_channel=entry_channel,
        )
        if self._active_trade_decision_context is not None:
            self._active_trade_decision_context.update(
                {
                    "quality_source_mult": float(_safe_float(token_data.get("_quality_source_mult"), 1.0)),
                    "quality_ev_signal": float(_safe_float(token_data.get("_quality_ev_signal"), 0.0)),
                    "quality_edge_mult": float(quality_edge_mult),
                    "quality_size_mult": float(quality_size_mult),
                    "quality_detail": str(quality_detail),
                }
            )
        token_ev_edge_mult = max(0.20, float(token_ev_memory.get("edge_mult", 1.0) or 1.0))
        token_ev_size_mult = max(0.05, float(token_ev_memory.get("size_mult", 1.0) or 1.0))
        regime_size_mult = max(0.10, min(1.50, float(regime_size_mult) * float(channel_size_mult)))
        regime_hold_mult = max(0.20, min(1.40, float(regime_hold_mult) * float(channel_hold_mult)))
        execution_lane = "scalp"
        execution_lane_detail = "off"
        execution_size_mult = 1.0
        execution_hold_mult = 1.0
        execution_edge_mult = 1.0
        cluster_route = {
            "detail": "off",
            "samples": 0.0,
            "avg_net_usd": 0.0,
            "loss_share": 0.0,
            "size_mult": 1.0,
            "hold_mult": 1.0,
            "edge_mult": 1.0,
            "entry_probability": 1.0,
        }

        position_size_usd = self._choose_position_size(expected_edge_percent)
        if bool(getattr(config, "POSITION_SIZE_QUALITY_ENABLED", True)):
            mult, mult_detail = self._quality_size_multiplier(
                score=score,
                liquidity_usd=liquidity_usd,
                volume_5m=volume_5m,
                price_change_5m=price_change_5m,
            )
            position_size_usd *= float(mult)
        else:
            mult, mult_detail = 1.0, "off"
        position_size_usd *= float(regime_size_mult)
        position_size_usd *= float(token_ev_size_mult)
        position_size_usd *= float(quality_size_mult)
        reinvest_mult, reinvest_detail = self._reinvest_multiplier(expected_edge_percent=float(expected_edge_percent))
        position_size_usd *= float(reinvest_mult)

        position_size_usd = min(position_size_usd, self._available_balance_usd())
        position_size_usd = self._apply_max_loss_per_trade_cap(
            position_size_usd=position_size_usd,
            stop_loss_percent=sl,
            total_cost_percent=cost_profile["total_percent"],
            gas_usd=cost_profile["gas_usd"],
        )
        max_buy_weth = float(config.MAX_BUY_AMOUNT)
        if max_buy_weth > 0:
            weth_price_usd = await self._resolve_weth_price_usd(token_data)
            cap_usd = max_buy_weth * weth_price_usd
            if cap_usd <= 0:
                self._bump_skip_reason("invalid_max_buy_cap")
                logger.info("AutoTrade skip token=%s reason=invalid_max_buy_cap", symbol)
                return None
            position_size_usd = min(position_size_usd, cap_usd)

        # Recompute costs on actual size and re-apply risk cap with the size-aware cost model.
        cost_profile = self._estimate_cost_profile(
            liquidity_usd,
            risk_level,
            score,
            position_size_usd=position_size_usd,
        )
        position_size_usd_capped = self._apply_max_loss_per_trade_cap(
            position_size_usd=position_size_usd,
            stop_loss_percent=sl,
            total_cost_percent=cost_profile["total_percent"],
            gas_usd=cost_profile["gas_usd"],
        )
        if position_size_usd_capped < position_size_usd:
            position_size_usd = position_size_usd_capped
            cost_profile = self._estimate_cost_profile(
                liquidity_usd,
                risk_level,
                score,
                position_size_usd=position_size_usd,
            )

        expected_edge_percent = self._estimate_edge_percent(score, tp, sl, cost_profile["total_percent"], risk_level)
        edge_percent_for_decision = float(expected_edge_percent) * float(regime_edge_mult)
        expected_edge_usd = float(position_size_usd) * float(edge_percent_for_decision) / 100.0
        ev_snapshot = self._ev_expected_snapshot(
            market_mode=market_mode,
            entry_tier=entry_tier,
            symbol=symbol_upper,
            cluster_key=cluster_key,
            score=int(score),
            fallback_edge_usd=float(expected_edge_usd),
        )
        kelly_mult, kelly_fraction = self._kelly_lite_multiplier(
            p_win=float(ev_snapshot.get("win_rate", 0.0) or 0.0),
            avg_win_usd=float(ev_snapshot.get("avg_win_usd", 0.0) or 0.0),
            avg_loss_usd=float(ev_snapshot.get("avg_loss_usd", 0.0) or 0.0),
            confidence=float(ev_snapshot.get("confidence", 0.0) or 0.0),
            expected_net_usd=float(ev_snapshot.get("expected_net_usd", 0.0) or 0.0),
        )
        position_size_usd *= float(kelly_mult)
        if bool(getattr(config, "EXECUTION_ARCH_ENABLED", True)):
            lane_profile = self._execution_lane_profile(
                symbol=symbol_upper,
                market_mode=market_mode,
                entry_tier=entry_tier,
                entry_channel=entry_channel,
                score=int(score),
                expected_edge_percent=float(edge_percent_for_decision),
                ev_snapshot=ev_snapshot,
            )
            execution_lane = str(lane_profile.get("lane", "scalp") or "scalp").strip().lower()
            execution_lane_detail = str(lane_profile.get("detail", "n/a") or "n/a")
            execution_size_mult *= max(0.05, float(lane_profile.get("size_mult", 1.0) or 1.0))
            execution_hold_mult *= max(0.10, float(lane_profile.get("hold_mult", 1.0) or 1.0))
            execution_edge_mult *= max(0.20, float(lane_profile.get("edge_mult", 1.0) or 1.0))

            cluster_route = self._cluster_ev_routing_adjustments(
                cluster_key=cluster_key,
                market_mode=market_mode,
                entry_tier=entry_tier,
                entry_channel=entry_channel,
                candidate_id=candidate_id,
            )
            cluster_prob = float(cluster_route.get("entry_probability", 1.0) or 1.0)
            cluster_seed = str(cluster_route.get("seed", "") or "")
            if cluster_prob < 1.0 and (not self._passes_probability_gate(seed=cluster_seed, probability=cluster_prob)):
                self._bump_skip_reason("cluster_ev_prob")
                logger.info(
                    "AutoTrade skip token=%s reason=cluster_ev_prob p=%.2f detail=%s",
                    symbol,
                    cluster_prob,
                    str(cluster_route.get("detail", "")),
                )
                return None
            execution_size_mult *= max(0.05, float(cluster_route.get("size_mult", 1.0) or 1.0))
            execution_hold_mult *= max(0.10, float(cluster_route.get("hold_mult", 1.0) or 1.0))
            execution_edge_mult *= max(0.20, float(cluster_route.get("edge_mult", 1.0) or 1.0))
            regime_hold_mult = max(0.20, min(2.20, float(regime_hold_mult) * float(execution_hold_mult)))
            position_size_usd *= float(execution_size_mult)

        position_size_usd = min(position_size_usd, self._available_balance_usd())
        if max_buy_weth > 0:
            position_size_usd = min(position_size_usd, cap_usd)
        cost_profile = self._estimate_cost_profile(
            liquidity_usd,
            risk_level,
            score,
            position_size_usd=position_size_usd,
        )
        position_size_usd = self._apply_max_loss_per_trade_cap(
            position_size_usd=position_size_usd,
            stop_loss_percent=sl,
            total_cost_percent=cost_profile["total_percent"],
            gas_usd=cost_profile["gas_usd"],
        )
        position_size_usd = min(position_size_usd, self._available_balance_usd())
        if max_buy_weth > 0:
            position_size_usd = min(position_size_usd, cap_usd)
        expected_edge_percent = self._estimate_edge_percent(score, tp, sl, cost_profile["total_percent"], risk_level)
        edge_percent_for_decision = float(expected_edge_percent) * float(regime_edge_mult)

        mm = str(market_mode or "").strip().upper()
        is_tier_a = str(entry_tier or "").strip().upper() == "A"
        is_core_channel = str(entry_channel or "").strip().lower() == "core"
        is_a_core = (
            is_tier_a
            and is_core_channel
        )
        if bool(getattr(config, "ENTRY_A_CORE_MIN_TRADE_USD_ENABLED", True)) and is_tier_a:
            allow_red = bool(getattr(config, "ENTRY_A_CORE_MIN_TRADE_APPLY_IN_RED", False))
            if allow_red or mm != "RED":
                floor_usd = max(0.0, float(getattr(config, "ENTRY_A_CORE_MIN_TRADE_USD", 0.45) or 0.45))
                available_usd = float(self._available_balance_usd())
                if max_buy_weth > 0:
                    floor_usd = min(floor_usd, float(cap_usd))
                floor_usd = min(floor_usd, available_usd)
                if floor_usd > position_size_usd and floor_usd > 0.0:
                    position_size_usd = floor_usd
                    cost_profile = self._estimate_cost_profile(
                        liquidity_usd,
                        risk_level,
                        score,
                        position_size_usd=position_size_usd,
                    )
                    expected_edge_percent = self._estimate_edge_percent(score, tp, sl, cost_profile["total_percent"], risk_level)
                    edge_percent_for_decision = float(expected_edge_percent) * float(regime_edge_mult)

        min_trade_usd = max(0.01, float(getattr(config, "MIN_TRADE_USD", 0.25) or 0.25))
        if position_size_usd < min_trade_usd:
            self._bump_skip_reason("min_trade_size")
            logger.info(
                "AutoTrade skip token=%s reason=min_trade_size size=$%.2f min=$%.2f",
                symbol,
                position_size_usd,
                min_trade_usd,
            )
            return None
        if position_size_usd < 0.1:
            self._bump_skip_reason("low_balance")
            logger.info("AutoTrade skip token=%s reason=low_balance", symbol)
            return None

        expected_edge_usd = float(position_size_usd) * float(edge_percent_for_decision) / 100.0
        core_probe_eligible = (
            bool(getattr(config, "EV_FIRST_ENTRY_CORE_PROBE_ENABLED", True))
            and is_a_core
        )
        core_probe_applied = False
        core_probe_ev_override = False
        core_probe_edge_override = False
        core_probe_detail = ""
        core_probe_budget_used = 0
        core_probe_budget_cap = max(0, int(getattr(config, "EV_FIRST_ENTRY_CORE_PROBE_MAX_OPENS", 1) or 1))
        if bool(getattr(config, "EV_FIRST_ENTRY_ENABLED", True)):
            ev_samples = float(ev_snapshot.get("samples", 0.0) or 0.0)
            ev_expected_net = float(ev_snapshot.get("expected_net_usd", 0.0) or 0.0)
            ev_min = float(getattr(config, "EV_FIRST_ENTRY_MIN_NET_USD", 0.0010) or 0.0010)
            if mm == "RED":
                ev_min *= float(getattr(config, "EV_FIRST_ENTRY_MIN_NET_USD_RED_MULT", 1.20) or 1.20)
            elif mm == "YELLOW":
                ev_min *= float(getattr(config, "EV_FIRST_ENTRY_MIN_NET_USD_YELLOW_MULT", 1.08) or 1.08)
            else:
                ev_min *= float(getattr(config, "EV_FIRST_ENTRY_MIN_NET_USD_GREEN_MULT", 1.00) or 1.00)
            if (
                bool(getattr(config, "ENTRY_EDGE_SOFTEN_GREEN_A_CORE_ENABLED", True))
                and mm == "GREEN"
                and is_a_core
            ):
                ev_min *= float(getattr(config, "EV_FIRST_ENTRY_MIN_NET_USD_GREEN_A_CORE_MULT", 0.95) or 0.95)
            if str(entry_channel) == "explore":
                ev_min *= float(getattr(config, "EV_FIRST_ENTRY_MIN_NET_USD_EXPLORE_MULT", 1.20) or 1.20)
            ev_min *= float(quality_edge_mult)
            ev_min *= float(token_ev_edge_mult)
            ev_min *= float(execution_edge_mult)
            if (
                bool(getattr(config, "ENTRY_SMALL_SIZE_EV_RELAX_ENABLED", True))
                and is_tier_a
                and mm in {"GREEN", "YELLOW"}
            ):
                nominal_size = max(0.1, float(getattr(config, "ENTRY_SMALL_SIZE_EV_RELAX_NOMINAL_USD", 1.00) or 1.00))
                min_factor = max(0.10, min(1.0, float(getattr(config, "ENTRY_SMALL_SIZE_EV_RELAX_MIN_FACTOR", 0.50) or 0.50)))
                size_factor = max(min_factor, min(1.0, float(position_size_usd) / nominal_size))
                ev_min *= float(size_factor)
            min_samples = max(5.0, float(getattr(config, "EV_FIRST_ENTRY_MIN_SAMPLES", 24) or 24))
            if ev_samples >= min_samples and ev_expected_net < ev_min:
                if core_probe_eligible:
                    probe_ok, probe_used, probe_cap = self._core_probe_budget_available()
                    core_probe_budget_used = int(probe_used)
                    core_probe_budget_cap = int(probe_cap)
                    if probe_ok:
                        probe_relax_only = bool(getattr(config, "EV_FIRST_ENTRY_CORE_PROBE_RELAX_ONLY", True))
                        probe_min_net = float(
                            getattr(config, "EV_FIRST_ENTRY_CORE_PROBE_MIN_NET_USD", -0.0015) or -0.0015
                        )
                        probe_tolerance = max(
                            0.0,
                            float(getattr(config, "EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD", 0.0030) or 0.0030),
                        )
                        probe_floor = probe_min_net if not probe_relax_only else max(probe_min_net, ev_min - probe_tolerance)
                        if ev_expected_net >= probe_floor:
                            core_probe_applied = True
                            core_probe_ev_override = True
                            core_probe_detail = (
                                f"ev_probe expected={ev_expected_net:.5f} floor={probe_floor:.5f} "
                                f"min={ev_min:.5f} budget={probe_used}/{probe_cap}"
                            )
                if not core_probe_ev_override:
                    probe_tail = ""
                    if core_probe_eligible and core_probe_budget_cap > 0:
                        probe_tail = (
                            f" probe_budget={core_probe_budget_used}/{core_probe_budget_cap} "
                            f"probe_detail={core_probe_detail or '-'}"
                        )
                self._bump_skip_reason("ev_net_low")
                logger.info(
                    (
                        "AutoTrade skip token=%s reason=ev_net_low expected_net=$%.5f min=$%.5f "
                        "samples=%.0f conf=%.2f pwin=%.2f avg_win=$%.4f avg_loss=$%.4f "
                        "kelly=%.3f mem=%s%s"
                    ),
                    symbol,
                    ev_expected_net,
                    ev_min,
                    ev_samples,
                    float(ev_snapshot.get("confidence", 0.0) or 0.0),
                    float(ev_snapshot.get("win_rate", 0.0) or 0.0),
                    float(ev_snapshot.get("avg_win_usd", 0.0) or 0.0),
                    float(ev_snapshot.get("avg_loss_usd", 0.0) or 0.0),
                    float(kelly_fraction),
                    str(token_ev_memory.get("detail", "")),
                    probe_tail,
                )
                return None
        if self._active_trade_decision_context is not None:
            self._active_trade_decision_context.update(
                {
                    "ev_samples": float(ev_snapshot.get("samples", 0.0) or 0.0),
                    "ev_confidence": float(ev_snapshot.get("confidence", 0.0) or 0.0),
                    "ev_win_rate": float(ev_snapshot.get("win_rate", 0.0) or 0.0),
                    "ev_avg_win_usd": float(ev_snapshot.get("avg_win_usd", 0.0) or 0.0),
                    "ev_avg_loss_usd": float(ev_snapshot.get("avg_loss_usd", 0.0) or 0.0),
                    "ev_avg_net_usd": float(ev_snapshot.get("avg_net_usd", 0.0) or 0.0),
                    "ev_expected_net_usd": float(ev_snapshot.get("expected_net_usd", 0.0) or 0.0),
                    "kelly_mult": float(kelly_mult),
                    "kelly_fraction": float(kelly_fraction),
                    "execution_lane": str(execution_lane or "scalp"),
                    "execution_size_mult": float(execution_size_mult),
                    "execution_hold_mult": float(execution_hold_mult),
                    "execution_edge_mult": float(execution_edge_mult),
                    "execution_lane_detail": str(execution_lane_detail),
                    "cluster_route_detail": str(cluster_route.get("detail", "")),
                    "cluster_route_samples": float(cluster_route.get("samples", 0.0) or 0.0),
                    "cluster_route_avg_net_usd": float(cluster_route.get("avg_net_usd", 0.0) or 0.0),
                    "cluster_route_loss_share": float(cluster_route.get("loss_share", 0.0) or 0.0),
                    "cluster_route_probability": float(cluster_route.get("entry_probability", 1.0) or 1.0),
                    "core_probe_eligible": bool(core_probe_eligible),
                    "core_probe_applied": bool(core_probe_applied),
                    "core_probe_ev_override": bool(core_probe_ev_override),
                    "core_probe_edge_override": bool(core_probe_edge_override),
                    "core_probe_detail": str(core_probe_detail),
                }
            )
        edge_dbg = self._edge_debug_components(
            score=score,
            tp_percent=tp,
            sl_percent=sl,
            total_cost_percent=float(cost_profile["total_percent"]),
            risk_level=risk_level,
        )
        if config.EDGE_FILTER_ENABLED:
            mode = str(getattr(config, "EDGE_FILTER_MODE", "usd") or "usd").strip().lower()
            min_edge_pct = (
                float(getattr(config, "MIN_EXPECTED_EDGE_PERCENT", 0.0) or 0.0)
                * float(channel_edge_pct_mult)
                * float(quality_edge_mult)
                * float(token_ev_edge_mult)
                * float(execution_edge_mult)
            )
            min_edge_usd = (
                float(getattr(config, "MIN_EXPECTED_EDGE_USD", 0.0) or 0.0)
                * float(channel_edge_usd_mult)
                * float(quality_edge_mult)
                * float(token_ev_edge_mult)
                * float(execution_edge_mult)
            )
            if (
                bool(getattr(config, "ENTRY_EDGE_SOFTEN_GREEN_A_CORE_ENABLED", True))
                and mm == "GREEN"
                and is_a_core
            ):
                min_edge_pct *= float(getattr(config, "ENTRY_EDGE_SOFTEN_GREEN_A_CORE_PCT_MULT", 0.92) or 0.92)
                min_edge_usd *= float(getattr(config, "ENTRY_EDGE_SOFTEN_GREEN_A_CORE_USD_MULT", 0.92) or 0.92)
            if (
                bool(getattr(config, "ENTRY_SMALL_SIZE_EDGE_RELAX_ENABLED", True))
                and is_tier_a
                and mm in {"GREEN", "YELLOW"}
            ):
                nominal_size = max(0.1, float(getattr(config, "ENTRY_SMALL_SIZE_EDGE_RELAX_NOMINAL_USD", 1.00) or 1.00))
                min_factor = max(0.10, min(1.0, float(getattr(config, "ENTRY_SMALL_SIZE_EDGE_RELAX_MIN_FACTOR", 0.50) or 0.50)))
                size_factor = max(min_factor, min(1.0, float(position_size_usd) / nominal_size))
                min_edge_usd *= float(size_factor)
            edge_min_pct_effective = float(min_edge_pct)
            edge_min_usd_effective = float(min_edge_usd)
            pct_ok = edge_percent_for_decision >= float(edge_min_pct_effective)
            usd_ok = expected_edge_usd >= float(edge_min_usd_effective)
            needs_probe = (
                (mode == "percent" and (not pct_ok))
                or (mode == "usd" and (not usd_ok))
                or (mode == "both" and (not (pct_ok and usd_ok)))
            )
            if core_probe_eligible and needs_probe:
                probe_ok = bool(core_probe_applied)
                probe_used = core_probe_budget_used
                probe_cap = core_probe_budget_cap
                if not probe_ok:
                    probe_ok, probe_used, probe_cap = self._core_probe_budget_available()
                core_probe_budget_used = int(probe_used)
                core_probe_budget_cap = int(probe_cap)
                if probe_ok:
                    probe_relax_only = bool(getattr(config, "EV_FIRST_ENTRY_CORE_PROBE_RELAX_ONLY", True))
                    if probe_relax_only:
                        relax_ratio = max(
                            0.0,
                            min(
                                0.80,
                                float(
                                    getattr(
                                        config,
                                        "EV_FIRST_ENTRY_CORE_PROBE_EDGE_PERCENT_RELAX_RATIO",
                                        0.20,
                                    )
                                    or 0.20
                                ),
                            ),
                        )
                        relax_abs = max(
                            0.0,
                            float(getattr(config, "EV_FIRST_ENTRY_CORE_PROBE_EDGE_USD_RELAX_ABS", 0.0020) or 0.0020),
                        )
                        probe_min_edge_pct = max(0.0, float(min_edge_pct) * (1.0 - relax_ratio))
                        probe_min_edge_usd = max(0.0, float(min_edge_usd) - relax_abs)
                    else:
                        probe_min_edge_pct = 0.0
                        probe_min_edge_usd = 0.0
                    probe_pct_ok = edge_percent_for_decision >= float(probe_min_edge_pct)
                    probe_usd_ok = expected_edge_usd >= float(probe_min_edge_usd)
                    probe_pass = False
                    if mode == "percent":
                        probe_pass = probe_pct_ok
                    elif mode == "usd":
                        probe_pass = probe_usd_ok
                    else:
                        probe_pass = probe_pct_ok and probe_usd_ok
                    if probe_pass:
                        edge_min_pct_effective = float(probe_min_edge_pct)
                        edge_min_usd_effective = float(probe_min_edge_usd)
                        pct_ok = probe_pct_ok
                        usd_ok = probe_usd_ok
                        core_probe_applied = True
                        core_probe_edge_override = True
                        core_probe_detail = (
                            f"edge_probe mode={mode} min_pct={probe_min_edge_pct:.4f} min_usd={probe_min_edge_usd:.5f} "
                            f"base_pct={min_edge_pct:.4f} base_usd={min_edge_usd:.5f} budget={probe_used}/{probe_cap}"
                        )
            if mode == "percent" and not pct_ok:
                self._bump_skip_reason("negative_edge")
                logger.info(
                    "AutoTrade skip token=%s reason=negative_edge edge=%.2f%% raw=%.2f%% mult=%.2f min=%.2f%% probe_min=%.2f%% edge_usd=$%.3f size=$%.2f gross=%.2f%% costs=%.2f%% pwin=%.2f probe=%s budget=%s/%s",
                    symbol,
                    edge_percent_for_decision,
                    expected_edge_percent,
                    regime_edge_mult,
                    min_edge_pct,
                    edge_min_pct_effective,
                    expected_edge_usd,
                    position_size_usd,
                    float(edge_dbg["gross_percent"]),
                    cost_profile["total_percent"],
                    float(edge_dbg["p_win"]),
                    str(core_probe_detail or "-"),
                    core_probe_budget_used,
                    core_probe_budget_cap,
                )
                return None
            if mode == "usd" and not usd_ok:
                self._bump_skip_reason("edge_usd_low")
                logger.info(
                    "AutoTrade skip token=%s reason=edge_usd_low edge_usd=$%.3f min=$%.3f probe_min=$%.3f edge=%.2f%% raw=%.2f%% mult=%.2f size=$%.2f gross=%.2f%% costs=%.2f%% pwin=%.2f probe=%s budget=%s/%s",
                    symbol,
                    expected_edge_usd,
                    min_edge_usd,
                    edge_min_usd_effective,
                    edge_percent_for_decision,
                    expected_edge_percent,
                    regime_edge_mult,
                    position_size_usd,
                    float(edge_dbg["gross_percent"]),
                    cost_profile["total_percent"],
                    float(edge_dbg["p_win"]),
                    str(core_probe_detail or "-"),
                    core_probe_budget_used,
                    core_probe_budget_cap,
                )
                return None
            if mode == "both" and not (pct_ok and usd_ok):
                self._bump_skip_reason("edge_low")
                logger.info(
                    "AutoTrade skip token=%s reason=edge_low mode=both edge=%.2f%% raw=%.2f%% mult=%.2f min=%.2f%% probe_min=%.2f%% edge_usd=$%.3f min_usd=$%.3f probe_min_usd=$%.3f size=$%.2f gross=%.2f%% costs=%.2f%% pwin=%.2f probe=%s budget=%s/%s",
                    symbol,
                    edge_percent_for_decision,
                    expected_edge_percent,
                    regime_edge_mult,
                    min_edge_pct,
                    edge_min_pct_effective,
                    expected_edge_usd,
                    min_edge_usd,
                    edge_min_usd_effective,
                    position_size_usd,
                    float(edge_dbg["gross_percent"]),
                    cost_profile["total_percent"],
                    float(edge_dbg["p_win"]),
                    str(core_probe_detail or "-"),
                    core_probe_budget_used,
                    core_probe_budget_cap,
                )
                return None

        if core_probe_applied:
            logger.info(
                "AUTO_CORE_PROBE token=%s mode=%s tier=%s channel=%s ev_override=%s edge_override=%s detail=%s",
                symbol,
                market_mode or "-",
                entry_tier or "-",
                entry_channel or "-",
                core_probe_ev_override,
                core_probe_edge_override,
                core_probe_detail or "applied",
            )

        breakdown = score_data.get("breakdown") if isinstance(score_data.get("breakdown"), dict) else {}
        logger.info(
            (
                "AUTO_DECISION token=%s mode=%s tier=%s channel=%s score=%s src=%s liq=%.0f vol5m=%.0f chg5m=%.2f%% "
                "edge=%.2f%% raw=%.2f%% regime_mult=%.2f edge_usd=$%.3f size=$%.2f mult=%.2f(%s) reinvest=%.2f(%s) "
                "size_mode=%.2f hold_mode=%.2f exec_lane=%s exec_mult(size=%.2f hold=%.2f edge=%.2f) "
                "edge_gate_mult(usd=%.2f,pct=%.2f) quality(edge=%.2f,size=%.2f detail=%s) costs=%.2f%% gas=$%.3f "
                "ev_net=$%.5f ev_samples=%.0f conf=%.2f pwin=%.2f avg_win=$%.4f avg_loss=$%.4f kelly=%.2f(k=%.3f) "
                "token_ev=%s cluster=%s cluster_route=%s breakdown=%s"
            ),
            symbol,
            market_mode or "-",
            entry_tier or "-",
            entry_channel,
            score,
            str(token_data.get("source", "-")),
            liquidity_usd,
            volume_5m,
            price_change_5m,
            edge_percent_for_decision,
            expected_edge_percent,
            regime_edge_mult,
            expected_edge_usd,
            position_size_usd,
            float(mult),
            str(mult_detail),
            float(reinvest_mult),
            str(reinvest_detail),
            float(regime_size_mult),
            float(regime_hold_mult),
            str(execution_lane),
            float(execution_size_mult),
            float(execution_hold_mult),
            float(execution_edge_mult),
            float(channel_edge_usd_mult),
            float(channel_edge_pct_mult),
            float(quality_edge_mult),
            float(quality_size_mult),
            str(quality_detail),
            float(cost_profile["total_percent"]),
            float(cost_profile["gas_usd"]),
            float(ev_snapshot.get("expected_net_usd", 0.0) or 0.0),
            float(ev_snapshot.get("samples", 0.0) or 0.0),
            float(ev_snapshot.get("confidence", 0.0) or 0.0),
            float(ev_snapshot.get("win_rate", 0.0) or 0.0),
            float(ev_snapshot.get("avg_win_usd", 0.0) or 0.0),
            float(ev_snapshot.get("avg_loss_usd", 0.0) or 0.0),
            float(kelly_mult),
            float(kelly_fraction),
            str(token_ev_memory.get("detail", "")),
            str(cluster_key),
            str(cluster_route.get("detail", "")),
            breakdown,
        )
        max_hold_seconds = self._choose_hold_seconds(
            score=score,
            risk_level=risk_level,
            liquidity_usd=liquidity_usd,
            volume_5m=volume_5m,
            price_change_5m=price_change_5m,
        )
        max_hold_seconds = int(max(30, round(float(max_hold_seconds) * float(regime_hold_mult))))

        if self._is_live_mode():
            if self.live_executor is None:
                self._bump_skip_reason("live_executor_unavailable")
                logger.error("AutoTrade skip token=%s reason=live_executor_unavailable", symbol)
                return None
            weth_price_usd = await self._resolve_weth_price_usd(token_data)
            if weth_price_usd <= 0:
                self._bump_skip_reason("no_weth_price")
                logger.info("AutoTrade skip token=%s reason=no_weth_price", symbol)
                return None
            spend_eth = position_size_usd / weth_price_usd
            # Route/roundtrip checks on tiny notional can false-fail due integer rounding/quote dust.
            # Use a bounded probe size for prechecks, while keeping the real buy size unchanged.
            probe_min_usd = float(getattr(config, "LIVE_PRECHECK_MIN_SPEND_USD", 2.0) or 2.0)
            probe_max_eth = float(getattr(config, "LIVE_PRECHECK_MAX_SPEND_ETH", 0.0030) or 0.0030)
            probe_spend_eth = max(float(spend_eth), float(probe_min_usd) / max(1e-9, float(weth_price_usd)))
            probe_spend_eth = min(probe_spend_eth, max(1e-8, float(probe_max_eth)))
            inv_ok, inv_reason = self._pre_buy_invariants(token_address, spend_eth)
            if not inv_ok:
                self._bump_skip_reason("pre_buy_invariants")
                logger.warning(
                    "AutoTrade skip token=%s reason=pre_buy_invariants detail=%s raw_address=%s normalized=%s source=%s",
                    symbol,
                    inv_reason,
                    raw_token_address,
                    token_address,
                    str(token_data.get("source", "-")),
                )
                return None
            # Ensure we keep enough native ETH for gas so we can still SELL/approve later.
            try:
                reserve_eth = float(getattr(config, "LIVE_MIN_GAS_RESERVE_ETH", 0.0) or 0.0)
            except Exception:
                reserve_eth = 0.0
            if reserve_eth > 0:
                try:
                    bal_eth = float(self.live_executor.native_balance_eth())
                    if (bal_eth - float(spend_eth)) < reserve_eth:
                        self._bump_skip_reason("gas_reserve")
                        logger.info(
                            "AutoTrade skip token=%s reason=gas_reserve balance_eth=%.6f spend_eth=%.6f reserve_eth=%.6f",
                            symbol,
                            bal_eth,
                            float(spend_eth),
                            reserve_eth,
                        )
                        return None
                except Exception:
                    pass
            route_ok, route_reason = await asyncio.to_thread(
                self.live_executor.is_buy_route_supported,
                token_address,
                probe_spend_eth,
            )
            if not route_ok:
                dex = str(token_data.get("dex", "unknown"))
                labels = token_data.get("dex_labels") or []
                labels_text = ",".join(str(x) for x in labels) if isinstance(labels, list) else str(labels)
                logger.info(
                    "AutoTrade skip token=%s reason=unsupported_live_route dex=%s labels=%s detail=%s",
                    symbol,
                    dex,
                    labels_text or "-",
                    self._short_error_text(route_reason),
                )
                self._bump_skip_reason("unsupported_live_route")
                self._blacklist_add(token_address, f"unsupported_buy_route:{self._short_error_text(route_reason)}")
                return None
            if bool(getattr(config, "LIVE_SELLABILITY_CHECK_ENABLED", True)):
                amt_tokens = float(getattr(config, "LIVE_SELLABILITY_CHECK_AMOUNT_TOKENS", 1.0) or 1.0)
                sell_ok, sell_reason = await asyncio.to_thread(
                    self.live_executor.is_sell_route_supported,
                    token_address,
                    amt_tokens,
                )
                if not sell_ok:
                    logger.info(
                        "AutoTrade skip token=%s reason=unsellable_or_no_quote detail=%s",
                        symbol,
                        self._short_error_text(sell_reason),
                    )
                    self._bump_skip_reason("unsellable_or_no_quote")
                    self._blacklist_add(token_address, f"unsupported_sell_route:{self._short_error_text(sell_reason)}")
                    return None
            if bool(getattr(config, "LIVE_ROUNDTRIP_CHECK_ENABLED", True)):
                ok, rt_reason, rt_ratio = await asyncio.to_thread(
                    self.live_executor.roundtrip_quote,
                    token_address,
                    probe_spend_eth,
                )
                if not ok:
                    logger.info(
                        "AutoTrade skip token=%s reason=roundtrip_quote_failed detail=%s",
                        symbol,
                        self._short_error_text(rt_reason),
                    )
                    self._bump_skip_reason("roundtrip_quote_failed")
                    self._blacklist_add(token_address, f"roundtrip_quote_failed:{self._short_error_text(rt_reason)}")
                    return None
                min_ratio = float(getattr(config, "LIVE_ROUNDTRIP_MIN_RETURN_RATIO", 0.70) or 0.70)
                if float(rt_ratio) < float(min_ratio):
                    logger.info(
                        "AutoTrade skip token=%s reason=roundtrip_ratio ratio=%.3f min=%.3f",
                        symbol,
                        float(rt_ratio),
                        float(min_ratio),
                    )
                    self._bump_skip_reason("roundtrip_ratio")
                    self._blacklist_add(token_address, f"roundtrip_ratio:{rt_ratio:.3f}")
                    return None
            hp_ok, hp_detail = await self._honeypot_guard_passes(token_address)
            if not hp_ok:
                self._bump_skip_reason("honeypot_guard")
                logger.warning(
                    "AutoTrade skip token=%s reason=honeypot_guard detail=%s",
                    symbol,
                    self._short_error_text(hp_detail),
                )
                self._blacklist_add(token_address, f"honeypot_guard:{self._short_error_text(hp_detail)}")
                return None
            try:
                buy_result = await asyncio.to_thread(self.live_executor.buy_token, token_address, spend_eth)
            except Exception as exc:
                # Conservative accounting: a failed live buy may still have consumed gas/nonce.
                self._record_tx_event()
                self._bump_skip_reason("live_buy_failed")
                logger.error("AUTO_BUY live_failed token=%s err=%s", symbol, exc)
                self._blacklist_add(token_address, f"live_buy_failed:{self._short_error_text(exc)}", ttl_seconds=6 * 3600)
                return None
            bought_raw = int(getattr(buy_result, "token_amount_raw", 0) or 0)
            if bought_raw <= 0:
                before_raw = int(getattr(buy_result, "balance_before_raw", 0) or 0)
                after_raw = int(getattr(buy_result, "balance_after_raw", 0) or 0)
                if after_raw > before_raw:
                    bought_raw = int(after_raw - before_raw)
                else:
                    attempts = max(1, int(getattr(config, "LIVE_BUY_BALANCE_RECHECK_ATTEMPTS", 8) or 8))
                    delay = max(
                        0.05,
                        float(getattr(config, "LIVE_BUY_BALANCE_RECHECK_DELAY_SECONDS", 0.30) or 0.30),
                    )
                    for idx in range(attempts):
                        try:
                            current_raw = int(await asyncio.to_thread(self.live_executor.token_balance_raw, token_address))
                        except Exception:
                            current_raw = 0
                        delta = max(0, current_raw - before_raw)
                        if delta > 0:
                            bought_raw = int(delta)
                            break
                        if idx < (attempts - 1):
                            await asyncio.sleep(delay)
            if bought_raw <= 0:
                self._record_tx_event()
                self._bump_skip_reason("live_buy_zero_amount")
                try:
                    raw_balance = int(await asyncio.to_thread(self.live_executor.token_balance_raw, token_address))
                except Exception:
                    raw_balance = 0
                if raw_balance > 0:
                    key = self._position_key(token_address)
                    prev = int(self.recovery_untracked.get(key, 0) or 0)
                    self.recovery_untracked[key] = raw_balance
                    if prev <= 0:
                        logger.warning(
                            "RECOVERY live_buy_untracked token=%s address=%s raw_balance=%s tx=%s",
                            symbol,
                            token_address,
                            raw_balance,
                            str(getattr(buy_result, "tx_hash", "")),
                        )
                    self._save_state()
                logger.error(
                    "AUTO_BUY live_failed token=%s err=zero_token_amount tx=%s spent_eth=%.8f",
                    symbol,
                    str(getattr(buy_result, "tx_hash", "")),
                    float(getattr(buy_result, "spent_eth", 0.0) or 0.0),
                )
                self._blacklist_add(token_address, "live_buy_zero_amount")
                return None
            pos = PaperPosition(
                token_address=token_address,
                candidate_id=candidate_id,
                symbol=symbol,
                entry_price_usd=entry_price_usd,
                current_price_usd=entry_price_usd,
                position_size_usd=position_size_usd,
                score=score,
                liquidity_usd=liquidity_usd,
                risk_level=risk_level,
                opened_at=datetime.now(timezone.utc),
                max_hold_seconds=max_hold_seconds,
                take_profit_percent=tp,
                stop_loss_percent=sl,
                expected_edge_percent=edge_percent_for_decision,
                buy_cost_percent=cost_profile["buy_percent"],
                sell_cost_percent=cost_profile["sell_percent"],
                gas_cost_usd=cost_profile["gas_usd"],
                token_amount_raw=int(bought_raw),
                buy_tx_hash=str(buy_result.tx_hash),
                buy_tx_status="confirmed",
                spent_eth=float(buy_result.spent_eth),
                original_position_size_usd=position_size_usd,
                market_mode=market_mode,
                entry_tier=entry_tier,
                entry_channel=entry_channel,
                partial_tp_trigger_mult=regime_partial_tp_trigger_mult,
                partial_tp_sell_mult=regime_partial_tp_sell_mult,
                token_cluster_key=cluster_key,
                ev_expected_net_usd=float(ev_snapshot.get("expected_net_usd", 0.0) or 0.0),
                ev_confidence=float(ev_snapshot.get("confidence", 0.0) or 0.0),
                kelly_mult=float(kelly_mult),
                execution_lane=str(execution_lane or "scalp"),
                cluster_ev_avg_net_usd=float(cluster_route.get("avg_net_usd", 0.0) or 0.0),
                cluster_ev_loss_share=float(cluster_route.get("loss_share", 0.0) or 0.0),
                cluster_ev_samples=float(cluster_route.get("samples", 0.0) or 0.0),
            )
            self._set_open_position(pos)
            self.total_plans += 1
            self.total_executed += 1
            self.trade_open_timestamps.append(datetime.now(timezone.utc).timestamp())
            if core_probe_applied:
                self._record_core_probe_open()
            self._record_open_symbol(pos.symbol)
            self._record_tx_event()
            logger.info(
                "AUTO_BUY Live BUY token=%s address=%s mode=%s tier=%s channel=%s lane=%s spend=%.8f ETH score=%s edge=%.2f%% edge_usd=$%.3f size=$%.2f mult=%.2f(%s) hold=%ss tx=%s",
                pos.symbol,
                pos.token_address,
                pos.market_mode or "-",
                pos.entry_tier or "-",
                pos.entry_channel or "-",
                pos.execution_lane or "-",
                pos.spent_eth,
                pos.score,
                pos.expected_edge_percent,
                float(pos.position_size_usd) * float(pos.expected_edge_percent) / 100.0,
                float(pos.position_size_usd),
                float(mult),
                str(mult_detail),
                pos.max_hold_seconds,
                pos.buy_tx_hash,
            )
            self._write_trade_decision(
                {
                    "ts": time.time(),
                    "decision_stage": "trade_open",
                    "decision": "open",
                    "reason": "buy_live",
                    "candidate_id": pos.candidate_id,
                    "token_address": pos.token_address,
                    "symbol": pos.symbol,
                    "score": int(pos.score),
                    "entry_tier": pos.entry_tier,
                    "entry_channel": pos.entry_channel,
                    "market_mode": pos.market_mode,
                    "position_size_usd": float(pos.position_size_usd),
                    "expected_edge_percent": float(pos.expected_edge_percent),
                    "token_cluster_key": str(pos.token_cluster_key or ""),
                    "ev_expected_net_usd": float(pos.ev_expected_net_usd),
                    "ev_confidence": float(pos.ev_confidence),
                    "kelly_mult": float(pos.kelly_mult),
                    "execution_lane": str(pos.execution_lane or ""),
                    "cluster_ev_avg_net_usd": float(pos.cluster_ev_avg_net_usd),
                    "cluster_ev_loss_share": float(pos.cluster_ev_loss_share),
                    "cluster_ev_samples": float(pos.cluster_ev_samples),
                    "core_probe": bool(core_probe_applied),
                    "core_probe_ev_override": bool(core_probe_ev_override),
                    "core_probe_edge_override": bool(core_probe_edge_override),
                    "core_probe_detail": str(core_probe_detail),
                    "max_hold_seconds": int(pos.max_hold_seconds),
                    "tx_hash": pos.buy_tx_hash,
                }
            )
            self._active_trade_decision_context = None
            self._save_state()
            return pos

        pos = PaperPosition(
            token_address=token_address,
            candidate_id=candidate_id,
            symbol=symbol,
            entry_price_usd=entry_price_usd,
            current_price_usd=entry_price_usd,
            position_size_usd=position_size_usd,
            score=score,
            liquidity_usd=liquidity_usd,
            risk_level=risk_level,
            opened_at=datetime.now(timezone.utc),
            max_hold_seconds=max_hold_seconds,
            take_profit_percent=tp,
            stop_loss_percent=sl,
            expected_edge_percent=edge_percent_for_decision,
            buy_cost_percent=cost_profile["buy_percent"],
            sell_cost_percent=cost_profile["sell_percent"],
            gas_cost_usd=cost_profile["gas_usd"],
            original_position_size_usd=position_size_usd,
            market_mode=market_mode,
            entry_tier=entry_tier,
            entry_channel=entry_channel,
            partial_tp_trigger_mult=regime_partial_tp_trigger_mult,
            partial_tp_sell_mult=regime_partial_tp_sell_mult,
            token_cluster_key=cluster_key,
            ev_expected_net_usd=float(ev_snapshot.get("expected_net_usd", 0.0) or 0.0),
            ev_confidence=float(ev_snapshot.get("confidence", 0.0) or 0.0),
            kelly_mult=float(kelly_mult),
            execution_lane=str(execution_lane or "scalp"),
            cluster_ev_avg_net_usd=float(cluster_route.get("avg_net_usd", 0.0) or 0.0),
            cluster_ev_loss_share=float(cluster_route.get("loss_share", 0.0) or 0.0),
            cluster_ev_samples=float(cluster_route.get("samples", 0.0) or 0.0),
        )

        self._set_open_position(pos)
        self.total_plans += 1
        self.total_executed += 1
        self.trade_open_timestamps.append(datetime.now(timezone.utc).timestamp())
        if core_probe_applied:
            self._record_core_probe_open()
        self._record_open_symbol(pos.symbol)
        self._record_tx_event()
        self.paper_balance_usd -= position_size_usd

        logger.info(
            "AUTO_BUY Paper BUY token=%s address=%s mode=%s tier=%s channel=%s lane=%s entry=$%.8f size=$%.2f score=%s edge=%.2f%% edge_usd=$%.3f mult=%.2f(%s) hold=%ss costs=%.2f%% gas=$%.2f",
            pos.symbol,
            pos.token_address,
            pos.market_mode or "-",
            pos.entry_tier or "-",
            pos.entry_channel or "-",
            pos.execution_lane or "-",
            pos.entry_price_usd,
            pos.position_size_usd,
            pos.score,
            pos.expected_edge_percent,
            float(pos.position_size_usd) * float(pos.expected_edge_percent) / 100.0,
            float(mult),
            str(mult_detail),
            pos.max_hold_seconds,
            pos.buy_cost_percent + pos.sell_cost_percent,
            pos.gas_cost_usd,
        )
        self._write_trade_decision(
            {
                "ts": time.time(),
                "decision_stage": "trade_open",
                "decision": "open",
                "reason": "buy_paper",
                "candidate_id": pos.candidate_id,
                "token_address": pos.token_address,
                "symbol": pos.symbol,
                "score": int(pos.score),
                "entry_tier": pos.entry_tier,
                "entry_channel": pos.entry_channel,
                "market_mode": pos.market_mode,
                "position_size_usd": float(pos.position_size_usd),
                "expected_edge_percent": float(pos.expected_edge_percent),
                "token_cluster_key": str(pos.token_cluster_key or ""),
                "ev_expected_net_usd": float(pos.ev_expected_net_usd),
                "ev_confidence": float(pos.ev_confidence),
                "kelly_mult": float(pos.kelly_mult),
                "execution_lane": str(pos.execution_lane or ""),
                "cluster_ev_avg_net_usd": float(pos.cluster_ev_avg_net_usd),
                "cluster_ev_loss_share": float(pos.cluster_ev_loss_share),
                "cluster_ev_samples": float(pos.cluster_ev_samples),
                "core_probe": bool(core_probe_applied),
                "core_probe_ev_override": bool(core_probe_ev_override),
                "core_probe_edge_override": bool(core_probe_edge_override),
                "core_probe_detail": str(core_probe_detail),
                "max_hold_seconds": int(pos.max_hold_seconds),
            }
        )
        self._active_trade_decision_context = None
        self._save_state()
        return pos

    def _passes_token_guards(self, token_data: dict[str, Any]) -> bool:
        if abs(float(token_data.get("price_change_5m") or 0)) > float(config.MAX_TOKEN_PRICE_CHANGE_5M_ABS_PERCENT):
            return False

        safety = token_data.get("safety") if isinstance(token_data.get("safety"), dict) else {}
        is_contract_safe = bool(token_data.get("is_contract_safe", safety.get("is_safe", False)))
        warning_flags = int(token_data.get("warning_flags", len((safety or {}).get("warnings") or [])) or 0)
        risk_level = str(token_data.get("risk_level", (safety or {}).get("risk_level", "HIGH"))).upper()

        if config.SAFE_TEST_MODE:
            if float(token_data.get("liquidity") or 0) < float(config.SAFE_MIN_LIQUIDITY_USD):
                return False
            if float(token_data.get("volume_5m") or 0) < float(config.SAFE_MIN_VOLUME_5M_USD):
                return False
            if int(token_data.get("age_seconds") or 0) < int(config.SAFE_MIN_AGE_SECONDS):
                return False
            if abs(float(token_data.get("price_change_5m") or 0)) > float(config.SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT):
                return False
            if config.SAFE_REQUIRE_CONTRACT_SAFE and not is_contract_safe:
                return False
            required_risk = str(config.SAFE_REQUIRE_RISK_LEVEL).upper()
            rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            if rank.get(risk_level, 2) > rank.get(required_risk, 1):
                return False
            if warning_flags > int(config.SAFE_MAX_WARNING_FLAGS):
                return False
        return True

    def _apply_max_loss_per_trade_cap(
        self,
        position_size_usd: float,
        stop_loss_percent: int,
        total_cost_percent: float,
        gas_usd: float,
    ) -> float:
        max_loss_pct = float(config.MAX_LOSS_PER_TRADE_PERCENT_BALANCE)
        if max_loss_pct <= 0:
            return position_size_usd
        equity = max(0.1, self._equity_usd())
        max_allowed_loss_usd = equity * (max_loss_pct / 100)
        worst_loss_ratio = max(0.0, abs(float(stop_loss_percent)) + float(total_cost_percent)) / 100
        if worst_loss_ratio <= 0:
            return position_size_usd
        max_size_by_risk = max(0.0, (max_allowed_loss_usd - gas_usd) / worst_loss_ratio)
        return max(0.0, min(position_size_usd, max_size_by_risk))

    @staticmethod
    def _partial_tp_stage(position: PaperPosition) -> int:
        try:
            stage = int(getattr(position, "partial_tp_stage", 0) or 0)
        except Exception:
            stage = 0
        if stage <= 0 and bool(getattr(position, "partial_tp_done", False)):
            stage = 1
        return max(0, stage)

    @staticmethod
    def _set_partial_tp_stage(position: PaperPosition, stage: int, trigger_percent: float = 0.0) -> None:
        safe_stage = max(0, int(stage))
        position.partial_tp_stage = safe_stage
        position.partial_tp_done = safe_stage > 0
        position.last_partial_tp_trigger_percent = max(0.0, float(trigger_percent))

    @staticmethod
    def _partial_tp_be_protect_floor_percent(position: PaperPosition) -> float:
        base_floor = float(getattr(config, "PAPER_PARTIAL_TP_BE_PROTECT_FLOOR_PERCENT", 0.05) or 0.05)
        floor = base_floor
        stage = AutoTrader._partial_tp_stage(position)
        if stage >= 2:
            floor += max(0.0, float(getattr(config, "PAPER_PARTIAL_TP_BE_PROTECT_STAGE2_BONUS_PERCENT", 0.10) or 0.10))
        return float(floor)

    def _exit_profile(
        self,
        position: PaperPosition,
        *,
        age_seconds: int = 0,
    ) -> dict[str, float | str | bool]:
        out: dict[str, float | str | bool] = {
            "enabled": bool(getattr(config, "EXIT_PROFILE_ENABLED", True)),
            "label": "neutral",
            "hold_mult": 1.0,
            "no_momentum_age_shift_seconds": 0,
            "no_momentum_pnl_shift": 0.0,
            "pre_partial_max_loss_shift": 0.0,
            "post_partial_giveback_mult": 1.0,
            "ev_expected_net_usd": float(getattr(position, "ev_expected_net_usd", 0.0) or 0.0),
            "ev_confidence": float(getattr(position, "ev_confidence", 0.0) or 0.0),
        }
        if not bool(out["enabled"]):
            return out

        ev_net = float(out["ev_expected_net_usd"])
        conf = float(out["ev_confidence"])
        lane = str(getattr(position, "execution_lane", "") or "").strip().lower()
        tier = str(getattr(position, "entry_tier", "") or "").strip().upper()
        channel = str(getattr(position, "entry_channel", "") or "").strip().lower()
        cluster_samples = float(getattr(position, "cluster_ev_samples", 0.0) or 0.0)
        cluster_loss_share = float(getattr(position, "cluster_ev_loss_share", 0.0) or 0.0)
        strong_ev = float(getattr(config, "EXIT_PROFILE_STRONG_MIN_EV_USD", 0.0065) or 0.0065)
        strong_conf = max(0.0, min(1.0, float(getattr(config, "EXIT_PROFILE_STRONG_MIN_CONFIDENCE", 0.35) or 0.35)))
        weak_ev = float(getattr(config, "EXIT_PROFILE_WEAK_MAX_EV_USD", 0.0012) or 0.0012)
        weak_conf = max(0.0, min(1.0, float(getattr(config, "EXIT_PROFILE_WEAK_MIN_CONFIDENCE", 0.25) or 0.25)))
        cluster_min_samples = max(0.0, float(getattr(config, "EXIT_PROFILE_CLUSTER_MIN_SAMPLES", 10) or 10))
        cluster_weak_loss = max(0.0, min(1.0, float(getattr(config, "EXIT_PROFILE_CLUSTER_WEAK_LOSS_SHARE", 0.68) or 0.68)))

        strong = (conf >= strong_conf) and (ev_net >= strong_ev)
        if (not strong) and lane == "runner" and tier == "A" and channel == "core":
            strong = ev_net >= (strong_ev * 0.75)
        weak = (conf >= weak_conf) and (ev_net <= weak_ev)
        if cluster_samples >= cluster_min_samples and cluster_loss_share >= cluster_weak_loss:
            weak = True

        if strong and weak:
            if ev_net >= strong_ev:
                weak = False
            else:
                strong = False

        if strong:
            out["label"] = "strong"
            out["hold_mult"] = max(0.20, float(getattr(config, "EXIT_PROFILE_STRONG_HOLD_MULT", 1.18) or 1.18))
            out["no_momentum_age_shift_seconds"] = int(
                getattr(config, "EXIT_PROFILE_STRONG_NO_MOMENTUM_AGE_SHIFT_SECONDS", 28) or 28
            )
            out["no_momentum_pnl_shift"] = float(
                getattr(config, "EXIT_PROFILE_STRONG_NO_MOMENTUM_PNL_SHIFT", -0.06) or -0.06
            )
            out["pre_partial_max_loss_shift"] = float(
                getattr(config, "EXIT_PROFILE_STRONG_PRE_PARTIAL_MAX_LOSS_SHIFT", -0.20) or -0.20
            )
            out["post_partial_giveback_mult"] = max(
                0.10,
                float(getattr(config, "EXIT_PROFILE_STRONG_POST_PARTIAL_GIVEBACK_MULT", 0.86) or 0.86),
            )
        elif weak:
            out["label"] = "weak"
            out["hold_mult"] = max(0.20, float(getattr(config, "EXIT_PROFILE_WEAK_HOLD_MULT", 0.82) or 0.82))
            out["no_momentum_age_shift_seconds"] = int(
                getattr(config, "EXIT_PROFILE_WEAK_NO_MOMENTUM_AGE_SHIFT_SECONDS", -22) or -22
            )
            out["no_momentum_pnl_shift"] = float(
                getattr(config, "EXIT_PROFILE_WEAK_NO_MOMENTUM_PNL_SHIFT", 0.14) or 0.14
            )
            out["pre_partial_max_loss_shift"] = float(
                getattr(config, "EXIT_PROFILE_WEAK_PRE_PARTIAL_MAX_LOSS_SHIFT", 0.24) or 0.24
            )
            out["post_partial_giveback_mult"] = max(
                0.10,
                float(getattr(config, "EXIT_PROFILE_WEAK_POST_PARTIAL_GIVEBACK_MULT", 1.18) or 1.18),
            )

        # Keep very old positions from being held forever by strong profile.
        if str(out.get("label", "")) == "strong":
            if int(age_seconds) >= max(1, int(position.max_hold_seconds or 1)) * 2:
                out["hold_mult"] = min(float(out.get("hold_mult", 1.0) or 1.0), 1.0)
                out["label"] = "strong_late_trim"
        return out

    def _effective_timeout_seconds(
        self,
        position: PaperPosition,
        age_seconds: int | None = None,
        exit_profile: dict[str, float | str | bool] | None = None,
    ) -> int:
        base_hold = max(1, int(position.max_hold_seconds or 1))
        if not bool(getattr(config, "EXIT_TIMEOUT_EXTENSION_ENABLED", False)):
            position.timeout_extension_seconds = 0
            out_hold = base_hold
        else:
            extension = 0
            max_ext = max(0, int(getattr(config, "EXIT_TIMEOUT_EXTENSION_MAX_SECONDS", 0) or 0))
            if max_ext <= 0:
                position.timeout_extension_seconds = 0
                out_hold = base_hold
            else:
                edge_gate = float(getattr(config, "EXIT_TIMEOUT_EXTENSION_MIN_EDGE_PERCENT", 0.0) or 0.0)
                if float(position.expected_edge_percent) >= edge_gate:
                    extension += max(0, int(getattr(config, "EXIT_TIMEOUT_EXTENSION_EDGE_SECONDS", 0) or 0))

                peak_gate = float(getattr(config, "EXIT_TIMEOUT_EXTENSION_MIN_PEAK_PERCENT", 0.0) or 0.0)
                pnl_gate = float(getattr(config, "EXIT_TIMEOUT_EXTENSION_MIN_PNL_PERCENT", -999.0) or -999.0)
                if float(position.peak_pnl_percent) >= peak_gate and float(position.pnl_percent) >= pnl_gate:
                    extension += max(0, int(getattr(config, "EXIT_TIMEOUT_EXTENSION_MOMENTUM_SECONDS", 0) or 0))

                if self._partial_tp_stage(position) > 0:
                    extension += max(0, int(getattr(config, "EXIT_TIMEOUT_EXTENSION_POST_PARTIAL_SECONDS", 0) or 0))

                if self._is_strong_trade(position):
                    strong_extra = max(0, int(getattr(config, "EXIT_TIMEOUT_STRONG_TRADE_EXTRA_SECONDS", 0) or 0))
                    extension += strong_extra
                    max_ext += strong_extra

                extension = max(0, min(max_ext, int(extension)))
                position.timeout_extension_seconds = extension
                out_hold = int(base_hold + extension)

        if bool(getattr(config, "STATEFUL_EXITS_ENABLED", True)):
            lane = str(getattr(position, "execution_lane", "") or "").strip().lower()
            if lane == "runner":
                out_hold += max(0, int(getattr(config, "STATEFUL_TIMEOUT_RUNNER_EXTRA_SECONDS", 26) or 26))
            elif lane == "scalp":
                out_hold -= max(0, int(getattr(config, "STATEFUL_TIMEOUT_SCALP_REDUCE_SECONDS", 14) or 14))
            if age_seconds is not None and lane == "runner":
                phase = self._execution_phase(position, int(age_seconds))
                if phase == "late":
                    out_hold += max(
                        0,
                        int(getattr(config, "STATEFUL_TIMEOUT_RUNNER_EXTRA_SECONDS", 26) or 26) // 2,
                    )
        profile = dict(exit_profile or {})
        if not profile:
            profile = self._exit_profile(position, age_seconds=max(0, int(age_seconds or 0)))
        out_hold = int(round(float(out_hold) * max(0.20, float(profile.get("hold_mult", 1.0) or 1.0))))
        return int(max(20, out_hold))

    @staticmethod
    def _is_strong_trade(position: PaperPosition) -> bool:
        edge_gate = float(getattr(config, "EXIT_TIMEOUT_STRONG_TRADE_MIN_EXPECTED_EDGE_PERCENT", 2.8) or 2.8)
        if float(position.expected_edge_percent) >= edge_gate:
            return True
        if bool(getattr(config, "EXIT_TIMEOUT_STRONG_TRADE_ALLOW_FOR_TIER_A", True)) and str(position.entry_tier or "").strip().upper() == "A":
            return True
        if bool(getattr(config, "EXIT_TIMEOUT_STRONG_TRADE_ALLOW_FOR_CORE", True)) and str(position.entry_channel or "").strip().lower() == "core":
            return True
        return False

    def _no_momentum_gate(
        self,
        position: PaperPosition,
        age_seconds: int,
        exit_profile: dict[str, float | str | bool] | None = None,
    ) -> tuple[int, float]:
        age_gate = max(
            int(position.max_hold_seconds * max(0.0, min(100.0, float(config.NO_MOMENTUM_EXIT_MIN_AGE_PERCENT))) / 100.0),
            int(config.NO_MOMENTUM_EXIT_MIN_HOLD_SECONDS),
        )
        pnl_gate = float(config.NO_MOMENTUM_EXIT_MAX_PNL_PERCENT)
        if bool(getattr(config, "NO_MOMENTUM_STRONG_TRADE_PROTECT_ENABLED", True)):
            strong_edge = float(getattr(config, "NO_MOMENTUM_STRONG_EXPECTED_EDGE_PERCENT", 2.8) or 2.8)
            if float(position.expected_edge_percent) >= strong_edge or self._is_strong_trade(position):
                age_gate += max(0, int(position.max_hold_seconds * max(0.0, float(getattr(config, "NO_MOMENTUM_STRONG_MIN_AGE_EXTRA_PERCENT", 18.0) or 18.0)) / 100.0))
                age_gate += max(0, int(getattr(config, "NO_MOMENTUM_STRONG_MIN_HOLD_EXTRA_SECONDS", 40) or 40))
                pnl_gate += float(getattr(config, "NO_MOMENTUM_STRONG_MAX_PNL_SHIFT", -0.10) or -0.10)
        if bool(getattr(config, "STATEFUL_EXITS_ENABLED", True)):
            lane = str(getattr(position, "execution_lane", "") or "").strip().lower()
            if lane == "runner":
                age_gate += int(getattr(config, "STATEFUL_NO_MOMENTUM_RUNNER_AGE_SHIFT_SECONDS", 24) or 24)
                pnl_gate += float(getattr(config, "STATEFUL_NO_MOMENTUM_RUNNER_PNL_SHIFT", -0.10) or -0.10)
            elif lane == "scalp":
                age_gate += int(getattr(config, "STATEFUL_NO_MOMENTUM_SCALP_AGE_SHIFT_SECONDS", -10) or -10)
                pnl_gate += float(getattr(config, "STATEFUL_NO_MOMENTUM_SCALP_PNL_SHIFT", 0.06) or 0.06)
            phase = self._execution_phase(position, int(age_seconds))
            if phase == "late" and lane == "runner":
                age_gate += 8
            if phase == "early" and lane == "scalp":
                age_gate = max(0, age_gate - 5)
        profile = dict(exit_profile or {})
        if not profile:
            profile = self._exit_profile(position, age_seconds=max(0, int(age_seconds)))
        age_gate += int(profile.get("no_momentum_age_shift_seconds", 0) or 0)
        pnl_gate += float(profile.get("no_momentum_pnl_shift", 0.0) or 0.0)
        return int(max(0, age_gate)), float(pnl_gate)

    def _weak_trade_early_gate(self, position: PaperPosition, age_seconds: int) -> bool:
        if not bool(getattr(config, "WEAK_TRADE_EARLY_EXIT_ENABLED", True)):
            return False
        if AutoTrader._partial_tp_stage(position) > 0:
            return False
        lane = str(getattr(position, "execution_lane", "") or "").strip().lower()
        if bool(getattr(config, "STATEFUL_EXITS_ENABLED", True)):
            phase = self._execution_phase(position, int(age_seconds))
            if phase == "late" and lane == "runner":
                return False
        edge_max = float(getattr(config, "WEAK_TRADE_EARLY_EDGE_MAX_PERCENT", 1.85) or 1.85)
        if bool(getattr(config, "STATEFUL_EXITS_ENABLED", True)):
            if lane == "runner":
                edge_max += float(
                    getattr(config, "STATEFUL_WEAK_EARLY_RUNNER_EDGE_BONUS_PERCENT", 0.35) or 0.35
                )
            elif lane == "scalp":
                edge_max -= float(
                    getattr(config, "STATEFUL_WEAK_EARLY_SCALP_EDGE_PENALTY_PERCENT", 0.25) or 0.25
                )
        if float(position.expected_edge_percent) > edge_max:
            return False
        min_age_ratio = max(
            0.0,
            min(100.0, float(getattr(config, "WEAK_TRADE_EARLY_MIN_AGE_PERCENT", 22.0) or 22.0)),
        ) / 100.0
        min_age = max(
            int(position.max_hold_seconds * min_age_ratio),
            int(getattr(config, "WEAK_TRADE_EARLY_MIN_HOLD_SECONDS", 55) or 55),
        )
        if bool(getattr(config, "STATEFUL_EXITS_ENABLED", True)):
            if lane == "runner":
                min_age += 8
            elif lane == "scalp":
                min_age = max(0, min_age - 6)
        if int(age_seconds) < int(min_age):
            return False
        max_peak = float(getattr(config, "WEAK_TRADE_EARLY_MAX_PEAK_PERCENT", 0.95) or 0.95)
        if float(position.peak_pnl_percent) > max_peak:
            return False
        max_pnl = float(getattr(config, "WEAK_TRADE_EARLY_MAX_PNL_PERCENT", 0.18) or 0.18)
        if bool(getattr(config, "STATEFUL_EXITS_ENABLED", True)):
            if lane == "runner":
                max_pnl += 0.08
            elif lane == "scalp":
                max_pnl -= 0.05
        if float(position.pnl_percent) > max_pnl:
            return False
        return True

    def _trailing_floor_percent(self, position: PaperPosition) -> float | None:
        if not bool(getattr(config, "PAPER_TRAILING_EXIT_ENABLED", False)):
            return None

        peak = float(position.peak_pnl_percent)
        activate_peak = float(getattr(config, "PAPER_TRAILING_ACTIVATE_PEAK_PERCENT", 1.6) or 1.6)
        if peak < activate_peak:
            return None

        giveback_abs = max(0.0, float(getattr(config, "PAPER_TRAILING_GIVEBACK_PERCENT", 1.1) or 1.1))
        giveback_ratio = max(
            0.05,
            min(0.95, float(getattr(config, "PAPER_TRAILING_GIVEBACK_RATIO", 0.55) or 0.55)),
        )
        giveback = max(giveback_abs, peak * giveback_ratio)
        min_floor = float(getattr(config, "PAPER_TRAILING_MIN_PNL_PERCENT", 0.2) or 0.2)
        floor = max(min_floor, peak - giveback)
        return float(floor)

    def _should_exit_momentum_decay(self, position: PaperPosition, age_seconds: int) -> bool:
        if not bool(getattr(config, "MOMENTUM_DECAY_EXIT_ENABLED", False)):
            return False

        age_ratio = max(0.0, min(100.0, float(getattr(config, "MOMENTUM_DECAY_MIN_AGE_PERCENT", 18.0) or 18.0))) / 100.0
        min_age_by_ratio = int(position.max_hold_seconds * age_ratio)
        min_age = max(min_age_by_ratio, int(getattr(config, "MOMENTUM_DECAY_MIN_HOLD_SECONDS", 60) or 60))
        if int(age_seconds) < int(min_age):
            return False

        peak = float(position.peak_pnl_percent)
        pnl = float(position.pnl_percent)
        if peak < float(getattr(config, "MOMENTUM_DECAY_MIN_PEAK_PERCENT", 2.0) or 2.0):
            return False
        if pnl > float(getattr(config, "MOMENTUM_DECAY_MAX_PNL_PERCENT", 0.8) or 0.8):
            return False

        retain_ratio = max(0.0, min(1.0, float(getattr(config, "MOMENTUM_DECAY_RETAIN_RATIO", 0.35) or 0.35)))
        min_drop = max(0.0, float(getattr(config, "MOMENTUM_DECAY_MIN_DROP_PERCENT", 1.0) or 1.0))
        drop = peak - pnl
        if drop < min_drop:
            return False
        if peak <= 0.0:
            return False
        return (pnl / peak) <= retain_ratio

    def _asym_pre_partial_should_exit(
        self,
        position: PaperPosition,
        age_seconds: int,
        exit_profile: dict[str, float | str | bool] | None = None,
    ) -> bool:
        if not bool(getattr(config, "ASYMMETRIC_EXITS_ENABLED", True)):
            return False
        if not bool(getattr(config, "ASYM_PRE_PARTIAL_RISK_ENABLED", True)):
            return False
        if self._partial_tp_stage(position) > 0:
            return False

        min_age = max(0, int(getattr(config, "ASYM_PRE_PARTIAL_MIN_AGE_SECONDS", 24) or 24))
        max_loss = float(getattr(config, "ASYM_PRE_PARTIAL_MAX_LOSS_PERCENT", -1.35) or -1.35)
        if bool(getattr(config, "STATEFUL_EXITS_ENABLED", True)):
            lane = str(getattr(position, "execution_lane", "") or "").strip().lower()
            if lane == "runner":
                min_age += int(getattr(config, "STATEFUL_PRE_PARTIAL_RUNNER_MIN_AGE_EXTRA_SECONDS", 14) or 14)
                max_loss += float(getattr(config, "STATEFUL_PRE_PARTIAL_RUNNER_MAX_LOSS_SHIFT", -0.18) or -0.18)
            elif lane == "scalp":
                min_age += int(getattr(config, "STATEFUL_PRE_PARTIAL_SCALP_MIN_AGE_SHIFT_SECONDS", -8) or -8)
                max_loss += float(getattr(config, "STATEFUL_PRE_PARTIAL_SCALP_MAX_LOSS_SHIFT", 0.15) or 0.15)
        profile = dict(exit_profile or {})
        if not profile:
            profile = self._exit_profile(position, age_seconds=max(0, int(age_seconds)))
        max_loss += float(profile.get("pre_partial_max_loss_shift", 0.0) or 0.0)
        if int(age_seconds) < min_age:
            return False
        peak_cap = float(getattr(config, "ASYM_PRE_PARTIAL_MAX_PEAK_PERCENT", 0.90) or 0.90)
        if float(position.peak_pnl_percent) > peak_cap:
            return False
        return float(position.pnl_percent) <= max_loss

    def _asym_post_partial_floor_percent(
        self,
        position: PaperPosition,
        age_seconds: int = 0,
        exit_profile: dict[str, float | str | bool] | None = None,
    ) -> float | None:
        if not bool(getattr(config, "ASYMMETRIC_EXITS_ENABLED", True)):
            return None
        if not bool(getattr(config, "ASYM_POST_PARTIAL_PROTECT_ENABLED", True)):
            return None
        if self._partial_tp_stage(position) <= 0:
            return None

        peak = float(position.peak_pnl_percent)
        min_peak = float(getattr(config, "ASYM_POST_PARTIAL_MIN_PEAK_PERCENT", 0.80) or 0.80)
        if peak < min_peak:
            return None
        min_floor = float(getattr(config, "ASYM_POST_PARTIAL_PROTECT_MIN_FLOOR_PERCENT", 0.12) or 0.12)
        giveback = max(0.0, float(getattr(config, "ASYM_POST_PARTIAL_PROTECT_GIVEBACK_PERCENT", 0.55) or 0.55))
        if bool(getattr(config, "STATEFUL_EXITS_ENABLED", True)):
            lane = str(getattr(position, "execution_lane", "") or "").strip().lower()
            if lane == "runner":
                giveback *= float(getattr(config, "STATEFUL_POST_PARTIAL_RUNNER_GIVEBACK_MULT", 0.82) or 0.82)
            elif lane == "scalp":
                giveback *= float(getattr(config, "STATEFUL_POST_PARTIAL_SCALP_GIVEBACK_MULT", 1.12) or 1.12)
            if self._execution_phase(position, int(age_seconds)) == "late" and lane == "runner":
                giveback *= 0.90
        profile = dict(exit_profile or {})
        if not profile:
            profile = self._exit_profile(position, age_seconds=max(0, int(age_seconds)))
        giveback *= max(0.10, float(profile.get("post_partial_giveback_mult", 1.0) or 1.0))
        return float(max(min_floor, peak - giveback))

    async def process_open_positions(self, bot=None) -> None:
        await self._process_recovery_queue()
        await self._check_critical_conditions()
        self._maybe_log_paper_summary()
        if self._is_live_mode():
            self._maybe_discover_untracked_live_positions()
        if not self.open_positions:
            return

        state_dirty = False
        addresses = list(self.open_positions.keys())
        updates = await asyncio.gather(*[self._fetch_current_price(address) for address in addresses])
        update_map: dict[str, float] = {}
        for update in updates:
            if not update:
                continue
            address, price_usd = update
            if price_usd > 0:
                update_map[address] = price_usd

        for address in addresses:
            position = self.open_positions.get(address)
            if not position:
                continue
            price_usd = float(update_map.get(address, 0.0) or 0.0)
            has_price_update = price_usd > 0 and self._accept_price_update(position, price_usd)
            if has_price_update:
                position.current_price_usd = price_usd
                state_dirty = True
            base_price = float(position.current_price_usd if position.current_price_usd > 0 else position.entry_price_usd)
            raw_price_pnl_percent = ((base_price - position.entry_price_usd) / position.entry_price_usd) * 100
            if has_price_update:
                position.pnl_percent = raw_price_pnl_percent
                position.pnl_usd = (position.position_size_usd * raw_price_pnl_percent) / 100
                position.peak_pnl_percent = max(float(position.peak_pnl_percent), float(position.pnl_percent))

            if has_price_update and (not self._is_live_mode()):
                self._maybe_partial_take_profit(position, raw_price_pnl_percent)
                # Recompute MTM for remaining slice after partial TP.
                position.pnl_percent = raw_price_pnl_percent
                position.pnl_usd = (position.position_size_usd * raw_price_pnl_percent) / 100
                position.peak_pnl_percent = max(float(position.peak_pnl_percent), float(position.pnl_percent))

            should_close = False
            close_reason = ""
            trailing_floor = self._trailing_floor_percent(position) if has_price_update else None
            partial_stage = self._partial_tp_stage(position)
            age_seconds = int((datetime.now(timezone.utc) - position.opened_at).total_seconds())
            exit_profile = self._exit_profile(position, age_seconds=age_seconds)
            position.execution_phase = self._execution_phase(position, age_seconds)
            be_protect_floor = (
                self._partial_tp_be_protect_floor_percent(position)
                if (has_price_update and partial_stage > 0 and bool(getattr(config, "PAPER_PARTIAL_TP_BE_PROTECT_ENABLED", True)))
                else None
            )
            asym_post_partial_floor = (
                self._asym_post_partial_floor_percent(position, age_seconds, exit_profile=exit_profile)
                if has_price_update
                else None
            )

            if has_price_update and position.pnl_percent >= position.take_profit_percent:
                should_close = True
                close_reason = "TP"
            elif has_price_update and self._asym_pre_partial_should_exit(
                position,
                age_seconds,
                exit_profile=exit_profile,
            ):
                should_close = True
                close_reason = "EARLY_RISK"
            elif has_price_update and position.pnl_percent <= -abs(position.stop_loss_percent):
                should_close = True
                close_reason = "SL"
            elif (
                has_price_update
                and
                config.PROFIT_LOCK_ENABLED
                and float(position.peak_pnl_percent) >= float(config.PROFIT_LOCK_TRIGGER_PERCENT)
                and float(position.pnl_percent) <= float(config.PROFIT_LOCK_FLOOR_PERCENT)
            ):
                should_close = True
                close_reason = "PROFIT_LOCK"
            elif (
                has_price_update
                and asym_post_partial_floor is not None
                and float(position.pnl_percent) <= float(asym_post_partial_floor)
            ):
                should_close = True
                close_reason = "POST_PARTIAL_PROTECT"
            elif (
                has_price_update
                and be_protect_floor is not None
                and float(position.pnl_percent) <= float(be_protect_floor)
            ):
                should_close = True
                close_reason = "BE_PROTECT"
            elif (
                has_price_update
                and trailing_floor is not None
                and float(position.pnl_percent) <= float(trailing_floor)
            ):
                should_close = True
                close_reason = "TRAIL_FLOOR"
            else:
                effective_hold_seconds = self._effective_timeout_seconds(
                    position,
                    age_seconds,
                    exit_profile=exit_profile,
                )
                no_momentum_age_gate, no_momentum_pnl_gate = self._no_momentum_gate(
                    position,
                    age_seconds,
                    exit_profile=exit_profile,
                )
                if has_price_update and self._weak_trade_early_gate(position, age_seconds):
                    should_close = True
                    close_reason = "WEAK_EARLY"
                # Close "flat" positions early: token never showed real impulse and still sits near breakeven.
                if (
                    (not should_close)
                    and
                    has_price_update
                    and
                    config.NO_MOMENTUM_EXIT_ENABLED
                    and age_seconds >= no_momentum_age_gate
                    and float(position.peak_pnl_percent) <= float(config.NO_MOMENTUM_EXIT_MAX_PEAK_PERCENT)
                    and float(position.pnl_percent) <= float(no_momentum_pnl_gate)
                ):
                    should_close = True
                    close_reason = "NO_MOMENTUM"
                if should_close:
                    pass
                else:
                    if has_price_update and self._should_exit_momentum_decay(position, age_seconds):
                        should_close = True
                        close_reason = "MOMENTUM_DECAY"
                    else:
                        min_age_ratio = max(0.0, min(100.0, float(config.WEAKNESS_EXIT_MIN_AGE_PERCENT))) / 100.0
                        if (
                            has_price_update
                            and
                            config.WEAKNESS_EXIT_ENABLED
                            and age_seconds >= int(position.max_hold_seconds * min_age_ratio)
                            and float(position.pnl_percent) <= float(config.WEAKNESS_EXIT_PNL_PERCENT)
                            and float(position.peak_pnl_percent) < float(config.PROFIT_LOCK_TRIGGER_PERCENT)
                        ):
                            should_close = True
                            close_reason = "WEAKNESS"
                        elif age_seconds >= int(effective_hold_seconds):
                            should_close = True
                            close_reason = "TIMEOUT"

            if should_close:
                if self._is_live_mode():
                    await self._close_position_live(position, close_reason)
                else:
                    self._close_position(position, close_reason)
                if bot and int(config.PERSONAL_TELEGRAM_ID or 0) > 0:
                    await self._send_close_message(bot, position)

        if state_dirty:
            now_ts = time.time()
            if (now_ts - float(self._last_state_flush_ts)) >= float(self._state_flush_interval_seconds):
                self._save_state()
                self._last_state_flush_ts = now_ts

    async def _process_recovery_queue(self) -> None:
        if (not self.recovery_queue) and (not self.recovery_untracked):
            return
        pending: list[str] = []
        max_per_cycle = 2
        processed = 0

        for address in list(self.recovery_queue):
            normalized = normalize_address(address)
            if not normalized:
                continue
            pos = self.open_positions.get(normalized)
            if not pos:
                self._recovery_clear_tracking(normalized)
                continue
            if int(pos.token_amount_raw or 0) <= 0:
                self._recovery_clear_tracking(normalized)
                continue

            if not self._is_live_mode() or self.live_executor is None:
                pending.append(normalized)
                continue

            if not self._recovery_attempt_allowed(normalized):
                pending.append(normalized)
                continue

            attempts = self._recovery_record_attempt(normalized)
            processed += 1
            logger.warning(
                "RECOVERY queued_position token=%s address=%s amount_raw=%s attempt=%s",
                pos.symbol,
                normalized,
                int(pos.token_amount_raw),
                attempts,
            )
            await self._close_position_live(pos, "RECOVERY")
            if normalized in self.open_positions:
                pending.append(normalized)
            else:
                self._recovery_clear_tracking(normalized)
            if processed >= max_per_cycle:
                pending.extend(
                    [normalize_address(x) for x in self.recovery_queue if normalize_address(x) and normalize_address(x) not in pending]
                )
                break
        self.recovery_queue = pending

        if self.recovery_untracked and self._is_live_mode() and self.live_executor is not None:
            for address in list(self.recovery_untracked.keys()):
                normalized = normalize_address(address)
                if not normalized:
                    self.recovery_untracked.pop(address, None)
                    continue

                # Keep recovery_untracked in sync with real on-chain balance to avoid
                # stale "UNTRACKED" rows after a token has already been sold/transferred.
                amount_raw = int(self.recovery_untracked.get(normalized, 0) or 0)
                try:
                    amount_raw = int(self.live_executor.token_balance_raw(normalized))
                except Exception:
                    pass
                self.recovery_untracked[normalized] = amount_raw
                if amount_raw <= 0:
                    self.recovery_untracked.pop(normalized, None)
                    self._recovery_clear_tracking(normalized)
                    continue
                if not self._recovery_attempt_allowed(normalized):
                    continue

                attempts = self._recovery_record_attempt(normalized)
                logger.warning(
                    "RECOVERY untracked_attempt address=%s amount_raw=%s attempt=%s",
                    normalized,
                    amount_raw,
                    attempts,
                )
                try:
                    sell_result = await asyncio.to_thread(
                        self.live_executor.sell_token,
                        normalized,
                        amount_raw,
                    )
                    logger.warning(
                        "RECOVERY untracked_sold address=%s recv_eth=%.8f tx=%s",
                        normalized,
                        float(getattr(sell_result, "received_eth", 0.0) or 0.0),
                        str(getattr(sell_result, "tx_hash", "")),
                    )
                    self.recovery_untracked.pop(normalized, None)
                    self._recovery_clear_tracking(normalized)
                except Exception as exc:
                    txt = str(exc).lower()
                    logger.warning("RECOVERY untracked_sell_failed address=%s err=%s", normalized, exc)
                    if bool(getattr(config, "LIVE_ABANDON_UNSELLABLE_POSITIONS", False)) and (
                        "execution reverted" in txt
                        or "quote_failed" in txt
                        or "unsupported" in txt
                    ):
                        logger.warning("RECOVERY untracked_abandoned address=%s", normalized)
                        self.recovery_untracked.pop(normalized, None)
                        self._recovery_clear_tracking(normalized)

        if self.recovery_queue or self.recovery_untracked:
            self._save_state()

    async def _check_critical_conditions(self) -> None:
        if self._kill_switch_active():
            if not self.emergency_halt_reason:
                self.emergency_halt_reason = "KILL_SWITCH_FILE"
                self.emergency_halt_ts = datetime.now(timezone.utc).timestamp()
                logger.warning("KILL_SWITCH reason=file_detected path=%s", config.KILL_SWITCH_FILE)
            if self.open_positions:
                await self._force_close_all_positions("KILL_SWITCH")
            return
        if self.emergency_halt_reason == "KILL_SWITCH_FILE":
            self._clear_emergency_halt("kill_switch_removed")

        # Optional "low balance stop" (separate from the open-trade guard).
        if bool(getattr(config, "LOW_BALANCE_GUARD_ENABLED", True)):
            min_available = max(0.1, float(config.AUTO_STOP_MIN_AVAILABLE_USD))
            available = self._available_balance_usd()
            if available < min_available and not self.open_positions:
                self._activate_emergency_halt(
                    "LOW_BALANCE_STOP",
                    f"available=${available:.2f} min=${min_available:.2f}",
                )
            elif self.emergency_halt_reason == "LOW_BALANCE_STOP" and available >= min_available:
                self._clear_emergency_halt("balance_recovered")
        elif self.emergency_halt_reason == "LOW_BALANCE_STOP":
            # If the feature gets disabled, clear the halt so trading can resume.
            self._clear_emergency_halt("low_balance_stop_disabled")

    def _activate_emergency_halt(self, reason: str, detail: str = "") -> None:
        if self.emergency_halt_reason == reason:
            return
        self.emergency_halt_reason = reason
        self.emergency_halt_ts = datetime.now(timezone.utc).timestamp()
        logger.error("CRITICAL_AUTO_RESET reason=%s detail=%s", reason, detail)
        if config.KILL_SWITCH_FILE:
            try:
                kill_path = config.KILL_SWITCH_FILE
                kill_dir = os.path.dirname(kill_path)
                if kill_dir:
                    os.makedirs(kill_dir, exist_ok=True)
                with open(kill_path, "w", encoding="utf-8") as f:
                    f.write(f"{datetime.now(timezone.utc).isoformat()} {reason} {detail}\n")
                logger.warning("KILL_SWITCH reason=auto_created path=%s", kill_path)
            except Exception as exc:
                logger.warning("KILL_SWITCH create failed: %s", exc)
        self._save_state()

    def _clear_emergency_halt(self, detail: str = "") -> None:
        if not self.emergency_halt_reason:
            return
        logger.warning("CRITICAL_AUTO_RESET cleared reason=%s detail=%s", self.emergency_halt_reason, detail)
        self.emergency_halt_reason = ""
        self.emergency_halt_ts = 0.0
        self._save_state()

    async def _force_close_all_positions(self, reason: str) -> None:
        if not self.open_positions:
            return
        closed = 0
        failed = 0
        for position in list(self.open_positions.values()):
            try:
                if self._is_live_mode():
                    await self._close_position_live(position, reason)
                else:
                    self._close_position(position, reason)
                closed += 1
            except Exception as exc:
                failed += 1
                logger.error("AUTO_SELL forced_failed token=%s reason=%s err=%s", position.symbol, reason, exc)
        logger.warning("CRITICAL_AUTO_RESET close_all reason=%s closed=%s failed=%s", reason, closed, failed)

    async def _close_position_live(self, position: PaperPosition, reason: str) -> None:
        def _abandon_live_position(abandon_reason: str, detail: str = "") -> None:
            # WARNING: This does not execute an on-chain sell. It only updates the local state so
            # the bot is not permanently blocked by an unsellable/stuck token.
            position.status = "CLOSED"
            position.close_reason = f"ABANDON:{abandon_reason}"
            position.closed_at = datetime.now(timezone.utc)
            position.pnl_usd = -float(position.position_size_usd)
            position.pnl_percent = -100.0
            position.sell_tx_hash = ""
            position.sell_tx_status = "failed"

            self._pop_open_position(position.token_address)
            self._recovery_forget_address(position.token_address)
            self.closed_positions.append(position)
            self._prune_closed_positions()
            self.total_closed += 1
            self.total_losses += 1
            self.current_loss_streak += 1
            self.realized_pnl_usd += position.pnl_usd
            self.day_realized_pnl_usd += position.pnl_usd

            try:
                self._blacklist_add(position.token_address, f"abandoned:{self._short_error_text(abandon_reason)}")
            except Exception:
                pass

            # Clear halt so new entries are possible again.
            if self.emergency_halt_reason:
                self._clear_emergency_halt(f"abandoned_position:{self._short_error_text(abandon_reason)}")
            self._risk_governor_after_close(close_reason=f"ABANDON:{abandon_reason}", pnl_usd=float(position.pnl_usd))
            self._session_profit_lock_after_close(
                close_reason=f"ABANDON:{abandon_reason}",
                pnl_usd=float(position.pnl_usd),
            )

            logger.error(
                "AUTO_SELL Live ABANDON token=%s reason=%s detail=%s",
                position.symbol,
                abandon_reason,
                self._short_error_text(detail),
            )
            self._save_state()

        if self.live_executor is None:
            logger.error("AUTO_SELL live_failed token=%s reason=no_executor", position.symbol)
            self._live_sell_failures += 1
            if self._live_sell_failures >= 3:
                self._activate_emergency_halt("LIVE_SELL_FAILED", "live_executor_unavailable")
            return
        inv_ok, inv_reason = self._pre_sell_invariants(position)
        if not inv_ok:
            logger.error("AUTO_SELL live_failed token=%s reason=pre_sell_invariants detail=%s", position.symbol, inv_reason)
            self._live_sell_failures += 1
            if self._live_sell_failures >= 3:
                self._activate_emergency_halt("LIVE_SELL_FAILED", inv_reason)
            return
        try:
            sell_result = await asyncio.to_thread(
                self.live_executor.sell_token,
                position.token_address,
                int(position.token_amount_raw),
            )
        except Exception as exc:
            logger.error("AUTO_SELL live_failed token=%s reason=%s", position.symbol, exc)
            # If we cannot even fund the gas for SELL, continuing to retry is pointless and can
            # trap the session in a noisy loop. Fail closed into emergency halt.
            txt = str(exc).lower()
            if "insufficient funds for gas" in txt:
                if bool(getattr(config, "LIVE_ABANDON_UNSELLABLE_POSITIONS", False)):
                    _abandon_live_position("insufficient_funds_for_gas", str(exc))
                else:
                    self._activate_emergency_halt("LIVE_SELL_FAILED", "insufficient_funds_for_gas")
                return
            if bool(getattr(config, "LIVE_ABANDON_UNSELLABLE_POSITIONS", False)) and (
                "execution reverted" in txt
                or "quote_failed" in txt
                or "no data" in txt
                or "unsupported" in txt
                or "gas_estimate_too_high" in txt
            ):
                _abandon_live_position("unsellable_or_reverted", str(exc))
                return
            self._live_sell_failures += 1
            if self._live_sell_failures >= 3:
                if bool(getattr(config, "LIVE_ABANDON_UNSELLABLE_POSITIONS", False)):
                    _abandon_live_position("sell_failed_3x", str(exc))
                else:
                    self._activate_emergency_halt("LIVE_SELL_FAILED", str(exc))
            return
        self._live_sell_failures = 0

        weth_price = await self._resolve_weth_price_usd({})
        if weth_price <= 0:
            weth_price = max(1.0, float(self.last_weth_price_usd))
        received_usd = float(sell_result.received_eth) * weth_price
        pnl_usd = received_usd - float(position.position_size_usd)
        pnl_percent = (pnl_usd / position.position_size_usd * 100) if position.position_size_usd > 0 else 0.0

        position.status = "CLOSED"
        position.close_reason = reason
        position.closed_at = datetime.now(timezone.utc)
        position.pnl_usd = pnl_usd
        position.pnl_percent = pnl_percent
        position.sell_tx_hash = str(sell_result.tx_hash)
        position.sell_tx_status = "confirmed"
        self._record_tx_event()

        self._pop_open_position(position.token_address)
        self._recovery_forget_address(position.token_address)
        self.closed_positions.append(position)
        self._prune_closed_positions()
        self.total_closed += 1
        self.realized_pnl_usd += position.pnl_usd
        self.day_realized_pnl_usd += position.pnl_usd

        outcome = self._pnl_outcome(position.pnl_usd)
        if outcome == "win":
            self.total_wins += 1
            self.current_loss_streak = 0
        elif outcome == "loss":
            self.total_losses += 1
            self.current_loss_streak += 1
        else:
            self.current_loss_streak = 0
        self._risk_governor_after_close(close_reason=reason, pnl_usd=float(position.pnl_usd))
        self._session_profit_lock_after_close(close_reason=reason, pnl_usd=float(position.pnl_usd))
        self._apply_token_cooldown_after_close(
            token_address=position.token_address,
            symbol=position.symbol,
            close_reason=reason,
            pnl_usd=float(position.pnl_usd),
            pnl_percent=float(position.pnl_percent),
        )

        logger.info(
            "AUTO_SELL Live SELL token=%s reason=%s recv=%.8f ETH pnl=%.2f%% ($%.2f) tx=%s",
            position.symbol,
            reason,
            float(sell_result.received_eth),
            position.pnl_percent,
            position.pnl_usd,
            position.sell_tx_hash,
        )
        self._write_trade_decision(
            {
                "ts": time.time(),
                "decision_stage": "trade_close",
                "decision": "close",
                "reason": str(reason),
                "candidate_id": str(position.candidate_id or ""),
                "token_address": position.token_address,
                "symbol": position.symbol,
                "score": int(position.score),
                "entry_tier": str(position.entry_tier or ""),
                "entry_channel": str(position.entry_channel or ""),
                "execution_lane": str(position.execution_lane or ""),
                "market_mode": str(position.market_mode or ""),
                "pnl_percent": float(position.pnl_percent),
                "pnl_usd": float(position.pnl_usd),
                "max_hold_seconds": int(position.max_hold_seconds),
                "closed_mode": "live",
            }
        )
        self._save_state()

    def _close_position(self, position: PaperPosition, reason: str) -> None:
        self._refresh_daily_window()
        raw_price_pnl_percent = ((position.current_price_usd - position.entry_price_usd) / position.entry_price_usd) * 100
        effective_price_pnl_percent = raw_price_pnl_percent
        if config.PAPER_REALISM_CAP_ENABLED:
            max_gain = max(0.0, float(config.PAPER_REALISM_MAX_GAIN_PERCENT))
            max_loss = max(0.0, float(config.PAPER_REALISM_MAX_LOSS_PERCENT))
            min_bound = -abs(max_loss)
            max_bound = abs(max_gain)
            capped = max(min_bound, min(max_bound, effective_price_pnl_percent))
            if capped != effective_price_pnl_percent:
                logger.info(
                    "AUTO_SELL realism_cap token=%s raw=%.2f%% capped=%.2f%%",
                    position.symbol,
                    effective_price_pnl_percent,
                    capped,
                )
                effective_price_pnl_percent = capped

        gross_value_usd = position.position_size_usd * (1 + effective_price_pnl_percent / 100)

        if config.PAPER_REALISM_ENABLED:
            total_percent_cost = (position.buy_cost_percent + position.sell_cost_percent) / 100
            final_value_usd = gross_value_usd * (1 - total_percent_cost) - position.gas_cost_usd
        else:
            final_value_usd = gross_value_usd
        final_value_usd = max(0.0, final_value_usd)
        pnl_usd_remaining = final_value_usd - position.position_size_usd
        total_pnl_usd = float(pnl_usd_remaining) + float(position.partial_realized_pnl_usd)
        base_size_usd = max(float(position.original_position_size_usd or 0.0), float(position.position_size_usd), 0.000001)
        total_pnl_percent = (total_pnl_usd / base_size_usd * 100) if base_size_usd > 0 else 0.0

        position.status = "CLOSED"
        position.close_reason = reason
        position.closed_at = datetime.now(timezone.utc)
        position.pnl_usd = total_pnl_usd
        position.pnl_percent = total_pnl_percent
        position.sell_tx_status = "simulated"
        self._record_tx_event()

        self._pop_open_position(position.token_address)
        self._recovery_forget_address(position.token_address)
        self.closed_positions.append(position)
        self._prune_closed_positions()
        self.total_closed += 1

        self.paper_balance_usd += final_value_usd
        self._update_stair_floor()
        self.realized_pnl_usd += pnl_usd_remaining
        self.day_realized_pnl_usd += pnl_usd_remaining

        outcome = self._pnl_outcome(total_pnl_usd)
        if outcome == "win":
            self.total_wins += 1
            self.current_loss_streak = 0
        elif outcome == "loss":
            self.total_losses += 1
            self.current_loss_streak += 1
        else:
            self.current_loss_streak = 0
        self._risk_governor_after_close(close_reason=reason, pnl_usd=float(total_pnl_usd))
        self._session_profit_lock_after_close(close_reason=reason, pnl_usd=float(total_pnl_usd))
        self._apply_token_cooldown_after_close(
            token_address=position.token_address,
            symbol=position.symbol,
            close_reason=reason,
            pnl_usd=float(total_pnl_usd),
            pnl_percent=float(total_pnl_percent),
        )

        logger.info(
            "AUTO_SELL Paper SELL token=%s reason=%s exit=$%.8f pnl=%.2f%% ($%.2f) remaining=$%.2f partial=$%.2f raw=%.2f%% cost=%.2f%% gas=$%.2f balance=$%.2f",
            position.symbol,
            reason,
            position.current_price_usd,
            position.pnl_percent,
            position.pnl_usd,
            pnl_usd_remaining,
            float(position.partial_realized_pnl_usd),
            raw_price_pnl_percent,
            position.buy_cost_percent + position.sell_cost_percent,
            position.gas_cost_usd,
            self.paper_balance_usd,
        )
        self._write_trade_decision(
            {
                "ts": time.time(),
                "decision_stage": "trade_close",
                "decision": "close",
                "reason": str(reason),
                "candidate_id": str(position.candidate_id or ""),
                "token_address": position.token_address,
                "symbol": position.symbol,
                "score": int(position.score),
                "entry_tier": str(position.entry_tier or ""),
                "entry_channel": str(position.entry_channel or ""),
                "execution_lane": str(position.execution_lane or ""),
                "market_mode": str(position.market_mode or ""),
                "pnl_percent": float(position.pnl_percent),
                "pnl_usd": float(position.pnl_usd),
                "max_hold_seconds": int(position.max_hold_seconds),
                "closed_mode": "paper",
            }
        )
        self._save_state()

    def _apply_partial_take_profit_slice(
        self,
        position: PaperPosition,
        *,
        stage: int,
        reason: str,
        trigger_percent: float,
        raw_price_pnl_percent: float,
        sell_fraction: float,
    ) -> bool:
        if float(position.position_size_usd) <= 0.0:
            return False

        frac = max(0.05, min(0.95, float(sell_fraction)))
        old_size = float(position.position_size_usd)
        min_remaining_usd = max(0.0, float(getattr(config, "PAPER_PARTIAL_TP_MIN_REMAINING_USD", 0.10) or 0.10))
        max_close_by_remaining = max(0.0, old_size - min_remaining_usd)
        close_size_usd = min(old_size * frac, max_close_by_remaining)
        if close_size_usd < 0.05:
            return False

        effective_price_pnl_percent = float(raw_price_pnl_percent)
        if config.PAPER_REALISM_CAP_ENABLED:
            max_gain = max(0.0, float(config.PAPER_REALISM_MAX_GAIN_PERCENT))
            max_loss = max(0.0, float(config.PAPER_REALISM_MAX_LOSS_PERCENT))
            effective_price_pnl_percent = max(-abs(max_loss), min(abs(max_gain), effective_price_pnl_percent))

        gross_value_usd = close_size_usd * (1 + effective_price_pnl_percent / 100.0)
        gas_fraction = 0.0 if old_size <= 0.0 else min(1.0, close_size_usd / old_size)
        gas_slice_usd = float(position.gas_cost_usd) * gas_fraction
        if config.PAPER_REALISM_ENABLED:
            total_percent_cost = (float(position.buy_cost_percent) + float(position.sell_cost_percent)) / 100.0
            final_value_usd = gross_value_usd * (1 - total_percent_cost) - gas_slice_usd
        else:
            final_value_usd = gross_value_usd
        final_value_usd = max(0.0, final_value_usd)
        realized_slice_pnl = final_value_usd - close_size_usd

        position.position_size_usd = max(0.0, old_size - close_size_usd)
        if float(position.original_position_size_usd or 0.0) <= 0.0:
            position.original_position_size_usd = old_size
        position.partial_realized_pnl_usd = float(position.partial_realized_pnl_usd) + float(realized_slice_pnl)
        self._set_partial_tp_stage(position, stage=stage, trigger_percent=trigger_percent)
        position.gas_cost_usd = max(0.0, float(position.gas_cost_usd) - gas_slice_usd)

        if int(position.token_amount_raw or 0) > 0:
            sold_raw = int(max(0, round(int(position.token_amount_raw) * gas_fraction)))
            position.token_amount_raw = max(0, int(position.token_amount_raw) - sold_raw)

        self.paper_balance_usd += final_value_usd
        self.realized_pnl_usd += realized_slice_pnl
        self.day_realized_pnl_usd += realized_slice_pnl
        self._update_stair_floor()

        if bool(getattr(config, "PAPER_PARTIAL_TP_MOVE_SL_TO_BREAK_EVEN", True)):
            be_floor = max(0.0, float(getattr(config, "PAPER_PARTIAL_TP_BREAK_EVEN_BUFFER_PERCENT", 0.2) or 0.2))
            if stage >= 2:
                be_floor += max(
                    0.0,
                    float(getattr(config, "PAPER_PARTIAL_TP_BREAK_EVEN_STAGE2_BONUS_PERCENT", 0.10) or 0.10),
                )
            position.stop_loss_percent = min(float(position.stop_loss_percent), be_floor)

        logger.info(
            "AUTO_SELL Paper PARTIAL token=%s reason=%s stage=%s trigger=%.2f%% sold=%.0f%% realized=$%.4f remaining_size=$%.2f sl=%.2f",
            position.symbol,
            reason,
            int(stage),
            float(trigger_percent),
            (close_size_usd / max(0.0001, old_size)) * 100.0,
            realized_slice_pnl,
            position.position_size_usd,
            float(position.stop_loss_percent),
        )
        self._write_trade_decision(
            {
                "ts": time.time(),
                "decision_stage": "trade_partial",
                "decision": "partial_take_profit",
                "reason": str(reason),
                "stage": int(stage),
                "trigger_percent": float(trigger_percent),
                "raw_price_pnl_percent": float(raw_price_pnl_percent),
                "candidate_id": str(position.candidate_id or ""),
                "token_address": position.token_address,
                "symbol": position.symbol,
                "score": int(position.score),
                "entry_tier": str(position.entry_tier or ""),
                "entry_channel": str(position.entry_channel or ""),
                "market_mode": str(position.market_mode or ""),
                "partial_realized_pnl_usd": float(realized_slice_pnl),
                "remaining_position_size_usd": float(position.position_size_usd),
            }
        )
        self._save_state()
        return True

    def _maybe_partial_take_profit(self, position: PaperPosition, raw_price_pnl_percent: float) -> None:
        if not bool(getattr(config, "PAPER_PARTIAL_TP_ENABLED", False)):
            return

        trigger_1 = float(getattr(config, "PAPER_PARTIAL_TP_TRIGGER_PERCENT", 2.5) or 2.5)
        trigger_1 *= max(0.2, float(getattr(position, "partial_tp_trigger_mult", 1.0) or 1.0))
        if float(raw_price_pnl_percent) < trigger_1:
            return

        stage = self._partial_tp_stage(position)
        if stage < 1:
            frac_1 = float(getattr(config, "PAPER_PARTIAL_TP_SELL_FRACTION", 0.5) or 0.5)
            frac_1 *= max(0.2, float(getattr(position, "partial_tp_sell_mult", 1.0) or 1.0))
            if self._apply_partial_take_profit_slice(
                position,
                stage=1,
                reason="TP1_PARTIAL",
                trigger_percent=trigger_1,
                raw_price_pnl_percent=raw_price_pnl_percent,
                sell_fraction=frac_1,
            ):
                stage = self._partial_tp_stage(position)

        if not bool(getattr(config, "PAPER_PARTIAL_TP_STAGE2_ENABLED", True)):
            return

        stage = self._partial_tp_stage(position)
        if stage >= 2:
            return

        stage2_mult = max(1.05, float(getattr(config, "PAPER_PARTIAL_TP_STAGE2_TRIGGER_MULT", 1.9) or 1.9))
        trigger_2 = trigger_1 * stage2_mult
        if float(raw_price_pnl_percent) < trigger_2:
            return

        frac_2 = float(getattr(config, "PAPER_PARTIAL_TP_STAGE2_SELL_FRACTION", 0.35) or 0.35)
        frac_2 *= max(0.2, float(getattr(position, "partial_tp_sell_mult", 1.0) or 1.0))
        self._apply_partial_take_profit_slice(
            position,
            stage=2,
            reason="TP2_PARTIAL",
            trigger_percent=trigger_2,
            raw_price_pnl_percent=raw_price_pnl_percent,
            sell_fraction=frac_2,
        )

    def clear_closed_positions(self, older_than_days: int = 0) -> int:
        before = len(self.closed_positions)
        if before == 0:
            return 0

        if older_than_days <= 0:
            self.closed_positions.clear()
        else:
            cutoff = datetime.now(timezone.utc).timestamp() - (older_than_days * 86400)
            kept: list[PaperPosition] = []
            for pos in self.closed_positions:
                closed_at = pos.closed_at or pos.opened_at
                ts = closed_at.timestamp()
                if ts >= cutoff:
                    kept.append(pos)
            self.closed_positions = kept

        removed = max(0, before - len(self.closed_positions))
        if removed > 0:
            self._save_state()
        return removed

    async def _send_close_message(self, bot, position: PaperPosition) -> None:
        try:
            await bot.send_message(
                chat_id=int(config.PERSONAL_TELEGRAM_ID),
                text=(
                    "<b>Paper Trade Closed</b>\n\n"
                    f"Token: <b>{position.symbol}</b>\n"
                    f"Reason: <b>{position.close_reason}</b>\n"
                    f"Entry: <code>${position.entry_price_usd:.8f}</code>\n"
                    f"Exit: <code>${position.current_price_usd:.8f}</code>\n"
                    f"PnL: <b>{position.pnl_percent:+.2f}% (${position.pnl_usd:+.2f})</b>\n"
                    f"Paper balance: <b>${self.paper_balance_usd:.2f}</b>"
                ),
                parse_mode="HTML",
            )
        except Exception as exc:
            logger.warning("Failed to send paper close message: %s", exc)

    def _maybe_log_paper_summary(self, force: bool = False) -> None:
        if self._is_live_mode():
            return
        interval = int(getattr(config, "PAPER_METRICS_SUMMARY_SECONDS", 900) or 900)
        if interval <= 0:
            return
        now_ts = datetime.now(timezone.utc).timestamp()
        if not force and (now_ts - float(self._last_paper_summary_ts)) < float(interval):
            return

        closed_delta = max(0, int(self.total_closed) - int(self._paper_summary_prev_closed))
        wins_delta = max(0, int(self.total_wins) - int(self._paper_summary_prev_wins))
        losses_delta = max(0, int(self.total_losses) - int(self._paper_summary_prev_losses))
        realized_delta_usd = float(self.realized_pnl_usd) - float(self._paper_summary_prev_realized_usd)
        win_rate_total = (float(self.total_wins) / float(self.total_closed) * 100.0) if int(self.total_closed) > 0 else 0.0

        logger.info(
            "PAPER_SUMMARY window=%ss closed_delta=%s wins_delta=%s losses_delta=%s realized_delta=$%.2f total_closed=%s winrate_total=%.1f%% realized_total=$%.2f balance=$%.2f equity=$%.2f open=%s",
            interval,
            closed_delta,
            wins_delta,
            losses_delta,
            realized_delta_usd,
            int(self.total_closed),
            win_rate_total,
            float(self.realized_pnl_usd),
            float(self.paper_balance_usd),
            float(self._equity_usd()),
            len(self.open_positions),
        )

        self._last_paper_summary_ts = now_ts
        self._paper_summary_prev_closed = int(self.total_closed)
        self._paper_summary_prev_wins = int(self.total_wins)
        self._paper_summary_prev_losses = int(self.total_losses)
        self._paper_summary_prev_realized_usd = float(self.realized_pnl_usd)

    async def _fetch_current_price(self, token_address: str) -> tuple[str, float] | None:
        token_address = normalize_address(token_address)
        if not token_address:
            return None
        url = f"{config.DEXSCREENER_API}/tokens/{token_address}"
        result = await self._http.get_json(
            url,
            source="dex_price",
            max_attempts=int(getattr(config, "HTTP_RETRY_ATTEMPTS", 3)),
        )
        if not result.ok or not isinstance(result.data, dict):
            return None
        data = result.data

        pairs = data.get("pairs", []) or []
        best_liq = -1.0
        best_price = 0.0
        for pair in pairs:
            if str(pair.get("chainId", "")).lower() != str(config.CHAIN_ID).lower():
                continue
            liq = float((pair.get("liquidity") or {}).get("usd") or 0)
            price = float(pair.get("priceUsd") or 0)
            if (not math.isfinite(price)) or price <= 0:
                continue
            if price > float(getattr(config, "PAPER_PRICE_MAX_ABSOLUTE_USD", 1_000_000.0) or 1_000_000.0):
                continue
            if liq > best_liq:
                best_liq = liq
                best_price = price
        if best_price <= 0:
            return None
        return token_address, best_price

    async def _resolve_weth_price_usd(self, token_data: dict[str, Any]) -> float:
        direct = float(token_data.get("weth_price_usd") or 0.0)
        if self._is_sane_weth_price_usd(direct):
            self.last_weth_price_usd = direct
            return direct
        if direct > 0:
            logger.warning("WETH price rejected as outlier source=token_data value=%.6f", direct)

        if self._is_sane_weth_price_usd(self.last_weth_price_usd):
            return self.last_weth_price_usd
        if self.last_weth_price_usd > 0:
            logger.warning(
                "WETH price cache rejected as outlier value=%.6f; trying on-chain/fallback",
                float(self.last_weth_price_usd),
            )

        weth_address = str(config.WETH_ADDRESS or "").strip().lower()
        if weth_address:
            fetched = await self._fetch_current_price(weth_address)
            if fetched:
                fetched_price = float(fetched[1] or 0.0)
                if self._is_sane_weth_price_usd(fetched_price):
                    self.last_weth_price_usd = fetched_price
                    return self.last_weth_price_usd
                if fetched_price > 0:
                    logger.warning("WETH price rejected as outlier source=dex_fetch value=%.6f", fetched_price)

        fallback = max(0.0, float(config.WETH_PRICE_FALLBACK_USD))
        if self._is_sane_weth_price_usd(fallback):
            self.last_weth_price_usd = fallback
            logger.info("AutoTrade cap fallback weth_price=$%.2f", fallback)
            return fallback

        return 0.0

    def _is_sane_weth_price_usd(self, price: float) -> bool:
        p = float(price or 0.0)
        if p <= 0:
            return False
        fallback = max(0.0, float(getattr(config, "WETH_PRICE_FALLBACK_USD", 0.0) or 0.0))
        # Guard against accidental poisoning by token-level quote fields.
        if fallback > 0:
            low = max(50.0, fallback * 0.20)
            high = max(low + 1.0, fallback * 5.00)
            return low <= p <= high
        return 50.0 <= p <= 20000.0

    def _accept_price_update(self, position: PaperPosition, next_price_usd: float) -> bool:
        if not config.PAPER_PRICE_GUARD_ENABLED:
            return True
        current_price = float(position.current_price_usd or 0.0)
        if current_price <= 0 or next_price_usd <= 0:
            return True

        # Hard reject physically implausible spikes even if repeated.
        ratio = max(next_price_usd, current_price) / max(min(next_price_usd, current_price), 1e-18)
        max_ratio = max(1.5, float(getattr(config, "PAPER_PRICE_GUARD_MAX_RATIO", 50.0) or 50.0))
        if (not math.isfinite(ratio)) or ratio > max_ratio:
            logger.warning(
                "AUTO_SELL price_guard_reject token=%s reason=abs_ratio ratio=%.6g max_ratio=%.2f current=%g next=%g",
                position.symbol,
                ratio,
                max_ratio,
                current_price,
                next_price_usd,
            )
            self.price_guard_pending.pop(position.token_address, None)
            return False

        jump_percent = abs((next_price_usd - current_price) / current_price) * 100
        if jump_percent <= float(config.PAPER_PRICE_GUARD_MAX_JUMP_PERCENT):
            self.price_guard_pending.pop(position.token_address, None)
            return True

        now_ts = datetime.now(timezone.utc).timestamp()
        pending = self.price_guard_pending.get(position.token_address)
        if pending:
            first_ts = float(pending.get("first_ts", 0.0))
            base_price = float(pending.get("price", 0.0))
            similar = False
            if base_price > 0:
                similar = abs((next_price_usd - base_price) / base_price) * 100 <= 20
            if (now_ts - first_ts) <= float(config.PAPER_PRICE_GUARD_WINDOW_SECONDS) and similar:
                pending["count"] = int(pending.get("count", 0)) + 1
                if int(pending["count"]) >= int(config.PAPER_PRICE_GUARD_CONFIRMATIONS):
                    self.price_guard_pending.pop(position.token_address, None)
                    logger.info(
                        "AUTO_SELL price_guard_confirm token=%s jump=%.2f%% confirmations=%s",
                        position.symbol,
                        jump_percent,
                        int(pending["count"]),
                    )
                    return True
                self.price_guard_pending[position.token_address] = pending
                logger.info(
                    "AUTO_SELL price_guard_pending token=%s jump=%.2f%% confirmations=%s/%s",
                    position.symbol,
                    jump_percent,
                    int(pending["count"]),
                    int(config.PAPER_PRICE_GUARD_CONFIRMATIONS),
                )
                return False

        self.price_guard_pending[position.token_address] = {
            "price": next_price_usd,
            "count": 1,
            "first_ts": now_ts,
        }
        logger.info(
            "AUTO_SELL price_guard_pending token=%s jump=%.2f%% confirmations=1/%s",
            position.symbol,
            jump_percent,
            int(config.PAPER_PRICE_GUARD_CONFIRMATIONS),
        )
        return False

    def get_stats(self) -> dict[str, float]:
        self._prune_hourly_window()
        self._prune_daily_tx_window()
        conc_window = max(300, int(getattr(config, "SYMBOL_CONCENTRATION_WINDOW_SECONDS", 3600) or 3600))
        self._prune_symbol_open_window(conc_window)
        symbol_counts: dict[str, int] = {}
        for _, sym in self._recent_open_symbols:
            symbol_counts[sym] = int(symbol_counts.get(sym, 0)) + 1
        top_symbol = ""
        top_count = 0
        if symbol_counts:
            top_symbol, top_count = max(symbol_counts.items(), key=lambda item: int(item[1]))
        conc_total = len(self._recent_open_symbols)
        top_share = (float(top_count) / float(conc_total)) if conc_total > 0 else 0.0
        win_rate = (self.total_wins / self.total_closed * 100) if self.total_closed else 0.0
        unrealized_pnl = sum(pos.pnl_usd for pos in self.open_positions.values())
        equity = self.paper_balance_usd + sum(pos.position_size_usd + pos.pnl_usd for pos in self.open_positions.values())
        drawdown_pct = self._risk_drawdown_percent()
        risk_block_reason = self._risk_governor_block_reason() or ""

        return {
            "open_trades": len(self.open_positions),
            "planned": self.total_plans,
            "executed": self.total_executed,
            "closed": self.total_closed,
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate_percent": round(win_rate, 2),
            "paper_balance_usd": round(self.paper_balance_usd, 2),
            # Keep control-loop inputs unrounded; cent rounding on a $7 bank hides real drift.
            "realized_pnl_usd": float(self.realized_pnl_usd),
            "session_peak_realized_pnl_usd": float(self._session_peak_realized_pnl_usd),
            "session_profit_lock_last_trigger_ts": round(self._session_profit_lock_last_trigger_ts, 2),
            "session_profit_lock_armed": bool(self._session_profit_lock_armed),
            "session_profit_lock_floor_usd": round(float(self._session_profit_lock_last_floor_usd or 0.0), 4),
            "session_profit_lock_metric_usd": round(float(self._session_profit_lock_last_metric_usd or 0.0), 4),
            "session_profit_lock_rearm_ready_ts": round(float(self._session_profit_lock_rearm_ready_ts or 0.0), 2),
            "day_realized_pnl_usd": float(self.day_realized_pnl_usd),
            "unrealized_pnl_usd": round(unrealized_pnl, 2),
            "equity_usd": round(equity, 2),
            "day_drawdown_percent": round(drawdown_pct, 2),
            "initial_balance_usd": round(self.initial_balance_usd, 2),
            "loss_streak": self.current_loss_streak,
            "trading_pause_until_ts": round(self.trading_pause_until_ts, 2),
            "last_pause_reason": self._last_pause_reason,
            "last_pause_trigger_ts": round(float(self._last_pause_trigger_ts or 0.0), 2),
            "risk_block_reason": risk_block_reason,
            "trades_last_hour": len(self.trade_open_timestamps),
            "tx_last_day": len(self.tx_event_timestamps),
            "stair_step_enabled": bool(config.STAIR_STEP_ENABLED),
            "stair_floor_usd": round(self.stair_floor_usd, 2),
            "stair_peak_balance_usd": round(self.stair_peak_balance_usd, 2),
            "reinvest_mult": round(float(self._last_reinvest_mult), 4),
            "available_balance_usd": round(self._available_balance_usd(), 2),
            "emergency_halt_reason": self.emergency_halt_reason,
            "emergency_halt_ts": round(self.emergency_halt_ts, 2),
            "data_policy_mode": self.data_policy_mode,
            "data_policy_reason": self.data_policy_reason,
            "recovery_queue": len(self.recovery_queue),
            "recovery_untracked": len(self.recovery_untracked),
            "skip_reasons_window": dict(self._skip_reason_counts_window),
            "anti_choke_active": self._anti_choke_active(),
            "entry_idle_seconds": round(self._entry_idle_seconds(), 1),
            "symbol_concentration_window_seconds": int(conc_window),
            "symbol_concentration_recent_opens": int(conc_total),
            "symbol_concentration_top_symbol": top_symbol,
            "symbol_concentration_top_share": round(float(top_share), 3),
        }

    def reset_paper_state(self, keep_closed: bool = True) -> None:
        self.initial_balance_usd = float(config.WALLET_BALANCE_USD)
        self.paper_balance_usd = float(config.WALLET_BALANCE_USD)
        self.realized_pnl_usd = 0.0
        self._session_peak_realized_pnl_usd = 0.0
        self._session_profit_lock_last_trigger_ts = 0.0
        self._session_profit_lock_armed = True
        self._session_profit_lock_rearm_ready_ts = 0.0
        self._session_profit_lock_rearm_floor_usd = 0.0
        self._session_profit_lock_last_floor_usd = 0.0
        self._session_profit_lock_last_metric_usd = 0.0
        self.day_id = self._current_day_id()
        self.day_start_equity_usd = float(config.WALLET_BALANCE_USD)
        self.day_realized_pnl_usd = 0.0
        self.current_loss_streak = 0
        self.trading_pause_until_ts = 0.0
        self._last_pause_reason = ""
        self._last_pause_detail = ""
        self._last_pause_trigger_ts = 0.0
        self.token_cooldowns.clear()
        self.token_cooldown_strikes.clear()
        self.symbol_cooldowns.clear()
        self.trade_open_timestamps.clear()
        self._recent_open_symbols.clear()
        self.tx_event_timestamps.clear()
        self.price_guard_pending.clear()
        self.emergency_halt_reason = ""
        self.emergency_halt_ts = 0.0
        self._live_sell_failures = 0
        self.stair_floor_usd = 0.0
        self.stair_peak_balance_usd = self.paper_balance_usd
        self._last_reinvest_mult = 1.0
        self.open_positions.clear()
        if not keep_closed:
            self.closed_positions.clear()
            self.total_closed = 0
            self.total_wins = 0
            self.total_losses = 0
        self._sync_stair_state()
        self._save_state()

    def flush_state(self) -> None:
        self._save_state()

    async def shutdown(self, reason: str = "shutdown") -> None:
        logger.info(
            "AUTOTRADER_SHUTDOWN reason=%s open=%s closed=%s",
            reason,
            len(self.open_positions),
            len(self.closed_positions),
        )
        self._save_state()
        await self._http.close()

    def _reconcile_live_state_on_startup(self) -> None:
        if not self._is_live_mode() or self.live_executor is None:
            return

        logger.warning("RECOVERY startup begin open_positions=%s", len(self.open_positions))
        now = datetime.now(timezone.utc)
        for key, position in list(self.open_positions.items()):
            addr = normalize_address(position.token_address or key)
            if not addr:
                continue
            try:
                onchain_amount = int(self.live_executor.token_balance_raw(addr))
            except Exception as exc:
                logger.warning("RECOVERY balance_check_failed token=%s err=%s", position.symbol, exc)
                continue

            if onchain_amount <= 0:
                logger.warning(
                    "RECOVERY closing_zero_balance token=%s address=%s stored_raw=%s",
                    position.symbol,
                    addr,
                    int(position.token_amount_raw or 0),
                )
                position.status = "CLOSED"
                position.close_reason = "RECOVERY_ZERO_BALANCE"
                position.closed_at = now
                position.pnl_percent = 0.0
                position.pnl_usd = 0.0
                position.sell_tx_status = "recovered"
                self.open_positions.pop(key, None)
                self.price_guard_pending.pop(key, None)
                self.closed_positions.append(position)
                self.total_closed += 1
                continue

            old_raw = int(position.token_amount_raw or 0)
            if old_raw != onchain_amount:
                position.token_amount_raw = onchain_amount
                self.recovery_queue.append(addr)
                logger.warning(
                    "RECOVERY sync_token_amount token=%s address=%s old_raw=%s new_raw=%s",
                    position.symbol,
                    addr,
                    old_raw,
                    onchain_amount,
                )

        self._discover_untracked_live_positions()
        self._save_state()

    def _candidate_log_path(self) -> str:
        raw = str(getattr(config, "CANDIDATE_DECISIONS_LOG_FILE", os.path.join("logs", "candidates.jsonl")) or "").strip()
        if os.path.isabs(raw):
            return raw
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        return os.path.join(root, raw)

    def _recent_candidate_addresses(self, max_rows: int = 500, max_bytes: int = 2_000_000) -> list[str]:
        path = self._candidate_log_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                start = max(0, size - int(max_bytes))
                f.seek(start)
                blob = f.read().decode("utf-8", errors="ignore")
        except Exception:
            return []
        lines = blob.splitlines()
        if len(lines) > max_rows:
            lines = lines[-max_rows:]
        out: list[str] = []
        seen: set[str] = set()
        for line in lines:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            addr = normalize_address(str(row.get("address", "") or ""))
            if not addr or addr in seen:
                continue
            if not bool(ADDRESS_RE.match(addr)):
                continue
            seen.add(addr)
            out.append(addr)
        return out

    def _discover_untracked_live_positions(self) -> None:
        if not self._is_live_mode() or self.live_executor is None:
            return
        max_discovery = int(getattr(config, "RECOVERY_DISCOVERY_MAX_ADDRESSES", 80) or 80)
        if max_discovery <= 0:
            return

        candidates: list[str] = []
        seen: set[str] = set()

        def _push(addr: str) -> None:
            a = normalize_address(addr)
            if not a or a in seen or a in self.open_positions:
                return
            if not bool(ADDRESS_RE.match(a)):
                return
            seen.add(a)
            candidates.append(a)

        for a in self._blacklist.keys():
            _push(str(a))
        for p in self.closed_positions[-300:]:
            _push(str(p.token_address))
        for a in self.token_cooldowns.keys():
            _push(str(a))
        for a in self._recent_candidate_addresses():
            _push(a)

        candidates = candidates[:max_discovery]
        for addr in candidates:
            try:
                amount_raw = int(self.live_executor.token_balance_raw(addr))
            except Exception:
                continue
            if amount_raw <= 0:
                continue
            if self._estimate_untracked_value_usd(addr, amount_raw) < float(
                getattr(config, "RECOVERY_UNTRACKED_MIN_VALUE_USD", 0.0) or 0.0
            ):
                continue
            prev = int(self.recovery_untracked.get(addr, 0) or 0)
            self.recovery_untracked[addr] = amount_raw
            if prev <= 0:
                logger.warning(
                    "RECOVERY untracked_detected address=%s amount_raw=%s",
                    addr,
                    amount_raw,
                )

    def _prune_recovery_untracked_live(self) -> None:
        if not self._is_live_mode() or self.live_executor is None:
            return
        if not self.recovery_untracked:
            return
        dirty = False
        for address in list(self.recovery_untracked.keys()):
            normalized = normalize_address(address)
            if not normalized:
                self.recovery_untracked.pop(address, None)
                dirty = True
                continue
            amount_raw = int(self.recovery_untracked.get(normalized, 0) or 0)
            try:
                amount_raw = int(self.live_executor.token_balance_raw(normalized))
            except Exception:
                pass
            if amount_raw <= 0:
                self.recovery_untracked.pop(normalized, None)
                self._recovery_clear_tracking(normalized)
                dirty = True
            else:
                min_usd = float(getattr(config, "RECOVERY_UNTRACKED_MIN_VALUE_USD", 0.0) or 0.0)
                if min_usd > 0 and self._estimate_untracked_value_usd(normalized, amount_raw) < min_usd:
                    self.recovery_untracked.pop(normalized, None)
                    self._recovery_clear_tracking(normalized)
                    dirty = True
                else:
                    self.recovery_untracked[normalized] = amount_raw
        if dirty:
            self._save_state()

    def _estimate_untracked_value_usd(self, token_address: str, amount_raw: int) -> float:
        if (not self._is_live_mode()) or self.live_executor is None:
            return 0.0
        if int(amount_raw or 0) <= 0:
            return 0.0
        try:
            out_eth = float(self.live_executor.quote_sell_eth(token_address, int(amount_raw)))
        except Exception:
            out_eth = 0.0
        if out_eth <= 0:
            return 0.0
        weth_usd = max(0.0, float(getattr(config, "WETH_PRICE_FALLBACK_USD", 0.0) or 0.0))
        return float(out_eth * weth_usd)

    def _maybe_discover_untracked_live_positions(self) -> None:
        now_ts = datetime.now(timezone.utc).timestamp()
        interval = float(getattr(config, "RECOVERY_DISCOVERY_INTERVAL_SECONDS", 45) or 45)
        if (now_ts - float(self._last_untracked_discovery_ts)) < max(10.0, interval):
            return
        self._last_untracked_discovery_ts = now_ts
        before = len(self.recovery_untracked)
        self._discover_untracked_live_positions()
        if len(self.recovery_untracked) != before:
            self._save_state()

    def _ev_expected_snapshot(
        self,
        *,
        market_mode: str,
        entry_tier: str,
        symbol: str,
        cluster_key: str,
        score: int,
        fallback_edge_usd: float,
    ) -> dict[str, float]:
        local_rows = self._closed_rows_window(
            window_minutes=int(getattr(config, "EV_FIRST_ENTRY_LOCAL_WINDOW_MINUTES", 360) or 360)
        )
        mm = str(market_mode or "").strip().upper()
        et = str(entry_tier or "").strip().upper()
        regime_local = self._ev_stats_from_pnls(
            [
                float(getattr(p, "pnl_usd", 0.0) or 0.0)
                for p in local_rows
                if str(getattr(p, "market_mode", "") or "").strip().upper() == mm
                and str(getattr(p, "entry_tier", "") or "").strip().upper() == et
            ]
        )
        cluster_local = self._cluster_window_metrics(cluster_key)
        symbol_local = self._symbol_ev_stats(symbol)
        regime_db = self._ev_db_stats(market_mode=mm, entry_tier=et, score=int(score))

        blended = self._blend_ev_stats([regime_db, regime_local, cluster_local, symbol_local])
        samples = float(blended.get("samples", 0.0) or 0.0)
        min_samples = max(5.0, float(getattr(config, "EV_FIRST_ENTRY_MIN_SAMPLES", 24) or 24))
        conf = max(0.0, min(1.0, samples / min_samples))
        # Keep some dependency on in-engine edge model for stability on sparse buckets.
        fallback_w_max = max(0.0, min(1.0, float(getattr(config, "EV_FIRST_ENTRY_FALLBACK_WEIGHT_MAX", 0.70) or 0.70)))
        fallback_w = (1.0 - conf) * fallback_w_max
        expected_net_usd = (
            (1.0 - fallback_w) * float(blended.get("avg_net_usd", 0.0) or 0.0)
            + fallback_w * float(fallback_edge_usd)
        )
        expected_net_usd -= max(0.0, float(getattr(config, "EV_FIRST_ENTRY_COST_HAIRCUT_USD", 0.0005) or 0.0005))
        return {
            "samples": float(samples),
            "confidence": float(conf),
            "win_rate": float(blended.get("win_rate", 0.0) or 0.0),
            "avg_win_usd": float(blended.get("avg_win_usd", 0.0) or 0.0),
            "avg_loss_usd": float(blended.get("avg_loss_usd", 0.0) or 0.0),
            "avg_net_usd": float(blended.get("avg_net_usd", 0.0) or 0.0),
            "expected_net_usd": float(expected_net_usd),
            "fallback_edge_usd": float(fallback_edge_usd),
            "loss_share": float(blended.get("loss_share", 0.0) or 0.0),
        }

    @staticmethod
    def _execution_phase(position: PaperPosition, age_seconds: int) -> str:
        hold = max(1, int(getattr(position, "max_hold_seconds", 1) or 1))
        age = max(0, int(age_seconds))
        ratio = float(age) / float(max(1, hold))
        early_ratio = max(
            0.05,
            min(0.95, float(getattr(config, "STATEFUL_EXITS_EARLY_PHASE_RATIO", 0.33) or 0.33)),
        )
        late_ratio = max(
            early_ratio + 0.05,
            min(0.99, float(getattr(config, "STATEFUL_EXITS_LATE_PHASE_RATIO", 0.78) or 0.78)),
        )
        if ratio < early_ratio:
            return "early"
        if ratio >= late_ratio:
            return "late"
        return "mid"

    def _execution_lane_profile(
        self,
        *,
        symbol: str,
        market_mode: str,
        entry_tier: str,
        entry_channel: str,
        score: int,
        expected_edge_percent: float,
        ev_snapshot: dict[str, float],
    ) -> dict[str, float | str]:
        out: dict[str, float | str] = {
            "enabled": bool(getattr(config, "EXECUTION_ARCH_ENABLED", True)),
            "lane": "scalp",
            "size_mult": 1.0,
            "hold_mult": 1.0,
            "edge_mult": 1.0,
            "detail": "disabled",
        }
        if not bool(out["enabled"]):
            return out

        if not bool(getattr(config, "EXECUTION_DUAL_LANE_ENABLED", True)):
            out["detail"] = "dual_lane_off"
            return out

        mm = str(market_mode or "").strip().upper()
        tier = str(entry_tier or "").strip().upper()
        channel = str(entry_channel or "").strip().lower()
        expected_net = float(ev_snapshot.get("expected_net_usd", 0.0) or 0.0)
        edge = float(expected_edge_percent)
        s = int(score)

        runner_min_edge = float(getattr(config, "EXECUTION_RUNNER_MIN_EDGE_PERCENT", 2.90) or 2.90)
        runner_min_ev = float(getattr(config, "EXECUTION_RUNNER_MIN_EV_USD", 0.0055) or 0.0055)
        runner_min_score = int(getattr(config, "EXECUTION_RUNNER_MIN_SCORE", 92) or 92)
        runner_allow_red = bool(getattr(config, "EXECUTION_RUNNER_ALLOW_IN_RED", False))
        runner_allow_explore = bool(getattr(config, "EXECUTION_RUNNER_ALLOW_EXPLORE", False))

        allow_runner = True
        reasons: list[str] = []
        if mm == "RED" and not runner_allow_red:
            allow_runner = False
            reasons.append("red_guard")
        if channel == "explore" and not runner_allow_explore:
            allow_runner = False
            reasons.append("explore_guard")

        runner_signal = (
            edge >= runner_min_edge
            or expected_net >= runner_min_ev
            or (s >= runner_min_score and tier == "A" and channel == "core")
        )
        lane = "runner" if (allow_runner and runner_signal) else "scalp"

        if lane == "runner":
            out["size_mult"] = float(getattr(config, "EXECUTION_RUNNER_SIZE_MULT", 1.10) or 1.10)
            out["hold_mult"] = float(getattr(config, "EXECUTION_RUNNER_HOLD_MULT", 1.32) or 1.32)
            out["edge_mult"] = float(getattr(config, "EXECUTION_RUNNER_EDGE_GATE_MULT", 0.94) or 0.94)
            out["detail"] = (
                f"runner edge={edge:.2f}/{runner_min_edge:.2f} ev=${expected_net:.4f}/${runner_min_ev:.4f} "
                f"score={s}/{runner_min_score}"
            )
        else:
            out["size_mult"] = float(getattr(config, "EXECUTION_SCALP_SIZE_MULT", 0.93) or 0.93)
            out["hold_mult"] = float(getattr(config, "EXECUTION_SCALP_HOLD_MULT", 0.78) or 0.78)
            out["edge_mult"] = float(getattr(config, "EXECUTION_SCALP_EDGE_GATE_MULT", 1.06) or 1.06)
            reason_txt = ",".join(reasons) if reasons else "signal_low"
            out["detail"] = (
                f"scalp reason={reason_txt} edge={edge:.2f}/{runner_min_edge:.2f} "
                f"ev=${expected_net:.4f}/${runner_min_ev:.4f}"
            )
        out["lane"] = lane
        return out

    def _cluster_ev_routing_adjustments(
        self,
        *,
        cluster_key: str,
        market_mode: str,
        entry_tier: str,
        entry_channel: str,
        candidate_id: str,
    ) -> dict[str, float | str | bool]:
        out: dict[str, float | str | bool] = {
            "enabled": bool(getattr(config, "EXECUTION_CLUSTER_ROUTING_ENABLED", True)),
            "size_mult": 1.0,
            "hold_mult": 1.0,
            "edge_mult": 1.0,
            "entry_probability": 1.0,
            "bad_ev": False,
            "severe_ev": False,
            "avg_net_usd": 0.0,
            "loss_share": 0.0,
            "samples": 0.0,
            "seed": "",
            "detail": "neutral",
        }
        if not bool(out["enabled"]):
            return out

        stats = self._cluster_window_metrics(cluster_key)
        samples = float(stats.get("samples", 0.0) or 0.0)
        avg_net = float(stats.get("avg_net_usd", 0.0) or 0.0)
        loss_share = float(stats.get("loss_share", 0.0) or 0.0)
        out["samples"] = samples
        out["avg_net_usd"] = avg_net
        out["loss_share"] = loss_share

        min_trades = max(2.0, float(getattr(config, "EXECUTION_CLUSTER_ROUTING_MIN_TRADES", 10) or 10))
        if samples < min_trades:
            out["detail"] = f"sparse samples={int(samples)}<{int(min_trades)}"
            return out

        bad_avg = float(getattr(config, "EXECUTION_CLUSTER_ROUTING_BAD_AVG_NET_USD", -0.0002) or -0.0002)
        bad_loss = float(getattr(config, "EXECUTION_CLUSTER_ROUTING_BAD_LOSS_SHARE", 0.60) or 0.60)
        severe_avg = float(getattr(config, "EXECUTION_CLUSTER_ROUTING_SEVERE_AVG_NET_USD", -0.0012) or -0.0012)
        severe_loss = float(getattr(config, "EXECUTION_CLUSTER_ROUTING_SEVERE_LOSS_SHARE", 0.72) or 0.72)

        bad = (avg_net <= bad_avg) or (loss_share >= bad_loss)
        severe = (avg_net <= severe_avg) or (loss_share >= severe_loss)
        out["bad_ev"] = bool(bad)
        out["severe_ev"] = bool(severe)
        if not bad:
            out["detail"] = f"ok samples={int(samples)} avg={avg_net:.5f} loss_share={loss_share:.2f}"
            return out

        if severe:
            out["size_mult"] = float(
                getattr(config, "EXECUTION_CLUSTER_ROUTING_SEVERE_SIZE_MULT", 0.62) or 0.62
            )
            out["hold_mult"] = float(
                getattr(config, "EXECUTION_CLUSTER_ROUTING_SEVERE_HOLD_MULT", 0.72) or 0.72
            )
            out["edge_mult"] = float(
                getattr(config, "EXECUTION_CLUSTER_ROUTING_SEVERE_EDGE_MULT", 1.28) or 1.28
            )
            out["entry_probability"] = float(
                getattr(config, "EXECUTION_CLUSTER_ROUTING_SEVERE_ENTRY_PROBABILITY", 0.48) or 0.48
            )
        else:
            out["size_mult"] = float(getattr(config, "EXECUTION_CLUSTER_ROUTING_BAD_SIZE_MULT", 0.84) or 0.84)
            out["hold_mult"] = float(getattr(config, "EXECUTION_CLUSTER_ROUTING_BAD_HOLD_MULT", 0.90) or 0.90)
            out["edge_mult"] = float(getattr(config, "EXECUTION_CLUSTER_ROUTING_BAD_EDGE_MULT", 1.10) or 1.10)
            out["entry_probability"] = float(
                getattr(config, "EXECUTION_CLUSTER_ROUTING_BAD_ENTRY_PROBABILITY", 0.80) or 0.80
            )

        if str(entry_tier or "").strip().upper() == "A" and str(entry_channel or "").strip().lower() == "core":
            out["edge_mult"] *= float(getattr(config, "EXECUTION_CLUSTER_ROUTING_A_CORE_RELAX_EDGE_MULT", 0.94) or 0.94)
        if str(market_mode or "").strip().upper() == "RED":
            out["edge_mult"] = float(out["edge_mult"]) * 1.04

        minute_slot = int(time.time() // 60)
        seed = (
            f"{str(cluster_key).lower()}|{str(market_mode).upper()}|{str(entry_tier).upper()}|"
            f"{str(entry_channel).lower()}|{str(candidate_id)}|{minute_slot}"
        )
        out["seed"] = seed
        out["detail"] = (
            f"bad={bad} severe={severe} samples={int(samples)} avg={avg_net:.5f} "
            f"loss_share={loss_share:.2f} size_mult={float(out['size_mult']):.2f} "
            f"hold_mult={float(out['hold_mult']):.2f} edge_mult={float(out['edge_mult']):.2f} "
            f"p={float(out['entry_probability']):.2f}"
        )
        return out

    def _token_ev_memory_adjustments(
        self,
        *,
        symbol: str,
        market_mode: str,
        entry_tier: str,
        candidate_id: str,
    ) -> dict[str, float | str | bool]:
        out: dict[str, float | str | bool] = {
            "enabled": bool(getattr(config, "TOKEN_EV_MEMORY_ENABLED", True)),
            "bad_ev": False,
            "severe_ev": False,
            "size_mult": 1.0,
            "edge_mult": 1.0,
            "entry_probability": 1.0,
            "avg_pnl_usd": 0.0,
            "loss_share": 0.0,
            "trades": 0.0,
            "seed": "",
            "detail": "neutral",
        }
        if not bool(out["enabled"]):
            return out

        stats = self._symbol_window_metrics(symbol)
        trades = float(stats.get("trades", 0) or 0.0)
        avg_pnl = float(stats.get("avg_pnl", 0.0) or 0.0)
        loss_share = float(stats.get("loss_share", 0.0) or 0.0)
        out["trades"] = trades
        out["avg_pnl_usd"] = avg_pnl
        out["loss_share"] = loss_share

        min_trades = max(2.0, float(getattr(config, "TOKEN_EV_MEMORY_MIN_TRADES", 4) or 4))
        if trades < min_trades:
            out["detail"] = f"sparse trades={int(trades)}<{int(min_trades)}"
            return out

        bad_avg = float(getattr(config, "TOKEN_EV_MEMORY_BAD_AVG_PNL_USD", 0.0002) or 0.0002)
        bad_loss_share = float(getattr(config, "TOKEN_EV_MEMORY_BAD_LOSS_SHARE", 0.62) or 0.62)
        severe_avg = float(getattr(config, "TOKEN_EV_MEMORY_SEVERE_AVG_PNL_USD", -0.0018) or -0.0018)
        severe_loss_share = float(getattr(config, "TOKEN_EV_MEMORY_SEVERE_LOSS_SHARE", 0.72) or 0.72)

        bad = (avg_pnl < bad_avg) or (loss_share > bad_loss_share)
        severe = (avg_pnl < severe_avg) or (loss_share > severe_loss_share)
        out["bad_ev"] = bool(bad)
        out["severe_ev"] = bool(severe)
        if not bad:
            out["detail"] = f"ok trades={int(trades)} avg={avg_pnl:.5f} loss_share={loss_share:.2f}"
            return out

        if severe:
            out["size_mult"] = float(getattr(config, "TOKEN_EV_MEMORY_SEVERE_SIZE_MULT", 0.58) or 0.58)
            out["edge_mult"] = float(getattr(config, "TOKEN_EV_MEMORY_SEVERE_EDGE_MULT", 1.28) or 1.28)
            out["entry_probability"] = float(getattr(config, "TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY", 0.42) or 0.42)
        else:
            out["size_mult"] = float(getattr(config, "TOKEN_EV_MEMORY_BAD_SIZE_MULT", 0.78) or 0.78)
            out["edge_mult"] = float(getattr(config, "TOKEN_EV_MEMORY_BAD_EDGE_MULT", 1.12) or 1.12)
            out["entry_probability"] = float(getattr(config, "TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY", 0.72) or 0.72)

        minute_slot = int(time.time() // 60)
        seed = f"{str(symbol).upper()}|{str(market_mode).upper()}|{str(entry_tier).upper()}|{str(candidate_id)}|{minute_slot}"
        out["seed"] = seed
        out["detail"] = (
            f"bad={bad} severe={severe} trades={int(trades)} avg={avg_pnl:.5f} "
            f"loss_share={loss_share:.2f} size_mult={float(out['size_mult']):.2f} "
            f"edge_mult={float(out['edge_mult']):.2f} p={float(out['entry_probability']):.2f}"
        )
        return out

    @staticmethod
    def _kelly_lite_multiplier(
        *,
        p_win: float,
        avg_win_usd: float,
        avg_loss_usd: float,
        confidence: float,
        expected_net_usd: float,
    ) -> tuple[float, float]:
        if not bool(getattr(config, "KELLY_LITE_ENABLED", True)):
            return 1.0, 0.0
        p = max(0.0, min(1.0, float(p_win)))
        w = max(0.0, float(avg_win_usd))
        l = max(0.000001, float(avg_loss_usd))
        if w <= 0.0 or float(expected_net_usd) <= 0.0:
            k = 0.0
        else:
            b = w / l
            q = 1.0 - p
            k = ((b * p) - q) / max(0.000001, b)
        k = max(0.0, min(float(getattr(config, "KELLY_LITE_MAX_FRACTION", 0.45) or 0.45), k))
        confidence_clamped = max(0.0, min(1.0, float(confidence)))
        min_samples = max(5.0, float(getattr(config, "KELLY_LITE_MIN_CONFIDENCE_SAMPLES", 18) or 18))
        conf_adj = min(1.0, confidence_clamped * (float(confidence_clamped) * (min_samples / max(5.0, min_samples))))
        min_mult = max(0.1, float(getattr(config, "KELLY_LITE_MIN_MULT", 0.55) or 0.55))
        max_mult = max(min_mult, float(getattr(config, "KELLY_LITE_MAX_MULT", 1.55) or 1.55))
        raw_mult = min_mult + ((max_mult - min_mult) * (k / max(0.000001, float(getattr(config, "KELLY_LITE_MAX_FRACTION", 0.45) or 0.45))))
        # Pull toward neutral when confidence is low.
        mult = 1.0 + ((raw_mult - 1.0) * conf_adj)
        return float(max(min_mult, min(max_mult, mult))), float(k)

    @staticmethod
    def _estimate_edge_percent(
        score: int,
        tp_percent: int,
        sl_percent: int,
        total_cost_percent: float,
        risk_level: str,
    ) -> float:
        dbg = AutoTrader._edge_debug_components(
            score=score,
            tp_percent=tp_percent,
            sl_percent=sl_percent,
            total_cost_percent=total_cost_percent,
            risk_level=risk_level,
        )
        expected = float(dbg["expected_percent"])
        return float(expected)

    @staticmethod
    def _edge_debug_components(
        *,
        score: int,
        tp_percent: int,
        sl_percent: int,
        total_cost_percent: float,
        risk_level: str,
    ) -> dict[str, float]:
        p_base = max(0.1, min(0.9, (score - 50) / 50))
        risk_level = str(risk_level).upper()
        risk_penalty = 0.0
        if risk_level == "HIGH":
            risk_penalty = 0.12
        elif risk_level == "MEDIUM":
            risk_penalty = 0.05
        p_win = max(0.05, min(0.95, p_base - risk_penalty))
        gross = (p_win * float(tp_percent)) - ((1.0 - p_win) * float(sl_percent))
        expected = float(gross) - float(total_cost_percent)
        return {
            "p_base": float(p_base),
            "p_win": float(p_win),
            "risk_penalty": float(risk_penalty),
            "gross_percent": float(gross),
            "expected_percent": float(expected),
        }

    def _reinvest_multiplier(self, expected_edge_percent: float) -> tuple[float, str]:
        if not bool(getattr(config, "V2_REINVEST_ENABLED", False)):
            self._last_reinvest_mult = 1.0
            return 1.0, "off"

        min_mult = max(0.2, float(getattr(config, "V2_REINVEST_MIN_MULT", 0.80) or 0.80))
        max_mult = max(min_mult, float(getattr(config, "V2_REINVEST_MAX_MULT", 1.50) or 1.50))
        growth_step_usd = max(0.05, float(getattr(config, "V2_REINVEST_GROWTH_STEP_USD", 0.60) or 0.60))
        growth_step_mult = max(0.0, float(getattr(config, "V2_REINVEST_STEP_MULT", 0.05) or 0.05))
        drawdown_cut_pct = max(0.0, float(getattr(config, "V2_REINVEST_DRAWDOWN_CUT_PERCENT", 2.5) or 2.5))
        drawdown_mult = max(0.2, float(getattr(config, "V2_REINVEST_DRAWDOWN_MULT", 0.80) or 0.80))
        loss_streak_step = max(0.0, float(getattr(config, "V2_REINVEST_LOSS_STREAK_STEP", 0.08) or 0.08))
        high_edge_threshold = float(getattr(config, "V2_REINVEST_HIGH_EDGE_THRESHOLD_PERCENT", 1.40) or 1.40)
        high_edge_bonus = max(0.0, float(getattr(config, "V2_REINVEST_HIGH_EDGE_BONUS", 0.06) or 0.06))

        equity = float(self._equity_usd())
        initial = max(0.01, float(self.initial_balance_usd or 0.01))
        growth_usd = max(0.0, equity - initial)
        growth_steps = int(growth_usd / growth_step_usd)
        mult = 1.0 + (float(growth_steps) * growth_step_mult)

        drawdown = float(self._risk_drawdown_percent())
        if drawdown <= -abs(drawdown_cut_pct):
            mult *= float(drawdown_mult)

        if int(self.current_loss_streak) > 0 and loss_streak_step > 0:
            mult *= max(0.35, 1.0 - (float(self.current_loss_streak) * loss_streak_step))

        if float(expected_edge_percent) >= high_edge_threshold and high_edge_bonus > 0:
            mult *= 1.0 + float(high_edge_bonus)

        mult = max(min_mult, min(max_mult, mult))
        self._last_reinvest_mult = float(mult)
        detail = (
            f"growth_steps={growth_steps} drawdown={drawdown:.2f}% "
            f"loss_streak={self.current_loss_streak} edge={float(expected_edge_percent):.2f}%"
        )
        return float(mult), detail

    def _choose_position_size(self, expected_edge_percent: float) -> float:
        min_size = max(0.1, float(config.PAPER_TRADE_SIZE_MIN_USD))
        max_size = max(min_size, float(config.PAPER_TRADE_SIZE_MAX_USD))
        if not config.DYNAMIC_POSITION_SIZING_ENABLED:
            return max_size

        # Map edge into [0,1] sizing factor.
        # 0% edge => min size, 25%+ edge => max size.
        factor = max(0.0, min(1.0, expected_edge_percent / 25.0))
        return min_size + (max_size - min_size) * factor

    @staticmethod
    def _estimate_cost_profile(
        liquidity_usd: float,
        risk_level: str,
        score: int,
        *,
        position_size_usd: float | None = None,
    ) -> dict[str, float]:
        buy_percent = float(config.PAPER_SWAP_FEE_BPS + config.PAPER_BASE_SLIPPAGE_BPS) / 100
        sell_percent = float(config.PAPER_SWAP_FEE_BPS + config.PAPER_BASE_SLIPPAGE_BPS) / 100

        # Token-sensitive slippage model.
        liq_extra_bps = 0.0
        if liquidity_usd < 5000:
            liq_extra_bps = 250.0
        elif liquidity_usd < 10000:
            liq_extra_bps = 150.0
        elif liquidity_usd < 20000:
            liq_extra_bps = 80.0
        else:
            liq_extra_bps = 30.0

        risk_extra_bps = 0.0
        risk_level = str(risk_level).upper()
        if risk_level == "HIGH":
            risk_extra_bps = 120.0
        elif risk_level == "MEDIUM":
            risk_extra_bps = 60.0
        else:
            risk_extra_bps = 20.0

        score_discount_bps = max(0.0, min(25.0, score - 70))
        side_slippage_bps = max(30.0, config.PAPER_BASE_SLIPPAGE_BPS + liq_extra_bps + risk_extra_bps - score_discount_bps)
        buy_percent = float(config.PAPER_SWAP_FEE_BPS + side_slippage_bps) / 100
        sell_percent = float(config.PAPER_SWAP_FEE_BPS + side_slippage_bps) / 100

        # Realistic gas model:
        # - buy + sell are always present,
        # - approve is not paid every trade when allowance is already set.
        approve_mult = 1.0 if bool(getattr(config, "PAPER_APPROVE_TX_ALWAYS", False)) else float(
            getattr(config, "PAPER_APPROVE_TX_PROBABILITY", 0.30) or 0.30
        )
        gas_tx_count = 2.0 + max(0.0, min(1.0, approve_mult))
        gas_usd = float(config.PAPER_GAS_PER_TX_USD) * float(gas_tx_count)
        if position_size_usd is None:
            size_ref_usd = float(getattr(config, "PAPER_TRADE_SIZE_USD", config.PAPER_TRADE_SIZE_MAX_USD) or 0.0)
        else:
            size_ref_usd = float(position_size_usd or 0.0)
        size_ref_usd = max(0.1, size_ref_usd)
        total_percent = buy_percent + sell_percent + (gas_usd / size_ref_usd * 100)

        return {
            "buy_percent": buy_percent,
            "sell_percent": sell_percent,
            "gas_usd": gas_usd,
            "total_percent": total_percent,
        }

    @staticmethod
    def _choose_hold_seconds(
        score: int,
        risk_level: str,
        liquidity_usd: float,
        volume_5m: float,
        price_change_5m: float,
    ) -> int:
        if not config.DYNAMIC_HOLD_ENABLED:
            return int(config.PAPER_MAX_HOLD_SECONDS)

        min_hold = int(config.HOLD_MIN_SECONDS)
        max_hold = int(config.HOLD_MAX_SECONDS)
        span = max(0, max_hold - min_hold)

        # Better score/liquidity/volume => can hold longer.
        score_factor = max(0.0, min(1.0, (score - 50) / 40.0))
        liq_factor = max(0.0, min(1.0, liquidity_usd / 25000.0))
        vol_factor = max(0.0, min(1.0, volume_5m / 15000.0))
        momentum_penalty = max(0.0, min(1.0, abs(price_change_5m) / 25.0))

        quality = (0.5 * score_factor) + (0.3 * liq_factor) + (0.2 * vol_factor)
        quality = max(0.0, quality - (0.15 * momentum_penalty))

        risk_level = str(risk_level).upper()
        if risk_level == "HIGH":
            quality *= 0.55
        elif risk_level == "MEDIUM":
            quality *= 0.8

        return int(min_hold + (span * max(0.0, min(1.0, quality))))

    def _save_state(self) -> None:
        auto_trader_state.save_state(self)

    def _load_state(self) -> None:
        auto_trader_state.load_state(self)

    @staticmethod
    def _serialize_pos(pos: PaperPosition) -> dict[str, Any]:
        return {
            "token_address": normalize_address(pos.token_address),
            "candidate_id": pos.candidate_id,
            "symbol": pos.symbol,
            "entry_price_usd": pos.entry_price_usd,
            "current_price_usd": pos.current_price_usd,
            "position_size_usd": pos.position_size_usd,
            "score": pos.score,
            "liquidity_usd": pos.liquidity_usd,
            "risk_level": pos.risk_level,
            "opened_at": pos.opened_at.isoformat(),
            "max_hold_seconds": pos.max_hold_seconds,
            "take_profit_percent": pos.take_profit_percent,
            "stop_loss_percent": pos.stop_loss_percent,
            "expected_edge_percent": pos.expected_edge_percent,
            "buy_cost_percent": pos.buy_cost_percent,
            "sell_cost_percent": pos.sell_cost_percent,
            "gas_cost_usd": pos.gas_cost_usd,
            "status": pos.status,
            "close_reason": pos.close_reason,
            "closed_at": pos.closed_at.isoformat() if pos.closed_at else None,
            "pnl_percent": pos.pnl_percent,
            "pnl_usd": pos.pnl_usd,
            "peak_pnl_percent": pos.peak_pnl_percent,
            "token_amount_raw": pos.token_amount_raw,
            "buy_tx_hash": pos.buy_tx_hash,
            "sell_tx_hash": pos.sell_tx_hash,
            "buy_tx_status": pos.buy_tx_status,
            "sell_tx_status": pos.sell_tx_status,
            "spent_eth": pos.spent_eth,
            "original_position_size_usd": pos.original_position_size_usd,
            "partial_tp_done": pos.partial_tp_done,
            "partial_tp_stage": pos.partial_tp_stage,
            "partial_realized_pnl_usd": pos.partial_realized_pnl_usd,
            "last_partial_tp_trigger_percent": pos.last_partial_tp_trigger_percent,
            "timeout_extension_seconds": pos.timeout_extension_seconds,
            "market_mode": pos.market_mode,
            "entry_tier": pos.entry_tier,
            "entry_channel": pos.entry_channel,
            "partial_tp_trigger_mult": pos.partial_tp_trigger_mult,
            "partial_tp_sell_mult": pos.partial_tp_sell_mult,
            "token_cluster_key": pos.token_cluster_key,
            "ev_expected_net_usd": pos.ev_expected_net_usd,
            "ev_confidence": pos.ev_confidence,
            "kelly_mult": pos.kelly_mult,
            "execution_lane": pos.execution_lane,
            "execution_phase": pos.execution_phase,
            "cluster_ev_avg_net_usd": pos.cluster_ev_avg_net_usd,
            "cluster_ev_loss_share": pos.cluster_ev_loss_share,
            "cluster_ev_samples": pos.cluster_ev_samples,
        }

    @staticmethod
    def _deserialize_pos(row: dict[str, Any]) -> PaperPosition | None:
        try:
            partial_stage_raw = row.get("partial_tp_stage", 1 if bool(row.get("partial_tp_done", False)) else 0)
            try:
                partial_stage = max(0, int(float(partial_stage_raw or 0)))
            except Exception:
                partial_stage = 1 if bool(row.get("partial_tp_done", False)) else 0

            opened_at = datetime.fromisoformat(str(row.get("opened_at")))
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            closed_at_raw = row.get("closed_at")
            closed_at = None
            if closed_at_raw:
                closed_at = datetime.fromisoformat(str(closed_at_raw))
                if closed_at.tzinfo is None:
                    closed_at = closed_at.replace(tzinfo=timezone.utc)
            return PaperPosition(
                token_address=normalize_address(str(row.get("token_address", ""))),
                candidate_id=str(row.get("candidate_id", "")),
                symbol=str(row.get("symbol", "N/A")),
                entry_price_usd=float(row.get("entry_price_usd", 0)),
                current_price_usd=float(row.get("current_price_usd", 0)),
                position_size_usd=float(row.get("position_size_usd", 0)),
                score=int(row.get("score", 0)),
                liquidity_usd=float(row.get("liquidity_usd", 0)),
                risk_level=str(row.get("risk_level", "MEDIUM")),
                opened_at=opened_at,
                max_hold_seconds=int(row.get("max_hold_seconds", config.PAPER_MAX_HOLD_SECONDS)),
                take_profit_percent=int(row.get("take_profit_percent", 50)),
                stop_loss_percent=float(row.get("stop_loss_percent", 30)),
                expected_edge_percent=float(row.get("expected_edge_percent", 0.0)),
                buy_cost_percent=float(row.get("buy_cost_percent", 0.0)),
                sell_cost_percent=float(row.get("sell_cost_percent", 0.0)),
                gas_cost_usd=float(row.get("gas_cost_usd", 0.0)),
                status=str(row.get("status", "OPEN")),
                close_reason=str(row.get("close_reason", "")),
                closed_at=closed_at,
                pnl_percent=float(row.get("pnl_percent", 0.0)),
                pnl_usd=float(row.get("pnl_usd", 0.0)),
                peak_pnl_percent=float(row.get("peak_pnl_percent", 0.0)),
                token_amount_raw=int(row.get("token_amount_raw", 0)),
                buy_tx_hash=str(row.get("buy_tx_hash", "")),
                sell_tx_hash=str(row.get("sell_tx_hash", "")),
                buy_tx_status=str(row.get("buy_tx_status", "none")),
                sell_tx_status=str(row.get("sell_tx_status", "none")),
                spent_eth=float(row.get("spent_eth", 0.0)),
                original_position_size_usd=float(row.get("original_position_size_usd", row.get("position_size_usd", 0.0))),
                partial_tp_done=bool(
                    row.get(
                        "partial_tp_done",
                        bool(partial_stage > 0),
                    )
                ),
                partial_tp_stage=partial_stage,
                partial_realized_pnl_usd=float(row.get("partial_realized_pnl_usd", 0.0)),
                last_partial_tp_trigger_percent=float(row.get("last_partial_tp_trigger_percent", 0.0)),
                timeout_extension_seconds=max(0, int(row.get("timeout_extension_seconds", 0) or 0)),
                market_mode=str(row.get("market_mode", "")),
                entry_tier=str(row.get("entry_tier", "")),
                entry_channel=str(row.get("entry_channel", "")),
                partial_tp_trigger_mult=float(row.get("partial_tp_trigger_mult", 1.0) or 1.0),
                partial_tp_sell_mult=float(row.get("partial_tp_sell_mult", 1.0) or 1.0),
                token_cluster_key=str(row.get("token_cluster_key", "")),
                ev_expected_net_usd=float(row.get("ev_expected_net_usd", 0.0) or 0.0),
                ev_confidence=float(row.get("ev_confidence", 0.0) or 0.0),
                kelly_mult=float(row.get("kelly_mult", 1.0) or 1.0),
                execution_lane=str(row.get("execution_lane", "") or ""),
                execution_phase=str(row.get("execution_phase", "") or ""),
                cluster_ev_avg_net_usd=float(row.get("cluster_ev_avg_net_usd", 0.0) or 0.0),
                cluster_ev_loss_share=float(row.get("cluster_ev_loss_share", 0.0) or 0.0),
                cluster_ev_samples=float(row.get("cluster_ev_samples", 0.0) or 0.0),
            )
        except Exception:
            return None

    def _prune_closed_positions(self) -> None:
        max_days = int(config.CLOSED_TRADES_MAX_AGE_DAYS)
        if max_days <= 0 or not self.closed_positions:
            return

        cutoff_ts = datetime.now(timezone.utc).timestamp() - (max_days * 86400)
        self.closed_positions = [
            pos
            for pos in self.closed_positions
            if (pos.closed_at or pos.opened_at).timestamp() >= cutoff_ts
        ]
