"""Auto-trading engine with full paper buy/sell cycle."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import config
from trading.live_executor import LiveExecutor
from utils.addressing import normalize_address
from utils.http_client import ResilientHttpClient

logger = logging.getLogger(__name__)
ADDRESS_RE = re.compile(r"^0x[a-f0-9]{40}$")


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
    stop_loss_percent: int
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


class AutoTrader:
    def __init__(self) -> None:
        self.state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_state.json")
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
        self.trade_open_timestamps: list[float] = []
        self.tx_event_timestamps: list[float] = []
        self.price_guard_pending: dict[str, dict[str, float | int]] = {}
        self.trading_pause_until_ts = 0.0
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
        ttl = int(ttl_seconds or int(getattr(config, "AUTOTRADE_BLACKLIST_TTL_SECONDS", 86400) or 86400))
        ttl = max(300, ttl)
        self._blacklist[key] = {
            "reason": str(reason or "blacklisted"),
            "added_ts": now_ts,
            "until_ts": now_ts + ttl,
        }
        self._blacklist_prune()
        self._save_blacklist()

    def _apply_startup_reset(self) -> None:
        if self.open_positions:
            logger.info(
                "PAPER_RESET_ON_START skipped: keeping %s open position(s).",
                len(self.open_positions),
            )
            return
        self.reset_paper_state(keep_closed=True)

    def can_open_trade(self) -> bool:
        if self.data_policy_mode != "OK":
            return False
        self._refresh_daily_window()
        # Manual control mode: no auto pause by daily drawdown or streak.
        self.trading_pause_until_ts = 0.0
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
        if self.data_policy_mode != "OK":
            return f"data_policy:{self.data_policy_mode}:{self.data_policy_reason}"
        self._refresh_daily_window()
        self.trading_pause_until_ts = 0.0
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
        if normalized not in {"OK", "DEGRADED", "FAIL_CLOSED"}:
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
        self.recovery_untracked.pop(key, None)
        self._recovery_clear_tracking(key)

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
    def _current_day_id() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _refresh_daily_window(self) -> None:
        now_ts = datetime.now(timezone.utc).timestamp()
        self.token_cooldowns = {k: v for k, v in self.token_cooldowns.items() if v > now_ts}
        self._prune_daily_tx_window()
        current_day = self._current_day_id()
        if self.day_id == current_day:
            return
        self.day_id = current_day
        self.day_realized_pnl_usd = 0.0
        self.day_start_equity_usd = self._equity_usd()
        self.current_loss_streak = 0

    @staticmethod
    def _kill_switch_active() -> bool:
        try:
            return bool(config.KILL_SWITCH_FILE) and os.path.exists(config.KILL_SWITCH_FILE)
        except Exception:
            return False

    def _prune_hourly_window(self) -> None:
        now_ts = datetime.now(timezone.utc).timestamp()
        self.trade_open_timestamps = [ts for ts in self.trade_open_timestamps if (now_ts - ts) <= 3600]

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
                return max(0.0, balance_eth * price_usd)
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

        if self.data_policy_mode != "OK":
            return False, f"data_policy_{self.data_policy_mode}"

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
        if not self.can_open_trade():
            logger.info("AutoTrade skip token=%s reason=disabled_or_limits detail=%s", symbol, self._cannot_open_trade_detail())
            return None
        if self._kill_switch_active():
            logger.warning("KILL_SWITCH active path=%s", config.KILL_SWITCH_FILE)
            return None

        recommendation = str(score_data.get("recommendation", "SKIP"))
        score = int(score_data.get("score", 0))
        if recommendation != "BUY" or score < int(config.MIN_TOKEN_SCORE):
            logger.info(
                "AutoTrade skip token=%s reason=signal recommendation=%s score=%s",
                symbol,
                recommendation,
                score,
            )
            return None

        raw_token_address = str(token_data.get("address", "") or "")
        token_address = normalize_address(raw_token_address)
        blocked, blk_reason = self._blacklist_is_blocked(token_address)
        if blocked:
            logger.info("AutoTrade skip token=%s reason=blacklist detail=%s", symbol, self._short_error_text(blk_reason))
            return None
        if not token_address or token_address in self.open_positions:
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
            logger.info("AutoTrade skip token=%s reason=cooldown_left_%ss", symbol, int(cooldown_until - now_ts))
            return None

        entry_price_usd = float(token_data.get("price_usd") or 0)
        if entry_price_usd <= 0:
            fetched = await self._fetch_current_price(token_address)
            if fetched:
                entry_price_usd = float(fetched[1])
        if entry_price_usd <= 0:
            logger.info("AutoTrade skip token=%s reason=no_price", symbol)
            return None

        tp = int(config.PROFIT_TARGET_PERCENT)
        sl = int(config.STOP_LOSS_PERCENT)
        liquidity_usd = float(token_data.get("liquidity") or 0)
        volume_5m = float(token_data.get("volume_5m") or 0)
        price_change_5m = float(token_data.get("price_change_5m") or 0)
        risk_level = str(token_data.get("risk_level", "MEDIUM")).upper()
        if not self._passes_token_guards(token_data):
            logger.info("AutoTrade skip token=%s reason=safety_guards", symbol)
            return None
        cost_profile = self._estimate_cost_profile(liquidity_usd, risk_level, score)
        expected_edge_percent = self._estimate_edge_percent(score, tp, sl, cost_profile["total_percent"], risk_level)

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
                logger.info("AutoTrade skip token=%s reason=invalid_max_buy_cap", symbol)
                return None
            position_size_usd = min(position_size_usd, cap_usd)
        min_trade_usd = max(0.01, float(getattr(config, "MIN_TRADE_USD", 0.25) or 0.25))
        if position_size_usd < min_trade_usd:
            logger.info(
                "AutoTrade skip token=%s reason=min_trade_size size=$%.2f min=$%.2f",
                symbol,
                position_size_usd,
                min_trade_usd,
            )
            return None
        if position_size_usd < 0.1:
            logger.info("AutoTrade skip token=%s reason=low_balance", symbol)
            return None

        expected_edge_usd = float(position_size_usd) * float(expected_edge_percent) / 100.0
        if config.EDGE_FILTER_ENABLED:
            mode = str(getattr(config, "EDGE_FILTER_MODE", "usd") or "usd").strip().lower()
            min_edge_pct = float(getattr(config, "MIN_EXPECTED_EDGE_PERCENT", 0.0) or 0.0)
            min_edge_usd = float(getattr(config, "MIN_EXPECTED_EDGE_USD", 0.0) or 0.0)
            pct_ok = expected_edge_percent >= float(min_edge_pct)
            usd_ok = expected_edge_usd >= float(min_edge_usd)
            if mode == "percent" and not pct_ok:
                logger.info(
                    "AutoTrade skip token=%s reason=negative_edge edge=%.2f%% min=%.2f%% edge_usd=$%.3f size=$%.2f costs=%.2f%%",
                    symbol,
                    expected_edge_percent,
                    min_edge_pct,
                    expected_edge_usd,
                    position_size_usd,
                    cost_profile["total_percent"],
                )
                return None
            if mode == "usd" and not usd_ok:
                logger.info(
                    "AutoTrade skip token=%s reason=edge_usd_low edge_usd=$%.3f min=$%.3f edge=%.2f%% size=$%.2f costs=%.2f%%",
                    symbol,
                    expected_edge_usd,
                    min_edge_usd,
                    expected_edge_percent,
                    position_size_usd,
                    cost_profile["total_percent"],
                )
                return None
            if mode == "both" and not (pct_ok and usd_ok):
                logger.info(
                    "AutoTrade skip token=%s reason=edge_low mode=both edge=%.2f%% min=%.2f%% edge_usd=$%.3f min_usd=$%.3f size=$%.2f costs=%.2f%%",
                    symbol,
                    expected_edge_percent,
                    min_edge_pct,
                    expected_edge_usd,
                    min_edge_usd,
                    position_size_usd,
                    cost_profile["total_percent"],
                )
                return None

        breakdown = score_data.get("breakdown") if isinstance(score_data.get("breakdown"), dict) else {}
        logger.info(
            "AUTO_DECISION token=%s score=%s src=%s liq=%.0f vol5m=%.0f chg5m=%.2f%% edge=%.2f%% edge_usd=$%.3f size=$%.2f mult=%.2f(%s) costs=%.2f%% gas=$%.3f breakdown=%s",
            symbol,
            score,
            str(token_data.get("source", "-")),
            liquidity_usd,
            volume_5m,
            price_change_5m,
            expected_edge_percent,
            expected_edge_usd,
            position_size_usd,
            float(mult),
            str(mult_detail),
            float(cost_profile["total_percent"]),
            float(cost_profile["gas_usd"]),
            breakdown,
        )
        max_hold_seconds = self._choose_hold_seconds(
            score=score,
            risk_level=risk_level,
            liquidity_usd=liquidity_usd,
            volume_5m=volume_5m,
            price_change_5m=price_change_5m,
        )

        if self._is_live_mode():
            if self.live_executor is None:
                logger.error("AutoTrade skip token=%s reason=live_executor_unavailable", symbol)
                return None
            weth_price_usd = await self._resolve_weth_price_usd(token_data)
            if weth_price_usd <= 0:
                logger.info("AutoTrade skip token=%s reason=no_weth_price", symbol)
                return None
            spend_eth = position_size_usd / weth_price_usd
            inv_ok, inv_reason = self._pre_buy_invariants(token_address, spend_eth)
            if not inv_ok:
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
                spend_eth,
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
                    self._blacklist_add(token_address, f"unsupported_sell_route:{self._short_error_text(sell_reason)}")
                    return None
            if bool(getattr(config, "LIVE_ROUNDTRIP_CHECK_ENABLED", True)):
                ok, rt_reason, rt_ratio = await asyncio.to_thread(
                    self.live_executor.roundtrip_quote,
                    token_address,
                    spend_eth,
                )
                if not ok:
                    logger.info(
                        "AutoTrade skip token=%s reason=roundtrip_quote_failed detail=%s",
                        symbol,
                        self._short_error_text(rt_reason),
                    )
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
                    self._blacklist_add(token_address, f"roundtrip_ratio:{rt_ratio:.3f}")
                    return None
            hp_ok, hp_detail = await self._honeypot_guard_passes(token_address)
            if not hp_ok:
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
                logger.error("AUTO_BUY live_failed token=%s err=%s", symbol, exc)
                self._blacklist_add(token_address, f"live_buy_failed:{self._short_error_text(exc)}", ttl_seconds=6 * 3600)
                return None
            if int(buy_result.token_amount_raw) <= 0:
                self._record_tx_event()
                logger.error(
                    "AUTO_BUY live_failed token=%s err=zero_token_amount tx=%s spent_eth=%.8f",
                    symbol,
                    str(getattr(buy_result, "tx_hash", "")),
                    float(getattr(buy_result, "spent_eth", 0.0) or 0.0),
                )
                self._blacklist_add(token_address, "live_buy_zero_amount", ttl_seconds=6 * 3600)
                return None
            pos = PaperPosition(
                token_address=token_address,
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
                expected_edge_percent=expected_edge_percent,
                buy_cost_percent=cost_profile["buy_percent"],
                sell_cost_percent=cost_profile["sell_percent"],
                gas_cost_usd=cost_profile["gas_usd"],
                token_amount_raw=int(buy_result.token_amount_raw),
                buy_tx_hash=str(buy_result.tx_hash),
                buy_tx_status="confirmed",
                spent_eth=float(buy_result.spent_eth),
            )
            self.open_positions[token_address] = pos
            self.total_plans += 1
            self.total_executed += 1
            self.trade_open_timestamps.append(datetime.now(timezone.utc).timestamp())
            self._record_tx_event()
            logger.info(
                "AUTO_BUY Live BUY token=%s address=%s spend=%.8f ETH score=%s edge=%.2f%% edge_usd=$%.3f size=$%.2f mult=%.2f(%s) hold=%ss tx=%s",
                pos.symbol,
                pos.token_address,
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
            self._save_state()
            return pos

        pos = PaperPosition(
            token_address=token_address,
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
            expected_edge_percent=expected_edge_percent,
            buy_cost_percent=cost_profile["buy_percent"],
            sell_cost_percent=cost_profile["sell_percent"],
            gas_cost_usd=cost_profile["gas_usd"],
        )

        self.open_positions[token_address] = pos
        self.total_plans += 1
        self.total_executed += 1
        self.trade_open_timestamps.append(datetime.now(timezone.utc).timestamp())
        self._record_tx_event()
        self.paper_balance_usd -= position_size_usd

        logger.info(
            "AUTO_BUY Paper BUY token=%s address=%s entry=$%.8f size=$%.2f score=%s edge=%.2f%% edge_usd=$%.3f mult=%.2f(%s) hold=%ss costs=%.2f%% gas=$%.2f",
            pos.symbol,
            pos.token_address,
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

    async def process_open_positions(self, bot=None) -> None:
        await self._process_recovery_queue()
        await self._check_critical_conditions()
        self._maybe_log_paper_summary()
        if not self.open_positions:
            return

        updates = await asyncio.gather(
            *[self._fetch_current_price(address) for address in list(self.open_positions.keys())]
        )
        for update in updates:
            if not update:
                continue
            address, price_usd = update
            position = self.open_positions.get(address)
            if not position:
                continue
            if price_usd <= 0:
                continue
            if not self._accept_price_update(position, price_usd):
                continue

            position.current_price_usd = price_usd
            raw_price_pnl_percent = ((price_usd - position.entry_price_usd) / position.entry_price_usd) * 100
            position.pnl_percent = raw_price_pnl_percent
            position.pnl_usd = (position.position_size_usd * raw_price_pnl_percent) / 100
            position.peak_pnl_percent = max(float(position.peak_pnl_percent), float(position.pnl_percent))

            should_close = False
            close_reason = ""

            if position.pnl_percent >= position.take_profit_percent:
                should_close = True
                close_reason = "TP"
            elif position.pnl_percent <= -abs(position.stop_loss_percent):
                should_close = True
                close_reason = "SL"
            elif (
                config.PROFIT_LOCK_ENABLED
                and float(position.peak_pnl_percent) >= float(config.PROFIT_LOCK_TRIGGER_PERCENT)
                and float(position.pnl_percent) <= float(config.PROFIT_LOCK_FLOOR_PERCENT)
            ):
                should_close = True
                close_reason = "PROFIT_LOCK"
            else:
                age_seconds = int((datetime.now(timezone.utc) - position.opened_at).total_seconds())
                min_age_ratio = max(0.0, min(100.0, float(config.WEAKNESS_EXIT_MIN_AGE_PERCENT))) / 100.0
                if (
                    config.WEAKNESS_EXIT_ENABLED
                    and age_seconds >= int(position.max_hold_seconds * min_age_ratio)
                    and float(position.pnl_percent) <= float(config.WEAKNESS_EXIT_PNL_PERCENT)
                    and float(position.peak_pnl_percent) < float(config.PROFIT_LOCK_TRIGGER_PERCENT)
                ):
                    should_close = True
                    close_reason = "WEAKNESS"
                elif age_seconds >= int(position.max_hold_seconds):
                    should_close = True
                    close_reason = "TIMEOUT"

            if should_close:
                if self._is_live_mode():
                    await self._close_position_live(position, close_reason)
                else:
                    self._close_position(position, close_reason)
                if bot and int(config.PERSONAL_TELEGRAM_ID or 0) > 0:
                    await self._send_close_message(bot, position)

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
                if not self._recovery_attempt_allowed(normalized):
                    continue

                amount_raw = int(self.recovery_untracked.get(normalized, 0) or 0)
                if amount_raw <= 0:
                    self.recovery_untracked.pop(normalized, None)
                    self._recovery_clear_tracking(normalized)
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

            self.open_positions.pop(position.token_address, None)
            self.price_guard_pending.pop(position.token_address, None)
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

        self.open_positions.pop(position.token_address, None)
        self.price_guard_pending.pop(position.token_address, None)
        self._recovery_forget_address(position.token_address)
        self.closed_positions.append(position)
        self._prune_closed_positions()
        self.total_closed += 1
        self.realized_pnl_usd += position.pnl_usd
        self.day_realized_pnl_usd += position.pnl_usd

        if position.pnl_usd >= 0:
            self.total_wins += 1
            self.current_loss_streak = 0
        else:
            self.total_losses += 1
            self.current_loss_streak += 1

        token_cooldown = int(config.MAX_TOKEN_COOLDOWN_SECONDS)
        if token_cooldown > 0:
            self.token_cooldowns[normalize_address(position.token_address)] = datetime.now(timezone.utc).timestamp() + token_cooldown

        logger.info(
            "AUTO_SELL Live SELL token=%s reason=%s recv=%.8f ETH pnl=%.2f%% ($%.2f) tx=%s",
            position.symbol,
            reason,
            float(sell_result.received_eth),
            position.pnl_percent,
            position.pnl_usd,
            position.sell_tx_hash,
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
        pnl_usd = final_value_usd - position.position_size_usd
        pnl_percent = (pnl_usd / position.position_size_usd * 100) if position.position_size_usd > 0 else 0.0

        position.status = "CLOSED"
        position.close_reason = reason
        position.closed_at = datetime.now(timezone.utc)
        position.pnl_usd = pnl_usd
        position.pnl_percent = pnl_percent
        position.sell_tx_status = "simulated"
        self._record_tx_event()

        self.open_positions.pop(position.token_address, None)
        self.price_guard_pending.pop(position.token_address, None)
        self._recovery_forget_address(position.token_address)
        self.closed_positions.append(position)
        self._prune_closed_positions()
        self.total_closed += 1

        self.paper_balance_usd += final_value_usd
        self._update_stair_floor()
        self.realized_pnl_usd += position.pnl_usd
        self.day_realized_pnl_usd += position.pnl_usd

        if position.pnl_usd >= 0:
            self.total_wins += 1
            self.current_loss_streak = 0
        else:
            self.total_losses += 1
            self.current_loss_streak += 1

        self.trading_pause_until_ts = 0.0

        token_cooldown = int(config.MAX_TOKEN_COOLDOWN_SECONDS)
        if token_cooldown > 0:
            self.token_cooldowns[normalize_address(position.token_address)] = datetime.now(timezone.utc).timestamp() + token_cooldown

        logger.info(
            "AUTO_SELL Paper SELL token=%s reason=%s exit=$%.8f pnl=%.2f%% ($%.2f) raw=%.2f%% cost=%.2f%% gas=$%.2f balance=$%.2f",
            position.symbol,
            reason,
            position.current_price_usd,
            position.pnl_percent,
            position.pnl_usd,
            raw_price_pnl_percent,
            position.buy_cost_percent + position.sell_cost_percent,
            position.gas_cost_usd,
            self.paper_balance_usd,
        )
        self._save_state()

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
            if price <= 0:
                continue
            if liq > best_liq:
                best_liq = liq
                best_price = price
        if best_price <= 0:
            return None
        return token_address, best_price

    async def _resolve_weth_price_usd(self, token_data: dict[str, Any]) -> float:
        direct = float(token_data.get("weth_price_usd") or token_data.get("base_quote_price_usd") or 0)
        if direct > 0:
            self.last_weth_price_usd = direct
            return direct

        if self.last_weth_price_usd > 0:
            return self.last_weth_price_usd

        weth_address = str(config.WETH_ADDRESS or "").strip().lower()
        if weth_address:
            fetched = await self._fetch_current_price(weth_address)
            if fetched and float(fetched[1]) > 0:
                self.last_weth_price_usd = float(fetched[1])
                return self.last_weth_price_usd

        fallback = max(0.0, float(config.WETH_PRICE_FALLBACK_USD))
        if fallback > 0:
            self.last_weth_price_usd = fallback
            logger.info("AutoTrade cap fallback weth_price=$%.2f", fallback)
            return fallback

        return 0.0

    def _accept_price_update(self, position: PaperPosition, next_price_usd: float) -> bool:
        if not config.PAPER_PRICE_GUARD_ENABLED:
            return True
        current_price = float(position.current_price_usd or 0.0)
        if current_price <= 0 or next_price_usd <= 0:
            return True

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
        win_rate = (self.total_wins / self.total_closed * 100) if self.total_closed else 0.0
        unrealized_pnl = sum(pos.pnl_usd for pos in self.open_positions.values())
        equity = self.paper_balance_usd + sum(pos.position_size_usd + pos.pnl_usd for pos in self.open_positions.values())

        return {
            "open_trades": len(self.open_positions),
            "planned": self.total_plans,
            "executed": self.total_executed,
            "closed": self.total_closed,
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate_percent": round(win_rate, 2),
            "paper_balance_usd": round(self.paper_balance_usd, 2),
            "realized_pnl_usd": round(self.realized_pnl_usd, 2),
            "day_realized_pnl_usd": round(self.day_realized_pnl_usd, 2),
            "unrealized_pnl_usd": round(unrealized_pnl, 2),
            "equity_usd": round(equity, 2),
            "initial_balance_usd": round(self.initial_balance_usd, 2),
            "loss_streak": self.current_loss_streak,
            "trading_pause_until_ts": round(self.trading_pause_until_ts, 2),
            "trades_last_hour": len(self.trade_open_timestamps),
            "tx_last_day": len(self.tx_event_timestamps),
            "stair_step_enabled": bool(config.STAIR_STEP_ENABLED),
            "stair_floor_usd": round(self.stair_floor_usd, 2),
            "stair_peak_balance_usd": round(self.stair_peak_balance_usd, 2),
            "available_balance_usd": round(self._available_balance_usd(), 2),
            "emergency_halt_reason": self.emergency_halt_reason,
            "emergency_halt_ts": round(self.emergency_halt_ts, 2),
            "data_policy_mode": self.data_policy_mode,
            "data_policy_reason": self.data_policy_reason,
            "recovery_queue": len(self.recovery_queue),
            "recovery_untracked": len(self.recovery_untracked),
        }

    def reset_paper_state(self, keep_closed: bool = True) -> None:
        self.initial_balance_usd = float(config.WALLET_BALANCE_USD)
        self.paper_balance_usd = float(config.WALLET_BALANCE_USD)
        self.realized_pnl_usd = 0.0
        self.day_id = self._current_day_id()
        self.day_start_equity_usd = float(config.WALLET_BALANCE_USD)
        self.day_realized_pnl_usd = 0.0
        self.current_loss_streak = 0
        self.trading_pause_until_ts = 0.0
        self.token_cooldowns.clear()
        self.trade_open_timestamps.clear()
        self.tx_event_timestamps.clear()
        self.price_guard_pending.clear()
        self.emergency_halt_reason = ""
        self.emergency_halt_ts = 0.0
        self._live_sell_failures = 0
        self.stair_floor_usd = 0.0
        self.stair_peak_balance_usd = self.paper_balance_usd
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

        # Discover potentially untracked tokens from known address universe.
        max_discovery = int(getattr(config, "RECOVERY_DISCOVERY_MAX_ADDRESSES", 80) or 80)
        if max_discovery > 0:
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

            candidates = candidates[:max_discovery]
            for addr in candidates:
                try:
                    amount_raw = int(self.live_executor.token_balance_raw(addr))
                except Exception:
                    continue
                if amount_raw <= 0:
                    continue
                self.recovery_untracked[addr] = amount_raw
                logger.warning(
                    "RECOVERY untracked_detected address=%s amount_raw=%s",
                    addr,
                    amount_raw,
                )
        self._save_state()

    @staticmethod
    def _estimate_edge_percent(
        score: int,
        tp_percent: int,
        sl_percent: int,
        total_cost_percent: float,
        risk_level: str,
    ) -> float:
        # Score-driven win probability with risk adjustment.
        p_win = max(0.1, min(0.9, (score - 50) / 50))
        risk_level = str(risk_level).upper()
        if risk_level == "HIGH":
            p_win -= 0.12
        elif risk_level == "MEDIUM":
            p_win -= 0.05
        p_win = max(0.05, min(0.95, p_win))

        expected = (p_win * tp_percent) - ((1 - p_win) * sl_percent) - total_cost_percent
        return float(expected)

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
    def _estimate_cost_profile(liquidity_usd: float, risk_level: str, score: int) -> dict[str, float]:
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

        # buy + approve + sell tx in real mode
        gas_usd = float(config.PAPER_GAS_PER_TX_USD) * 3.0
        total_percent = buy_percent + sell_percent + (gas_usd / max(float(config.PAPER_TRADE_SIZE_MAX_USD), 0.1) * 100)

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
        try:
            self._prune_closed_positions()
            payload = {
                "initial_balance_usd": self.initial_balance_usd,
                "paper_balance_usd": self.paper_balance_usd,
                "realized_pnl_usd": self.realized_pnl_usd,
                "total_plans": self.total_plans,
                "total_executed": self.total_executed,
                "total_closed": self.total_closed,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "current_loss_streak": self.current_loss_streak,
                "trading_pause_until_ts": self.trading_pause_until_ts,
                "day_id": self.day_id,
                "day_start_equity_usd": self.day_start_equity_usd,
                "day_realized_pnl_usd": self.day_realized_pnl_usd,
                "token_cooldowns": self.token_cooldowns,
                "trade_open_timestamps": self.trade_open_timestamps[-500:],
                "tx_event_timestamps": self.tx_event_timestamps[-1000:],
                "price_guard_pending": self.price_guard_pending,
                "recovery_queue": self.recovery_queue[-200:],
                "recovery_untracked": self.recovery_untracked,
                "recovery_attempts": self._recovery_attempts,
                "recovery_last_attempt_ts": self._recovery_last_attempt_ts,
                "data_policy_mode": self.data_policy_mode,
                "data_policy_reason": self.data_policy_reason,
                "emergency_halt_reason": self.emergency_halt_reason,
                "emergency_halt_ts": self.emergency_halt_ts,
                "stair_floor_usd": self.stair_floor_usd,
                "stair_peak_balance_usd": self.stair_peak_balance_usd,
                "live_start_ts": self.live_start_ts,
                "live_start_balance_eth": self.live_start_balance_eth,
                "live_start_balance_usd": self.live_start_balance_usd,
                "open_positions": [self._serialize_pos(p) for p in self.open_positions.values()],
                "closed_positions": [self._serialize_pos(p) for p in self.closed_positions[-500:]],
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("AutoTrade state save failed: %s", exc)

    def _load_state(self) -> None:
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8-sig") as f:
                payload = json.load(f)
            self.initial_balance_usd = float(payload.get("initial_balance_usd", self.initial_balance_usd))
            self.paper_balance_usd = float(payload.get("paper_balance_usd", self.paper_balance_usd))
            self.realized_pnl_usd = float(payload.get("realized_pnl_usd", 0.0))
            self.total_plans = int(payload.get("total_plans", 0))
            self.total_executed = int(payload.get("total_executed", 0))
            self.total_closed = int(payload.get("total_closed", 0))
            self.total_wins = int(payload.get("total_wins", 0))
            self.total_losses = int(payload.get("total_losses", 0))
            self.current_loss_streak = int(payload.get("current_loss_streak", 0))
            self.trading_pause_until_ts = 0.0
            self.day_id = str(payload.get("day_id", self._current_day_id()))
            self.day_start_equity_usd = float(payload.get("day_start_equity_usd", self.paper_balance_usd))
            self.day_realized_pnl_usd = float(payload.get("day_realized_pnl_usd", 0.0))
            raw_cooldowns = payload.get("token_cooldowns", {}) or {}
            self.token_cooldowns = {normalize_address(str(k)): float(v) for k, v in raw_cooldowns.items() if normalize_address(str(k))}
            raw_trade_ts = payload.get("trade_open_timestamps", []) or []
            self.trade_open_timestamps = []
            for value in raw_trade_ts:
                try:
                    self.trade_open_timestamps.append(float(value))
                except (TypeError, ValueError):
                    continue
            raw_tx_ts = payload.get("tx_event_timestamps", []) or []
            self.tx_event_timestamps = []
            for value in raw_tx_ts:
                try:
                    self.tx_event_timestamps.append(float(value))
                except (TypeError, ValueError):
                    continue
            pending_raw = payload.get("price_guard_pending", {}) or {}
            self.price_guard_pending = {}
            if isinstance(pending_raw, dict):
                for address, record in pending_raw.items():
                    if not isinstance(record, dict):
                        continue
                    normalized = normalize_address(str(address))
                    if not normalized:
                        continue
                    self.price_guard_pending[normalized] = {
                        "price": float(record.get("price", 0.0)),
                        "count": int(record.get("count", 0)),
                        "first_ts": float(record.get("first_ts", 0.0)),
                    }
            self.recovery_queue = []
            raw_recovery = payload.get("recovery_queue", []) or []
            if isinstance(raw_recovery, list):
                for value in raw_recovery:
                    addr = normalize_address(value)
                    if addr:
                        self.recovery_queue.append(addr)
            self.recovery_untracked = {}
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
                        self.recovery_untracked[addr] = amount
            self._recovery_attempts = {}
            raw_attempts = payload.get("recovery_attempts", {}) or {}
            if isinstance(raw_attempts, dict):
                for k, v in raw_attempts.items():
                    addr = normalize_address(k)
                    if not addr:
                        continue
                    try:
                        self._recovery_attempts[addr] = int(v)
                    except Exception:
                        continue
            self._recovery_last_attempt_ts = {}
            raw_attempt_ts = payload.get("recovery_last_attempt_ts", {}) or {}
            if isinstance(raw_attempt_ts, dict):
                for k, v in raw_attempt_ts.items():
                    addr = normalize_address(k)
                    if not addr:
                        continue
                    try:
                        self._recovery_last_attempt_ts[addr] = float(v)
                    except Exception:
                        continue
            self.data_policy_mode = str(payload.get("data_policy_mode", self.data_policy_mode) or "OK").upper()
            if self.data_policy_mode not in {"OK", "DEGRADED", "FAIL_CLOSED"}:
                self.data_policy_mode = "OK"
            self.data_policy_reason = str(payload.get("data_policy_reason", self.data_policy_reason) or "")
            self.emergency_halt_reason = str(payload.get("emergency_halt_reason", "") or "")
            self.emergency_halt_ts = float(payload.get("emergency_halt_ts", 0.0) or 0.0)
            self.stair_floor_usd = float(payload.get("stair_floor_usd", self.stair_floor_usd))
            self.stair_peak_balance_usd = float(payload.get("stair_peak_balance_usd", self.paper_balance_usd))
            self.live_start_ts = float(payload.get("live_start_ts", self.live_start_ts) or 0.0)
            self.live_start_balance_eth = float(payload.get("live_start_balance_eth", self.live_start_balance_eth) or 0.0)
            self.live_start_balance_usd = float(payload.get("live_start_balance_usd", self.live_start_balance_usd) or 0.0)
            self._prune_hourly_window()

            self.open_positions.clear()
            for row in payload.get("open_positions", []):
                pos = self._deserialize_pos(row)
                if pos and pos.status == "OPEN":
                    self.open_positions[normalize_address(pos.token_address)] = pos

            self.closed_positions.clear()
            for row in payload.get("closed_positions", []):
                pos = self._deserialize_pos(row)
                if pos:
                    self.closed_positions.append(pos)
            self._prune_closed_positions()
            self._sync_stair_state()

            logger.info(
                "AutoTrade state loaded open=%s closed=%s balance=$%.2f",
                len(self.open_positions),
                len(self.closed_positions),
                self.paper_balance_usd,
            )
        except Exception as exc:
            logger.warning("AutoTrade state load failed: %s", exc)

    @staticmethod
    def _serialize_pos(pos: PaperPosition) -> dict[str, Any]:
        return {
            "token_address": normalize_address(pos.token_address),
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
        }

    @staticmethod
    def _deserialize_pos(row: dict[str, Any]) -> PaperPosition | None:
        try:
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
                stop_loss_percent=int(row.get("stop_loss_percent", 30)),
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
