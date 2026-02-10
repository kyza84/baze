"""Auto-trading engine with full paper buy/sell cycle."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiohttp

import config

logger = logging.getLogger(__name__)


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


class AutoTrader:
    def __init__(self) -> None:
        self.state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_state.json")
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
        self.price_guard_pending: dict[str, dict[str, float | int]] = {}
        self.trading_pause_until_ts = 0.0
        self.day_id = self._current_day_id()
        self.day_start_equity_usd = float(config.WALLET_BALANCE_USD)
        self.day_realized_pnl_usd = 0.0

        self.initial_balance_usd = float(config.WALLET_BALANCE_USD)
        self.paper_balance_usd = float(config.WALLET_BALANCE_USD)
        self.realized_pnl_usd = 0.0
        self._load_state()
        if config.PAPER_RESET_ON_START:
            self._apply_startup_reset()
        self._prune_closed_positions()

    def _apply_startup_reset(self) -> None:
        if self.open_positions:
            logger.info(
                "PAPER_RESET_ON_START skipped: keeping %s open position(s).",
                len(self.open_positions),
            )
            return
        self.reset_paper_state(keep_closed=True)

    def can_open_trade(self) -> bool:
        self._refresh_daily_window()
        # Manual control mode: no auto pause by daily drawdown or streak.
        self.trading_pause_until_ts = 0.0
        if self._kill_switch_active():
            logger.warning("KILL_SWITCH active path=%s", config.KILL_SWITCH_FILE)
            return False
        self._prune_hourly_window()
        max_trades_per_hour = int(config.MAX_TRADES_PER_HOUR)
        if max_trades_per_hour > 0 and len(self.trade_open_timestamps) >= max_trades_per_hour:
            return False
        max_open = int(config.MAX_OPEN_TRADES)
        if max_open > 0 and len(self.open_positions) >= max_open:
            return False
        if not config.AUTO_TRADE_ENABLED:
            return False
        if not config.AUTO_TRADE_PAPER:
            # Live execution is intentionally blocked until explicit implementation.
            return False
        return self.paper_balance_usd >= 0.1

    @staticmethod
    def _current_day_id() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _refresh_daily_window(self) -> None:
        now_ts = datetime.now(timezone.utc).timestamp()
        self.token_cooldowns = {k: v for k, v in self.token_cooldowns.items() if v > now_ts}
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

    def _equity_usd(self) -> float:
        return self.paper_balance_usd + sum(
            pos.position_size_usd + pos.pnl_usd for pos in self.open_positions.values()
        )

    async def plan_batch(self, candidates: list[tuple[dict[str, Any], dict[str, Any]]]) -> int:
        if not candidates:
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

        eligible.sort(key=lambda item: int(item[1].get("score", 0)), reverse=True)
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

    async def plan_trade(self, token_data: dict[str, Any], score_data: dict[str, Any]) -> PaperPosition | None:
        symbol = str(token_data.get("symbol", "N/A"))
        if not self.can_open_trade():
            logger.info("AutoTrade skip token=%s reason=disabled_or_limits", symbol)
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

        token_address = str(token_data.get("address", "")).strip()
        if not token_address or token_address in self.open_positions:
            logger.info("AutoTrade skip token=%s reason=address_or_duplicate", symbol)
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
        if config.EDGE_FILTER_ENABLED and expected_edge_percent < float(config.MIN_EXPECTED_EDGE_PERCENT):
            logger.info(
                "AutoTrade skip token=%s reason=negative_edge edge=%.2f cost=%.2f",
                symbol,
                expected_edge_percent,
                cost_profile["total_percent"],
            )
            return None

        position_size_usd = self._choose_position_size(expected_edge_percent)
        position_size_usd = min(position_size_usd, self.paper_balance_usd)
        position_size_usd = self._apply_max_loss_per_trade_cap(
            position_size_usd=position_size_usd,
            stop_loss_percent=sl,
            total_cost_percent=cost_profile["total_percent"],
            gas_usd=cost_profile["gas_usd"],
        )
        max_buy_weth = float(config.MAX_BUY_AMOUNT)
        if max_buy_weth > 0:
            weth_price_usd = float(token_data.get("weth_price_usd") or token_data.get("base_quote_price_usd") or 0)
            if weth_price_usd <= 0:
                logger.info("AutoTrade skip token=%s reason=missing_weth_price_for_cap", symbol)
                return None
            cap_usd = max_buy_weth * weth_price_usd
            if cap_usd <= 0:
                logger.info("AutoTrade skip token=%s reason=invalid_max_buy_cap", symbol)
                return None
            position_size_usd = min(position_size_usd, cap_usd)
        if position_size_usd < 0.1:
            logger.info("AutoTrade skip token=%s reason=low_balance", symbol)
            return None
        max_hold_seconds = self._choose_hold_seconds(
            score=score,
            risk_level=risk_level,
            liquidity_usd=liquidity_usd,
            volume_5m=volume_5m,
            price_change_5m=price_change_5m,
        )

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
        self.paper_balance_usd -= position_size_usd

        logger.info(
            "AUTO_BUY Paper BUY token=%s address=%s entry=$%.8f size=$%.2f score=%s edge=%.2f%% hold=%ss costs=%.2f%% gas=$%.2f",
            pos.symbol,
            pos.token_address,
            pos.entry_price_usd,
            pos.position_size_usd,
            pos.score,
            pos.expected_edge_percent,
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

            should_close = False
            close_reason = ""

            if position.pnl_percent >= position.take_profit_percent:
                should_close = True
                close_reason = "TP"
            elif position.pnl_percent <= -abs(position.stop_loss_percent):
                should_close = True
                close_reason = "SL"
            else:
                age_seconds = int((datetime.now(timezone.utc) - position.opened_at).total_seconds())
                if age_seconds >= int(position.max_hold_seconds):
                    should_close = True
                    close_reason = "TIMEOUT"

            if should_close:
                self._close_position(position, close_reason)
                if bot and int(config.PERSONAL_TELEGRAM_ID or 0) > 0:
                    await self._send_close_message(bot, position)

    def _close_position(self, position: PaperPosition, reason: str) -> None:
        self._refresh_daily_window()
        raw_price_pnl_percent = ((position.current_price_usd - position.entry_price_usd) / position.entry_price_usd) * 100
        gross_value_usd = position.position_size_usd * (1 + raw_price_pnl_percent / 100)

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

        self.open_positions.pop(position.token_address, None)
        self.price_guard_pending.pop(position.token_address, None)
        self.closed_positions.append(position)
        self._prune_closed_positions()
        self.total_closed += 1

        self.paper_balance_usd += final_value_usd
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
            self.token_cooldowns[position.token_address] = datetime.now(timezone.utc).timestamp() + token_cooldown

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
                    "📉 <b>Paper Trade Closed</b>\n\n"
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

    async def _fetch_current_price(self, token_address: str) -> tuple[str, float] | None:
        url = f"{config.DEXSCREENER_API}/tokens/{token_address}"
        timeout = aiohttp.ClientTimeout(total=config.DEX_TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    data = await response.json()
        except Exception:
            return None

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
        self.price_guard_pending.clear()
        self.open_positions.clear()
        if not keep_closed:
            self.closed_positions.clear()
            self.total_closed = 0
            self.total_wins = 0
            self.total_losses = 0
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
                "price_guard_pending": self.price_guard_pending,
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
            self.token_cooldowns = {str(k): float(v) for k, v in raw_cooldowns.items()}
            raw_trade_ts = payload.get("trade_open_timestamps", []) or []
            self.trade_open_timestamps = []
            for value in raw_trade_ts:
                try:
                    self.trade_open_timestamps.append(float(value))
                except (TypeError, ValueError):
                    continue
            pending_raw = payload.get("price_guard_pending", {}) or {}
            self.price_guard_pending = {}
            if isinstance(pending_raw, dict):
                for address, record in pending_raw.items():
                    if not isinstance(record, dict):
                        continue
                    self.price_guard_pending[str(address)] = {
                        "price": float(record.get("price", 0.0)),
                        "count": int(record.get("count", 0)),
                        "first_ts": float(record.get("first_ts", 0.0)),
                    }
            self._prune_hourly_window()

            self.open_positions.clear()
            for row in payload.get("open_positions", []):
                pos = self._deserialize_pos(row)
                if pos and pos.status == "OPEN":
                    self.open_positions[pos.token_address] = pos

            self.closed_positions.clear()
            for row in payload.get("closed_positions", []):
                pos = self._deserialize_pos(row)
                if pos:
                    self.closed_positions.append(pos)
            self._prune_closed_positions()

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
            "token_address": pos.token_address,
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
                token_address=str(row.get("token_address", "")),
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
