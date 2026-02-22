"""V2 runtime controllers: universe flow, budgets, calibration, matrix guard."""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
import hashlib
from collections import defaultdict, deque
from typing import Any

import config
from utils.addressing import normalize_address

logger = logging.getLogger(__name__)


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _parse_float_map(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for chunk in str(raw or "").split(","):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        k = str(key or "").strip().lower()
        if not k:
            continue
        try:
            out[k] = float(value.strip())
        except Exception:
            continue
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    xs = sorted(float(v) for v in values)
    pp = _clamp(float(p), 0.0, 100.0) / 100.0
    idx = pp * float(len(xs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(xs[lo])
    frac = idx - float(lo)
    return float(xs[lo] + ((xs[hi] - xs[lo]) * frac))


def _deterministic_probability(seed: str) -> float:
    digest = hashlib.sha1(str(seed or "").encode("utf-8")).hexdigest()
    top = int(digest[:8], 16)
    return float(top % 10000) / 10000.0


def _datetime_to_ts(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value.timestamp())
    except Exception:
        return 0.0


class UniverseFlowController:
    """Flow-shaping before heavy checks: diversity + anti-repeat."""

    def __init__(self) -> None:
        def _cfg(name: str, default: Any) -> Any:
            value = getattr(config, name, default)
            return default if value is None else value

        self.enabled = bool(getattr(config, "V2_UNIVERSE_ENABLED", False))
        self.max_total_per_cycle = max(0, int(getattr(config, "V2_UNIVERSE_MAX_TOTAL_PER_CYCLE", 0) or 0))
        self.novelty_window_seconds = max(60, int(getattr(config, "V2_UNIVERSE_NOVELTY_WINDOW_SECONDS", 1800) or 1800))
        self.novelty_min_share = _clamp(
            float(getattr(config, "V2_UNIVERSE_NOVELTY_MIN_SHARE", 0.30) or 0.30),
            0.0,
            1.0,
        )
        self.novelty_min_abs = max(0, int(getattr(config, "V2_UNIVERSE_NOVELTY_MIN_ABS", 2) or 2))
        self.pass_repeat_cooldown_seconds = max(
            0,
            int(getattr(config, "V2_UNIVERSE_PASS_REPEAT_COOLDOWN_SECONDS", 180) or 180),
        )
        self.pass_repeat_override_vol_mult = max(
            1.0,
            float(getattr(config, "V2_UNIVERSE_PASS_REPEAT_OVERRIDE_VOL_MULT", 2.2) or 2.2),
        )
        self.source_caps = {
            k: max(0, int(v))
            for k, v in _parse_float_map(
                str(
                    getattr(
                        config,
                        "V2_UNIVERSE_SOURCE_CAPS",
                        "onchain:120,onchain+market:120,dexscreener:100,geckoterminal:100,watchlist:35,dex_boosts:35",
                    )
                    or ""
                )
            ).items()
        }
        self.source_weights = _parse_float_map(
            str(
                getattr(
                    config,
                    "V2_UNIVERSE_SOURCE_WEIGHTS",
                    "onchain:1.15,onchain+market:1.10,dexscreener:1.00,geckoterminal:1.05,watchlist:0.90,dex_boosts:1.10",
                )
                or ""
            )
        )
        self._last_seen_ts: dict[str, float] = {}
        self._last_pass_ts: dict[str, float] = {}
        self.symbol_repeat_window_seconds = max(
            60,
            int(getattr(config, "V2_UNIVERSE_SYMBOL_REPEAT_WINDOW_SECONDS", 1800) or 1800),
        )
        self.symbol_repeat_soft_cap = max(
            0,
            int(getattr(config, "V2_UNIVERSE_SYMBOL_REPEAT_SOFT_CAP", 4) or 4),
        )
        self.symbol_repeat_hard_cap = max(
            self.symbol_repeat_soft_cap,
            int(getattr(config, "V2_UNIVERSE_SYMBOL_REPEAT_HARD_CAP", 8) or 8),
        )
        self.symbol_repeat_penalty_mult = _clamp(
            float(getattr(config, "V2_UNIVERSE_SYMBOL_REPEAT_PENALTY_MULT", 0.72) or 0.72),
            0.10,
            1.0,
        )
        self.symbol_repeat_override_vol_mult = max(
            1.0,
            float(getattr(config, "V2_UNIVERSE_SYMBOL_REPEAT_OVERRIDE_VOL_MULT", 2.5) or 2.5),
        )
        self.surge_enabled = bool(_cfg("V2_UNIVERSE_SURGE_ENABLED", False))
        self.surge_min_abs_change_5m = max(0.0, float(_cfg("V2_UNIVERSE_SURGE_MIN_ABS_CHANGE_5M", 4.0)))
        self.surge_min_volume_5m = max(0.0, float(_cfg("V2_UNIVERSE_SURGE_MIN_VOLUME_5M_USD", 2500.0)))
        self.surge_min_vol_to_liq = max(0.0, float(_cfg("V2_UNIVERSE_SURGE_MIN_VOL_TO_LIQ", 0.06)))
        self.surge_reserve_share = _clamp(float(_cfg("V2_UNIVERSE_SURGE_RESERVE_SHARE", 0.22)), 0.0, 0.50)
        self.surge_reserve_min_abs = max(0, int(_cfg("V2_UNIVERSE_SURGE_RESERVE_MIN_ABS", 2)))
        self.surge_size_mult = _clamp(float(_cfg("V2_UNIVERSE_SURGE_SIZE_MULT", 0.88)), 0.10, 1.60)
        self.surge_hold_mult = _clamp(float(_cfg("V2_UNIVERSE_SURGE_HOLD_MULT", 0.90)), 0.20, 1.40)
        self.surge_edge_usd_mult = _clamp(float(_cfg("V2_UNIVERSE_SURGE_EDGE_USD_MULT", 1.08)), 0.05, 1.80)
        self.surge_edge_percent_mult = _clamp(float(_cfg("V2_UNIVERSE_SURGE_EDGE_PERCENT_MULT", 1.06)), 0.05, 1.80)
        self._symbol_pass_history: dict[str, deque[float]] = {}

    def _is_surge_row(self, row: dict[str, Any]) -> bool:
        if not self.surge_enabled:
            return False
        token = dict(row.get("token") or {})
        vol5m = max(0.0, _safe_float(row.get("vol5m"), _safe_float(token.get("volume_5m"), 0.0)))
        liq = max(0.0, _safe_float(token.get("liquidity"), 0.0))
        change5m = abs(_safe_float(token.get("price_change_5m"), 0.0))
        vol_to_liq = vol5m / max(liq, 1.0)
        return (
            change5m >= float(self.surge_min_abs_change_5m)
            and vol5m >= float(self.surge_min_volume_5m)
            and vol_to_liq >= float(self.surge_min_vol_to_liq)
        )

    def _prune(self, now_ts: float) -> None:
        ttl = max(float(self.novelty_window_seconds) * 1.8, 3600.0)
        self._last_seen_ts = {
            k: v
            for k, v in self._last_seen_ts.items()
            if float(v or 0.0) >= (now_ts - ttl)
        }
        self._last_pass_ts = {
            k: v
            for k, v in self._last_pass_ts.items()
            if float(v or 0.0) >= (now_ts - max(float(self.pass_repeat_cooldown_seconds) * 8.0, 1800.0))
        }
        sym_cutoff = now_ts - float(self.symbol_repeat_window_seconds)
        for symbol in list(self._symbol_pass_history.keys()):
            queue = self._symbol_pass_history.get(symbol)
            if not queue:
                self._symbol_pass_history.pop(symbol, None)
                continue
            while queue and float(queue[0]) < sym_cutoff:
                queue.popleft()
            if not queue:
                self._symbol_pass_history.pop(symbol, None)

    def _symbol_key(self, token: dict[str, Any]) -> str:
        sym = str(token.get("symbol", "") or "").strip().upper()
        if sym:
            return sym
        name = str(token.get("name", "") or "").strip().upper()
        if name:
            return name
        return "-"

    def _symbol_recent_count(self, symbol: str, now_ts: float) -> int:
        if not symbol:
            return 0
        queue = self._symbol_pass_history.get(symbol)
        if not queue:
            return 0
        cutoff = now_ts - float(self.symbol_repeat_window_seconds)
        while queue and float(queue[0]) < cutoff:
            queue.popleft()
        if not queue:
            self._symbol_pass_history.pop(symbol, None)
            return 0
        return int(len(queue))

    def record_candidate_pass(self, token_address: str, symbol: str = "") -> None:
        addr = normalize_address(token_address)
        if not addr:
            return
        now_ts = time.time()
        self._last_pass_ts[addr] = now_ts
        sym_key = str(symbol or "").strip().upper()
        if sym_key:
            queue = self._symbol_pass_history.setdefault(sym_key, deque())
            queue.append(now_ts)

    def filter_tokens(self, tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.enabled or not tokens:
            return tokens

        now_ts = time.time()
        self._prune(now_ts)

        dedup: dict[str, dict[str, Any]] = {}
        for token in tokens:
            addr = normalize_address(str(token.get("address", "") or ""))
            if not addr:
                continue
            cur = dedup.get(addr)
            if cur is None:
                dedup[addr] = token
                continue
            cur_score = self._priority_score(cur)
            new_score = self._priority_score(token)
            if new_score > cur_score:
                dedup[addr] = token

        rows = []
        for addr, token in dedup.items():
            src = str(token.get("source", "unknown") or "unknown").strip().lower()
            if src == "onchain":
                # Keep explicit distinction for parallel mode.
                if str(token.get("dex", "") or "").strip():
                    src = str(token.get("source", "onchain") or "onchain").strip().lower()
            vol5m = _safe_float(token.get("volume_5m"), 0.0)
            seen_ts = float(self._last_seen_ts.get(addr, 0.0) or 0.0)
            is_novel = seen_ts <= 0.0 or (now_ts - seen_ts) >= float(self.novelty_window_seconds)
            pass_ts = float(self._last_pass_ts.get(addr, 0.0) or 0.0)
            pass_cooldown_left = max(0.0, float(self.pass_repeat_cooldown_seconds) - (now_ts - pass_ts))
            symbol = self._symbol_key(token)
            symbol_repeat_count = self._symbol_recent_count(symbol, now_ts)
            rows.append(
                {
                    "addr": addr,
                    "src": src,
                    "token": token,
                    "priority": self._priority_score(token),
                    "is_novel": bool(is_novel),
                    "pass_cooldown_left": pass_cooldown_left,
                    "vol5m": vol5m,
                    "symbol": symbol,
                    "symbol_repeat_count": symbol_repeat_count,
                }
            )

        if not rows:
            return []
        rows.sort(key=lambda row: float(row.get("priority", 0.0)), reverse=True)

        pass_override_floor = (
            float(getattr(config, "SAFE_MIN_VOLUME_5M_USD", 0.0) or 0.0)
            * float(self.pass_repeat_override_vol_mult)
        )
        pass_filtered: list[dict[str, Any]] = []
        dropped_repeat = 0
        for row in rows:
            if float(row.get("pass_cooldown_left", 0.0)) <= 0.0:
                pass_filtered.append(row)
                continue
            if float(row.get("vol5m", 0.0)) >= pass_override_floor:
                pass_filtered.append(row)
                continue
            dropped_repeat += 1

        rows = pass_filtered
        if not rows:
            if dropped_repeat > 0:
                logger.info("V2_UNIVERSE all candidates suppressed by repeat cooldown dropped=%s", dropped_repeat)
            return []

        symbol_dropped = 0
        symbol_override_floor = float(getattr(config, "SAFE_MIN_VOLUME_5M_USD", 0.0) or 0.0) * float(
            self.symbol_repeat_override_vol_mult
        )
        symbol_filtered: list[dict[str, Any]] = []
        for row in rows:
            rep_count = int(row.get("symbol_repeat_count", 0) or 0)
            vol5m = float(row.get("vol5m", 0.0) or 0.0)
            if (
                self.symbol_repeat_hard_cap > 0
                and rep_count >= self.symbol_repeat_hard_cap
                and vol5m < symbol_override_floor
            ):
                symbol_dropped += 1
                continue
            if self.symbol_repeat_soft_cap > 0 and rep_count > self.symbol_repeat_soft_cap:
                overflow = max(0, rep_count - self.symbol_repeat_soft_cap)
                penalty = float(self.symbol_repeat_penalty_mult) ** float(overflow)
                row["priority"] = float(row.get("priority", 0.0) or 0.0) * penalty
            symbol_filtered.append(row)

        rows = symbol_filtered
        rows.sort(key=lambda row: float(row.get("priority", 0.0)), reverse=True)
        if not rows:
            logger.info(
                "V2_UNIVERSE all candidates suppressed by symbol_repeat dropped=%s repeat_dropped=%s",
                symbol_dropped,
                dropped_repeat,
            )
            return []

        if self.max_total_per_cycle > 0:
            target = int(self.max_total_per_cycle)
        else:
            target = len(rows)

        target = max(1, min(target, len(rows)))
        novelty_target = max(int(math.ceil(float(target) * float(self.novelty_min_share))), int(self.novelty_min_abs))
        novelty_target = min(novelty_target, target)

        source_counts: dict[str, int] = defaultdict(int)
        selected: list[dict[str, Any]] = []
        selected_addrs: set[str] = set()
        surge_in = 0

        def _source_cap(src: str) -> int:
            if src in self.source_caps:
                return max(0, int(self.source_caps[src]))
            return 0 if self.source_caps else 10**9

        def _try_take(row: dict[str, Any]) -> bool:
            src = str(row.get("src", "unknown"))
            cap = _source_cap(src)
            if cap > 0 and int(source_counts[src]) >= int(cap):
                return False
            addr = str(row.get("addr", ""))
            if not addr or addr in selected_addrs:
                return False
            selected_addrs.add(addr)
            source_counts[src] = int(source_counts[src]) + 1
            selected.append(row)
            return True

        if self.surge_enabled:
            for row in rows:
                is_surge = self._is_surge_row(row)
                row["is_surge"] = bool(is_surge)
                if is_surge:
                    surge_in += 1
        else:
            for row in rows:
                row["is_surge"] = False

        surge_target = 0
        if self.surge_enabled and target > 0 and surge_in > 0:
            surge_target = int(math.ceil(float(target) * float(self.surge_reserve_share)))
            surge_target = max(surge_target, int(self.surge_reserve_min_abs))
            surge_target = min(surge_target, int(surge_in), int(target))

        surge_rows = [row for row in rows if bool(row.get("is_surge", False))]
        surge_novel_rows = [row for row in surge_rows if bool(row.get("is_novel"))]
        surge_selected = 0
        if surge_target > 0:
            for row in surge_novel_rows:
                if surge_selected >= surge_target:
                    break
                if _try_take(row):
                    surge_selected += 1
            if surge_selected < surge_target:
                for row in surge_rows:
                    if surge_selected >= surge_target:
                        break
                    if _try_take(row):
                        surge_selected += 1

        novel = [row for row in rows if bool(row.get("is_novel"))]
        known = [row for row in rows if not bool(row.get("is_novel"))]
        for row in novel:
            if len(selected) >= novelty_target:
                break
            _try_take(row)
        for row in rows:
            if len(selected) >= target:
                break
            _try_take(row)
        if len(selected) < target:
            # Fallback: ignore source caps in extreme scarcity.
            for row in rows:
                if len(selected) >= target:
                    break
                addr = str(row.get("addr", ""))
                if not addr or addr in selected_addrs:
                    continue
                selected_addrs.add(addr)
                selected.append(row)

        out = [dict(row.get("token") or {}) for row in selected]
        surge_out = 0
        for idx, row in enumerate(selected):
            if not bool(row.get("is_surge", False)):
                continue
            token2 = dict(out[idx] or {})
            token2["_universe_lane"] = "surge"
            token2["_entry_channel_size_mult"] = float(token2.get("_entry_channel_size_mult", 1.0) or 1.0) * float(
                self.surge_size_mult
            )
            token2["_entry_channel_hold_mult"] = float(token2.get("_entry_channel_hold_mult", 1.0) or 1.0) * float(
                self.surge_hold_mult
            )
            token2["_entry_channel_edge_usd_mult"] = float(token2.get("_entry_channel_edge_usd_mult", 1.0) or 1.0) * float(
                self.surge_edge_usd_mult
            )
            token2["_entry_channel_edge_pct_mult"] = float(token2.get("_entry_channel_edge_pct_mult", 1.0) or 1.0) * float(
                self.surge_edge_percent_mult
            )
            out[idx] = token2
            surge_out += 1
        for row in selected:
            addr = str(row.get("addr", ""))
            if addr:
                self._last_seen_ts[addr] = now_ts

        logger.info(
            (
                "V2_UNIVERSE in=%s dedup=%s out=%s target=%s novel_target=%s novel_out=%s "
                "repeat_dropped=%s symbol_dropped=%s surge_in=%s surge_target=%s surge_out=%s"
            ),
            len(tokens),
            len(dedup),
            len(out),
            target,
            novelty_target,
            sum(1 for row in selected if bool(row.get("is_novel"))),
            dropped_repeat,
            symbol_dropped,
            surge_in,
            surge_target,
            surge_out,
        )
        return out

    def _priority_score(self, token: dict[str, Any]) -> float:
        src = str(token.get("source", "unknown") or "unknown").strip().lower()
        src_w = float(self.source_weights.get(src, 1.0))
        liq = max(0.0, _safe_float(token.get("liquidity"), 0.0))
        vol5m = max(0.0, _safe_float(token.get("volume_5m"), 0.0))
        score = float((0.65 * math.log1p(vol5m)) + (0.35 * math.log1p(liq)))
        return float(score * src_w)


class UniverseQualityGateController:
    """EV-first pool shaping: core/explore/cooldown + source budgets + anti-concentration."""

    def __init__(self) -> None:
        def _cfg(name: str, default: Any) -> Any:
            value = getattr(config, name, default)
            return default if value is None else value

        self.enabled = bool(_cfg("V2_QUALITY_GATE_ENABLED", False))
        self.refresh_seconds = max(60, int(_cfg("V2_QUALITY_REFRESH_SECONDS", 1800)))
        self.window_seconds = max(300, int(_cfg("V2_QUALITY_WINDOW_SECONDS", 14400)))
        self.min_symbol_trades = max(1, int(_cfg("V2_QUALITY_MIN_SYMBOL_TRADES", 12)))
        self.min_cluster_trades = max(1, int(_cfg("V2_QUALITY_MIN_CLUSTER_TRADES", 16)))
        self.min_avg_pnl_usd = float(_cfg("V2_QUALITY_MIN_AVG_PNL_USD", 0.0))
        self.max_loss_share = _clamp(float(_cfg("V2_QUALITY_MAX_LOSS_SHARE", 0.60)), 0.0, 1.0)
        self.bad_avg_pnl_usd = float(_cfg("V2_QUALITY_BAD_AVG_PNL_USD", -0.0008))
        self.bad_loss_share = _clamp(float(_cfg("V2_QUALITY_BAD_LOSS_SHARE", 0.67)), 0.0, 1.0)
        self.explore_max_share = _clamp(float(_cfg("V2_QUALITY_EXPLORE_MAX_SHARE", 0.18)), 0.0, 0.60)
        self.explore_min_abs = max(0, int(_cfg("V2_QUALITY_EXPLORE_MIN_ABS", 1)))
        self.explore_reserve_share = _clamp(float(_cfg("V2_QUALITY_EXPLORE_RESERVE_SHARE", 0.18)), 0.0, 0.60)
        self.cooldown_probe_probability = _clamp(
            float(_cfg("V2_QUALITY_COOLDOWN_PROBE_PROBABILITY", 0.15)),
            0.0,
            1.0,
        )
        self.cooldown_size_mult = _clamp(float(_cfg("V2_QUALITY_COOLDOWN_SIZE_MULT", 0.55)), 0.05, 1.0)
        self.cooldown_hold_mult = _clamp(float(_cfg("V2_QUALITY_COOLDOWN_HOLD_MULT", 0.75)), 0.10, 1.2)
        self.cooldown_edge_usd_mult = max(0.05, float(_cfg("V2_QUALITY_COOLDOWN_EDGE_USD_MULT", 1.35)))
        self.cooldown_edge_percent_mult = max(
            0.05,
            float(_cfg("V2_QUALITY_COOLDOWN_EDGE_PERCENT_MULT", 1.30)),
        )
        self.symbol_concentration_window_seconds = max(
            120,
            int(_cfg("V2_QUALITY_SYMBOL_CONCENTRATION_WINDOW_SECONDS", 3600)),
        )
        self.symbol_max_share = _clamp(float(_cfg("V2_QUALITY_SYMBOL_MAX_SHARE", 0.10)), 0.02, 0.50)
        self.symbol_min_abs_cap = max(1, int(_cfg("V2_QUALITY_SYMBOL_MIN_ABS_CAP", 2)))
        self.symbol_reentry_min_seconds = max(0, int(_cfg("V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS", 240)))
        self.symbol_reentry_override_vol_mult = max(1.0, float(_cfg("V2_QUALITY_SYMBOL_REENTRY_OVERRIDE_VOL_MULT", 2.3)))
        self.fresh_symbol_window_seconds = max(300, int(_cfg("V2_QUALITY_FRESH_SYMBOL_WINDOW_SECONDS", 3600)))
        self.fresh_symbol_boost = max(0.0, float(_cfg("V2_QUALITY_FRESH_SYMBOL_BOOST", 2.6)))
        self.source_budget_enabled = bool(_cfg("V2_QUALITY_SOURCE_BUDGET_ENABLED", True))
        self.source_min_trades = max(1, int(_cfg("V2_QUALITY_SOURCE_MIN_TRADES", 10)))
        self.source_window_seconds = max(300, int(_cfg("V2_QUALITY_SOURCE_WINDOW_SECONDS", 14400)))
        self.source_good_avg = float(_cfg("V2_QUALITY_SOURCE_GOOD_AVG_PNL_USD", 0.0020))
        self.source_bad_avg = float(_cfg("V2_QUALITY_SOURCE_BAD_AVG_PNL_USD", -0.0015))
        self.source_bad_loss_share = _clamp(float(_cfg("V2_QUALITY_SOURCE_BAD_LOSS_SHARE", 0.64)), 0.0, 1.0)
        self.source_severe_avg = float(_cfg("V2_QUALITY_SOURCE_SEVERE_AVG_PNL_USD", -0.0030))
        self.source_severe_loss_share = _clamp(float(_cfg("V2_QUALITY_SOURCE_SEVERE_LOSS_SHARE", 0.76)), 0.0, 1.0)
        self.source_bad_entry_probability = _clamp(
            float(_cfg("V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY", 0.72)),
            0.0,
            1.0,
        )
        self.source_severe_entry_probability = _clamp(
            float(_cfg("V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY", 0.46)),
            0.0,
            1.0,
        )
        self.source_cooldown_in_red_mult = _clamp(
            float(_cfg("V2_QUALITY_SOURCE_COOLDOWN_IN_RED_MULT", 0.88)),
            0.10,
            1.0,
        )
        self.source_boost_mult = max(0.50, float(_cfg("V2_QUALITY_SOURCE_BOOST_MULT", 1.28)))
        self.source_cut_mult = _clamp(float(_cfg("V2_QUALITY_SOURCE_CUT_MULT", 0.62)), 0.10, 1.0)
        self.source_min_share = _clamp(float(_cfg("V2_QUALITY_SOURCE_MIN_SHARE", 0.08)), 0.0, 0.40)
        self.source_max_share = _clamp(float(_cfg("V2_QUALITY_SOURCE_MAX_SHARE", 0.58)), 0.10, 1.0)
        self.provisional_core_enabled = bool(_cfg("V2_QUALITY_PROVISIONAL_CORE_ENABLED", True))
        self.provisional_core_min_score = max(0.0, _safe_float(_cfg("V2_QUALITY_PROVISIONAL_CORE_MIN_SCORE", 92), 92.0))
        self.provisional_core_min_liquidity = max(
            0.0,
            _safe_float(_cfg("V2_QUALITY_PROVISIONAL_CORE_MIN_LIQUIDITY_USD", 120000), 120000.0),
        )
        self.provisional_core_min_volume_5m = max(
            0.0,
            _safe_float(_cfg("V2_QUALITY_PROVISIONAL_CORE_MIN_VOLUME_5M_USD", 3500), 3500.0),
        )
        self.provisional_core_min_abs_change_5m = max(
            0.0,
            _safe_float(_cfg("V2_QUALITY_PROVISIONAL_CORE_MIN_ABS_CHANGE_5M", 2.8), 2.8),
        )
        self.provisional_core_max_share = _clamp(
            _safe_float(_cfg("V2_QUALITY_PROVISIONAL_CORE_MAX_SHARE", 0.35), 0.35),
            0.0,
            0.80,
        )
        self.log_top_symbols = max(1, int(_cfg("V2_QUALITY_LOG_TOP_SYMBOLS", 8)))
        self.core_min_abs = max(0, int(_cfg("V2_QUALITY_CORE_MIN_ABS", 1)))
        self.core_fallback_from_explore_enabled = bool(_cfg("V2_QUALITY_CORE_FALLBACK_FROM_EXPLORE_ENABLED", True))
        self.core_fallback_score_min = max(
            0.0,
            _safe_float(_cfg("V2_QUALITY_CORE_FALLBACK_SCORE_MIN", 90), 90.0),
        )
        self.core_fallback_max_share = _clamp(
            _safe_float(_cfg("V2_QUALITY_CORE_FALLBACK_MAX_SHARE", 0.25), 0.25),
            0.0,
            0.60,
        )
        self._next_refresh_ts = 0.0
        self._symbol_stats: dict[str, dict[str, float]] = {}
        self._cluster_stats: dict[str, dict[str, float]] = {}
        self._source_stats: dict[str, dict[str, float]] = {}
        self._candidate_source: dict[str, tuple[str, float]] = {}
        self._symbol_history: deque[tuple[float, str]] = deque()
        self._symbol_last_selected_ts: dict[str, float] = {}
        self._last_rotation_signature = ""

    @staticmethod
    def _symbol_key(token: dict[str, Any]) -> str:
        sym = str(token.get("symbol", "") or "").strip().upper()
        if sym:
            return sym
        name = str(token.get("name", "") or "").strip().upper()
        return name or "-"

    @staticmethod
    def _cluster_key(token: dict[str, Any], score_data: dict[str, Any]) -> str:
        score = int(_safe_float((score_data or {}).get("score"), 0.0))
        if score >= 90:
            score_band = "s3"
        elif score >= 80:
            score_band = "s2"
        elif score >= 70:
            score_band = "s1"
        else:
            score_band = "s0"

        liq = _safe_float(token.get("liquidity"), 0.0)
        if liq >= 500_000:
            liq_band = "l3"
        elif liq >= 120_000:
            liq_band = "l2"
        elif liq >= 35_000:
            liq_band = "l1"
        else:
            liq_band = "l0"

        vol = _safe_float(token.get("volume_5m"), 0.0)
        if vol >= 12_000:
            vol_band = "v3"
        elif vol >= 3_500:
            vol_band = "v2"
        elif vol >= 800:
            vol_band = "v1"
        else:
            vol_band = "v0"

        mom_abs = abs(_safe_float(token.get("price_change_5m"), 0.0))
        if mom_abs >= 8.0:
            mom_band = "m3"
        elif mom_abs >= 3.0:
            mom_band = "m2"
        elif mom_abs >= 1.0:
            mom_band = "m1"
        else:
            mom_band = "m0"

        risk = str(token.get("risk_level", "MEDIUM") or "MEDIUM").strip().upper()
        risk_band = {"LOW": "r0", "MEDIUM": "r1", "HIGH": "r2"}.get(risk, "r1")
        return f"{score_band}|{liq_band}|{vol_band}|{mom_band}|{risk_band}"

    @staticmethod
    def _roll_stats(pnls: list[float]) -> dict[str, float]:
        vals = [float(x or 0.0) for x in (pnls or [])]
        if not vals:
            return {"trades": 0.0, "avg_pnl": 0.0, "loss_share": 0.0, "win_rate": 0.0, "sum_pnl": 0.0}
        wins = sum(1 for x in vals if x > 0.0)
        losses = sum(1 for x in vals if x <= 0.0)
        return {
            "trades": float(len(vals)),
            "avg_pnl": float(sum(vals) / float(max(1, len(vals)))),
            "loss_share": float(losses) / float(max(1, len(vals))),
            "win_rate": float(wins) / float(max(1, len(vals))),
            "sum_pnl": float(sum(vals)),
        }

    def _is_good(self, stats: dict[str, float], *, min_trades: int) -> bool:
        trades = int(_safe_float(stats.get("trades"), 0.0))
        if trades < int(min_trades):
            return False
        avg = _safe_float(stats.get("avg_pnl"), 0.0)
        loss_share = _safe_float(stats.get("loss_share"), 0.0)
        return (
            float(avg) > float(self.min_avg_pnl_usd)
            and float(loss_share) <= float(self.max_loss_share)
        )

    def _is_bad(self, stats: dict[str, float], *, min_trades: int) -> bool:
        trades = int(_safe_float(stats.get("trades"), 0.0))
        if trades < int(min_trades):
            return False
        avg = _safe_float(stats.get("avg_pnl"), 0.0)
        loss_share = _safe_float(stats.get("loss_share"), 0.0)
        return (
            float(avg) <= float(self.bad_avg_pnl_usd)
            or float(loss_share) >= float(self.bad_loss_share)
        )

    def _source_mult(self, source: str) -> float:
        if not self.source_budget_enabled:
            return 1.0
        src = str(source or "unknown").strip().lower() or "unknown"
        stats = dict(self._source_stats.get(src, {}))
        trades = int(stats.get("trades", 0.0) or 0.0)
        if trades < int(self.source_min_trades):
            return 1.0
        avg = float(stats.get("avg_pnl", 0.0) or 0.0)
        good = float(self.source_good_avg)
        bad = float(self.source_bad_avg)
        if avg >= good:
            return float(self.source_boost_mult)
        if avg <= bad:
            return float(self.source_cut_mult)
        if avg >= 0.0:
            span = max(0.000001, good)
            k = _clamp(avg / span, 0.0, 1.0)
            return float(1.0 + ((float(self.source_boost_mult) - 1.0) * k))
        span = max(0.000001, abs(bad))
        k = _clamp(abs(avg) / span, 0.0, 1.0)
        return float(1.0 - ((1.0 - float(self.source_cut_mult)) * k))

    def _source_quality_flags(self, source: str) -> tuple[bool, bool]:
        src = str(source or "unknown").strip().lower() or "unknown"
        stats = dict(self._source_stats.get(src, {}))
        trades = int(stats.get("trades", 0.0) or 0.0)
        if trades < int(self.source_min_trades):
            return False, False
        avg = float(stats.get("avg_pnl", 0.0) or 0.0)
        loss_share = float(stats.get("loss_share", 0.0) or 0.0)
        bad = (avg <= float(self.source_bad_avg)) or (loss_share >= float(self.source_bad_loss_share))
        severe = (avg <= float(self.source_severe_avg)) or (loss_share >= float(self.source_severe_loss_share))
        return bool(bad), bool(severe)

    def _source_entry_probability(self, source: str, regime: str) -> float:
        if not self.source_budget_enabled:
            return 1.0
        bad, severe = self._source_quality_flags(source)
        if severe:
            p = float(self.source_severe_entry_probability)
        elif bad:
            p = float(self.source_bad_entry_probability)
        else:
            p = 1.0
        if str(regime or "").strip().upper() == "RED" and p < 1.0:
            p *= float(self.source_cooldown_in_red_mult)
        return _clamp(float(p), 0.0, 1.0)

    def _prune_candidate_source(self, now_ts: float) -> None:
        ttl = max(float(self.source_window_seconds) * 2.0, float(self.window_seconds) * 2.0, 7200.0)
        self._candidate_source = {
            cid: row
            for cid, row in self._candidate_source.items()
            if float((row or ("", 0.0))[1] or 0.0) >= (now_ts - ttl)
        }

    def _refresh_stats(self, *, auto_trader: Any, now_ts: float) -> dict[str, Any]:
        closed_positions = list(getattr(auto_trader, "closed_positions", []) or [])
        cutoff = float(now_ts) - float(self.window_seconds)
        cutoff_source = float(now_ts) - float(self.source_window_seconds)
        symbol_rows: dict[str, list[float]] = defaultdict(list)
        cluster_rows: dict[str, list[float]] = defaultdict(list)
        source_rows: dict[str, list[float]] = defaultdict(list)

        self._prune_candidate_source(now_ts)
        for pos in closed_positions:
            closed_ts = _datetime_to_ts(getattr(pos, "closed_at", None))
            if closed_ts <= 0.0 or closed_ts < cutoff:
                continue
            pnl = float(getattr(pos, "pnl_usd", 0.0) or 0.0)
            symbol = str(getattr(pos, "symbol", "") or "").strip().upper() or "-"
            cluster = str(getattr(pos, "token_cluster_key", "") or "").strip().lower()
            if symbol:
                symbol_rows[symbol].append(pnl)
            if cluster:
                cluster_rows[cluster].append(pnl)

            if closed_ts >= cutoff_source:
                cid = str(getattr(pos, "candidate_id", "") or "").strip()
                src = "unknown"
                if cid and cid in self._candidate_source:
                    src = str((self._candidate_source.get(cid) or ("unknown", 0.0))[0] or "unknown").strip().lower() or "unknown"
                source_rows[src].append(pnl)

        self._symbol_stats = {k: self._roll_stats(v) for k, v in symbol_rows.items()}
        self._cluster_stats = {k: self._roll_stats(v) for k, v in cluster_rows.items()}
        self._source_stats = {k: self._roll_stats(v) for k, v in source_rows.items()}
        self._next_refresh_ts = float(now_ts) + float(self.refresh_seconds)

        core_symbols = [
            (k, v)
            for k, v in self._symbol_stats.items()
            if self._is_good(v, min_trades=int(self.min_symbol_trades))
        ]
        cooldown_symbols = [
            (k, v)
            for k, v in self._symbol_stats.items()
            if self._is_bad(v, min_trades=int(self.min_symbol_trades))
        ]
        core_symbols.sort(key=lambda row: float((row[1] or {}).get("avg_pnl", 0.0) or 0.0), reverse=True)
        cooldown_symbols.sort(key=lambda row: float((row[1] or {}).get("avg_pnl", 0.0) or 0.0))
        top_n = int(self.log_top_symbols)
        top_core = [
            f"{sym}:{float(stats.get('avg_pnl', 0.0) or 0.0):.4f}/{int(stats.get('trades', 0.0) or 0)}"
            for sym, stats in core_symbols[:top_n]
        ]
        top_cooldown = [
            f"{sym}:{float(stats.get('avg_pnl', 0.0) or 0.0):.4f}/{int(stats.get('trades', 0.0) or 0)}"
            for sym, stats in cooldown_symbols[:top_n]
        ]
        signature = json.dumps(
            {
                "core": top_core,
                "cooldown": top_cooldown,
                "source_count": len(self._source_stats),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        changed = signature != self._last_rotation_signature
        if changed:
            self._last_rotation_signature = signature
            logger.info(
                "V2_QUALITY_ROTATE symbols_core=%s symbols_cooldown=%s sources=%s top_core=%s top_cooldown=%s",
                len(core_symbols),
                len(cooldown_symbols),
                len(self._source_stats),
                ",".join(top_core) if top_core else "-",
                ",".join(top_cooldown) if top_cooldown else "-",
            )
        return {
            "refreshed": True,
            "rotation_changed": bool(changed),
            "symbols_core": int(len(core_symbols)),
            "symbols_cooldown": int(len(cooldown_symbols)),
            "sources_tracked": int(len(self._source_stats)),
        }

    def _source_caps(self, rows: list[dict[str, Any]], *, target: int) -> dict[str, int]:
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            src = str(row.get("source", "unknown") or "unknown").strip().lower() or "unknown"
            by_source[src].append(row)
        if not by_source:
            return {}
        if not self.source_budget_enabled:
            return {src: len(items) for src, items in by_source.items()}

        base_target = int(max(1, target))
        weighted: dict[str, float] = {}
        for src, items in by_source.items():
            mult = max(0.10, float(self._source_mult(src)))
            weighted[src] = float(len(items)) * mult
        total_w = float(sum(weighted.values()))
        if total_w <= 0.0:
            return {src: len(items) for src, items in by_source.items()}

        shares: dict[str, float] = {}
        for src, items in by_source.items():
            raw_share = float(weighted.get(src, 0.0)) / total_w
            st = dict(self._source_stats.get(src, {}))
            min_share = float(self.source_min_share) if int(st.get("trades", 0.0) or 0.0) >= int(self.source_min_trades) else 0.0
            shares[src] = max(0.0, min(float(self.source_max_share), max(raw_share, min_share)))

        share_sum = float(sum(shares.values())) or 1.0
        caps: dict[str, int] = {}
        rem = int(base_target)
        srcs = sorted(shares.keys(), key=lambda s: float(shares[s]), reverse=True)
        for idx, src in enumerate(srcs):
            available = len(by_source.get(src, []))
            src_max = int(math.ceil(float(base_target) * float(self.source_max_share)))
            src_max = max(1, src_max)
            if idx == len(srcs) - 1:
                cap = min(available, max(0, rem))
            else:
                cap = int(math.ceil((float(shares[src]) / share_sum) * float(base_target)))
                cap = min(available, max(0, cap))
                rem = max(0, rem - cap)
            if cap <= 0 and available > 0:
                cap = 1
                rem = max(0, rem - 1)
            caps[src] = int(min(cap, src_max))
        return caps

    def _prune_symbol_history(self, now_ts: float) -> None:
        cutoff = float(now_ts) - float(self.symbol_concentration_window_seconds)
        while self._symbol_history and float(self._symbol_history[0][0]) < cutoff:
            self._symbol_history.popleft()
        reentry_keep = max(float(self.symbol_reentry_min_seconds) * 4.0, float(self.symbol_concentration_window_seconds))
        self._symbol_last_selected_ts = {
            str(k): float(v)
            for k, v in self._symbol_last_selected_ts.items()
            if float(v) >= (float(now_ts) - reentry_keep)
        }

    def _register_candidate_sources(self, candidates: list[tuple[dict[str, Any], dict[str, Any]]], now_ts: float) -> None:
        for token, _ in candidates:
            cid = str((token or {}).get("_candidate_id", "") or "").strip()
            if not cid:
                continue
            src = str((token or {}).get("source", "unknown") or "unknown").strip().lower() or "unknown"
            self._candidate_source[cid] = (src, float(now_ts))

    def filter_candidates(
        self,
        *,
        candidates: list[tuple[dict[str, Any], dict[str, Any]]],
        auto_trader: Any,
        market_mode: str,
    ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
        regime = str(market_mode or "YELLOW").strip().upper() or "YELLOW"
        if not candidates:
            return [], {
                "enabled": bool(self.enabled),
                "in_total": 0,
                "out_total": 0,
                "core_out": 0,
                "explore_out": 0,
                "probe_out": 0,
                "dropped": [],
                "drop_counts": {},
                "regime": regime,
            }
        if not self.enabled:
            return list(candidates), {
                "enabled": False,
                "in_total": len(candidates),
                "out_total": len(candidates),
                "core_out": sum(
                    1
                    for token, _ in candidates
                    if str((token or {}).get("_entry_tier", "")).strip().upper() == "A"
                ),
                "explore_out": sum(
                    1
                    for token, _ in candidates
                    if str((token or {}).get("_entry_tier", "")).strip().upper() != "A"
                ),
                "probe_out": 0,
                "dropped": [],
                "drop_counts": {},
                "regime": regime,
            }

        now_ts = time.time()
        self._register_candidate_sources(candidates, now_ts)
        refresh_meta: dict[str, Any] = {}
        if now_ts >= float(self._next_refresh_ts):
            refresh_meta = self._refresh_stats(auto_trader=auto_trader, now_ts=now_ts)

        self._prune_symbol_history(now_ts)
        hist_by_symbol: dict[str, int] = defaultdict(int)
        fresh_cutoff = float(now_ts) - float(self.fresh_symbol_window_seconds)
        fresh_seen_symbols: set[str] = set()
        for ts, sym in self._symbol_history:
            if not sym:
                continue
            key = str(sym)
            hist_by_symbol[key] = int(hist_by_symbol.get(key, 0)) + 1
            if float(ts) >= fresh_cutoff:
                fresh_seen_symbols.add(key)

        rows: list[dict[str, Any]] = []
        for token, score_data in candidates:
            token_d = dict(token or {})
            score_d = dict(score_data or {})
            symbol = self._symbol_key(token_d)
            cluster_key = self._cluster_key(token_d, score_d)
            src = str(token_d.get("source", "unknown") or "unknown").strip().lower() or "unknown"
            sym_stats = dict(self._symbol_stats.get(symbol, {}))
            clu_stats = dict(self._cluster_stats.get(cluster_key.lower(), {}))
            sym_good = self._is_good(sym_stats, min_trades=int(self.min_symbol_trades))
            clu_good = self._is_good(clu_stats, min_trades=int(self.min_cluster_trades))
            sym_bad = self._is_bad(sym_stats, min_trades=int(self.min_symbol_trades))
            clu_bad = self._is_bad(clu_stats, min_trades=int(self.min_cluster_trades))

            score = _safe_float(score_d.get("score"), 0.0)
            liq = max(0.0, _safe_float(token_d.get("liquidity"), 0.0))
            vol = max(0.0, _safe_float(token_d.get("volume_5m"), 0.0))
            abs_change = abs(_safe_float(token_d.get("price_change_5m"), 0.0))
            provisional_core = (
                bool(self.provisional_core_enabled)
                and (not sym_bad)
                and (not clu_bad)
                and score >= float(self.provisional_core_min_score)
                and liq >= float(self.provisional_core_min_liquidity)
                and vol >= float(self.provisional_core_min_volume_5m)
                and abs_change >= float(self.provisional_core_min_abs_change_5m)
            )
            if sym_good or clu_good:
                bucket = "core"
            elif provisional_core:
                bucket = "core_provisional"
            elif sym_bad or clu_bad:
                bucket = "cooldown"
            else:
                bucket = "explore"

            is_fresh_symbol = str(symbol) not in fresh_seen_symbols
            base_priority = float(score + (1.3 * math.log1p(liq / 10000.0)) + (1.8 * math.log1p(vol / 1200.0)))
            ev_signal = max(float(sym_stats.get("avg_pnl", 0.0) or 0.0), float(clu_stats.get("avg_pnl", 0.0) or 0.0))
            ev_bonus = _clamp(ev_signal * 360.0, -6.0, 8.0)
            src_mult = self._source_mult(src)
            source_entry_prob = self._source_entry_probability(src, regime)
            if bucket in {"core", "core_provisional"}:
                bucket_bias = 6.0
            elif bucket == "cooldown":
                bucket_bias = -6.0
            else:
                bucket_bias = 0.0
            fresh_bonus = float(self.fresh_symbol_boost) if bool(is_fresh_symbol) else 0.0
            priority = (base_priority + ev_bonus + bucket_bias + fresh_bonus) * float(src_mult)
            rows.append(
                {
                    "token": token_d,
                    "score_data": score_d,
                    "symbol": symbol,
                    "cluster_key": cluster_key.lower(),
                    "source": src,
                    "bucket": bucket,
                    "priority": float(priority),
                    "ev_signal": float(ev_signal),
                    "src_mult": float(src_mult),
                    "source_entry_probability": float(source_entry_prob),
                    "is_probe": False,
                    "is_fresh_symbol": bool(is_fresh_symbol),
                    "is_provisional_core": bool(provisional_core),
                }
            )

        rows.sort(key=lambda row: float(row.get("priority", 0.0) or 0.0), reverse=True)
        core_rows_native = [row for row in rows if str(row.get("bucket")) == "core"]
        core_rows_provisional = [row for row in rows if str(row.get("bucket")) == "core_provisional"]
        core_rows = list(core_rows_native) + list(core_rows_provisional)
        explore_rows = [row for row in rows if str(row.get("bucket")) == "explore"]
        cooldown_rows = [row for row in rows if str(row.get("bucket")) == "cooldown"]
        core_fallback_rows: list[dict[str, Any]] = []

        # Anti-stall fallback: when quality says "no core", promote best explore rows to A/core.
        if (
            bool(self.core_fallback_from_explore_enabled)
            and str(regime) != "RED"
            and int(self.core_min_abs) > 0
            and len(core_rows) < int(self.core_min_abs)
            and explore_rows
        ):
            need = int(self.core_min_abs) - len(core_rows)
            fallback_cap = int(math.floor(float(len(rows)) * float(self.core_fallback_max_share)))
            if fallback_cap <= 0 and need > 0:
                fallback_cap = 1
            fallback_cap = max(0, fallback_cap)
            used_cids: set[str] = set()
            for row in core_rows:
                cid = str((row.get("token") or {}).get("_candidate_id", "") or "")
                if cid:
                    used_cids.add(cid)
            for row in explore_rows:
                if need <= 0 or len(core_fallback_rows) >= fallback_cap:
                    break
                score_data = dict(row.get("score_data") or {})
                score = _safe_float(score_data.get("score"), 0.0)
                if score < float(self.core_fallback_score_min):
                    continue
                cid = str((row.get("token") or {}).get("_candidate_id", "") or "")
                if cid and cid in used_cids:
                    continue
                row2 = dict(row)
                row2["bucket"] = "core_fallback"
                row2["priority"] = float(row2.get("priority", 0.0) or 0.0) + 5.0
                core_fallback_rows.append(row2)
                if cid:
                    used_cids.add(cid)
                need -= 1
            if core_fallback_rows:
                promoted_cids = {
                    str((row.get("token") or {}).get("_candidate_id", "") or "")
                    for row in core_fallback_rows
                    if str((row.get("token") or {}).get("_candidate_id", "") or "")
                }
                explore_rows = [
                    row
                    for row in explore_rows
                    if str((row.get("token") or {}).get("_candidate_id", "") or "") not in promoted_cids
                ]
                core_rows_native.extend(core_fallback_rows)
                core_rows.extend(core_fallback_rows)

        dropped: list[dict[str, Any]] = []
        drop_counts: dict[str, int] = defaultdict(int)
        probe_rows: list[dict[str, Any]] = []
        probe_p = float(self.cooldown_probe_probability)
        if regime == "RED":
            probe_p *= 0.75
        for row in cooldown_rows:
            token = dict(row.get("token") or {})
            cid = str(token.get("_candidate_id", "") or "")
            seed = f"{cid}|{int(now_ts // 300)}|{regime}"
            if _deterministic_probability(seed) <= probe_p:
                row2 = dict(row)
                row2["is_probe"] = True
                row2["bucket"] = "explore_probe"
                row2["priority"] = float(row2.get("priority", 0.0) or 0.0) * 0.92
                probe_rows.append(row2)
            else:
                drop_counts["cooldown_bucket"] += 1
                dropped.append(
                    {
                        "candidate_id": str(token.get("_candidate_id", "") or ""),
                        "symbol": str(token.get("symbol", "") or ""),
                        "reason": "cooldown_bucket",
                        "bucket": "cooldown",
                    }
                )

        explore_pool = sorted(explore_rows + probe_rows, key=lambda row: float(row.get("priority", 0.0) or 0.0), reverse=True)
        target_total = len(rows)
        explore_cap = int(math.floor(float(target_total) * float(self.explore_max_share)))
        if regime == "RED":
            explore_cap = min(explore_cap, 1)
        explore_cap = max(0, min(len(explore_pool), explore_cap))
        reserve = int(math.ceil(float(target_total) * float(self.explore_reserve_share)))
        reserve = max(reserve, int(self.explore_min_abs))
        if regime == "RED":
            reserve = min(reserve, 1)
        reserve = max(0, min(len(explore_pool), reserve))
        explore_target = int(min(len(explore_pool), max(explore_cap, reserve)))
        core_target = max(0, int(target_total) - int(explore_target))

        source_caps = self._source_caps(core_rows + explore_pool, target=target_total)
        source_used: dict[str, int] = defaultdict(int)
        selected_symbol_counts: dict[str, int] = defaultdict(int)
        selected: list[dict[str, Any]] = []
        selected_addrs: set[str] = set()
        symbol_reentry_override_floor = float(getattr(config, "SAFE_MIN_VOLUME_5M_USD", 0.0) or 0.0) * float(
            self.symbol_reentry_override_vol_mult
        )

        def _try_take(row: dict[str, Any]) -> bool:
            token = dict(row.get("token") or {})
            addr_norm = normalize_address(str(token.get("address", "") or ""))
            candidate_id = str(token.get("_candidate_id", "") or "")
            if addr_norm:
                uniq_key = f"addr:{addr_norm}"
            elif candidate_id:
                uniq_key = f"cid:{candidate_id}"
            else:
                uniq_key = f"row:{id(row)}"
            source = str(row.get("source", "unknown") or "unknown")
            symbol = str(row.get("symbol", "-") or "-")
            if uniq_key in selected_addrs:
                drop_counts["duplicate_address"] += 1
                return False
            source_p = _clamp(_safe_float(row.get("source_entry_probability"), 1.0), 0.0, 1.0)
            if source_p < 1.0:
                slot = int(now_ts // 600)
                seed = f"{candidate_id or uniq_key}|{source}|{regime}|srcp|{slot}"
                if _deterministic_probability(seed) > source_p:
                    drop_counts["source_ev_cooldown"] += 1
                    dropped.append(
                        {
                            "candidate_id": str(token.get("_candidate_id", "") or ""),
                            "symbol": str(token.get("symbol", "") or ""),
                            "reason": "source_ev_cooldown",
                            "bucket": str(row.get("bucket", "")),
                        }
                    )
                    return False
            cap = int(source_caps.get(source, 10**9))
            if int(source_used.get(source, 0)) >= cap:
                drop_counts["source_budget"] += 1
                dropped.append(
                    {
                        "candidate_id": str(token.get("_candidate_id", "") or ""),
                        "symbol": str(token.get("symbol", "") or ""),
                        "reason": "source_budget",
                        "bucket": str(row.get("bucket", "")),
                    }
                )
                return False

            if int(self.symbol_reentry_min_seconds) > 0:
                last_selected_ts = float(self._symbol_last_selected_ts.get(symbol, 0.0) or 0.0)
                reentry_left = float(self.symbol_reentry_min_seconds) - (float(now_ts) - last_selected_ts)
                vol5m = max(0.0, _safe_float(token.get("volume_5m"), 0.0))
                if reentry_left > 0.0 and vol5m < symbol_reentry_override_floor:
                    drop_counts["symbol_reentry_cooldown"] += 1
                    dropped.append(
                        {
                            "candidate_id": str(token.get("_candidate_id", "") or ""),
                            "symbol": str(token.get("symbol", "") or ""),
                            "reason": "symbol_reentry_cooldown",
                            "bucket": str(row.get("bucket", "")),
                        }
                    )
                    return False

            projected_total = int(len(self._symbol_history)) + int(len(selected)) + 1
            allowed = max(int(self.symbol_min_abs_cap), int(math.floor(float(projected_total) * float(self.symbol_max_share))))
            used_symbol = int(hist_by_symbol.get(symbol, 0)) + int(selected_symbol_counts.get(symbol, 0))
            if used_symbol >= allowed:
                drop_counts["symbol_concentration"] += 1
                dropped.append(
                    {
                        "candidate_id": str(token.get("_candidate_id", "") or ""),
                        "symbol": str(token.get("symbol", "") or ""),
                        "reason": "symbol_concentration",
                        "bucket": str(row.get("bucket", "")),
                    }
                )
                return False

            selected.append(row)
            selected_addrs.add(uniq_key)
            source_used[source] = int(source_used.get(source, 0)) + 1
            selected_symbol_counts[symbol] = int(selected_symbol_counts.get(symbol, 0)) + 1
            return True

        core_taken = 0
        for row in core_rows_native:
            if core_taken >= int(core_target):
                break
            if _try_take(row):
                core_taken += 1

        provisional_cap = int(math.ceil(float(target_total) * float(self.provisional_core_max_share)))
        provisional_cap = max(0, provisional_cap)
        provisional_taken = 0
        for row in core_rows_provisional:
            if core_taken >= int(core_target):
                break
            if provisional_cap > 0 and provisional_taken >= provisional_cap:
                drop_counts["provisional_core_cap"] += 1
                dropped.append(
                    {
                        "candidate_id": str((row.get("token") or {}).get("_candidate_id", "") or ""),
                        "symbol": str((row.get("token") or {}).get("symbol", "") or ""),
                        "reason": "provisional_core_cap",
                        "bucket": "core_provisional",
                    }
                )
                continue
            if _try_take(row):
                core_taken += 1
                provisional_taken += 1

        explore_taken = 0
        for row in explore_pool:
            if explore_taken >= int(explore_target):
                break
            if _try_take(row):
                explore_taken += 1

        for row in (core_rows_native + core_rows_provisional + explore_pool):
            if len(selected) >= int(target_total):
                break
            _try_take(row)

        if not selected and (core_rows or explore_pool):
            for row in (core_rows_native + core_rows_provisional + explore_pool):
                token = dict(row.get("token") or {})
                addr_norm = normalize_address(str(token.get("address", "") or ""))
                candidate_id = str(token.get("_candidate_id", "") or "")
                if addr_norm:
                    uniq_key = f"addr:{addr_norm}"
                elif candidate_id:
                    uniq_key = f"cid:{candidate_id}"
                else:
                    uniq_key = f"row:{id(row)}"
                selected = [row]
                selected_addrs = {uniq_key}
                source = str(row.get("source", "unknown") or "unknown")
                symbol = str(row.get("symbol", "-") or "-")
                source_used[source] = int(source_used.get(source, 0)) + 1
                selected_symbol_counts[symbol] = int(selected_symbol_counts.get(symbol, 0)) + 1
                break

        for row in selected:
            sym = str(row.get("symbol", "") or "")
            if sym:
                self._symbol_history.append((float(now_ts), sym))
                self._symbol_last_selected_ts[sym] = float(now_ts)

        tagged: list[tuple[dict[str, Any], dict[str, Any]]] = []
        core_out = 0
        provisional_core_out = 0
        fallback_core_out = 0
        explore_out = 0
        probe_out = 0
        fresh_symbol_out = 0
        for row in selected:
            token = dict(row.get("token") or {})
            score_data = dict(row.get("score_data") or {})
            bucket = str(row.get("bucket", "explore") or "explore")
            token["_quality_bucket"] = bucket
            token["_quality_priority"] = float(row.get("priority", 0.0) or 0.0)
            token["_quality_ev_signal"] = float(row.get("ev_signal", 0.0) or 0.0)
            token["_quality_source_mult"] = float(row.get("src_mult", 1.0) or 1.0)
            token["_quality_source_entry_probability"] = float(
                _safe_float(row.get("source_entry_probability"), 1.0)
            )
            token["_quality_source"] = str(row.get("source", "unknown") or "unknown")
            token["_quality_cluster_key"] = str(row.get("cluster_key", "") or "")
            token["_quality_is_fresh_symbol"] = bool(row.get("is_fresh_symbol", False))
            if bool(row.get("is_fresh_symbol", False)):
                fresh_symbol_out += 1
            if bucket in {"core", "core_provisional", "core_fallback"}:
                token["_entry_tier"] = "A"
                core_out += 1
                if bucket == "core_provisional":
                    provisional_core_out += 1
                    token["_quality_provisional_core"] = True
                elif bucket == "core_fallback":
                    fallback_core_out += 1
                    token["_quality_core_fallback"] = True
            else:
                token["_entry_tier"] = "B"
                explore_out += 1
                if bool(row.get("is_probe", False)):
                    probe_out += 1
                    token["_entry_channel_size_mult"] = float(token.get("_entry_channel_size_mult", 1.0) or 1.0) * float(
                        self.cooldown_size_mult
                    )
                    token["_entry_channel_hold_mult"] = float(token.get("_entry_channel_hold_mult", 1.0) or 1.0) * float(
                        self.cooldown_hold_mult
                    )
                    token["_entry_channel_edge_usd_mult"] = float(token.get("_entry_channel_edge_usd_mult", 1.0) or 1.0) * float(
                        self.cooldown_edge_usd_mult
                    )
                    token["_entry_channel_edge_pct_mult"] = float(token.get("_entry_channel_edge_pct_mult", 1.0) or 1.0) * float(
                        self.cooldown_edge_percent_mult
                    )
            tagged.append((token, score_data))

        total_out = len(tagged)
        symbol_counts_out: dict[str, int] = defaultdict(int)
        for token, _ in tagged:
            sk = self._symbol_key(token)
            symbol_counts_out[sk] = int(symbol_counts_out.get(sk, 0)) + 1
        max_symbol_share = 0.0
        if total_out > 0 and symbol_counts_out:
            max_symbol_share = max(float(v) / float(total_out) for v in symbol_counts_out.values())
        source_share_out = {
            str(k): round(float(v) / float(max(1, total_out)), 6)
            for k, v in source_used.items()
        }

        meta = {
            "enabled": True,
            "regime": regime,
            "in_total": len(candidates),
            "out_total": int(total_out),
            "core_in": len(core_rows),
            "core_native_in": len(core_rows_native),
            "core_provisional_in": len(core_rows_provisional),
            "core_fallback_in": len(core_fallback_rows),
            "explore_in": len(explore_rows),
            "cooldown_in": len(cooldown_rows),
            "core_out": int(core_out),
            "core_provisional_out": int(provisional_core_out),
            "core_fallback_out": int(fallback_core_out),
            "explore_out": int(explore_out),
            "probe_out": int(probe_out),
            "fresh_symbol_out": int(fresh_symbol_out),
            "unique_symbols_out": int(len(symbol_counts_out)),
            "core_target": int(core_target),
            "explore_cap": int(explore_cap),
            "explore_target": int(explore_target),
            "explore_quota": int(explore_target),
            "source_caps": dict(source_caps),
            "source_used": dict(source_used),
            "source_share_out": source_share_out,
            "drop_counts": dict(drop_counts),
            "dropped": dropped,
            "core_share_out": round((float(core_out) / float(max(1, total_out))), 6),
            "explore_share_out": round((float(explore_out) / float(max(1, total_out))), 6),
            "max_symbol_share_out": round(float(max_symbol_share), 6),
            **dict(refresh_meta or {}),
        }
        return tagged, meta


class SafetyBudgetController:
    """Per-cycle cap for expensive safety checks."""

    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "V2_SAFETY_BUDGET_ENABLED", False))
        self.max_checks_per_cycle = max(1, int(getattr(config, "V2_SAFETY_BUDGET_MAX_PER_CYCLE", 80) or 80))
        self.per_source_caps = {
            k: max(1, int(v))
            for k, v in _parse_float_map(
                str(
                    getattr(
                        config,
                        "V2_SAFETY_BUDGET_PER_SOURCE",
                        "onchain:48,onchain+market:48,dexscreener:42,geckoterminal:42,watchlist:18,dex_boosts:18",
                    )
                    or ""
                )
            ).items()
        }
        self._used_total = 0
        self._used_by_source: dict[str, int] = defaultdict(int)

    def reset_cycle(self) -> None:
        self._used_total = 0
        self._used_by_source.clear()

    def allow(self, token: dict[str, Any]) -> bool:
        if not self.enabled:
            return True
        if int(self._used_total) >= int(self.max_checks_per_cycle):
            return False
        src = str(token.get("source", "unknown") or "unknown").strip().lower()
        cap = int(self.per_source_caps.get(src, self.max_checks_per_cycle))
        if int(self._used_by_source[src]) >= cap:
            return False
        self._used_total += 1
        self._used_by_source[src] = int(self._used_by_source[src]) + 1
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "used_total": int(self._used_total),
            "max_total": int(self.max_checks_per_cycle),
            "used_by_source": dict(self._used_by_source),
        }


class UnifiedCalibrator:
    """Runtime calibration from unified dataset (safe bounded in-memory adjustments)."""

    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "V2_CALIBRATION_ENABLED", False))
        self.apply_interval_seconds = max(120, int(getattr(config, "V2_CALIBRATION_INTERVAL_SECONDS", 900) or 900))
        self.min_closed = max(20, int(getattr(config, "V2_CALIBRATION_MIN_CLOSED", 120) or 120))
        self.lookback_rows = max(50, int(getattr(config, "V2_CALIBRATION_LOOKBACK_ROWS", 2000) or 2000))
        self.smooth_alpha = _clamp(float(getattr(config, "V2_CALIBRATION_SMOOTH_ALPHA", 0.35) or 0.35), 0.05, 1.0)
        self.edge_usd_min = max(0.0, float(getattr(config, "V2_CALIBRATION_EDGE_USD_MIN", 0.010) or 0.010))
        self.edge_usd_max = max(self.edge_usd_min, float(getattr(config, "V2_CALIBRATION_EDGE_USD_MAX", 0.120) or 0.120))
        self.edge_percent_min = max(0.0, float(getattr(config, "V2_CALIBRATION_EDGE_PERCENT_MIN", 0.35) or 0.35))
        self.edge_percent_max = max(
            self.edge_percent_min,
            float(getattr(config, "V2_CALIBRATION_EDGE_PERCENT_MAX", 3.00) or 3.00),
        )
        self.edge_percent_step_max = max(
            0.0,
            float(getattr(config, "V2_CALIBRATION_EDGE_PERCENT_STEP_MAX", 0.06) or 0.06),
        )
        self.edge_usd_step_max = max(
            0.0,
            float(getattr(config, "V2_CALIBRATION_EDGE_USD_STEP_MAX", 0.0030) or 0.0030),
        )
        self.edge_percent_drift_up_24h = max(
            0.0,
            float(getattr(config, "V2_CALIBRATION_EDGE_PERCENT_DRIFT_UP_24H", 0.25) or 0.25),
        )
        self.edge_percent_drift_down_24h = max(
            0.0,
            float(getattr(config, "V2_CALIBRATION_EDGE_PERCENT_DRIFT_DOWN_24H", 0.20) or 0.20),
        )
        self.edge_usd_drift_up_24h = max(
            0.0,
            float(getattr(config, "V2_CALIBRATION_EDGE_USD_DRIFT_UP_24H", 0.0060) or 0.0060),
        )
        self.edge_usd_drift_down_24h = max(
            0.0,
            float(getattr(config, "V2_CALIBRATION_EDGE_USD_DRIFT_DOWN_24H", 0.0060) or 0.0060),
        )
        self.reanchor_interval_seconds = max(
            1800,
            int(getattr(config, "V2_CALIBRATION_REANCHOR_INTERVAL_SECONDS", 14400) or 14400),
        )
        self.reanchor_blend = _clamp(float(getattr(config, "V2_CALIBRATION_REANCHOR_BLEND", 0.22) or 0.22), 0.0, 1.0)
        self.volume_floor_min = max(0.0, float(getattr(config, "V2_CALIBRATION_VOLUME_MIN", 20.0) or 20.0))
        self.volume_floor_max = max(self.volume_floor_min, float(getattr(config, "V2_CALIBRATION_VOLUME_MAX", 450.0) or 450.0))
        self.ev_min_net_min = max(0.0, float(getattr(config, "V2_CALIBRATION_EV_MIN_NET_MIN", 0.0004) or 0.0004))
        self.ev_min_net_max = max(
            self.ev_min_net_min,
            float(getattr(config, "V2_CALIBRATION_EV_MIN_NET_MAX", 0.0038) or 0.0038),
        )
        self.runner_min_ev_min = max(
            0.0,
            float(getattr(config, "V2_CALIBRATION_RUNNER_MIN_EV_MIN", 0.0030) or 0.0030),
        )
        self.runner_min_ev_max = max(
            self.runner_min_ev_min,
            float(getattr(config, "V2_CALIBRATION_RUNNER_MIN_EV_MAX", 0.0140) or 0.0140),
        )
        self.source_bad_entry_prob_min = _clamp(
            float(getattr(config, "V2_CALIBRATION_SOURCE_BAD_ENTRY_PROB_MIN", 0.45) or 0.45),
            0.0,
            1.0,
        )
        self.source_bad_entry_prob_max = _clamp(
            float(getattr(config, "V2_CALIBRATION_SOURCE_BAD_ENTRY_PROB_MAX", 0.86) or 0.86),
            self.source_bad_entry_prob_min,
            1.0,
        )
        self.source_severe_entry_prob_min = _clamp(
            float(getattr(config, "V2_CALIBRATION_SOURCE_SEVERE_ENTRY_PROB_MIN", 0.25) or 0.25),
            0.0,
            1.0,
        )
        self.source_severe_entry_prob_max = _clamp(
            float(getattr(config, "V2_CALIBRATION_SOURCE_SEVERE_ENTRY_PROB_MAX", 0.68) or 0.68),
            self.source_severe_entry_prob_min,
            1.0,
        )
        self._next_eval_ts = time.time() + max(30, int(self.apply_interval_seconds))
        self.run_tag = str(getattr(config, "RUN_TAG", os.getenv("RUN_TAG", "")) or "").strip()
        self.scope_by_run_tag = bool(getattr(config, "V2_CALIBRATION_SCOPE_BY_RUN_TAG", True))
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        db_default = os.path.join(root, "data", "unified_dataset", "unified.db")
        self.db_path = str(getattr(config, "V2_CALIBRATION_DB_PATH", db_default) or db_default)
        out_default = os.path.join(root, "data", "analysis", "v2_calibration_latest.json")
        self.out_json_path = str(getattr(config, "V2_CALIBRATION_OUTPUT_JSON", out_default) or out_default)
        self._last_applied_signature = ""
        self._baseline_edge_percent = _safe_float(getattr(config, "MIN_EXPECTED_EDGE_PERCENT", 1.0), 1.0)
        self._baseline_edge_usd = _safe_float(getattr(config, "MIN_EXPECTED_EDGE_USD", 0.02), 0.02)
        self._baseline_volume_floor = _safe_float(getattr(config, "SAFE_MIN_VOLUME_5M_USD", 0.0), 0.0)
        self._baseline_no_momentum = _safe_float(getattr(config, "NO_MOMENTUM_EXIT_MAX_PNL_PERCENT", 0.25), 0.25)
        self._baseline_weakness = _safe_float(getattr(config, "WEAKNESS_EXIT_PNL_PERCENT", -2.6), -2.6)
        self._baseline_hold_max = float(max(90, int(getattr(config, "HOLD_MAX_SECONDS", 150) or 150)))
        self._baseline_ev_min_net = _safe_float(getattr(config, "EV_FIRST_ENTRY_MIN_NET_USD", 0.0010), 0.0010)
        self._baseline_runner_min_ev = _safe_float(getattr(config, "EXECUTION_RUNNER_MIN_EV_USD", 0.0055), 0.0055)
        self._baseline_source_bad_prob = _safe_float(
            getattr(config, "V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY", 0.72),
            0.72,
        )
        self._baseline_source_severe_prob = _safe_float(
            getattr(config, "V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY", 0.46),
            0.46,
        )
        self.clamp_enabled = bool(getattr(config, "V2_CALIBRATION_CLAMP_ENABLED", True))
        self.clamp_edge_percent_max_from_base = max(
            0.0,
            _safe_float(getattr(config, "V2_CALIBRATION_EDGE_PERCENT_MAX_FROM_BASE", 0.08), 0.08),
        )
        self.clamp_edge_percent_min_from_base = max(
            0.0,
            _safe_float(getattr(config, "V2_CALIBRATION_EDGE_PERCENT_MIN_FROM_BASE", 0.10), 0.10),
        )
        self.clamp_edge_usd_max_from_base = max(
            0.0,
            _safe_float(getattr(config, "V2_CALIBRATION_EDGE_USD_MAX_FROM_BASE", 0.0020), 0.0020),
        )
        self.clamp_edge_usd_min_from_base = max(
            0.0,
            _safe_float(getattr(config, "V2_CALIBRATION_EDGE_USD_MIN_FROM_BASE", 0.0030), 0.0030),
        )
        self.clamp_safe_volume_max_from_base = max(
            0.0,
            _safe_float(getattr(config, "V2_CALIBRATION_SAFE_VOLUME_MAX_FROM_BASE", 60.0), 60.0),
        )
        self.clamp_no_momentum_min = _safe_float(getattr(config, "V2_CALIBRATION_NO_MOMENTUM_MIN", 0.08), 0.08)
        self.clamp_weakness_max_neg_drift = max(
            0.0,
            _safe_float(getattr(config, "V2_CALIBRATION_WEAKNESS_MAX_NEG_DRIFT", 0.35), 0.35),
        )
        self.clamp_hold_max_abs = float(max(90, int(getattr(config, "V2_CALIBRATION_HOLD_MAX_ABS", 280) or 280)))
        self._anchor_edge_percent = float(self._baseline_edge_percent)
        self._anchor_edge_usd = float(self._baseline_edge_usd)
        self._anchor_ts = time.time()

    def maybe_apply(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        now_ts = time.time()
        if now_ts < float(self._next_eval_ts):
            return None
        self._next_eval_ts = now_ts + max(120, int(self.apply_interval_seconds))
        if not os.path.exists(self.db_path):
            logger.warning("V2_CALIBRATION db not found path=%s", self.db_path)
            return None
        self._maybe_reanchor(now_ts)

        payload = self._build_payload()
        if not payload:
            return None
        signature = json.dumps(payload.get("applied", {}), ensure_ascii=False, sort_keys=True)
        if signature == self._last_applied_signature:
            return None
        self._last_applied_signature = signature

        applied = payload.get("applied", {})
        self._apply_config_overrides(applied)
        self._write_payload(payload)
        logger.warning(
            (
                "V2_CALIBRATION applied scope=%s:%s closed=%s edge_usd=%.4f volume_floor=%.0f hold_max=%s "
                "no_momentum=%.2f weakness=%.2f ev_min=$%.4f runner_min_ev=$%.4f src_bad_p=%.2f src_severe_p=%.2f"
            ),
            str(payload.get("scope_field", "") or "global"),
            str(payload.get("scope_run_tag", "") or "-"),
            int(payload.get("rows_closed", 0) or 0),
            float(applied.get("MIN_EXPECTED_EDGE_USD", 0.0) or 0.0),
            float(applied.get("SAFE_MIN_VOLUME_5M_USD", 0.0) or 0.0),
            int(applied.get("HOLD_MAX_SECONDS", 0) or 0),
            float(applied.get("NO_MOMENTUM_EXIT_MAX_PNL_PERCENT", 0.0) or 0.0),
            float(applied.get("WEAKNESS_EXIT_PNL_PERCENT", 0.0) or 0.0),
            float(applied.get("EV_FIRST_ENTRY_MIN_NET_USD", 0.0) or 0.0),
            float(applied.get("EXECUTION_RUNNER_MIN_EV_USD", 0.0) or 0.0),
            float(applied.get("V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY", 0.0) or 0.0),
            float(applied.get("V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY", 0.0) or 0.0),
        )
        return payload

    def _build_payload(self) -> dict[str, Any] | None:
        conn = sqlite3.connect(self.db_path)
        try:
            closed_cols = {
                str(row[1] or "").strip().lower()
                for row in conn.execute("PRAGMA table_info(closed_trades)").fetchall()
            }
            scope_clause = ""
            scope_params: list[Any] = []
            scope_field = ""
            if bool(self.scope_by_run_tag) and bool(self.run_tag):
                if "run_tag" in closed_cols:
                    hit = conn.execute(
                        "SELECT 1 FROM closed_trades WHERE run_tag = ? LIMIT 1",
                        (str(self.run_tag),),
                    ).fetchone()
                    if hit:
                        scope_clause = "WHERE run_tag = ?"
                        scope_params = [str(self.run_tag)]
                        scope_field = "run_tag"
                if (not scope_clause) and ("profile" in closed_cols):
                    hit = conn.execute(
                        "SELECT 1 FROM closed_trades WHERE profile = ? LIMIT 1",
                        (str(self.run_tag),),
                    ).fetchone()
                    if hit:
                        scope_clause = "WHERE profile = ?"
                        scope_params = [str(self.run_tag)]
                        scope_field = "profile"
                if (not scope_clause) and ("run_tag" in closed_cols):
                    scope_clause = "WHERE run_tag = ?"
                    scope_params = [str(self.run_tag)]
                    scope_field = "run_tag"

            closed_rows = conn.execute(
                f"""
                SELECT pnl_usd, pnl_percent, position_size_usd, close_reason, max_hold_seconds
                FROM closed_trades
                {scope_clause}
                ORDER BY rowid DESC
                LIMIT ?
                """,
                tuple(list(scope_params) + [int(self.lookback_rows)]),
            ).fetchall()
            if not closed_rows:
                return None
            pnl_usd = [_safe_float(row[0], 0.0) for row in closed_rows]
            if len(pnl_usd) < int(self.min_closed):
                return None
            pnl_pct = [_safe_float(row[1], 0.0) for row in closed_rows]
            pos_sizes = [max(0.01, _safe_float(row[2], 0.0)) for row in closed_rows]
            reasons = [str(row[3] or "") for row in closed_rows]
            hold_values = [max(30.0, _safe_float(row[4], 0.0)) for row in closed_rows if _safe_float(row[4], 0.0) > 0.0]

            wins = [x for x in pnl_usd if x > 0]
            losses = [abs(x) for x in pnl_usd if x <= 0]
            win_rate = float(len(wins)) / float(max(1, len(pnl_usd)))
            p50_win = float(_percentile(wins, 50.0)) if wins else 0.0
            p65_win = float(_percentile(wins, 65.0)) if wins else p50_win
            p50_loss = float(_percentile(losses, 50.0)) if losses else 0.0
            size_median = float(_percentile(pos_sizes, 50.0)) if pos_sizes else 0.5
            ev_proxy = float((win_rate * p50_win) - ((1.0 - win_rate) * p50_loss))

            edge_from_wins = max(self.edge_usd_min, p65_win * 0.45)
            if ev_proxy < 0:
                edge_from_wins *= 1.15
            edge_target = _clamp(edge_from_wins, self.edge_usd_min, self.edge_usd_max)
            edge_pct_target = _clamp(
                (edge_target / max(0.1, size_median)) * 100.0,
                self.edge_percent_min,
                self.edge_percent_max,
            )

            # Winner volume via candidates.raw_json (portable across schema migrations).
            winner_scope_clause = ""
            winner_scope_params: list[Any] = []
            if scope_clause and scope_field:
                winner_scope_clause = f" AND {scope_field} = ?"
                winner_scope_params = [str(self.run_tag)]
            winner_ids = [
                str(row[0] or "")
                for row in conn.execute(
                    f"""
                    SELECT candidate_id
                    FROM closed_trades
                    WHERE pnl_usd > 0
                      AND candidate_id != ''
                      {winner_scope_clause}
                    ORDER BY rowid DESC
                    LIMIT ?
                    """,
                    tuple(list(winner_scope_params) + [int(self.lookback_rows)]),
                ).fetchall()
            ]
            winner_vol: list[float] = []
            if winner_ids:
                batch_size = 250
                for start in range(0, len(winner_ids), batch_size):
                    chunk = [cid for cid in winner_ids[start : start + batch_size] if cid]
                    if not chunk:
                        continue
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"""
                        SELECT raw_json
                        FROM candidates
                        WHERE decision_stage = 'post_filters'
                          AND candidate_id IN ({placeholders})
                        """,
                        tuple(chunk),
                    ).fetchall()
                    for raw_row in rows:
                        raw_json = str(raw_row[0] or "")
                        if not raw_json:
                            continue
                        try:
                            payload = json.loads(raw_json)
                        except Exception:
                            continue
                        vol = _safe_float(payload.get("volume_5m_usd", payload.get("volume_5m")), 0.0)
                        if vol > 0:
                            winner_vol.append(vol)
            if winner_vol:
                vol_target = _percentile(winner_vol, 30.0)
            else:
                vol_target = float(getattr(config, "SAFE_MIN_VOLUME_5M_USD", 0.0) or 0.0)
            vol_target = _clamp(vol_target, self.volume_floor_min, self.volume_floor_max)

            no_momentum_vals = [
                _safe_float(pct, 0.0)
                for pct, reason in zip(pnl_pct, reasons)
                if str(reason).upper().startswith("NO_MOMENTUM")
            ]
            weakness_vals = [
                _safe_float(pct, 0.0)
                for pct, reason in zip(pnl_pct, reasons)
                if str(reason).upper().startswith("WEAKNESS")
            ]
            nm_target = _clamp(
                (_percentile(no_momentum_vals, 70.0) if no_momentum_vals else float(getattr(config, "NO_MOMENTUM_EXIT_MAX_PNL_PERCENT", 0.25))),
                -0.2,
                0.8,
            )
            nm_floor = _safe_float(getattr(config, "V2_CALIBRATION_NO_MOMENTUM_MAX_PNL_FLOOR", 0.02), 0.02)
            nm_target = max(float(nm_floor), float(nm_target))
            weak_target = _clamp(
                (_percentile(weakness_vals, 55.0) if weakness_vals else float(getattr(config, "WEAKNESS_EXIT_PNL_PERCENT", -2.6))),
                -6.5,
                -0.9,
            )
            hold_target = int(
                _clamp(
                    round(_percentile(hold_values, 55.0) if hold_values else float(getattr(config, "HOLD_MAX_SECONDS", 150) or 150)),
                    90,
                    360,
                )
            )
            reason_u = [str(r or "").upper() for r in reasons]
            timeout_count = sum(1 for r in reason_u if r.startswith("TIMEOUT"))
            no_momentum_count = sum(1 for r in reason_u if r.startswith("NO_MOMENTUM"))
            tp_count = sum(1 for r in reason_u if r.startswith("TP"))
            sl_count = sum(1 for r in reason_u if r.startswith("SL"))
            total_closed = max(1, len(reason_u))
            timeout_nm_share = float(timeout_count + no_momentum_count) / float(total_closed)
            tp_share = float(tp_count) / float(total_closed)
            sl_share = float(sl_count) / float(total_closed)
            loss_share_total = float(len(losses)) / float(max(1, len(pnl_usd)))

            ev_min_net_target = (
                (p50_win * 0.14)
                + (max(0.0, ev_proxy) * 0.35)
                + (p50_loss * 0.02)
            )
            if timeout_nm_share >= 0.45 and ev_proxy < 0.0:
                ev_min_net_target *= 1.08
            if tp_share >= 0.14 and ev_proxy > 0.0:
                ev_min_net_target *= 0.96
            ev_min_net_target = _clamp(
                ev_min_net_target,
                float(self.ev_min_net_min),
                float(self.ev_min_net_max),
            )

            runner_min_ev_target = max(
                ev_min_net_target * 2.20,
                p65_win * 0.30,
                p50_win * 0.45,
            )
            if timeout_nm_share >= 0.50 and tp_share <= 0.10:
                runner_min_ev_target *= 1.06
            if tp_share >= 0.16 and loss_share_total <= 0.52:
                runner_min_ev_target *= 0.96
            runner_min_ev_target = _clamp(
                runner_min_ev_target,
                float(self.runner_min_ev_min),
                float(self.runner_min_ev_max),
            )

            source_penalty = (
                (max(0.0, loss_share_total - 0.50) * 0.60)
                + (max(0.0, timeout_nm_share - 0.40) * 0.45)
                - (max(0.0, tp_share - 0.12) * 0.35)
            )
            source_penalty = _clamp(source_penalty, -0.18, 0.25)
            source_bad_prob_target = _clamp(
                0.74 - source_penalty,
                float(self.source_bad_entry_prob_min),
                float(self.source_bad_entry_prob_max),
            )
            source_severe_prob_target = _clamp(
                source_bad_prob_target - 0.22,
                float(self.source_severe_entry_prob_min),
                float(self.source_severe_entry_prob_max),
            )

            edge_usd_next = self._bounded_edge_usd_target(edge_target)
            edge_pct_next = self._bounded_edge_percent_target(edge_pct_target)
            relax_until_ts = _safe_float(getattr(config, "V2_RUNTIME_EDGE_RELAX_UNTIL_TS", 0.0), 0.0)
            if bool(getattr(config, "V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW", True)) and (time.time() < float(relax_until_ts)):
                edge_usd_now = _safe_float(getattr(config, "MIN_EXPECTED_EDGE_USD", edge_usd_next), edge_usd_next)
                edge_pct_now = _safe_float(getattr(config, "MIN_EXPECTED_EDGE_PERCENT", edge_pct_next), edge_pct_next)
                edge_usd_next = min(float(edge_usd_next), float(edge_usd_now))
                edge_pct_next = min(float(edge_pct_next), float(edge_pct_now))

            applied = {
                "MIN_EXPECTED_EDGE_USD": float(edge_usd_next),
                "MIN_EXPECTED_EDGE_PERCENT": float(edge_pct_next),
                "SAFE_MIN_VOLUME_5M_USD": self._smooth("SAFE_MIN_VOLUME_5M_USD", vol_target),
                "NO_MOMENTUM_EXIT_MAX_PNL_PERCENT": self._smooth("NO_MOMENTUM_EXIT_MAX_PNL_PERCENT", nm_target),
                "WEAKNESS_EXIT_PNL_PERCENT": self._smooth("WEAKNESS_EXIT_PNL_PERCENT", weak_target),
                "HOLD_MAX_SECONDS": int(round(self._smooth("HOLD_MAX_SECONDS", float(hold_target)))),
                "EV_FIRST_ENTRY_MIN_NET_USD": self._smooth("EV_FIRST_ENTRY_MIN_NET_USD", ev_min_net_target),
                "EXECUTION_RUNNER_MIN_EV_USD": self._smooth("EXECUTION_RUNNER_MIN_EV_USD", runner_min_ev_target),
                "V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY": self._smooth(
                    "V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY",
                    source_bad_prob_target,
                ),
                "V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY": self._smooth(
                    "V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY",
                    source_severe_prob_target,
                ),
            }
            applied = self._apply_runtime_clamp(applied)
            return {
                "ts": int(time.time()),
                "db_path": self.db_path,
                "rows_closed": len(pnl_usd),
                "win_rate": round(win_rate, 6),
                "p50_win_usd": round(p50_win, 6),
                "p50_loss_usd": round(p50_loss, 6),
                "ev_proxy_usd": round(ev_proxy, 6),
                "timeout_nm_share": round(timeout_nm_share, 6),
                "tp_share": round(tp_share, 6),
                "sl_share": round(sl_share, 6),
                "loss_share": round(loss_share_total, 6),
                "scope_applied": bool(scope_clause),
                "scope_field": scope_field,
                "scope_run_tag": str(self.run_tag) if scope_clause else "",
                "anchor_edge_percent": round(self._anchor_edge_percent, 6),
                "anchor_edge_usd": round(self._anchor_edge_usd, 6),
                "applied": applied,
            }
        finally:
            conn.close()

    def _maybe_reanchor(self, now_ts: float) -> None:
        if (now_ts - float(self._anchor_ts)) < float(self.reanchor_interval_seconds):
            return
        edge_pct_now = _safe_float(getattr(config, "MIN_EXPECTED_EDGE_PERCENT", self._anchor_edge_percent), self._anchor_edge_percent)
        edge_usd_now = _safe_float(getattr(config, "MIN_EXPECTED_EDGE_USD", self._anchor_edge_usd), self._anchor_edge_usd)
        blend = float(self.reanchor_blend)
        # Slow recentering toward baseline prevents long-run drift lock.
        self._anchor_edge_percent = _clamp(
            (edge_pct_now * (1.0 - blend)) + (float(self._baseline_edge_percent) * blend),
            self.edge_percent_min,
            self.edge_percent_max,
        )
        self._anchor_edge_usd = _clamp(
            (edge_usd_now * (1.0 - blend)) + (float(self._baseline_edge_usd) * blend),
            self.edge_usd_min,
            self.edge_usd_max,
        )
        self._anchor_ts = float(now_ts)

    def _bounded_edge_percent_target(self, target: float) -> float:
        now_ts = time.time()
        current = _safe_float(getattr(config, "MIN_EXPECTED_EDGE_PERCENT", target), target)
        smoothed = self._smooth("MIN_EXPECTED_EDGE_PERCENT", target)
        if self.edge_percent_step_max > 0.0:
            smoothed = _clamp(smoothed, current - self.edge_percent_step_max, current + self.edge_percent_step_max)
        drift_low = float(self._anchor_edge_percent) - float(self.edge_percent_drift_down_24h)
        drift_high = float(self._anchor_edge_percent) + float(self.edge_percent_drift_up_24h)
        out = _clamp(smoothed, max(self.edge_percent_min, drift_low), min(self.edge_percent_max, drift_high))
        return _clamp(out, self.edge_percent_min, self.edge_percent_max)

    def _bounded_edge_usd_target(self, target: float) -> float:
        current = _safe_float(getattr(config, "MIN_EXPECTED_EDGE_USD", target), target)
        smoothed = self._smooth("MIN_EXPECTED_EDGE_USD", target)
        if self.edge_usd_step_max > 0.0:
            smoothed = _clamp(smoothed, current - self.edge_usd_step_max, current + self.edge_usd_step_max)
        drift_low = float(self._anchor_edge_usd) - float(self.edge_usd_drift_down_24h)
        drift_high = float(self._anchor_edge_usd) + float(self.edge_usd_drift_up_24h)
        out = _clamp(smoothed, max(self.edge_usd_min, drift_low), min(self.edge_usd_max, drift_high))
        return _clamp(out, self.edge_usd_min, self.edge_usd_max)

    def _smooth(self, field: str, target: float) -> float:
        old = _safe_float(getattr(config, field, target), target)
        new = (old * (1.0 - float(self.smooth_alpha))) + (float(target) * float(self.smooth_alpha))
        return float(new)

    def _apply_config_overrides(self, applied: dict[str, Any]) -> None:
        for key, value in (applied or {}).items():
            try:
                setattr(config, str(key), value)
            except Exception:
                continue

    def _apply_runtime_clamp(self, applied: dict[str, Any]) -> dict[str, Any]:
        out = dict(applied or {})
        if not bool(self.clamp_enabled):
            return out

        base_edge_pct = float(self._baseline_edge_percent)
        base_edge_usd = float(self._baseline_edge_usd)
        edge_pct_lo = max(float(self.edge_percent_min), base_edge_pct - float(self.clamp_edge_percent_min_from_base))
        edge_pct_hi = min(float(self.edge_percent_max), base_edge_pct + float(self.clamp_edge_percent_max_from_base))
        edge_usd_lo = max(float(self.edge_usd_min), base_edge_usd - float(self.clamp_edge_usd_min_from_base))
        edge_usd_hi = min(float(self.edge_usd_max), base_edge_usd + float(self.clamp_edge_usd_max_from_base))

        out["MIN_EXPECTED_EDGE_PERCENT"] = _clamp(
            _safe_float(out.get("MIN_EXPECTED_EDGE_PERCENT"), base_edge_pct),
            edge_pct_lo,
            edge_pct_hi,
        )
        out["MIN_EXPECTED_EDGE_USD"] = _clamp(
            _safe_float(out.get("MIN_EXPECTED_EDGE_USD"), base_edge_usd),
            edge_usd_lo,
            edge_usd_hi,
        )

        volume_cap = float(self._baseline_volume_floor) + float(self.clamp_safe_volume_max_from_base)
        volume_cap = max(float(self.volume_floor_min), min(float(self.volume_floor_max), volume_cap))
        out["SAFE_MIN_VOLUME_5M_USD"] = _clamp(
            _safe_float(out.get("SAFE_MIN_VOLUME_5M_USD"), self._baseline_volume_floor),
            float(self.volume_floor_min),
            volume_cap,
        )

        out["NO_MOMENTUM_EXIT_MAX_PNL_PERCENT"] = max(
            float(self.clamp_no_momentum_min),
            _safe_float(out.get("NO_MOMENTUM_EXIT_MAX_PNL_PERCENT"), self._baseline_no_momentum),
        )

        weakness_floor = float(self._baseline_weakness) - float(self.clamp_weakness_max_neg_drift)
        out["WEAKNESS_EXIT_PNL_PERCENT"] = _clamp(
            _safe_float(out.get("WEAKNESS_EXIT_PNL_PERCENT"), self._baseline_weakness),
            weakness_floor,
            -0.20,
        )

        hold_now = float(_safe_float(out.get("HOLD_MAX_SECONDS"), self._baseline_hold_max))
        out["HOLD_MAX_SECONDS"] = int(round(_clamp(hold_now, 90.0, float(self.clamp_hold_max_abs))))

        out["EV_FIRST_ENTRY_MIN_NET_USD"] = _clamp(
            _safe_float(out.get("EV_FIRST_ENTRY_MIN_NET_USD"), self._baseline_ev_min_net),
            float(self.ev_min_net_min),
            float(self.ev_min_net_max),
        )
        out["EXECUTION_RUNNER_MIN_EV_USD"] = _clamp(
            _safe_float(out.get("EXECUTION_RUNNER_MIN_EV_USD"), self._baseline_runner_min_ev),
            float(self.runner_min_ev_min),
            float(self.runner_min_ev_max),
        )
        out["V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY"] = _clamp(
            _safe_float(out.get("V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY"), self._baseline_source_bad_prob),
            float(self.source_bad_entry_prob_min),
            float(self.source_bad_entry_prob_max),
        )
        out["V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY"] = _clamp(
            _safe_float(
                out.get("V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY"),
                self._baseline_source_severe_prob,
            ),
            float(self.source_severe_entry_prob_min),
            float(self.source_severe_entry_prob_max),
        )
        if float(out["V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY"]) > float(out["V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY"]):
            out["V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY"] = float(
                max(
                    self.source_severe_entry_prob_min,
                    float(out["V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY"]) - 0.05,
                )
            )
        return out

    def _write_payload(self, payload: dict[str, Any]) -> None:
        try:
            path = self.out_json_path
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception:
            logger.exception("V2_CALIBRATION write failed path=%s", self.out_json_path)


class PolicyEntryRouter:
    """Decouple data policy health from hard global BUY shutdown."""

    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "V2_POLICY_ROUTER_ENABLED", False))
        self.fail_closed_action = str(getattr(config, "V2_POLICY_FAIL_CLOSED_ACTION", "limited") or "limited").strip().lower()
        self.degraded_action = str(getattr(config, "V2_POLICY_DEGRADED_ACTION", "limited") or "limited").strip().lower()
        self.limited_entry_ratio = _clamp(
            float(getattr(config, "V2_POLICY_LIMITED_ENTRY_RATIO", 0.45) or 0.45),
            0.05,
            1.0,
        )
        self.limited_min_per_cycle = max(1, int(getattr(config, "V2_POLICY_LIMITED_MIN_PER_CYCLE", 2) or 2))
        self.limited_only_strict = bool(getattr(config, "V2_POLICY_LIMITED_ONLY_STRICT", True))
        self.allow_explore_in_red = bool(getattr(config, "V2_POLICY_LIMITED_ALLOW_EXPLORE_IN_RED", False))
        self.hard_block_enabled = bool(getattr(config, "DATA_POLICY_HARD_BLOCK_ENABLED", False))

    @staticmethod
    def _sort_key(candidate: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, float, float]:
        token, score_data = candidate
        score = _safe_float(score_data.get("score"), 0.0)
        liq = _safe_float(token.get("liquidity"), 0.0)
        vol = _safe_float(token.get("volume_5m"), 0.0)
        return (score, liq, vol)

    @staticmethod
    def _is_strict(candidate: tuple[dict[str, Any], dict[str, Any]]) -> bool:
        token, _ = candidate
        return str(token.get("_entry_tier", "") or "").strip().upper() == "A"

    def route(
        self,
        *,
        candidates: list[tuple[dict[str, Any], dict[str, Any]]],
        policy_state: str,
        policy_reason: str,
        market_mode: str,
    ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
        mode = str(policy_state or "UNKNOWN").strip().upper() or "UNKNOWN"
        regime = str(market_mode or "YELLOW").strip().upper() or "YELLOW"
        total = len(candidates)
        if total <= 0:
            return [], {
                "enabled": bool(self.enabled),
                "policy_state": mode,
                "effective_mode": "OK" if mode == "OK" else "BLOCKED",
                "action": "empty",
                "in_total": 0,
                "out_total": 0,
                "reason": str(policy_reason or ""),
            }

        if not self.enabled:
            if mode == "OK":
                return candidates, {
                    "enabled": False,
                    "policy_state": mode,
                    "effective_mode": "OK",
                    "action": "passthrough",
                    "in_total": total,
                    "out_total": total,
                    "reason": str(policy_reason or ""),
                }
            return [], {
                "enabled": False,
                "policy_state": mode,
                "effective_mode": "BLOCKED",
                "action": "legacy_block",
                "in_total": total,
                "out_total": 0,
                "reason": str(policy_reason or ""),
            }

        if self.hard_block_enabled and mode != "OK":
            return [], {
                "enabled": True,
                "policy_state": mode,
                "effective_mode": "BLOCKED",
                "action": "hard_block",
                "in_total": total,
                "out_total": 0,
                "reason": str(policy_reason or ""),
            }

        if mode == "OK":
            return candidates, {
                "enabled": True,
                "policy_state": mode,
                "effective_mode": "OK",
                "action": "allow_all",
                "in_total": total,
                "out_total": total,
                "reason": str(policy_reason or ""),
            }

        action = "block"
        if mode == "FAIL_CLOSED":
            action = self.fail_closed_action if self.fail_closed_action in {"limited", "block"} else "block"
        elif mode == "DEGRADED":
            action = self.degraded_action if self.degraded_action in {"limited", "block"} else "block"
        if action != "limited":
            return [], {
                "enabled": True,
                "policy_state": mode,
                "effective_mode": "BLOCKED",
                "action": "policy_block",
                "in_total": total,
                "out_total": 0,
                "reason": str(policy_reason or ""),
            }

        sorted_rows = sorted(candidates, key=self._sort_key, reverse=True)
        strict_rows = [row for row in sorted_rows if self._is_strict(row)]
        soft_rows = [row for row in sorted_rows if not self._is_strict(row)]
        allowed_total = max(
            int(self.limited_min_per_cycle),
            int(math.ceil(float(total) * float(self.limited_entry_ratio))),
        )
        allowed_total = max(1, min(total, allowed_total))

        selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
        selected.extend(strict_rows[:allowed_total])
        soft_allowed = not self.limited_only_strict
        if regime == "RED" and not self.allow_explore_in_red:
            soft_allowed = False
        if soft_allowed and len(selected) < allowed_total:
            selected.extend(soft_rows[: max(0, allowed_total - len(selected))])

        return selected, {
            "enabled": True,
            "policy_state": mode,
            "effective_mode": "LIMITED",
            "action": "limited",
            "in_total": total,
            "out_total": len(selected),
            "strict_in": len(strict_rows),
            "soft_in": len(soft_rows),
            "strict_out": sum(1 for row in selected if self._is_strict(row)),
            "soft_out": sum(1 for row in selected if not self._is_strict(row)),
            "ratio": float(self.limited_entry_ratio),
            "reason": str(policy_reason or ""),
            "regime": regime,
        }


class DualEntryController:
    """Split entry flow into core/explore channels with explicit quotas."""

    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "V2_ENTRY_DUAL_CHANNEL_ENABLED", False))
        self.explore_max_share = _clamp(
            float(getattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE", 0.35) or 0.35),
            0.0,
            1.0,
        )
        self.explore_max_per_cycle = max(0, int(getattr(config, "V2_ENTRY_EXPLORE_MAX_PER_CYCLE", 3) or 3))
        self.explore_allow_in_red = bool(getattr(config, "V2_ENTRY_EXPLORE_ALLOW_IN_RED", False))
        self.core_min_per_cycle = max(0, int(getattr(config, "V2_ENTRY_CORE_MIN_PER_CYCLE", 1) or 1))
        self.explore_size_mult = _clamp(float(getattr(config, "V2_ENTRY_EXPLORE_SIZE_MULT", 0.45) or 0.45), 0.1, 1.2)
        self.explore_hold_mult = _clamp(float(getattr(config, "V2_ENTRY_EXPLORE_HOLD_MULT", 0.75) or 0.75), 0.2, 1.4)
        self.explore_edge_usd_mult = max(0.05, float(getattr(config, "V2_EXPLORE_EDGE_USD_MULT", 0.75) or 0.75))
        self.explore_edge_percent_mult = max(
            0.05,
            float(getattr(config, "V2_EXPLORE_EDGE_PERCENT_MULT", 0.80) or 0.80),
        )

    @staticmethod
    def _sort_key(candidate: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, float, float]:
        token, score_data = candidate
        score = _safe_float(score_data.get("score"), 0.0)
        liq = _safe_float(token.get("liquidity"), 0.0)
        vol = _safe_float(token.get("volume_5m"), 0.0)
        return (score, liq, vol)

    @staticmethod
    def _tag_candidate(
        row: tuple[dict[str, Any], dict[str, Any]],
        *,
        channel: str,
        size_mult: float,
        hold_mult: float,
        edge_usd_mult: float,
        edge_pct_mult: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        token, score_data = row
        token2 = dict(token or {})
        prev_size_mult = _safe_float(token2.get("_entry_channel_size_mult"), 1.0)
        prev_hold_mult = _safe_float(token2.get("_entry_channel_hold_mult"), 1.0)
        prev_edge_usd_mult = _safe_float(token2.get("_entry_channel_edge_usd_mult"), 1.0)
        prev_edge_pct_mult = _safe_float(token2.get("_entry_channel_edge_pct_mult"), 1.0)
        token2["_entry_channel"] = str(channel).strip().lower()
        token2["_entry_channel_size_mult"] = float(size_mult) * float(prev_size_mult)
        token2["_entry_channel_hold_mult"] = float(hold_mult) * float(prev_hold_mult)
        token2["_entry_channel_edge_usd_mult"] = float(edge_usd_mult) * float(prev_edge_usd_mult)
        token2["_entry_channel_edge_pct_mult"] = float(edge_pct_mult) * float(prev_edge_pct_mult)
        return token2, score_data

    def allocate(
        self,
        *,
        candidates: list[tuple[dict[str, Any], dict[str, Any]]],
        market_mode: str,
    ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
        regime = str(market_mode or "YELLOW").strip().upper() or "YELLOW"
        if not candidates:
            return [], {
                "enabled": bool(self.enabled),
                "regime": regime,
                "in_total": 0,
                "out_total": 0,
                "core_out": 0,
                "explore_out": 0,
            }

        sorted_rows = sorted(candidates, key=self._sort_key, reverse=True)
        strict_rows = [row for row in sorted_rows if str((row[0] or {}).get("_entry_tier", "")).strip().upper() == "A"]
        soft_rows = [row for row in sorted_rows if str((row[0] or {}).get("_entry_tier", "")).strip().upper() != "A"]

        if not self.enabled:
            tagged = [
                self._tag_candidate(
                    row,
                    channel="core",
                    size_mult=1.0,
                    hold_mult=1.0,
                    edge_usd_mult=1.0,
                    edge_pct_mult=1.0,
                )
                for row in sorted_rows
            ]
            return tagged, {
                "enabled": False,
                "regime": regime,
                "in_total": len(sorted_rows),
                "out_total": len(tagged),
                "core_out": len(tagged),
                "explore_out": 0,
            }

        allow_explore = regime != "RED" or bool(self.explore_allow_in_red)
        explore_quota = int(math.floor(float(len(sorted_rows)) * float(self.explore_max_share)))
        if self.explore_max_per_cycle > 0:
            explore_quota = min(explore_quota, int(self.explore_max_per_cycle))
        if allow_explore and explore_quota <= 0 and soft_rows and float(self.explore_max_share) > 0.0:
            explore_quota = 1
        explore_quota = max(0, min(len(soft_rows), explore_quota))

        core_selected = list(strict_rows)
        explore_selected = list(soft_rows[:explore_quota]) if allow_explore else []

        # In scarcity, keep at least `core_min_per_cycle` by promoting best soft rows to core.
        if self.core_min_per_cycle > 0 and len(core_selected) < int(self.core_min_per_cycle):
            needed = int(self.core_min_per_cycle) - len(core_selected)
            promoted = [row for row in soft_rows if row not in explore_selected][:needed]
            core_selected.extend(promoted)

        tagged: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in core_selected:
            tagged.append(
                self._tag_candidate(
                    row,
                    channel="core",
                    size_mult=1.0,
                    hold_mult=1.0,
                    edge_usd_mult=1.0,
                    edge_pct_mult=1.0,
                )
            )
        for row in explore_selected:
            if row in core_selected:
                continue
            tagged.append(
                self._tag_candidate(
                    row,
                    channel="explore",
                    size_mult=float(self.explore_size_mult),
                    hold_mult=float(self.explore_hold_mult),
                    edge_usd_mult=float(self.explore_edge_usd_mult),
                    edge_pct_mult=float(self.explore_edge_percent_mult),
                )
            )

        tagged.sort(key=self._sort_key, reverse=True)
        return tagged, {
            "enabled": True,
            "regime": regime,
            "in_total": len(sorted_rows),
            "out_total": len(tagged),
            "core_in": len(strict_rows),
            "soft_in": len(soft_rows),
            "core_out": sum(1 for row in tagged if str((row[0] or {}).get("_entry_channel", "")) == "core"),
            "explore_out": sum(1 for row in tagged if str((row[0] or {}).get("_entry_channel", "")) == "explore"),
            "explore_quota": int(explore_quota),
            "allow_explore": bool(allow_explore),
        }


class RollingEdgeGovernor:
    """Runtime edge-floor governor using recent outcomes + skip-pressure."""

    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "V2_ROLLING_EDGE_ENABLED", False))
        self.interval_seconds = max(60, int(getattr(config, "V2_ROLLING_EDGE_INTERVAL_SECONDS", 240) or 240))
        self.min_closed = max(10, int(getattr(config, "V2_ROLLING_EDGE_MIN_CLOSED", 24) or 24))
        self.window_closed = max(20, int(getattr(config, "V2_ROLLING_EDGE_WINDOW_CLOSED", 120) or 120))
        self.relax_step_usd = max(0.0, float(getattr(config, "V2_ROLLING_EDGE_RELAX_STEP_USD", 0.0020) or 0.0020))
        self.tighten_step_usd = max(0.0, float(getattr(config, "V2_ROLLING_EDGE_TIGHTEN_STEP_USD", 0.0025) or 0.0025))
        self.relax_step_percent = max(
            0.0,
            float(getattr(config, "V2_ROLLING_EDGE_RELAX_STEP_PERCENT", 0.08) or 0.08),
        )
        self.tighten_step_percent = max(
            0.0,
            float(getattr(config, "V2_ROLLING_EDGE_TIGHTEN_STEP_PERCENT", 0.10) or 0.10),
        )
        self.min_usd = max(0.0, float(getattr(config, "V2_ROLLING_EDGE_MIN_USD", 0.008) or 0.008))
        self.max_usd = max(self.min_usd, float(getattr(config, "V2_ROLLING_EDGE_MAX_USD", 0.120) or 0.120))
        self.min_percent = max(0.0, float(getattr(config, "V2_ROLLING_EDGE_MIN_PERCENT", 0.35) or 0.35))
        self.max_percent = max(
            self.min_percent,
            float(getattr(config, "V2_ROLLING_EDGE_MAX_PERCENT", 3.20) or 3.20),
        )
        self.edge_low_share_relax = _clamp(
            float(getattr(config, "V2_ROLLING_EDGE_EDGE_LOW_SHARE_RELAX", 0.65) or 0.65),
            0.0,
            1.0,
        )
        self.loss_share_tighten = _clamp(
            float(getattr(config, "V2_ROLLING_EDGE_LOSS_SHARE_TIGHTEN", 0.58) or 0.58),
            0.0,
            1.0,
        )
        self.anti_self_tighten_enabled = bool(getattr(config, "V2_ANTI_SELF_TIGHTEN_ENABLED", True))
        self.anti_self_tighten_open_rate_threshold = max(
            0.0,
            float(getattr(config, "V2_ANTI_SELF_TIGHTEN_OPEN_RATE_THRESHOLD", 0.035) or 0.035),
        )
        self.anti_self_tighten_edge_low_share_threshold = _clamp(
            float(getattr(config, "V2_ANTI_SELF_TIGHTEN_EDGE_LOW_SHARE_THRESHOLD", 0.62) or 0.62),
            0.0,
            1.0,
        )
        self.anti_self_tighten_cooldown_seconds = max(
            60,
            int(getattr(config, "V2_ANTI_SELF_TIGHTEN_COOLDOWN_SECONDS", 1800) or 1800),
        )
        self.max_tighten_per_hour = max(
            0,
            int(getattr(config, "V2_ROLLING_EDGE_MAX_TIGHTEN_PER_HOUR", 1) or 1),
        )
        self._next_eval_ts = time.time() + float(self.interval_seconds)
        self._skip_totals: dict[str, int] = defaultdict(int)
        self._window_candidates = 0
        self._window_opened = 0
        self._window_cycles = 0
        self._tighten_apply_ts: deque[float] = deque(maxlen=16)

    def record_cycle(self, *, candidates: int, opened: int, skip_reasons_cycle: dict[str, int]) -> None:
        self._window_cycles += 1
        self._window_candidates += int(candidates or 0)
        self._window_opened += int(opened or 0)
        for key, value in (skip_reasons_cycle or {}).items():
            k = str(key or "").strip().lower() or "unknown"
            self._skip_totals[k] = int(self._skip_totals.get(k, 0)) + int(value or 0)

    def _reset_window(self) -> None:
        self._skip_totals.clear()
        self._window_candidates = 0
        self._window_opened = 0
        self._window_cycles = 0

    def maybe_apply(self, *, auto_trader: Any, auto_stats: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        now_ts = time.time()
        if now_ts < float(self._next_eval_ts):
            return None
        self._next_eval_ts = now_ts + float(self.interval_seconds)

        closed_total = int(auto_stats.get("closed", 0) or 0)
        if closed_total < int(self.min_closed):
            self._reset_window()
            return None

        recent_closed = list(getattr(auto_trader, "closed_positions", []) or [])
        if not recent_closed:
            self._reset_window()
            return None
        recent = recent_closed[-int(self.window_closed) :]
        pnl_rows = [float(getattr(pos, "pnl_usd", 0.0) or 0.0) for pos in recent]
        if len(pnl_rows) < int(self.min_closed):
            self._reset_window()
            return None
        losses = sum(1 for p in pnl_rows if p < 0.0)
        loss_share = float(losses) / float(max(1, len(pnl_rows)))
        avg_pnl = float(sum(pnl_rows)) / float(max(1, len(pnl_rows)))

        skip_total = int(sum(int(v or 0) for v in self._skip_totals.values()))
        edge_low = (
            int(self._skip_totals.get("edge_usd_low", 0))
            + int(self._skip_totals.get("edge_low", 0))
            + int(self._skip_totals.get("negative_edge", 0))
            + int(self._skip_totals.get("ev_net_low", 0))
        )
        edge_low_share = (float(edge_low) / float(skip_total)) if skip_total > 0 else 0.0
        open_rate = (
            float(self._window_opened) / float(self._window_candidates)
            if int(self._window_candidates) > 0
            else 0.0
        )

        edge_usd_now = float(getattr(config, "MIN_EXPECTED_EDGE_USD", 0.0) or 0.0)
        edge_pct_now = float(getattr(config, "MIN_EXPECTED_EDGE_PERCENT", 0.0) or 0.0)
        edge_usd_next = edge_usd_now
        edge_pct_next = edge_pct_now
        action = "hold"
        reason = "steady"
        relax_until_ts = _safe_float(getattr(config, "V2_RUNTIME_EDGE_RELAX_UNTIL_TS", 0.0), 0.0)
        anti_stall_window = now_ts < float(relax_until_ts)
        low_flow_now = (
            open_rate <= float(self.anti_self_tighten_open_rate_threshold)
            or edge_low_share >= float(self.anti_self_tighten_edge_low_share_threshold)
        )
        if bool(self.anti_self_tighten_enabled) and low_flow_now:
            relax_until_next = now_ts + float(self.anti_self_tighten_cooldown_seconds)
            if relax_until_next > float(relax_until_ts):
                setattr(config, "V2_RUNTIME_EDGE_RELAX_UNTIL_TS", float(relax_until_next))
                relax_until_ts = float(relax_until_next)
                anti_stall_window = True
        while self._tighten_apply_ts and (now_ts - float(self._tighten_apply_ts[0])) > 3600.0:
            self._tighten_apply_ts.popleft()
        tighten_allowed = (self.max_tighten_per_hour <= 0) or (len(self._tighten_apply_ts) < int(self.max_tighten_per_hour))

        if (
            (loss_share >= float(self.loss_share_tighten) or avg_pnl < -0.0004)
            and (not anti_stall_window)
            and (not (self.anti_self_tighten_enabled and low_flow_now))
            and tighten_allowed
        ):
            edge_usd_next = _clamp(edge_usd_now + float(self.tighten_step_usd), self.min_usd, self.max_usd)
            edge_pct_next = _clamp(edge_pct_now + float(self.tighten_step_percent), self.min_percent, self.max_percent)
            action = "tighten"
            reason = f"loss_share={loss_share:.3f} avg_pnl={avg_pnl:.5f}"
        elif edge_low_share >= float(self.edge_low_share_relax) and open_rate < 0.08:
            edge_usd_next = _clamp(edge_usd_now - float(self.relax_step_usd), self.min_usd, self.max_usd)
            edge_pct_next = _clamp(edge_pct_now - float(self.relax_step_percent), self.min_percent, self.max_percent)
            action = "relax"
            reason = f"edge_low_share={edge_low_share:.3f} open_rate={open_rate:.3f}"
        elif anti_stall_window:
            action = "hold"
            reason = "anti_stall_recovery_window"

        changed = (abs(edge_usd_next - edge_usd_now) >= 0.0001) or (abs(edge_pct_next - edge_pct_now) >= 0.0001)
        payload = {
            "event_type": "V2_ROLLING_EDGE",
            "action": action,
            "changed": bool(changed),
            "reason": reason,
            "edge_usd_prev": round(edge_usd_now, 6),
            "edge_usd_next": round(edge_usd_next, 6),
            "edge_percent_prev": round(edge_pct_now, 6),
            "edge_percent_next": round(edge_pct_next, 6),
            "loss_share": round(loss_share, 6),
            "avg_pnl_usd": round(avg_pnl, 6),
            "edge_low_share": round(edge_low_share, 6),
            "open_rate": round(open_rate, 6),
            "closed_window": len(pnl_rows),
            "skip_total": int(skip_total),
        }
        if changed:
            setattr(config, "MIN_EXPECTED_EDGE_USD", float(edge_usd_next))
            setattr(config, "MIN_EXPECTED_EDGE_PERCENT", float(edge_pct_next))
            if action == "tighten":
                self._tighten_apply_ts.append(float(now_ts))
            logger.warning(
                "V2_ROLLING_EDGE action=%s reason=%s edge_usd=%.4f->%.4f edge_pct=%.2f->%.2f",
                action,
                reason,
                edge_usd_now,
                edge_usd_next,
                edge_pct_now,
                edge_pct_next,
            )
        self._reset_window()
        return payload if changed else None


class RuntimeKpiLoop:
    """Rolling KPI loop for throughput/diversity controls."""

    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "V2_KPI_LOOP_ENABLED", False))
        self.interval_seconds = max(60, int(getattr(config, "V2_KPI_LOOP_INTERVAL_SECONDS", 300) or 300))
        self.window_cycles = max(3, int(getattr(config, "V2_KPI_LOOP_WINDOW_CYCLES", 20) or 20))
        self.edge_low_relax_trigger = _clamp(
            float(getattr(config, "V2_KPI_EDGE_LOW_RELAX_TRIGGER", 0.70) or 0.70),
            0.0,
            1.0,
        )
        self.open_rate_low_trigger = max(0.0, float(getattr(config, "V2_KPI_OPEN_RATE_LOW_TRIGGER", 0.03) or 0.03))
        self.policy_block_trigger = _clamp(
            float(getattr(config, "V2_KPI_POLICY_BLOCK_TRIGGER", 0.35) or 0.35),
            0.0,
            1.0,
        )
        self.unique_symbols_min = max(1, int(getattr(config, "V2_KPI_UNIQUE_SYMBOLS_MIN", 6) or 6))
        self.max_buys_boost_step = max(0, int(getattr(config, "V2_KPI_MAX_BUYS_BOOST_STEP", 4) or 4))
        self.max_buys_cap = max(1, int(getattr(config, "V2_KPI_MAX_BUYS_CAP", 96) or 96))
        self.topn_boost_step = max(0, int(getattr(config, "V2_KPI_TOPN_BOOST_STEP", 1) or 1))
        self.topn_cap = max(1, int(getattr(config, "V2_KPI_TOPN_CAP", 24) or 24))
        self.explore_share_step = _clamp(
            float(getattr(config, "V2_KPI_EXPLORE_SHARE_STEP", 0.03) or 0.03),
            0.0,
            0.30,
        )
        self.explore_share_max = _clamp(
            float(getattr(config, "V2_KPI_EXPLORE_SHARE_MAX", 0.55) or 0.55),
            0.05,
            1.0,
        )
        self.novelty_share_step = _clamp(
            float(getattr(config, "V2_KPI_NOVELTY_SHARE_STEP", 0.03) or 0.03),
            0.0,
            0.30,
        )
        self.novelty_share_max = _clamp(
            float(getattr(config, "V2_KPI_NOVELTY_SHARE_MAX", 0.60) or 0.60),
            0.05,
            1.0,
        )
        self.quality_rebalance_enabled = bool(getattr(config, "V2_KPI_QUALITY_REBALANCE_ENABLED", True))
        self.edge_low_hard_trigger = _clamp(
            float(getattr(config, "V2_KPI_EDGE_LOW_HARD_TRIGGER", 0.80) or 0.80),
            0.0,
            1.0,
        )
        self.quality_rebalance_explore_step = _clamp(
            float(getattr(config, "V2_KPI_QUALITY_REBALANCE_EXPLORE_STEP", 0.04) or 0.04),
            0.0,
            0.30,
        )
        self.quality_rebalance_novelty_step = _clamp(
            float(getattr(config, "V2_KPI_QUALITY_REBALANCE_NOVELTY_STEP", 0.03) or 0.03),
            0.0,
            0.30,
        )
        self.quality_rebalance_topn_step = max(
            0,
            int(getattr(config, "V2_KPI_QUALITY_REBALANCE_TOPN_STEP", 1) or 1),
        )
        self.quality_rebalance_max_buys_step = max(
            0,
            int(getattr(config, "V2_KPI_QUALITY_REBALANCE_MAX_BUYS_STEP", 4) or 4),
        )
        self.explore_failsafe_enabled = bool(getattr(config, "V2_KPI_EXPLORE_FAILSAFE_ENABLED", True))
        self.explore_failsafe_window_closed = max(
            8,
            int(getattr(config, "V2_KPI_EXPLORE_FAILSAFE_WINDOW_CLOSED", 40) or 40),
        )
        self.explore_failsafe_min_closed = max(
            6,
            int(getattr(config, "V2_KPI_EXPLORE_FAILSAFE_MIN_CLOSED", 12) or 12),
        )
        self.explore_failsafe_loss_share_trigger = _clamp(
            float(getattr(config, "V2_KPI_EXPLORE_FAILSAFE_LOSS_SHARE_TRIGGER", 0.62) or 0.62),
            0.0,
            1.0,
        )
        self.explore_failsafe_avg_pnl_trigger = float(
            getattr(config, "V2_KPI_EXPLORE_FAILSAFE_AVG_PNL_TRIGGER_USD", -0.0015) or -0.0015
        )
        self.explore_failsafe_shrink_step = _clamp(
            float(getattr(config, "V2_KPI_EXPLORE_FAILSAFE_SHRINK_STEP", 0.06) or 0.06),
            0.0,
            0.30,
        )
        self.explore_failsafe_min_share = _clamp(
            float(getattr(config, "V2_KPI_EXPLORE_FAILSAFE_MIN_SHARE", 0.08) or 0.08),
            0.01,
            0.60,
        )
        self.explore_failsafe_recovery_winrate_min = _clamp(
            float(getattr(config, "V2_KPI_EXPLORE_FAILSAFE_RECOVERY_WINRATE_MIN", 0.54) or 0.54),
            0.0,
            1.0,
        )
        self.explore_failsafe_recovery_avg_pnl_min = float(
            getattr(config, "V2_KPI_EXPLORE_FAILSAFE_RECOVERY_AVG_PNL_MIN_USD", 0.0005) or 0.0005
        )
        self.explore_failsafe_recovery_step = _clamp(
            float(getattr(config, "V2_KPI_EXPLORE_FAILSAFE_RECOVERY_STEP", 0.03) or 0.03),
            0.0,
            0.30,
        )
        self.explore_failsafe_cooldown_seconds = max(
            0,
            int(getattr(config, "V2_KPI_EXPLORE_FAILSAFE_COOLDOWN_SECONDS", 900) or 900),
        )
        self._explore_share_baseline = _clamp(
            float(getattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE", 0.35) or 0.35),
            0.0,
            1.0,
        )
        self._explore_failsafe_until_ts = 0.0
        self.fast_antistall_enabled = bool(getattr(config, "V2_KPI_FAST_ANTISTALL_ENABLED", True))
        self.fast_antistall_interval_seconds = max(
            60,
            int(getattr(config, "V2_KPI_FAST_ANTISTALL_INTERVAL_SECONDS", 900) or 900),
        )
        self.fast_antistall_streak_trigger = max(
            1,
            int(getattr(config, "V2_KPI_FAST_ANTISTALL_STREAK_TRIGGER", 2) or 2),
        )
        self.fast_antistall_edge_low_share_trigger = _clamp(
            float(getattr(config, "V2_KPI_FAST_ANTISTALL_EDGE_LOW_SHARE_TRIGGER", 0.80) or 0.80),
            0.0,
            1.0,
        )
        self.fast_antistall_open_rate_trigger = max(
            0.0,
            float(getattr(config, "V2_KPI_FAST_ANTISTALL_OPEN_RATE_TRIGGER", 0.01) or 0.01),
        )
        self.fast_antistall_edge_percent_drop = max(
            0.0,
            float(getattr(config, "V2_KPI_FAST_ANTISTALL_EDGE_PERCENT_DROP", 0.12) or 0.12),
        )
        self.fast_antistall_edge_usd_drop = max(
            0.0,
            float(getattr(config, "V2_KPI_FAST_ANTISTALL_EDGE_USD_DROP", 0.0015) or 0.0015),
        )
        self.fast_antistall_recovery_window_seconds = max(
            120,
            int(getattr(config, "V2_KPI_FAST_ANTISTALL_RECOVERY_WINDOW_SECONDS", 1200) or 1200),
        )
        self.fast_antistall_max_apply_per_hour = max(
            1,
            int(getattr(config, "V2_KPI_FAST_ANTISTALL_MAX_APPLY_PER_HOUR", 2) or 2),
        )
        self._rows: deque[dict[str, Any]] = deque(maxlen=self.window_cycles)
        self._next_eval_ts = time.time() + float(self.interval_seconds)
        self._next_fast_eval_ts = time.time() + float(min(self.fast_antistall_interval_seconds, self.interval_seconds))
        self._low_flow_streak = 0
        self._fast_apply_ts: deque[float] = deque(maxlen=16)

    def record_cycle(
        self,
        *,
        candidates: int,
        opened: int,
        policy_state: str,
        symbols: list[str],
        skip_reasons_cycle: dict[str, int],
    ) -> None:
        sym_set = {str(x or "").strip().upper() for x in (symbols or []) if str(x or "").strip()}
        self._rows.append(
            {
                "candidates": int(candidates or 0),
                "opened": int(opened or 0),
                "policy_state": str(policy_state or "UNKNOWN").strip().upper() or "UNKNOWN",
                "symbols": sym_set,
                "skip_reasons": dict(skip_reasons_cycle or {}),
            }
        )

    def _explore_window_metrics(self, auto_trader: Any | None) -> dict[str, Any] | None:
        if auto_trader is None:
            return None
        closed = list(getattr(auto_trader, "closed_positions", []) or [])
        if not closed:
            return None
        pnl_rows: list[float] = []
        for pos in reversed(closed):
            ch = str(getattr(pos, "entry_channel", "") or "").strip().lower()
            if ch != "explore":
                continue
            pnl_rows.append(float(getattr(pos, "pnl_usd", 0.0) or 0.0))
            if len(pnl_rows) >= int(self.explore_failsafe_window_closed):
                break
        if len(pnl_rows) < int(self.explore_failsafe_min_closed):
            return None
        wins = sum(1 for x in pnl_rows if x > 0.0)
        losses = sum(1 for x in pnl_rows if x < 0.0)
        count = len(pnl_rows)
        avg_pnl = float(sum(pnl_rows)) / float(max(1, count))
        loss_share = float(losses) / float(max(1, count))
        win_rate = float(wins) / float(max(1, count))
        return {
            "count": int(count),
            "avg_pnl_usd": float(avg_pnl),
            "loss_share": float(loss_share),
            "win_rate": float(win_rate),
            "sum_pnl_usd": float(sum(pnl_rows)),
        }

    def maybe_apply(self, *, auto_trader: Any | None = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        now_ts = time.time()
        fast_payload = self._maybe_apply_fast_antistall(now_ts=now_ts)
        if isinstance(fast_payload, dict):
            return fast_payload
        if now_ts < float(self._next_eval_ts):
            return None
        self._next_eval_ts = now_ts + float(self.interval_seconds)
        if len(self._rows) < int(self.window_cycles):
            return None

        rows = list(self._rows)
        cand_total = int(sum(int(row.get("candidates", 0) or 0) for row in rows))
        opened_total = int(sum(int(row.get("opened", 0) or 0) for row in rows))
        open_rate = (float(opened_total) / float(cand_total)) if cand_total > 0 else 0.0
        policy_bad = sum(1 for row in rows if str(row.get("policy_state", "UNKNOWN")) != "OK")
        policy_block_share = float(policy_bad) / float(max(1, len(rows)))

        symbols: set[str] = set()
        skip_total = 0
        skip_edge_low = 0
        for row in rows:
            symbols.update(set(row.get("symbols", set()) or set()))
            skip_map = dict(row.get("skip_reasons", {}) or {})
            for key, value in skip_map.items():
                iv = int(value or 0)
                skip_total += iv
                kk = str(key or "").strip().lower()
                if kk in {"negative_edge", "edge_low", "edge_usd_low", "ev_net_low"}:
                    skip_edge_low += iv
        edge_low_share = (float(skip_edge_low) / float(skip_total)) if skip_total > 0 else 0.0
        unique_symbols = len(symbols)

        max_buys_now = int(getattr(config, "MAX_BUYS_PER_HOUR", 24) or 24)
        top_n_now = int(getattr(config, "AUTO_TRADE_TOP_N", 10) or 10)
        explore_share_now = _clamp(
            float(getattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE", 0.35) or 0.35),
            0.0,
            1.0,
        )
        self._explore_share_baseline = max(float(self._explore_share_baseline), float(explore_share_now))
        novelty_share_now = _clamp(
            float(getattr(config, "V2_UNIVERSE_NOVELTY_MIN_SHARE", 0.30) or 0.30),
            0.0,
            1.0,
        )

        max_buys_next = max_buys_now
        top_n_next = top_n_now
        explore_share_next = explore_share_now
        novelty_share_next = novelty_share_now
        action_parts: list[str] = []
        explore_stats = self._explore_window_metrics(auto_trader=auto_trader)
        explore_failsafe_active = now_ts < float(self._explore_failsafe_until_ts)
        explore_failsafe_shrunk = False

        if bool(self.explore_failsafe_enabled) and isinstance(explore_stats, dict):
            exp_loss_share = float(explore_stats.get("loss_share", 0.0) or 0.0)
            exp_avg_pnl = float(explore_stats.get("avg_pnl_usd", 0.0) or 0.0)
            exp_win_rate = float(explore_stats.get("win_rate", 0.0) or 0.0)
            degrade = (
                exp_loss_share >= float(self.explore_failsafe_loss_share_trigger)
                or exp_avg_pnl <= float(self.explore_failsafe_avg_pnl_trigger)
            )
            recover = (
                exp_win_rate >= float(self.explore_failsafe_recovery_winrate_min)
                and exp_avg_pnl >= float(self.explore_failsafe_recovery_avg_pnl_min)
            )
            if (not explore_failsafe_active) and degrade:
                explore_floor = max(0.0, float(self.explore_failsafe_min_share))
                explore_share_next = max(explore_floor, float(explore_share_now) - float(self.explore_failsafe_shrink_step))
                self._explore_failsafe_until_ts = now_ts + float(self.explore_failsafe_cooldown_seconds)
                explore_failsafe_active = True
                explore_failsafe_shrunk = abs(explore_share_next - explore_share_now) >= 0.0001
                if explore_failsafe_shrunk:
                    action_parts.append("explore_failsafe_shrink")
            elif recover and float(explore_share_now) < float(self._explore_share_baseline):
                recovery_cap = min(float(self.explore_share_max), float(self._explore_share_baseline))
                explore_share_next = min(recovery_cap, float(explore_share_now) + float(self.explore_failsafe_recovery_step))
                if abs(explore_share_next - explore_share_now) >= 0.0001:
                    action_parts.append("explore_failsafe_recover")

        low_flow = open_rate <= float(self.open_rate_low_trigger) or edge_low_share >= float(self.edge_low_relax_trigger)
        hard_edge_regime = edge_low_share >= float(self.edge_low_hard_trigger)
        policy_drag = policy_block_share >= float(self.policy_block_trigger)
        if bool(getattr(config, "V2_ANTI_SELF_TIGHTEN_ENABLED", True)) and low_flow:
            cooldown = max(60, int(getattr(config, "V2_ANTI_SELF_TIGHTEN_COOLDOWN_SECONDS", 1800) or 1800))
            relax_until_now = _safe_float(getattr(config, "V2_RUNTIME_EDGE_RELAX_UNTIL_TS", 0.0), 0.0)
            relax_until_next = now_ts + float(cooldown)
            if relax_until_next > float(relax_until_now):
                setattr(config, "V2_RUNTIME_EDGE_RELAX_UNTIL_TS", float(relax_until_next))
        if bool(self.quality_rebalance_enabled) and hard_edge_regime:
            if self.quality_rebalance_max_buys_step > 0:
                max_buys_next = max(1, max_buys_now - int(self.quality_rebalance_max_buys_step))
            if self.quality_rebalance_topn_step > 0:
                top_n_next = max(1, top_n_now - int(self.quality_rebalance_topn_step))
            explore_share_next = max(0.05, explore_share_now - float(self.quality_rebalance_explore_step))
            novelty_share_next = max(0.10, novelty_share_now - float(self.quality_rebalance_novelty_step))
            setattr(config, "V2_QUALITY_SOURCE_BUDGET_ENABLED", True)
            action_parts.append("quality_rebalance")
        elif low_flow:
            if self.max_buys_boost_step > 0:
                max_buys_next = min(int(self.max_buys_cap), max_buys_now + int(self.max_buys_boost_step))
            if self.topn_boost_step > 0:
                top_n_next = min(int(self.topn_cap), top_n_now + int(self.topn_boost_step))
            if (not explore_failsafe_active) and (not explore_failsafe_shrunk):
                explore_share_next = min(float(self.explore_share_max), explore_share_now + float(self.explore_share_step))
            action_parts.append("throughput_boost")
        if unique_symbols < int(self.unique_symbols_min):
            novelty_share_next = min(float(self.novelty_share_max), novelty_share_now + float(self.novelty_share_step))
            action_parts.append("diversity_boost")
        if policy_drag and low_flow and self.max_buys_boost_step > 0:
            max_buys_next = min(int(self.max_buys_cap), max_buys_next + int(max(1, self.max_buys_boost_step // 2)))
            action_parts.append("policy_drag_boost")

        changed = (
            (max_buys_next != max_buys_now)
            or (top_n_next != top_n_now)
            or (abs(explore_share_next - explore_share_now) >= 0.0001)
            or (abs(novelty_share_next - novelty_share_now) >= 0.0001)
        )
        if not changed:
            return None

        setattr(config, "MAX_BUYS_PER_HOUR", int(max_buys_next))
        setattr(config, "AUTO_TRADE_TOP_N", int(top_n_next))
        setattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE", float(explore_share_next))
        setattr(config, "V2_UNIVERSE_NOVELTY_MIN_SHARE", float(novelty_share_next))

        action = ",".join(action_parts) if action_parts else "adjust"
        logger.warning(
            (
                "V2_KPI_LOOP action=%s open_rate=%.3f edge_low_share=%.3f policy_share=%.3f "
                "unique_symbols=%s max_buys=%s->%s top_n=%s->%s explore_share=%.2f->%.2f novelty_share=%.2f->%.2f "
                "explore_failsafe(active=%s until=%.0f count=%s wr=%.3f loss=%.3f avg=$%.4f)"
            ),
            action,
            open_rate,
            edge_low_share,
            policy_block_share,
            unique_symbols,
            max_buys_now,
            max_buys_next,
            top_n_now,
            top_n_next,
            explore_share_now,
            explore_share_next,
            novelty_share_now,
            novelty_share_next,
            bool(explore_failsafe_active),
            float(self._explore_failsafe_until_ts or 0.0),
            int((explore_stats or {}).get("count", 0) or 0),
            float((explore_stats or {}).get("win_rate", 0.0) or 0.0),
            float((explore_stats or {}).get("loss_share", 0.0) or 0.0),
            float((explore_stats or {}).get("avg_pnl_usd", 0.0) or 0.0),
        )
        return {
            "event_type": "V2_KPI_LOOP",
            "action": action,
            "open_rate": round(open_rate, 6),
            "edge_low_share": round(edge_low_share, 6),
            "policy_block_share": round(policy_block_share, 6),
            "unique_symbols": int(unique_symbols),
            "max_buys_prev": int(max_buys_now),
            "max_buys_next": int(max_buys_next),
            "top_n_prev": int(top_n_now),
            "top_n_next": int(top_n_next),
            "explore_share_prev": round(explore_share_now, 6),
            "explore_share_next": round(explore_share_next, 6),
            "explore_share_baseline": round(float(self._explore_share_baseline), 6),
            "explore_failsafe_active": bool(explore_failsafe_active),
            "explore_failsafe_until_ts": round(float(self._explore_failsafe_until_ts or 0.0), 3),
            "explore_perf_count": int((explore_stats or {}).get("count", 0) or 0),
            "explore_perf_win_rate": round(float((explore_stats or {}).get("win_rate", 0.0) or 0.0), 6),
            "explore_perf_loss_share": round(float((explore_stats or {}).get("loss_share", 0.0) or 0.0), 6),
            "explore_perf_avg_pnl_usd": round(float((explore_stats or {}).get("avg_pnl_usd", 0.0) or 0.0), 6),
            "novelty_share_prev": round(novelty_share_now, 6),
            "novelty_share_next": round(novelty_share_next, 6),
            "window_cycles": int(len(rows)),
        }

    def _recent_window_metrics(self) -> tuple[float, float, float]:
        rows = list(self._rows)
        if not rows:
            return 0.0, 0.0, 0.0
        cand_total = int(sum(int(row.get("candidates", 0) or 0) for row in rows))
        opened_total = int(sum(int(row.get("opened", 0) or 0) for row in rows))
        open_rate = (float(opened_total) / float(cand_total)) if cand_total > 0 else 0.0
        skip_total = 0
        skip_edge_low = 0
        for row in rows:
            skip_map = dict(row.get("skip_reasons", {}) or {})
            for key, value in skip_map.items():
                iv = int(value or 0)
                skip_total += iv
                kk = str(key or "").strip().lower()
                if kk in {"negative_edge", "edge_low", "edge_usd_low", "ev_net_low"}:
                    skip_edge_low += iv
        edge_low_share = (float(skip_edge_low) / float(skip_total)) if skip_total > 0 else 0.0
        policy_bad = sum(1 for row in rows if str(row.get("policy_state", "UNKNOWN")) != "OK")
        policy_block_share = float(policy_bad) / float(max(1, len(rows)))
        return open_rate, edge_low_share, policy_block_share

    def _maybe_apply_fast_antistall(self, *, now_ts: float) -> dict[str, Any] | None:
        if not self.fast_antistall_enabled:
            return None
        if now_ts < float(self._next_fast_eval_ts):
            return None
        self._next_fast_eval_ts = now_ts + float(self.fast_antistall_interval_seconds)
        if len(self._rows) < max(3, min(self.window_cycles, 6)):
            return None

        open_rate, edge_low_share, policy_block_share = self._recent_window_metrics()
        low_flow_now = (
            open_rate <= float(self.fast_antistall_open_rate_trigger)
            and edge_low_share >= float(self.fast_antistall_edge_low_share_trigger)
        )
        if bool(self.quality_rebalance_enabled) and edge_low_share >= float(self.edge_low_hard_trigger):
            self._low_flow_streak = 0
            return None
        if low_flow_now:
            self._low_flow_streak += 1
        else:
            self._low_flow_streak = 0
            return None
        if int(self._low_flow_streak) < int(self.fast_antistall_streak_trigger):
            return None

        keep: deque[float] = deque(maxlen=self._fast_apply_ts.maxlen)
        for ts in self._fast_apply_ts:
            if (now_ts - float(ts)) <= 3600.0:
                keep.append(float(ts))
        self._fast_apply_ts = keep
        if len(self._fast_apply_ts) >= int(self.fast_antistall_max_apply_per_hour):
            return None

        edge_pct_now = _safe_float(getattr(config, "MIN_EXPECTED_EDGE_PERCENT", 1.0), 1.0)
        edge_usd_now = _safe_float(getattr(config, "MIN_EXPECTED_EDGE_USD", 0.02), 0.02)
        edge_pct_floor = max(
            _safe_float(getattr(config, "V2_CALIBRATION_EDGE_PERCENT_MIN", 0.35), 0.35),
            _safe_float(getattr(config, "V2_ROLLING_EDGE_MIN_PERCENT", 0.35), 0.35),
        )
        edge_usd_floor = max(
            _safe_float(getattr(config, "V2_CALIBRATION_EDGE_USD_MIN", 0.008), 0.008),
            _safe_float(getattr(config, "V2_ROLLING_EDGE_MIN_USD", 0.008), 0.008),
        )
        edge_pct_next = _clamp(
            edge_pct_now - float(self.fast_antistall_edge_percent_drop),
            edge_pct_floor,
            _safe_float(getattr(config, "V2_CALIBRATION_EDGE_PERCENT_MAX", 3.0), 3.0),
        )
        edge_usd_next = _clamp(
            edge_usd_now - float(self.fast_antistall_edge_usd_drop),
            edge_usd_floor,
            _safe_float(getattr(config, "V2_CALIBRATION_EDGE_USD_MAX", 0.12), 0.12),
        )
        max_buys_now = int(getattr(config, "MAX_BUYS_PER_HOUR", 24) or 24)
        top_n_now = int(getattr(config, "AUTO_TRADE_TOP_N", 10) or 10)
        max_buys_cap = max(1, int(getattr(config, "V2_KPI_MAX_BUYS_CAP", max_buys_now) or max_buys_now))
        top_n_cap = max(1, int(getattr(config, "V2_KPI_TOPN_CAP", top_n_now) or top_n_now))
        max_buys_next = min(max_buys_cap, max_buys_now + max(1, int(self.max_buys_boost_step)))
        top_n_next = min(top_n_cap, top_n_now + max(1, int(self.topn_boost_step)))
        recovery_until_ts = now_ts + float(self.fast_antistall_recovery_window_seconds)

        setattr(config, "MIN_EXPECTED_EDGE_PERCENT", float(edge_pct_next))
        setattr(config, "MIN_EXPECTED_EDGE_USD", float(edge_usd_next))
        setattr(config, "MAX_BUYS_PER_HOUR", int(max_buys_next))
        setattr(config, "AUTO_TRADE_TOP_N", int(top_n_next))
        setattr(config, "V2_RUNTIME_EDGE_RELAX_UNTIL_TS", float(recovery_until_ts))
        self._fast_apply_ts.append(float(now_ts))
        self._low_flow_streak = 0

        logger.warning(
            (
                "V2_FAST_ANTISTALL applied open_rate=%.3f edge_low_share=%.3f policy_share=%.3f "
                "edge_pct=%.3f->%.3f edge_usd=%.4f->%.4f max_buys=%s->%s top_n=%s->%s recovery=%ss"
            ),
            open_rate,
            edge_low_share,
            policy_block_share,
            edge_pct_now,
            edge_pct_next,
            edge_usd_now,
            edge_usd_next,
            max_buys_now,
            max_buys_next,
            top_n_now,
            top_n_next,
            int(self.fast_antistall_recovery_window_seconds),
        )
        return {
            "event_type": "V2_FAST_ANTISTALL",
            "action": "edge_relax_recovery",
            "open_rate": round(open_rate, 6),
            "edge_low_share": round(edge_low_share, 6),
            "policy_block_share": round(policy_block_share, 6),
            "streak_trigger": int(self.fast_antistall_streak_trigger),
            "edge_percent_prev": round(edge_pct_now, 6),
            "edge_percent_next": round(edge_pct_next, 6),
            "edge_usd_prev": round(edge_usd_now, 6),
            "edge_usd_next": round(edge_usd_next, 6),
            "max_buys_prev": int(max_buys_now),
            "max_buys_next": int(max_buys_next),
            "top_n_prev": int(top_n_now),
            "top_n_next": int(top_n_next),
            "recovery_until_ts": float(recovery_until_ts),
            "recovery_window_seconds": int(self.fast_antistall_recovery_window_seconds),
        }


class MatrixChampionGuard:
    """Champion/challenger auto-stop: stop profile if it lags behind active champion."""

    def __init__(self, *, run_tag: str, paper_state_file: str, graceful_stop_file: str) -> None:
        self.enabled = bool(getattr(config, "V2_CHAMPION_GUARD_ENABLED", False))
        self.eval_interval_seconds = max(30, int(getattr(config, "V2_CHAMPION_GUARD_INTERVAL_SECONDS", 120) or 120))
        self.min_runtime_seconds = max(180, int(getattr(config, "V2_CHAMPION_GUARD_MIN_RUNTIME_SECONDS", 3600) or 3600))
        self.min_closed_trades = max(1, int(getattr(config, "V2_CHAMPION_GUARD_MIN_CLOSED_TRADES", 20) or 20))
        self.max_lag_usd = max(0.0, float(getattr(config, "V2_CHAMPION_GUARD_MAX_LAG_USD", 0.22) or 0.22))
        self.fail_windows = max(1, int(getattr(config, "V2_CHAMPION_GUARD_FAIL_WINDOWS", 3) or 3))
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.matrix_active_path = str(
            getattr(
                config,
                "V2_CHAMPION_GUARD_ACTIVE_MATRIX_PATH",
                os.path.join(root, "data", "matrix", "runs", "active_matrix.json"),
            )
            or os.path.join(root, "data", "matrix", "runs", "active_matrix.json")
        )
        self.run_tag = str(run_tag or "").strip()
        self.paper_state_file = str(paper_state_file or "")
        self.graceful_stop_file = str(graceful_stop_file or "")
        self.started_ts = time.time()
        self.next_eval_ts = self.started_ts + float(self.eval_interval_seconds)
        self.bad_windows = 0
        self.triggered = False

    def maybe_stop(self, *, auto_stats: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        if not self.enabled or self.triggered:
            return False, None
        now_ts = time.time()
        if now_ts < float(self.next_eval_ts):
            return False, None
        self.next_eval_ts = now_ts + float(self.eval_interval_seconds)
        runtime = int(now_ts - float(self.started_ts))
        if runtime < int(self.min_runtime_seconds):
            return False, None
        own_closed = int(auto_stats.get("closed", 0) or 0)
        if own_closed < int(self.min_closed_trades):
            return False, None

        score_rows = self._load_scoreboard()
        if len(score_rows) < 2:
            return False, None
        best = max(score_rows, key=lambda row: float(row.get("realized", 0.0)))
        own = None
        for row in score_rows:
            if str(row.get("run_tag", "")).strip() == self.run_tag:
                own = row
                break
        if own is None:
            own = {
                "run_tag": self.run_tag,
                "realized": _safe_float(auto_stats.get("realized_pnl_usd"), 0.0),
                "closed": own_closed,
                "paper_state_file": self.paper_state_file,
            }

        lag = float(best.get("realized", 0.0)) - float(own.get("realized", 0.0))
        losing = (
            str(best.get("run_tag", "")) != self.run_tag
            and lag > float(self.max_lag_usd)
            and int(own.get("closed", own_closed) or own_closed) >= int(self.min_closed_trades)
        )
        if losing:
            self.bad_windows += 1
        else:
            self.bad_windows = max(0, int(self.bad_windows) - 1)

        logger.warning(
            "V2_CHAMPION_GUARD run=%s best=%s lag=$%.4f bad_windows=%s/%s own_closed=%s",
            self.run_tag,
            str(best.get("run_tag", "")),
            lag,
            self.bad_windows,
            self.fail_windows,
            own_closed,
        )
        if self.bad_windows < int(self.fail_windows):
            return False, None

        self.triggered = True
        reason = (
            f"lagging_champion best={best.get('run_tag','')} best_realized={float(best.get('realized',0.0)):.4f} "
            f"own_realized={float(own.get('realized',0.0)):.4f} lag={lag:.4f} "
            f"windows={self.bad_windows}/{self.fail_windows}"
        )
        self._write_stop_file(reason=reason)
        event = {
            "event_type": "PROFILE_AUTOSTOP",
            "symbol": "PROFILE_AUTOSTOP",
            "score": 0,
            "recommendation": "STOP",
            "risk_level": "WARNING",
            "name": f"{self.run_tag} stopped by champion_guard",
            "breakdown": {
                "run_tag": self.run_tag,
                "reason": reason,
                "best_run_tag": str(best.get("run_tag", "")),
                "best_realized_pnl_usd": float(best.get("realized", 0.0)),
                "own_realized_pnl_usd": float(own.get("realized", 0.0)),
                "lag_usd": float(lag),
                "bad_windows": int(self.bad_windows),
            },
            "address": "",
            "liquidity": 0.0,
            "volume_5m": 0.0,
            "price_change_5m": 0.0,
            "warning_flags": int(self.bad_windows),
        }
        return True, event

    def _load_scoreboard(self) -> list[dict[str, Any]]:
        path = self.matrix_active_path
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            profiles = payload.get("profiles", []) if isinstance(payload, dict) else []
            out: list[dict[str, Any]] = []
            for row in profiles:
                if not isinstance(row, dict):
                    continue
                run_tag = str(row.get("id", "") or "")
                ps = str(row.get("paper_state_file", "") or "")
                if not os.path.isabs(ps):
                    ps = os.path.abspath(os.path.join(os.path.dirname(path), "..", "..", "..", ps))
                realized = 0.0
                closed = 0
                if ps and os.path.exists(ps):
                    try:
                        with open(ps, "r", encoding="utf-8", errors="ignore") as sf:
                            state = json.load(sf)
                        realized = _safe_float(state.get("realized_pnl_usd"), 0.0)
                        closed = _safe_int(state.get("total_closed"), 0)
                    except Exception:
                        pass
                out.append(
                    {
                        "run_tag": run_tag,
                        "realized": realized,
                        "closed": closed,
                        "paper_state_file": ps,
                    }
                )
            return out
        except Exception:
            logger.exception("V2_CHAMPION_GUARD failed to read active matrix path=%s", path)
            return []

    def _write_stop_file(self, *, reason: str) -> None:
        try:
            path = self.graceful_stop_file
            if not path:
                return
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "run_tag": self.run_tag,
                            "event": "CHAMPION_GUARD_AUTOSTOP",
                            "reason": reason,
                        },
                        ensure_ascii=False,
                        sort_keys=False,
                    )
                )
        except Exception:
            logger.exception("V2_CHAMPION_GUARD failed to write stop file")
