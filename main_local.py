"""Local fast runner without Telegram polling/webhooks."""

from __future__ import annotations

import asyncio
import atexit
import ctypes
import hashlib
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

try:
    import msvcrt  # Windows-only, used for a robust single-instance file lock.
except Exception:  # pragma: no cover
    msvcrt = None  # type: ignore[assignment]

import config
from config import APP_LOG_FILE, LOG_DIR, LOG_LEVEL, OUT_LOG_FILE, SCAN_INTERVAL
from monitor.dexscreener import DexScreenerMonitor
from monitor.local_alerter import LocalAlerter
from monitor.onchain_factory import OnChainFactoryMonitor, OnChainRPCError
from monitor.token_checker import TokenChecker
from monitor.token_scorer import TokenScorer
from monitor.watchlist import WatchlistMonitor
from trading.auto_trader import AutoTrader
from utils.addressing import normalize_address

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INSTANCE_ID = str(getattr(config, "BOT_INSTANCE_ID", "") or "").strip()
RUN_TAG = str(getattr(config, "RUN_TAG", INSTANCE_ID) or INSTANCE_ID or "single").strip()
WINDOWS_MUTEX_NAME = (
    f"Global\\solana_alert_bot_main_local_{INSTANCE_ID}"
    if INSTANCE_ID
    else "Global\\solana_alert_bot_main_local_single_instance"
)


def _graceful_stop_file_path() -> str:
    raw = str(getattr(config, "GRACEFUL_STOP_FILE", os.path.join("data", "graceful_stop.signal")) or "").strip()
    if not raw:
        raw = os.path.join("data", "graceful_stop.signal")
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(PROJECT_ROOT, raw))


def _graceful_stop_requested() -> bool:
    try:
        return os.path.exists(_graceful_stop_file_path())
    except Exception:
        return False


def _clear_graceful_stop_flag() -> None:
    try:
        path = _graceful_stop_file_path()
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _rss_memory_mb() -> float:
    if os.name != "nt":
        return 0.0
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return 0.0
        return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _merge_source_stats(*parts: dict[str, dict[str, int | float]]) -> dict[str, dict[str, int | float]]:
    merged: dict[str, dict[str, int | float]] = {}
    for block in parts:
        for source, row in (block or {}).items():
            cur = merged.setdefault(
                source,
                {
                    "ok": 0,
                    "fail": 0,
                    "total": 0,
                    "rate_limited": 0,
                    "limiter_waits": 0,
                    "cooldown_waits": 0,
                    "cooldown_active": 0,
                    "cooldown_remaining_sec": 0.0,
                    "retries": 0,
                    "error_percent": 0.0,
                    "latency_avg_ms": 0.0,
                    "latency_max_ms": 0.0,
                    "_latency_weighted_sum_ms": 0.0,
                    "_latency_total_samples": 0,
                },
            )
            cur["ok"] = int(cur["ok"]) + int(row.get("ok", 0))
            cur["fail"] = int(cur["fail"]) + int(row.get("fail", 0))
            cur["total"] = int(cur["total"]) + int(row.get("total", 0))
            cur["rate_limited"] = int(cur["rate_limited"]) + int(row.get("rate_limited", 0))
            cur["limiter_waits"] = int(cur["limiter_waits"]) + int(row.get("limiter_waits", 0))
            cur["cooldown_waits"] = int(cur["cooldown_waits"]) + int(row.get("cooldown_waits", 0))
            cur["cooldown_active"] = max(int(cur["cooldown_active"]), int(row.get("cooldown_active", 0)))
            cur["cooldown_remaining_sec"] = max(
                float(cur["cooldown_remaining_sec"]),
                float(row.get("cooldown_remaining_sec", 0.0) or 0.0),
            )
            cur["retries"] = int(cur["retries"]) + int(row.get("retries", 0))
            row_total = int(row.get("total", 0) or 0)
            row_avg = float(row.get("latency_avg_ms", 0.0) or 0.0)
            cur["_latency_weighted_sum_ms"] = float(cur["_latency_weighted_sum_ms"]) + (row_avg * row_total)
            cur["_latency_total_samples"] = int(cur["_latency_total_samples"]) + row_total
            cur["latency_max_ms"] = max(float(cur["latency_max_ms"]), float(row.get("latency_max_ms", 0.0) or 0.0))
    for source, row in merged.items():
        total = int(row.get("total", 0))
        fail = int(row.get("fail", 0))
        row["error_percent"] = round((float(fail) / total * 100.0) if total > 0 else 0.0, 2)
        lat_n = int(row.get("_latency_total_samples", 0) or 0)
        lat_sum = float(row.get("_latency_weighted_sum_ms", 0.0) or 0.0)
        row["latency_avg_ms"] = round((lat_sum / lat_n), 2) if lat_n > 0 else 0.0
        row.pop("_latency_weighted_sum_ms", None)
        row.pop("_latency_total_samples", None)
    return merged


def _policy_state(
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


def _apply_policy_hysteresis(
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

    # raw_mode == OK
    good_streak += 1
    bad_streak = 0
    if current_mode in {"FAIL_CLOSED", "DEGRADED"} and good_streak < exit_req:
        return current_mode, f"hysteresis_exit_wait {good_streak}/{exit_req} raw=OK", bad_streak, good_streak
    return "OK", raw_reason, bad_streak, good_streak


def _format_source_stats_brief(source_stats: dict[str, dict[str, int | float]]) -> str:
    if not source_stats:
        return "none"
    parts: list[str] = []
    for source in sorted(source_stats.keys()):
        row = source_stats.get(source) or {}
        parts.append(
            (
                f"{source}:ok={int(row.get('ok', 0))}"
                f"/fail={int(row.get('fail', 0))}"
                f"/429={int(row.get('rate_limited', 0))}"
                f"/lim_wait={int(row.get('limiter_waits', 0))}"
                f"/cd_wait={int(row.get('cooldown_waits', 0))}"
                f"/err={float(row.get('error_percent', 0.0)):.1f}%"
                f"/avg={float(row.get('latency_avg_ms', 0.0)):.0f}ms"
                f"{('/cd=' + str(int(float(row.get('cooldown_remaining_sec', 0.0) or 0.0))) + 's') if int(row.get('cooldown_active', 0)) else ''}"
            )
        )
    return "; ".join(parts)


def _format_safety_reasons_brief(safety_stats: dict[str, int | float]) -> str:
    counts = safety_stats.get("fail_reason_counts")
    if not isinstance(counts, dict) or not counts:
        return "none"
    pairs: list[tuple[str, int]] = []
    for key, value in counts.items():
        try:
            pairs.append((str(key), int(value)))
        except Exception:
            continue
    if not pairs:
        return "none"
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    return ",".join(f"{k}:{v}" for k, v in pairs[:4])


def _format_ingest_stats_brief(ingest_stats: dict[str, int | float]) -> str:
    if not ingest_stats:
        return "none"
    return (
        f"q={int(ingest_stats.get('queue_size', 0))}"
        f"/oldest={float(ingest_stats.get('oldest_item_age_sec', 0.0)):.1f}s"
        f"/drop_old={int(ingest_stats.get('dropped_oldest_count', 0))}"
        f"/dedup_skip={int(ingest_stats.get('ingest_dedup_skipped', 0))}"
        f"/enq={int(ingest_stats.get('enqueued_count', 0))}"
        f"/drain={int(ingest_stats.get('drained_count', 0))}"
    )


def _format_top_filter_reasons(filter_fail_reasons: dict[str, int], limit: int = 4) -> str:
    if not filter_fail_reasons:
        return "none"
    rows = sorted(filter_fail_reasons.items(), key=lambda kv: int(kv[1]), reverse=True)
    return ",".join(f"{k}:{int(v)}" for k, v in rows[: max(1, int(limit))])


def _belongs_to_candidate_shard(address: str) -> bool:
    mod = max(1, int(getattr(config, "CANDIDATE_SHARD_MOD", 1) or 1))
    slot = int(getattr(config, "CANDIDATE_SHARD_SLOT", 0) or 0)
    if mod <= 1:
        return True
    if slot < 0:
        slot = 0
    if slot >= mod:
        slot = slot % mod
    addr = normalize_address(address)
    if not addr:
        return True
    digest = hashlib.sha1(addr.encode("ascii", errors="ignore")).digest()
    bucket = int.from_bytes(digest[:4], byteorder="big", signed=False) % mod
    return bucket == slot


def _format_top_counts(counts: dict[str, int], limit: int = 4) -> str:
    if not counts:
        return "none"
    rows = sorted(counts.items(), key=lambda kv: int(kv[1]), reverse=True)
    return ",".join(f"{k}:{int(v)}" for k, v in rows[: max(1, int(limit))])


def _candidate_quality_features(token: dict, score_data: dict) -> dict[str, object]:
    score = int(score_data.get("score", 0) or 0)
    liquidity = float(token.get("liquidity") or 0.0)
    volume_5m = float(token.get("volume_5m") or 0.0)
    price_change_5m = float(token.get("price_change_5m") or 0.0)
    age_seconds = int(token.get("age_seconds") or 0)
    risk_level = str(token.get("risk_level", "UNKNOWN")).upper()

    liq_to_vol_ratio = (volume_5m / liquidity) if liquidity > 0 else 0.0
    momentum_positive = price_change_5m > 0.0
    volatility_abs_5m = abs(price_change_5m)

    quality_band = "C"
    if score >= 85 and momentum_positive and liq_to_vol_ratio >= 0.08:
        quality_band = "A+"
    elif score >= 75 and liq_to_vol_ratio >= 0.05:
        quality_band = "A"
    elif score >= 65:
        quality_band = "B"

    return {
        "quality_band": quality_band,
        "score": score,
        "risk_level": risk_level,
        "liquidity_usd": liquidity,
        "volume_5m_usd": volume_5m,
        "liq_to_vol_5m_ratio": round(liq_to_vol_ratio, 6),
        "price_change_5m": price_change_5m,
        "volatility_abs_5m": volatility_abs_5m,
        "momentum_positive_5m": momentum_positive,
        "age_seconds": age_seconds,
    }


def _detect_market_regime(
    *,
    policy_state: str,
    source_stats: dict[str, dict[str, int | float]],
    safety_stats: dict[str, int | float],
    avg_candidates_recent: float,
) -> tuple[str, str]:
    mode = str(policy_state).upper()
    if mode == "FAIL_CLOSED":
        return "RISK_OFF", "policy_fail_closed"
    if mode == "DEGRADED":
        return "CAUTION", "policy_degraded"

    checks_total = int(safety_stats.get("checks_total", 0) or 0)
    fail_closed = int(safety_stats.get("fail_closed", 0) or 0)
    fail_closed_ratio = (float(fail_closed) / checks_total * 100.0) if checks_total > 0 else 0.0
    max_err = 0.0
    for row in (source_stats or {}).values():
        max_err = max(max_err, float((row or {}).get("error_percent", 0.0) or 0.0))

    if (
        fail_closed_ratio >= float(getattr(config, "MARKET_REGIME_FAIL_CLOSED_RATIO", 20.0))
        or max_err >= float(getattr(config, "MARKET_REGIME_SOURCE_ERROR_PERCENT", 20.0))
    ):
        return "FRAGILE", f"fail_closed={fail_closed_ratio:.1f}% max_err={max_err:.1f}%"

    if avg_candidates_recent >= float(getattr(config, "MARKET_REGIME_MOMENTUM_CANDIDATES", 2.5)):
        return "MOMENTUM", f"avg_cand={avg_candidates_recent:.2f}"

    if avg_candidates_recent <= float(getattr(config, "MARKET_REGIME_THIN_CANDIDATES", 0.8)):
        return "THIN", f"avg_cand={avg_candidates_recent:.2f}"

    return "BALANCED", f"avg_cand={avg_candidates_recent:.2f}"


def _regime_entry_overrides(market_regime: str) -> tuple[int, float, float]:
    mode = str(market_regime or "BALANCED").strip().upper()
    # score_delta, volume_mult, edge_mult
    if mode == "MOMENTUM":
        # Slightly loosen thresholds to capture continuation bursts.
        return -1, 0.90, 0.90
    if mode == "THIN":
        # Thin tape: be pickier, avoid low-energy chop.
        return 2, 1.15, 1.10
    if mode in {"FRAGILE", "CAUTION", "RISK_OFF"}:
        # Source/safety unstable: strongly tighten BUY gate.
        return 4, 1.30, 1.20
    return 0, 1.00, 1.00


def _excluded_trade_addresses() -> set[str]:
    out: set[str] = set()
    weth = normalize_address(getattr(config, "WETH_ADDRESS", ""))
    if weth:
        out.add(weth)
    for addr in (getattr(config, "AUTO_TRADE_EXCLUDED_ADDRESSES", []) or []):
        n = normalize_address(str(addr))
        if n:
            out.add(n)
    return out


def _merge_token_streams(*groups: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for group in groups:
        for token in group or []:
            address = normalize_address(token.get("address", ""))
            if not address:
                continue
            existing = merged.get(address)
            if not existing:
                merged[address] = token
                continue
            existing_score = (
                (1 if float(existing.get("liquidity") or 0) > 0 else 0)
                + (1 if float(existing.get("volume_5m") or 0) > 0 else 0)
                + (1 if float(existing.get("price_usd") or 0) > 0 else 0)
            )
            candidate_score = (
                (1 if float(token.get("liquidity") or 0) > 0 else 0)
                + (1 if float(token.get("volume_5m") or 0) > 0 else 0)
                + (1 if float(token.get("price_usd") or 0) > 0 else 0)
            )
            if candidate_score > existing_score:
                merged[address] = token
    return list(merged.values())


class AdaptiveFilterController:
    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "ADAPTIVE_FILTERS_ENABLED", False))
        self.mode = str(getattr(config, "ADAPTIVE_FILTERS_MODE", "dry_run") or "dry_run").strip().lower()
        self.paper_only = bool(getattr(config, "ADAPTIVE_FILTERS_PAPER_ONLY", True))
        self.interval_seconds = int(getattr(config, "ADAPTIVE_FILTERS_INTERVAL_SECONDS", 900) or 900)
        self.min_window_cycles = int(getattr(config, "ADAPTIVE_FILTERS_MIN_WINDOW_CYCLES", 5) or 5)
        self.cooldown_windows = int(getattr(config, "ADAPTIVE_FILTERS_COOLDOWN_WINDOWS", 1) or 1)
        self.target_cand_min = float(getattr(config, "ADAPTIVE_FILTERS_TARGET_CAND_MIN", 2.0) or 2.0)
        self.target_cand_max = float(getattr(config, "ADAPTIVE_FILTERS_TARGET_CAND_MAX", 12.0) or 12.0)
        self.target_open_min = float(getattr(config, "ADAPTIVE_FILTERS_TARGET_OPEN_MIN", 0.10) or 0.10)
        self.zero_open_reset_enabled = bool(getattr(config, "ADAPTIVE_ZERO_OPEN_RESET_ENABLED", True))
        self.zero_open_windows_before_reset = int(getattr(config, "ADAPTIVE_ZERO_OPEN_WINDOWS_BEFORE_RESET", 2) or 2)
        self.zero_open_min_candidates = float(getattr(config, "ADAPTIVE_ZERO_OPEN_MIN_CANDIDATES", 1.0) or 1.0)
        self.neg_realized_trigger = float(getattr(config, "ADAPTIVE_FILTERS_NEG_REALIZED_TRIGGER_USD", 0.60) or 0.60)
        self.neg_closed_min = int(getattr(config, "ADAPTIVE_FILTERS_NEG_CLOSED_MIN", 3) or 3)
        self.pnl_min_closed = int(getattr(config, "ADAPTIVE_FILTERS_PNL_MIN_CLOSED", 2) or 2)

        self.score_min_bound = int(getattr(config, "ADAPTIVE_SCORE_MIN", 60) or 60)
        self.score_max_bound = int(getattr(config, "ADAPTIVE_SCORE_MAX", 72) or 72)
        self.score_step = int(getattr(config, "ADAPTIVE_SCORE_STEP", 1) or 1)
        self.volume_min_bound = float(getattr(config, "ADAPTIVE_SAFE_VOLUME_MIN", 150.0) or 150.0)
        self.volume_max_bound = float(getattr(config, "ADAPTIVE_SAFE_VOLUME_MAX", 1200.0) or 1200.0)
        self.volume_step = float(getattr(config, "ADAPTIVE_SAFE_VOLUME_STEP", 50.0) or 50.0)
        self.volume_step_pct = float(getattr(config, "ADAPTIVE_SAFE_VOLUME_STEP_PCT", 12.0) or 12.0)
        self.ttl_min_bound = int(getattr(config, "ADAPTIVE_DEDUP_TTL_MIN", 60) or 60)
        self.ttl_max_bound = int(getattr(config, "ADAPTIVE_DEDUP_TTL_MAX", 900) or 900)
        self.ttl_step = int(getattr(config, "ADAPTIVE_DEDUP_TTL_STEP", 30) or 30)
        self.ttl_step_pct = float(getattr(config, "ADAPTIVE_DEDUP_TTL_STEP_PCT", 25.0) or 25.0)
        self.dedup_relax_enabled = bool(getattr(config, "ADAPTIVE_DEDUP_RELAX_ENABLED", False))
        self.dedup_dynamic_enabled = bool(getattr(config, "ADAPTIVE_DEDUP_DYNAMIC_ENABLED", False))
        self.dedup_dynamic_min = int(getattr(config, "ADAPTIVE_DEDUP_DYNAMIC_MIN", 8) or 8)
        self.dedup_dynamic_max = int(getattr(config, "ADAPTIVE_DEDUP_DYNAMIC_MAX", 30) or 30)
        self.dedup_dynamic_target_percentile = float(
            getattr(config, "ADAPTIVE_DEDUP_DYNAMIC_TARGET_PERCENTILE", 90.0) or 90.0
        )
        self.dedup_dynamic_factor = float(getattr(config, "ADAPTIVE_DEDUP_DYNAMIC_FACTOR", 1.2) or 1.2)
        self.dedup_dynamic_min_samples = int(getattr(config, "ADAPTIVE_DEDUP_DYNAMIC_MIN_SAMPLES", 200) or 200)
        self.edge_enabled = bool(getattr(config, "ADAPTIVE_EDGE_ENABLED", True))
        self.edge_min_bound = float(getattr(config, "ADAPTIVE_EDGE_MIN", 1.5) or 1.5)
        self.edge_max_bound = float(getattr(config, "ADAPTIVE_EDGE_MAX", 4.0) or 4.0)
        self.edge_step = float(getattr(config, "ADAPTIVE_EDGE_STEP", 0.25) or 0.25)
        self.edge_step_pct = float(getattr(config, "ADAPTIVE_EDGE_STEP_PCT", 0.0) or 0.0)
        self.exit_enabled = bool(getattr(config, "ADAPTIVE_EXIT_ENABLED", True))
        self.profit_lock_floor_min = float(getattr(config, "ADAPTIVE_PROFIT_LOCK_FLOOR_MIN", 1.0) or 1.0)
        self.profit_lock_floor_max = float(getattr(config, "ADAPTIVE_PROFIT_LOCK_FLOOR_MAX", 8.0) or 8.0)
        self.profit_lock_floor_step = float(getattr(config, "ADAPTIVE_PROFIT_LOCK_FLOOR_STEP", 0.5) or 0.5)
        self.no_mom_pnl_min = float(getattr(config, "ADAPTIVE_NO_MOMENTUM_MAX_PNL_MIN", -0.5) or -0.5)
        self.no_mom_pnl_max = float(getattr(config, "ADAPTIVE_NO_MOMENTUM_MAX_PNL_MAX", 1.5) or 1.5)
        self.no_mom_pnl_step = float(getattr(config, "ADAPTIVE_NO_MOMENTUM_MAX_PNL_STEP", 0.2) or 0.2)
        self.weakness_pnl_min = float(getattr(config, "ADAPTIVE_WEAKNESS_PNL_MIN", -15.0) or -15.0)
        self.weakness_pnl_max = float(getattr(config, "ADAPTIVE_WEAKNESS_PNL_MAX", -2.0) or -2.0)
        self.weakness_pnl_step = float(getattr(config, "ADAPTIVE_WEAKNESS_PNL_STEP", 0.5) or 0.5)

        self.last_eval_ts = time.time()
        self.cooldown_until_ts = 0.0
        self.window_cycles = 0
        self.window_candidates = 0
        self.window_opened = 0
        self.window_filter_fails: dict[str, int] = {}
        self.window_skip_reasons: dict[str, int] = {}
        self.window_regime_counts: dict[str, int] = {}
        self.window_dedup_repeat_intervals_sec: list[float] = []
        self.prev_realized_usd = 0.0
        self.prev_closed = 0
        self.baseline_thresholds = self._snapshot_thresholds()
        self.zero_open_streak = 0

    @staticmethod
    def _clamp_int(value: int, lower: int, upper: int) -> int:
        return max(lower, min(upper, value))

    @staticmethod
    def _clamp_float(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _snapshot_thresholds(self) -> tuple[int, float, int, float, float, float, float]:
        return (
            int(getattr(config, "MIN_TOKEN_SCORE", 70)),
            float(getattr(config, "SAFE_MIN_VOLUME_5M_USD", 1200.0)),
            int(getattr(config, "HEAVY_CHECK_DEDUP_TTL_SECONDS", 900)),
            float(getattr(config, "MIN_EXPECTED_EDGE_PERCENT", 2.0)),
            float(getattr(config, "PROFIT_LOCK_FLOOR_PERCENT", 4.0)),
            float(getattr(config, "NO_MOMENTUM_EXIT_MAX_PNL_PERCENT", 0.3)),
            float(getattr(config, "WEAKNESS_EXIT_PNL_PERCENT", -9.0)),
        )

    @staticmethod
    def _step_by_pct(value: float, pct: float, fallback_abs: float) -> float:
        pct = max(0.0, float(pct))
        if pct <= 0.0:
            return max(1.0, float(fallback_abs))
        return max(1.0, float(value) * (pct / 100.0))

    @staticmethod
    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        vals = sorted(float(v) for v in values if float(v) >= 0.0)
        if not vals:
            return 0.0
        pct = max(0.0, min(100.0, float(p)))
        idx = int(round((pct / 100.0) * float(len(vals) - 1)))
        idx = max(0, min(len(vals) - 1, idx))
        return float(vals[idx])

    @staticmethod
    def _source_pressure(source_stats: dict[str, dict[str, int | float]]) -> bool:
        for row in (source_stats or {}).values():
            if int((row or {}).get("rate_limited", 0) or 0) > 0:
                return True
            if int((row or {}).get("limiter_waits", 0) or 0) > 0:
                return True
            if int((row or {}).get("cooldown_waits", 0) or 0) > 0:
                return True
            if int((row or {}).get("cooldown_active", 0) or 0) > 0:
                return True
        return False

    def _apply_thresholds(
        self,
        score_min: int,
        safe_volume_5m_min: float,
        dedup_ttl: int,
        edge_min_percent: float,
        profit_lock_floor: float,
        no_momentum_max_pnl: float,
        weakness_pnl: float,
    ) -> None:
        setattr(config, "MIN_TOKEN_SCORE", int(score_min))
        setattr(config, "SAFE_MIN_VOLUME_5M_USD", float(safe_volume_5m_min))
        setattr(config, "HEAVY_CHECK_DEDUP_TTL_SECONDS", int(dedup_ttl))
        setattr(config, "MIN_EXPECTED_EDGE_PERCENT", float(edge_min_percent))
        setattr(config, "PROFIT_LOCK_FLOOR_PERCENT", float(profit_lock_floor))
        setattr(config, "NO_MOMENTUM_EXIT_MAX_PNL_PERCENT", float(no_momentum_max_pnl))
        setattr(config, "WEAKNESS_EXIT_PNL_PERCENT", float(weakness_pnl))

    def record_cycle(
        self,
        *,
        candidates: int,
        opened: int,
        filter_fails_cycle: dict[str, int],
        skip_reasons_cycle: dict[str, int],
        market_regime: str,
        dedup_repeat_intervals_cycle: list[float] | None = None,
    ) -> None:
        self.window_cycles += 1
        self.window_candidates += int(candidates)
        self.window_opened += int(opened)
        for reason, count in (filter_fails_cycle or {}).items():
            key = str(reason or "unknown").strip().lower() or "unknown"
            self.window_filter_fails[key] = int(self.window_filter_fails.get(key, 0)) + int(count or 0)
        for reason, count in (skip_reasons_cycle or {}).items():
            key = str(reason or "unknown").strip().lower() or "unknown"
            self.window_skip_reasons[key] = int(self.window_skip_reasons.get(key, 0)) + int(count or 0)
        regime_key = str(market_regime or "unknown").strip().upper() or "UNKNOWN"
        self.window_regime_counts[regime_key] = int(self.window_regime_counts.get(regime_key, 0)) + 1
        if dedup_repeat_intervals_cycle:
            for v in dedup_repeat_intervals_cycle:
                fv = float(v or 0.0)
                if 0.0 < fv <= 300.0:
                    self.window_dedup_repeat_intervals_sec.append(fv)

    def _reset_window(self, *, now_ts: float, realized_usd: float, closed: int) -> None:
        self.last_eval_ts = now_ts
        self.window_cycles = 0
        self.window_candidates = 0
        self.window_opened = 0
        self.window_filter_fails = {}
        self.window_skip_reasons = {}
        self.window_regime_counts = {}
        self.window_dedup_repeat_intervals_sec = []
        self.prev_realized_usd = float(realized_usd)
        self.prev_closed = int(closed)

    def maybe_adapt(
        self,
        *,
        policy_state: str,
        auto_stats: dict[str, float | int],
        source_stats: dict[str, dict[str, int | float]],
    ) -> None:
        if not self.enabled:
            return
        if self.mode not in {"dry_run", "apply"}:
            return
        if self.paper_only and not bool(getattr(config, "AUTO_TRADE_PAPER", False)):
            return

        now_ts = time.time()
        if (now_ts - float(self.last_eval_ts)) < float(self.interval_seconds):
            return
        if int(self.window_cycles) < int(self.min_window_cycles):
            return

        if now_ts < float(self.cooldown_until_ts):
            rem = max(0.0, float(self.cooldown_until_ts) - now_ts)
            logger.warning(
                "ADAPTIVE_FILTERS mode=%s action=hold changed=False reason=cooldown_active remaining=%.1fs",
                self.mode,
                rem,
            )
            self._reset_window(
                now_ts=now_ts,
                realized_usd=float(auto_stats.get("realized_pnl_usd", 0.0) or 0.0),
                closed=int(auto_stats.get("closed", 0) or 0),
            )
            return

        (
            score_now,
            volume_now,
            ttl_now,
            edge_now,
            lock_floor_now,
            no_mom_now,
            weakness_now,
        ) = self._snapshot_thresholds()
        score_next, volume_next, ttl_next, edge_next = score_now, volume_now, ttl_now, edge_now
        lock_floor_next, no_mom_next, weakness_next = lock_floor_now, no_mom_now, weakness_now
        action = "hold"
        reason = "steady_state"

        avg_candidates = float(self.window_candidates) / max(1, int(self.window_cycles))
        avg_opened = float(self.window_opened) / max(1, int(self.window_cycles))
        fail_score = int(self.window_filter_fails.get("score_min", 0))
        fail_volume = int(self.window_filter_fails.get("safe_volume", 0))
        fail_dedup = int(self.window_filter_fails.get("heavy_dedup_ttl", 0))
        skip_negative_edge = int(self.window_skip_reasons.get("negative_edge", 0))
        skip_edge_usd_low = int(self.window_skip_reasons.get("edge_usd_low", 0))
        skip_edge_low = int(self.window_skip_reasons.get("edge_low", 0))
        skip_min_trade = int(self.window_skip_reasons.get("min_trade_size", 0))
        regime_top = _format_top_counts(self.window_regime_counts, limit=1)

        closed_now = int(auto_stats.get("closed", 0) or 0)
        realized_now = float(auto_stats.get("realized_pnl_usd", 0.0) or 0.0)
        closed_delta = int(closed_now - int(self.prev_closed))
        realized_delta = float(realized_now - float(self.prev_realized_usd))
        has_pnl_signal = closed_delta >= int(self.pnl_min_closed)
        pressure = self._source_pressure(source_stats)
        vol_step = self._step_by_pct(volume_now, self.volume_step_pct, self.volume_step)
        ttl_step = self._step_by_pct(float(ttl_now), self.ttl_step_pct, float(self.ttl_step))
        edge_step = self._step_by_pct(float(edge_now), self.edge_step_pct, float(self.edge_step))
        ttl_min_bound = int(self.ttl_min_bound)
        ttl_max_bound = int(self.ttl_max_bound)
        dedup_dyn_p = 0.0
        dedup_dyn_target = 0
        if (
            self.dedup_dynamic_enabled
            and len(self.window_dedup_repeat_intervals_sec) >= int(self.dedup_dynamic_min_samples)
        ):
            dedup_dyn_p = self._percentile(self.window_dedup_repeat_intervals_sec, self.dedup_dynamic_target_percentile)
            raw_target = dedup_dyn_p * max(0.8, float(self.dedup_dynamic_factor))
            dyn_floor = max(1, min(int(self.dedup_dynamic_min), int(self.dedup_dynamic_max)))
            dyn_ceil = max(dyn_floor, int(self.dedup_dynamic_max))
            dedup_dyn_target = self._clamp_int(int(round(raw_target)), dyn_floor, dyn_ceil)
            ttl_min_bound = max(dyn_floor, self._clamp_int(dedup_dyn_target - 2, dyn_floor, dyn_ceil))
            ttl_max_bound = max(ttl_min_bound, self._clamp_int(dedup_dyn_target + 8, dyn_floor, dyn_ceil))
            if ttl_now < ttl_min_bound:
                ttl_next = ttl_min_bound
            elif ttl_now > ttl_max_bound:
                ttl_next = ttl_max_bound
        edge_mode = str(getattr(config, "EDGE_FILTER_MODE", "usd") or "usd").strip().lower()
        edge_knob_movable = self.edge_enabled and bool(getattr(config, "EDGE_FILTER_ENABLED", True))
        edge_negative_blocked = (skip_negative_edge + skip_edge_low) > 0 and edge_mode in {"percent", "both"}
        if str(policy_state).upper() == "OK" and int(self.window_opened) <= 0 and avg_candidates >= float(self.zero_open_min_candidates):
            self.zero_open_streak += 1
        elif int(self.window_opened) > 0:
            self.zero_open_streak = 0

        if self.exit_enabled:
            if realized_delta <= -abs(float(self.neg_realized_trigger)) and closed_delta >= int(self.neg_closed_min):
                lock_floor_next = self._clamp_float(
                    lock_floor_now + float(self.profit_lock_floor_step),
                    self.profit_lock_floor_min,
                    self.profit_lock_floor_max,
                )
                no_mom_next = self._clamp_float(
                    no_mom_now + float(self.no_mom_pnl_step),
                    self.no_mom_pnl_min,
                    self.no_mom_pnl_max,
                )
                weakness_next = self._clamp_float(
                    weakness_now + float(self.weakness_pnl_step),
                    self.weakness_pnl_min,
                    self.weakness_pnl_max,
                )
            elif has_pnl_signal and realized_delta > 0:
                lock_floor_next = self._clamp_float(
                    lock_floor_now - float(self.profit_lock_floor_step),
                    self.profit_lock_floor_min,
                    self.profit_lock_floor_max,
                )
                no_mom_next = self._clamp_float(
                    no_mom_now - float(self.no_mom_pnl_step),
                    self.no_mom_pnl_min,
                    self.no_mom_pnl_max,
                )
                weakness_next = self._clamp_float(
                    weakness_now - float(self.weakness_pnl_step),
                    self.weakness_pnl_min,
                    self.weakness_pnl_max,
                )

        if str(policy_state).upper() != "OK":
            action = "hold"
            reason = f"policy_{policy_state.lower()}"
        elif pressure and ttl_now < ttl_max_bound:
            ttl_next = self._clamp_int(int(round(ttl_now + ttl_step)), ttl_min_bound, ttl_max_bound)
            action = "tighten_dedup_ttl"
            reason = "source_pressure_detected"
        elif (
            has_pnl_signal
            and closed_delta >= int(self.neg_closed_min)
            and realized_delta <= -abs(float(self.neg_realized_trigger))
        ):
            score_next = self._clamp_int(score_now + int(self.score_step), self.score_min_bound, self.score_max_bound)
            volume_next = self._clamp_float(volume_now + float(vol_step), self.volume_min_bound, self.volume_max_bound)
            ttl_next = self._clamp_int(int(round(ttl_now + ttl_step)), ttl_min_bound, ttl_max_bound)
            if edge_knob_movable and edge_now < self.edge_max_bound:
                edge_next = self._clamp_float(edge_now + float(edge_step), self.edge_min_bound, self.edge_max_bound)
            action = "tighten_all"
            reason = f"negative_window realized_delta=${realized_delta:.2f} closed_delta={closed_delta}"
        elif (
            self.zero_open_reset_enabled
            and self.mode == "apply"
            and self.zero_open_streak >= int(max(1, self.zero_open_windows_before_reset))
            and avg_candidates >= float(self.zero_open_min_candidates)
        ):
            base_score, base_volume, base_ttl, base_edge, _, _, _ = self.baseline_thresholds
            target_score = self._clamp_int(min(score_now, int(base_score)), self.score_min_bound, self.score_max_bound)
            if target_score == score_now and score_now > self.score_min_bound:
                target_score = self._clamp_int(score_now - int(self.score_step), self.score_min_bound, self.score_max_bound)

            target_volume = self._clamp_float(min(volume_now, float(base_volume)), self.volume_min_bound, self.volume_max_bound)
            if abs(target_volume - volume_now) < 0.5 and volume_now > self.volume_min_bound:
                target_volume = self._clamp_float(volume_now - float(vol_step), self.volume_min_bound, self.volume_max_bound)

            target_ttl = self._clamp_int(min(ttl_now, int(base_ttl)), ttl_min_bound, ttl_max_bound)
            if target_ttl == ttl_now and self.dedup_relax_enabled and ttl_now > ttl_min_bound:
                target_ttl = self._clamp_int(int(round(ttl_now - ttl_step)), ttl_min_bound, ttl_max_bound)

            target_edge = edge_now
            if edge_knob_movable:
                target_edge = self._clamp_float(min(edge_now, float(base_edge)), self.edge_min_bound, self.edge_max_bound)
                if abs(target_edge - edge_now) < 0.01 and edge_now > self.edge_min_bound:
                    target_edge = self._clamp_float(edge_now - float(edge_step), self.edge_min_bound, self.edge_max_bound)

            score_next = target_score
            volume_next = target_volume
            ttl_next = target_ttl
            edge_next = target_edge
            action = "anti_stall_reset"
            reason = (
                f"zero_open_windows={self.zero_open_streak}/{self.zero_open_windows_before_reset} "
                f"avg_cand={avg_candidates:.2f}"
            )
        elif (
            edge_knob_movable
            and edge_negative_blocked
            and avg_candidates >= float(self.target_cand_min)
            and avg_opened < float(self.target_open_min)
            and edge_now > self.edge_min_bound
        ):
            edge_next = self._clamp_float(edge_now - float(edge_step), self.edge_min_bound, self.edge_max_bound)
            action = "loosen_edge"
            reason = (
                f"blocked_by_edge avg_cand={avg_candidates:.2f} "
                f"neg_edge={skip_negative_edge + skip_edge_low}"
            )
        elif avg_candidates < float(self.target_cand_min) and avg_opened < float(self.target_open_min):
            if fail_volume >= fail_score and volume_now > self.volume_min_bound:
                volume_next = self._clamp_float(volume_now - float(vol_step), self.volume_min_bound, self.volume_max_bound)
                action = "loosen_volume"
                reason = f"low_flow avg_cand={avg_candidates:.2f} fail_safe_volume={fail_volume}"
            elif fail_score > 0 and score_now > self.score_min_bound:
                score_next = self._clamp_int(score_now - int(self.score_step), self.score_min_bound, self.score_max_bound)
                action = "loosen_score"
                reason = f"low_flow avg_cand={avg_candidates:.2f} fail_score_min={fail_score}"
            elif self.dedup_relax_enabled and fail_dedup > 0 and ttl_now > ttl_min_bound:
                ttl_next = self._clamp_int(int(round(ttl_now - ttl_step)), ttl_min_bound, ttl_max_bound)
                action = "loosen_dedup_ttl"
                reason = f"low_flow avg_cand={avg_candidates:.2f} fail_heavy_dedup={fail_dedup}"
            else:
                action = "hold"
                reason = f"low_flow_no_movable_knob avg_cand={avg_candidates:.2f}"
        elif avg_candidates > float(self.target_cand_max):
            score_next = self._clamp_int(score_now + int(self.score_step), self.score_min_bound, self.score_max_bound)
            if edge_knob_movable and edge_now < self.edge_max_bound:
                edge_next = self._clamp_float(edge_now + float(edge_step), self.edge_min_bound, self.edge_max_bound)
            action = "tighten_score"
            reason = f"too_many_candidates avg_cand={avg_candidates:.2f}"

        changed = (
            (score_next != score_now)
            or (abs(volume_next - volume_now) >= 0.5)
            or (ttl_next != ttl_now)
            or (abs(edge_next - edge_now) >= 0.01)
            or (abs(lock_floor_next - lock_floor_now) >= 0.01)
            or (abs(no_mom_next - no_mom_now) >= 0.01)
            or (abs(weakness_next - weakness_now) >= 0.01)
        )
        if self.mode == "apply" and changed:
            self._apply_thresholds(
                score_next,
                volume_next,
                ttl_next,
                edge_next,
                lock_floor_next,
                no_mom_next,
                weakness_next,
            )
            self.cooldown_until_ts = now_ts + (max(0, self.cooldown_windows) * max(1, self.interval_seconds))
            if action == "anti_stall_reset":
                self.zero_open_streak = 0

        logger.warning(
            "ADAPTIVE_FILTERS mode=%s action=%s changed=%s reason=%s regime=%s avg_cand=%.2f avg_opened=%.2f closed_delta=%s realized_delta=$%.2f pnl_signal=%s pressure=%s score=%s->%s volume=%.0f->%.0f dedup_ttl=%s->%s dyn_dedup(p=%.1f,target=%s,range=%s-%s,samples=%s) edge=%.2f->%.2f lock_floor=%.2f->%.2f no_mom_pnl=%.2f->%.2f weakness_pnl=%.2f->%.2f fails(score=%s,volume=%s,dedup=%s) skips(neg_edge=%s,edge_usd=%s,min_trade=%s) zero_open=%s/%s",
            self.mode,
            action,
            changed,
            reason,
            regime_top,
            avg_candidates,
            avg_opened,
            closed_delta,
            realized_delta,
            has_pnl_signal,
            pressure,
            score_now,
            score_next,
            volume_now,
            volume_next,
            ttl_now,
            ttl_next,
            dedup_dyn_p,
            dedup_dyn_target,
            ttl_min_bound,
            ttl_max_bound,
            len(self.window_dedup_repeat_intervals_sec),
            edge_now,
            edge_next,
            lock_floor_now,
            lock_floor_next,
            no_mom_now,
            no_mom_next,
            weakness_now,
            weakness_next,
            fail_score,
            fail_volume,
            fail_dedup,
            (skip_negative_edge + skip_edge_low),
            skip_edge_usd_low,
            skip_min_trade,
            self.zero_open_streak,
            self.zero_open_windows_before_reset,
        )

        self._reset_window(now_ts=now_ts, realized_usd=realized_now, closed=closed_now)


class CandidateDecisionWriter:
    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "CANDIDATE_DECISIONS_LOG_ENABLED", True))
        raw = str(getattr(config, "CANDIDATE_DECISIONS_LOG_FILE", os.path.join("logs", "candidates.jsonl")) or "").strip()
        if not raw:
            raw = os.path.join("logs", "candidates.jsonl")
        self.path = raw if os.path.isabs(raw) else os.path.abspath(os.path.join(PROJECT_ROOT, raw))

    def write(self, event: dict[str, object]) -> None:
        if not self.enabled:
            return
        try:
            event = dict(event)
            event.setdefault("run_tag", RUN_TAG)
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, sort_keys=False) + "\n")
        except Exception:
            logger.exception("CANDIDATE_LOG write failed")


class MiniAnalyzer:
    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "MINI_ANALYZER_ENABLED", True))
        self.interval_seconds = int(getattr(config, "MINI_ANALYZER_INTERVAL_SECONDS", 900) or 900)
        self.last_emit_ts = time.time()
        self.cycles = 0
        self.scanned = 0
        self.candidates = 0
        self.opened = 0
        self.filter_reasons: dict[str, int] = {}
        self.skip_reasons: dict[str, int] = {}
        self.policy_counts: dict[str, int] = {}
        self.regime_counts: dict[str, int] = {}
        self.prev_closed = 0
        self.prev_realized = 0.0

    def prime(self, *, closed: int, realized_usd: float) -> None:
        self.prev_closed = int(closed)
        self.prev_realized = float(realized_usd)

    def record(
        self,
        *,
        scanned: int,
        candidates: int,
        opened: int,
        filter_fails_cycle: dict[str, int],
        skip_reasons_cycle: dict[str, int],
        policy_state: str,
        market_regime: str,
    ) -> None:
        if not self.enabled:
            return
        self.cycles += 1
        self.scanned += int(scanned)
        self.candidates += int(candidates)
        self.opened += int(opened)
        pkey = str(policy_state or "UNKNOWN").upper()
        self.policy_counts[pkey] = int(self.policy_counts.get(pkey, 0)) + 1
        rkey = str(market_regime or "UNKNOWN").upper()
        self.regime_counts[rkey] = int(self.regime_counts.get(rkey, 0)) + 1
        for reason, count in (filter_fails_cycle or {}).items():
            key = str(reason or "unknown").strip().lower() or "unknown"
            self.filter_reasons[key] = int(self.filter_reasons.get(key, 0)) + int(count or 0)
        for reason, count in (skip_reasons_cycle or {}).items():
            key = str(reason or "unknown").strip().lower() or "unknown"
            self.skip_reasons[key] = int(self.skip_reasons.get(key, 0)) + int(count or 0)

    def maybe_emit(self, *, auto_stats: dict[str, float | int]) -> None:
        if not self.enabled:
            return
        now_ts = time.time()
        if (now_ts - float(self.last_emit_ts)) < float(self.interval_seconds):
            return
        if self.cycles <= 0:
            self.last_emit_ts = now_ts
            return

        closed_now = int(auto_stats.get("closed", 0) or 0)
        realized_now = float(auto_stats.get("realized_pnl_usd", 0.0) or 0.0)
        closed_delta = max(0, closed_now - int(self.prev_closed))
        realized_delta = realized_now - float(self.prev_realized)

        cand_per_cycle = float(self.candidates) / max(1, int(self.cycles))
        opened_per_cycle = float(self.opened) / max(1, int(self.cycles))
        conv = (float(self.opened) / float(self.candidates) * 100.0) if self.candidates > 0 else 0.0

        logger.warning(
            "MINI_ANALYZER window=%ss cycles=%s scanned=%s cand=%s opened=%s cand_per_cycle=%.2f open_per_cycle=%.2f conv=%.1f%% closed_delta=%s realized_delta=$%.2f policy_top=%s regime_top=%s filter_top=%s skips_top=%s",
            int(self.interval_seconds),
            self.cycles,
            self.scanned,
            self.candidates,
            self.opened,
            cand_per_cycle,
            opened_per_cycle,
            conv,
            closed_delta,
            realized_delta,
            _format_top_counts(self.policy_counts),
            _format_top_counts(self.regime_counts),
            _format_top_counts(self.filter_reasons),
            _format_top_counts(self.skip_reasons),
        )

        self.last_emit_ts = now_ts
        self.cycles = 0
        self.scanned = 0
        self.candidates = 0
        self.opened = 0
        self.filter_reasons = {}
        self.skip_reasons = {}
        self.policy_counts = {}
        self.regime_counts = {}
        self.prev_closed = closed_now
        self.prev_realized = realized_now


class InstanceLock:
    def __init__(self) -> None:
        self._handle = None
        self._lock_fh = None
        self.acquired = False

    def _acquire_lock_file(self) -> bool:
        lock_name = f"main_local.{INSTANCE_ID}.lock" if INSTANCE_ID else "main_local.lock"
        lock_file = os.path.join(PROJECT_ROOT, lock_name)
        try:
            fh = open(lock_file, "a+b")
        except OSError:
            return False

        try:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"1")
                fh.flush()
            fh.seek(0)

            if os.name == "nt" and msvcrt is not None:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                pass

            # Human-friendly: store the PID in the lock file (lock is still held via the open file handle).
            try:
                fh.seek(0)
                fh.truncate(0)
                fh.write(f"{os.getpid()}\n".encode("ascii", errors="ignore"))
                fh.flush()
                fh.seek(0)
            except Exception:
                pass
        except OSError:
            try:
                fh.close()
            except Exception:
                pass
            return False

        self._lock_fh = fh
        return True

    def acquire(self) -> bool:
        # Prefer a real file lock: robust across admin/non-admin tokens, no PID reuse issues.
        if not self._acquire_lock_file():
            return False

        # NOTE: When using ctypes, prefer get_last_error() with use_last_error=True.
        # Calling kernel32.GetLastError() directly is unreliable here and can fail to detect duplicates.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        mutex = kernel32.CreateMutexW(None, False, WINDOWS_MUTEX_NAME)
        if not mutex:
            return False
        error_already_exists = 183
        if ctypes.get_last_error() == error_already_exists:
            kernel32.CloseHandle(mutex)
            return False
        self._handle = mutex
        self.acquired = True
        return True

    def release(self) -> None:
        if not self.acquired:
            return
        if self._lock_fh is not None:
            try:
                if os.name == "nt" and msvcrt is not None:
                    self._lock_fh.seek(0)
                    msvcrt.locking(self._lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            try:
                self._lock_fh.close()
            except Exception:
                pass
            self._lock_fh = None
        if self._handle:
            ctypes.windll.kernel32.CloseHandle(self._handle)
            self._handle = None
        self.acquired = False


def configure_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    session_logs_dir = os.path.join(LOG_DIR, "sessions")
    os.makedirs(session_logs_dir, exist_ok=True)
    class _RunTagFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            setattr(record, "run_tag", RUN_TAG)
            return True

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(run_tag)s] %(name)s: %(message)s")

    file_handler = RotatingFileHandler(APP_LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    out_handler = RotatingFileHandler(OUT_LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    out_handler.setFormatter(formatter)
    session_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_log_file = os.path.join(session_logs_dir, f"main_local_{session_stamp}_pid{os.getpid()}.log")
    session_handler = logging.FileHandler(session_log_file, encoding="utf-8")
    session_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(out_handler)
    root.addHandler(session_handler)
    root.addHandler(console_handler)
    root.addFilter(_RunTagFilter())

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger(__name__).info("SESSION_LOG file=%s", session_log_file)


async def run_local_loop() -> None:
    dex_monitor = DexScreenerMonitor()
    watchlist_monitor = WatchlistMonitor()
    onchain_monitor: OnChainFactoryMonitor | None = None
    onchain_error_streak = 0
    fallback_until_ts = 0.0

    if str(config.SIGNAL_SOURCE).lower() == "onchain":
        try:
            onchain_monitor = OnChainFactoryMonitor()
        except Exception as exc:
            logger.warning("On-chain source init failed (%s). Falling back to DexScreener.", exc)
            onchain_monitor = None

    scorer = TokenScorer()
    checker = TokenChecker()
    local_alerter = LocalAlerter(os.path.join(LOG_DIR, "local_alerts.jsonl"))
    auto_trader = AutoTrader()
    candidate_writer = CandidateDecisionWriter()
    auto_trader.set_data_policy("OK", "init_reset_manual")
    logger.info("AUTOTRADER_INIT state=ready reason=manual mode=%s", ("paper" if bool(config.AUTO_TRADE_PAPER) else "live"))
    cycle_times = deque(maxlen=120)
    last_rss_log_ts = 0.0
    policy_mode_current = "OK"
    policy_bad_streak = 0
    policy_good_streak = 0
    heavy_seen_until: dict[str, float] = {}
    heavy_last_liquidity: dict[str, float] = {}
    heavy_last_volume_5m: dict[str, float] = {}
    heavy_last_seen_ts: dict[str, float] = {}
    filter_fail_reasons_session: dict[str, int] = {}
    adaptive_filters = AdaptiveFilterController()
    adaptive_init_stats = auto_trader.get_stats()
    adaptive_filters.prev_closed = int(adaptive_init_stats.get("closed", 0) or 0)
    adaptive_filters.prev_realized_usd = float(adaptive_init_stats.get("realized_pnl_usd", 0.0) or 0.0)
    mini_analyzer = MiniAnalyzer()
    mini_analyzer.prime(
        closed=int(adaptive_init_stats.get("closed", 0) or 0),
        realized_usd=float(adaptive_init_stats.get("realized_pnl_usd", 0.0) or 0.0),
    )
    recent_candidate_counts: deque[int] = deque(
        maxlen=max(3, int(getattr(config, "MARKET_REGIME_WINDOW_CYCLES", 12) or 12))
    )
    market_regime_current = "BALANCED"
    market_regime_reason = "init"
    cycle_index = 0

    logger.info("Local mode started (Telegram disabled).")
    try:
        while True:
            cycle_index += 1
            if _graceful_stop_requested():
                logger.warning("GRACEFUL_STOP requested path=%s", _graceful_stop_file_path())
                break
            cycle_started = time.perf_counter()
            try:
                source_mode = "dexscreener"
                now_ts = time.time()
                if str(config.SIGNAL_SOURCE).lower() == "onchain" and onchain_monitor is not None:
                    if now_ts < fallback_until_ts:
                        source_mode = "dexscreener_fallback"
                        try:
                            await onchain_monitor.advance_cursor_only()
                        except OnChainRPCError as exc:
                            logger.warning("On-chain cursor sync failed during fallback: %s", exc)
                        tokens = await dex_monitor.fetch_new_tokens()
                    else:
                        try:
                            source_mode = "onchain"
                            if config.ONCHAIN_PARALLEL_MARKET_SOURCES:
                                onchain_task = asyncio.create_task(onchain_monitor.fetch_new_tokens())
                                dex_task = asyncio.create_task(dex_monitor.fetch_new_tokens())
                                onchain_tokens, dex_tokens = await asyncio.gather(onchain_task, dex_task)
                                tokens = _merge_token_streams(onchain_tokens, dex_tokens)
                                source_mode = "onchain+market"
                            else:
                                tokens = await onchain_monitor.fetch_new_tokens()
                            onchain_error_streak = 0
                        except OnChainRPCError as exc:
                            onchain_error_streak += 1
                            logger.warning(
                                "On-chain source error streak=%s err=%s",
                                onchain_error_streak,
                                exc,
                            )
                            if onchain_error_streak >= 3:
                                fallback_until_ts = now_ts + 60
                                onchain_error_streak = 0
                                logger.warning("On-chain source fallback enabled for 60 seconds.")
                            source_mode = "dexscreener_fallback"
                            tokens = await dex_monitor.fetch_new_tokens()
                else:
                    tokens = await dex_monitor.fetch_new_tokens()

                # Stability flow: dynamic watchlist tokens (vetted) in parallel to new-pairs sources.
                if bool(getattr(config, "WATCHLIST_ENABLED", False)):
                    try:
                        wl = await watchlist_monitor.fetch_tokens()
                        if wl:
                            tokens = _merge_token_streams(tokens or [], wl)
                            if wl:
                                logger.info("WATCHLIST merged count=%s", len(wl))
                    except Exception as exc:
                        logger.warning("WATCHLIST fetch failed: %s", exc)

                high_quality = 0
                alerts_sent = 0
                trade_candidates: list[tuple[dict, dict]] = []
                safety_checked = 0
                safety_fail_closed_hits = 0
                heavy_dedup_skipped = 0
                heavy_dedup_override = 0
                dedup_repeat_intervals_cycle: list[float] = []
                excluded_addresses = _excluded_trade_addresses()
                now_mono = time.monotonic()
                if heavy_seen_until:
                    expired = [a for a, ts in heavy_seen_until.items() if float(ts or 0.0) <= now_mono]
                    for a in expired:
                        heavy_seen_until.pop(a, None)
                        heavy_last_liquidity.pop(a, None)
                        heavy_last_volume_5m.pop(a, None)
                if tokens:
                    cycle_filter_fails: dict[str, int] = {}
                    regime_score_delta, regime_volume_mult, regime_edge_mult = _regime_entry_overrides(market_regime_current)

                    def _filter_fail(reason: str, token: dict, extra: str = "") -> None:
                        key = str(reason or "unknown").strip().lower() or "unknown"
                        quality = _candidate_quality_features(token, (token.get("score_data") or {}))
                        filter_fail_reasons_session[key] = int(filter_fail_reasons_session.get(key, 0)) + 1
                        cycle_filter_fails[key] = int(cycle_filter_fails.get(key, 0)) + 1
                        candidate_writer.write(
                            {
                                "ts": time.time(),
                                "cycle_index": cycle_index,
                                "source_mode": source_mode,
                                "decision_stage": "filter_fail",
                                "decision": "skip",
                                "reason": key,
                                "extra": extra,
                                "market_regime": market_regime_current,
                                "address": normalize_address(token.get("address", "")),
                                "symbol": str(token.get("symbol", "N/A")),
                                "score": int((token.get("score_data") or {}).get("score", 0) if isinstance(token.get("score_data"), dict) else 0),
                                "recommendation": str((token.get("score_data") or {}).get("recommendation", "") if isinstance(token.get("score_data"), dict) else ""),
                                "liquidity_usd": float(token.get("liquidity") or 0.0),
                                "volume_5m_usd": float(token.get("volume_5m") or 0.0),
                                "age_seconds": int(token.get("age_seconds") or 0),
                                "price_change_5m": float(token.get("price_change_5m") or 0.0),
                                "quality": quality,
                            }
                        )
                        if extra:
                            logger.info(
                                "FILTER_FAIL token=%s reason=%s %s",
                                token.get("symbol", "N/A"),
                                reason,
                                extra,
                            )
                        else:
                            logger.info(
                                "FILTER_FAIL token=%s reason=%s",
                                token.get("symbol", "N/A"),
                                reason,
                            )

                    for token in tokens:
                        token_address = normalize_address(token.get("address", ""))
                        if token_address:
                            seen_now = time.monotonic()
                            prev_seen = float(heavy_last_seen_ts.get(token_address, 0.0) or 0.0)
                            if prev_seen > 0.0:
                                delta = float(seen_now - prev_seen)
                                if 0.0 < delta <= 300.0:
                                    dedup_repeat_intervals_cycle.append(delta)
                            heavy_last_seen_ts[token_address] = seen_now
                        if token_address in excluded_addresses:
                            _filter_fail("excluded_base_token", token)
                            continue
                        if not _belongs_to_candidate_shard(token_address):
                            _filter_fail("shard_skip", token)
                            continue
                        score_data = scorer.calculate_score(token)
                        token["score_data"] = score_data
                        if int(score_data.get("score", 0)) >= 70:
                            high_quality += 1
                        score_min_base = int(config.MIN_TOKEN_SCORE)
                        score_min_regime = max(0, score_min_base + int(regime_score_delta))
                        if config.AUTO_FILTER_ENABLED and int(score_data.get("score", 0)) < int(score_min_regime):
                            _filter_fail(
                                "score_min",
                                token,
                                f"score={score_data.get('score', 0)} min={int(score_min_regime)} regime={market_regime_current}",
                            )
                            continue

                        if config.SAFE_TEST_MODE:
                            if float(token.get("liquidity") or 0) < float(config.SAFE_MIN_LIQUIDITY_USD):
                                _filter_fail("safe_liquidity", token)
                                continue
                            safe_volume_floor = float(config.SAFE_MIN_VOLUME_5M_USD) * max(0.1, float(regime_volume_mult))
                            if float(token.get("volume_5m") or 0) < safe_volume_floor:
                                _filter_fail(
                                    "safe_volume",
                                    token,
                                    f"vol5m={float(token.get('volume_5m') or 0):.0f} min={safe_volume_floor:.0f} regime={market_regime_current}",
                                )
                                continue
                            if int(token.get("age_seconds") or 0) < int(config.SAFE_MIN_AGE_SECONDS):
                                _filter_fail("safe_age", token)
                                continue
                            if abs(float(token.get("price_change_5m") or 0)) > float(config.SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT):
                                _filter_fail("safe_change_5m", token)
                                continue

                        # Heavy safety-check dedup: avoid re-checking same token address too often.
                        heavy_ttl = int(getattr(config, "HEAVY_CHECK_DEDUP_TTL_SECONDS", 900) or 900)
                        if heavy_ttl > 0 and token_address:
                            now_mono = time.monotonic()
                            until = float(heavy_seen_until.get(token_address, 0.0) or 0.0)
                            token_liquidity = float(token.get("liquidity") or 0.0)
                            token_volume_5m = float(token.get("volume_5m") or 0.0)
                            prev_liquidity = float(heavy_last_liquidity.get(token_address, 0.0) or 0.0)
                            prev_volume_5m = float(heavy_last_volume_5m.get(token_address, 0.0) or 0.0)
                            override_mult = float(getattr(config, "HEAVY_CHECK_OVERRIDE_LIQ_MULT", 2.0) or 2.0)
                            override_mult = max(1.1, override_mult)
                            override_vol_mult = float(getattr(config, "HEAVY_CHECK_OVERRIDE_VOL_MULT", 3.0) or 3.0)
                            override_vol_mult = max(1.1, override_vol_mult)
                            override_vol_min_abs = float(
                                getattr(config, "HEAVY_CHECK_OVERRIDE_VOL_MIN_ABS_USD", 500.0) or 500.0
                            )
                            override_vol_min_abs = max(0.0, override_vol_min_abs)
                            if until > now_mono:
                                liq_override = prev_liquidity > 0 and token_liquidity >= (prev_liquidity * override_mult)
                                vol_override = (
                                    prev_volume_5m > 0
                                    and token_volume_5m >= (prev_volume_5m * override_vol_mult)
                                    and token_volume_5m >= override_vol_min_abs
                                )
                                if liq_override or vol_override:
                                    heavy_dedup_override += 1
                                else:
                                    heavy_dedup_skipped += 1
                                    _filter_fail("heavy_dedup_ttl", token)
                                    continue

                        # Expensive safety API check only after cheap filters.
                        safety = await checker.check_token_safety(
                            token.get("address", ""),
                            token.get("liquidity", 0),
                        )
                        if heavy_ttl > 0 and token_address:
                            heavy_seen_until[token_address] = time.monotonic() + float(heavy_ttl)
                            heavy_last_liquidity[token_address] = float(token.get("liquidity") or 0.0)
                            heavy_last_volume_5m[token_address] = float(token.get("volume_5m") or 0.0)
                        safety_checked += 1
                        if str((safety or {}).get("source", "")).lower() == "fail_closed":
                            safety_fail_closed_hits += 1
                        token["safety"] = safety or {}
                        token["risk_level"] = str((safety or {}).get("risk_level", "HIGH")).upper()
                        token["warning_flags"] = int((safety or {}).get("warning_flags", 0))
                        token["is_contract_safe"] = bool((safety or {}).get("is_safe", False))

                        if config.SAFE_TEST_MODE:
                            if config.SAFE_REQUIRE_CONTRACT_SAFE and not bool(token.get("is_contract_safe", False)):
                                _filter_fail("safe_contract", token)
                                continue
                            required_risk = str(config.SAFE_REQUIRE_RISK_LEVEL).upper()
                            risk = str(token.get("risk_level", "HIGH")).upper()
                            rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
                            if rank.get(risk, 2) > rank.get(required_risk, 1):
                                _filter_fail("safe_risk", token)
                                continue
                            if int(token.get("warning_flags") or 0) > int(config.SAFE_MAX_WARNING_FLAGS):
                                _filter_fail("safe_warnings", token)
                                continue

                        token["_regime_edge_mult"] = float(regime_edge_mult)
                        token["_regime_name"] = str(market_regime_current)

                        logger.info(
                            "FILTER_PASS token=%s score=%s risk=%s liq=%.0f vol5m=%.0f",
                            token.get("symbol", "N/A"),
                            score_data.get("score", 0),
                            token.get("risk_level", "N/A"),
                            float(token.get("liquidity") or 0),
                            float(token.get("volume_5m") or 0),
                        )
                        candidate_writer.write(
                            {
                                "ts": time.time(),
                                "cycle_index": cycle_index,
                                "source_mode": source_mode,
                                "decision_stage": "post_filters",
                                "decision": "candidate_pass",
                                "reason": "passed_all_filters",
                                "market_regime": market_regime_current,
                                "address": token_address,
                                "symbol": str(token.get("symbol", "N/A")),
                                "score": int(score_data.get("score", 0)),
                                "recommendation": str(score_data.get("recommendation", "")),
                                "liquidity_usd": float(token.get("liquidity") or 0.0),
                                "volume_5m_usd": float(token.get("volume_5m") or 0.0),
                                "age_seconds": int(token.get("age_seconds") or 0),
                                "price_change_5m": float(token.get("price_change_5m") or 0.0),
                                "risk_level": str(token.get("risk_level", "")),
                                "warning_flags": int(token.get("warning_flags") or 0),
                                "is_contract_safe": bool(token.get("is_contract_safe", False)),
                                "quality": _candidate_quality_features(token, score_data),
                            }
                        )
                        trade_candidates.append((token, score_data))
                        # Optional: avoid alert spam from watchlist flow.
                        if str(token.get("source", "")).lower() == "watchlist" and not bool(
                            getattr(config, "WATCHLIST_ALERTS_ENABLED", False)
                        ):
                            pass
                        else:
                            alerts_sent += await local_alerter.send_alert(token, score_data, safety=safety)
                else:
                    cycle_filter_fails = {}

                dex_stats_all = dex_monitor.runtime_stats(reset=True)
                gecko_ingest_stats = dict(dex_stats_all.get("gecko_ingest", {}) or {})
                dex_stats = {k: v for k, v in dex_stats_all.items() if k != "gecko_ingest"}
                watch_stats = watchlist_monitor.runtime_stats(reset=True)
                onchain_stats = onchain_monitor.runtime_stats(reset=True) if onchain_monitor is not None else {}
                source_stats = _merge_source_stats(dex_stats, watch_stats, onchain_stats)
                safety_stats = checker.runtime_stats(reset=True)
                raw_policy_state, raw_policy_reason = _policy_state(
                    source_stats=source_stats,
                    safety_stats=safety_stats,
                )
                policy_state, policy_reason, policy_bad_streak, policy_good_streak = _apply_policy_hysteresis(
                    raw_mode=raw_policy_state,
                    raw_reason=raw_policy_reason,
                    current_mode=policy_mode_current,
                    bad_streak=policy_bad_streak,
                    good_streak=policy_good_streak,
                )
                recent_candidate_counts.append(len(trade_candidates))
                avg_candidates_recent = (
                    float(sum(recent_candidate_counts)) / float(len(recent_candidate_counts))
                    if recent_candidate_counts
                    else float(len(trade_candidates))
                )
                market_regime_current, market_regime_reason = _detect_market_regime(
                    policy_state=policy_state,
                    source_stats=source_stats,
                    safety_stats=safety_stats,
                    avg_candidates_recent=avg_candidates_recent,
                )
                policy_mode_current = policy_state
                auto_trader.set_data_policy(policy_state, policy_reason)
                if policy_state == "FAIL_CLOSED":
                    logger.warning(
                        "SAFETY_MODE fail_closed active; BUY disabled until safety API recovers. reason=%s",
                        policy_reason,
                    )
                elif policy_state == "DEGRADED":
                    logger.warning(
                        "DATA_MODE degraded; BUY disabled by policy. reason=%s",
                        policy_reason,
                    )
                logger.info(
                    "MARKET_REGIME mode=%s reason=%s avg_cand_recent=%.2f",
                    market_regime_current,
                    market_regime_reason,
                    avg_candidates_recent,
                )

                opened_trades = 0
                if trade_candidates and policy_state == "OK":
                    opened_trades = await auto_trader.plan_batch(trade_candidates)
                elif trade_candidates:
                    for token_data, score_data in trade_candidates:
                        candidate_writer.write(
                            {
                                "ts": time.time(),
                                "cycle_index": cycle_index,
                                "source_mode": source_mode,
                                "decision_stage": "policy_gate",
                                "decision": "skip",
                                "reason": f"policy_{str(policy_state).lower()}",
                                "policy_reason": str(policy_reason),
                                "market_regime": market_regime_current,
                                "market_regime_reason": market_regime_reason,
                                "address": normalize_address(token_data.get("address", "")),
                                "symbol": str(token_data.get("symbol", "N/A")),
                                "score": int(score_data.get("score", 0)),
                                "recommendation": str(score_data.get("recommendation", "")),
                                "quality": _candidate_quality_features(token_data, score_data),
                            }
                        )
                    logger.warning(
                        "AUTO_POLICY mode=%s action=no_buy reason=%s candidates=%s",
                        policy_state,
                        policy_reason,
                        len(trade_candidates),
                    )
                await auto_trader.process_open_positions(bot=None)
                auto_stats = auto_trader.get_stats()
                skip_reasons_cycle = auto_trader.pop_skip_reason_counts_window()
                adaptive_filters.record_cycle(
                    candidates=len(trade_candidates),
                    opened=opened_trades,
                    filter_fails_cycle=cycle_filter_fails,
                    skip_reasons_cycle=skip_reasons_cycle,
                    market_regime=market_regime_current,
                    dedup_repeat_intervals_cycle=dedup_repeat_intervals_cycle,
                )
                mini_analyzer.record(
                    scanned=len(tokens or []),
                    candidates=len(trade_candidates),
                    opened=opened_trades,
                    filter_fails_cycle=cycle_filter_fails,
                    skip_reasons_cycle=skip_reasons_cycle,
                    policy_state=policy_state,
                    market_regime=market_regime_current,
                )
                adaptive_filters.maybe_adapt(
                    policy_state=policy_state,
                    auto_stats=auto_stats,
                    source_stats=source_stats,
                )
                mini_analyzer.maybe_emit(auto_stats=auto_stats)

                active_tasks = len(asyncio.all_tasks())
                now_ts = time.time()
                rss_mb = 0.0
                if (now_ts - last_rss_log_ts) >= int(getattr(config, "METRICS_RSS_LOG_SECONDS", 900)):
                    last_rss_log_ts = now_ts
                    rss_mb = _rss_memory_mb()
                thresh_every = max(1, int(getattr(config, "FILTER_THRESH_LOG_EVERY_CYCLES", 30) or 30))
                if cycle_index == 1 or (cycle_index % thresh_every) == 0:
                    logger.info(
                        "FILTER_THRESHOLDS safe_volume_5m_min=$%.0f safe_liquidity_min=$%.0f safe_age_min=%ss score_min=%s edge_min=%.2f%% lock_floor=%.2f no_momentum_max_pnl=%.2f weakness_pnl=%.2f",
                        float(config.SAFE_MIN_VOLUME_5M_USD),
                        float(config.SAFE_MIN_LIQUIDITY_USD),
                        int(config.SAFE_MIN_AGE_SECONDS),
                        int(config.MIN_TOKEN_SCORE),
                        float(config.MIN_EXPECTED_EDGE_PERCENT),
                        float(config.PROFIT_LOCK_FLOOR_PERCENT),
                        float(config.NO_MOMENTUM_EXIT_MAX_PNL_PERCENT),
                        float(config.WEAKNESS_EXIT_PNL_PERCENT),
                    )

                logger.info(
                    "Scanned %s tokens | High quality: %s | Alerts sent: %s | Trade candidates: %s | Opened: %s | Mode: local | Source: %s | Policy: %s(%s) | Regime: %s(%s) | Safety: checked=%s fail_closed=%s reasons=%s | FiltersTop(session): %s | Dedup: heavy_skip=%s override=%s | Ingest: %s | Sources: %s | Tasks: %s | RSS: %.1fMB | CycleAvg: %.2fs",
                    len(tokens or []),
                    high_quality,
                    alerts_sent,
                    len(trade_candidates),
                    opened_trades,
                    source_mode,
                    policy_state,
                    policy_reason,
                    market_regime_current,
                    market_regime_reason,
                    safety_checked,
                    safety_fail_closed_hits,
                    _format_safety_reasons_brief(safety_stats),
                    _format_top_filter_reasons(filter_fail_reasons_session),
                    heavy_dedup_skipped,
                    heavy_dedup_override,
                    _format_ingest_stats_brief(gecko_ingest_stats),
                    _format_source_stats_brief(source_stats),
                    active_tasks,
                    rss_mb,
                    (sum(cycle_times) / len(cycle_times)) if cycle_times else 0.0,
                )
            except Exception:
                logger.exception("Local monitoring loop error")
            finally:
                cycle_times.append(max(0.0, time.perf_counter() - cycle_started))

            sleep_seconds = (
                max(1, int(config.ONCHAIN_POLL_INTERVAL_SECONDS))
                if str(config.SIGNAL_SOURCE).lower() == "onchain"
                else int(SCAN_INTERVAL)
            )
            await asyncio.sleep(sleep_seconds)
    finally:
        auto_trader.flush_state()
        await auto_trader.shutdown("main_local_shutdown")
        if onchain_monitor is not None:
            await onchain_monitor.close()
        await dex_monitor.close()
        await watchlist_monitor.close()
        await checker.close()
        await local_alerter.close()
        _clear_graceful_stop_flag()


def main() -> None:
    lock = InstanceLock()
    if not lock.acquire():
        print("Another main_local.py instance is already running. Exit.")
        return
    atexit.register(lock.release)
    configure_logging()
    _clear_graceful_stop_flag()
    try:
        asyncio.run(run_local_loop())
    finally:
        lock.release()


if __name__ == "__main__":
    main()
