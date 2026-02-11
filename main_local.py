"""Local fast runner without Telegram polling/webhooks."""

from __future__ import annotations

import asyncio
import atexit
import ctypes
import logging
import os
import time
from collections import deque
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
WINDOWS_MUTEX_NAME = "Global\\solana_alert_bot_main_local_single_instance"


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
                f"/err={float(row.get('error_percent', 0.0)):.1f}%"
                f"/avg={float(row.get('latency_avg_ms', 0.0)):.0f}ms"
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


class InstanceLock:
    def __init__(self) -> None:
        self._handle = None
        self._lock_fh = None
        self.acquired = False

    def _acquire_lock_file(self) -> bool:
        lock_file = os.path.join(PROJECT_ROOT, "main_local.lock")
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
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = RotatingFileHandler(APP_LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    out_handler = RotatingFileHandler(OUT_LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    out_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(out_handler)
    root.addHandler(console_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)


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
    auto_trader.set_data_policy("OK", "init_reset_manual")
    logger.info("AUTOTRADER_INIT state=ready reason=manual mode=%s", ("paper" if bool(config.AUTO_TRADE_PAPER) else "live"))
    cycle_times = deque(maxlen=120)
    last_rss_log_ts = 0.0
    policy_mode_current = "OK"
    policy_bad_streak = 0
    policy_good_streak = 0

    logger.info("Local mode started (Telegram disabled).")
    try:
        while True:
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
                excluded_addresses = _excluded_trade_addresses()
                if tokens:
                    for token in tokens:
                        token_address = normalize_address(token.get("address", ""))
                        if token_address in excluded_addresses:
                            logger.info(
                                "FILTER_FAIL token=%s reason=excluded_base_token",
                                token.get("symbol", "N/A"),
                            )
                            continue
                        score_data = scorer.calculate_score(token)
                        token["score_data"] = score_data
                        if int(score_data.get("score", 0)) >= 70:
                            high_quality += 1
                        if config.AUTO_FILTER_ENABLED and int(score_data.get("score", 0)) < int(config.MIN_TOKEN_SCORE):
                            logger.info(
                                "FILTER_FAIL token=%s reason=score_min score=%s min=%s",
                                token.get("symbol", "N/A"),
                                score_data.get("score", 0),
                                int(config.MIN_TOKEN_SCORE),
                            )
                            continue

                        if config.SAFE_TEST_MODE:
                            if float(token.get("liquidity") or 0) < float(config.SAFE_MIN_LIQUIDITY_USD):
                                logger.info("FILTER_FAIL token=%s reason=safe_liquidity", token.get("symbol", "N/A"))
                                continue
                            if float(token.get("volume_5m") or 0) < float(config.SAFE_MIN_VOLUME_5M_USD):
                                logger.info("FILTER_FAIL token=%s reason=safe_volume", token.get("symbol", "N/A"))
                                continue
                            if int(token.get("age_seconds") or 0) < int(config.SAFE_MIN_AGE_SECONDS):
                                logger.info("FILTER_FAIL token=%s reason=safe_age", token.get("symbol", "N/A"))
                                continue
                            if abs(float(token.get("price_change_5m") or 0)) > float(config.SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT):
                                logger.info("FILTER_FAIL token=%s reason=safe_change_5m", token.get("symbol", "N/A"))
                                continue

                        # Expensive safety API check only after cheap filters.
                        safety = await checker.check_token_safety(
                            token.get("address", ""),
                            token.get("liquidity", 0),
                        )
                        safety_checked += 1
                        if str((safety or {}).get("source", "")).lower() == "fail_closed":
                            safety_fail_closed_hits += 1
                        token["safety"] = safety or {}
                        token["risk_level"] = str((safety or {}).get("risk_level", "HIGH")).upper()
                        token["warning_flags"] = int((safety or {}).get("warning_flags", 0))
                        token["is_contract_safe"] = bool((safety or {}).get("is_safe", False))

                        if config.SAFE_TEST_MODE:
                            if config.SAFE_REQUIRE_CONTRACT_SAFE and not bool(token.get("is_contract_safe", False)):
                                logger.info("FILTER_FAIL token=%s reason=safe_contract", token.get("symbol", "N/A"))
                                continue
                            required_risk = str(config.SAFE_REQUIRE_RISK_LEVEL).upper()
                            risk = str(token.get("risk_level", "HIGH")).upper()
                            rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
                            if rank.get(risk, 2) > rank.get(required_risk, 1):
                                logger.info("FILTER_FAIL token=%s reason=safe_risk", token.get("symbol", "N/A"))
                                continue
                            if int(token.get("warning_flags") or 0) > int(config.SAFE_MAX_WARNING_FLAGS):
                                logger.info("FILTER_FAIL token=%s reason=safe_warnings", token.get("symbol", "N/A"))
                                continue

                        logger.info(
                            "FILTER_PASS token=%s score=%s risk=%s liq=%.0f vol5m=%.0f",
                            token.get("symbol", "N/A"),
                            score_data.get("score", 0),
                            token.get("risk_level", "N/A"),
                            float(token.get("liquidity") or 0),
                            float(token.get("volume_5m") or 0),
                        )
                        trade_candidates.append((token, score_data))
                        # Optional: avoid alert spam from watchlist flow.
                        if str(token.get("source", "")).lower() == "watchlist" and not bool(
                            getattr(config, "WATCHLIST_ALERTS_ENABLED", False)
                        ):
                            pass
                        else:
                            alerts_sent += await local_alerter.send_alert(token, score_data, safety=safety)

                dex_stats = dex_monitor.runtime_stats(reset=True)
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

                opened_trades = 0
                if trade_candidates and policy_state == "OK":
                    opened_trades = await auto_trader.plan_batch(trade_candidates)
                elif trade_candidates:
                    logger.warning(
                        "AUTO_POLICY mode=%s action=no_buy reason=%s candidates=%s",
                        policy_state,
                        policy_reason,
                        len(trade_candidates),
                    )
                await auto_trader.process_open_positions(bot=None)

                active_tasks = len(asyncio.all_tasks())
                now_ts = time.time()
                rss_mb = 0.0
                if (now_ts - last_rss_log_ts) >= int(getattr(config, "METRICS_RSS_LOG_SECONDS", 900)):
                    last_rss_log_ts = now_ts
                    rss_mb = _rss_memory_mb()

                logger.info(
                    "Scanned %s tokens | High quality: %s | Alerts sent: %s | Trade candidates: %s | Opened: %s | Mode: local | Source: %s | Policy: %s(%s) | Safety: checked=%s fail_closed=%s reasons=%s | Sources: %s | Tasks: %s | RSS: %.1fMB | CycleAvg: %.2fs",
                    len(tokens or []),
                    high_quality,
                    alerts_sent,
                    len(trade_candidates),
                    opened_trades,
                    source_mode,
                    policy_state,
                    policy_reason,
                    safety_checked,
                    safety_fail_closed_hits,
                    _format_safety_reasons_brief(safety_stats),
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
