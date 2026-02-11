"""Local fast runner without Telegram polling/webhooks."""

from __future__ import annotations

import asyncio
import atexit
import ctypes
import logging
import os
import time
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

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WINDOWS_MUTEX_NAME = "Global\\solana_alert_bot_main_local_single_instance"


def _merge_token_streams(*groups: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for group in groups:
        for token in group or []:
            address = str(token.get("address", "")).strip().lower()
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

    logger.info("Local mode started (Telegram disabled).")
    while True:
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
            if tokens:
                for token in tokens:
                    safety = await checker.check_token_safety(
                        token.get("address", ""),
                        token.get("liquidity", 0),
                    )
                    token["safety"] = safety or {}
                    token["risk_level"] = str((safety or {}).get("risk_level", "HIGH")).upper()
                    token["warning_flags"] = int((safety or {}).get("warning_flags", 0))
                    token["is_contract_safe"] = bool((safety or {}).get("is_safe", False))

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

            opened_trades = 0
            if trade_candidates:
                opened_trades = await auto_trader.plan_batch(trade_candidates)
            await auto_trader.process_open_positions(bot=None)

            logger.info(
                "Scanned %s tokens | High quality: %s | Alerts sent: %s | Trade candidates: %s | Opened: %s | Mode: local | Source: %s",
                len(tokens or []),
                high_quality,
                alerts_sent,
                len(trade_candidates),
                opened_trades,
                source_mode,
            )
        except Exception:
            logger.exception("Local monitoring loop error")

        sleep_seconds = (
            max(1, int(config.ONCHAIN_POLL_INTERVAL_SECONDS))
            if str(config.SIGNAL_SOURCE).lower() == "onchain"
            else int(SCAN_INTERVAL)
        )
        await asyncio.sleep(sleep_seconds)


def main() -> None:
    lock = InstanceLock()
    if not lock.acquire():
        print("Another main_local.py instance is already running. Exit.")
        return
    atexit.register(lock.release)
    configure_logging()
    try:
        asyncio.run(run_local_loop())
    finally:
        lock.release()


if __name__ == "__main__":
    main()
