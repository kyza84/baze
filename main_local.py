"""Local fast runner without Telegram polling/webhooks."""

from __future__ import annotations

import asyncio
import atexit
import ctypes
import logging
import os
from logging.handlers import RotatingFileHandler

import config
from config import APP_LOG_FILE, LOG_DIR, LOG_LEVEL, SCAN_INTERVAL
from monitor.dexscreener import DexScreenerMonitor
from monitor.local_alerter import LocalAlerter
from monitor.token_checker import TokenChecker
from monitor.token_scorer import TokenScorer
from trading.auto_trader import AutoTrader

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WINDOWS_MUTEX_NAME = "Global\\solana_alert_bot_main_local_single_instance"


class InstanceLock:
    def __init__(self) -> None:
        self._handle = None
        self.acquired = False

    def acquire(self) -> bool:
        if os.name != "nt":
            # Local fallback for non-Windows environments.
            lock_file = os.path.join(PROJECT_ROOT, "main_local.lock")
            try:
                fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="ascii") as f:
                    f.write(str(os.getpid()))
            except FileExistsError:
                return False
            self.acquired = True
            return True

        kernel32 = ctypes.windll.kernel32
        mutex = kernel32.CreateMutexW(None, False, WINDOWS_MUTEX_NAME)
        if not mutex:
            return False
        error_already_exists = 183
        if kernel32.GetLastError() == error_already_exists:
            kernel32.CloseHandle(mutex)
            return False
        self._handle = mutex
        self.acquired = True
        return True

    def release(self) -> None:
        if not self.acquired:
            return
        if os.name != "nt":
            lock_file = os.path.join(PROJECT_ROOT, "main_local.lock")
            try:
                os.remove(lock_file)
            except OSError:
                pass
            self.acquired = False
            return
        if self._handle:
            ctypes.windll.kernel32.CloseHandle(self._handle)
            self._handle = None
        self.acquired = False


def configure_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = RotatingFileHandler(APP_LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)


async def run_local_loop() -> None:
    monitor = DexScreenerMonitor()
    scorer = TokenScorer()
    checker = TokenChecker()
    local_alerter = LocalAlerter(os.path.join(LOG_DIR, "local_alerts.jsonl"))
    auto_trader = AutoTrader()

    logger.info("Local mode started (Telegram disabled).")
    while True:
        try:
            tokens = await monitor.fetch_new_tokens()
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
                        continue

                    if config.SAFE_TEST_MODE:
                        if float(token.get("liquidity") or 0) < float(config.SAFE_MIN_LIQUIDITY_USD):
                            continue
                        if float(token.get("volume_5m") or 0) < float(config.SAFE_MIN_VOLUME_5M_USD):
                            continue
                        if int(token.get("age_seconds") or 0) < int(config.SAFE_MIN_AGE_SECONDS):
                            continue
                        if abs(float(token.get("price_change_5m") or 0)) > float(config.SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT):
                            continue
                        if config.SAFE_REQUIRE_CONTRACT_SAFE and not bool(token.get("is_contract_safe", False)):
                            continue
                        required_risk = str(config.SAFE_REQUIRE_RISK_LEVEL).upper()
                        risk = str(token.get("risk_level", "HIGH")).upper()
                        rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
                        if rank.get(risk, 2) > rank.get(required_risk, 1):
                            continue
                        if int(token.get("warning_flags") or 0) > int(config.SAFE_MAX_WARNING_FLAGS):
                            continue

                    trade_candidates.append((token, score_data))
                    alerts_sent += await local_alerter.send_alert(token, score_data, safety=safety)

            opened_trades = 0
            if trade_candidates:
                opened_trades = await auto_trader.plan_batch(trade_candidates)
            await auto_trader.process_open_positions(bot=None)

            logger.info(
                "Scanned %s tokens | High quality: %s | Alerts sent: %s | Trade candidates: %s | Opened: %s | Mode: local",
                len(tokens or []),
                high_quality,
                alerts_sent,
                len(trade_candidates),
                opened_trades,
            )
        except Exception:
            logger.exception("Local monitoring loop error")

        await asyncio.sleep(SCAN_INTERVAL)


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
