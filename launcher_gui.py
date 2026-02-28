"""Dark control panel for bot process, activity feed, and settings."""

from __future__ import annotations

import os
import re
import hashlib
import math
import subprocess
import tkinter as tk
import ctypes
import threading
import time
from datetime import datetime, timedelta, timezone
import json
import ssl
import urllib.request
from tkinter import messagebox, ttk

from monitor.gui_engine_control import (
    EngineContext,
    run_powershell_script as engine_run_powershell_script,
    start_bot as engine_start_bot,
    stop_bot as engine_stop_bot,
)
from utils.state_file import StateFileLockError, write_json_atomic_locked

try:
    import msvcrt  # Windows-only, used for a robust single-instance file lock.
except Exception:  # pragma: no cover
    msvcrt = None  # type: ignore[assignment]

try:
    from web3 import HTTPProvider, Web3
except Exception:  # pragma: no cover
    HTTPProvider = None  # type: ignore[assignment]
    Web3 = None  # type: ignore[assignment]

def _fix_mojibake(text: str) -> str:
    """Fix common UTF-8 mojibake variants (e.g. Ã..., Ð..., Ñ...)."""
    if not isinstance(text, str):
        return str(text)
    if not any(marker in text for marker in ("Ã", "Ð", "Ñ")):
        return text
    for source_encoding in ("latin1", "cp1252"):
        try:
            fixed = text.encode(source_encoding).decode("utf-8")
        except Exception:
            continue
        if fixed != text and re.search(r"[А-Яа-яЁё]", fixed):
            return fixed
    return text

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_PATH = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
PID_FILE = os.path.join(PROJECT_ROOT, "bot.pid")
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUT_LOG = os.path.join(LOG_DIR, "bot.out.log")
ERR_LOG = os.path.join(LOG_DIR, "bot.err.log")
APP_LOG = os.path.join(LOG_DIR, "app.log")
LOCAL_ALERTS_FILE = os.path.join(LOG_DIR, "local_alerts.jsonl")
CANDIDATE_DECISIONS_LOG_FILE = os.path.join("logs", "candidates.jsonl")
TRADE_DECISIONS_LOG_FILE = os.path.join("logs", "trade_decisions.jsonl")
PAPER_STATE_FILE = os.path.join(PROJECT_ROOT, "trading", "paper_state.json")
MATRIX_ACTIVE_FILE = os.path.join(PROJECT_ROOT, "data", "matrix", "runs", "active_matrix.json")
MATRIX_LAUNCHER = os.path.join(PROJECT_ROOT, "tools", "matrix_paper_launcher.ps1")
MATRIX_STOPPER = os.path.join(PROJECT_ROOT, "tools", "matrix_paper_stop.ps1")
MATRIX_SUMMARY = os.path.join(PROJECT_ROOT, "tools", "matrix_paper_summary.ps1")
MATRIX_PROFILE_CATALOG = os.path.join(PROJECT_ROOT, "tools", "matrix_profile_catalog.ps1")
MATRIX_USER_PRESETS = os.path.join(PROJECT_ROOT, "tools", "matrix_user_presets.ps1")
GRACEFUL_STOP_FILE_DEFAULT = os.path.join(PROJECT_ROOT, "data", "graceful_stop.signal")
GRACEFUL_STOP_TIMEOUT_SECONDS_DEFAULT = 12
WALLET_MODE_KEY = "WALLET_MODE"
LIVE_WALLET_BALANCE_KEY = "LIVE_WALLET_BALANCE_USD"
STAIR_STEP_ENABLED_KEY = "STAIR_STEP_ENABLED"
GUI_MUTEX_NAME = "Global\\solana_alert_bot_launcher_gui_single_instance"
LIVE_BALANCE_POLL_SECONDS = 12
GUI_INITIAL_REFRESH_DELAY_MS = 300
try:
    GUI_REFRESH_INTERVAL_MS = max(400, int(os.getenv("GUI_REFRESH_INTERVAL_MS", "500")))
except Exception:
    GUI_REFRESH_INTERVAL_MS = 500
try:
    GUI_UI_TICK_MS = max(120, int(os.getenv("GUI_UI_TICK_MS", "150")))
except Exception:
    GUI_UI_TICK_MS = 150
try:
    STATE_FILE_LOCK_TIMEOUT_SECONDS = max(0.1, float(os.getenv("STATE_FILE_LOCK_TIMEOUT_SECONDS", "2.0")))
except Exception:
    STATE_FILE_LOCK_TIMEOUT_SECONDS = 2.0
try:
    STATE_FILE_LOCK_RETRY_SECONDS = max(0.01, float(os.getenv("STATE_FILE_LOCK_RETRY_SECONDS", "0.05")))
except Exception:
    STATE_FILE_LOCK_RETRY_SECONDS = 0.05
UI_FONT_MAIN = "Bahnschrift"
UI_FONT_MONO = "Cascadia Mono"
GUI_STATS_EXCLUDED_SYMBOLS_DEFAULT = "ZORA"
MATRIX_META_CACHE_TTL_SECONDS = 0.5
ENGINE_CONTEXT = EngineContext(
    project_root=PROJECT_ROOT,
    python_path=PYTHON_PATH,
    pid_file=PID_FILE,
    log_dir=LOG_DIR,
    out_log=OUT_LOG,
    err_log=ERR_LOG,
)
_MATRIX_META_CACHE: dict[str, object] = {"ts": 0.0, "mtime": 0.0, "payload": {}}

EVENT_KEYS = {
    "BUY": ("Paper BUY", "#67e8f9"),
    "SELL": ("Paper SELL", "#f472b6"),
    "SCAN": ("Scanned ", "#a7f3d0"),
    "ALERT": ("Alert dispatch", "#fcd34d"),
    "SKIP": ("AutoTrade skip", "#f59e0b"),
    "ERROR": ("[ERROR]", "#f87171"),
}
IMPORTANT_LOG_SNIPPETS = (
    "Scanned ",
    "Trade candidates:",
    "Opened:",
    "SAFETY_MODE",
    "DATA_MODE",
    "Paper BUY",
    "Paper SELL",
    "AutoTrade skip",
    "AutoTrade batch",
    "Alert dispatch",
    "Local alert",
    "[ERROR]",
)
NOISE_LOG_SNIPPETS = (
    "telegram.ext.Updater:",
    "telegram.ext.Application:",
    "httpx.RemoteProtocolError",
    "httpcore.RemoteProtocolError",
    "Traceback (most recent call last):",
    "site-packages\\telegram\\",
    "site-packages\\httpx\\",
    "site-packages\\httpcore\\",
)
CRITICAL_LOG_MARKERS = (
    "CRITICAL_AUTO_RESET",
    "KILL_SWITCH",
    "AUTO_SELL live_failed",
    "AUTO_SELL forced_failed",
    "Local monitoring loop error",
    "[ERROR]",
)

BUY_RE = re.compile(
    r"Paper BUY token=(?P<symbol>\S+) address=(?P<address>\S+) entry=\$(?P<entry>[0-9.]+) size=\$(?P<size>[0-9.]+) score=(?P<score>\d+)"
)
SELL_RE = re.compile(
    r"Paper SELL token=(?P<symbol>\S+) reason=(?P<reason>\S+) exit=\$(?P<exit>[0-9.]+) pnl=(?P<pnl_pct>[-+0-9.]+)% \(\$(?P<pnl_usd>[-+0-9.]+)\)(?: raw=[-+0-9.]+% cost=[-+0-9.]+% gas=\$[-+0-9.]+)? balance=\$(?P<balance>[0-9.]+)"
)
PAIR_SOURCE_RE = re.compile(r"PAIR_DETECTED source=(?P<source>[a-zA-Z0-9_:-]+)")
SCAN_SUMMARY_RE = re.compile(
    r"Scanned (?P<scanned>\d+) tokens \| High quality: (?P<hq>\d+) \| Alerts sent: (?P<alerts>\d+) "
    r"\| Trade candidates: (?P<candidates>\d+) \| Opened: (?P<opened>\d+) \| Mode: (?P<mode>[^|]+) "
    r"\| Source: (?P<source>[^|]+) \| Policy: (?P<policy>[A-Z_]+)\((?P<reason>[^)]*)\) "
    r"\| Safety: checked=(?P<safety_checked>\d+) fail_closed=(?P<safety_fc>\d+)(?: reasons=(?P<safety_reasons>.*?))? "
    r"\| Sources: (?P<sources>.*?) \| Tasks: (?P<tasks>\d+) \| RSS: (?P<rss>[0-9.]+)MB \| CycleAvg: (?P<cycle>[0-9.]+)s"
)
AUTO_POLICY_RE = re.compile(
    r"AUTO_POLICY mode=(?P<policy>[A-Z_]+) action=(?P<action>[a-z_]+) reason=(?P<reason>.+?) candidates=(?P<candidates>\d+)"
)
AUTO_TRADE_SKIP_RE = re.compile(r"AutoTrade skip token=\S+ reason=(?P<reason>[a-zA-Z0-9_:-]+)")

SETTINGS_FIELDS_RAW = [
    ("PERSONAL_MODE", "Ð›Ð¸Ñ‡Ð½Ñ‹Ð¹ Ñ€ÐµÐ¶Ð¸Ð¼"),
    ("PERSONAL_TELEGRAM_ID", "Telegram ID"),
    ("MIN_TOKEN_SCORE", "ÐœÐ¸Ð½. ÑÐºÐ¾Ñ€"),
    ("AUTO_FILTER_ENABLED", "ÐÐ²Ñ‚Ð¾Ñ„Ð¸Ð»ÑŒÑ‚Ñ€"),
    ("AUTO_TRADE_ENABLED", "ÐÐ²Ñ‚Ð¾Ñ‚Ñ€ÐµÐ¹Ð´"),
    ("AUTO_TRADE_PAPER", "Ð‘ÑƒÐ¼Ð°Ð¶Ð½Ñ‹Ð¹ Ñ€ÐµÐ¶Ð¸Ð¼"),
    ("AUTO_TRADE_ENTRY_MODE", "Ð ÐµÐ¶Ð¸Ð¼ Ð²Ñ…Ð¾Ð´Ð°"),
    ("AUTO_TRADE_TOP_N", "Ð¢Ð¾Ð¿ N"),
    ("AUTONOMOUS_CONTROL_ENABLED", "ÐÐ²Ñ‚Ð¾Ð½Ð¾Ð¼Ð½Ñ‹Ð¹ ÐºÐ¾Ð½Ñ‚Ñ€Ð¾Ð»Ð»ÐµÑ€"),
    ("AUTONOMOUS_CONTROL_MODE", "Ð ÐµÐ¶Ð¸Ð¼ Ð°Ð²Ñ‚Ð¾Ð½Ð¾Ð¼Ð½Ð¾Ð³Ð¾ ÐºÐ¾Ð½Ñ‚Ñ€Ð¾Ð»Ð»ÐµÑ€Ð°"),
    ("AUTONOMOUS_CONTROL_INTERVAL_SECONDS", "ÐžÐºÐ½Ð¾ Ð°Ð²Ñ‚Ð¾ÐºÐ¾Ð½Ñ‚Ñ€Ð¾Ð»Ñ (ÑÐµÐº)"),
    ("AUTONOMOUS_CONTROL_MIN_WINDOW_CYCLES", "ÐœÐ¸Ð½. Ñ†Ð¸ÐºÐ»Ð¾Ð² Ð² Ð¾ÐºÐ½Ðµ"),
    ("AUTONOMOUS_CONTROL_TARGET_CANDIDATES_MIN", "Ð¦ÐµÐ»ÑŒ cand min"),
    ("AUTONOMOUS_CONTROL_TARGET_CANDIDATES_HIGH", "Ð¦ÐµÐ»ÑŒ cand high"),
    ("AUTONOMOUS_CONTROL_TARGET_OPENED_MIN", "Ð¦ÐµÐ»ÑŒ opened/cycle"),
    ("AUTONOMOUS_CONTROL_NEG_REALIZED_TRIGGER_USD", "ÐŸÐ¾Ñ€Ð¾Ð³ Ð»Ð¾ÑÑÐ° Ð¾ÐºÐ½Ð° $"),
    ("AUTONOMOUS_CONTROL_POS_REALIZED_TRIGGER_USD", "ÐŸÐ¾Ñ€Ð¾Ð³ Ð¿Ñ€Ð¾Ñ„Ð¸Ñ‚Ð° Ð¾ÐºÐ½Ð° $"),
    ("AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MIN", "Ð“Ñ€Ð°Ð½Ð¸Ñ†Ð° open trades min"),
    ("AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MAX", "Ð“Ñ€Ð°Ð½Ð¸Ñ†Ð° open trades max"),
    ("AUTONOMOUS_CONTROL_TOP_N_MIN", "Ð“Ñ€Ð°Ð½Ð¸Ñ†Ð° top N min"),
    ("AUTONOMOUS_CONTROL_TOP_N_MAX", "Ð“Ñ€Ð°Ð½Ð¸Ñ†Ð° top N max"),
    ("AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN", "Ð“Ñ€Ð°Ð½Ð¸Ñ†Ð° buys/hour min"),
    ("AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MAX", "Ð“Ñ€Ð°Ð½Ð¸Ñ†Ð° buys/hour max"),
    ("WALLET_BALANCE_USD", "Ð‘ÑƒÐ¼Ð°Ð¶Ð½Ñ‹Ð¹ ÐºÐ¾ÑˆÐµÐ»ÐµÐº $"),
    ("PAPER_TRADE_SIZE_USD", "Ð‘Ð°Ð·Ð¾Ð²Ñ‹Ð¹ Ñ€Ð°Ð·Ð¼ÐµÑ€ ÑÐ´ÐµÐ»ÐºÐ¸ $"),
    ("PAPER_TRADE_SIZE_MIN_USD", "ÐœÐ¸Ð½. Ñ€Ð°Ð·Ð¼ÐµÑ€ ÑÐ´ÐµÐ»ÐºÐ¸ $"),
    ("PAPER_TRADE_SIZE_MAX_USD", "ÐœÐ°ÐºÑ. Ñ€Ð°Ð·Ð¼ÐµÑ€ ÑÐ´ÐµÐ»ÐºÐ¸ $"),
    ("PAPER_MAX_HOLD_SECONDS", "ÐœÐ°ÐºÑ. ÑƒÐ´ÐµÑ€Ð¶Ð°Ð½Ð¸Ðµ (ÑÐµÐº)"),
    ("DYNAMIC_HOLD_ENABLED", "Ð”Ð¸Ð½Ð°Ð¼Ð¸Ñ‡ÐµÑÐºÐ¸Ð¹ Ñ…Ð¾Ð»Ð´"),
    ("HOLD_MIN_SECONDS", "Ð¥Ð¾Ð»Ð´ Ð¼Ð¸Ð½ (ÑÐµÐº)"),
    ("HOLD_MAX_SECONDS", "Ð¥Ð¾Ð»Ð´ Ð¼Ð°ÐºÑ (ÑÐµÐº)"),
    ("MAX_OPEN_TRADES", "ÐœÐ°ÐºÑ. Ð¾Ñ‚ÐºÑ€Ñ‹Ñ‚Ñ‹Ñ… ÑÐ´ÐµÐ»Ð¾Ðº"),
    ("CLOSED_TRADES_MAX_AGE_DAYS", "Ð¥Ñ€Ð°Ð½Ð¸Ñ‚ÑŒ Ð·Ð°ÐºÑ€Ñ‹Ñ‚Ñ‹Ðµ (Ð´Ð½ÐµÐ¹)"),
    ("PAPER_REALISM_ENABLED", "Ð ÐµÐ°Ð»Ð¸ÑÑ‚Ð¸Ñ‡Ð½Ñ‹Ð¹ Ñ€ÐµÐ¶Ð¸Ð¼"),
    ("PAPER_REALISM_CAP_ENABLED", "Realism cap"),
    ("PAPER_REALISM_MAX_GAIN_PERCENT", "Realism max gain %"),
    ("PAPER_REALISM_MAX_LOSS_PERCENT", "Realism max loss %"),
    ("PAPER_GAS_PER_TX_USD", "Ð“Ð°Ð· Ð·Ð° tx $"),
    ("PAPER_SWAP_FEE_BPS", "ÐšÐ¾Ð¼Ð¸ÑÑÐ¸Ñ ÑÐ²Ð°Ð¿Ð° (bps)"),
    ("PAPER_BASE_SLIPPAGE_BPS", "Ð‘Ð°Ð·Ð¾Ð²Ñ‹Ð¹ slippage (bps)"),
    ("DYNAMIC_POSITION_SIZING_ENABLED", "Ð”Ð¸Ð½Ð°Ð¼Ð¸Ñ‡ÐµÑÐºÐ¸Ð¹ Ñ€Ð°Ð·Ð¼ÐµÑ€"),
    ("EDGE_FILTER_ENABLED", "Ð¤Ð¸Ð»ÑŒÑ‚Ñ€ edge"),
    ("MIN_EXPECTED_EDGE_PERCENT", "ÐœÐ¸Ð½. Ð¾Ð¶Ð¸Ð´Ð°ÐµÐ¼Ñ‹Ð¹ edge %"),
    ("PROFIT_TARGET_PERCENT", "Ð¢ÐµÐ¹Ðº-Ð¿Ñ€Ð¾Ñ„Ð¸Ñ‚ %"),
    ("STOP_LOSS_PERCENT", "Ð¡Ñ‚Ð¾Ð¿-Ð»Ð¾ÑÑ %"),
    ("SAFE_TEST_MODE", "Safe test mode"),
    ("SAFE_MIN_LIQUIDITY_USD", "Safe min liquidity $"),
    ("SAFE_MIN_VOLUME_5M_USD", "Safe min volume 5m $"),
    ("SAFE_MIN_AGE_SECONDS", "Safe min age (sec)"),
    ("SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT", "Safe max change 5m %"),
    ("SAFE_REQUIRE_CONTRACT_SAFE", "Safe require contract"),
    ("SAFE_REQUIRE_RISK_LEVEL", "Safe max risk"),
    ("SAFE_MAX_WARNING_FLAGS", "Safe max warnings"),
    ("MAX_TOKEN_PRICE_CHANGE_5M_ABS_PERCENT", "Hard max change 5m %"),
    ("MAX_TOKEN_COOLDOWN_SECONDS", "Token cooldown (sec)"),
    ("MAX_LOSS_PER_TRADE_PERCENT_BALANCE", "Max loss/trade % eq"),
    ("DAILY_MAX_DRAWDOWN_PERCENT", "Daily max drawdown %"),
    ("MAX_CONSECUTIVE_LOSSES", "Max loss streak"),
    ("LOSS_STREAK_COOLDOWN_SECONDS", "Loss cooldown (sec)"),
    ("DEX_BOOSTS_SOURCE_ENABLED", "Dex boosts source"),
    ("DEX_BOOSTS_MAX_TOKENS", "Dex boosts max tokens"),
    ("DEX_SEARCH_QUERIES", "ÐŸÐ¾Ð¸ÑÐº Dex (Ñ‡ÐµÑ€ÐµÐ· Ð·Ð°Ð¿ÑÑ‚ÑƒÑŽ)"),
    ("GECKO_NEW_POOLS_PAGES", "Ð¡Ñ‚Ñ€Ð°Ð½Ð¸Ñ† Gecko"),
    ("WETH_PRICE_FALLBACK_USD", "WETH fallback $"),
    ("STAIR_STEP_ENABLED", "Step protection"),
    ("STAIR_STEP_START_BALANCE_USD", "Step start floor $"),
    ("STAIR_STEP_SIZE_USD", "Step size $"),
    ("STAIR_STEP_TRADABLE_BUFFER_USD", "Step tradable buffer $"),
    ("AUTO_STOP_MIN_AVAILABLE_USD", "Stop min available $"),
]
SETTINGS_FIELDS = [(k, _fix_mojibake(v)) for (k, v) in SETTINGS_FIELDS_RAW]
GUI_SETTINGS_FIELDS = [
    ("AUTO_TRADE_ENABLED", "Auto trade enabled"),
    ("AUTO_TRADE_PAPER", "Paper mode"),
    ("WALLET_MODE", "Wallet mode"),
    ("WALLET_BALANCE_USD", "Paper wallet balance (USD)"),
    ("LIVE_WALLET_BALANCE_USD", "Live wallet balance (USD)"),
    ("MIN_TOKEN_SCORE", "Min token score"),
    ("AUTO_TRADE_ENTRY_MODE", "Entry mode"),
    ("AUTO_TRADE_TOP_N", "Top-N candidates"),
    ("MAX_OPEN_TRADES", "Max open trades"),
    ("MAX_BUYS_PER_HOUR", "Max buys per hour"),
    ("MAX_TX_PER_DAY", "Max tx per day"),
    ("PAPER_TRADE_SIZE_MIN_USD", "Trade size min (USD)"),
    ("PAPER_TRADE_SIZE_MAX_USD", "Trade size max (USD)"),
    ("MIN_EXPECTED_EDGE_PERCENT", "Min expected edge (%)"),
    ("MIN_EXPECTED_EDGE_USD", "Min expected edge (USD)"),
    ("PROFIT_TARGET_PERCENT", "TP percent"),
    ("STOP_LOSS_PERCENT", "SL percent"),
    ("PAPER_MAX_HOLD_SECONDS", "Max hold (seconds)"),
    ("HOLD_MIN_SECONDS", "Hold min (seconds)"),
    ("HOLD_MAX_SECONDS", "Hold max (seconds)"),
    ("PAPER_PARTIAL_TP_ENABLED", "Partial TP enabled"),
    ("PAPER_PARTIAL_TP_TRIGGER_PERCENT", "Partial TP trigger (%)"),
    ("PAPER_PARTIAL_TP_SELL_FRACTION", "Partial TP sell fraction"),
    ("NO_MOMENTUM_EXIT_MAX_PNL_PERCENT", "No-momentum exit max pnl (%)"),
    ("WEAKNESS_EXIT_PNL_PERCENT", "Weakness exit pnl (%)"),
    ("SAFE_MIN_LIQUIDITY_USD", "Safe min liquidity (USD)"),
    ("SAFE_MIN_VOLUME_5M_USD", "Safe min volume 5m (USD)"),
    ("SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT", "Safe max 5m move (%)"),
    ("MAX_TOKEN_COOLDOWN_SECONDS", "Token cooldown (seconds)"),
    ("V2_POLICY_ROUTER_ENABLED", "V2 policy router"),
    ("V2_POLICY_FAIL_CLOSED_ACTION", "V2 fail-closed action"),
    ("V2_POLICY_DEGRADED_ACTION", "V2 degraded action"),
    ("V2_ENTRY_DUAL_CHANNEL_ENABLED", "V2 dual entry"),
    ("V2_ENTRY_EXPLORE_MAX_SHARE", "V2 explore share"),
    ("V2_UNIVERSE_NOVELTY_MIN_SHARE", "V2 novelty min share"),
]

MATRIX_OVERRIDE_COMMON_KEYS = (
    "AUTO_TRADE_TOP_N",
    "MAX_OPEN_TRADES",
    "MAX_BUYS_PER_HOUR",
    "MIN_TOKEN_SCORE",
    "MIN_EXPECTED_EDGE_PERCENT",
    "MIN_EXPECTED_EDGE_USD",
    "SAFE_MIN_VOLUME_5M_USD",
    "SAFE_MIN_LIQUIDITY_USD",
    "PAPER_TRADE_SIZE_MIN_USD",
    "PAPER_TRADE_SIZE_MAX_USD",
    "PAPER_MAX_HOLD_SECONDS",
    "HOLD_MIN_SECONDS",
    "HOLD_MAX_SECONDS",
    "PAPER_PARTIAL_TP_TRIGGER_PERCENT",
    "PAPER_PARTIAL_TP_SELL_FRACTION",
    "NO_MOMENTUM_EXIT_MAX_PNL_PERCENT",
    "WEAKNESS_EXIT_PNL_PERCENT",
    "MAX_TOKEN_COOLDOWN_SECONDS",
    "STOP_LOSS_PERCENT",
    "PROFIT_TARGET_PERCENT",
)

FIELD_OPTIONS = {
    "WALLET_MODE": ["paper", "live"],
    "PERSONAL_MODE": ["true", "false"],
    "AUTO_FILTER_ENABLED": ["true", "false"],
    "AUTO_TRADE_ENABLED": ["false", "true"],
    "AUTO_TRADE_PAPER": ["true", "false"],
    "AUTO_TRADE_ENTRY_MODE": ["single", "all", "top_n"],
    "AUTO_TRADE_TOP_N": ["3", "5", "10", "20", "50"],
    "AUTONOMOUS_CONTROL_ENABLED": ["false", "true"],
    "AUTONOMOUS_CONTROL_MODE": ["off", "dry_run", "apply"],
    "AUTONOMOUS_CONTROL_INTERVAL_SECONDS": ["120", "180", "240", "300", "600"],
    "AUTONOMOUS_CONTROL_MIN_WINDOW_CYCLES": ["1", "2", "3", "4", "5"],
    "AUTONOMOUS_CONTROL_TARGET_CANDIDATES_MIN": ["1.0", "1.5", "2.0", "3.0"],
    "AUTONOMOUS_CONTROL_TARGET_CANDIDATES_HIGH": ["4.0", "6.0", "8.0", "12.0"],
    "AUTONOMOUS_CONTROL_TARGET_OPENED_MIN": ["0.10", "0.15", "0.20", "0.30"],
    "AUTONOMOUS_CONTROL_NEG_REALIZED_TRIGGER_USD": ["0.03", "0.05", "0.08", "0.12"],
    "AUTONOMOUS_CONTROL_POS_REALIZED_TRIGGER_USD": ["0.03", "0.05", "0.08", "0.12"],
    "AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MIN": ["1", "2", "3"],
    "AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MAX": ["2", "3", "4", "5", "6"],
    "AUTONOMOUS_CONTROL_TOP_N_MIN": ["1", "2", "4", "6", "8"],
    "AUTONOMOUS_CONTROL_TOP_N_MAX": ["6", "8", "10", "12", "16", "20"],
    "AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN": ["6", "12", "18", "24", "36"],
    "AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MAX": ["24", "36", "48", "72", "96"],
    "MIN_TOKEN_SCORE": ["40", "50", "60", "70", "80", "90"],
    "WALLET_BALANCE_USD": ["2.75", "5", "10", "25", "50", "100"],
    "PAPER_TRADE_SIZE_USD": ["0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "5.0"],
    "PAPER_TRADE_SIZE_MIN_USD": ["0.1", "0.25", "0.5", "0.75", "1.0"],
    "PAPER_TRADE_SIZE_MAX_USD": ["0.5", "1.0", "1.5", "2.0", "3.0", "5.0"],
    "PAPER_MAX_HOLD_SECONDS": ["300", "600", "900", "1800", "3600"],
    "DYNAMIC_HOLD_ENABLED": ["true", "false"],
    "HOLD_MIN_SECONDS": ["120", "180", "300", "600", "900"],
    "HOLD_MAX_SECONDS": ["600", "900", "1200", "1800", "2400", "3600"],
    "MAX_OPEN_TRADES": ["0", "1", "2", "3", "5", "10"],
    "CLOSED_TRADES_MAX_AGE_DAYS": ["0", "3", "7", "14", "30", "60", "90"],
    "PAPER_REALISM_ENABLED": ["true", "false"],
    "PAPER_REALISM_CAP_ENABLED": ["true", "false"],
    "PAPER_REALISM_MAX_GAIN_PERCENT": ["200", "300", "600", "1000"],
    "PAPER_REALISM_MAX_LOSS_PERCENT": ["50", "70", "85", "95"],
    "PAPER_GAS_PER_TX_USD": ["0.01", "0.02", "0.03", "0.05", "0.08", "0.1"],
    "PAPER_SWAP_FEE_BPS": ["5", "10", "20", "30", "40", "50"],
    "PAPER_BASE_SLIPPAGE_BPS": ["30", "50", "80", "100", "150", "200"],
    "DYNAMIC_POSITION_SIZING_ENABLED": ["true", "false"],
    "EDGE_FILTER_ENABLED": ["true", "false"],
    "MIN_EXPECTED_EDGE_PERCENT": ["0.5", "1.0", "2.0", "3.0", "5.0", "8.0"],
    "PROFIT_TARGET_PERCENT": ["20", "30", "40", "50", "75", "100"],
    "STOP_LOSS_PERCENT": ["10", "15", "20", "25", "30", "40"],
    "SAFE_TEST_MODE": ["true", "false"],
    "SAFE_REQUIRE_CONTRACT_SAFE": ["true", "false"],
    "SAFE_REQUIRE_RISK_LEVEL": ["LOW", "MEDIUM", "HIGH"],
    "SAFE_MAX_WARNING_FLAGS": ["0", "1", "2", "3", "5"],
    "MAX_CONSECUTIVE_LOSSES": ["2", "3", "4", "5", "6"],
    "DEX_BOOSTS_SOURCE_ENABLED": ["true", "false"],
    "DEX_BOOSTS_MAX_TOKENS": ["5", "10", "20", "30", "50"],
    "GECKO_NEW_POOLS_PAGES": ["1", "2", "3", "4", "5"],
    "STAIR_STEP_ENABLED": ["false", "true"],
    "STAIR_STEP_START_BALANCE_USD": ["1.5", "2.75", "5", "10"],
    "STAIR_STEP_SIZE_USD": ["2", "5", "10"],
    "STAIR_STEP_TRADABLE_BUFFER_USD": ["0.25", "0.35", "0.5", "0.75", "1.0"],
    "AUTO_STOP_MIN_AVAILABLE_USD": ["0.10", "0.25", "0.35", "0.50", "1.00"],
    "V2_POLICY_ROUTER_ENABLED": ["true", "false"],
    "V2_POLICY_FAIL_CLOSED_ACTION": ["limited", "block", "allow_all"],
    "V2_POLICY_DEGRADED_ACTION": ["limited", "block", "allow_all"],
    "V2_ENTRY_DUAL_CHANNEL_ENABLED": ["true", "false"],
}


def read_pid() -> int | None:
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r", encoding="ascii") as f:
            return int(f.read().strip())
    except Exception:
        return None


def is_running(pid: int | None) -> bool:
    if not pid:
        return False
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def process_commandline(pid: int | None) -> str:
    if not pid:
        return ""
    cmd = (
        "Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" "
        "| Select-Object -ExpandProperty CommandLine"
    ).format(pid=int(pid))
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def is_main_local_process(pid: int | None) -> bool:
    if not is_running(pid):
        return False
    cmd = process_commandline(pid)
    if not cmd:
        return False
    lower = cmd.lower()
    return "python" in lower and "main_local.py" in lower


def list_main_local_pids() -> list[int]:
    cmd = (
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" "
        "| Where-Object { $_.CommandLine -match 'main_local.py' } "
        "| Select-Object -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    out: list[int] = []
    for line in result.stdout.splitlines():
        raw = line.strip()
        if raw.isdigit():
            out.append(int(raw))
    return out


def _resolve_graceful_stop_file() -> str:
    raw = "data/graceful_stop.signal"
    try:
        if os.path.exists(ENV_FILE):
            with open(ENV_FILE, "r", encoding="utf-8-sig") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    if k.strip() == "GRACEFUL_STOP_FILE":
                        raw = v.strip().strip("'").strip('"') or raw
                        break
    except Exception:
        pass
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(PROJECT_ROOT, raw))


def _resolve_graceful_stop_timeout_seconds() -> int:
    value = GRACEFUL_STOP_TIMEOUT_SECONDS_DEFAULT
    try:
        if os.path.exists(ENV_FILE):
            with open(ENV_FILE, "r", encoding="utf-8-sig") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    if k.strip() == "GRACEFUL_STOP_TIMEOUT_SECONDS":
                        try:
                            value = int(v.strip().strip("'").strip('"'))
                        except Exception:
                            pass
                        break
    except Exception:
        pass
    return max(2, int(value))


def _signal_graceful_stop() -> tuple[bool, str]:
    path = _resolve_graceful_stop_file()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} gui_graceful_stop\n")
        return True, path
    except Exception as exc:
        return False, str(exc)


def _clear_graceful_stop_flag() -> None:
    path = _resolve_graceful_stop_file()
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def start_bot() -> tuple[bool, str]:
    return engine_start_bot(
        ENGINE_CONTEXT,
        matrix_alive_count=matrix_alive_count,
        list_main_local_pids=list_main_local_pids,
        is_main_local_process=is_main_local_process,
        read_pid=read_pid,
        is_running=is_running,
        clear_graceful_stop_flag=_clear_graceful_stop_flag,
    )


def stop_bot() -> tuple[bool, str]:
    return engine_stop_bot(
        ENGINE_CONTEXT,
        list_main_local_pids=list_main_local_pids,
        is_main_local_process=is_main_local_process,
        read_pid=read_pid,
        signal_graceful_stop=_signal_graceful_stop,
        resolve_graceful_timeout=_resolve_graceful_stop_timeout_seconds,
        clear_graceful_stop_flag=_clear_graceful_stop_flag,
    )


class GuiInstanceLock:
    def __init__(self) -> None:
        self._handle = None
        self._lock_fh = None
        self.acquired = False
        self._lock_file = os.path.join(PROJECT_ROOT, "launcher_gui.lock")

    def _acquire_lock_file(self) -> bool:
        # Keep the file handle open for the lifetime of the process.
        os.makedirs(os.path.dirname(self._lock_file), exist_ok=True)
        try:
            fh = open(self._lock_file, "a+b")
        except OSError:
            return False

        try:
            # Ensure there is at least 1 byte to lock.
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"1")
                fh.flush()
            fh.seek(0)

            if os.name == "nt" and msvcrt is not None:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                # Best-effort fallback (non-Windows): no locking available here.
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
        mutex = kernel32.CreateMutexW(None, False, GUI_MUTEX_NAME)
        if not mutex:
            return False
        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
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


def read_tail(path: str, max_lines: int) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f.readlines()[-max_lines:]]


def read_tail_bytes(path: str, max_bytes: int) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max(0, int(max_bytes))))
            blob = f.read()
        return blob.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def latest_session_log(log_dir: str) -> str | None:
    sessions_dir = os.path.join(log_dir, "sessions")
    if not os.path.isdir(sessions_dir):
        return None
    files: list[tuple[float, str]] = []
    try:
        for name in os.listdir(sessions_dir):
            if not name.lower().endswith(".log"):
                continue
            full = os.path.join(sessions_dir, name)
            if os.path.isfile(full):
                files.append((os.path.getmtime(full), full))
    except Exception:
        return None
    if not files:
        return None
    files.sort(key=lambda x: x[0], reverse=True)
    return files[0][1]


def truncate_file(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    for attempt in range(8):
        try:
            # Accept files with or without UTF-8 BOM (PowerShell often writes BOM by default).
            with open(path, "r", encoding="utf-8-sig") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
            return {}
        except json.JSONDecodeError:
            # State can be read mid-write; retry briefly before falling back.
            time.sleep(0.03 + (attempt * 0.02))
            continue
        except OSError:
            # Windows can raise sharing violations briefly while writer replaces file.
            time.sleep(0.03 + (attempt * 0.02))
            continue
        except Exception:
            return {}
    return {}


def read_matrix_meta() -> dict:
    now = time.time()
    try:
        mtime = float(os.path.getmtime(MATRIX_ACTIVE_FILE))
    except Exception:
        mtime = 0.0
    cache_payload = _MATRIX_META_CACHE.get("payload")
    cache_ts = float(_MATRIX_META_CACHE.get("ts", 0.0) or 0.0)
    cache_mtime = float(_MATRIX_META_CACHE.get("mtime", 0.0) or 0.0)
    if isinstance(cache_payload, dict):
        if mtime > 0.0 and cache_mtime == mtime and (now - cache_ts) <= MATRIX_META_CACHE_TTL_SECONDS:
            return cache_payload
    payload = read_json(MATRIX_ACTIVE_FILE)
    if isinstance(payload, dict) and payload:
        _MATRIX_META_CACHE["payload"] = payload
        _MATRIX_META_CACHE["mtime"] = mtime
        _MATRIX_META_CACHE["ts"] = now
        return payload
    if isinstance(cache_payload, dict) and cache_payload:
        return cache_payload
    return {}


def matrix_alive_count() -> int:
    meta = read_matrix_meta()
    items = meta.get("items")
    alive = 0
    if isinstance(items, list):
        for row in items:
            if not isinstance(row, dict):
                continue
            try:
                pid = int(row.get("pid", 0) or 0)
            except Exception:
                pid = 0
            if pid > 0 and is_main_local_process(pid):
                alive += 1
    return alive


def run_powershell_script(
    script_path: str,
    args: list[str] | None = None,
    *,
    timeout_seconds: int = 90,
) -> tuple[bool, str]:
    return engine_run_powershell_script(
        ENGINE_CONTEXT,
        script_path=script_path,
        args=args,
        timeout_seconds=timeout_seconds,
    )


def read_jsonl_tail(path: str, max_lines: int) -> list[dict]:
    rows: list[dict] = []
    for line in read_tail(path, max_lines):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _parse_ts(raw: object) -> float | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        try:
            return float(raw)
        except Exception:
            return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _format_matrix_candidate_tail(log_dir: str, inst_id: str, max_rows: int = 80) -> list[str]:
    path = os.path.join(log_dir, "candidates.jsonl")
    rows = read_jsonl_tail(path, max_rows)
    if not rows:
        return []
    out: list[str] = []
    for row in rows[-max_rows:]:
        stage = str(row.get("decision_stage", "") or "").strip() or "-"
        decision = str(row.get("decision", "") or "").strip() or "-"
        reason = str(row.get("reason", "") or "").strip() or "-"
        symbol = str(row.get("symbol", "") or "").strip() or "-"
        score = row.get("score", 0)
        src = str(row.get("source_mode", "") or "").strip() or "-"
        out.append(
            f"[{inst_id}] cand stage={stage} decision={decision} reason={reason} symbol={symbol} score={score} src={src}"
        )
    return out


def _format_matrix_alert_tail(log_dir: str, inst_id: str, max_rows: int = 40) -> list[str]:
    path = os.path.join(log_dir, "local_alerts.jsonl")
    rows = read_jsonl_tail(path, max_rows)
    if not rows:
        return []
    out: list[str] = []
    for row in rows[-max_rows:]:
        symbol = str(row.get("symbol", "N/A"))
        score = row.get("score", 0)
        rec = str(row.get("recommendation", ""))
        risk = str(row.get("risk_level", ""))
        out.append(f"[{inst_id}] alert symbol={symbol} score={score} rec={rec} risk={risk}")
    return out


def _collect_matrix_runtime_sections(items: list[dict], max_lines_each: int = 120) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        inst_id = str(row.get("id", "") or "").strip() or "matrix"
        log_dir = str(row.get("log_dir", "") or "").strip()
        if not log_dir:
            continue
        abs_dir = log_dir if os.path.isabs(log_dir) else os.path.join(PROJECT_ROOT, log_dir)
        block: list[str] = []
        seen_line_keys: set[str] = set()

        def _append_unique(lines: list[str]) -> None:
            for raw_line in lines:
                line = str(raw_line or "")
                key = line.strip()
                if not key:
                    continue
                if key in seen_line_keys:
                    continue
                seen_line_keys.add(key)
                block.append(line)

        latest = latest_session_log(abs_dir)
        if latest:
            label = os.path.basename(latest)
            block.append(f"--- session: {label} ---")
            _append_unique(compact_runtime_lines(read_tail(latest, max_lines_each))[-max_lines_each:])

        for name in ("app.log", "out.log"):
            p = os.path.join(abs_dir, name)
            if not os.path.exists(p):
                continue
            try:
                if os.path.getsize(p) <= 0:
                    continue
            except Exception:
                continue
            block.append(f"--- {name} ---")
            _append_unique(compact_runtime_lines(read_tail(p, max_lines_each))[-max_lines_each:])

        cand_lines = _format_matrix_candidate_tail(abs_dir, inst_id, max_rows=80)
        if cand_lines:
            block.append("--- candidates.jsonl (tail) ---")
            block.extend(cand_lines[-80:])

        alert_lines = _format_matrix_alert_tail(abs_dir, inst_id, max_rows=40)
        if alert_lines:
            block.append("--- local_alerts.jsonl (tail) ---")
            block.extend(alert_lines[-40:])

        if block:
            sections.append((inst_id, block))
    return sections


def parse_activity(lines: list[str]) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for line in lines:
        if any(noise in line for noise in NOISE_LOG_SNIPPETS):
            continue
        # Left feed is intentionally compact: skip per-token filter spam and low-level discovery noise.
        if (
            "FILTER_FAIL" in line
            or "FILTER_PASS" in line
            or "PAIR_DETECTED" in line
            or "ENRICH_RETRY" in line
            or "WATCHLIST merged count=" in line
        ):
            continue
        pretty = _compact_activity_line(line)
        matched = "INFO"
        if "[ERROR]" in line:
            matched = "ERROR"
        elif "[WARNING]" in line:
            matched = "WARNING"
        elif "[INFO]" in line:
            matched = "INFO"

        for key, (needle, _) in EVENT_KEYS.items():
            if needle in line:
                matched = key
                break
        if "Policy: FAIL_CLOSED(" in line or "AUTO_POLICY mode=FAIL_CLOSED" in line:
            matched = "POLICY_FAIL_CLOSED"
        elif "Policy: DEGRADED(" in line or "AUTO_POLICY mode=DEGRADED" in line:
            matched = "POLICY_DEGRADED"
        elif "Policy: OK(" in line:
            matched = "POLICY_OK"
        elif "SAFETY_MODE fail_closed active" in line:
            matched = "POLICY_FAIL_CLOSED"
        elif "DATA_MODE degraded" in line:
            matched = "POLICY_DEGRADED"
        elif "FILTER_FAIL" in line:
            matched = "FILTER_FAIL"
        elif "FILTER_PASS" in line:
            matched = "FILTER_PASS"
        elif "RATE_LIMIT" in line or "HTTP_RETRY" in line:
            matched = "API"
        elif "WATCHLIST" in line:
            matched = "WATCHLIST"
        elif "PAIR_DETECTED" in line or "ENRICH_RETRY" in line or "On-chain" in line:
            matched = "ONCHAIN"
        # Keep feed focused on summaries and critical events.
        elif (
            "Scanned " not in line
            and "AUTO_POLICY" not in line
            and "SAFETY_MODE" not in line
            and "DATA_MODE" not in line
            and "Paper BUY" not in line
            and "Paper SELL" not in line
            and "GRACEFUL_STOP" not in line
            and "AUTOTRADER_SHUTDOWN" not in line
            and "AUTOTRADER_INIT" not in line
            and "[ERROR]" not in line
            and "[WARNING]" not in line
        ):
            continue
        events.append((matched, pretty))
    return events


def _compact_activity_line(line: str) -> str:
    if "SAFETY_MODE fail_closed active;" in line:
        reason = line.split("reason=", 1)[1].strip() if "reason=" in line else "unknown"
        if reason.startswith("safety_api_down") or reason.startswith("safety_api_unreliable"):
            reason = "safety_api_unreliable -> buy paused"
        return f"[FAIL_CLOSED] BUY paused until safety API recovers | {reason}"
    if "DATA_MODE degraded;" in line:
        reason = line.split("reason=", 1)[1].strip() if "reason=" in line else "unknown"
        return f"[DEGRADED] BUY paused by data policy | {reason}"

    scan = SCAN_SUMMARY_RE.search(line)
    if scan:
        policy = str(scan.group("policy")).strip().upper()
        reason = str(scan.group("reason")).strip()
        reason_short = reason
        if reason.startswith("safety_api_down") or reason.startswith("safety_api_unreliable"):
            reason_short = "safety_api_unreliable -> buy paused"
        safety_reasons = str(scan.group("safety_reasons") or "").strip()
        safety_tail = f" safety_reasons={safety_reasons}" if safety_reasons else ""
        return (
            f"[{policy}] scan={scan.group('scanned')} hq={scan.group('hq')} "
            f"cand={scan.group('candidates')} opened={scan.group('opened')} "
            f"src={scan.group('source').strip()} safety_fc={scan.group('safety_fc')}/{scan.group('safety_checked')} "
            f"tasks={scan.group('tasks')} cycle={scan.group('cycle')}s rss={scan.group('rss')}MB "
            f"reason={reason_short}{safety_tail}"
        )

    auto_policy = AUTO_POLICY_RE.search(line)
    if auto_policy:
        policy = auto_policy.group("policy").strip().upper()
        reason = auto_policy.group("reason").strip()
        candidates = auto_policy.group("candidates").strip()
        if reason.startswith("safety_api_down") or reason.startswith("safety_api_unreliable"):
            reason = "safety_api_unreliable -> buy paused"
        return f"[{policy}] BUY blocked candidates={candidates} reason={reason}"

    return line


def compact_runtime_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    skipping_trace = False
    for line in lines:
        if not line.strip():
            if skipping_trace:
                skipping_trace = False
            continue
        if "Traceback (most recent call last):" in line:
            skipping_trace = True
            continue
        if skipping_trace:
            # Drop traceback body and site-package stack frames.
            if line.startswith(" ") or line.startswith("\t") or "File \"" in line:
                continue
            skipping_trace = False
        if any(noise in line for noise in NOISE_LOG_SNIPPETS):
            continue
        out.append(line)
    return out


def load_env_lines() -> list[str]:
    return load_env_lines_from(ENV_FILE)


def load_env_lines_from(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def read_env_map() -> dict[str, str]:
    return read_env_map_from(ENV_FILE)


def resolve_runtime_env_file() -> str:
    root_map = read_env_map_from(ENV_FILE)
    raw = str(root_map.get("BOT_ENV_FILE", "") or "").strip()
    if not raw:
        return ENV_FILE
    return raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)


def read_runtime_env_map() -> dict[str, str]:
    # Runtime can be redirected via BOT_ENV_FILE (used by live promotion).
    env_map = read_env_map_from(ENV_FILE)
    env_path = resolve_runtime_env_file()
    if os.path.abspath(env_path) == os.path.abspath(ENV_FILE):
        return env_map
    if not os.path.exists(env_path):
        return env_map
    nested = read_env_map_from(env_path)
    if nested:
        env_map.update(nested)
    return env_map


def read_env_map_from(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in load_env_lines_from(path):
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def save_env_map(values: dict[str, str]) -> None:
    save_env_map_to(ENV_FILE, values)


def save_runtime_env_map(values: dict[str, str]) -> None:
    save_env_map_to(resolve_runtime_env_file(), values)


def save_env_map_to(path: str, values: dict[str, str]) -> None:
    lines = load_env_lines_from(path)
    existing_keys = set()
    updated: list[str] = []

    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in values:
                updated.append(f"{key}={values[key]}")
                existing_keys.add(key)
                continue
        updated.append(line)

    for key, value in values.items():
        if key not in existing_keys:
            updated.append(f"{key}={value}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(updated).rstrip() + "\n")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Base Bot Control Center")
        self.geometry("1180x760")
        self.minsize(1020, 680)
        self.configure(bg="#0b1220")

        self._configure_theme()
        self.env_vars: dict[str, tk.StringVar] = {}

        self.status_var = tk.StringVar(value="Status: unknown")
        self.hint_var = tk.StringVar(value="Hint: validate in paper mode first, then enable live.")
        self.matrix_count_var = tk.StringVar(value="2")
        self.matrix_profiles_hint_var = tk.StringVar(value="Matrix profiles: default by count")
        self.matrix_user_name_var = tk.StringVar(value="")
        self.matrix_user_base_var = tk.StringVar(value="")
        self.matrix_user_note_var = tk.StringVar(value="")
        self.matrix_override_key_var = tk.StringVar(value="")
        self.matrix_override_value_var = tk.StringVar(value="")
        self.matrix_recent_winners_var = tk.StringVar(value="Recent winners: --")
        self._matrix_selected_profile_ids: list[str] = []
        self._matrix_catalog_rows: dict[str, dict] = {}
        self._matrix_user_overrides_map: dict[str, str] = {}
        self.positions_source_var = tk.StringVar(value="single")
        self.positions_source_map: dict[str, str] = {"single": PAPER_STATE_FILE}
        self._state_cache: dict[str, dict] = {}
        self.refresh_meta_var = tk.StringVar(value="Refresh: --")
        self.header_limits_var = tk.StringVar(value="Limits: --")
        self.header_health_var = tk.StringVar(value="Health: --")

        self._build_header()
        self._build_tabs()
        self._load_settings()
        self._refresh_matrix_catalog()

        self._last_feed_signature = ""
        self._last_raw_signature = ""
        self._last_refresh_ts = 0.0
        self._open_row_expiry_ts: dict[str, float] = {}
        self._open_row_opened_ts: dict[str, float] = {}
        self._open_row_pnl_usd: dict[str, float] = {}
        self._open_ttl_col_idx = -1
        self._open_pnl_hour_col_idx = -1
        self.after(GUI_INITIAL_REFRESH_DELAY_MS, self.refresh)
        self.after(GUI_REFRESH_INTERVAL_MS, self.auto_refresh)
        self.after(GUI_UI_TICK_MS, self._ui_tick)
        self._live_wallet_eth = 0.0
        self._live_wallet_usd = 0.0
        self._live_wallet_last_ts = 0.0
        self._live_wallet_last_err = ""
        self._live_wallet_polling = False
        self.after(800, self._schedule_live_wallet_poll)

    def _configure_theme(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#080f1f", foreground="#dbe7ff", fieldbackground="#0d172d")
        style.configure("Card.TFrame", background="#0d172d", borderwidth=1, relief="solid")
        style.configure("Header.TFrame", background="#0a1326")
        style.configure("Title.TLabel", font=(UI_FONT_MAIN, 18, "bold"), foreground="#f8fafc", background="#0f172a")
        style.configure("Hint.TLabel", font=(UI_FONT_MAIN, 9), foreground="#8eb0e8", background="#0a1326")
        style.configure("Status.TLabel", font=(UI_FONT_MAIN, 10, "bold"), foreground="#8df0b7", background="#0a1326")
        style.configure("TLabel", font=(UI_FONT_MAIN, 10))
        style.configure("TButton", font=(UI_FONT_MAIN, 10, "bold"), padding=9, borderwidth=1)
        style.map("TButton", background=[("active", "#15325f"), ("pressed", "#0f2749")], foreground=[("active", "#f8fafc")])
        style.configure("Accent.TButton", font=(UI_FONT_MAIN, 10, "bold"), padding=9, borderwidth=1, background="#14532d", foreground="#eafff1")
        style.map("Accent.TButton", background=[("active", "#166534"), ("pressed", "#14532d")], foreground=[("active", "#f0fdf4")])
        style.configure("KPI.TFrame", background="#0a1933")
        style.configure("KPILabel.TLabel", font=(UI_FONT_MAIN, 9), foreground="#93a4c6", background="#0a1933")
        style.configure("KPIValue.TLabel", font=(UI_FONT_MAIN, 13, "bold"), foreground="#e6efff", background="#0a1933")
        style.configure("TEntry", padding=6)
        style.configure(
            "TCombobox",
            fieldbackground="#0d172d",
            background="#0d172d",
            foreground="#e5e7eb",
            arrowcolor="#93c5fd",
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#0f172a")],
            foreground=[("readonly", "#e5e7eb")],
            selectbackground=[("readonly", "#1e293b")],
            selectforeground=[("readonly", "#f8fafc")],
        )
        style.configure("TNotebook", background="#080f1f", borderwidth=0)
        style.configure("TNotebook.Tab", background="#152238", foreground="#b7c7e8", padding=(14, 10), font=(UI_FONT_MAIN, 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#0d172d")], foreground=[("selected", "#f8fafc")])
        style.configure(
            "Treeview",
            background="#0b1428",
            fieldbackground="#0b1428",
            foreground="#e5e7eb",
            borderwidth=0,
            rowheight=32,
            font=(UI_FONT_MAIN, 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#14243f",
            foreground="#bfd9ff",
            borderwidth=0,
            font=(UI_FONT_MAIN, 10, "bold"),
        )
        style.map(
            "Treeview",
            background=[("selected", "#1d4ed8")],
            foreground=[("selected", "#f8fafc")],
        )

    def _build_header(self) -> None:
        hdr = ttk.Frame(self, style="Header.TFrame", padding=(16, 14))
        hdr.pack(fill=tk.X)

        left = ttk.Frame(hdr, style="Header.TFrame")
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(left, text="Base Bot Control Center", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, textvariable=self.hint_var, style="Hint.TLabel").pack(anchor="w", pady=(3, 0))
        ttk.Label(left, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(left, textvariable=self.refresh_meta_var, style="Hint.TLabel").pack(anchor="w", pady=(2, 0))
        metrics = ttk.Frame(left, style="Header.TFrame")
        metrics.pack(anchor="w", pady=(6, 0), fill=tk.X)
        ttk.Label(metrics, textvariable=self.header_limits_var, style="Hint.TLabel").pack(anchor="w")
        self.health_label = ttk.Label(metrics, textvariable=self.header_health_var, style="Hint.TLabel")
        self.health_label.pack(anchor="w", pady=(2, 0))

        right = ttk.Frame(hdr, style="Header.TFrame")
        right.pack(side=tk.RIGHT)
        ttk.Button(right, text="Start", command=self.on_start, style="Accent.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Stop", command=self.on_stop).pack(side=tk.LEFT, padx=4)
        ttk.Separator(right, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(right, text="Matrix").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Combobox(
            right,
            width=3,
            state="readonly",
            textvariable=self.matrix_count_var,
            values=("1", "2", "3", "4"),
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(right, text="Matrix Start", command=self.on_matrix_start, style="Accent.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Matrix Stop", command=self.on_matrix_stop).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Matrix Summary", command=self.on_matrix_summary).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Save", command=self._save_settings).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Restart", command=self.on_restart).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Clear Logs", command=self.on_clear_logs).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Refresh", command=self.refresh).pack(side=tk.LEFT, padx=4)

    def _build_tabs(self) -> None:
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        self.activity_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.signals_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.wallet_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.positions_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.matrix_presets_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.settings_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.emergency_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.tabs.add(self.activity_tab, text="Activity")
        self.tabs.add(self.signals_tab, text="Signals")
        self.tabs.add(self.wallet_tab, text="Wallet")
        self.tabs.add(self.positions_tab, text="Trades")
        self.tabs.add(self.matrix_presets_tab, text="Matrix Presets")
        self.tabs.add(self.settings_tab, text="Settings")
        self.tabs.add(self.emergency_tab, text="Emergency")

        self._build_activity_tab()
        self._build_signals_tab()
        self._build_wallet_tab()
        self._build_positions_tab()
        self._build_matrix_presets_tab()
        self._build_settings_tab()
        self._build_emergency_tab()

    def _build_activity_tab(self) -> None:
        activity_top = ttk.Frame(self.activity_tab, style="Card.TFrame")
        activity_top.pack(fill=tk.BOTH, expand=True)
        activity_top.columnconfigure(0, weight=1)
        activity_top.columnconfigure(1, weight=1)
        activity_top.rowconfigure(0, weight=1)

        left_card = ttk.Frame(activity_top, style="Card.TFrame")
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(left_card, text="Signal and Trade Feed", font=("Segoe UI Semibold", 12)).pack(anchor="w", pady=(0, 8))
        self.feed_text = tk.Text(
            left_card,
            bg="#091224",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief=tk.FLAT,
            wrap="word",
            font=(UI_FONT_MONO, 10),
        )
        self.feed_text.pack(fill=tk.BOTH, expand=True)
        self._attach_text_copy_support(self.feed_text)
        self.feed_text.tag_configure("BUY", foreground=EVENT_KEYS["BUY"][1])
        self.feed_text.tag_configure("SELL", foreground=EVENT_KEYS["SELL"][1])
        self.feed_text.tag_configure("SCAN", foreground=EVENT_KEYS["SCAN"][1])
        self.feed_text.tag_configure("ALERT", foreground=EVENT_KEYS["ALERT"][1])
        self.feed_text.tag_configure("SKIP", foreground=EVENT_KEYS["SKIP"][1])
        self.feed_text.tag_configure("ERROR", foreground=EVENT_KEYS["ERROR"][1])
        self.feed_text.tag_configure("WARNING", foreground="#fb7185")
        self.feed_text.tag_configure("FILTER_FAIL", foreground="#f59e0b")
        self.feed_text.tag_configure("FILTER_PASS", foreground="#93c5fd")
        self.feed_text.tag_configure("API", foreground="#c4b5fd")
        self.feed_text.tag_configure("WATCHLIST", foreground="#67e8f9")
        self.feed_text.tag_configure("ONCHAIN", foreground="#a7f3d0")
        self.feed_text.tag_configure("POLICY_OK", foreground="#86efac")
        self.feed_text.tag_configure("POLICY_DEGRADED", foreground="#fbbf24")
        self.feed_text.tag_configure("POLICY_FAIL_CLOSED", foreground="#f87171")
        self.feed_text.tag_configure("INFO", foreground="#cbd5e1")

        right_card = ttk.Frame(activity_top, style="Card.TFrame")
        right_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(right_card, text="Raw runtime logs", font=("Segoe UI Semibold", 12)).pack(anchor="w", pady=(0, 8))
        self.raw_text = tk.Text(
            right_card,
            bg="#091224",
            fg="#cbd5e1",
            insertbackground="#e5e7eb",
            relief=tk.FLAT,
            wrap="none",
            font=(UI_FONT_MONO, 9),
        )
        self.raw_text.pack(fill=tk.BOTH, expand=True)
        self._attach_text_copy_support(self.raw_text)

    def _build_signals_tab(self) -> None:
        top = ttk.Frame(self.signals_tab, style="Card.TFrame")
        top.pack(fill=tk.X, pady=(0, 10))
        self.signals_summary_var = tk.StringVar(value="Ð’Ñ…Ð¾Ð´ÑÑ‰Ð¸Ñ… ÑÐ¸Ð³Ð½Ð°Ð»Ð¾Ð² Ð¿Ð¾ÐºÐ° Ð½ÐµÑ‚.")
        ttk.Label(top, text="Ð’Ñ…Ð¾Ð´ÑÑ‰Ð¸Ðµ ÑÐ¸Ð³Ð½Ð°Ð»Ñ‹", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        ttk.Label(top, textvariable=self.signals_summary_var, foreground="#93c5fd").pack(anchor="w", pady=(4, 0))
        self.signals_diag_var = tk.StringVar(value="Signal check: not started.")
        ttk.Label(top, textvariable=self.signals_diag_var, foreground="#a7f3d0").pack(anchor="w", pady=(2, 0))
        self.signals_sources_var = tk.StringVar(value="Sources: v2=0 | v3=0 | total=0")
        ttk.Label(top, textvariable=self.signals_sources_var, foreground="#fcd34d").pack(anchor="w", pady=(2, 0))
        actions = ttk.Frame(top, style="Card.TFrame")
        actions.pack(anchor="w", pady=(8, 0))
        ttk.Button(actions, text="\u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c \u0441\u0438\u0433\u043d\u0430\u043b\u044b", command=self.on_clear_signals).pack(side=tk.LEFT)
        ttk.Button(actions, text="Signal Check", command=self.on_signal_check).pack(side=tk.LEFT, padx=8)

        self.signals_tree = ttk.Treeview(
            self.signals_tab,
            columns=("time", "symbol", "score", "recommendation", "risk", "liquidity", "volume", "change"),
            show="headings",
            height=18,
        )
        for col, title, width in (
            ("time", "Ð’Ñ€ÐµÐ¼Ñ", 110),
            ("symbol", "Ð¡Ð¸Ð¼Ð²Ð¾Ð»", 90),
            ("score", "Score", 80),
            ("recommendation", "Ð ÐµÐºÐ¾Ð¼ÐµÐ½Ð´Ð°Ñ†Ð¸Ñ", 120),
            ("risk", "Ð Ð¸ÑÐº", 90),
            ("liquidity", "Liquidity $", 120),
            ("volume", "Volume 5m $", 120),
            ("change", "Change 5m %", 110),
        ):
            self.signals_tree.heading(col, text=title)
            self.signals_tree.column(col, width=width, anchor="center")
        self.signals_tree.pack(fill=tk.BOTH, expand=True)
        self.signals_tree.tag_configure("row_even", background="#0b1428")
        self.signals_tree.tag_configure("row_odd", background="#0d1a31")

    def _build_matrix_presets_tab(self) -> None:
        top = ttk.Frame(self.matrix_presets_tab, style="Card.TFrame")
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(top, text="Matrix Preset Manager", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        ttk.Label(top, textvariable=self.matrix_profiles_hint_var, foreground="#93c5fd").pack(anchor="w", pady=(4, 0))
        ttk.Label(top, textvariable=self.matrix_recent_winners_var, foreground="#a7f3d0").pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(self.matrix_presets_tab, style="Card.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(left, text="All Profiles (built-in + user)", font=("Segoe UI Semibold", 11)).pack(anchor="w", pady=(0, 6))
        self.matrix_profiles_tree = ttk.Treeview(
            left,
            columns=("kind", "name", "base", "updated"),
            show="headings",
            height=18,
            selectmode="extended",
        )
        for col, title, width in (
            ("kind", "Kind", 90),
            ("name", "Name", 240),
            ("base", "Base", 200),
            ("updated", "Updated", 190),
        ):
            self.matrix_profiles_tree.heading(col, text=title)
            self.matrix_profiles_tree.column(col, width=width, anchor="center")
        self.matrix_profiles_tree.pack(fill=tk.BOTH, expand=True)
        self.matrix_profiles_tree.tag_configure("builtin", foreground="#bfdbfe")
        self.matrix_profiles_tree.tag_configure("user", foreground="#86efac")
        self.matrix_profiles_tree.bind("<Double-1>", lambda _e: self._on_load_selected_user_preset())

        left_actions = ttk.Frame(left, style="Card.TFrame")
        left_actions.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(left_actions, text="Refresh Catalog", command=self._refresh_matrix_catalog).pack(side=tk.LEFT)
        ttk.Button(left_actions, text="Use Selected For Matrix", command=self._on_use_selected_matrix_profiles).pack(side=tk.LEFT, padx=8)
        ttk.Button(left_actions, text="Clear Matrix Selection", command=self._on_clear_matrix_profile_selection).pack(side=tk.LEFT)

        right = ttk.Frame(body, style="Card.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(right, text="Create / Update User Preset", font=("Segoe UI Semibold", 11)).pack(anchor="w", pady=(0, 6))

        form = ttk.Frame(right, style="Card.TFrame")
        form.pack(fill=tk.X)
        form.columnconfigure(0, weight=1)

        ttk.Label(form, text="Preset name", foreground="#bfdbfe").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.matrix_user_name_var, width=46).grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(form, text="Base profile", foreground="#bfdbfe").grid(row=2, column=0, sticky="w")
        self.matrix_user_base_combo = ttk.Combobox(
            form,
            textvariable=self.matrix_user_base_var,
            state="readonly",
            values=(),
            width=44,
        )
        self.matrix_user_base_combo.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(form, text="Note", foreground="#bfdbfe").grid(row=4, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.matrix_user_note_var, width=46).grid(row=5, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(form, text="Overrides editor", foreground="#bfdbfe").grid(row=6, column=0, sticky="w")
        key_row = ttk.Frame(form, style="Card.TFrame")
        key_row.grid(row=7, column=0, sticky="ew", pady=(2, 6))
        key_row.columnconfigure(0, weight=1)
        key_row.columnconfigure(1, weight=1)
        self.matrix_override_key_combo = ttk.Combobox(
            key_row,
            textvariable=self.matrix_override_key_var,
            values=MATRIX_OVERRIDE_COMMON_KEYS,
            width=26,
        )
        self.matrix_override_key_combo.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Entry(key_row, textvariable=self.matrix_override_value_var, width=24).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(key_row, text="Add / Update", command=self._on_add_or_update_override).grid(row=0, column=2, sticky="w")

        self.matrix_overrides_tree = ttk.Treeview(
            form,
            columns=("key", "value"),
            show="headings",
            height=10,
            selectmode="browse",
        )
        self.matrix_overrides_tree.heading("key", text="Key")
        self.matrix_overrides_tree.heading("value", text="Value")
        self.matrix_overrides_tree.column("key", width=210, anchor="w")
        self.matrix_overrides_tree.column("value", width=220, anchor="w")
        self.matrix_overrides_tree.grid(row=8, column=0, sticky="nsew", pady=(0, 6))
        self.matrix_overrides_tree.bind("<<TreeviewSelect>>", self._on_override_row_selected)

        overrides_actions = ttk.Frame(form, style="Card.TFrame")
        overrides_actions.grid(row=9, column=0, sticky="w", pady=(0, 8))
        ttk.Button(overrides_actions, text="Remove Selected", command=self._on_remove_selected_override).pack(side=tk.LEFT)
        ttk.Button(overrides_actions, text="Clear Overrides", command=self._on_clear_overrides).pack(side=tk.LEFT, padx=8)

        right_actions = ttk.Frame(right, style="Card.TFrame")
        right_actions.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(right_actions, text="Create / Update User Preset", command=self._on_create_or_update_user_preset).pack(side=tk.LEFT)
        ttk.Button(right_actions, text="Delete Selected User Preset", command=self._on_delete_selected_user_preset).pack(side=tk.LEFT, padx=8)
        ttk.Button(right_actions, text="Fill Base From Selected", command=self._on_fill_base_from_selected).pack(side=tk.LEFT)
        ttk.Button(right_actions, text="Load Selected User Preset", command=self._on_load_selected_user_preset).pack(side=tk.LEFT, padx=8)

    def _on_fill_base_from_selected(self) -> None:
        if not hasattr(self, "matrix_profiles_tree"):
            return
        sel = self.matrix_profiles_tree.selection()
        if not sel:
            messagebox.showwarning("Matrix Presets", "Select one profile in catalog first.")
            return
        row = self._matrix_catalog_rows.get(sel[0], {})
        name = str(row.get("name", "") or "").strip()
        if not name:
            return
        self.matrix_user_base_var.set(name)
        self.hint_var.set(f"Base profile selected: {name}")

    def _refresh_overrides_tree(self) -> None:
        if not hasattr(self, "matrix_overrides_tree"):
            return
        tree = self.matrix_overrides_tree
        tree.delete(*tree.get_children())
        for idx, key in enumerate(sorted(self._matrix_user_overrides_map.keys())):
            tree.insert("", tk.END, iid=f"ovr:{idx}", values=(key, self._matrix_user_overrides_map[key]))

    def _on_add_or_update_override(self) -> None:
        key = str(self.matrix_override_key_var.get() or "").strip()
        value = str(self.matrix_override_value_var.get() or "").strip()
        if not key:
            messagebox.showwarning("Matrix Presets", "Override key is required.")
            return
        self._matrix_user_overrides_map[key] = value
        self._refresh_overrides_tree()
        self.hint_var.set(f"Override set: {key}={value}")

    def _on_override_row_selected(self, _event=None) -> None:
        if not hasattr(self, "matrix_overrides_tree"):
            return
        sel = self.matrix_overrides_tree.selection()
        if not sel:
            return
        vals = self.matrix_overrides_tree.item(sel[0], "values") or ()
        if len(vals) >= 2:
            self.matrix_override_key_var.set(str(vals[0]))
            self.matrix_override_value_var.set(str(vals[1]))

    def _on_remove_selected_override(self) -> None:
        if not hasattr(self, "matrix_overrides_tree"):
            return
        sel = self.matrix_overrides_tree.selection()
        if not sel:
            return
        vals = self.matrix_overrides_tree.item(sel[0], "values") or ()
        if not vals:
            return
        key = str(vals[0] or "").strip()
        if key in self._matrix_user_overrides_map:
            del self._matrix_user_overrides_map[key]
        self._refresh_overrides_tree()
        self.hint_var.set(f"Override removed: {key}")

    def _on_clear_overrides(self) -> None:
        self._matrix_user_overrides_map = {}
        self._refresh_overrides_tree()
        self.hint_var.set("Overrides cleared.")

    def _on_load_selected_user_preset(self) -> None:
        if not hasattr(self, "matrix_profiles_tree"):
            return
        sel = self.matrix_profiles_tree.selection()
        if not sel:
            messagebox.showwarning("Matrix Presets", "Select user preset row first.")
            return
        row = self._matrix_catalog_rows.get(sel[0], {})
        kind = str(row.get("kind", "") or "").strip()
        name = str(row.get("name", "") or "").strip()
        if kind != "user" or not name:
            messagebox.showwarning("Matrix Presets", "Selected row is not a user preset.")
            return
        self.hint_var.set(f"Loading user preset {name}...")

        def _worker() -> tuple[bool, str]:
            ok, msg = run_powershell_script(
                MATRIX_USER_PRESETS,
                ["show", "--name", name, "--json-only"],
                timeout_seconds=60,
            )
            if not ok:
                return False, msg
            row_data = json.loads(msg)
            if not isinstance(row_data, dict):
                return False, "Invalid user preset payload."

            def _finish() -> None:
                self.matrix_user_name_var.set(str(row_data.get("name", "") or ""))
                self.matrix_user_base_var.set(str(row_data.get("base", "") or ""))
                self.matrix_user_note_var.set(str(row_data.get("note", "") or ""))
                overrides = row_data.get("overrides", {})
                parsed: dict[str, str] = {}
                if isinstance(overrides, dict):
                    for k, v in overrides.items():
                        kk = str(k or "").strip()
                        if kk:
                            parsed[kk] = str(v)
                self._matrix_user_overrides_map = parsed
                self._refresh_overrides_tree()
                self.hint_var.set(f"User preset loaded: {name}")

            self.after(0, _finish)
            return True, "ok"

        def _task() -> None:
            ok, msg = _worker()
            if ok:
                return
            self.after(0, lambda: messagebox.showerror("Matrix Presets", str(msg)[:4000]))
            self.after(0, lambda: self.hint_var.set("User preset load failed."))

        threading.Thread(target=_task, daemon=True).start()

    def _refresh_matrix_catalog(self) -> None:
        self.hint_var.set("Matrix catalog refresh in progress...")

        def _worker() -> tuple[bool, str]:
            ok, msg = run_powershell_script(MATRIX_PROFILE_CATALOG, ["--json"], timeout_seconds=90)
            if not ok:
                return False, msg
            payload = json.loads(msg)
            if not isinstance(payload, dict):
                return False, "Invalid catalog payload."
            ok_allowed, msg_allowed = run_powershell_script(MATRIX_USER_PRESETS, ["allowed", "--json"], timeout_seconds=90)
            allowed_payload: dict = {}
            if ok_allowed:
                try:
                    allowed_payload = json.loads(msg_allowed)
                    if not isinstance(allowed_payload, dict):
                        allowed_payload = {}
                except Exception:
                    allowed_payload = {}

            def _finish() -> None:
                profiles = payload.get("profiles", [])
                winners = payload.get("recent_winners", [])
                self._populate_matrix_catalog_ui(profiles, winners)
                self._apply_allowed_override_keys(allowed_payload)
                self.hint_var.set("Matrix catalog refreshed.")

            self.after(0, _finish)
            return True, "ok"

        def _task() -> None:
            ok, msg = _worker()
            if ok:
                return
            self.after(0, lambda: messagebox.showerror("Matrix Presets", str(msg)[:4000]))
            self.after(0, lambda: self.hint_var.set("Matrix catalog refresh failed."))

        threading.Thread(target=_task, daemon=True).start()

    def _apply_allowed_override_keys(self, payload: dict) -> None:
        if not hasattr(self, "matrix_override_key_combo"):
            return
        allow = payload.get("allowed_keys", {}) if isinstance(payload, dict) else {}
        keys = sorted([str(k) for k in (allow or {}).keys() if str(k).strip()])
        if not keys:
            keys = list(MATRIX_OVERRIDE_COMMON_KEYS)
        self.matrix_override_key_combo.configure(values=tuple(keys))

    def _populate_matrix_catalog_ui(self, profiles: list, winners: list) -> None:
        if not hasattr(self, "matrix_profiles_tree"):
            return
        tree = self.matrix_profiles_tree
        tree.delete(*tree.get_children())
        self._matrix_catalog_rows = {}

        base_candidates: list[str] = []
        for idx, row in enumerate(profiles or []):
            if not isinstance(row, dict):
                continue
            kind = str(row.get("kind", "") or "").strip() or "builtin"
            name = str(row.get("name", "") or "").strip()
            base = str(row.get("base", "") or "").strip()
            upd = str(row.get("updated_at", "") or "").strip()
            if not name:
                continue
            iid = f"{kind}:{name}:{idx}"
            tree.insert("", tk.END, iid=iid, values=(kind, name, base, upd), tags=(kind,))
            self._matrix_catalog_rows[iid] = {"kind": kind, "name": name, "base": base, "updated_at": upd}
            base_candidates.append(name)

        base_candidates = sorted(set(base_candidates))
        if hasattr(self, "matrix_user_base_combo"):
            self.matrix_user_base_combo.configure(values=tuple(base_candidates))
            if not self.matrix_user_base_var.get().strip() and base_candidates:
                self.matrix_user_base_var.set(base_candidates[0])

        winner_parts: list[str] = []
        for row in winners or []:
            if not isinstance(row, dict):
                continue
            winner = str(row.get("winner_id", "") or "").strip()
            if winner:
                winner_parts.append(winner)
        if winner_parts:
            self.matrix_recent_winners_var.set("Recent winners: " + ", ".join(winner_parts[:6]))
        else:
            self.matrix_recent_winners_var.set("Recent winners: --")

    def _on_use_selected_matrix_profiles(self) -> None:
        if not hasattr(self, "matrix_profiles_tree"):
            return
        names: list[str] = []
        for iid in self.matrix_profiles_tree.selection():
            row = self._matrix_catalog_rows.get(iid, {})
            name = str(row.get("name", "") or "").strip()
            if name:
                names.append(name)
        names = list(dict.fromkeys(names))
        if not names:
            messagebox.showwarning("Matrix Presets", "Select profiles in catalog first.")
            return
        if len(names) > 4:
            messagebox.showwarning("Matrix Presets", "Select up to 4 profiles.")
            return
        self._matrix_selected_profile_ids = names
        self.matrix_count_var.set(str(len(names)))
        self.matrix_profiles_hint_var.set("Matrix profiles: " + ",".join(names))
        self.hint_var.set(f"Matrix profile selection set: {','.join(names)}")

    def _on_clear_matrix_profile_selection(self) -> None:
        self._matrix_selected_profile_ids = []
        self.matrix_profiles_hint_var.set("Matrix profiles: default by count")
        self.hint_var.set("Matrix profile selection cleared. Default launcher selection will be used.")

    def _on_create_or_update_user_preset(self) -> None:
        name = str(self.matrix_user_name_var.get() or "").strip()
        base = str(self.matrix_user_base_var.get() or "").strip()
        note = str(self.matrix_user_note_var.get() or "").strip()
        if not name:
            messagebox.showwarning("Matrix Presets", "Preset name is required.")
            return
        if not base:
            messagebox.showwarning("Matrix Presets", "Base profile is required.")
            return
        sets: list[str] = [f"{k}={v}" for k, v in sorted(self._matrix_user_overrides_map.items())]
        args = ["create", "--name", name, "--base", base, "--force"]
        if note:
            args.extend(["--note", note])
        for item in sets:
            args.extend(["--set", item])
        self.hint_var.set(f"Saving user preset {name}...")
        self._run_background(
            title="Matrix User Preset",
            worker=lambda: run_powershell_script(MATRIX_USER_PRESETS, args, timeout_seconds=90),
            on_success=lambda _msg: self._refresh_matrix_catalog(),
            on_error=lambda _msg: self.hint_var.set("User preset save failed."),
        )

    def _on_delete_selected_user_preset(self) -> None:
        if not hasattr(self, "matrix_profiles_tree"):
            return
        sel = self.matrix_profiles_tree.selection()
        if not sel:
            messagebox.showwarning("Matrix Presets", "Select user preset row first.")
            return
        row = self._matrix_catalog_rows.get(sel[0], {})
        kind = str(row.get("kind", "") or "").strip()
        name = str(row.get("name", "") or "").strip()
        if kind != "user" or not name:
            messagebox.showwarning("Matrix Presets", "Selected row is not a user preset.")
            return
        if not messagebox.askyesno("Matrix Presets", f"Delete user preset '{name}'?"):
            return
        self.hint_var.set(f"Deleting user preset {name}...")
        self._run_background(
            title="Matrix User Preset",
            worker=lambda: run_powershell_script(MATRIX_USER_PRESETS, ["delete", "--name", name], timeout_seconds=60),
            on_success=lambda _msg: self._refresh_matrix_catalog(),
            on_error=lambda _msg: self.hint_var.set("User preset delete failed."),
        )

    def _build_settings_tab(self) -> None:
        shell = ttk.Frame(self.settings_tab, style="Card.TFrame")
        shell.pack(fill=tk.BOTH, expand=True)

        self.settings_canvas = tk.Canvas(
            shell,
            bg="#111827",
            highlightthickness=0,
            relief=tk.FLAT,
        )
        settings_scroll = ttk.Scrollbar(shell, orient="vertical", command=self.settings_canvas.yview)
        self.settings_canvas.configure(yscrollcommand=settings_scroll.set)
        settings_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.settings_content = ttk.Frame(self.settings_canvas, style="Card.TFrame")
        self.settings_window = self.settings_canvas.create_window(
            (0, 0),
            window=self.settings_content,
            anchor="nw",
        )
        self.settings_content.bind(
            "<Configure>",
            lambda _e: self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all")),
        )
        self.settings_canvas.bind(
            "<Configure>",
            lambda e: self.settings_canvas.itemconfigure(self.settings_window, width=e.width),
        )
        self.settings_canvas.bind("<Enter>", self._bind_settings_scroll)
        self.settings_canvas.bind("<Leave>", self._unbind_settings_scroll)

        title = ttk.Label(self.settings_content, text="Runtime Settings (.env)", font=("Segoe UI Semibold", 12))
        title.pack(anchor="w", pady=(0, 10))

        grid = ttk.Frame(self.settings_content, style="Card.TFrame")
        grid.pack(fill=tk.X)
        grid.columnconfigure(0, weight=1)

        for idx, (key, label) in enumerate(GUI_SETTINGS_FIELDS):
            row = idx * 2
            ttk.Label(
                grid,
                text=label,
                foreground="#bfdbfe",
                font=("Segoe UI Semibold", 10),
            ).grid(row=row, column=0, sticky="w", pady=(6, 2))
            var = tk.StringVar()
            self.env_vars[key] = var
            if key in FIELD_OPTIONS:
                combo = ttk.Combobox(
                    grid,
                    textvariable=var,
                    values=FIELD_OPTIONS[key],
                    width=54,
                    state="readonly",
                )
                combo.grid(row=row + 1, column=0, sticky="ew", pady=(0, 6))
            else:
                ttk.Entry(grid, textvariable=var, width=57).grid(row=row + 1, column=0, sticky="ew", pady=(0, 6))

        btns = ttk.Frame(self.settings_content, style="Card.TFrame")
        btns.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(btns, text="Reload .env", command=self._load_settings).pack(side=tk.LEFT)
        ttk.Button(btns, text="Save", command=self._save_settings).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save + Restart", command=self._save_restart).pack(side=tk.LEFT)

        tips = (
            "Tips:\n"
            "- Validate presets in matrix before live.\n"
            "- Keep policy router enabled to avoid full dead-stop in degraded windows.\n"
            "- Tune entry edge + exits together; isolated tweaks usually hurt EV.\n"
            "- After changing runtime settings, restart the bot process."
        )
        ttk.Label(self.settings_content, text=tips, foreground="#94a3b8").pack(anchor="w", pady=(16, 0))

    def _build_wallet_tab(self) -> None:
        top = ttk.Frame(self.wallet_tab, style="Card.TFrame")
        top.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(top, text="Wallet Center", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        self.wallet_summary_var = tk.StringVar(value="No wallet data yet.")
        ttk.Label(top, textvariable=self.wallet_summary_var, foreground="#93c5fd").pack(anchor="w", pady=(4, 0))
        self.live_onchain_var = tk.StringVar(value="Live on-chain: (not checked yet)")
        ttk.Label(top, textvariable=self.live_onchain_var, foreground="#a7f3d0").pack(anchor="w", pady=(4, 0))

        grid = ttk.Frame(self.wallet_tab, style="Card.TFrame")
        grid.pack(fill=tk.X, pady=(6, 0))
        grid.columnconfigure(1, weight=1)

        self.wallet_mode_var = tk.StringVar(value="paper")
        ttk.Label(grid, text="Wallet mode", foreground="#bfdbfe", font=("Segoe UI Semibold", 10)).grid(
            row=0, column=0, sticky="w", pady=(6, 2)
        )
        ttk.Combobox(
            grid,
            textvariable=self.wallet_mode_var,
            values=["paper", "live"],
            width=18,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", pady=(6, 2))
        ttk.Button(grid, text="Apply mode", command=self.on_apply_wallet_mode).grid(row=0, column=2, sticky="w", padx=8)

        self.paper_balance_set_var = tk.StringVar(value="2.75")
        ttk.Label(grid, text="Set paper balance ($)", foreground="#bfdbfe", font=("Segoe UI Semibold", 10)).grid(
            row=1, column=0, sticky="w", pady=(6, 2)
        )
        ttk.Entry(grid, textvariable=self.paper_balance_set_var, width=20).grid(row=1, column=1, sticky="w", pady=(6, 2))
        ttk.Button(grid, text="Apply paper", command=self.on_apply_paper_balance).grid(
            row=1, column=2, sticky="w", padx=8
        )

        self.live_balance_var = tk.StringVar(value="0")
        ttk.Label(grid, text="Live balance ($)", foreground="#bfdbfe", font=("Segoe UI Semibold", 10)).grid(
            row=2, column=0, sticky="w", pady=(6, 2)
        )
        ttk.Entry(grid, textvariable=self.live_balance_var, width=20).grid(row=2, column=1, sticky="w", pady=(6, 2))
        ttk.Button(grid, text="Apply live", command=self.on_apply_live_balance).grid(row=2, column=2, sticky="w", padx=8)

        self.stair_enabled_var = tk.StringVar(value="false")
        ttk.Label(grid, text="Step protection", foreground="#bfdbfe", font=("Segoe UI Semibold", 10)).grid(
            row=3, column=0, sticky="w", pady=(6, 2)
        )
        ttk.Combobox(
            grid,
            textvariable=self.stair_enabled_var,
            values=["false", "true"],
            width=18,
            state="readonly",
        ).grid(row=3, column=1, sticky="w", pady=(6, 2))
        ttk.Button(grid, text="Apply step", command=self.on_apply_stair_mode).grid(row=3, column=2, sticky="w", padx=8)

        self.stair_start_var = tk.StringVar(value="2.00")
        ttk.Label(grid, text="Step start floor ($)", foreground="#bfdbfe", font=("Segoe UI Semibold", 10)).grid(
            row=4, column=0, sticky="w", pady=(6, 2)
        )
        ttk.Entry(grid, textvariable=self.stair_start_var, width=20).grid(row=4, column=1, sticky="w", pady=(6, 2))

        self.stair_size_var = tk.StringVar(value="2")
        ttk.Label(grid, text="Step size ($)", foreground="#bfdbfe", font=("Segoe UI Semibold", 10)).grid(
            row=5, column=0, sticky="w", pady=(6, 2)
        )
        ttk.Entry(grid, textvariable=self.stair_size_var, width=20).grid(row=5, column=1, sticky="w", pady=(6, 2))
        ttk.Button(grid, text="Apply step params", command=self.on_apply_stair_params).grid(
            row=5, column=2, sticky="w", padx=8
        )

        ttk.Button(grid, text="Set paper 2.75", command=self.on_set_paper_275).grid(row=6, column=1, sticky="w", pady=(10, 2))
        ttk.Button(grid, text="Refresh wallet", command=self._refresh_wallet).grid(row=6, column=2, sticky="w", padx=8, pady=(10, 2))
        ttk.Label(
            grid,
            text="Critical actions moved to tab: ÐÐ²Ð°Ñ€Ð¸Ð¹Ð½Ñ‹Ð¹",
            foreground="#fca5a5",
            font=("Segoe UI", 10),
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(12, 2))

    def _build_emergency_tab(self) -> None:
        top = ttk.Frame(self.emergency_tab, style="Card.TFrame")
        top.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(top, text="ÐÐ²Ð°Ñ€Ð¸Ð¹Ð½Ñ‹Ð¹ Ð±Ð»Ð¾Ðº ÑƒÐ¿Ñ€Ð°Ð²Ð»ÐµÐ½Ð¸Ñ", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        self.emergency_summary_var = tk.StringVar(
            value="KILL_SWITCH: OFF | CRITICAL_AUTO_RESET: 0 | live_failed: 0 | errors: 0"
        )
        ttk.Label(top, textvariable=self.emergency_summary_var, foreground="#fca5a5").pack(anchor="w", pady=(4, 0))

        actions = ttk.Frame(top, style="Card.TFrame")
        actions.pack(anchor="w", pady=(8, 0))
        ttk.Button(actions, text="Ð’ÐºÐ»ÑŽÑ‡Ð¸Ñ‚ÑŒ KILL SWITCH", command=self.on_enable_kill_switch).pack(side=tk.LEFT)
        ttk.Button(actions, text="Ð’Ñ‹ÐºÐ»ÑŽÑ‡Ð¸Ñ‚ÑŒ KILL SWITCH", command=self.on_disable_kill_switch).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="ÐšÑ€Ð¸Ñ‚Ð¸Ñ‡ÐµÑÐºÐ¸Ð¹ ÑÐ±Ñ€Ð¾Ñ", command=self.on_critical_reset).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Signal Check", command=self.on_signal_check).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="ÐžÑ‡Ð¸ÑÑ‚Ð¸Ñ‚ÑŒ Ð»Ð¾Ð³Ð¸", command=self.on_clear_logs).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="ÐžÑ‡Ð¸ÑÑ‚Ð¸Ñ‚ÑŒ GUI", command=self.on_clear_gui).pack(side=tk.LEFT, padx=8)

        self.critical_text = tk.Text(
            self.emergency_tab,
            bg="#091224",
            fg="#fecaca",
            insertbackground="#fecaca",
            relief=tk.FLAT,
            wrap="word",
            font=(UI_FONT_MONO, 10),
        )
        self.critical_text.pack(fill=tk.BOTH, expand=True)
        self._attach_text_copy_support(self.critical_text)

    def _bind_settings_scroll(self, _event=None) -> None:
        self.bind_all("<MouseWheel>", self._on_settings_mousewheel)
        self.bind_all("<Button-4>", self._on_settings_scroll_up)
        self.bind_all("<Button-5>", self._on_settings_scroll_down)

    def _unbind_settings_scroll(self, _event=None) -> None:
        self.unbind_all("<MouseWheel>")
        self.unbind_all("<Button-4>")
        self.unbind_all("<Button-5>")

    def _on_settings_mousewheel(self, event) -> None:
        if hasattr(self, "settings_canvas"):
            self.settings_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_settings_scroll_up(self, _event=None) -> None:
        if hasattr(self, "settings_canvas"):
            self.settings_canvas.yview_scroll(-1, "units")

    def _on_settings_scroll_down(self, _event=None) -> None:
        if hasattr(self, "settings_canvas"):
            self.settings_canvas.yview_scroll(1, "units")

    def _build_positions_tab(self) -> None:
        top = ttk.Frame(self.positions_tab, style="Card.TFrame")
        top.pack(fill=tk.X, pady=(0, 10))
        self.pos_summary_var = tk.StringVar(value="ÐŸÐ¾ÐºÐ° Ð½ÐµÑ‚ Ð´Ð°Ð½Ð½Ñ‹Ñ… Ð¿Ð¾ ÑÐ´ÐµÐ»ÐºÐ°Ð¼.")
        self.positions_title_var = tk.StringVar(value="ÐœÐ¾Ð½Ð¸Ñ‚Ð¾Ñ€ ÑÐ´ÐµÐ»Ð¾Ðº")
        ttk.Label(top, textvariable=self.positions_title_var, font=("Segoe UI Semibold", 12)).pack(anchor="w")
        ttk.Label(top, textvariable=self.pos_summary_var, foreground="#93c5fd").pack(anchor="w", pady=(4, 0))
        self.untracked_summary_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.untracked_summary_var, foreground="#fcd34d").pack(anchor="w", pady=(2, 0))
        kpi = ttk.Frame(top, style="Card.TFrame")
        kpi.pack(fill=tk.X, pady=(8, 4))
        for idx in range(6):
            kpi.columnconfigure(idx, weight=1)
        self.kpi_open_var = tk.StringVar(value="0")
        self.kpi_closed_var = tk.StringVar(value="0")
        self.kpi_winrate_var = tk.StringVar(value="0%")
        self.kpi_pnl_var = tk.StringVar(value="$+0.00")
        self.kpi_pnl_1h_var = tk.StringVar(value="$+0.00")
        self.kpi_updated_var = tk.StringVar(value="--:--:--")
        cards = (
            ("Open", self.kpi_open_var),
            ("Closed", self.kpi_closed_var),
            ("Winrate", self.kpi_winrate_var),
            ("Realized PnL", self.kpi_pnl_var),
            ("PnL 1h", self.kpi_pnl_1h_var),
            ("Updated", self.kpi_updated_var),
        )
        for idx, (title, var) in enumerate(cards):
            card = ttk.Frame(kpi, style="KPI.TFrame", padding=(10, 8))
            card.grid(row=0, column=idx, sticky="nsew", padx=(0 if idx == 0 else 6, 0))
            ttk.Label(card, text=title, style="KPILabel.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=var, style="KPIValue.TLabel").pack(anchor="w", pady=(3, 0))
        actions = ttk.Frame(top, style="Card.TFrame")
        actions.pack(anchor="w", pady=(8, 0))
        ttk.Label(actions, text="Ð˜ÑÑ‚Ð¾Ñ‡Ð½Ð¸Ðº:").pack(side=tk.LEFT)
        self.positions_source_combo = ttk.Combobox(
            actions,
            width=22,
            state="readonly",
            textvariable=self.positions_source_var,
            values=("single",),
        )
        self.positions_source_combo.pack(side=tk.LEFT, padx=(6, 10))
        self.positions_source_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        ttk.Button(actions, text="Ð£Ð´Ð°Ð»Ð¸Ñ‚ÑŒ ÑÑ‚Ð°Ñ€Ñ‹Ðµ Ð·Ð°ÐºÑ€Ñ‹Ñ‚Ñ‹Ðµ", command=self.on_prune_closed_trades).pack(side=tk.LEFT)
        ttk.Button(actions, text="ÐžÑ‡Ð¸ÑÑ‚Ð¸Ñ‚ÑŒ Ð²ÑÐµ Ð·Ð°ÐºÑ€Ñ‹Ñ‚Ñ‹Ðµ", command=self.on_clear_closed_trades).pack(side=tk.LEFT, padx=8)
        idle = ttk.Frame(top, style="Card.TFrame")
        idle.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(idle, text="ÐŸÑ€Ð¸Ñ‡Ð¸Ð½Ñ‹ Ð¿Ñ€Ð¾ÑÑ‚Ð¾Ñ (15Ð¼)", font=("Segoe UI Semibold", 11)).pack(anchor="w")
        self.idle_reasons_var = tk.StringVar(value="--")
        ttk.Label(idle, textvariable=self.idle_reasons_var, foreground="#fcd34d").pack(anchor="w", pady=(4, 0))
        ttk.Label(idle, text="Ð‘Ð»Ð¾Ðº Ð¿Ð¾ÑÐ»Ðµ pass (live 15Ð¼)", font=("Segoe UI Semibold", 10)).pack(anchor="w", pady=(6, 0))
        self.live_block_reasons_var = tk.StringVar(value="--")
        ttk.Label(idle, textvariable=self.live_block_reasons_var, foreground="#fda4af").pack(anchor="w", pady=(2, 0))

        panels = ttk.Frame(self.positions_tab, style="Card.TFrame")
        panels.pack(fill=tk.BOTH, expand=True)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(0, weight=1)

        left = ttk.Frame(panels, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(left, text="ÐžÑ‚ÐºÑ€Ñ‹Ñ‚Ñ‹Ðµ Ð¿Ð¾Ð·Ð¸Ñ†Ð¸Ð¸", font=("Segoe UI Semibold", 11)).pack(anchor="w", pady=(0, 6))
        self.open_tree = ttk.Treeview(
            left,
            columns=("symbol", "entry", "size", "score", "opened", "pnl_pct", "pnl_usd", "pnl_hour", "peak_pct", "ttl"),
            show="headings",
            height=14,
        )
        for col, title, width in (
            ("symbol", "Ð¡Ð¸Ð¼Ð²Ð¾Ð»", 90),
            ("entry", "Ð’Ñ…Ð¾Ð´ $", 120),
            ("size", "Ð Ð°Ð·Ð¼ÐµÑ€ $", 90),
            ("score", "Ð¡ÐºÐ¾Ñ€", 80),
            ("opened", "Opened", 130),
            ("ttl", "ÐžÑÑ‚Ð°Ð»Ð¾ÑÑŒ (ÑÐµÐº)", 120),
        ):
            self.open_tree.heading(col, text=title)
            self.open_tree.column(col, width=width, anchor="center")
        self.open_tree.heading("pnl_pct", text="PnL %")
        self.open_tree.heading("pnl_usd", text="PnL $")
        self.open_tree.heading("pnl_hour", text="PnL/hour $")
        self.open_tree.heading("peak_pct", text="Peak %")
        self.open_tree.column("pnl_pct", width=80, anchor="center")
        self.open_tree.column("pnl_usd", width=80, anchor="center")
        self.open_tree.column("pnl_hour", width=95, anchor="center")
        self.open_tree.column("peak_pct", width=80, anchor="center")
        try:
            open_columns = list(self.open_tree["columns"])
            self._open_ttl_col_idx = open_columns.index("ttl")
            self._open_pnl_hour_col_idx = open_columns.index("pnl_hour")
        except Exception:
            self._open_ttl_col_idx = -1
            self._open_pnl_hour_col_idx = -1
        self.open_tree.pack(fill=tk.BOTH, expand=True)
        self.open_tree.tag_configure("row_even", background="#0b1428")
        self.open_tree.tag_configure("row_odd", background="#0d1a31")
        self.open_tree.tag_configure("open_pos", foreground="#86efac")
        self.open_tree.tag_configure("open_neg", foreground="#fda4af")
        self.open_tree.tag_configure("open_flat", foreground="#bfdbfe")
        self.open_tree.tag_configure("open_untracked", foreground="#fcd34d")

        right = ttk.Frame(panels, style="Card.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(right, text="Ð—Ð°ÐºÑ€Ñ‹Ñ‚Ñ‹Ðµ Ð¿Ð¾Ð·Ð¸Ñ†Ð¸Ð¸", font=("Segoe UI Semibold", 11)).pack(anchor="w", pady=(0, 6))
        self.closed_tree = ttk.Treeview(
            right,
            columns=("symbol", "reason", "opened", "closed", "pnl_pct", "pnl_usd"),
            show="headings",
            height=14,
        )
        for col, title, width in (
            ("symbol", "Ð¡Ð¸Ð¼Ð²Ð¾Ð»", 110),
            ("reason", "Ð’Ñ‹Ñ…Ð¾Ð´", 90),
            ("opened", "Opened", 130),
            ("closed", "Closed", 130),
            ("pnl_pct", "PnL %", 90),
            ("pnl_usd", "PnL $", 90),
        ):
            self.closed_tree.heading(col, text=title)
            self.closed_tree.column(col, width=width, anchor="center")
        self.closed_tree.pack(fill=tk.BOTH, expand=True)
        self.closed_tree.tag_configure("row_even", background="#0b1428")
        self.closed_tree.tag_configure("row_odd", background="#0d1a31")
        self.closed_tree.tag_configure("closed_pos", foreground="#86efac")
        self.closed_tree.tag_configure("closed_neg", foreground="#fda4af")
        self.closed_tree.tag_configure("closed_flat", foreground="#bfdbfe")

    @staticmethod
    def _to_float(raw: str | None, default: float) -> float:
        try:
            if raw is None:
                return default
            return float(str(raw).strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(raw: object, default: float = 0.0) -> float:
        try:
            value = float(raw)
            if not math.isfinite(value):
                return default
            return value
        except Exception:
            return default

    @staticmethod
    def _parse_iso_ts(raw: object) -> datetime | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    @classmethod
    def _format_local_ts(cls, raw: object) -> str:
        dt = cls._parse_iso_ts(raw)
        if dt is None:
            return "--"
        return dt.astimezone().strftime("%m-%d %H:%M:%S")

    def _effective_wallet_balance(self, preset: dict[str, str]) -> float:
        if "WALLET_BALANCE_USD" in preset:
            return max(0.1, self._to_float(preset.get("WALLET_BALANCE_USD"), 2.75))
        env_value = self.env_vars.get("WALLET_BALANCE_USD")
        if env_value and env_value.get().strip():
            return max(0.1, self._to_float(env_value.get(), 2.75))
        env_map = read_runtime_env_map()
        return max(0.1, self._to_float(env_map.get("WALLET_BALANCE_USD"), 2.75))

    def _adapt_preset_by_balance(self, preset: dict[str, str]) -> tuple[dict[str, str], float, str]:
        tuned = dict(preset)
        balance = self._effective_wallet_balance(tuned)

        if balance <= 5:
            # Small-bank profile: keep flow alive on tiny balance while preserving guardrails.
            tuned.update(
                {
                    "AUTO_TRADE_ENTRY_MODE": "top_n",
                    "AUTO_TRADE_TOP_N": str(min(3, max(2, int(self._to_float(tuned.get("AUTO_TRADE_TOP_N"), 2))))),
                    "MAX_OPEN_TRADES": str(min(2, max(1, int(self._to_float(tuned.get("MAX_OPEN_TRADES"), 2))))),
                    "PAPER_TRADE_SIZE_USD": "0.30",
                    "PAPER_TRADE_SIZE_MIN_USD": "0.12",
                    "PAPER_TRADE_SIZE_MAX_USD": "0.45",
                    "MAX_LOSS_PER_TRADE_PERCENT_BALANCE": str(max(6.0, self._to_float(tuned.get("MAX_LOSS_PER_TRADE_PERCENT_BALANCE"), 6.0))),
                    "PAPER_GAS_PER_TX_USD": "0.02",
                }
            )
            if str(tuned.get("SAFE_TEST_MODE", "")).lower() == "true":
                tuned["SAFE_MIN_LIQUIDITY_USD"] = str(min(14000, int(self._to_float(tuned.get("SAFE_MIN_LIQUIDITY_USD"), 14000))))
                tuned["SAFE_MIN_VOLUME_5M_USD"] = str(min(800, int(self._to_float(tuned.get("SAFE_MIN_VOLUME_5M_USD"), 800))))
                tuned["SAFE_MIN_AGE_SECONDS"] = str(min(45, int(self._to_float(tuned.get("SAFE_MIN_AGE_SECONDS"), 45))))
            return tuned, balance, "small-bank"

        if balance <= 25:
            tuned.update(
                {
                    "AUTO_TRADE_ENTRY_MODE": str(tuned.get("AUTO_TRADE_ENTRY_MODE", "top_n")).lower()
                    if str(tuned.get("AUTO_TRADE_ENTRY_MODE", "top_n")).lower() in {"single", "top_n", "all"}
                    else "top_n",
                    "AUTO_TRADE_TOP_N": str(max(3, int(self._to_float(tuned.get("AUTO_TRADE_TOP_N"), 5)))),
                    "MAX_OPEN_TRADES": str(min(3, max(1, int(self._to_float(tuned.get("MAX_OPEN_TRADES"), 2))))),
                    "MAX_LOSS_PER_TRADE_PERCENT_BALANCE": str(max(5.0, self._to_float(tuned.get("MAX_LOSS_PER_TRADE_PERCENT_BALANCE"), 5.0))),
                    "PAPER_GAS_PER_TX_USD": str(min(0.03, self._to_float(tuned.get("PAPER_GAS_PER_TX_USD"), 0.03))),
                }
            )
            return tuned, balance, "mid-bank"

        return tuned, balance, "large-bank"

    def _apply_preset_map(self, preset: dict[str, str]) -> None:
        tuned, balance, profile = self._adapt_preset_by_balance(preset)
        normalized = dict(tuned)
        legacy_max_trades = str(normalized.get("MAX_TRADES_PER_HOUR", "") or "").strip()
        max_buys = str(normalized.get("MAX_BUYS_PER_HOUR", "") or "").strip()
        if legacy_max_trades and not max_buys:
            max_buys = legacy_max_trades
            normalized["MAX_BUYS_PER_HOUR"] = max_buys
        if max_buys:
            # Keep legacy key aligned for fallback code paths.
            normalized["MAX_TRADES_PER_HOUR"] = max_buys
        extra_env_values: dict[str, str] = {}
        for key, value in normalized.items():
            if key in self.env_vars:
                self.env_vars[key].set(value)
            else:
                extra_env_values[key] = value
        if extra_env_values:
            save_runtime_env_map(extra_env_values)
        if hasattr(self, "hint_var"):
            self.hint_var.set(f"Preset auto-tuned for balance ${balance:.2f} ({profile}).")

    def _apply_super_safe_preset(self) -> None:
        self._apply_preset_map(
            {
                "SIGNAL_SOURCE": "onchain",
                "ONCHAIN_ENABLE_UNISWAP_V3": "true",
                "ONCHAIN_POLL_INTERVAL_SECONDS": "5",
                "DEX_SEARCH_QUERIES": "base",
                "GECKO_NEW_POOLS_PAGES": "1",
                "DEX_BOOSTS_SOURCE_ENABLED": "false",
                "DEX_BOOSTS_MAX_TOKENS": "10",
                "AUTO_TRADE_ENABLED": "true",
                "AUTO_TRADE_PAPER": "true",
                "AUTO_FILTER_ENABLED": "true",
                "AUTO_TRADE_ENTRY_MODE": "single",
                "AUTO_TRADE_TOP_N": "1",
                "MAX_BUYS_PER_HOUR": "2",
                "MAX_BUY_AMOUNT": "0.00020",
                "MIN_TOKEN_SCORE": "82",
                "SAFE_TEST_MODE": "true",
                "SAFE_MIN_LIQUIDITY_USD": "35000",
                "SAFE_MIN_VOLUME_5M_USD": "9000",
                "SAFE_MIN_AGE_SECONDS": "180",
                "SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT": "12",
                "SAFE_REQUIRE_CONTRACT_SAFE": "true",
                "SAFE_REQUIRE_RISK_LEVEL": "LOW",
                "SAFE_MAX_WARNING_FLAGS": "0",
                "MAX_OPEN_TRADES": "1",
                "PAPER_TRADE_SIZE_USD": "0.35",
                "PAPER_TRADE_SIZE_MIN_USD": "0.20",
                "PAPER_TRADE_SIZE_MAX_USD": "0.45",
                "MAX_LOSS_PER_TRADE_PERCENT_BALANCE": "0.7",
                "DAILY_MAX_DRAWDOWN_PERCENT": "2.5",
                "MAX_CONSECUTIVE_LOSSES": "2",
                "LOSS_STREAK_COOLDOWN_SECONDS": "3600",
                "MAX_TOKEN_COOLDOWN_SECONDS": "1200",
                "MAX_TOKEN_PRICE_CHANGE_5M_ABS_PERCENT": "20",
                "EDGE_FILTER_ENABLED": "true",
                "MIN_EXPECTED_EDGE_PERCENT": "4.0",
                "PROFIT_TARGET_PERCENT": "35",
                "STOP_LOSS_PERCENT": "12",
                "PAPER_BASE_SLIPPAGE_BPS": "50",
            }
        )

    def _apply_medium_preset(self) -> None:
        self._apply_preset_map(
            {
                "SIGNAL_SOURCE": "onchain",
                "ONCHAIN_ENABLE_UNISWAP_V3": "true",
                "ONCHAIN_POLL_INTERVAL_SECONDS": "5",
                "DEX_SEARCH_QUERIES": "base",
                "GECKO_NEW_POOLS_PAGES": "1",
                "DEX_BOOSTS_SOURCE_ENABLED": "false",
                "DEX_BOOSTS_MAX_TOKENS": "10",
                "AUTO_TRADE_ENABLED": "true",
                "AUTO_TRADE_PAPER": "true",
                "AUTO_FILTER_ENABLED": "true",
                "AUTO_TRADE_ENTRY_MODE": "top_n",
                "AUTO_TRADE_TOP_N": "2",
                "MAX_BUYS_PER_HOUR": "4",
                "MAX_BUY_AMOUNT": "0.00030",
                "MIN_TOKEN_SCORE": "75",
                "SAFE_TEST_MODE": "true",
                "SAFE_MIN_LIQUIDITY_USD": "22000",
                "SAFE_MIN_VOLUME_5M_USD": "5000",
                "SAFE_MIN_AGE_SECONDS": "120",
                "SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT": "18",
                "SAFE_REQUIRE_CONTRACT_SAFE": "true",
                "SAFE_REQUIRE_RISK_LEVEL": "MEDIUM",
                "SAFE_MAX_WARNING_FLAGS": "1",
                "MAX_OPEN_TRADES": "2",
                "PAPER_TRADE_SIZE_USD": "0.60",
                "PAPER_TRADE_SIZE_MIN_USD": "0.25",
                "PAPER_TRADE_SIZE_MAX_USD": "0.85",
                "MAX_LOSS_PER_TRADE_PERCENT_BALANCE": "1.1",
                "DAILY_MAX_DRAWDOWN_PERCENT": "4.0",
                "MAX_CONSECUTIVE_LOSSES": "3",
                "LOSS_STREAK_COOLDOWN_SECONDS": "2400",
                "MAX_TOKEN_COOLDOWN_SECONDS": "900",
                "MAX_TOKEN_PRICE_CHANGE_5M_ABS_PERCENT": "30",
                "EDGE_FILTER_ENABLED": "true",
                "MIN_EXPECTED_EDGE_PERCENT": "2.5",
                "PROFIT_TARGET_PERCENT": "40",
                "STOP_LOSS_PERCENT": "15",
                "PAPER_BASE_SLIPPAGE_BPS": "60",
            }
        )

    def _apply_hard_preset(self) -> None:
        self._apply_preset_map(
            {
                "SIGNAL_SOURCE": "onchain",
                "ONCHAIN_ENABLE_UNISWAP_V3": "true",
                "ONCHAIN_POLL_INTERVAL_SECONDS": "4",
                "DEX_SEARCH_QUERIES": "base",
                "GECKO_NEW_POOLS_PAGES": "1",
                "DEX_BOOSTS_SOURCE_ENABLED": "false",
                "DEX_BOOSTS_MAX_TOKENS": "10",
                "AUTO_TRADE_ENABLED": "true",
                "AUTO_TRADE_PAPER": "true",
                "AUTO_FILTER_ENABLED": "false",
                "AUTO_TRADE_ENTRY_MODE": "all",
                "AUTO_TRADE_TOP_N": "10",
                "MAX_BUYS_PER_HOUR": "8",
                "MAX_BUY_AMOUNT": "0.00050",
                "MIN_TOKEN_SCORE": "65",
                "SAFE_TEST_MODE": "false",
                "SAFE_REQUIRE_CONTRACT_SAFE": "true",
                "SAFE_REQUIRE_RISK_LEVEL": "MEDIUM",
                "SAFE_MAX_WARNING_FLAGS": "2",
                "MAX_OPEN_TRADES": "4",
                "PAPER_TRADE_SIZE_USD": "1.20",
                "PAPER_TRADE_SIZE_MIN_USD": "0.35",
                "PAPER_TRADE_SIZE_MAX_USD": "1.80",
                "MAX_LOSS_PER_TRADE_PERCENT_BALANCE": "1.8",
                "DAILY_MAX_DRAWDOWN_PERCENT": "6.0",
                "MAX_CONSECUTIVE_LOSSES": "4",
                "LOSS_STREAK_COOLDOWN_SECONDS": "1800",
                "MAX_TOKEN_COOLDOWN_SECONDS": "600",
                "MAX_TOKEN_PRICE_CHANGE_5M_ABS_PERCENT": "40",
                "EDGE_FILTER_ENABLED": "false",
                "MIN_EXPECTED_EDGE_PERCENT": "0.0",
                "PROFIT_TARGET_PERCENT": "45",
                "STOP_LOSS_PERCENT": "20",
                "PAPER_BASE_SLIPPAGE_BPS": "90",
            }
        )

    def _apply_hard_lite_preset(self) -> None:
        self._apply_preset_map(
            {
                "SIGNAL_SOURCE": "onchain",
                "ONCHAIN_ENABLE_UNISWAP_V3": "true",
                "ONCHAIN_POLL_INTERVAL_SECONDS": "5",
                "DEX_SEARCH_QUERIES": "base",
                "GECKO_NEW_POOLS_PAGES": "1",
                "DEX_BOOSTS_SOURCE_ENABLED": "false",
                "DEX_BOOSTS_MAX_TOKENS": "10",
                "AUTO_TRADE_ENABLED": "true",
                "AUTO_TRADE_PAPER": "true",
                "AUTO_FILTER_ENABLED": "false",
                "AUTO_TRADE_ENTRY_MODE": "single",
                "AUTO_TRADE_TOP_N": "3",
                "MAX_BUYS_PER_HOUR": "4",
                "MAX_BUY_AMOUNT": "0.00035",
                "MIN_TOKEN_SCORE": "65",
                "SAFE_TEST_MODE": "false",
                "SAFE_REQUIRE_CONTRACT_SAFE": "true",
                "SAFE_REQUIRE_RISK_LEVEL": "MEDIUM",
                "SAFE_MAX_WARNING_FLAGS": "2",
                "MAX_OPEN_TRADES": "1",
                "PAPER_TRADE_SIZE_USD": "0.55",
                "PAPER_TRADE_SIZE_MIN_USD": "0.20",
                "PAPER_TRADE_SIZE_MAX_USD": "0.90",
                "MAX_LOSS_PER_TRADE_PERCENT_BALANCE": "8.0",
                "DAILY_MAX_DRAWDOWN_PERCENT": "8.0",
                "MAX_CONSECUTIVE_LOSSES": "4",
                "LOSS_STREAK_COOLDOWN_SECONDS": "1200",
                "MAX_TOKEN_COOLDOWN_SECONDS": "450",
                "MAX_TOKEN_PRICE_CHANGE_5M_ABS_PERCENT": "35",
                "EDGE_FILTER_ENABLED": "false",
                "MIN_EXPECTED_EDGE_PERCENT": "0.0",
                "PROFIT_TARGET_PERCENT": "45",
                "STOP_LOSS_PERCENT": "20",
                "PAPER_GAS_PER_TX_USD": "0.02",
                "PAPER_BASE_SLIPPAGE_BPS": "80",
            }
        )

    def _apply_safe_flow_preset(self) -> None:
        self._apply_preset_map(
            {
                "SIGNAL_SOURCE": "onchain",
                "ONCHAIN_ENABLE_UNISWAP_V3": "true",
                "ONCHAIN_POLL_INTERVAL_SECONDS": "4",
                "DEX_SEARCH_QUERIES": "base",
                "GECKO_NEW_POOLS_PAGES": "1",
                "DEX_BOOSTS_SOURCE_ENABLED": "false",
                "DEX_BOOSTS_MAX_TOKENS": "10",
                "AUTO_TRADE_ENABLED": "true",
                "AUTO_TRADE_PAPER": "true",
                "AUTO_FILTER_ENABLED": "true",
                "AUTO_TRADE_ENTRY_MODE": "top_n",
                "AUTO_TRADE_TOP_N": "3",
                "AUTONOMOUS_CONTROL_ENABLED": "true",
                "AUTONOMOUS_CONTROL_MODE": "apply",
                "AUTONOMOUS_CONTROL_INTERVAL_SECONDS": "300",
                "AUTONOMOUS_CONTROL_MIN_WINDOW_CYCLES": "2",
                "AUTONOMOUS_CONTROL_TARGET_CANDIDATES_MIN": "1.2",
                "AUTONOMOUS_CONTROL_TARGET_CANDIDATES_HIGH": "5.0",
                "AUTONOMOUS_CONTROL_TARGET_OPENED_MIN": "0.12",
                "AUTONOMOUS_CONTROL_NEG_REALIZED_TRIGGER_USD": "0.05",
                "AUTONOMOUS_CONTROL_POS_REALIZED_TRIGGER_USD": "0.05",
                "AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MIN": "1",
                "AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MAX": "3",
                "AUTONOMOUS_CONTROL_TOP_N_MIN": "2",
                "AUTONOMOUS_CONTROL_TOP_N_MAX": "8",
                "AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN": "12",
                "AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MAX": "36",
                "MAX_BUYS_PER_HOUR": "6",
                "MAX_BUY_AMOUNT": "0.00025",
                "MIN_TOKEN_SCORE": "70",
                "SAFE_TEST_MODE": "true",
                "SAFE_MIN_LIQUIDITY_USD": "14000",
                "SAFE_MIN_VOLUME_5M_USD": "800",
                "SAFE_MIN_AGE_SECONDS": "45",
                "SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT": "24",
                "SAFE_REQUIRE_CONTRACT_SAFE": "true",
                "SAFE_REQUIRE_RISK_LEVEL": "MEDIUM",
                "SAFE_MAX_WARNING_FLAGS": "2",
                "MAX_OPEN_TRADES": "2",
                "PAPER_TRADE_SIZE_USD": "0.30",
                "PAPER_TRADE_SIZE_MIN_USD": "0.12",
                "PAPER_TRADE_SIZE_MAX_USD": "0.45",
                "MAX_LOSS_PER_TRADE_PERCENT_BALANCE": "6.0",
                "DAILY_MAX_DRAWDOWN_PERCENT": "0",
                "MAX_CONSECUTIVE_LOSSES": "0",
                "LOSS_STREAK_COOLDOWN_SECONDS": "0",
                "MAX_TOKEN_COOLDOWN_SECONDS": "300",
                "MAX_TOKEN_PRICE_CHANGE_5M_ABS_PERCENT": "42",
                "EDGE_FILTER_ENABLED": "true",
                "MIN_EXPECTED_EDGE_PERCENT": "1.0",
                "PROFIT_TARGET_PERCENT": "42",
                "STOP_LOSS_PERCENT": "15",
                "PAPER_GAS_PER_TX_USD": "0.02",
                "PAPER_BASE_SLIPPAGE_BPS": "60",
                "PAPER_REALISM_CAP_ENABLED": "true",
                "PAPER_REALISM_MAX_GAIN_PERCENT": "600",
                "PAPER_REALISM_MAX_LOSS_PERCENT": "95",
                "WETH_PRICE_FALLBACK_USD": "3000",
                "STAIR_STEP_ENABLED": "true",
                "STAIR_STEP_START_BALANCE_USD": "2.00",
                "STAIR_STEP_SIZE_USD": "2",
                "STAIR_STEP_TRADABLE_BUFFER_USD": "0.35",
                "AUTO_STOP_MIN_AVAILABLE_USD": "0.25",
            }
        )

    def _apply_medium_flow_preset(self) -> None:
        self._apply_preset_map(
            {
                "SIGNAL_SOURCE": "onchain",
                "ONCHAIN_ENABLE_UNISWAP_V3": "true",
                "ONCHAIN_POLL_INTERVAL_SECONDS": "5",
                "DEX_SEARCH_QUERIES": "base",
                "GECKO_NEW_POOLS_PAGES": "1",
                "DEX_BOOSTS_SOURCE_ENABLED": "false",
                "DEX_BOOSTS_MAX_TOKENS": "10",
                "AUTO_TRADE_ENABLED": "true",
                "AUTO_TRADE_PAPER": "true",
                "AUTO_FILTER_ENABLED": "true",
                "AUTO_TRADE_ENTRY_MODE": "top_n",
                "AUTO_TRADE_TOP_N": "3",
                "AUTONOMOUS_CONTROL_ENABLED": "true",
                "AUTONOMOUS_CONTROL_MODE": "apply",
                "AUTONOMOUS_CONTROL_INTERVAL_SECONDS": "240",
                "AUTONOMOUS_CONTROL_MIN_WINDOW_CYCLES": "2",
                "AUTONOMOUS_CONTROL_TARGET_CANDIDATES_MIN": "1.5",
                "AUTONOMOUS_CONTROL_TARGET_CANDIDATES_HIGH": "6.5",
                "AUTONOMOUS_CONTROL_TARGET_OPENED_MIN": "0.15",
                "AUTONOMOUS_CONTROL_NEG_REALIZED_TRIGGER_USD": "0.06",
                "AUTONOMOUS_CONTROL_POS_REALIZED_TRIGGER_USD": "0.06",
                "AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MIN": "1",
                "AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MAX": "4",
                "AUTONOMOUS_CONTROL_TOP_N_MIN": "2",
                "AUTONOMOUS_CONTROL_TOP_N_MAX": "10",
                "AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN": "12",
                "AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MAX": "48",
                "MAX_BUYS_PER_HOUR": "5",
                "MAX_BUY_AMOUNT": "0.00035",
                "MIN_TOKEN_SCORE": "72",
                "SAFE_TEST_MODE": "true",
                "SAFE_MIN_LIQUIDITY_USD": "16000",
                "SAFE_MIN_VOLUME_5M_USD": "1200",
                "SAFE_MIN_AGE_SECONDS": "60",
                "SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT": "22",
                "SAFE_REQUIRE_CONTRACT_SAFE": "true",
                "SAFE_REQUIRE_RISK_LEVEL": "MEDIUM",
                "SAFE_MAX_WARNING_FLAGS": "2",
                "MAX_OPEN_TRADES": "3",
                "PAPER_TRADE_SIZE_USD": "0.60",
                "PAPER_TRADE_SIZE_MIN_USD": "0.20",
                "PAPER_TRADE_SIZE_MAX_USD": "0.95",
                "MAX_LOSS_PER_TRADE_PERCENT_BALANCE": "8.0",
                "DAILY_MAX_DRAWDOWN_PERCENT": "0",
                "MAX_CONSECUTIVE_LOSSES": "0",
                "LOSS_STREAK_COOLDOWN_SECONDS": "0",
                "MAX_TOKEN_COOLDOWN_SECONDS": "450",
                "MAX_TOKEN_PRICE_CHANGE_5M_ABS_PERCENT": "40",
                "EDGE_FILTER_ENABLED": "true",
                "MIN_EXPECTED_EDGE_PERCENT": "1.0",
                "PROFIT_TARGET_PERCENT": "45",
                "STOP_LOSS_PERCENT": "18",
                "PAPER_GAS_PER_TX_USD": "0.02",
                "PAPER_BASE_SLIPPAGE_BPS": "70",
                "PAPER_REALISM_CAP_ENABLED": "true",
                "PAPER_REALISM_MAX_GAIN_PERCENT": "600",
                "PAPER_REALISM_MAX_LOSS_PERCENT": "95",
                "WETH_PRICE_FALLBACK_USD": "3000",
                "STAIR_STEP_ENABLED": "false",
                "STAIR_STEP_START_BALANCE_USD": "2.75",
                "STAIR_STEP_SIZE_USD": "5",
            }
        )

    def _apply_live_wallet_preset(self) -> None:
        self._apply_preset_map(
            {
                "WALLET_MODE": "live",
                "SIGNAL_SOURCE": "onchain",
                "ONCHAIN_ENABLE_UNISWAP_V3": "true",
                "ONCHAIN_PARALLEL_MARKET_SOURCES": "true",
                "ONCHAIN_POLL_INTERVAL_SECONDS": "3",
                "ONCHAIN_FINALITY_BLOCKS": "2",
                "ONCHAIN_BLOCK_CHUNK": "300",
                "DEX_SEARCH_QUERIES": "base,memecoin,new",
                "GECKO_NEW_POOLS_PAGES": "2",
                "DEX_BOOSTS_SOURCE_ENABLED": "true",
                "DEX_BOOSTS_MAX_TOKENS": "20",
                "AUTO_TRADE_ENABLED": "true",
                "AUTO_TRADE_PAPER": "true",
                "AUTO_FILTER_ENABLED": "true",
                "AUTO_TRADE_ENTRY_MODE": "top_n",
                "AUTO_TRADE_TOP_N": "2",
                "MAX_BUYS_PER_HOUR": "0",
                "MAX_OPEN_TRADES": "2",
                "MAX_BUY_AMOUNT": "0.00020",
                "MIN_TOKEN_SCORE": "70",
                "SAFE_TEST_MODE": "true",
                "SAFE_MIN_LIQUIDITY_USD": "14000",
                "SAFE_MIN_VOLUME_5M_USD": "800",
                "SAFE_MIN_AGE_SECONDS": "45",
                "SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT": "24",
                "SAFE_REQUIRE_CONTRACT_SAFE": "true",
                "SAFE_REQUIRE_RISK_LEVEL": "MEDIUM",
                "SAFE_MAX_WARNING_FLAGS": "1",
                "PAPER_TRADE_SIZE_USD": "0.28",
                "PAPER_TRADE_SIZE_MIN_USD": "0.12",
                "PAPER_TRADE_SIZE_MAX_USD": "0.40",
                "PAPER_GAS_PER_TX_USD": "0.02",
                "PAPER_BASE_SLIPPAGE_BPS": "60",
                "PAPER_REALISM_CAP_ENABLED": "true",
                "PAPER_REALISM_MAX_GAIN_PERCENT": "300",
                "PAPER_REALISM_MAX_LOSS_PERCENT": "85",
                "EDGE_FILTER_ENABLED": "true",
                "MIN_EXPECTED_EDGE_PERCENT": "1.0",
                "PROFIT_TARGET_PERCENT": "35",
                "STOP_LOSS_PERCENT": "14",
                "MAX_TOKEN_COOLDOWN_SECONDS": "300",
                "MAX_TOKEN_PRICE_CHANGE_5M_ABS_PERCENT": "38",
                "MAX_LOSS_PER_TRADE_PERCENT_BALANCE": "6.0",
                "DAILY_MAX_DRAWDOWN_PERCENT": "0",
                "MAX_CONSECUTIVE_LOSSES": "0",
                "LOSS_STREAK_COOLDOWN_SECONDS": "0",
                "WALLET_BALANCE_USD": "2.00",
                "WETH_PRICE_FALLBACK_USD": "3000",
                "STAIR_STEP_ENABLED": "true",
                "STAIR_STEP_START_BALANCE_USD": "1.25",
                "STAIR_STEP_SIZE_USD": "1.00",
                "STAIR_STEP_TRADABLE_BUFFER_USD": "0.35",
                "AUTO_STOP_MIN_AVAILABLE_USD": "0.25",
            }
        )
        if "WALLET_MODE" in self.env_vars:
            self.env_vars["WALLET_MODE"].set("live")
        if hasattr(self, "wallet_mode_var"):
            self.wallet_mode_var.set("live")
        if hasattr(self, "paper_balance_set_var"):
            self.paper_balance_set_var.set("2.00")
        if hasattr(self, "hint_var"):
            self.hint_var.set("Live Wallet preset applied (live profile + paper execution until live executor is enabled).")

    def _apply_ultra_safe_live_preset(self) -> None:
        env_map = read_runtime_env_map()

        # Prefer the most recent on-chain poll (non-blocking). If not available yet, fall back to .env value.
        live_eth = 0.0
        if float(getattr(self, "_live_wallet_last_ts", 0.0) or 0.0) > 0:
            live_eth = float(getattr(self, "_live_wallet_eth", 0.0) or 0.0)

        price_usd = self._to_float(env_map.get("WETH_PRICE_FALLBACK_USD"), 2000.0)
        live_usd = (live_eth * price_usd) if live_eth > 0 else self._to_float(env_map.get(LIVE_WALLET_BALANCE_KEY), 0.0)

        reserve_eth = self._to_float(env_map.get("LIVE_MIN_GAS_RESERVE_ETH"), 0.0007)
        # If the wallet is tiny, make sure we don't reserve more than ~85% of the available ETH.
        if live_eth > 0 and reserve_eth > (live_eth * 0.85):
            reserve_eth = max(0.0002, live_eth * 0.65)

        reserve_usd = reserve_eth * max(0.0, price_usd)
        tradable_usd = max(0.0, live_usd - reserve_usd)

        # Entry sizes are derived from the *tradable* portion of the wallet.
        trade_max_usd = max(0.10, min(tradable_usd * 0.25, tradable_usd, 0.35))
        trade_min_usd = max(0.10, min(trade_max_usd, tradable_usd * 0.10))

        max_buy_eth = 0.0
        if price_usd > 0:
            max_buy_eth = (trade_max_usd / price_usd) * 1.05  # small headroom for rounding
        if live_eth > 0:
            max_buy_eth = min(max_buy_eth, max(0.0, live_eth - reserve_eth) * 0.90)
        max_buy_eth = max(0.0, max_buy_eth)

        stop_min_available_usd = max(0.10, reserve_usd + 0.10)

        preset = {
            "WALLET_MODE": "live",
            "SIGNAL_SOURCE": "onchain",
            "ONCHAIN_ENABLE_UNISWAP_V3": "true",
            "ONCHAIN_PARALLEL_MARKET_SOURCES": "true",
            "ONCHAIN_POLL_INTERVAL_SECONDS": "4",
            "ONCHAIN_FINALITY_BLOCKS": "2",
            "ONCHAIN_BLOCK_CHUNK": "300",
            "AUTO_TRADE_ENABLED": "true",
            "AUTO_TRADE_PAPER": "false",
            "AUTO_FILTER_ENABLED": "true",
            "AUTO_TRADE_ENTRY_MODE": "single",
            "AUTO_TRADE_TOP_N": "1",
            "MAX_OPEN_TRADES": "1",
            "MAX_BUYS_PER_HOUR": "3",
            "MIN_TOKEN_SCORE": "80",
            "SAFE_TEST_MODE": "true",
            "SAFE_MIN_LIQUIDITY_USD": "100000",
            "SAFE_MIN_VOLUME_5M_USD": "8000",
            "SAFE_MIN_AGE_SECONDS": "1800",
            "SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT": "15",
            "SAFE_REQUIRE_CONTRACT_SAFE": "true",
            "SAFE_REQUIRE_RISK_LEVEL": "LOW",
            "SAFE_MAX_WARNING_FLAGS": "0",
            "HONEYPOT_API_ENABLED": "true",
            "HONEYPOT_API_FAIL_CLOSED": "true",
            "HONEYPOT_MAX_BUY_TAX_PERCENT": "5",
            "HONEYPOT_MAX_SELL_TAX_PERCENT": "5",
            "TOKEN_SAFETY_FAIL_CLOSED": "true",
            "LIVE_MAX_SWAP_GAS": "300000",
            "LIVE_MIN_GAS_RESERVE_ETH": "0.00075",
            "LIVE_ROUNDTRIP_CHECK_ENABLED": "true",
            "LIVE_ROUNDTRIP_MIN_RETURN_RATIO": "0.80",
            "DEX_BOOSTS_SOURCE_ENABLED": "false",
            "PAPER_TRADE_SIZE_MIN_USD": f"{trade_min_usd:.2f}",
            "PAPER_TRADE_SIZE_MAX_USD": f"{trade_max_usd:.2f}",
            "MAX_BUY_AMOUNT": f"{max_buy_eth:.6f}" if max_buy_eth > 0 else "0",
            "DYNAMIC_POSITION_SIZING_ENABLED": "true",
            "EDGE_FILTER_ENABLED": "true",
            "EDGE_FILTER_MODE": "percent",
            "MIN_EXPECTED_EDGE_PERCENT": "10.0",
            "PROFIT_TARGET_PERCENT": "12",
            "STOP_LOSS_PERCENT": "7",
            "PROFIT_LOCK_ENABLED": "true",
            "PROFIT_LOCK_TRIGGER_PERCENT": "6",
            "PROFIT_LOCK_FLOOR_PERCENT": "1",
            "WEAKNESS_EXIT_ENABLED": "true",
            "WEAKNESS_EXIT_MIN_AGE_PERCENT": "30",
            "WEAKNESS_EXIT_PNL_PERCENT": "-3",
            "MAX_TOKEN_COOLDOWN_SECONDS": "1200",
            "DYNAMIC_HOLD_ENABLED": "true",
            "HOLD_MIN_SECONDS": "90",
            "HOLD_MAX_SECONDS": "600",
            "PAPER_MAX_HOLD_SECONDS": "600",
            "AUTO_STOP_MIN_AVAILABLE_USD": f"{stop_min_available_usd:.2f}",
            "LIVE_STOP_AFTER_PROFIT_USD": "5.0",
            "LIVE_SESSION_RESET_ON_START": "true",
        }

        # Keep GUI and .env in sync.
        save_runtime_env_map(preset)
        for k, v in preset.items():
            if k in self.env_vars:
                self.env_vars[k].set(v)

        if hasattr(self, "wallet_mode_var"):
            self.wallet_mode_var.set("live")
        self._refresh_wallet()
        if hasattr(self, "hint_var"):
            self.hint_var.set(
                f"Ultra Safe Live applied. Wallet~${live_usd:.2f}, tradable~${tradable_usd:.2f}, size ${trade_min_usd:.2f}-${trade_max_usd:.2f}."
            )

    def _apply_working_live_preset(self) -> None:
        # "Working" profile: keeps core safety checks but allows more entries than Ultra Safe.
        env_map = read_runtime_env_map()
        live_usd = self._to_float(env_map.get(LIVE_WALLET_BALANCE_KEY), 0.0)
        reserve_usd = self._to_float(env_map.get("AUTO_STOP_MIN_AVAILABLE_USD"), 0.25)
        tradable_usd = max(0.0, live_usd - reserve_usd)
        trade_max_usd = max(0.10, min(tradable_usd * 0.35, max(0.20, tradable_usd), 0.45))
        trade_min_usd = max(0.10, min(trade_max_usd, max(0.10, tradable_usd * 0.15)))

        preset = {
            "WALLET_MODE": "live",
            "AUTO_TRADE_ENABLED": "true",
            "AUTO_TRADE_PAPER": "false",
            "AUTO_TRADE_ENTRY_MODE": "single",
            "AUTO_TRADE_TOP_N": "1",
            "MAX_OPEN_TRADES": "1",
            "MAX_BUYS_PER_HOUR": "3",
            "MIN_TOKEN_SCORE": "75",
            "SAFE_TEST_MODE": "true",
            "SAFE_REQUIRE_CONTRACT_SAFE": "true",
            "SAFE_REQUIRE_RISK_LEVEL": "LOW",
            "SAFE_MAX_WARNING_FLAGS": "0",
            "SAFE_MIN_LIQUIDITY_USD": "100000",
            "SAFE_MIN_VOLUME_5M_USD": "5000",
            "SAFE_MIN_AGE_SECONDS": "1800",
            "SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT": "15",
            "EDGE_FILTER_ENABLED": "true",
            "EDGE_FILTER_MODE": "percent",
            "MIN_EXPECTED_EDGE_PERCENT": "10.0",
            "LIVE_ROUNDTRIP_CHECK_ENABLED": "true",
            "LIVE_ROUNDTRIP_MIN_RETURN_RATIO": "0.74",
            "HONEYPOT_API_ENABLED": "true",
            "HONEYPOT_API_FAIL_CLOSED": "true",
            "TOKEN_SAFETY_FAIL_CLOSED": "true",
            "LIVE_MIN_GAS_RESERVE_ETH": "0.00075",
            "LIVE_MAX_SWAP_GAS": "300000",
            "MAX_BUY_AMOUNT": "0.00020",
            "PAPER_TRADE_SIZE_MIN_USD": f"{trade_min_usd:.2f}",
            "PAPER_TRADE_SIZE_MAX_USD": f"{trade_max_usd:.2f}",
            "PROFIT_TARGET_PERCENT": "12",
            "STOP_LOSS_PERCENT": "7",
            "PROFIT_LOCK_ENABLED": "true",
            "PROFIT_LOCK_TRIGGER_PERCENT": "6",
            "PROFIT_LOCK_FLOOR_PERCENT": "1",
            "WEAKNESS_EXIT_ENABLED": "true",
            "WEAKNESS_EXIT_MIN_AGE_PERCENT": "30",
            "WEAKNESS_EXIT_PNL_PERCENT": "-3",
            "MAX_TOKEN_COOLDOWN_SECONDS": "1200",
            "HOLD_MIN_SECONDS": "90",
            "HOLD_MAX_SECONDS": "600",
            "PAPER_MAX_HOLD_SECONDS": "600",
            "LIVE_STOP_AFTER_PROFIT_USD": "5.0",
            "LIVE_SESSION_RESET_ON_START": "true",
            "DEX_BOOSTS_SOURCE_ENABLED": "false",
        }

        save_runtime_env_map(preset)
        for k, v in preset.items():
            if k in self.env_vars:
                self.env_vars[k].set(v)
        if hasattr(self, "wallet_mode_var"):
            self.wallet_mode_var.set("live")
        self._refresh_wallet()
        if hasattr(self, "hint_var"):
            self.hint_var.set("Working Live applied. Safer core checks kept, entry flow widened moderately.")

    def _load_settings(self) -> None:
        env_map = read_runtime_env_map()
        for key, _ in GUI_SETTINGS_FIELDS:
            self.env_vars[key].set(env_map.get(key, ""))
        if hasattr(self, "wallet_mode_var"):
            self.wallet_mode_var.set(str(env_map.get(WALLET_MODE_KEY, "paper")).strip().lower() or "paper")
        if hasattr(self, "live_balance_var"):
            self.live_balance_var.set(str(env_map.get(LIVE_WALLET_BALANCE_KEY, "0")).strip() or "0")
        self._refresh_wallet()

    def _save_settings(self) -> None:
        values = {k: v.get().strip() for k, v in self.env_vars.items()}
        # Keep mode flags coherent. Bot's runtime decides "live" via:
        # AUTO_TRADE_ENABLED && !AUTO_TRADE_PAPER.
        try:
            mode = str(self.wallet_mode_var.get()).strip().lower() if hasattr(self, "wallet_mode_var") else ""
        except Exception:
            mode = ""
        if mode == "live":
            # Prevent accidental paper flag re-enabling live and causing "AUTO_TRADE_ENABLED=false" confusion.
            values["AUTO_TRADE_PAPER"] = "false"
        elif mode == "paper":
            values["AUTO_TRADE_PAPER"] = "true"
        try:
            save_runtime_env_map(values)
        except Exception as exc:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ° ÑÐ¾Ñ…Ñ€Ð°Ð½ÐµÐ½Ð¸Ñ", str(exc))
            return
        messagebox.showinfo("Ð“Ð¾Ñ‚Ð¾Ð²Ð¾", ".env ÑƒÑÐ¿ÐµÑˆÐ½Ð¾ Ð¾Ð±Ð½Ð¾Ð²Ð»ÐµÐ½.")

    def _save_restart(self) -> None:
        self._save_settings()
        self.on_restart()

    def refresh(self) -> None:
        self._last_refresh_ts = time.time()
        self._refresh_positions_source_options()
        pid = read_pid()
        single_alive = bool(pid and is_main_local_process(pid))
        if not single_alive:
            # Self-heal stale/missing bot.pid after manual restarts outside GUI.
            live_pids = sorted(set(list_main_local_pids()))
            if live_pids:
                pid = int(live_pids[0])
                try:
                    with open(PID_FILE, "w", encoding="ascii") as f:
                        f.write(str(pid))
                except Exception:
                    pass
                single_alive = bool(is_main_local_process(pid))
        matrix_alive = 0
        alive_ids: list[str] = []
        meta = read_matrix_meta()
        items = meta.get("items")
        if isinstance(items, list):
            for row in items:
                if not isinstance(row, dict):
                    continue
                try:
                    p = int(row.get("pid", 0) or 0)
                except Exception:
                    p = 0
                if p > 0 and is_main_local_process(p):
                    matrix_alive += 1
                    alive_ids.append(str(row.get("id", "") or "").strip())

        if (not single_alive) and matrix_alive > 0 and str(self.positions_source_var.get() or "single") == "single":
            first = next((x for x in alive_ids if x), "")
            if first:
                self.positions_source_var.set(first)

        if single_alive:
            if matrix_alive > 0:
                self.status_var.set(f"Status: RUNNING (PID {pid}) | Matrix alive: {matrix_alive}")
            else:
                self.status_var.set(f"Status: RUNNING (PID {pid})")
        else:
            if matrix_alive > 0:
                self.status_var.set(f"Status: MATRIX MODE ({matrix_alive} instance)")
            else:
                self.status_var.set("Status: STOPPED")

        log_paths: list[str] = []
        if single_alive:
            log_paths = self._runtime_log_paths_for_signals()
        elif matrix_alive > 0:
            if isinstance(items, list):
                for row in items:
                    if not isinstance(row, dict):
                        continue
                    log_dir = str(row.get("log_dir", "") or "").strip()
                    if not log_dir:
                        continue
                    abs_dir = log_dir if os.path.isabs(log_dir) else os.path.join(PROJECT_ROOT, log_dir)
                    latest = latest_session_log(abs_dir)
                    if latest:
                        log_paths.append(latest)
        if not log_paths:
            log_paths = [APP_LOG]

        matrix_sections: list[tuple[str, list[str]]] = []
        if matrix_alive > 0 and isinstance(items, list):
            matrix_sections = _collect_matrix_runtime_sections(items, max_lines_each=120)

        feed_bottom = self._is_near_bottom(self.feed_text)
        feed_y = self.feed_text.yview()
        app_lines: list[str] = []
        for path in log_paths:
            app_lines.extend(read_tail(path, 140))
        events = self._activity_feed_events(app_lines)
        feed_signature = "\n".join(f"{lvl}|{ln}" for lvl, ln in events[-180:])
        if feed_signature != self._last_feed_signature and (not self._text_has_active_selection(self.feed_text)):
            self._last_feed_signature = feed_signature
            self.feed_text.delete("1.0", tk.END)
            if not events:
                self.feed_text.insert(tk.END, "No market mode changes or trade-open events yet.\n", ("INFO",))
            for level, line in events[-180:]:
                self.feed_text.insert(tk.END, line + "\n", (level,))
            self._restore_scroll(self.feed_text, feed_y, feed_bottom)

        raw_bottom = self._is_near_bottom(self.raw_text)
        raw_y = self.raw_text.yview()
        combined = []
        if matrix_sections:
            for inst_id, lines in matrix_sections:
                combined.append(f"===== MATRIX {inst_id} runtime =====")
                combined.extend(lines[-220:] if lines else ["<empty>"])
                combined.append("")
        else:
            for path in log_paths:
                label = os.path.basename(path)
                lines = compact_runtime_lines(read_tail(path, 180))
                combined.append(f"===== {label} LOG (clean) =====")
                combined.extend(lines[-120:] or ["<empty>"])
                combined.append("")
        raw_blob = "\n".join(combined)
        if raw_blob != self._last_raw_signature and (not self._text_has_active_selection(self.raw_text)):
            self._last_raw_signature = raw_blob
            self.raw_text.delete("1.0", tk.END)
            self.raw_text.insert(tk.END, raw_blob)
            self._restore_scroll(self.raw_text, raw_y, raw_bottom)
        self._refresh_header_metrics(app_lines)
        self._refresh_idle_reasons(app_lines)
        self._refresh_signals()
        self._refresh_wallet()
        self._refresh_positions(app_lines)
        self._refresh_emergency_tab(app_lines)

    def _refresh_positions_source_options(self) -> None:
        env_map = self._runtime_env_map()
        single_state_raw = str(env_map.get("PAPER_STATE_FILE", PAPER_STATE_FILE) or "").strip() or PAPER_STATE_FILE
        single_state_abs = single_state_raw if os.path.isabs(single_state_raw) else os.path.join(PROJECT_ROOT, single_state_raw)
        source_map: dict[str, str] = {"single": single_state_abs}
        meta = read_matrix_meta()
        items = meta.get("items")
        if isinstance(items, list):
            for row in items:
                if not isinstance(row, dict):
                    continue
                inst = str(row.get("id", "")).strip()
                path = str(row.get("paper_state_file", "")).strip()
                if not inst or not path:
                    continue
                abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
                source_map[inst] = abs_path
        self.positions_source_map = source_map
        labels = tuple(source_map.keys())
        if hasattr(self, "positions_source_combo"):
            self.positions_source_combo.configure(values=labels)
        cur = str(self.positions_source_var.get() or "single")
        if cur not in source_map:
            self.positions_source_var.set("single")

    def _selected_positions_state_file(self) -> str:
        self._refresh_positions_source_options()
        key = str(self.positions_source_var.get() or "single")
        return str(self.positions_source_map.get(key, PAPER_STATE_FILE))

    def _runtime_state_file(self) -> str:
        env_map = read_runtime_env_map()
        raw = str(env_map.get("PAPER_STATE_FILE", PAPER_STATE_FILE) or "").strip() or PAPER_STATE_FILE
        return raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)

    def _matrix_items(self) -> list[dict]:
        meta = read_matrix_meta()
        items = meta.get("items")
        if not isinstance(items, list):
            return []
        return [row for row in items if isinstance(row, dict)]

    def _selected_matrix_item(self) -> dict | None:
        selected = str(self.positions_source_var.get() or "single").strip()
        if selected and selected != "single":
            for row in self._matrix_items():
                if str(row.get("id", "")).strip() == selected:
                    return row
        return None

    def _runtime_env_map(self) -> dict[str, str]:
        item = self._selected_matrix_item()
        if isinstance(item, dict):
            env_file = str(item.get("env_file", "") or "").strip()
            if env_file:
                env_abs = env_file if os.path.isabs(env_file) else os.path.join(PROJECT_ROOT, env_file)
                env_map = read_env_map_from(env_abs)
                if env_map:
                    if (
                        str(env_map.get("MAX_BUYS_PER_HOUR", "") or "").strip() == ""
                        and str(env_map.get("MAX_TRADES_PER_HOUR", "") or "").strip() != ""
                    ):
                        env_map["MAX_BUYS_PER_HOUR"] = str(env_map.get("MAX_TRADES_PER_HOUR", "") or "").strip()
                    return env_map
        env_map = read_runtime_env_map()
        if (
            str(env_map.get("MAX_BUYS_PER_HOUR", "") or "").strip() == ""
            and str(env_map.get("MAX_TRADES_PER_HOUR", "") or "").strip() != ""
        ):
            env_map["MAX_BUYS_PER_HOUR"] = str(env_map.get("MAX_TRADES_PER_HOUR", "") or "").strip()
        return env_map

    def _selected_signals_file(self) -> tuple[str, str]:
        item = self._selected_matrix_item()
        if isinstance(item, dict):
            inst = str(item.get("id", "matrix")).strip() or "matrix"
            log_dir = str(item.get("log_dir", "") or "").strip()
            if log_dir:
                abs_dir = log_dir if os.path.isabs(log_dir) else os.path.join(PROJECT_ROOT, log_dir)
                return inst, os.path.join(abs_dir, "local_alerts.jsonl")
        env_map = self._runtime_env_map()
        raw_log_dir = str(env_map.get("LOG_DIR", LOG_DIR) or "").strip() or LOG_DIR
        abs_log_dir = raw_log_dir if os.path.isabs(raw_log_dir) else os.path.join(PROJECT_ROOT, raw_log_dir)
        return "single", os.path.join(abs_log_dir, "local_alerts.jsonl")

    def _runtime_log_paths_for_signals(self) -> list[str]:
        item = self._selected_matrix_item()
        if not isinstance(item, dict):
            env_map = self._runtime_env_map()
            raw_log_dir = str(env_map.get("LOG_DIR", LOG_DIR) or "").strip() or LOG_DIR
            abs_log_dir = raw_log_dir if os.path.isabs(raw_log_dir) else os.path.join(PROJECT_ROOT, raw_log_dir)
            paths: list[str] = []
            latest = latest_session_log(abs_log_dir)
            if latest:
                paths.append(latest)
            for name in ("app.log", "out.log"):
                p = os.path.join(abs_log_dir, name)
                if os.path.exists(p):
                    paths.append(p)
            if not paths:
                paths.append(APP_LOG)
            return paths
        paths: list[str] = []
        log_dir = str(item.get("log_dir", "") or "").strip()
        if log_dir:
            abs_dir = log_dir if os.path.isabs(log_dir) else os.path.join(PROJECT_ROOT, log_dir)
            latest = latest_session_log(abs_dir)
            if latest:
                paths.append(latest)
            for name in ("app.log", "out.log"):
                p = os.path.join(abs_dir, name)
                if os.path.exists(p):
                    paths.append(p)
        if not paths:
            paths.append(APP_LOG)
        return paths

    def _resolve_candidates_log_path(self) -> str:
        item = self._selected_matrix_item()
        if isinstance(item, dict):
            log_dir = str(item.get("log_dir", "") or "").strip()
            if log_dir:
                abs_dir = log_dir if os.path.isabs(log_dir) else os.path.join(PROJECT_ROOT, log_dir)
                return os.path.join(abs_dir, "candidates.jsonl")
        env_map = self._runtime_env_map()
        raw = str(env_map.get("CANDIDATE_DECISIONS_LOG_FILE", "") or "").strip()
        if not raw:
            raw = CANDIDATE_DECISIONS_LOG_FILE
        if os.path.isabs(raw):
            return raw
        return os.path.join(PROJECT_ROOT, raw)

    def _resolve_trade_decisions_log_path(self) -> str:
        item = self._selected_matrix_item()
        if isinstance(item, dict):
            log_dir = str(item.get("log_dir", "") or "").strip()
            if log_dir:
                abs_dir = log_dir if os.path.isabs(log_dir) else os.path.join(PROJECT_ROOT, log_dir)
                return os.path.join(abs_dir, "trade_decisions.jsonl")
        env_map = self._runtime_env_map()
        raw = str(env_map.get("TRADE_DECISIONS_LOG_FILE", "") or "").strip()
        if not raw:
            raw = TRADE_DECISIONS_LOG_FILE
        if os.path.isabs(raw):
            return raw
        return os.path.join(PROJECT_ROOT, raw)

    def _market_mode_activity_events(self, max_rows: int = 160) -> list[tuple[float, str, str]]:
        _src, alerts_path = self._selected_signals_file()
        rows = read_jsonl_tail(alerts_path, max_rows * 6)
        out: list[tuple[float, str, str]] = []
        for row in rows:
            event_type = str(row.get("event_type", "") or "").strip().upper()
            symbol = str(row.get("symbol", "") or "").strip().upper()
            if event_type != "MARKET_MODE_CHANGE" and symbol != "MARKET_MODE":
                continue
            ts = _parse_ts(row.get("ts", row.get("timestamp")))
            if ts is None:
                ts = time.time()
            breakdown = row.get("breakdown", {})
            prev_mode = ""
            new_mode = str(row.get("recommendation", "") or "").strip().upper()
            reason = ""
            if isinstance(breakdown, dict):
                prev_mode = str(breakdown.get("previous_mode", "") or "").strip().upper()
                if not new_mode:
                    new_mode = str(breakdown.get("new_mode", "") or "").strip().upper()
                reason = str(breakdown.get("reason", "") or "").strip()
            if not new_mode:
                new_mode = "UNKNOWN"
            ts_text = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            if prev_mode:
                msg = f"[{ts_text}] market mode {prev_mode}->{new_mode} reason={reason or '-'}"
            else:
                msg = f"[{ts_text}] market mode {new_mode} reason={reason or '-'}"
            level = "INFO"
            if new_mode == "GREEN":
                level = "POLICY_OK"
            elif new_mode == "YELLOW":
                level = "POLICY_DEGRADED"
            elif new_mode == "RED":
                level = "POLICY_FAIL_CLOSED"
            out.append((float(ts), level, msg))
        return out[-max_rows:]

    def _trade_activity_events(self, max_rows: int = 220) -> list[tuple[float, str, str]]:
        path = self._resolve_trade_decisions_log_path()
        rows = read_jsonl_tail(path, max_rows * 8)
        out: list[tuple[float, str, str]] = []
        for row in rows:
            stage = str(row.get("decision_stage", "") or "").strip().lower()
            if stage != "trade_open":
                continue
            ts = _parse_ts(row.get("ts", row.get("timestamp")))
            if ts is None:
                ts = time.time()
            symbol = str(row.get("symbol", "N/A") or "N/A").strip() or "N/A"
            mode = str(row.get("market_mode", "") or "").strip().upper() or "-"
            tier = str(row.get("entry_tier", "") or "").strip().upper() or "-"
            channel = str(row.get("entry_channel", "") or "").strip().lower() or "-"
            score = int(self._safe_float(row.get("score", 0), 0.0))
            size_usd = self._safe_float(row.get("position_size_usd", 0.0), 0.0)
            edge_pct = self._safe_float(row.get("expected_edge_percent", 0.0), 0.0)
            reason = str(row.get("reason", "") or "").strip().lower()
            ts_text = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            msg = (
                f"[{ts_text}] trade open {symbol} mode={mode} tier={tier}/{channel} "
                f"size=${size_usd:.2f} score={score} edge={edge_pct:.2f}%"
            )
            if reason and reason not in {"-", "none", "ok"}:
                msg = f"{msg} reason={reason}"
            out.append((float(ts), "BUY", msg))
        return out[-max_rows:]

    def _recent_positions_from_trade_decisions(self, max_rows: int = 1200) -> tuple[list[dict], list[dict]]:
        path = self._resolve_trade_decisions_log_path()
        if not os.path.exists(path):
            return [], []
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return [], []

        cache = getattr(self, "_trade_positions_cache", None)
        if (
            isinstance(cache, dict)
            and cache.get("path") == path
            and float(cache.get("mtime", -1.0)) == float(mtime)
        ):
            return list(cache.get("open", []) or []), list(cache.get("closed", []) or [])

        rows = read_jsonl_tail(path, max_rows)
        if not rows:
            self._trade_positions_cache = {"path": path, "mtime": float(mtime), "open": [], "closed": []}
            return [], []

        def _row_iso_ts(row: dict) -> str:
            text = str(row.get("timestamp", "") or "").strip()
            if text:
                return text
            ts = _parse_ts(row.get("ts"))
            if ts is None:
                return ""
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        open_map: dict[str, dict] = {}
        closed_rows: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            stage = str(row.get("decision_stage", "") or "").strip().lower()
            if stage not in {"trade_open", "trade_close"}:
                continue

            candidate_id = str(row.get("candidate_id", "") or "").strip()
            token_address = str(row.get("token_address", "") or "").strip()
            symbol = str(row.get("symbol", "N/A") or "N/A").strip() or "N/A"
            key = candidate_id or token_address or f"{symbol}|{_row_iso_ts(row)}"

            if stage == "trade_open":
                open_map[key] = {
                    "token_address": token_address,
                    "symbol": symbol,
                    "score": int(self._safe_float(row.get("score", 0), 0.0)),
                    "entry_price_usd": self._safe_float(row.get("entry_price_usd", 0.0), 0.0),
                    "current_price_usd": self._safe_float(row.get("entry_price_usd", 0.0), 0.0),
                    "position_size_usd": self._safe_float(row.get("position_size_usd", 0.0), 0.0),
                    "pnl_percent": 0.0,
                    "pnl_usd": 0.0,
                    "opened_at": _row_iso_ts(row),
                    "max_hold_seconds": int(self._safe_float(row.get("max_hold_seconds", 180), 180.0)),
                }
                continue

            # trade_close
            open_row = open_map.pop(key, None)
            opened_at = ""
            if isinstance(open_row, dict):
                opened_at = str(open_row.get("opened_at", "") or "")
            if not opened_at:
                opened_at = str(row.get("opened_at", "") or "")
            closed_rows.append(
                {
                    "candidate_id": candidate_id,
                    "token_address": token_address,
                    "symbol": symbol,
                    "close_reason": str(row.get("reason", "") or "").strip(),
                    "reason": str(row.get("reason", "") or "").strip(),
                    "pnl_percent": self._safe_float(row.get("pnl_percent", 0.0), 0.0),
                    "pnl_usd": self._safe_float(row.get("pnl_usd", 0.0), 0.0),
                    "opened_at": opened_at,
                    "closed_at": _row_iso_ts(row),
                }
            )

        open_rows = list(open_map.values())
        self._trade_positions_cache = {
            "path": path,
            "mtime": float(mtime),
            "open": open_rows,
            "closed": closed_rows,
        }
        return open_rows, closed_rows

    def _profile_stop_activity_events(self, max_rows: int = 80) -> list[tuple[float, str, str]]:
        _src, alerts_path = self._selected_signals_file()
        rows = read_jsonl_tail(alerts_path, max_rows * 8)
        out: list[tuple[float, str, str]] = []
        for row in rows:
            event_type = str(row.get("event_type", "") or "").strip().upper()
            symbol = str(row.get("symbol", "") or "").strip().upper()
            if event_type not in {"PROFILE_AUTOSTOP", "CHAMPION_GUARD_AUTOSTOP"} and symbol not in {
                "PROFILE_STOPPED",
                "PROFILE_AUTOSTOP",
            }:
                continue
            ts = _parse_ts(row.get("ts", row.get("timestamp")))
            if ts is None:
                ts = time.time()
            breakdown = row.get("breakdown", {})
            if not isinstance(breakdown, dict):
                breakdown = {}
            tag = str(
                breakdown.get("event_tag")
                or row.get("tag")
                or ("[AUTOSTOP][PROFILE_STOPPED]" if event_type == "PROFILE_AUTOSTOP" else "[AUTOSTOP][CHAMPION_GUARD]")
            ).strip()
            run_tag = str(breakdown.get("run_tag") or row.get("run_tag") or "-").strip()
            reason = str(breakdown.get("reason") or row.get("reason") or "-").strip()
            ts_text = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            msg = f"[{ts_text}] {tag} profile={run_tag} reason={reason}"
            out.append((float(ts), "WARNING", msg))
        return out[-max_rows:]

    def _activity_feed_events(self, app_lines: list[str]) -> list[tuple[str, str]]:
        events = (
            self._market_mode_activity_events(max_rows=160)
            + self._trade_activity_events(max_rows=240)
            + self._profile_stop_activity_events(max_rows=80)
        )
        if not events:
            return parse_activity(app_lines)
        events.sort(key=lambda x: x[0])
        return [(level, text) for _ts, level, text in events[-220:]]

    def _read_candidate_rows_window(self, window_seconds: int, max_bytes: int = 2_000_000) -> list[dict]:
        path = self._resolve_candidates_log_path()
        blob = read_tail_bytes(path, max_bytes)
        if not blob:
            return []
        rows: list[dict] = []
        now_ts = time.time()
        cutoff = now_ts - max(0, int(window_seconds))
        for line in blob.splitlines():
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            ts_raw = row.get("ts")
            ts = _parse_ts(ts_raw)
            if ts is not None and ts < cutoff:
                continue
            rows.append(row)
        return rows

    @staticmethod
    def _count_entries_since(state: dict, since_ts: float) -> int:
        open_rows = state.get("open_positions", []) or []
        closed_rows = state.get("closed_positions", []) or []
        if not isinstance(open_rows, list):
            open_rows = []
        if not isinstance(closed_rows, list):
            closed_rows = []
        count = 0
        for row in open_rows:
            if not isinstance(row, dict):
                continue
            ts = _parse_ts(row.get("opened_at"))
            if ts is not None and ts >= since_ts:
                count += 1
        for row in closed_rows:
            if not isinstance(row, dict):
                continue
            ts = _parse_ts(row.get("opened_at"))
            if ts is not None and ts >= since_ts:
                count += 1
        return count

    @staticmethod
    def _count_exits_since(state: dict, since_ts: float) -> int:
        closed_rows = state.get("closed_positions", []) or []
        if not isinstance(closed_rows, list):
            closed_rows = []
        count = 0
        for row in closed_rows:
            if not isinstance(row, dict):
                continue
            ts = _parse_ts(row.get("closed_at"))
            if ts is not None and ts >= since_ts:
                count += 1
        return count

    @staticmethod
    def _extract_policy_state(app_lines: list[str]) -> str:
        policy = ""
        for line in app_lines:
            if "DATA_POLICY mode=" in line:
                policy = line
        if policy:
            try:
                part = policy.split("DATA_POLICY mode=", 1)[1]
                mode = part.split(" ", 1)[0].strip().upper()
                return mode or "UNKNOWN"
            except Exception:
                pass
        for line in app_lines[::-1]:
            if "DATA_MODE degraded" in line:
                return "DEGRADED"
            if "FAIL_CLOSED" in line or "DATA_MODE fail_closed" in line:
                return "FAIL_CLOSED"
        return "UNKNOWN"

    @staticmethod
    def _extract_risk_governor_state(app_lines: list[str]) -> str:
        last = ""
        for line in app_lines:
            if "RISK_GOVERNOR block reason=" in line:
                last = line
            elif "RISK_GOVERNOR pause reason=" in line:
                last = line
        if not last:
            return "ok"
        if "block reason=" in last:
            return last.split("block reason=", 1)[1].strip()
        if "pause reason=" in last:
            return last.split("pause reason=", 1)[1].strip()
        return "ok"

    @staticmethod
    def _extract_last_exception_age_seconds(app_lines: list[str]) -> float | None:
        candidates = [line for line in app_lines if ("exception" in line.lower() or "[ERROR]" in line)]
        if not candidates:
            return None
        last = candidates[-1]
        ts = None
        try:
            head = last[:19].replace("T", " ")
            dt = datetime.fromisoformat(head)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = dt.timestamp()
        except Exception:
            ts = None
        if ts is None:
            return None
        return max(0.0, time.time() - ts)

    @staticmethod
    def _estimate_latency_p95_ms(app_lines: list[str]) -> float | None:
        vals: list[float] = []
        for line in app_lines:
            if "/avg=" in line and "ms" in line:
                try:
                    part = line.split("/avg=", 1)[1]
                    num = part.split("ms", 1)[0]
                    vals.append(float(num))
                except Exception:
                    continue
        if not vals:
            return None
        vals.sort()
        idx = int(round(0.95 * (len(vals) - 1)))
        return vals[idx]

    def _latest_candidate_age_seconds(self) -> float | None:
        rows = self._read_candidate_rows_window(1800, max_bytes=2_000_000)
        if not rows:
            return None
        ts_vals = [t for t in (_parse_ts(r.get("ts")) for r in rows) if t is not None]
        if not ts_vals:
            return None
        latest = max(ts_vals)
        return max(0.0, time.time() - float(latest))

    def _stats_excluded_symbols(self) -> set[str]:
        env_map = read_runtime_env_map()
        raw = str(env_map.get("GUI_STATS_EXCLUDED_SYMBOLS", GUI_STATS_EXCLUDED_SYMBOLS_DEFAULT) or "").strip()
        if not raw:
            return set()
        return {s.strip().upper() for s in raw.split(",") if s.strip()}

    def _kill_switch_path(self) -> str:
        env_map = read_runtime_env_map()
        raw = str(env_map.get("KILL_SWITCH_FILE", os.path.join("data", "kill.txt")) or "").strip()
        if not raw:
            raw = os.path.join("data", "kill.txt")
        if os.path.isabs(raw):
            return raw
        return os.path.join(PROJECT_ROOT, raw)

    def _refresh_emergency_tab(self, app_lines: list[str]) -> None:
        if not hasattr(self, "critical_text"):
            return
        kill_path = self._kill_switch_path()
        kill_on = os.path.exists(kill_path)
        critical_lines = [line for line in app_lines if any(marker in line for marker in CRITICAL_LOG_MARKERS)]
        reset_count = sum(1 for line in app_lines if "CRITICAL_AUTO_RESET" in line)
        live_fail = sum(1 for line in app_lines if "AUTO_SELL live_failed" in line)
        err_count = sum(1 for line in app_lines if "[ERROR]" in line)
        if hasattr(self, "emergency_summary_var"):
            self.emergency_summary_var.set(
                f"KILL_SWITCH: {'ON' if kill_on else 'OFF'} | CRITICAL_AUTO_RESET: {reset_count} | live_failed: {live_fail} | errors: {err_count}"
            )

        critical_bottom = self._is_near_bottom(self.critical_text)
        critical_y = self.critical_text.yview()
        if not self._text_has_active_selection(self.critical_text):
            self.critical_text.delete("1.0", tk.END)
            if critical_lines:
                self.critical_text.insert(tk.END, "\n".join(critical_lines[-220:]))
            else:
                self.critical_text.insert(
                    tk.END,
                    "No critical events in latest app.log tail.\n"
                    f"KILL_SWITCH path: {kill_path}\n",
                )
            self._restore_scroll(self.critical_text, critical_y, critical_bottom)

    def _refresh_header_metrics(self, app_lines: list[str]) -> None:
        try:
            state = read_json(self._selected_positions_state_file())
        except Exception:
            state = {}
        env_map = self._runtime_env_map()
        open_rows = state.get("open_positions", []) or []
        open_now = len(open_rows) if isinstance(open_rows, list) else 0

        now_ts = time.time()
        buys_1h = self._count_entries_since(state, now_ts - 3600)
        buys_24h = self._count_entries_since(state, now_ts - 86400)
        exits_24h = self._count_exits_since(state, now_ts - 86400)

        def _limit_fmt(used: int, raw_max: str) -> str:
            try:
                max_val = int(float(raw_max))
            except Exception:
                max_val = 0
            if max_val <= 0:
                return f"{used}/âˆž"
            return f"{used}/{max_val}"

        max_buys = str(
            env_map.get("MAX_BUYS_PER_HOUR", env_map.get("MAX_TRADES_PER_HOUR", "0")) or "0"
        )
        max_tx_day = str(env_map.get("MAX_TX_PER_DAY", "0") or "0")
        max_open = str(env_map.get("MAX_OPEN_TRADES", "0") or "0")

        rows_10m = self._read_candidate_rows_window(600, max_bytes=1_000_000)
        dedup_10m = sum(
            1 for row in rows_10m if str(row.get("reason", "") or "").strip() == "heavy_dedup_ttl"
        )
        risk_state = self._extract_risk_governor_state(app_lines)

        limits = (
            f"Limits: buys/h {_limit_fmt(buys_1h, max_buys)} | "
            f"tx/day {_limit_fmt(buys_24h + exits_24h, max_tx_day)} | "
            f"open {_limit_fmt(open_now, max_open)} | "
            f"dedup(10m) {dedup_10m} | risk {risk_state}"
        )
        if hasattr(self, "header_limits_var"):
            self.header_limits_var.set(limits)

        policy = str(state.get("data_policy_mode", "") or "").strip().upper()
        if policy not in {"OK", "DEGRADED", "FAIL_CLOSED"}:
            policy = self._extract_policy_state(app_lines)
        latency_p95 = self._estimate_latency_p95_ms(app_lines)
        exc_age = self._extract_last_exception_age_seconds(app_lines)
        sync_delay = self._latest_candidate_age_seconds()

        lat_text = f"{latency_p95:.0f}ms" if latency_p95 is not None else "--"
        exc_text = f"{exc_age:.0f}s" if exc_age is not None else "--"
        sync_text = f"{sync_delay:.0f}s" if sync_delay is not None else "--"
        health = f"Health: policy {policy} | latency p95 {lat_text} | last exc {exc_text} | sync delay {sync_text}"
        if hasattr(self, "header_health_var"):
            self.header_health_var.set(health)

        if hasattr(self, "health_label"):
            color = "#cbd5e1"
            if policy == "OK":
                color = "#86efac"
            elif policy == "DEGRADED":
                color = "#fbbf24"
            elif policy == "FAIL_CLOSED":
                color = "#f87171"
            try:
                self.health_label.configure(foreground=color)
            except Exception:
                pass

    @staticmethod
    def _line_ts_local(line: str) -> datetime | None:
        text = str(line or "")
        if len(text) < 19:
            return None
        head = text[:19]
        try:
            return datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    def _refresh_idle_reasons(self, app_lines: list[str]) -> None:
        rows_15m = self._read_candidate_rows_window(900, max_bytes=1_500_000)
        counts: dict[str, int] = {}
        for row in rows_15m:
            reason = str(row.get("reason", "") or "").strip()
            if not reason:
                continue
            counts[reason] = int(counts.get(reason, 0)) + 1

        def _pick(label: str, key: str) -> str:
            return f"{label}={int(counts.get(key, 0))}"

        summary = " | ".join(
            [
                _pick("safe_volume", "safe_volume"),
                _pick("score_min", "score_min"),
                _pick("safe_risk", "safe_contract"),
                _pick("dedup_ttl", "heavy_dedup_ttl"),
                _pick("shard_skip", "shard_skip"),
            ]
        )
        if hasattr(self, "idle_reasons_var"):
            self.idle_reasons_var.set(summary or "--")

        # Live execution blockers after candidate pass (from runtime app logs).
        cutoff = datetime.now() - timedelta(minutes=15)
        live_counts: dict[str, int] = {}
        for line in app_lines:
            m = AUTO_TRADE_SKIP_RE.search(line)
            if not m:
                continue
            ts = self._line_ts_local(line)
            if ts is not None and ts < cutoff:
                continue
            reason = str(m.group("reason") or "").strip()
            if not reason:
                continue
            live_counts[reason] = int(live_counts.get(reason, 0)) + 1
        top = sorted(live_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        live_summary = " | ".join(f"{k}={v}" for k, v in top) if top else "--"
        if hasattr(self, "live_block_reasons_var"):
            self.live_block_reasons_var.set(live_summary)

    def on_session_snapshot(self) -> None:
        env_map = self._runtime_env_map()
        state_path = self._selected_positions_state_file()
        state = read_json(state_path)
        closed_rows = [x for x in (state.get("closed_positions") or []) if isinstance(x, dict)]
        closed_rows.sort(key=lambda r: str(r.get("closed_at", "")))
        last_closed = closed_rows[-50:]

        rows_15m = self._read_candidate_rows_window(900, max_bytes=2_000_000)
        rows_1h = self._read_candidate_rows_window(3600, max_bytes=3_000_000)

        def _skip_stats(rows: list[dict]) -> dict[str, int]:
            out: dict[str, int] = {}
            for row in rows:
                reason = str(row.get("reason", "") or "").strip()
                if not reason:
                    continue
                out[reason] = int(out.get(reason, 0)) + 1
            return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:20])

        commit = "unknown"
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=6,
            )
            if proc.returncode == 0:
                commit = (proc.stdout or "").strip()[:12] or commit
        except Exception:
            pass

        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": str(self.positions_source_var.get() or "single"),
            "state_file": state_path,
            "commit": commit,
            "effective_env": env_map,
            "state_summary": {
                "open_count": len(state.get("open_positions", []) or []),
                "closed_count": len(closed_rows),
                "realized_pnl_usd": float(state.get("realized_pnl_usd") or 0.0),
                "wins": sum(1 for x in closed_rows if float(x.get("pnl_usd") or 0.0) > 0),
                "losses": sum(1 for x in closed_rows if float(x.get("pnl_usd") or 0.0) < 0),
            },
            "last_closed": last_closed,
            "skip_stats_15m": _skip_stats(rows_15m),
            "skip_stats_1h": _skip_stats(rows_1h),
        }

        out_dir = os.path.join(PROJECT_ROOT, "snapshots")
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"session_snapshot_{stamp}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Session Snapshot", f"Snapshot saved:\n{out_path}")
        except Exception as exc:
            messagebox.showerror("Session Snapshot", f"Snapshot failed:\n{exc}")

    def on_enable_kill_switch(self) -> None:
        path = self._kill_switch_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} manual_kill_switch\n")
        self.refresh()
        messagebox.showinfo("KILL SWITCH", f"KILL SWITCH enabled:\n{path}")

    def on_disable_kill_switch(self) -> None:
        path = self._kill_switch_path()
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                messagebox.showerror("KILL SWITCH", f"Cannot disable kill switch:\n{exc}")
                return
        self.refresh()
        messagebox.showinfo("KILL SWITCH", "KILL SWITCH disabled.")

    @staticmethod
    def _text_has_active_selection(widget: tk.Text) -> bool:
        try:
            widget.index("sel.first")
            widget.index("sel.last")
            return True
        except tk.TclError:
            return False

    def _attach_text_copy_support(self, widget: tk.Text) -> None:
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Copy", command=lambda w=widget: self._copy_text_selection(w))
        widget.bind("<Control-c>", lambda _e, w=widget: self._copy_text_selection(w))
        widget.bind("<Button-3>", lambda e, m=menu: self._show_context_menu(e, m))

    @staticmethod
    def _show_context_menu(event, menu: tk.Menu) -> str:
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def _copy_text_selection(self, widget: tk.Text) -> str:
        try:
            text = widget.get("sel.first", "sel.last")
        except tk.TclError:
            return "break"
        if not text:
            return "break"
        self.clipboard_clear()
        self.clipboard_append(text)
        return "break"

    @staticmethod
    def _is_near_bottom(widget: tk.Text) -> bool:
        top, bottom = widget.yview()
        return bottom > 0.98

    @staticmethod
    def _restore_scroll(widget: tk.Text, prev_y: tuple[float, float], was_bottom: bool) -> None:
        if was_bottom:
            widget.see("end")
            return
        if prev_y:
            widget.yview_moveto(prev_y[0])

    @staticmethod
    def _stable_iid(prefix: str, raw_key: str) -> str:
        digest = hashlib.md5(raw_key.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    def _sync_tree_rows(
        self,
        tree: ttk.Treeview,
        rows: list[tuple[str, tuple[str, ...], tuple[str, ...]]],
    ) -> None:
        wanted_ids: list[str] = []
        wanted_set: set[str] = set()
        for idx, (iid, values, tags) in enumerate(rows):
            sid = str(iid)
            wanted_ids.append(sid)
            wanted_set.add(sid)
            base_tags = tuple(tags or ())
            stripe_tag = "row_even" if (idx % 2 == 0) else "row_odd"
            final_tags = base_tags + (stripe_tag,)
            if tree.exists(sid):
                tree.item(sid, values=values, tags=final_tags)
            else:
                tree.insert("", tk.END, iid=sid, values=values, tags=final_tags)

        for existing in tree.get_children():
            if existing not in wanted_set:
                tree.delete(existing)

        for idx, sid in enumerate(wanted_ids):
            tree.move(sid, "", idx)

    def _update_positions_kpi(
        self,
        open_count: int,
        closed_count: int,
        wins: int,
        losses: int,
        realized_pnl: float,
        realized_pnl_1h: float = 0.0,
    ) -> None:
        if hasattr(self, "kpi_open_var"):
            self.kpi_open_var.set(str(open_count))
        if hasattr(self, "kpi_closed_var"):
            self.kpi_closed_var.set(str(closed_count))
        total = wins + losses
        winrate = (wins * 100.0 / total) if total > 0 else 0.0
        if hasattr(self, "kpi_winrate_var"):
            self.kpi_winrate_var.set(f"{winrate:.1f}%")
        if hasattr(self, "kpi_pnl_var"):
            self.kpi_pnl_var.set(f"${realized_pnl:+.2f}")
        if hasattr(self, "kpi_pnl_1h_var"):
            self.kpi_pnl_1h_var.set(f"${realized_pnl_1h:+.2f}")
        if hasattr(self, "kpi_updated_var"):
            self.kpi_updated_var.set(datetime.now().strftime("%H:%M:%S"))

    def _ui_tick(self) -> None:
        now_ts = time.time()
        age = max(0.0, now_ts - float(self._last_refresh_ts or 0.0))
        if hasattr(self, "refresh_meta_var"):
            if self._last_refresh_ts > 0:
                self.refresh_meta_var.set(f"Refresh: {age:.1f}s ago | full sync {GUI_REFRESH_INTERVAL_MS}ms")
            else:
                self.refresh_meta_var.set("Refresh: --")

        if hasattr(self, "open_tree"):
            for iid in self.open_tree.get_children():
                vals = list(self.open_tree.item(iid, "values") or ())
                if not vals:
                    continue
                changed = False

                ttl_idx = int(getattr(self, "_open_ttl_col_idx", -1))
                expiry_ts = self._open_row_expiry_ts.get(iid)
                if expiry_ts is not None and ttl_idx >= 0 and ttl_idx < len(vals):
                    left = max(0, int(expiry_ts - now_ts))
                    ttl_text = f"{left} sec"
                    if vals[ttl_idx] != ttl_text:
                        vals[ttl_idx] = ttl_text
                        changed = True

                pnl_hour_idx = int(getattr(self, "_open_pnl_hour_col_idx", -1))
                opened_ts = self._open_row_opened_ts.get(iid)
                pnl_usd = self._open_row_pnl_usd.get(iid)
                if pnl_hour_idx >= 0 and pnl_hour_idx < len(vals):
                    pnl_hour_text = "--"
                    if opened_ts is not None and pnl_usd is not None and opened_ts > 0.0:
                        age_hours = max(1.0 / 3600.0, (now_ts - float(opened_ts)) / 3600.0)
                        pnl_hour_text = f"{(float(pnl_usd) / age_hours):+.2f}"
                    if vals[pnl_hour_idx] != pnl_hour_text:
                        vals[pnl_hour_idx] = pnl_hour_text
                        changed = True

                if changed:
                    self.open_tree.item(iid, values=tuple(vals))

        self.after(GUI_UI_TICK_MS, self._ui_tick)

    def _refresh_positions(self, app_lines: list[str]) -> None:
        state_file = self._selected_positions_state_file()
        state = read_json(state_file)
        if state:
            self._state_cache[state_file] = state
        elif os.path.exists(state_file):
            cached = self._state_cache.get(state_file)
            if isinstance(cached, dict) and cached:
                state = cached
        src = str(self.positions_source_var.get() or "single")
        excluded_symbols = self._stats_excluded_symbols()
        if hasattr(self, "positions_title_var"):
            env_map = self._runtime_env_map()
            mode = str(env_map.get(WALLET_MODE_KEY, "paper") or "paper").strip().lower()
            prefix = "ÐœÐ¾Ð½Ð¸Ñ‚Ð¾Ñ€ live-ÑÐ´ÐµÐ»Ð¾Ðº" if mode == "live" else "ÐœÐ¾Ð½Ð¸Ñ‚Ð¾Ñ€ Ð±ÑƒÐ¼Ð°Ð¶Ð½Ñ‹Ñ… ÑÐ´ÐµÐ»Ð¾Ðº"
            suffix = "" if src == "single" else f" [{src}]"
            self.positions_title_var.set(f"{prefix}{suffix}")
        if state:
            self._refresh_positions_from_state(state)
            if hasattr(self, "pos_summary_var"):
                self.pos_summary_var.set(f"[{src}] {self.pos_summary_var.get()}")
            return

        # Fallback to log parsing if state file is missing.
        open_map: dict[str, dict[str, str]] = {}
        closed: list[dict[str, str]] = []
        wins = 0
        losses = 0
        pnl_sum = 0.0

        for line in app_lines:
            b = BUY_RE.search(line)
            if b:
                address = b.group("address")
                symbol = str(b.group("symbol") or "").upper()
                if symbol in excluded_symbols:
                    continue
                open_map[address] = {
                    "symbol": b.group("symbol"),
                    "entry": b.group("entry"),
                    "size": b.group("size"),
                    "score": b.group("score"),
                }
                continue
            s = SELL_RE.search(line)
            if s:
                symbol = str(s.group("symbol") or "").upper()
                if symbol in excluded_symbols:
                    continue
                pnl_usd = float(s.group("pnl_usd"))
                pnl_sum += pnl_usd
                if pnl_usd >= 0:
                    wins += 1
                else:
                    losses += 1
                closed.append(
                    {
                        "symbol": s.group("symbol"),
                        "reason": s.group("reason"),
                        "pnl_pct": s.group("pnl_pct"),
                        "pnl_usd": s.group("pnl_usd"),
                    }
                )
                # remove one matching open by symbol if exists
                for addr, pos in list(open_map.items()):
                    if pos["symbol"] == s.group("symbol"):
                        open_map.pop(addr, None)
                        break

        open_tree_rows: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        self._open_row_expiry_ts = {}
        self._open_row_opened_ts = {}
        self._open_row_pnl_usd = {}
        for addr, pos in open_map.items():
            iid = self._stable_iid("open_fallback", f"{addr}:{pos['symbol']}")
            open_tree_rows.append(
                (
                    iid,
                    (pos["symbol"], pos["entry"], pos["size"], pos["score"], "--", "--", "--", "--", "--", "--"),
                    ("open_flat",),
                )
            )
        self._sync_tree_rows(self.open_tree, open_tree_rows)

        closed_tree_rows: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        for idx, pos in enumerate(closed[-200:]):
            raw_key = f"{idx}:{pos['symbol']}:{pos['reason']}:{pos['pnl_pct']}:{pos['pnl_usd']}"
            iid = self._stable_iid("closed_fallback", raw_key)
            pnl = self._to_float(pos.get("pnl_usd"), 0.0)
            closed_tag = "closed_pos" if pnl > 0 else ("closed_neg" if pnl < 0 else "closed_flat")
            closed_tree_rows.append((iid, (pos["symbol"], pos["reason"], "--", "--", pos["pnl_pct"], pos["pnl_usd"]), (closed_tag,)))
        self._sync_tree_rows(self.closed_tree, closed_tree_rows)

        self._update_positions_kpi(len(open_map), len(closed), wins, losses, pnl_sum, 0.0)
        excl_note = f" | excluded: {','.join(sorted(excluded_symbols))}" if excluded_symbols else ""
        self.pos_summary_var.set(
            f"[{src}] Open: {len(open_map)} | Closed: {len(closed)} | Wins/Losses: {wins}/{losses} | Total PnL: ${pnl_sum:+.2f}{excl_note}"
        )

    def _refresh_signals(self) -> None:
        if not hasattr(self, "signals_tree"):
            return
        src_label, signals_file = self._selected_signals_file()
        rows = read_jsonl_tail(signals_file, 400)
        if not rows:
            if hasattr(self, "signals_summary_var"):
                self.signals_summary_var.set(f"[{src_label}] No incoming signals yet.")
            self._sync_tree_rows(self.signals_tree, [])
            self._refresh_signal_sources()
            return

        signal_rows: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        for row in reversed(rows[-250:]):
            ts = str(row.get("timestamp", ""))
            ts_text = "-"
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ts_text = dt.astimezone().strftime("%H:%M:%S")
                except Exception:
                    ts_text = ts
            symbol = str(row.get("symbol", "N/A"))
            score = str(row.get("score", 0))
            recommendation = str(row.get("recommendation", "SKIP"))
            risk = str(row.get("risk_level", "N/A"))
            liquidity = f"{float(row.get('liquidity', 0) or 0):.0f}"
            volume = f"{float(row.get('volume_5m', 0) or 0):.0f}"
            change = f"{float(row.get('price_change_5m', 0) or 0):+.2f}"
            iid = self._stable_iid("sig", f"{ts}:{symbol}:{score}:{recommendation}:{risk}:{liquidity}:{volume}:{change}")
            signal_rows.append((iid, (ts_text, symbol, score, recommendation, risk, liquidity, volume, change), ()))

        self._sync_tree_rows(self.signals_tree, signal_rows)
        if hasattr(self, "signals_summary_var"):
            self.signals_summary_var.set(f"[{src_label}] Signals: {len(rows)} | Shown: {min(len(rows), 250)}")
        self._refresh_signal_sources()

    def _refresh_signal_sources(self) -> None:
        if not hasattr(self, "signals_sources_var"):
            return
        src_label, _signals_file = self._selected_signals_file()
        lines: list[str] = []
        for path in self._runtime_log_paths_for_signals():
            lines.extend(read_tail(path, 360))
        counts: dict[str, int] = {}
        total = 0
        for line in lines:
            match = PAIR_SOURCE_RE.search(line)
            if not match:
                continue
            source = match.group("source").strip().lower() or "unknown"
            counts[source] = counts.get(source, 0) + 1
            total += 1
        v2_count = counts.get("uniswap_v2", 0)
        v3_count = counts.get("uniswap_v3", 0)
        other_count = max(0, total - v2_count - v3_count)
        self.signals_sources_var.set(
            f"Sources (tail, {src_label}): v2={v2_count} | v3={v3_count} | other={other_count} | total={total}"
        )

    def _refresh_positions_from_state(self, state: dict) -> None:
        open_rows = state.get("open_positions", []) or []
        closed_rows = state.get("closed_positions", []) or []
        if not isinstance(open_rows, list):
            open_rows = []
        else:
            open_rows = [row for row in open_rows if isinstance(row, dict)]
        if not isinstance(closed_rows, list):
            closed_rows = []
        else:
            closed_rows = [row for row in closed_rows if isinstance(row, dict)]

        fallback_open_rows, fallback_closed_rows = self._recent_positions_from_trade_decisions(max_rows=1400)
        if not open_rows and fallback_open_rows:
            open_rows = [row for row in fallback_open_rows if isinstance(row, dict)]

        def _norm_close_ts(value: object) -> str:
            dt = self._parse_iso_ts(value)
            if dt is None:
                return str(value or "").strip()
            return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()

        def _closed_row_identity(row: dict) -> str:
            # Keep dedup key stable across state/fallback sources:
            # source-specific reason/pnl formatting can differ for the same close event.
            position_id_k = str((row or {}).get("position_id", "") or "").strip()
            candidate_id_k = str((row or {}).get("candidate_id", "") or "").strip()
            token_k = str((row or {}).get("token_address", "") or "").strip().lower()
            opened_k = _norm_close_ts((row or {}).get("opened_at", ""))
            closed_k = _norm_close_ts((row or {}).get("closed_at", ""))
            if position_id_k:
                return f"pid:{position_id_k}"
            if candidate_id_k:
                return f"cid:{candidate_id_k}"
            if token_k and opened_k and closed_k:
                return f"tok:{token_k}|opened:{opened_k}|closed:{closed_k}"
            if token_k and opened_k:
                return f"tok:{token_k}|opened:{opened_k}"
            symbol_k = str((row or {}).get("symbol", "") or "").strip().upper()
            return f"fallback:{token_k}|{symbol_k}|{opened_k}|{closed_k}"

        merged_closed = [row for row in fallback_closed_rows if isinstance(row, dict)]
        merged_closed.extend(closed_rows)
        dedup_closed: dict[str, dict] = {}
        for row in merged_closed:
            dedup_closed[_closed_row_identity(row)] = row
        closed_rows = list(dedup_closed.values())
        raw_untracked = state.get("recovery_untracked", {}) or {}
        untracked_map: dict[str, int] = {}
        if isinstance(raw_untracked, dict):
            for key, value in raw_untracked.items():
                addr = str(key or "").strip().lower()
                if not addr:
                    continue
                try:
                    amount_raw = int(value or 0)
                except Exception:
                    amount_raw = 0
                if amount_raw > 0:
                    untracked_map[addr] = amount_raw
        tx_events = state.get("tx_event_timestamps", []) or []
        tx_events_recent = 0
        if isinstance(tx_events, list):
            now_ts_recent = time.time()
            for raw_ts in tx_events:
                ts = _parse_ts(raw_ts)
                if ts is None:
                    continue
                if (now_ts_recent - ts) <= 20 * 60:
                    tx_events_recent += 1

        excluded_symbols = self._stats_excluded_symbols()
        if excluded_symbols:
            open_rows = [
                row
                for row in open_rows
                if str((row or {}).get("symbol", "") or "").strip().upper() not in excluded_symbols
            ]
            closed_rows = [
                row
                for row in closed_rows
                if str((row or {}).get("symbol", "") or "").strip().upper() not in excluded_symbols
            ]

        try:
            open_rows.sort(
                key=lambda row: (
                    self._parse_iso_ts((row or {}).get("opened_at")) or datetime.fromtimestamp(0, tz=timezone.utc)
                ).timestamp()
            )
        except Exception:
            pass
        try:
            closed_rows.sort(
                key=lambda row: (
                    self._parse_iso_ts((row or {}).get("closed_at")) or datetime.fromtimestamp(0, tz=timezone.utc)
                ).timestamp()
            )
        except Exception:
            pass

        now = datetime.now(timezone.utc)
        cutoff_1h_ts = now.timestamp() - 3600.0

        wins = 0
        losses = 0
        realized_pnl = 0.0
        realized_pnl_1h = 0.0
        for row in closed_rows:
            pnl_usd_v = self._safe_float((row or {}).get("pnl_usd", 0.0), 0.0)
            realized_pnl += pnl_usd_v
            if pnl_usd_v > 0.0:
                wins += 1
            elif pnl_usd_v < 0.0:
                losses += 1
            closed_dt = self._parse_iso_ts((row or {}).get("closed_at"))
            if closed_dt is not None and closed_dt.timestamp() >= cutoff_1h_ts:
                realized_pnl_1h += pnl_usd_v

        env_map = self._runtime_env_map()
        fallback_hold = int(max(1.0, self._safe_float(env_map.get("PAPER_MAX_HOLD_SECONDS", "1800"), 1800.0)))
        breakeven_eps = max(0.0, self._safe_float(env_map.get("PNL_BREAKEVEN_EPSILON_USD", "0.0"), 0.0))

        self._open_row_expiry_ts = {}
        self._open_row_opened_ts = {}
        self._open_row_pnl_usd = {}

        addr_to_symbol: dict[str, str] = {}
        for row in (open_rows + closed_rows):
            addr = str((row or {}).get("token_address", "") or "").strip().lower()
            sym = str((row or {}).get("symbol", "") or "").strip()
            if addr and sym and addr not in addr_to_symbol:
                addr_to_symbol[addr] = sym

        open_tree_rows: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        now_ts = now.timestamp()
        for row in open_rows:
            token_address = str(row.get("token_address", "") or "")
            opened_at_raw = row.get("opened_at", "")
            opened_dt = self._parse_iso_ts(opened_at_raw)
            opened_ts = 0.0
            opened_text = self._format_local_ts(opened_at_raw)
            ttl = "--"
            max_hold = int(max(1.0, self._safe_float(row.get("max_hold_seconds", fallback_hold), float(fallback_hold))))
            if opened_dt is not None:
                opened_ts = opened_dt.timestamp()
                age = int(max(0.0, now_ts - opened_ts))
                left = max(0, max_hold - age)
                ttl = f"{left} sec"

            entry_price = self._safe_float(row.get("entry_price_usd", 0.0), 0.0)
            current_price = self._safe_float(row.get("current_price_usd", 0.0), 0.0)
            position_size = self._safe_float(row.get("position_size_usd", 0.0), 0.0)
            pnl_pct = self._safe_float(row.get("pnl_percent", row.get("pnl_pct", 0.0)), 0.0)
            pnl_usd = self._safe_float(row.get("pnl_usd", 0.0), 0.0)

            if entry_price > 0.0 and current_price > 0.0 and position_size > 0.0:
                derived_pct = ((current_price - entry_price) / entry_price) * 100.0
                derived_usd = (position_size * derived_pct) / 100.0
                if abs(pnl_pct) < 1e-9 and abs(derived_pct) > 1e-6:
                    pnl_pct = derived_pct
                if abs(pnl_usd) < 1e-9 and abs(derived_usd) > 1e-6:
                    pnl_usd = derived_usd

            peak_pct = self._safe_float(row.get("peak_pnl_percent", pnl_pct), pnl_pct)
            if peak_pct < pnl_pct:
                peak_pct = pnl_pct

            if pnl_usd > breakeven_eps:
                tag = "open_pos"
            elif pnl_usd < -breakeven_eps:
                tag = "open_neg"
            else:
                tag = "open_flat"

            pnl_hour_text = "--"
            if opened_ts > 0.0:
                age_hours = max(1.0 / 3600.0, (now_ts - opened_ts) / 3600.0)
                pnl_hour_text = f"{(pnl_usd / age_hours):+.2f}"

            values = (
                str(row.get("symbol", "N/A")),
                f"{entry_price:.8f}",
                f"{position_size:.2f}",
                str(row.get("score", 0)),
                opened_text,
                f"{pnl_pct:+.2f}",
                f"{pnl_usd:+.2f}",
                pnl_hour_text,
                f"{peak_pct:+.2f}",
                ttl,
            )
            key = token_address or f"{row.get('symbol', 'N/A')}|{str(opened_at_raw)}"
            iid = self._stable_iid("open", key)
            open_tree_rows.append((iid, values, (tag,)))
            self._open_row_pnl_usd[iid] = pnl_usd
            if opened_ts > 0.0:
                self._open_row_opened_ts[iid] = opened_ts
                self._open_row_expiry_ts[iid] = opened_ts + float(max_hold)

        for addr, amount_raw in sorted(untracked_map.items()):
            symbol = addr_to_symbol.get(addr, "UNTRACKED")
            if symbol.strip().upper() in excluded_symbols:
                continue
            values = (
                symbol,
                "--",
                f"raw:{amount_raw}",
                "wallet",
                "--",
                "--",
                "--",
                "--",
                "--",
                "wallet",
            )
            iid = self._stable_iid("open_untracked", addr)
            open_tree_rows.append((iid, values, ("open_untracked",)))

        self._sync_tree_rows(self.open_tree, open_tree_rows)

        closed_tree_rows: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        for row in closed_rows[-200:]:
            opened_at_raw = row.get("opened_at", "")
            closed_at_raw = row.get("closed_at", "")
            symbol = str(row.get("symbol", "N/A"))
            reason = str(row.get("close_reason", row.get("reason", "")))
            pnl_pct_v = self._safe_float(row.get("pnl_percent", row.get("pnl_pct", 0.0)), 0.0)
            pnl_usd_v = self._safe_float(row.get("pnl_usd", 0.0), 0.0)
            values = (
                symbol,
                reason,
                self._format_local_ts(opened_at_raw),
                self._format_local_ts(closed_at_raw),
                f"{pnl_pct_v:+.2f}",
                f"{pnl_usd_v:+.2f}",
            )
            key = _closed_row_identity(row)
            iid = self._stable_iid("closed", key)
            if pnl_usd_v > breakeven_eps:
                closed_tag = "closed_pos"
            elif pnl_usd_v < -breakeven_eps:
                closed_tag = "closed_neg"
            else:
                closed_tag = "closed_flat"
            closed_tree_rows.append((iid, values, (closed_tag,)))

        self._sync_tree_rows(self.closed_tree, closed_tree_rows)
        self._update_positions_kpi(len(open_rows), len(closed_rows), wins, losses, realized_pnl, realized_pnl_1h)
        excl_note = f" | excluded: {','.join(sorted(excluded_symbols))}" if excluded_symbols else ""
        self.pos_summary_var.set(
            f"Open: {len(open_rows)} | Closed: {len(closed_rows)} | Wins/Losses: {wins}/{losses} | Realized PnL: ${realized_pnl:+.2f}{excl_note}"
        )
        if hasattr(self, "untracked_summary_var"):
            if untracked_map:
                self.untracked_summary_var.set(
                    f"Untracked in wallet: {len(untracked_map)} (present in wallet, missing in bot open_positions)"
                )
            elif len(open_rows) == 0 and tx_events_recent > 0:
                self.untracked_summary_var.set(
                    f"Visibility alert: recent live tx={tx_events_recent} in last 20m, but open_positions is empty"
                )
            else:
                self.untracked_summary_var.set("")
    def _schedule_live_wallet_poll(self) -> None:
        # Keep UI responsive: RPC calls happen in a background thread.
        if getattr(self, "_live_wallet_polling", False):
            self.after(int(LIVE_BALANCE_POLL_SECONDS * 1000), self._schedule_live_wallet_poll)
            return

        env_map = read_runtime_env_map()
        address = str(env_map.get("LIVE_WALLET_ADDRESS", "") or "").strip()
        rpc_primary = str(env_map.get("RPC_PRIMARY", "") or "").strip()
        rpc_secondary = str(env_map.get("RPC_SECONDARY", "") or "").strip()
        rpc_urls = [u for u in (rpc_primary, rpc_secondary) if u]

        if not address:
            self._live_wallet_last_err = "no_live_wallet_address"
            self._live_wallet_last_ts = 0.0
            self.after(int(LIVE_BALANCE_POLL_SECONDS * 1000), self._schedule_live_wallet_poll)
            return
        if Web3 is None or HTTPProvider is None:
            self._live_wallet_last_err = "web3_not_available"
            self._live_wallet_last_ts = 0.0
            self.after(int(LIVE_BALANCE_POLL_SECONDS * 1000), self._schedule_live_wallet_poll)
            return
        if not rpc_urls:
            self._live_wallet_last_err = "no_rpc_configured"
            self._live_wallet_last_ts = 0.0
            self.after(int(LIVE_BALANCE_POLL_SECONDS * 1000), self._schedule_live_wallet_poll)
            return

        try:
            weth_price = float(env_map.get("WETH_PRICE_FALLBACK_USD", "0") or 0)
        except Exception:
            weth_price = 0.0

        self._live_wallet_polling = True
        threading.Thread(
            target=self._poll_live_wallet_worker,
            args=(address, rpc_urls, weth_price),
            daemon=True,
        ).start()

    def _poll_live_wallet_worker(self, address: str, rpc_urls: list[str], weth_price_fallback_usd: float) -> None:
        eth = 0.0
        err = ""
        ts = datetime.now(timezone.utc).timestamp()

        try:
            checksum = Web3.to_checksum_address(address)  # type: ignore[union-attr]
        except Exception:
            checksum = address

        for url in rpc_urls:
            try:
                w3 = Web3(HTTPProvider(url, request_kwargs={"timeout": 10}))  # type: ignore[misc]
                bal_wei = int(w3.eth.get_balance(checksum))
                eth = bal_wei / 1e18
                err = ""
                break
            except Exception as exc:
                err = f"rpc_error:{exc}"
                continue

        def _apply() -> None:
            self._live_wallet_eth = float(eth)
            self._live_wallet_usd = float(eth * max(0.0, float(weth_price_fallback_usd)))
            self._live_wallet_last_ts = float(ts) if not err else 0.0
            self._live_wallet_last_err = str(err or "")
            self._live_wallet_polling = False
            try:
                self._refresh_wallet()
            except Exception:
                pass
            self.after(int(LIVE_BALANCE_POLL_SECONDS * 1000), self._schedule_live_wallet_poll)

        try:
            self.after(0, _apply)
        except Exception:
            return

    def _refresh_wallet(self) -> None:
        env_map = read_runtime_env_map()
        mode = str(env_map.get(WALLET_MODE_KEY, self.wallet_mode_var.get() if hasattr(self, "wallet_mode_var") else "paper")).strip().lower()
        live_balance = float(env_map.get(LIVE_WALLET_BALANCE_KEY, "0") or 0)

        state = read_json(self._runtime_state_file())
        paper_balance = float(state.get("paper_balance_usd", env_map.get("WALLET_BALANCE_USD", "0")) or 0)
        paper_open = len(state.get("open_positions", []) or [])
        paper_closed = len(state.get("closed_positions", []) or [])
        stair_floor = float(state.get("stair_floor_usd", env_map.get("STAIR_STEP_START_BALANCE_USD", "0")) or 0)
        stair_enabled = str(env_map.get(STAIR_STEP_ENABLED_KEY, "false")).strip().lower()
        emergency_reason = str(state.get("emergency_halt_reason", "") or "").strip()

        active_live = self._live_wallet_usd if float(getattr(self, "_live_wallet_last_ts", 0.0) or 0.0) > 0 else live_balance
        active = paper_balance if mode == "paper" else active_live
        mode_label = "PAPER" if mode == "paper" else "LIVE"
        if hasattr(self, "wallet_summary_var"):
            self.wallet_summary_var.set(
                f"Mode: {mode_label} | Active balance: ${active:.2f} | Paper: ${paper_balance:.2f} | Live: ${live_balance:.2f} | Stair: {stair_enabled.upper()} floor=${stair_floor:.2f} | Open: {paper_open} | Closed: {paper_closed} | Halt: {emergency_reason or 'none'}"
            )
        if hasattr(self, "live_onchain_var"):
            if float(getattr(self, "_live_wallet_last_ts", 0.0) or 0.0) > 0:
                when = datetime.fromtimestamp(float(self._live_wallet_last_ts), tz=timezone.utc).astimezone().strftime("%H:%M:%S")
                try:
                    price = float(env_map.get("WETH_PRICE_FALLBACK_USD", "0") or 0)
                except Exception:
                    price = 0.0
                usd_text = f"${(self._live_wallet_eth * price):.2f}" if price > 0 else "(USD n/a)"
                self.live_onchain_var.set(f"Live on-chain: {self._live_wallet_eth:.6f} ETH  ~ {usd_text}  (updated {when})")
            elif str(getattr(self, "_live_wallet_last_err", "") or ""):
                self.live_onchain_var.set(f"Live on-chain: error: {self._live_wallet_last_err}")
            else:
                self.live_onchain_var.set("Live on-chain: (not checked yet)")
        if hasattr(self, "wallet_mode_var"):
            self.wallet_mode_var.set(mode if mode in {"paper", "live"} else "paper")
        if hasattr(self, "live_balance_var"):
            self.live_balance_var.set(f"{live_balance:.2f}")
        if hasattr(self, "paper_balance_set_var"):
            self.paper_balance_set_var.set(f"{paper_balance:.2f}")
        if "WALLET_BALANCE_USD" in self.env_vars:
            self.env_vars["WALLET_BALANCE_USD"].set(f"{paper_balance:.2f}")
        if hasattr(self, "stair_enabled_var"):
            self.stair_enabled_var.set("true" if stair_enabled == "true" else "false")
        if hasattr(self, "stair_start_var"):
            self.stair_start_var.set(str(env_map.get("STAIR_STEP_START_BALANCE_USD", "0")).strip() or "0")
        if hasattr(self, "stair_size_var"):
            self.stair_size_var.set(str(env_map.get("STAIR_STEP_SIZE_USD", "5")).strip() or "5")
        if "STAIR_STEP_START_BALANCE_USD" in self.env_vars:
            self.env_vars["STAIR_STEP_START_BALANCE_USD"].set(
                str(env_map.get("STAIR_STEP_START_BALANCE_USD", "0")).strip() or "0"
            )
        if "STAIR_STEP_SIZE_USD" in self.env_vars:
            self.env_vars["STAIR_STEP_SIZE_USD"].set(str(env_map.get("STAIR_STEP_SIZE_USD", "5")).strip() or "5")

    def on_apply_wallet_mode(self) -> None:
        mode = str(self.wallet_mode_var.get()).strip().lower()
        if mode not in {"paper", "live"}:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ°", "Ð ÐµÐ¶Ð¸Ð¼ ÐºÐ¾ÑˆÐµÐ»ÑŒÐºÐ° Ð´Ð¾Ð»Ð¶ÐµÐ½ Ð±Ñ‹Ñ‚ÑŒ paper Ð¸Ð»Ð¸ live.")
            return
        save_runtime_env_map({WALLET_MODE_KEY: mode})
        self._refresh_wallet()
        if mode == "live":
            messagebox.showinfo(
                "Live mode",
                "Ð’Ð½Ð¸Ð¼Ð°Ð½Ð¸Ðµ: live execution Ð¿Ð¾ÐºÐ° Ð½Ðµ Ð¿Ð¾Ð´ÐºÐ»ÑŽÑ‡ÐµÐ½. Ð¡ÐµÐ¹Ñ‡Ð°Ñ ÑÑ‚Ð¾ Ñ€ÐµÐ¶Ð¸Ð¼ Ð¾Ñ‚Ð¾Ð±Ñ€Ð°Ð¶ÐµÐ½Ð¸Ñ Ð±Ð°Ð»Ð°Ð½ÑÐ°.",
            )

    def on_apply_stair_mode(self) -> None:
        mode = str(self.stair_enabled_var.get()).strip().lower()
        if mode not in {"true", "false"}:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ°", "Step protection Ð´Ð¾Ð»Ð¶ÐµÐ½ Ð±Ñ‹Ñ‚ÑŒ true Ð¸Ð»Ð¸ false.")
            return
        save_runtime_env_map({STAIR_STEP_ENABLED_KEY: mode})
        if STAIR_STEP_ENABLED_KEY in self.env_vars:
            self.env_vars[STAIR_STEP_ENABLED_KEY].set(mode)
        self._refresh_wallet()

    def on_apply_stair_params(self) -> None:
        try:
            start_floor = float(str(self.stair_start_var.get()).strip())
            step_size = float(str(self.stair_size_var.get()).strip())
        except ValueError:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ°", "Step start/size Ð´Ð¾Ð»Ð¶Ð½Ñ‹ Ð±Ñ‹Ñ‚ÑŒ Ñ‡Ð¸ÑÐ»Ð°Ð¼Ð¸.")
            return
        if start_floor < 0:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ°", "Step start floor Ð½Ðµ Ð¼Ð¾Ð¶ÐµÑ‚ Ð±Ñ‹Ñ‚ÑŒ Ð¾Ñ‚Ñ€Ð¸Ñ†Ð°Ñ‚ÐµÐ»ÑŒÐ½Ñ‹Ð¼.")
            return
        if step_size <= 0:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ°", "Step size Ð´Ð¾Ð»Ð¶ÐµÐ½ Ð±Ñ‹Ñ‚ÑŒ Ð±Ð¾Ð»ÑŒÑˆÐµ 0.")
            return
        save_runtime_env_map(
            {
                "STAIR_STEP_START_BALANCE_USD": f"{start_floor:.2f}",
                "STAIR_STEP_SIZE_USD": f"{step_size:.2f}",
            }
        )
        if "STAIR_STEP_START_BALANCE_USD" in self.env_vars:
            self.env_vars["STAIR_STEP_START_BALANCE_USD"].set(f"{start_floor:.2f}")
        if "STAIR_STEP_SIZE_USD" in self.env_vars:
            self.env_vars["STAIR_STEP_SIZE_USD"].set(f"{step_size:.2f}")
        self._refresh_wallet()

    def on_apply_live_balance(self) -> None:
        try:
            val = float(self.live_balance_var.get().strip())
        except ValueError:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ°", "Live balance Ð´Ð¾Ð»Ð¶ÐµÐ½ Ð±Ñ‹Ñ‚ÑŒ Ñ‡Ð¸ÑÐ»Ð¾Ð¼.")
            return
        if val < 0:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ°", "Live balance Ð½Ðµ Ð¼Ð¾Ð¶ÐµÑ‚ Ð±Ñ‹Ñ‚ÑŒ Ð¾Ñ‚Ñ€Ð¸Ñ†Ð°Ñ‚ÐµÐ»ÑŒÐ½Ñ‹Ð¼.")
            return
        save_runtime_env_map({LIVE_WALLET_BALANCE_KEY: f"{val:.2f}"})
        self._refresh_wallet()

    def on_set_paper_275(self) -> None:
        self.paper_balance_set_var.set("2.75")
        self.on_apply_paper_balance()

    def on_apply_paper_balance(self) -> None:
        try:
            val = float(self.paper_balance_set_var.get().strip())
        except ValueError:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ°", "Paper balance Ð´Ð¾Ð»Ð¶ÐµÐ½ Ð±Ñ‹Ñ‚ÑŒ Ñ‡Ð¸ÑÐ»Ð¾Ð¼.")
            return
        if val < 0:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ°", "Paper balance Ð½Ðµ Ð¼Ð¾Ð¶ÐµÑ‚ Ð±Ñ‹Ñ‚ÑŒ Ð¾Ñ‚Ñ€Ð¸Ñ†Ð°Ñ‚ÐµÐ»ÑŒÐ½Ñ‹Ð¼.")
            return

        state_file = self._runtime_state_file()
        state = read_json(state_file)
        open_positions = state.get("open_positions", []) or []
        if open_positions:
            if not messagebox.askyesno(
                "Open trades detected",
                "Open paper trades exist in the active state file.\n\n"
                "To change paper balance, open trades must be cleared (removed from paper tracking).\n\n"
                "Clear open trades and set the new paper balance?",
            ):
                return
            # Soft reset: clear only the parts that can block paper trading, keep closed history if any.
            state["open_positions"] = []
            state["price_guard_pending"] = {}
            state["token_cooldowns"] = {}
            state["trade_open_timestamps"] = []
            state["trading_pause_until_ts"] = 0.0
            state["emergency_halt_reason"] = ""
            state["emergency_halt_ts"] = 0.0
        state["initial_balance_usd"] = float(val)
        state["paper_balance_usd"] = float(val)
        state["realized_pnl_usd"] = 0.0
        state["day_start_equity_usd"] = float(val)
        state["day_realized_pnl_usd"] = 0.0
        state["stair_floor_usd"] = 0.0
        state["stair_peak_balance_usd"] = float(val)
        try:
            write_json_atomic_locked(
                state_file,
                state,
                timeout_seconds=STATE_FILE_LOCK_TIMEOUT_SECONDS,
                poll_seconds=STATE_FILE_LOCK_RETRY_SECONDS,
                encoding="utf-8",
                ensure_ascii=False,
                indent=2,
            )
        except StateFileLockError:
            messagebox.showerror(
                "Error",
                "State file is busy (locked by another process). Retry in a moment.",
            )
            return
        except OSError as exc:
            messagebox.showerror("Error", f"Failed to save state: {exc}")
            return
        save_runtime_env_map({"WALLET_BALANCE_USD": f"{val:.2f}"})
        if "WALLET_BALANCE_USD" in self.env_vars:
            self.env_vars["WALLET_BALANCE_USD"].set(f"{val:.2f}")
        self._refresh_wallet()

    def on_critical_reset(self) -> None:
        if not messagebox.askyesno(
            "ÐšÑ€Ð¸Ñ‚Ð¸Ñ‡ÐµÑÐºÐ¸Ð¹ ÑÐ±Ñ€Ð¾Ñ",
            "Ð¡Ð±Ñ€Ð¾ÑÐ¸Ñ‚ÑŒ paper-ÑÐ¾ÑÑ‚Ð¾ÑÐ½Ð¸Ðµ Ð¿Ð¾Ð»Ð½Ð¾ÑÑ‚ÑŒÑŽ? ÐžÑ‚ÐºÑ€Ñ‹Ñ‚Ñ‹Ðµ Ð¸ Ð·Ð°ÐºÑ€Ñ‹Ñ‚Ñ‹Ðµ ÑÐ´ÐµÐ»ÐºÐ¸ Ð±ÑƒÐ´ÑƒÑ‚ ÑƒÐ´Ð°Ð»ÐµÐ½Ñ‹.",
        ):
            return

        pid = read_pid()
        if pid and is_running(pid):
            ok, msg = stop_bot()
            if not ok:
                messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ°", f"ÐÐµ ÑƒÐ´Ð°Ð»Ð¾ÑÑŒ Ð¾ÑÑ‚Ð°Ð½Ð¾Ð²Ð¸Ñ‚ÑŒ Ð±Ð¾Ñ‚Ð° Ð¿ÐµÑ€ÐµÐ´ ÑÐ±Ñ€Ð¾ÑÐ¾Ð¼: {msg}")
                return

        env_map = read_runtime_env_map()
        try:
            balance = float(env_map.get("WALLET_BALANCE_USD", "2.75") or 2.75)
        except ValueError:
            balance = 2.75

        state = {
            "initial_balance_usd": balance,
            "paper_balance_usd": balance,
            "realized_pnl_usd": 0.0,
            "total_plans": 0,
            "total_executed": 0,
            "total_closed": 0,
            "total_wins": 0,
            "total_losses": 0,
            "stair_floor_usd": 0.0,
            "stair_peak_balance_usd": balance,
            "open_positions": [],
            "closed_positions": [],
        }
        state_file = self._runtime_state_file()
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        try:
            write_json_atomic_locked(
                state_file,
                state,
                timeout_seconds=STATE_FILE_LOCK_TIMEOUT_SECONDS,
                poll_seconds=STATE_FILE_LOCK_RETRY_SECONDS,
                encoding="utf-8",
                ensure_ascii=False,
                indent=2,
            )
        except StateFileLockError:
            messagebox.showerror(
                "Error",
                "State file is busy (locked by another process). Retry in a moment.",
            )
            return
        except OSError as exc:
            messagebox.showerror("Error", f"Failed to save state: {exc}")
            return
        save_runtime_env_map({"WALLET_BALANCE_USD": f"{balance:.2f}"})
        if "WALLET_BALANCE_USD" in self.env_vars:
            self.env_vars["WALLET_BALANCE_USD"].set(f"{balance:.2f}")
        self._refresh_wallet()
        self._refresh_positions([])
        messagebox.showinfo("Ð“Ð¾Ñ‚Ð¾Ð²Ð¾", f"Paper-ÑÐ¾ÑÑ‚Ð¾ÑÐ½Ð¸Ðµ ÑÐ±Ñ€Ð¾ÑˆÐµÐ½Ð¾. Ð¢ÐµÐºÑƒÑ‰Ð¸Ð¹ Ð±Ð°Ð»Ð°Ð½Ñ: ${balance:.2f}")

    def auto_refresh(self) -> None:
        try:
            self.refresh()
        except Exception as exc:
            # Keep periodic refresh alive even if one cycle fails.
            try:
                self.hint_var.set(f"Refresh error: {exc.__class__.__name__}. Retrying...")
            except Exception:
                pass
        finally:
            self.after(GUI_REFRESH_INTERVAL_MS, self.auto_refresh)

    def on_start(self) -> None:
        # Avoid mixing single run with matrix mode.
        ok, msg = start_bot()
        if not ok:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ° Ð·Ð°Ð¿ÑƒÑÐºÐ°", msg)
        self.refresh()

    def on_stop(self) -> None:
        ok, msg = stop_bot()
        m_ok, m_msg = run_powershell_script(MATRIX_STOPPER, ["-HardKill"])
        if not ok:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ° Ð¾ÑÑ‚Ð°Ð½Ð¾Ð²ÐºÐ¸", msg)
        elif not m_ok:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ° Ð¾ÑÑ‚Ð°Ð½Ð¾Ð²ÐºÐ¸ Matrix", m_msg)
        self.refresh()

    def on_restart(self) -> None:
        stop_bot()
        ok, msg = start_bot()
        if not ok:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ° Ñ€ÐµÑÑ‚Ð°Ñ€Ñ‚Ð°", msg)
        self.refresh()

    def on_matrix_start(self) -> None:
        profile_ids = list(self._matrix_selected_profile_ids)
        if profile_ids:
            count = str(len(profile_ids))
            self.matrix_count_var.set(count)
        else:
            count = str(self.matrix_count_var.get() or "2").strip()
        if count not in {"1", "2", "3", "4"}:
            count = "2"
            self.matrix_count_var.set(count)
        profile_hint = f" profiles={','.join(profile_ids)}" if profile_ids else ""
        self.hint_var.set(f"Matrix start in progress ({count})...{profile_hint}")

        def _matrix_start_worker() -> tuple[bool, str]:
            args = ["-Run"]
            if profile_ids:
                args.extend(["-ProfileIds", ",".join(profile_ids)])
            else:
                args = ["-Count", count, "-Run"]
            ok, msg = run_powershell_script(
                MATRIX_LAUNCHER,
                args,
                timeout_seconds=180,
            )
            if ok:
                return ok, msg
            # If launcher timed out but instances are already alive, treat as successful start.
            txt = str(msg or "").lower()
            if "timed out" in txt:
                alive = matrix_alive_count()
                target = len(profile_ids) if profile_ids else int(count)
                if alive >= target:
                    return True, f"Matrix started (timeout fallback). Alive: {alive}/{target}"
            return False, msg

        self._run_background(
            title="Matrix Start",
            worker=_matrix_start_worker,
            on_success=lambda _msg: self.hint_var.set(
                f"Matrix started: {count} instances." if not profile_ids else f"Matrix started: {','.join(profile_ids)}"
            ),
            on_error=lambda _msg: self.hint_var.set("Matrix start failed. Check popup/logs."),
        )

    def on_matrix_stop(self) -> None:
        self.hint_var.set("Matrix stop in progress...")
        self._run_background(
            title="Matrix Stop",
            worker=lambda: run_powershell_script(MATRIX_STOPPER, ["-HardKill"]),
            on_success=lambda _msg: self.hint_var.set("Matrix stopped."),
            on_error=lambda _msg: self.hint_var.set("Matrix stop failed. Check popup/logs."),
        )

    def on_matrix_summary(self) -> None:
        self._run_background(
            title="Matrix Summary",
            worker=lambda: run_powershell_script(MATRIX_SUMMARY, []),
            on_success=None,
        )

    def _run_background(
        self,
        *,
        title: str,
        worker,
        on_success=None,
        on_error=None,
    ) -> None:
        def _task() -> None:
            try:
                ok, msg = worker()
            except Exception as exc:
                ok, msg = False, str(exc)

            def _finish() -> None:
                try:
                    self.refresh()
                except Exception:
                    pass
                if ok:
                    if callable(on_success):
                        try:
                            on_success(msg)
                        except Exception:
                            pass
                    messagebox.showinfo(title, str(msg)[:4000])
                else:
                    if callable(on_error):
                        try:
                            on_error(msg)
                        except Exception:
                            pass
                    messagebox.showerror(title, str(msg)[:4000])

            self.after(0, _finish)

        threading.Thread(target=_task, daemon=True).start()

    def on_reload_env(self) -> None:
        # GUI keeps values in memory; restarting the bot does not refresh input fields.
        # This explicitly reloads `.env` into the GUI controls.
        try:
            self._load_settings()
        except Exception as exc:
            messagebox.showerror("ÐžÑˆÐ¸Ð±ÐºÐ°", f"ÐÐµ ÑƒÐ´Ð°Ð»Ð¾ÑÑŒ Ð¿ÐµÑ€ÐµÑ‡Ð¸Ñ‚Ð°Ñ‚ÑŒ .env: {exc}")
            return
        self.refresh()
        messagebox.showinfo("Ð“Ð¾Ñ‚Ð¾Ð²Ð¾", "ÐÐ°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ¸ Ð¿ÐµÑ€ÐµÑ‡Ð¸Ñ‚Ð°Ð½Ñ‹ Ð¸Ð· .env.")

    def on_clear_logs(self) -> None:
        paths = {APP_LOG, OUT_LOG, ERR_LOG}
        for p in self._runtime_log_paths_for_signals():
            paths.add(p)
        paths.add(self._resolve_candidates_log_path())
        paths.add(self._resolve_trade_decisions_log_path())
        _, signals_file = self._selected_signals_file()
        paths.add(signals_file)
        for path in paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            truncate_file(path)
        self.refresh()

    def on_clear_gui(self) -> None:
        # UI-only cleanup: clears views and truncates log files. Does not touch wallet state or KILL SWITCH.
        self.on_clear_logs()
        self.on_clear_signals()
        for attr in ("feed_text", "raw_text", "critical_text", "log_text"):
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            try:
                widget.delete("1.0", tk.END)
            except Exception:
                pass
        for attr in ("open_tree", "closed_tree", "signals_tree"):
            tree = getattr(self, attr, None)
            if tree is None:
                continue
            try:
                tree.delete(*tree.get_children())
            except Exception:
                pass
        if hasattr(self, "pos_summary_var"):
            self.pos_summary_var.set("ÐžÑ‡Ð¸ÑÑ‚ÐºÐ° Ð²Ñ‹Ð¿Ð¾Ð»Ð½ÐµÐ½Ð°.")
        if hasattr(self, "hint_var"):
            self.hint_var.set("GUI Ð¾Ñ‡Ð¸Ñ‰ÐµÐ½: Ð»Ð¾Ð³Ð¸/ÑÐ¸Ð³Ð½Ð°Ð»Ñ‹/Ñ‚Ð°Ð±Ð»Ð¸Ñ†Ñ‹. KILL SWITCH Ð½Ðµ Ñ‚Ñ€Ð¾Ð½ÑƒÑ‚.")

    def on_clear_signals(self) -> None:
        src_label, signals_file = self._selected_signals_file()
        os.makedirs(os.path.dirname(signals_file), exist_ok=True)
        truncate_file(signals_file)
        if hasattr(self, "signals_tree"):
            self.signals_tree.delete(*self.signals_tree.get_children())
        if hasattr(self, "signals_summary_var"):
            self.signals_summary_var.set(f"[{src_label}] No incoming signals yet.")

    def on_signal_check(self) -> None:
        env_map = self._runtime_env_map()
        signal_source = str(env_map.get("SIGNAL_SOURCE", "dexscreener") or "dexscreener").strip().lower()

        src_label, signals_file = self._selected_signals_file()
        rows = read_jsonl_tail(signals_file, 400)
        last_alert = "-"
        if rows:
            ts = str(rows[-1].get("timestamp", "")).strip()
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    last_alert = dt.astimezone().strftime("%H:%M:%S")
                except Exception:
                    last_alert = ts

        app_lines: list[str] = []
        for path in self._runtime_log_paths_for_signals():
            app_lines.extend(read_tail(path, 260))
        scan_count = sum(1 for line in app_lines if "Scanned " in line)
        pair_detected = sum(1 for line in app_lines if "PAIR_DETECTED" in line)
        filter_pass = sum(1 for line in app_lines if "FILTER_PASS" in line)
        auto_buy = sum(1 for line in app_lines if "AUTO_BUY" in line)
        auto_sell = sum(1 for line in app_lines if "AUTO_SELL" in line)
        critical_events = sum(1 for line in app_lines if "CRITICAL_AUTO_RESET" in line)
        kill_events = sum(1 for line in app_lines if "KILL_SWITCH" in line)

        state = read_json(self._selected_positions_state_file())
        paper_balance = float(state.get("paper_balance_usd", env_map.get("WALLET_BALANCE_USD", "0")) or 0)
        open_count = len(state.get("open_positions", []) or [])
        closed_count = len(state.get("closed_positions", []) or [])

        onchain_probe = "n/a"
        if signal_source == "onchain":
            if not os.path.exists(PYTHON_PATH):
                onchain_probe = f"python_not_found:{PYTHON_PATH}"
            else:
                try:
                    diag_env = os.environ.copy()
                    diag_env["ONCHAIN_LAST_BLOCK_FILE"] = os.path.join(LOG_DIR, "diag_last_block_base.txt")
                    diag_env["ONCHAIN_SEEN_PAIRS_FILE"] = os.path.join(LOG_DIR, "diag_seen_pairs_base.json")
                    result = subprocess.run(
                        [PYTHON_PATH, "-m", "monitor.onchain_factory", "--once"],
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=diag_env,
                    )
                    output = (result.stdout or "").strip()
                    if not output:
                        output = (result.stderr or "").strip()
                    onchain_probe = output.splitlines()[-1] if output else "no_output"
                except subprocess.TimeoutExpired:
                    onchain_probe = "timeout"
                except Exception as exc:
                    onchain_probe = f"error:{exc}"

        summary_line = (
            f"source={src_label} signals={len(rows)} last={last_alert} scans={scan_count} "
            f"pair={pair_detected} pass={filter_pass} buy={auto_buy} sell={auto_sell} "
            f"paper=${paper_balance:.2f} open={open_count} closed={closed_count}"
        )
        if hasattr(self, "signals_diag_var"):
            self.signals_diag_var.set(f"Signal check: {summary_line}")

        details = [
            f"Source: {signal_source}",
            f"On-chain probe: {onchain_probe}",
            f"Signals in feed: {len(rows)} (last: {last_alert})",
            f"Recent runtime (app.log tail): scans={scan_count}, PAIR_DETECTED={pair_detected}, FILTER_PASS={filter_pass}",
            f"Paper activity: AUTO_BUY={auto_buy}, AUTO_SELL={auto_sell}",
            f"Critical: CRITICAL_AUTO_RESET={critical_events}, KILL_SWITCH={kill_events}",
            f"Paper state: balance=${paper_balance:.2f}, open={open_count}, closed={closed_count}",
        ]
        messagebox.showinfo("Signal Check", "\n".join(details))

    def on_prune_closed_trades(self) -> None:
        env_map = self._runtime_env_map()
        days_raw = env_map.get("CLOSED_TRADES_MAX_AGE_DAYS", "14").strip() or "14"
        try:
            days = max(0, int(days_raw))
        except ValueError:
            days = 14
        removed = self._cleanup_closed_trades(
            days=days,
            remove_all=False,
            state_file=self._selected_positions_state_file(),
        )
        messagebox.showinfo("Ð“Ð¾Ñ‚Ð¾Ð²Ð¾", f"Ð£Ð´Ð°Ð»ÐµÐ½Ð¾ Ð·Ð°ÐºÑ€Ñ‹Ñ‚Ñ‹Ñ… ÑÐ´ÐµÐ»Ð¾Ðº: {removed}")
        self.refresh()

    def on_clear_closed_trades(self) -> None:
        removed = self._cleanup_closed_trades(
            days=0,
            remove_all=True,
            state_file=self._selected_positions_state_file(),
        )
        messagebox.showinfo("Ð“Ð¾Ñ‚Ð¾Ð²Ð¾", f"Ð£Ð´Ð°Ð»ÐµÐ½Ð¾ Ð·Ð°ÐºÑ€Ñ‹Ñ‚Ñ‹Ñ… ÑÐ´ÐµÐ»Ð¾Ðº: {removed}")
        self.refresh()

    @staticmethod
    def _cleanup_closed_trades(days: int, remove_all: bool, state_file: str) -> int:
        state = read_json(state_file)
        if not state:
            return 0
        closed_rows = state.get("closed_positions", []) or []
        before = len(closed_rows)
        if before == 0:
            return 0

        if remove_all or days <= 0:
            state["closed_positions"] = []
        else:
            now_ts = datetime.now(timezone.utc).timestamp()
            cutoff_ts = now_ts - (days * 86400)
            kept: list[dict] = []
            for row in closed_rows:
                ts = 0.0
                closed_raw = str(row.get("closed_at") or "")
                opened_raw = str(row.get("opened_at") or "")
                if closed_raw:
                    try:
                        dt = datetime.fromisoformat(closed_raw)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        ts = dt.timestamp()
                    except Exception:
                        ts = 0.0
                if ts == 0.0 and opened_raw:
                    try:
                        dt = datetime.fromisoformat(opened_raw)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        ts = dt.timestamp()
                    except Exception:
                        ts = 0.0
                if ts >= cutoff_ts:
                    kept.append(row)
            state["closed_positions"] = kept

        try:
            write_json_atomic_locked(
                state_file,
                state,
                timeout_seconds=STATE_FILE_LOCK_TIMEOUT_SECONDS,
                poll_seconds=STATE_FILE_LOCK_RETRY_SECONDS,
                encoding="utf-8",
                ensure_ascii=False,
                indent=2,
            )
        except StateFileLockError:
            messagebox.showerror(
                "Error",
                "State file is busy (locked by another process). Retry in a moment.",
            )
            return 0
        except OSError as exc:
            messagebox.showerror("Error", f"Failed to save state: {exc}")
            return 0
        return max(0, before - len(state.get("closed_positions", [])))


if __name__ == "__main__":
    gui_lock = GuiInstanceLock()
    if not gui_lock.acquire():
        print("Another launcher_gui.py instance is already running. Exit.")
    else:
        try:
            App().mainloop()
        finally:
            gui_lock.release()

