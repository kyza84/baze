"""Dark control panel for bot process, activity feed, and settings."""

from __future__ import annotations

import os
import re
import subprocess
import tkinter as tk
import ctypes
import threading
from datetime import datetime, timezone
import json
import ssl
import urllib.request
from tkinter import messagebox, ttk

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
    """Fix UTF-8 bytes that were mistakenly decoded as latin-1 (common Ð...Ñ... mojibake)."""
    if not isinstance(text, str):
        return str(text)
    if "Ð" not in text and "Ñ" not in text:
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
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
PAPER_STATE_FILE = os.path.join(PROJECT_ROOT, "trading", "paper_state.json")
WALLET_MODE_KEY = "WALLET_MODE"
LIVE_WALLET_BALANCE_KEY = "LIVE_WALLET_BALANCE_USD"
STAIR_STEP_ENABLED_KEY = "STAIR_STEP_ENABLED"
GUI_MUTEX_NAME = "Global\\solana_alert_bot_launcher_gui_single_instance"
LIVE_BALANCE_POLL_SECONDS = 12

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

SETTINGS_FIELDS_RAW = [
    ("PERSONAL_MODE", "Личный режим"),
    ("PERSONAL_TELEGRAM_ID", "Telegram ID"),
    ("MIN_TOKEN_SCORE", "Мин. скор"),
    ("AUTO_FILTER_ENABLED", "Автофильтр"),
    ("AUTO_TRADE_ENABLED", "Автотрейд"),
    ("AUTO_TRADE_PAPER", "Бумажный режим"),
    ("AUTO_TRADE_ENTRY_MODE", "Режим входа"),
    ("AUTO_TRADE_TOP_N", "Топ N"),
    ("WALLET_BALANCE_USD", "Бумажный кошелек $"),
    ("PAPER_TRADE_SIZE_USD", "Базовый размер сделки $"),
    ("PAPER_TRADE_SIZE_MIN_USD", "Мин. размер сделки $"),
    ("PAPER_TRADE_SIZE_MAX_USD", "Макс. размер сделки $"),
    ("PAPER_MAX_HOLD_SECONDS", "Макс. удержание (сек)"),
    ("DYNAMIC_HOLD_ENABLED", "Динамический холд"),
    ("HOLD_MIN_SECONDS", "Холд мин (сек)"),
    ("HOLD_MAX_SECONDS", "Холд макс (сек)"),
    ("MAX_OPEN_TRADES", "Макс. открытых сделок"),
    ("CLOSED_TRADES_MAX_AGE_DAYS", "Хранить закрытые (дней)"),
    ("PAPER_REALISM_ENABLED", "Реалистичный режим"),
    ("PAPER_REALISM_CAP_ENABLED", "Realism cap"),
    ("PAPER_REALISM_MAX_GAIN_PERCENT", "Realism max gain %"),
    ("PAPER_REALISM_MAX_LOSS_PERCENT", "Realism max loss %"),
    ("PAPER_GAS_PER_TX_USD", "Газ за tx $"),
    ("PAPER_SWAP_FEE_BPS", "Комиссия свапа (bps)"),
    ("PAPER_BASE_SLIPPAGE_BPS", "Базовый slippage (bps)"),
    ("DYNAMIC_POSITION_SIZING_ENABLED", "Динамический размер"),
    ("EDGE_FILTER_ENABLED", "Фильтр edge"),
    ("MIN_EXPECTED_EDGE_PERCENT", "Мин. ожидаемый edge %"),
    ("PROFIT_TARGET_PERCENT", "Тейк-профит %"),
    ("STOP_LOSS_PERCENT", "Стоп-лосс %"),
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
    ("DEX_SEARCH_QUERIES", "Поиск Dex (через запятую)"),
    ("GECKO_NEW_POOLS_PAGES", "Страниц Gecko"),
    ("WETH_PRICE_FALLBACK_USD", "WETH fallback $"),
    ("STAIR_STEP_ENABLED", "Step protection"),
    ("STAIR_STEP_START_BALANCE_USD", "Step start floor $"),
    ("STAIR_STEP_SIZE_USD", "Step size $"),
    ("STAIR_STEP_TRADABLE_BUFFER_USD", "Step tradable buffer $"),
    ("AUTO_STOP_MIN_AVAILABLE_USD", "Stop min available $"),
]
SETTINGS_FIELDS = [(k, _fix_mojibake(v)) for (k, v) in SETTINGS_FIELDS_RAW]

FIELD_OPTIONS = {
    "PERSONAL_MODE": ["true", "false"],
    "AUTO_FILTER_ENABLED": ["true", "false"],
    "AUTO_TRADE_ENABLED": ["false", "true"],
    "AUTO_TRADE_PAPER": ["true", "false"],
    "AUTO_TRADE_ENTRY_MODE": ["single", "all", "top_n"],
    "AUTO_TRADE_TOP_N": ["3", "5", "10", "20", "50"],
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


def start_bot() -> tuple[bool, str]:
    if not os.path.exists(PYTHON_PATH):
        return False, f"Python not found: {PYTHON_PATH}"

    existing = sorted(set(list_main_local_pids()))
    if existing:
        keep_pid = existing[0]
        # Kill duplicated workers if any, keep one process alive.
        for pid in existing[1:]:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        with open(PID_FILE, "w", encoding="ascii") as f:
            f.write(str(keep_pid))
        if len(existing) > 1:
            return True, f"Already running main_local.py (PID {keep_pid}), duplicates removed: {len(existing) - 1}"
        return True, f"Already running main_local.py (PID {keep_pid})"

    pid = read_pid()
    if pid and is_running(pid):
        return True, f"Already running (PID {pid})"

    os.makedirs(LOG_DIR, exist_ok=True)
    out_file = open(OUT_LOG, "a", encoding="utf-8")
    err_file = open(ERR_LOG, "a", encoding="utf-8")

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        [PYTHON_PATH, "main_local.py"],
        cwd=PROJECT_ROOT,
        stdout=out_file,
        stderr=err_file,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )
    with open(PID_FILE, "w", encoding="ascii") as f:
        f.write(str(proc.pid))
    return True, f"Started main_local.py (PID {proc.pid})"


def stop_bot() -> tuple[bool, str]:
    pids = sorted(set(list_main_local_pids()))
    if not pids:
        pid = read_pid()
        if pid and is_running(pid):
            pids = [pid]
        else:
            try:
                os.remove(PID_FILE)
            except OSError:
                pass
            return True, "Already stopped"

    failed: list[int] = []
    for pid in pids:
        result = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        if result.returncode != 0 and is_running(pid):
            failed.append(pid)
    try:
        os.remove(PID_FILE)
    except OSError:
        pass
    if failed:
        return False, f"Failed to stop PID(s): {', '.join(str(pid) for pid in failed)}"
    return True, f"Stopped main_local.py instances: {', '.join(str(pid) for pid in pids)}"


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


def truncate_file(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


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


def parse_activity(lines: list[str]) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for line in lines:
        if any(noise in line for noise in NOISE_LOG_SNIPPETS):
            continue
        matched = "INFO"
        for key, (needle, _) in EVENT_KEYS.items():
            if needle in line:
                matched = key
                break
        if matched == "INFO" and not any(snippet in line for snippet in IMPORTANT_LOG_SNIPPETS):
            continue
        events.append((matched, line))
    return events


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
    if not os.path.exists(ENV_FILE):
        return []
    with open(ENV_FILE, "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def read_env_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in load_env_lines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def save_env_map(values: dict[str, str]) -> None:
    lines = load_env_lines()
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

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(updated).rstrip() + "\n")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Панель управления Base Bot")
        self.geometry("1180x760")
        self.minsize(1020, 680)
        self.configure(bg="#0b1220")

        self._configure_theme()
        self.env_vars: dict[str, tk.StringVar] = {}

        self.status_var = tk.StringVar(value="Статус: неизвестно")
        self.hint_var = tk.StringVar(value="Подсказка: сначала тестируй в paper-режиме, потом подключай live.")

        self._build_header()
        self._build_tabs()
        self._load_settings()

        self.after(200, self.refresh)
        self.after(2200, self.auto_refresh)
        self._live_wallet_eth = 0.0
        self._live_wallet_usd = 0.0
        self._live_wallet_last_ts = 0.0
        self._live_wallet_last_err = ""
        self._live_wallet_polling = False
        self.after(800, self._schedule_live_wallet_poll)

    def _configure_theme(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#0b1220", foreground="#dbeafe", fieldbackground="#0f172a")
        style.configure("Card.TFrame", background="#111827")
        style.configure("Header.TFrame", background="#0f172a")
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 16), foreground="#f8fafc", background="#0f172a")
        style.configure("Hint.TLabel", font=("Segoe UI", 9), foreground="#93c5fd", background="#0f172a")
        style.configure("Status.TLabel", font=("Segoe UI", 10), foreground="#86efac", background="#0f172a")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=8)
        style.configure("TEntry", padding=6)
        style.configure(
            "TCombobox",
            fieldbackground="#0f172a",
            background="#0f172a",
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
        style.configure("TNotebook", background="#0b1220", borderwidth=0)
        style.configure("TNotebook.Tab", background="#1f2937", foreground="#d1d5db", padding=(12, 8))
        style.map("TNotebook.Tab", background=[("selected", "#111827")], foreground=[("selected", "#f8fafc")])
        style.configure(
            "Treeview",
            background="#0f172a",
            fieldbackground="#0f172a",
            foreground="#e5e7eb",
            borderwidth=0,
            rowheight=26,
        )
        style.configure(
            "Treeview.Heading",
            background="#1f2937",
            foreground="#bfdbfe",
            borderwidth=0,
            font=("Segoe UI Semibold", 10),
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
        ttk.Label(left, text="Панель управления Base Bot", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, textvariable=self.hint_var, style="Hint.TLabel").pack(anchor="w", pady=(3, 0))
        ttk.Label(left, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(6, 0))

        right = ttk.Frame(hdr, style="Header.TFrame")
        right.pack(side=tk.RIGHT)
        ttk.Button(right, text="Старт", command=self.on_start).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Стоп", command=self.on_stop).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Сохранить", command=self._save_settings).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Сохранить + Рестарт", command=self._save_restart).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Рестарт", command=self.on_restart).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Перечитать .env", command=self.on_reload_env).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Очистить логи", command=self.on_clear_logs).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Обновить", command=self.refresh).pack(side=tk.LEFT, padx=4)

    def _build_tabs(self) -> None:
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        self.activity_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.signals_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.wallet_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.positions_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.settings_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.emergency_tab = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.tabs.add(self.activity_tab, text="Активность")
        self.tabs.add(self.signals_tab, text="Сигналы")
        self.tabs.add(self.wallet_tab, text="Кошелек")
        self.tabs.add(self.positions_tab, text="Сделки")
        self.tabs.add(self.settings_tab, text="Настройки")
        self.tabs.add(self.emergency_tab, text="Аварийный")

        self._build_activity_tab()
        self._build_signals_tab()
        self._build_wallet_tab()
        self._build_positions_tab()
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
        ttk.Label(left_card, text="Лента сигналов и сделок", font=("Segoe UI Semibold", 12)).pack(anchor="w", pady=(0, 8))
        self.feed_text = tk.Text(
            left_card,
            bg="#0b1220",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief=tk.FLAT,
            wrap="word",
            font=("Consolas", 10),
        )
        self.feed_text.pack(fill=tk.BOTH, expand=True)
        self._attach_text_copy_support(self.feed_text)
        self.feed_text.tag_configure("BUY", foreground=EVENT_KEYS["BUY"][1])
        self.feed_text.tag_configure("SELL", foreground=EVENT_KEYS["SELL"][1])
        self.feed_text.tag_configure("SCAN", foreground=EVENT_KEYS["SCAN"][1])
        self.feed_text.tag_configure("ALERT", foreground=EVENT_KEYS["ALERT"][1])
        self.feed_text.tag_configure("SKIP", foreground=EVENT_KEYS["SKIP"][1])
        self.feed_text.tag_configure("ERROR", foreground=EVENT_KEYS["ERROR"][1])
        self.feed_text.tag_configure("INFO", foreground="#cbd5e1")

        right_card = ttk.Frame(activity_top, style="Card.TFrame")
        right_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(right_card, text="Сырые runtime-логи", font=("Segoe UI Semibold", 12)).pack(anchor="w", pady=(0, 8))
        self.raw_text = tk.Text(
            right_card,
            bg="#0b1220",
            fg="#cbd5e1",
            insertbackground="#e5e7eb",
            relief=tk.FLAT,
            wrap="none",
            font=("Consolas", 9),
        )
        self.raw_text.pack(fill=tk.BOTH, expand=True)
        self._attach_text_copy_support(self.raw_text)

    def _build_signals_tab(self) -> None:
        top = ttk.Frame(self.signals_tab, style="Card.TFrame")
        top.pack(fill=tk.X, pady=(0, 10))
        self.signals_summary_var = tk.StringVar(value="Входящих сигналов пока нет.")
        ttk.Label(top, text="Входящие сигналы", font=("Segoe UI Semibold", 12)).pack(anchor="w")
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
            ("time", "Время", 110),
            ("symbol", "Символ", 90),
            ("score", "Score", 80),
            ("recommendation", "Рекомендация", 120),
            ("risk", "Риск", 90),
            ("liquidity", "Liquidity $", 120),
            ("volume", "Volume 5m $", 120),
            ("change", "Change 5m %", 110),
        ):
            self.signals_tree.heading(col, text=title)
            self.signals_tree.column(col, width=width, anchor="center")
        self.signals_tree.pack(fill=tk.BOTH, expand=True)

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

        title = ttk.Label(self.settings_content, text="Настройки runtime (.env)", font=("Segoe UI Semibold", 12))
        title.pack(anchor="w", pady=(0, 10))

        presets = ttk.Frame(self.settings_content, style="Card.TFrame")
        presets.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(presets, text="Быстрые пресеты:", foreground="#93c5fd").pack(side=tk.LEFT)
        ttk.Button(presets, text="Safe Flow", command=self._apply_safe_flow_preset).pack(side=tk.LEFT, padx=6)
        ttk.Button(presets, text="Medium Flow", command=self._apply_medium_flow_preset).pack(side=tk.LEFT, padx=6)
        ttk.Button(presets, text="Live Wallet", command=self._apply_live_wallet_preset).pack(side=tk.LEFT, padx=6)
        ttk.Button(presets, text="Ultra Safe Live", command=self._apply_ultra_safe_live_preset).pack(side=tk.LEFT, padx=6)
        ttk.Button(presets, text="Working Live", command=self._apply_working_live_preset).pack(side=tk.LEFT, padx=6)

        grid = ttk.Frame(self.settings_content, style="Card.TFrame")
        grid.pack(fill=tk.X)
        grid.columnconfigure(0, weight=1)

        for idx, (key, label) in enumerate(SETTINGS_FIELDS):
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
        ttk.Button(btns, text="Перечитать .env", command=self._load_settings).pack(side=tk.LEFT)
        ttk.Button(btns, text="Сохранить", command=self._save_settings).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Сохранить + Рестарт", command=self._save_restart).pack(side=tk.LEFT)

        tips = (
            "Подсказка:\n"
            "- Сначала используй paper-режим (AUTO_TRADE_PAPER=true).\n"
            "- Для маленького баланса: Безопасный или Сбалансированный (3).\n"
            "- Держи EDGE_FILTER_ENABLED=true, чтобы резать слабые входы.\n"
            "- После изменений делай рестарт бота."
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
            text="Critical actions moved to tab: Аварийный",
            foreground="#fca5a5",
            font=("Segoe UI", 10),
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(12, 2))

    def _build_emergency_tab(self) -> None:
        top = ttk.Frame(self.emergency_tab, style="Card.TFrame")
        top.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(top, text="Аварийный блок управления", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        self.emergency_summary_var = tk.StringVar(
            value="KILL_SWITCH: OFF | CRITICAL_AUTO_RESET: 0 | live_failed: 0 | errors: 0"
        )
        ttk.Label(top, textvariable=self.emergency_summary_var, foreground="#fca5a5").pack(anchor="w", pady=(4, 0))

        actions = ttk.Frame(top, style="Card.TFrame")
        actions.pack(anchor="w", pady=(8, 0))
        ttk.Button(actions, text="Включить KILL SWITCH", command=self.on_enable_kill_switch).pack(side=tk.LEFT)
        ttk.Button(actions, text="Выключить KILL SWITCH", command=self.on_disable_kill_switch).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Критический сброс", command=self.on_critical_reset).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Signal Check", command=self.on_signal_check).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Очистить логи", command=self.on_clear_logs).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Очистить GUI", command=self.on_clear_gui).pack(side=tk.LEFT, padx=8)

        self.critical_text = tk.Text(
            self.emergency_tab,
            bg="#0b1220",
            fg="#fecaca",
            insertbackground="#fecaca",
            relief=tk.FLAT,
            wrap="word",
            font=("Consolas", 10),
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
        self.pos_summary_var = tk.StringVar(value="Пока нет данных по сделкам.")
        ttk.Label(top, text="Монитор бумажных сделок", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        ttk.Label(top, textvariable=self.pos_summary_var, foreground="#93c5fd").pack(anchor="w", pady=(4, 0))
        actions = ttk.Frame(top, style="Card.TFrame")
        actions.pack(anchor="w", pady=(8, 0))
        ttk.Button(actions, text="Удалить старые закрытые", command=self.on_prune_closed_trades).pack(side=tk.LEFT)
        ttk.Button(actions, text="Очистить все закрытые", command=self.on_clear_closed_trades).pack(side=tk.LEFT, padx=8)

        panels = ttk.Frame(self.positions_tab, style="Card.TFrame")
        panels.pack(fill=tk.BOTH, expand=True)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(0, weight=1)

        left = ttk.Frame(panels, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(left, text="Открытые позиции", font=("Segoe UI Semibold", 11)).pack(anchor="w", pady=(0, 6))
        self.open_tree = ttk.Treeview(
            left,
            columns=("symbol", "entry", "size", "score", "ttl"),
            show="headings",
            height=14,
        )
        for col, title, width in (
            ("symbol", "Символ", 90),
            ("entry", "Вход $", 120),
            ("size", "Размер $", 90),
            ("score", "Скор", 80),
            ("ttl", "Осталось (сек)", 120),
        ):
            self.open_tree.heading(col, text=title)
            self.open_tree.column(col, width=width, anchor="center")
        self.open_tree.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(panels, style="Card.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(right, text="Закрытые позиции", font=("Segoe UI Semibold", 11)).pack(anchor="w", pady=(0, 6))
        self.closed_tree = ttk.Treeview(
            right,
            columns=("symbol", "reason", "pnl_pct", "pnl_usd"),
            show="headings",
            height=14,
        )
        for col, title, width in (
            ("symbol", "Символ", 110),
            ("reason", "Выход", 90),
            ("pnl_pct", "PnL %", 90),
            ("pnl_usd", "PnL $", 90),
        ):
            self.closed_tree.heading(col, text=title)
            self.closed_tree.column(col, width=width, anchor="center")
        self.closed_tree.pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def _to_float(raw: str | None, default: float) -> float:
        try:
            if raw is None:
                return default
            return float(str(raw).strip())
        except (TypeError, ValueError):
            return default

    def _effective_wallet_balance(self, preset: dict[str, str]) -> float:
        if "WALLET_BALANCE_USD" in preset:
            return max(0.1, self._to_float(preset.get("WALLET_BALANCE_USD"), 2.75))
        env_value = self.env_vars.get("WALLET_BALANCE_USD")
        if env_value and env_value.get().strip():
            return max(0.1, self._to_float(env_value.get(), 2.75))
        env_map = read_env_map()
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
        extra_env_values: dict[str, str] = {}
        for key, value in tuned.items():
            if key in self.env_vars:
                self.env_vars[key].set(value)
            else:
                extra_env_values[key] = value
        if extra_env_values:
            save_env_map(extra_env_values)
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
                "MAX_TRADES_PER_HOUR": "2",
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
                "MAX_TRADES_PER_HOUR": "4",
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
                "MAX_TRADES_PER_HOUR": "8",
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
                "MAX_TRADES_PER_HOUR": "4",
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
                "MAX_TRADES_PER_HOUR": "6",
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
                "MAX_TRADES_PER_HOUR": "5",
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
                "MAX_TRADES_PER_HOUR": "0",
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
        env_map = read_env_map()

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
            "MAX_TRADES_PER_HOUR": "3",
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
        save_env_map(preset)
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
        env_map = read_env_map()
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
            "MAX_TRADES_PER_HOUR": "3",
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

        save_env_map(preset)
        for k, v in preset.items():
            if k in self.env_vars:
                self.env_vars[k].set(v)
        if hasattr(self, "wallet_mode_var"):
            self.wallet_mode_var.set("live")
        self._refresh_wallet()
        if hasattr(self, "hint_var"):
            self.hint_var.set("Working Live applied. Safer core checks kept, entry flow widened moderately.")

    def _load_settings(self) -> None:
        env_map = read_env_map()
        for key, _ in SETTINGS_FIELDS:
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
            save_env_map(values)
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", str(exc))
            return
        messagebox.showinfo("Готово", ".env успешно обновлен.")

    def _save_restart(self) -> None:
        self._save_settings()
        self.on_restart()

    def refresh(self) -> None:
        pid = read_pid()
        if pid and is_running(pid):
            self.status_var.set(f"Статус: ЗАПУЩЕН (PID {pid})")
        else:
            self.status_var.set("Статус: ОСТАНОВЛЕН")

        feed_bottom = self._is_near_bottom(self.feed_text)
        feed_y = self.feed_text.yview()
        app_lines = read_tail(APP_LOG, 220)
        events = parse_activity(app_lines)
        if not self._text_has_active_selection(self.feed_text):
            self.feed_text.delete("1.0", tk.END)
            if not events:
                self.feed_text.insert(tk.END, "Пока нет активности. Запусти бота и дождись цикла сканирования.\n", ("INFO",))
            for level, line in events[-180:]:
                self.feed_text.insert(tk.END, line + "\n", (level,))
            self._restore_scroll(self.feed_text, feed_y, feed_bottom)

        raw_bottom = self._is_near_bottom(self.raw_text)
        raw_y = self.raw_text.yview()
        combined = []
        for label, path in (("APP", APP_LOG),):
            lines = compact_runtime_lines(read_tail(path, 180))
            combined.append(f"===== {label} LOG (clean) =====")
            combined.extend(lines[-120:] or ["<empty>"])
            combined.append("")
        if not self._text_has_active_selection(self.raw_text):
            self.raw_text.delete("1.0", tk.END)
            self.raw_text.insert(tk.END, "\n".join(combined))
            self._restore_scroll(self.raw_text, raw_y, raw_bottom)
        self._refresh_signals()
        self._refresh_wallet()
        self._refresh_positions(app_lines)
        self._refresh_emergency_tab(app_lines)

    @staticmethod
    def _kill_switch_path() -> str:
        env_map = read_env_map()
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

    def _refresh_positions(self, app_lines: list[str]) -> None:
        state = read_json(PAPER_STATE_FILE)
        if state:
            self._refresh_positions_from_state(state)
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
                open_map[address] = {
                    "symbol": b.group("symbol"),
                    "entry": b.group("entry"),
                    "size": b.group("size"),
                    "score": b.group("score"),
                }
                continue
            s = SELL_RE.search(line)
            if s:
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

        self.open_tree.delete(*self.open_tree.get_children())
        for _, pos in open_map.items():
            self.open_tree.insert("", tk.END, values=(pos["symbol"], pos["entry"], pos["size"], pos["score"], "--:--"))

        self.closed_tree.delete(*self.closed_tree.get_children())
        for pos in closed[-200:]:
            self.closed_tree.insert("", tk.END, values=(pos["symbol"], pos["reason"], pos["pnl_pct"], pos["pnl_usd"]))

        self.pos_summary_var.set(
            f"Открыто: {len(open_map)} | Закрыто: {len(closed)} | Побед/Поражений: {wins}/{losses} | Итог PnL: ${pnl_sum:+.2f}"
        )

    def _refresh_signals(self) -> None:
        if not hasattr(self, "signals_tree"):
            return
        rows = read_jsonl_tail(LOCAL_ALERTS_FILE, 400)
        self.signals_tree.delete(*self.signals_tree.get_children())
        if not rows:
            self.signals_summary_var.set("Входящих сигналов пока нет.")
            self._refresh_signal_sources()
            return

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
            self.signals_tree.insert(
                "",
                tk.END,
                values=(
                    ts_text,
                    str(row.get("symbol", "N/A")),
                    str(row.get("score", 0)),
                    str(row.get("recommendation", "SKIP")),
                    str(row.get("risk_level", "N/A")),
                    f"{float(row.get('liquidity', 0) or 0):.0f}",
                    f"{float(row.get('volume_5m', 0) or 0):.0f}",
                    f"{float(row.get('price_change_5m', 0) or 0):+.2f}",
                ),
            )
        self.signals_summary_var.set(f"Сигналов: {len(rows)} | Показано: {min(len(rows), 250)}")
        self._refresh_signal_sources()

    def _refresh_signal_sources(self) -> None:
        if not hasattr(self, "signals_sources_var"):
            return
        lines = read_tail(APP_LOG, 500)
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
            f"Sources (tail): v2={v2_count} | v3={v3_count} | other={other_count} | total={total}"
        )

    def _refresh_positions_from_state(self, state: dict) -> None:
        open_rows = state.get("open_positions", []) or []
        closed_rows = state.get("closed_positions", []) or []

        wins = int(state.get("total_wins", 0))
        losses = int(state.get("total_losses", 0))
        realized_pnl = float(state.get("realized_pnl_usd", 0.0))

        env_map = read_env_map()
        fallback_hold = int(env_map.get("PAPER_MAX_HOLD_SECONDS", "1800") or "1800")
        now = datetime.now(timezone.utc)

        self.open_tree.delete(*self.open_tree.get_children())
        for row in open_rows:
            opened_at_raw = str(row.get("opened_at", ""))
            ttl = "--"
            try:
                opened_at = datetime.fromisoformat(opened_at_raw)
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
                max_hold = int(row.get("max_hold_seconds", fallback_hold))
                age = int((now - opened_at).total_seconds())
                left = max(0, max_hold - age)
                ttl = f"{left} сек"
            except Exception:
                ttl = "--"

            self.open_tree.insert(
                "",
                tk.END,
                values=(
                    str(row.get("symbol", "N/A")),
                    f"{float(row.get('entry_price_usd', 0)):.8f}",
                    f"{float(row.get('position_size_usd', 0)):.2f}",
                    str(row.get("score", 0)),
                    ttl,
                ),
            )

        self.closed_tree.delete(*self.closed_tree.get_children())
        for row in closed_rows[-200:]:
            self.closed_tree.insert(
                "",
                tk.END,
                values=(
                    str(row.get("symbol", "N/A")),
                    str(row.get("close_reason", "")),
                    f"{float(row.get('pnl_percent', 0.0)):+.2f}",
                    f"{float(row.get('pnl_usd', 0.0)):+.2f}",
                ),
            )

        self.pos_summary_var.set(
            f"Открыто: {len(open_rows)} | Закрыто: {len(closed_rows)} | Побед/Поражений: {wins}/{losses} | Реализованный PnL: ${realized_pnl:+.2f}"
        )

    def _schedule_live_wallet_poll(self) -> None:
        # Keep UI responsive: RPC calls happen in a background thread.
        if getattr(self, "_live_wallet_polling", False):
            self.after(int(LIVE_BALANCE_POLL_SECONDS * 1000), self._schedule_live_wallet_poll)
            return

        env_map = read_env_map()
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
        env_map = read_env_map()
        mode = str(env_map.get(WALLET_MODE_KEY, self.wallet_mode_var.get() if hasattr(self, "wallet_mode_var") else "paper")).strip().lower()
        live_balance = float(env_map.get(LIVE_WALLET_BALANCE_KEY, "0") or 0)

        state = read_json(PAPER_STATE_FILE)
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
            messagebox.showerror("Ошибка", "Режим кошелька должен быть paper или live.")
            return
        save_env_map({WALLET_MODE_KEY: mode})
        self._refresh_wallet()
        if mode == "live":
            messagebox.showinfo(
                "Live mode",
                "Внимание: live execution пока не подключен. Сейчас это режим отображения баланса.",
            )

    def on_apply_stair_mode(self) -> None:
        mode = str(self.stair_enabled_var.get()).strip().lower()
        if mode not in {"true", "false"}:
            messagebox.showerror("Ошибка", "Step protection должен быть true или false.")
            return
        save_env_map({STAIR_STEP_ENABLED_KEY: mode})
        if STAIR_STEP_ENABLED_KEY in self.env_vars:
            self.env_vars[STAIR_STEP_ENABLED_KEY].set(mode)
        self._refresh_wallet()

    def on_apply_stair_params(self) -> None:
        try:
            start_floor = float(str(self.stair_start_var.get()).strip())
            step_size = float(str(self.stair_size_var.get()).strip())
        except ValueError:
            messagebox.showerror("Ошибка", "Step start/size должны быть числами.")
            return
        if start_floor < 0:
            messagebox.showerror("Ошибка", "Step start floor не может быть отрицательным.")
            return
        if step_size <= 0:
            messagebox.showerror("Ошибка", "Step size должен быть больше 0.")
            return
        save_env_map(
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
            messagebox.showerror("Ошибка", "Live balance должен быть числом.")
            return
        if val < 0:
            messagebox.showerror("Ошибка", "Live balance не может быть отрицательным.")
            return
        save_env_map({LIVE_WALLET_BALANCE_KEY: f"{val:.2f}"})
        self._refresh_wallet()

    def on_set_paper_275(self) -> None:
        self.paper_balance_set_var.set("2.75")
        self.on_apply_paper_balance()

    def on_apply_paper_balance(self) -> None:
        try:
            val = float(self.paper_balance_set_var.get().strip())
        except ValueError:
            messagebox.showerror("Ошибка", "Paper balance должен быть числом.")
            return
        if val < 0:
            messagebox.showerror("Ошибка", "Paper balance не может быть отрицательным.")
            return

        state = read_json(PAPER_STATE_FILE)
        open_positions = state.get("open_positions", []) or []
        if open_positions:
            if not messagebox.askyesno(
                "Open trades detected",
                "Open paper trades exist in paper_state.json.\n\n"
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
        with open(PAPER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        save_env_map({"WALLET_BALANCE_USD": f"{val:.2f}"})
        if "WALLET_BALANCE_USD" in self.env_vars:
            self.env_vars["WALLET_BALANCE_USD"].set(f"{val:.2f}")
        self._refresh_wallet()

    def on_critical_reset(self) -> None:
        if not messagebox.askyesno(
            "Критический сброс",
            "Сбросить paper-состояние полностью? Открытые и закрытые сделки будут удалены.",
        ):
            return

        pid = read_pid()
        if pid and is_running(pid):
            ok, msg = stop_bot()
            if not ok:
                messagebox.showerror("Ошибка", f"Не удалось остановить бота перед сбросом: {msg}")
                return

        env_map = read_env_map()
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
        os.makedirs(os.path.dirname(PAPER_STATE_FILE), exist_ok=True)
        with open(PAPER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        save_env_map({"WALLET_BALANCE_USD": f"{balance:.2f}"})
        if "WALLET_BALANCE_USD" in self.env_vars:
            self.env_vars["WALLET_BALANCE_USD"].set(f"{balance:.2f}")
        self._refresh_wallet()
        self._refresh_positions([])
        messagebox.showinfo("Готово", f"Paper-состояние сброшено. Текущий баланс: ${balance:.2f}")

    def auto_refresh(self) -> None:
        self.refresh()
        self.after(2200, self.auto_refresh)

    def on_start(self) -> None:
        ok, msg = start_bot()
        if not ok:
            messagebox.showerror("Ошибка запуска", msg)
        self.refresh()

    def on_stop(self) -> None:
        ok, msg = stop_bot()
        if not ok:
            messagebox.showerror("Ошибка остановки", msg)
        self.refresh()

    def on_restart(self) -> None:
        stop_bot()
        ok, msg = start_bot()
        if not ok:
            messagebox.showerror("Ошибка рестарта", msg)
        self.refresh()

    def on_reload_env(self) -> None:
        # GUI keeps values in memory; restarting the bot does not refresh input fields.
        # This explicitly reloads `.env` into the GUI controls.
        try:
            self._load_settings()
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось перечитать .env: {exc}")
            return
        self.refresh()
        messagebox.showinfo("Готово", "Настройки перечитаны из .env.")

    def on_clear_logs(self) -> None:
        for path in (APP_LOG, OUT_LOG, ERR_LOG):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            truncate_file(path)
        self.refresh()

    def on_clear_gui(self) -> None:
        # UI-only cleanup: clears views and truncates log files. Does not touch wallet state or KILL SWITCH.
        self.on_clear_logs()
        self.on_clear_signals()
        for attr in ("log_text", "critical_text"):
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
            self.pos_summary_var.set("Очистка выполнена.")
        if hasattr(self, "hint_var"):
            self.hint_var.set("GUI очищен: логи/сигналы/таблицы. KILL SWITCH не тронут.")

    def on_clear_signals(self) -> None:
        os.makedirs(os.path.dirname(LOCAL_ALERTS_FILE), exist_ok=True)
        truncate_file(LOCAL_ALERTS_FILE)
        if hasattr(self, "signals_tree"):
            self.signals_tree.delete(*self.signals_tree.get_children())
        if hasattr(self, "signals_summary_var"):
            self.signals_summary_var.set("\u0412\u0445\u043e\u0434\u044f\u0449\u0438\u0445 \u0441\u0438\u0433\u043d\u0430\u043b\u043e\u0432 \u043f\u043e\u043a\u0430 \u043d\u0435\u0442.")

    def on_signal_check(self) -> None:
        env_map = read_env_map()
        signal_source = str(env_map.get("SIGNAL_SOURCE", "dexscreener") or "dexscreener").strip().lower()

        rows = read_jsonl_tail(LOCAL_ALERTS_FILE, 400)
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

        app_lines = read_tail(APP_LOG, 260)
        scan_count = sum(1 for line in app_lines if "Scanned " in line)
        pair_detected = sum(1 for line in app_lines if "PAIR_DETECTED" in line)
        filter_pass = sum(1 for line in app_lines if "FILTER_PASS" in line)
        auto_buy = sum(1 for line in app_lines if "AUTO_BUY" in line)
        auto_sell = sum(1 for line in app_lines if "AUTO_SELL" in line)
        critical_events = sum(1 for line in app_lines if "CRITICAL_AUTO_RESET" in line)
        kill_events = sum(1 for line in app_lines if "KILL_SWITCH" in line)

        state = read_json(PAPER_STATE_FILE)
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
            f"signals={len(rows)} last={last_alert} scans={scan_count} "
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
        env_map = read_env_map()
        days_raw = env_map.get("CLOSED_TRADES_MAX_AGE_DAYS", "14").strip() or "14"
        try:
            days = max(0, int(days_raw))
        except ValueError:
            days = 14
        removed = self._cleanup_closed_trades(days=days, remove_all=False)
        messagebox.showinfo("Готово", f"Удалено закрытых сделок: {removed}")
        self.refresh()

    def on_clear_closed_trades(self) -> None:
        removed = self._cleanup_closed_trades(days=0, remove_all=True)
        messagebox.showinfo("Готово", f"Удалено закрытых сделок: {removed}")
        self.refresh()

    @staticmethod
    def _cleanup_closed_trades(days: int, remove_all: bool) -> int:
        state = read_json(PAPER_STATE_FILE)
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

        with open(PAPER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
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

