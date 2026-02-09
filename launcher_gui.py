"""Dark control panel for bot process, activity feed, and settings."""

from __future__ import annotations

import os
import re
import subprocess
import tkinter as tk
from datetime import datetime, timezone
import json
from tkinter import messagebox, ttk

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

BUY_RE = re.compile(
    r"Paper BUY token=(?P<symbol>\S+) address=(?P<address>\S+) entry=\$(?P<entry>[0-9.]+) size=\$(?P<size>[0-9.]+) score=(?P<score>\d+)"
)
SELL_RE = re.compile(
    r"Paper SELL token=(?P<symbol>\S+) reason=(?P<reason>\S+) exit=\$(?P<exit>[0-9.]+) pnl=(?P<pnl_pct>[-+0-9.]+)% \(\$(?P<pnl_usd>[-+0-9.]+)\)(?: raw=[-+0-9.]+% cost=[-+0-9.]+% gas=\$[-+0-9.]+)? balance=\$(?P<balance>[0-9.]+)"
)

SETTINGS_FIELDS = [
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
]

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

    existing = list_main_local_pids()
    if existing:
        with open(PID_FILE, "w", encoding="ascii") as f:
            f.write(str(existing[0]))
        return True, f"Already running main_local.py (PID {existing[0]})"

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
        self.tabs.add(self.activity_tab, text="Активность")
        self.tabs.add(self.signals_tab, text="Сигналы")
        self.tabs.add(self.wallet_tab, text="Кошелек")
        self.tabs.add(self.positions_tab, text="Сделки")
        self.tabs.add(self.settings_tab, text="Настройки")

        self._build_activity_tab()
        self._build_signals_tab()
        self._build_wallet_tab()
        self._build_positions_tab()
        self._build_settings_tab()

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

    def _build_signals_tab(self) -> None:
        top = ttk.Frame(self.signals_tab, style="Card.TFrame")
        top.pack(fill=tk.X, pady=(0, 10))
        self.signals_summary_var = tk.StringVar(value="Входящих сигналов пока нет.")
        ttk.Label(top, text="Входящие сигналы", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        ttk.Label(top, textvariable=self.signals_summary_var, foreground="#93c5fd").pack(anchor="w", pady=(4, 0))
        actions = ttk.Frame(top, style="Card.TFrame")
        actions.pack(anchor="w", pady=(8, 0))
        ttk.Button(actions, text="\u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c \u0441\u0438\u0433\u043d\u0430\u043b\u044b", command=self.on_clear_signals).pack(side=tk.LEFT)

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
        ttk.Button(presets, text="\u0421\u0443\u043f\u0435\u0440 \u0431\u0435\u0437\u043e\u043f\u0430\u0441\u043d\u044b\u0439", command=self._apply_super_safe_preset).pack(side=tk.LEFT, padx=6)
        ttk.Button(presets, text="Medium", command=self._apply_medium_preset).pack(side=tk.LEFT, padx=6)
        ttk.Button(presets, text="Hard", command=self._apply_hard_preset).pack(side=tk.LEFT)
        ttk.Button(presets, text="Hard Lite", command=self._apply_hard_lite_preset).pack(side=tk.LEFT, padx=6)

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

        ttk.Button(grid, text="Set paper 2.75", command=self.on_set_paper_275).grid(row=3, column=1, sticky="w", pady=(10, 2))
        ttk.Button(grid, text="Refresh wallet", command=self._refresh_wallet).grid(row=3, column=2, sticky="w", padx=8, pady=(10, 2))
        ttk.Button(grid, text="Критический сброс", command=self.on_critical_reset).grid(
            row=4, column=1, sticky="w", pady=(12, 2)
        )

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

    def _apply_preset_map(self, preset: dict[str, str]) -> None:
        for key, value in preset.items():
            if key in self.env_vars:
                self.env_vars[key].set(value)

    def _apply_super_safe_preset(self) -> None:
        self._apply_preset_map(
            {
                "AUTO_TRADE_ENABLED": "true",
                "AUTO_TRADE_PAPER": "true",
                "AUTO_FILTER_ENABLED": "true",
                "AUTO_TRADE_ENTRY_MODE": "single",
                "AUTO_TRADE_TOP_N": "3",
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
                "DEX_BOOSTS_SOURCE_ENABLED": "true",
                "DEX_BOOSTS_MAX_TOKENS": "10",
                "DEX_SEARCH_QUERIES": "base",
                "GECKO_NEW_POOLS_PAGES": "1",
            }
        )

    def _apply_medium_preset(self) -> None:
        self._apply_preset_map(
            {
                "AUTO_TRADE_ENABLED": "true",
                "AUTO_TRADE_PAPER": "true",
                "AUTO_FILTER_ENABLED": "true",
                "AUTO_TRADE_ENTRY_MODE": "top_n",
                "AUTO_TRADE_TOP_N": "5",
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
                "DEX_BOOSTS_SOURCE_ENABLED": "true",
                "DEX_BOOSTS_MAX_TOKENS": "20",
                "DEX_SEARCH_QUERIES": "base,new",
                "GECKO_NEW_POOLS_PAGES": "2",
            }
        )

    def _apply_hard_preset(self) -> None:
        self._apply_preset_map(
            {
                "AUTO_TRADE_ENABLED": "true",
                "AUTO_TRADE_PAPER": "true",
                "AUTO_FILTER_ENABLED": "false",
                "AUTO_TRADE_ENTRY_MODE": "all",
                "AUTO_TRADE_TOP_N": "10",
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
                "DEX_BOOSTS_SOURCE_ENABLED": "true",
                "DEX_BOOSTS_MAX_TOKENS": "30",
                "DEX_SEARCH_QUERIES": "base,new,meme",
                "GECKO_NEW_POOLS_PAGES": "3",
            }
        )

    def _apply_hard_lite_preset(self) -> None:
        self._apply_preset_map(
            {
                "AUTO_TRADE_ENABLED": "true",
                "AUTO_TRADE_PAPER": "true",
                "AUTO_FILTER_ENABLED": "false",
                "AUTO_TRADE_ENTRY_MODE": "single",
                "AUTO_TRADE_TOP_N": "3",
                "MIN_TOKEN_SCORE": "65",
                "SAFE_TEST_MODE": "false",
                "SAFE_REQUIRE_CONTRACT_SAFE": "true",
                "SAFE_REQUIRE_RISK_LEVEL": "MEDIUM",
                "SAFE_MAX_WARNING_FLAGS": "2",
                "MAX_OPEN_TRADES": "1",
                "WALLET_BALANCE_USD": "2.75",
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
                "DEX_BOOSTS_SOURCE_ENABLED": "true",
                "DEX_BOOSTS_MAX_TOKENS": "25",
                "DEX_SEARCH_QUERIES": "base,new,meme",
                "GECKO_NEW_POOLS_PAGES": "3",
            }
        )

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
        self.raw_text.delete("1.0", tk.END)
        self.raw_text.insert(tk.END, "\n".join(combined))
        self._restore_scroll(self.raw_text, raw_y, raw_bottom)
        self._refresh_signals()
        self._refresh_wallet()
        self._refresh_positions(app_lines)

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

    def _refresh_wallet(self) -> None:
        env_map = read_env_map()
        mode = str(env_map.get(WALLET_MODE_KEY, self.wallet_mode_var.get() if hasattr(self, "wallet_mode_var") else "paper")).strip().lower()
        live_balance = float(env_map.get(LIVE_WALLET_BALANCE_KEY, "0") or 0)

        state = read_json(PAPER_STATE_FILE)
        paper_balance = float(state.get("paper_balance_usd", env_map.get("WALLET_BALANCE_USD", "0")) or 0)
        paper_open = len(state.get("open_positions", []) or [])
        paper_closed = len(state.get("closed_positions", []) or [])

        active = paper_balance if mode == "paper" else live_balance
        mode_label = "PAPER" if mode == "paper" else "LIVE"
        if hasattr(self, "wallet_summary_var"):
            self.wallet_summary_var.set(
                f"Mode: {mode_label} | Active balance: ${active:.2f} | Paper: ${paper_balance:.2f} | Live: ${live_balance:.2f} | Open: {paper_open} | Closed: {paper_closed}"
            )
        if hasattr(self, "wallet_mode_var"):
            self.wallet_mode_var.set(mode if mode in {"paper", "live"} else "paper")
        if hasattr(self, "live_balance_var"):
            self.live_balance_var.set(f"{live_balance:.2f}")
        if hasattr(self, "paper_balance_set_var"):
            self.paper_balance_set_var.set(f"{paper_balance:.2f}")

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
            messagebox.showerror("Ошибка", "Нельзя менять paper balance при открытых сделках.")
            return

        state["initial_balance_usd"] = float(val)
        state["paper_balance_usd"] = float(val)
        state["realized_pnl_usd"] = 0.0
        with open(PAPER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        save_env_map({"WALLET_BALANCE_USD": f"{val:.2f}"})
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
            "open_positions": [],
            "closed_positions": [],
        }
        os.makedirs(os.path.dirname(PAPER_STATE_FILE), exist_ok=True)
        with open(PAPER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        save_env_map({"WALLET_BALANCE_USD": f"{balance:.2f}"})
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

    def on_clear_logs(self) -> None:
        for path in (APP_LOG, OUT_LOG, ERR_LOG):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            truncate_file(path)
        self.refresh()

    def on_clear_signals(self) -> None:
        os.makedirs(os.path.dirname(LOCAL_ALERTS_FILE), exist_ok=True)
        truncate_file(LOCAL_ALERTS_FILE)
        if hasattr(self, "signals_tree"):
            self.signals_tree.delete(*self.signals_tree.get_children())
        if hasattr(self, "signals_summary_var"):
            self.signals_summary_var.set("\u0412\u0445\u043e\u0434\u044f\u0449\u0438\u0445 \u0441\u0438\u0433\u043d\u0430\u043b\u043e\u0432 \u043f\u043e\u043a\u0430 \u043d\u0435\u0442.")

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
    App().mainloop()
