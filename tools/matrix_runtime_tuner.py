from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from matrix_preset_guard import load_contract, validate_overrides


SUMMARY_RE = re.compile(r"Scanned\s+(\d+)\s+tokens.*Trade candidates:\s+(\d+).*Opened:\s+(\d+)")
FILTER_FAIL_RE = re.compile(r"FILTER_FAIL .*reason=([^\s|]+)")
AUTOTRADE_SKIP_RE = re.compile(r"AutoTrade skip token=.* reason=([^\s]+)")
AUTOTRADE_SKIP_DETAIL_RE = re.compile(r"AutoTrade skip token=.* reason=([^\s]+)(?:\s+detail=([^\r\n]+))?")
AUTOTRADE_BATCH_RE = re.compile(
    r"AutoTrade batch .*selected=(\d+)\s+opened=(\d+)\s+pe=([^\s]+)"
)
AUTO_BUY_RE = re.compile(r"AUTO_BUY .*token=([^\s]+)")
SYMBOL_CONCENTRATION_DROP_RE = re.compile(r"symbol_concentration['\"]?\s*:\s*(\d+)")
PAPER_SUMMARY_RE = re.compile(
    r"PAPER_SUMMARY .*total_closed=(\d+)\s+winrate_total=([0-9.]+)%\s+realized_total=\$(-?[0-9.]+).*open=(\d+)"
)


MODE_CHOICES = ("conveyor", "sniper", "calm", "fast")

SOURCE_BLOCK_REASONS = {"safe_source", "watchlist_strict_guard", "source_disabled"}
PRECHECK_BLOCK_REASONS = {
    "safe_risky_flags",
    "blacklist",
    "honeypot_guard",
    "safe_age",
    "safe_liquidity",
    "safe_volume",
    "safe_change_5m",
}
THRESHOLD_BLOCK_REASONS = {
    "score_min",
    "safety_budget",
    "excluded_base_token",
    "heavy_dedup_ttl",
}
QUARANTINE_BLOCK_REASONS = {
    "quarantine",
    "quarantine_unstable",
    "quarantine_mismatch",
    "symbol_concentration",
    "source_budget",
}
EXEC_ROUTE_FAIL_REASONS = {"no_route", "unsupported_live_route", "route_unstable", "sell_fail", "buy_fail"}
SOURCE_CAP_ORDER = ("onchain", "onchain+market", "dexscreener", "geckoterminal", "watchlist", "dex_boosts")
CHURN_CLOSE_REASONS = {
    "NO_MOMENTUM",
    "TIMEOUT",
    "WEAK_EARLY",
    "EARLY_RISK",
    "POST_PARTIAL_PI",
    "POST_PARTIAL_PROFIT",
}
CHURN_MIN_OPENS = 3
CHURN_MIN_CLOSES = 2
CHURN_MIN_OPEN_SHARE = 0.65
CHURN_MIN_FLAT_CLOSE_SHARE = 0.60
CHURN_FLAT_PNL_ABS_USD = 0.0015
TUNER_RUNTIME_FALLBACK_KEYS = {
    "MARKET_MODE_STRICT_SCORE",
    "MARKET_MODE_SOFT_SCORE",
    "SAFE_MIN_VOLUME_5M_USD",
    "MIN_EXPECTED_EDGE_PERCENT",
    "MIN_EXPECTED_EDGE_USD",
    "V2_ROLLING_EDGE_MIN_PERCENT",
    "V2_ROLLING_EDGE_MIN_USD",
    "V2_CALIBRATION_ENABLED",
    "V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW",
    "V2_CALIBRATION_EDGE_USD_MIN",
    "V2_CALIBRATION_VOLUME_MIN",
    "EV_FIRST_ENTRY_MIN_NET_USD",
    "EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD",
    "MAX_TOKEN_COOLDOWN_SECONDS",
    "MIN_TRADE_USD",
    "PAPER_TRADE_SIZE_MIN_USD",
    "PAPER_TRADE_SIZE_MAX_USD",
    "ENTRY_A_CORE_MIN_TRADE_USD",
    "AUTO_TRADE_TOP_N",
    "AUTO_TRADE_EXCLUDED_SYMBOLS",
    "HEAVY_CHECK_DEDUP_TTL_SECONDS",
    "PLAN_MAX_SINGLE_SOURCE_SHARE",
    "PLAN_MAX_WATCHLIST_SHARE",
    "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
    "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
    "V2_SOURCE_QOS_TOPK_PER_CYCLE",
    "V2_SOURCE_QOS_SOURCE_CAPS",
    "V2_UNIVERSE_SOURCE_CAPS",
    "SOURCE_ROUTER_MIN_TRADES",
    "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY",
    "SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY",
    "TOKEN_EV_MEMORY_MIN_TRADES",
    "TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY",
    "TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY",
    "TOKEN_AGE_MAX",
    "SEEN_TOKEN_TTL",
    "WATCHLIST_REFRESH_SECONDS",
    "WATCHLIST_MAX_TOKENS",
    "WATCHLIST_MIN_LIQUIDITY_USD",
    "WATCHLIST_MIN_VOLUME_24H_USD",
    "PAPER_WATCHLIST_MIN_SCORE",
    "PAPER_WATCHLIST_MIN_LIQUIDITY_USD",
    "PAPER_WATCHLIST_MIN_VOLUME_5M_USD",
    "V2_QUALITY_SOURCE_BUDGET_ENABLED",
}
TUNER_MUTABLE_KEYS = {
    "AUTO_TRADE_TOP_N",
    "AUTO_TRADE_EXCLUDED_SYMBOLS",
    "ENTRY_A_CORE_MIN_TRADE_USD",
    "EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD",
    "EV_FIRST_ENTRY_MIN_NET_USD",
    "HEAVY_CHECK_DEDUP_TTL_SECONDS",
    "MARKET_MODE_SOFT_SCORE",
    "MARKET_MODE_STRICT_SCORE",
    "MAX_TOKEN_COOLDOWN_SECONDS",
    "MIN_EXPECTED_EDGE_PERCENT",
    "MIN_EXPECTED_EDGE_USD",
    "V2_ROLLING_EDGE_MIN_PERCENT",
    "V2_ROLLING_EDGE_MIN_USD",
    "V2_CALIBRATION_ENABLED",
    "V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW",
    "V2_CALIBRATION_EDGE_USD_MIN",
    "V2_CALIBRATION_VOLUME_MIN",
    "MIN_TRADE_USD",
    "PAPER_TRADE_SIZE_MAX_USD",
    "PAPER_TRADE_SIZE_MIN_USD",
    "PLAN_MAX_SINGLE_SOURCE_SHARE",
    "PLAN_MAX_WATCHLIST_SHARE",
    "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
    "PROFIT_ENGINE_ENABLED",
    "SAFE_MIN_VOLUME_5M_USD",
    "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY",
    "SOURCE_ROUTER_MIN_TRADES",
    "SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY",
    "SYMBOL_CONCENTRATION_MAX_SHARE",
    "TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY",
    "TOKEN_EV_MEMORY_MIN_TRADES",
    "TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY",
    "TOKEN_AGE_MAX",
    "SEEN_TOKEN_TTL",
    "WATCHLIST_REFRESH_SECONDS",
    "WATCHLIST_MAX_TOKENS",
    "WATCHLIST_MIN_LIQUIDITY_USD",
    "WATCHLIST_MIN_VOLUME_24H_USD",
    "PAPER_WATCHLIST_MIN_SCORE",
    "PAPER_WATCHLIST_MIN_LIQUIDITY_USD",
    "PAPER_WATCHLIST_MIN_VOLUME_5M_USD",
    "V2_QUALITY_SOURCE_BUDGET_ENABLED",
    "V2_QUALITY_SYMBOL_MAX_SHARE",
    "V2_QUALITY_SYMBOL_MIN_ABS_CAP",
    "V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS",
    "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
    "V2_SOURCE_QOS_TOPK_PER_CYCLE",
    "V2_SOURCE_QOS_SOURCE_CAPS",
    "V2_UNIVERSE_NOVELTY_MIN_SHARE",
    "V2_UNIVERSE_SOURCE_CAPS",
}
TUNER_EPHEMERAL_KEYS = {"AUTO_TRADE_EXCLUDED_SYMBOLS"}
# Keys that can be hot-applied into a running `main_local.py` without process restart.
# They are still persisted into preset/env; runtime patch just removes restart dependency.
TUNER_HOT_APPLY_KEYS = set(TUNER_MUTABLE_KEYS)
# Explicitly restart-required keys (kept empty by default, can be extended later safely).
TUNER_RESTART_REQUIRED_KEYS: set[str] = set()
FORCE_ACTION_REASONS = {"soft_must_be_leq_strict"}
TIGHTEN_FLOW_ESCAPE_SIGNALS = {
    "source_qos_cap_rebalance",
    "duplicate_choke",
    "quality_duplicate_choke",
    "feed_starvation",
    "route_floor_normalize",
}
PHASE_CHOICES = ("expand", "hold", "tighten")
STATE_MACHINE_CHOICES = ("auto",) + PHASE_CHOICES
PROCESS_PROBE_TIMEOUT_SECONDS = 3
LOCK_STALE_SECONDS = 900
LOCK_STALE_TERMINATE_TIMEOUT_SECONDS = 8
DEAD_PROFILE_HEARTBEAT_MAX_AGE_SECONDS = 90.0


@dataclass
class TargetPolicy:
    target_trades_per_hour: float = 12.0
    target_pnl_per_hour_usd: float = 0.05
    min_open_rate_15m: float = 0.04
    min_selected_15m: int = 16
    min_closed_for_risk_checks: int = 6
    min_winrate_closed_15m: float = 0.35
    max_blacklist_share_15m: float = 0.45
    max_blacklist_added_15m: int = 80
    rollback_degrade_streak: int = 3
    hold_hysteresis_open_rate: float = 0.07
    hold_hysteresis_trades_per_hour: float = 6.0
    pre_risk_min_plan_attempts_15m: int = 8
    pre_risk_route_fail_rate_15m: float = 0.35
    pre_risk_buy_fail_rate_15m: float = 0.35
    pre_risk_sell_fail_rate_15m: float = 0.30
    pre_risk_roundtrip_loss_median_pct_15m: float = -1.2
    tail_loss_min_closes_60m: int = 6
    tail_loss_ratio_max: float = 8.0
    diversity_min_buys_15m: int = 4
    diversity_min_unique_symbols_15m: int = 2
    diversity_max_top1_open_share_15m: float = 0.72
    adaptive_target_enabled: bool = True
    adaptive_target_floor_trades_per_hour: float = 4.0
    adaptive_target_step_up_trades_per_hour: float = 1.5
    adaptive_target_step_down_trades_per_hour: float = 3.0
    adaptive_target_headroom_mult: float = 1.35
    adaptive_target_headroom_add_trades_per_hour: float = 2.0
    adaptive_target_stable_ticks_for_step_up: int = 2
    adaptive_target_fail_ticks_for_step_down: int = 2


@dataclass
class PolicyDecision:
    phase: str
    reasons: list[str]
    target_trades_per_hour_effective: float
    target_trades_per_hour_requested: float
    throughput_est_trades_h: float
    pnl_hour_usd: float
    blacklist_added_15m: int
    blacklist_share_15m: float
    open_rate_15m: float
    risk_fail: bool
    flow_fail: bool
    blacklist_fail: bool
    pre_risk_fail: bool = False
    diversity_fail: bool = False


@dataclass
class RuntimeState:
    degrade_streak: int = 0
    stable_hash: str = ""
    stable_overrides: dict[str, str] = field(default_factory=dict)
    last_phase: str = "expand"
    last_effective_hash: str = ""
    last_effective_at: str = ""
    churn_lock_symbol: str = ""
    churn_lock_until_ts: float = 0.0
    churn_lock_last_reason: str = ""
    effective_target_trades_per_hour: float = 0.0
    effective_target_stable_ticks: int = 0
    effective_target_fail_ticks: int = 0
    effective_target_last_reason: str = ""
    restart_history_ts: list[float] = field(default_factory=list)



@dataclass
class ModeSpec:
    name: str
    max_actions_per_tick: int
    strict_floor: int
    soft_floor: int
    strict_ceiling: int
    soft_ceiling: int
    volume_floor: float
    volume_ceiling: float
    edge_floor: float
    edge_ceiling: float
    edge_usd_floor: float
    edge_usd_ceiling: float
    cooldown_floor: int
    cooldown_ceiling: int


MODE_SPECS: dict[str, ModeSpec] = {
    "fast": ModeSpec(
        name="fast",
        max_actions_per_tick=7,
        strict_floor=45,
        soft_floor=40,
        strict_ceiling=70,
        soft_ceiling=65,
        volume_floor=8.0,
        volume_ceiling=160.0,
        edge_floor=0.05,
        edge_ceiling=1.20,
        edge_usd_floor=0.0005,
        edge_usd_ceiling=0.020,
        cooldown_floor=10,
        cooldown_ceiling=600,
    ),
    "conveyor": ModeSpec(
        name="conveyor",
        max_actions_per_tick=6,
        strict_floor=45,
        soft_floor=40,
        strict_ceiling=75,
        soft_ceiling=70,
        volume_floor=8.0,
        volume_ceiling=200.0,
        edge_floor=0.20,
        edge_ceiling=1.40,
        edge_usd_floor=0.002,
        edge_usd_ceiling=0.030,
        cooldown_floor=10,
        cooldown_ceiling=900,
    ),
    "calm": ModeSpec(
        name="calm",
        max_actions_per_tick=2,
        strict_floor=50,
        soft_floor=45,
        strict_ceiling=88,
        soft_ceiling=82,
        volume_floor=10.0,
        volume_ceiling=300.0,
        edge_floor=0.25,
        edge_ceiling=2.00,
        edge_usd_floor=0.003,
        edge_usd_ceiling=0.040,
        cooldown_floor=20,
        cooldown_ceiling=1200,
    ),
    "sniper": ModeSpec(
        name="sniper",
        max_actions_per_tick=2,
        strict_floor=58,
        soft_floor=52,
        strict_ceiling=95,
        soft_ceiling=90,
        volume_floor=16.0,
        volume_ceiling=600.0,
        edge_floor=0.35,
        edge_ceiling=3.00,
        edge_usd_floor=0.004,
        edge_usd_ceiling=0.050,
        cooldown_floor=30,
        cooldown_ceiling=1800,
    ),
}


@dataclass
class WindowMetrics:
    scanned: int = 0
    trade_candidates: int = 0
    opened_from_summary: int = 0
    selected_from_batch: int = 0
    opened_from_batch: int = 0
    filter_fail_reasons: Counter[str] = field(default_factory=Counter)
    autotrade_skip_reasons: Counter[str] = field(default_factory=Counter)
    pe_reasons: Counter[str] = field(default_factory=Counter)
    buy_symbol_counts: Counter[str] = field(default_factory=Counter)
    blacklist_detail_reasons: Counter[str] = field(default_factory=Counter)
    symbol_concentration_drop_hits: int = 0
    open_positions: int = 0
    total_closed: int = 0
    winrate_total: float | None = None
    realized_total: float | None = None
    lines_seen: int = 0

    @property
    def has_flow_problem(self) -> bool:
        return self.selected_from_batch <= 0 or self.opened_from_batch <= 0

    @property
    def autobuy_total(self) -> int:
        return int(sum(int(v) for v in self.buy_symbol_counts.values()))

    @property
    def unique_buy_symbols(self) -> int:
        return int(len(self.buy_symbol_counts))

    @property
    def top_buy_symbol(self) -> str:
        if not self.buy_symbol_counts:
            return ""
        return str(self.buy_symbol_counts.most_common(1)[0][0])

    @property
    def top_buy_symbol_count(self) -> int:
        if not self.buy_symbol_counts:
            return 0
        return int(self.buy_symbol_counts.most_common(1)[0][1])

    @property
    def top_buy_symbol_share(self) -> float:
        total = self.autobuy_total
        if total <= 0:
            return 0.0
        return float(self.top_buy_symbol_count) / float(total)

    @property
    def open_rate(self) -> float:
        selected = int(self.selected_from_batch)
        if selected <= 0:
            return 0.0
        return float(int(self.opened_from_batch)) / float(selected)


@dataclass
class Action:
    key: str
    old_value: str
    new_value: str
    reason: str


@dataclass
class AdaptiveBounds:
    strict_floor: int
    soft_floor: int
    strict_ceiling: int
    soft_ceiling: int
    volume_floor: float
    volume_ceiling: float
    edge_floor: float
    edge_ceiling: float
    edge_usd_floor: float
    edge_usd_ceiling: float
    cooldown_floor: int
    cooldown_ceiling: int


def _project_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preset_path(root: Path, profile_id: str) -> Path:
    return root / "data" / "matrix" / "user_presets" / f"{profile_id}.json"


def _session_dir(root: Path, profile_id: str) -> Path:
    return root / "logs" / "matrix" / profile_id / "sessions"


def _runtime_log_path(root: Path, profile_id: str) -> Path:
    return root / "logs" / "matrix" / profile_id / "runtime_tuner.jsonl"


def _active_matrix_path(root: Path) -> Path:
    return root / "data" / "matrix" / "runs" / "active_matrix.json"


def _runtime_state_path(root: Path, profile_id: str) -> Path:
    return root / "logs" / "matrix" / profile_id / "runtime_tuner_state.json"


def _runtime_lock_path(root: Path, profile_id: str) -> Path:
    return root / "logs" / "matrix" / profile_id / "runtime_tuner.lock.json"


def _runtime_patch_path(root: Path, profile_id: str) -> Path:
    return root / "logs" / "matrix" / profile_id / "runtime_tuner_runtime_overrides.json"


def _session_age_seconds(session_path: Path) -> float:
    try:
        return max(0.0, float(time.time()) - float(session_path.stat().st_mtime))
    except Exception:
        return 0.0


def _heartbeat_recent(root: Path, profile_id: str, *, max_age_seconds: float = 90.0) -> bool:
    hb_path = root / "logs" / "matrix" / str(profile_id) / "heartbeat.json"
    if not hb_path.exists():
        return False
    try:
        payload = json.loads(hb_path.read_text(encoding="utf-8"))
        ts = float(payload.get("ts", 0.0) or 0.0)
        if ts <= 0.0:
            return False
        age = max(0.0, float(time.time()) - ts)
        return age <= float(max_age_seconds)
    except Exception:
        return False


def _pid_is_running(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except OSError:
        pass
    except Exception:
        pass
    if os.name == "nt":
        try:
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            probe = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
                timeout=max(1, int(PROCESS_PROBE_TIMEOUT_SECONDS)),
                creationflags=creationflags,
            )
            line = str(probe.stdout or "").strip()
            if not line:
                return False
            if "No tasks are running" in line:
                return False
            return str(int(pid)) in line
        except Exception:
            return False
    return False


def _runtime_activity_age_seconds(root: Path, profile_id: str) -> float:
    now_ts = float(time.time())
    mtimes: list[float] = []
    for path in (
        _runtime_state_path(root, profile_id),
        _runtime_log_path(root, profile_id),
        root / "logs" / "matrix" / str(profile_id) / "heartbeat.json",
    ):
        try:
            if path.exists():
                mtimes.append(float(path.stat().st_mtime))
        except Exception:
            continue
    if not mtimes:
        return float("inf")
    return max(0.0, now_ts - max(mtimes))


def _terminate_pid(pid: int, *, timeout_seconds: int = LOCK_STALE_TERMINATE_TIMEOUT_SECONDS) -> bool:
    pid = int(pid or 0)
    if pid <= 0:
        return True
    if not _pid_is_running(pid):
        return True
    try:
        if os.name == "nt":
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                timeout=max(2, int(timeout_seconds)),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        else:
            try:
                os.kill(pid, 15)
            except Exception:
                pass
    except Exception:
        pass
    deadline = float(time.time()) + float(max(2, int(timeout_seconds)))
    while float(time.time()) < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.25)
    return not _pid_is_running(pid)


def _acquire_runtime_lock(root: Path, profile_id: str, owner_label: str) -> tuple[bool, str]:
    path = _runtime_lock_path(root, profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    current_pid = int(os.getpid())
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                existing = payload
        except Exception:
            existing = {}
    try:
        owner_pid = int(existing.get("pid", 0) or 0)
    except Exception:
        owner_pid = 0
    owner = str(existing.get("owner", "") or "").strip() or "unknown"
    started = str(existing.get("started_at", "") or "").strip() or "unknown"
    started_dt = _parse_any_ts(started)
    lock_age_seconds = 0.0
    if started_dt is not None:
        lock_age_seconds = max(0.0, float(time.time()) - float(started_dt.timestamp()))
    activity_age_seconds = _runtime_activity_age_seconds(root, profile_id)
    if owner_pid > 0 and owner_pid != current_pid and _pid_is_running(owner_pid):
        stale_activity = (
            activity_age_seconds != float("inf")
            and activity_age_seconds >= float(LOCK_STALE_SECONDS)
        )
        stale_lock = lock_age_seconds >= float(LOCK_STALE_SECONDS)
        if stale_activity and stale_lock:
            terminated = _terminate_pid(owner_pid, timeout_seconds=LOCK_STALE_TERMINATE_TIMEOUT_SECONDS)
            if not terminated and _pid_is_running(owner_pid):
                return (
                    False,
                    "runtime_tuner stale lock takeover failed: "
                    f"profile={profile_id} pid={owner_pid} owner={owner} "
                    f"activity_age={activity_age_seconds:.1f}s lock_age={lock_age_seconds:.1f}s",
                )
        else:
            activity_age_text = (
                "inf" if activity_age_seconds == float("inf") else f"{activity_age_seconds:.1f}s"
            )
            lock_age_text = f"{lock_age_seconds:.1f}s"
            return (
                False,
                "runtime_tuner lock busy: "
                f"profile={profile_id} pid={owner_pid} owner={owner} started_at={started} "
                f"activity_age={activity_age_text} lock_age={lock_age_text}",
            )
    if owner_pid > 0 and owner_pid != current_pid and _pid_is_running(owner_pid):
        return (
            False,
            f"runtime_tuner lock busy: profile={profile_id} pid={owner_pid} owner={owner} started_at={started}",
        )
    payload = {
        "pid": current_pid,
        "owner": str(owner_label or "runtime_tuner"),
        "profile_id": str(profile_id),
        "started_at": _now_iso(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return True, ""


def _release_runtime_lock(root: Path, profile_id: str) -> None:
    path = _runtime_lock_path(root, profile_id)
    if not path.exists():
        return
    current_pid = int(os.getpid())
    owner_pid = 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            owner_pid = int(payload.get("pid", 0) or 0)
    except Exception:
        owner_pid = 0
    if owner_pid > 0 and owner_pid != current_pid and _pid_is_running(owner_pid):
        return
    try:
        path.unlink()
    except Exception:
        return


def _load_runtime_state(root: Path, profile_id: str) -> RuntimeState:
    path = _runtime_state_path(root, profile_id)
    if not path.exists():
        return RuntimeState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return RuntimeState()
    if not isinstance(payload, dict):
        return RuntimeState()
    stable_raw = payload.get("stable_overrides", {})
    stable_map: dict[str, str] = {}
    if isinstance(stable_raw, dict):
        stable_map = {str(k): str(v) for k, v in stable_raw.items()}
    restart_history_raw = payload.get("restart_history_ts", [])
    restart_history: list[float] = []
    if isinstance(restart_history_raw, list):
        for item in restart_history_raw:
            try:
                restart_history.append(float(item))
            except Exception:
                continue
    return RuntimeState(
        degrade_streak=int(payload.get("degrade_streak", 0) or 0),
        stable_hash=str(payload.get("stable_hash", "") or ""),
        stable_overrides=stable_map,
        last_phase=str(payload.get("last_phase", "expand") or "expand"),
        last_effective_hash=str(payload.get("last_effective_hash", "") or ""),
        last_effective_at=str(payload.get("last_effective_at", "") or ""),
        churn_lock_symbol=str(payload.get("churn_lock_symbol", "") or "").strip().upper(),
        churn_lock_until_ts=float(payload.get("churn_lock_until_ts", 0.0) or 0.0),
        churn_lock_last_reason=str(payload.get("churn_lock_last_reason", "") or ""),
        effective_target_trades_per_hour=float(payload.get("effective_target_trades_per_hour", 0.0) or 0.0),
        effective_target_stable_ticks=int(payload.get("effective_target_stable_ticks", 0) or 0),
        effective_target_fail_ticks=int(payload.get("effective_target_fail_ticks", 0) or 0),
        effective_target_last_reason=str(payload.get("effective_target_last_reason", "") or ""),
        restart_history_ts=restart_history,
    )


def _save_runtime_state(root: Path, profile_id: str, state: RuntimeState) -> None:
    path = _runtime_state_path(root, profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "degrade_streak": int(max(0, state.degrade_streak)),
        "stable_hash": str(state.stable_hash or ""),
        "stable_overrides": {str(k): str(v) for k, v in (state.stable_overrides or {}).items()},
        "last_phase": str(state.last_phase or "expand"),
        "last_effective_hash": str(state.last_effective_hash or ""),
        "last_effective_at": str(state.last_effective_at or ""),
        "churn_lock_symbol": str(state.churn_lock_symbol or "").strip().upper(),
        "churn_lock_until_ts": float(state.churn_lock_until_ts or 0.0),
        "churn_lock_last_reason": str(state.churn_lock_last_reason or ""),
        "effective_target_trades_per_hour": float(state.effective_target_trades_per_hour or 0.0),
        "effective_target_stable_ticks": int(max(0, state.effective_target_stable_ticks)),
        "effective_target_fail_ticks": int(max(0, state.effective_target_fail_ticks)),
        "effective_target_last_reason": str(state.effective_target_last_reason or ""),
        "restart_history_ts": [
            float(x)
            for x in (state.restart_history_ts or [])
            if isinstance(x, (int, float)) and float(x) > 0.0
        ][-96:],
        "updated_at": _now_iso(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prune_restart_history(
    history: list[float],
    *,
    now_ts: float,
    window_seconds: float = 3600.0,
    max_points: int = 96,
) -> list[float]:
    cutoff = float(now_ts) - float(max(1.0, window_seconds))
    out: list[float] = []
    for item in history or []:
        try:
            ts = float(item)
        except Exception:
            continue
        if ts <= 0.0 or ts < cutoff or ts > (float(now_ts) + 30.0):
            continue
        out.append(ts)
    out.sort()
    if len(out) > int(max(1, max_points)):
        out = out[-int(max(1, max_points)) :]
    return out


def _restart_gate(
    *,
    now_ts: float,
    history: list[float],
    restart_cooldown_seconds: int,
    restart_max_per_hour: int,
) -> tuple[bool, float, int]:
    max_per_hour = max(1, int(restart_max_per_hour))
    cooldown = float(max(0, int(restart_cooldown_seconds)))
    last_ts = float(history[-1]) if history else 0.0
    cooldown_left = 0.0
    if last_ts > 0.0 and cooldown > 0.0:
        cooldown_left = max(0.0, cooldown - (float(now_ts) - last_ts))
    budget_left = max(0, max_per_hour - len(history))
    can_restart = bool(budget_left > 0 and cooldown_left <= 0.0)
    return can_restart, float(cooldown_left), int(budget_left)


def _latest_session_log(root: Path, profile_id: str) -> Path | None:
    base = _session_dir(root, profile_id)
    if not base.exists():
        return None
    files = sorted(base.glob("main_local_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _parse_ts(line: str) -> datetime | None:
    if len(line) < 23:
        return None
    prefix = line[:23]
    try:
        return datetime.strptime(prefix, "%Y-%m-%d %H:%M:%S,%f")
    except Exception:
        return None


def _normalize_blacklist_detail(detail: str) -> str:
    text = str(detail or "").strip().lower()
    if not text:
        return "unknown"
    chunk = text.split()[0]
    head = chunk.split(":", 1)[0].strip()
    return head or "unknown"


def _blacklist_detail_class(detail: str) -> str:
    head = _normalize_blacklist_detail(detail)
    if head.startswith("honeypot_guard"):
        return "honeypot"
    if head in {"", "unknown", "blacklist"}:
        return "unknown"
    if head.startswith("safety_guard") or head.startswith("hard_blocklist"):
        return "safety"
    if head.startswith("unsupported_") or head.startswith("roundtrip_"):
        return "route"
    if head.startswith("sell_fail") or head.startswith("proof_sell_fail"):
        return "sell_fail"
    return "other"


def _read_window_metrics(session_log: Path, window_minutes: int) -> WindowMetrics:
    metrics = WindowMetrics()
    if not session_log.exists():
        return metrics
    cutoff = datetime.now() - timedelta(minutes=max(1, window_minutes))
    for line in session_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        ts = _parse_ts(line)
        if ts is not None and ts < cutoff:
            continue
        metrics.lines_seen += 1
        m = SUMMARY_RE.search(line)
        if m:
            metrics.scanned += int(m.group(1))
            metrics.trade_candidates += int(m.group(2))
            metrics.opened_from_summary += int(m.group(3))
        m = FILTER_FAIL_RE.search(line)
        if m:
            metrics.filter_fail_reasons[m.group(1)] += 1
        m = AUTOTRADE_SKIP_RE.search(line)
        if m:
            metrics.autotrade_skip_reasons[m.group(1)] += 1
        m = AUTOTRADE_SKIP_DETAIL_RE.search(line)
        if m and str(m.group(1) or "").strip().lower() == "blacklist":
            metrics.blacklist_detail_reasons[_normalize_blacklist_detail(str(m.group(2) or ""))] += 1
        m = AUTOTRADE_BATCH_RE.search(line)
        if m:
            metrics.selected_from_batch += int(m.group(1))
            metrics.opened_from_batch += int(m.group(2))
            metrics.pe_reasons[m.group(3)] += 1
        m = AUTO_BUY_RE.search(line)
        if m:
            symbol = str(m.group(1) or "").strip()
            if symbol:
                metrics.buy_symbol_counts[symbol] += 1
        if "V2_QUALITY_GATE" in line and "drop=" in line and "symbol_concentration" in line:
            m = SYMBOL_CONCENTRATION_DROP_RE.search(line)
            if m:
                metrics.symbol_concentration_drop_hits += int(m.group(1))
            else:
                metrics.symbol_concentration_drop_hits += 1
        m = PAPER_SUMMARY_RE.search(line)
        if m:
            metrics.total_closed = int(m.group(1))
            metrics.winrate_total = float(m.group(2))
            metrics.realized_total = float(m.group(3))
            metrics.open_positions = int(m.group(4))
    return metrics


def _parse_any_ts(raw: Any) -> datetime | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except Exception:
            return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _read_jsonl_window(path: Path, *, cutoff: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            ts = _parse_any_ts(row.get("timestamp"))
            if ts is None:
                ts = _parse_any_ts(row.get("ts"))
            if ts is not None and ts < cutoff:
                continue
            rows.append(row)
    return rows


def _row_candidate_key(row: dict[str, Any], idx: int) -> str:
    candidate_id = str(row.get("candidate_id", "") or "").strip()
    if candidate_id:
        return candidate_id
    token = str(row.get("token_address", "") or row.get("address", "") or row.get("symbol", "") or "").strip()
    ts = str(row.get("timestamp", "") or row.get("ts", "") or "").strip()
    if token or ts:
        return f"{token}|{ts}"
    return f"row_{idx}"


def _counter_top(counter: Counter[str], limit: int = 6) -> list[tuple[str, int]]:
    return [(str(k), int(v)) for k, v in counter.most_common(limit)]


def _to_share_rows(counter: Counter[str], *, limit: int = 8) -> list[dict[str, Any]]:
    total = int(sum(int(v) for v in counter.values()))
    out: list[dict[str, Any]] = []
    for reason, count in counter.most_common(limit):
        c = int(count)
        share = (float(c) / float(total)) if total > 0 else 0.0
        out.append({"reason_code": str(reason), "count": c, "share": round(share, 6)})
    return out


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _funnel_15m(candidate_rows: list[dict[str, Any]], trade_rows: list[dict[str, Any]]) -> dict[str, int]:
    raw_ids: set[str] = set()
    blocked_source: set[str] = set()
    blocked_precheck: set[str] = set()
    blocked_threshold: set[str] = set()
    post_filters_passed: set[str] = set()
    quarantine_blocked: set[str] = set()
    quality_skip_total: set[str] = set()
    for idx, row in enumerate(candidate_rows):
        key = _row_candidate_key(row, idx)
        raw_ids.add(key)
        stage = str(row.get("decision_stage", "") or "").strip().lower()
        decision = str(row.get("decision", "") or "").strip().lower()
        reason = str(row.get("reason", "") or "").strip().lower()
        if stage == "filter_fail":
            if reason in SOURCE_BLOCK_REASONS:
                blocked_source.add(key)
            elif reason in PRECHECK_BLOCK_REASONS:
                blocked_precheck.add(key)
            elif reason in THRESHOLD_BLOCK_REASONS:
                blocked_threshold.add(key)
        elif stage == "post_filters" and decision in {"candidate_pass", "pass", "ok"}:
            post_filters_passed.add(key)
        elif stage == "quality_gate" and decision == "skip":
            quality_skip_total.add(key)
            if reason in QUARANTINE_BLOCK_REASONS:
                quarantine_blocked.add(key)

    plan_attempts = 0
    plan_skips = 0
    buys = 0
    sells = 0
    for row in trade_rows:
        stage = str(row.get("decision_stage", "") or "").strip().lower()
        decision = str(row.get("decision", "") or "").strip().lower()
        if stage == "plan_trade":
            plan_attempts += 1
            if decision == "skip":
                plan_skips += 1
        elif stage == "trade_open" and decision == "open":
            buys += 1
        elif stage == "trade_close" and decision == "close":
            sells += 1

    raw = int(len(raw_ids))
    source = max(0, raw - int(len(blocked_source)))
    pre = max(0, source - int(len(blocked_precheck)))
    thr = int(len(post_filters_passed)) if post_filters_passed else max(0, pre - int(len(blocked_threshold)))
    quarantine = max(0, thr - int(len(quarantine_blocked)))
    exec_ready = max(0, int(plan_attempts) - int(plan_skips))
    return {
        "raw": int(raw),
        "source": int(source),
        "pre": int(pre),
        "thr": int(thr),
        "quarantine": int(quarantine),
        "exec": int(exec_ready),
        "buy": int(buys),
        "raw_candidates": int(raw),
        "post_filters_passed": int(len(post_filters_passed)),
        "quality_gate_skips": int(len(quality_skip_total)),
        "plan_attempts": int(plan_attempts),
        "plan_skips": int(plan_skips),
        "trade_open": int(buys),
        "trade_close": int(sells),
    }


def _top_reasons_15m(candidate_rows: list[dict[str, Any]], trade_rows: list[dict[str, Any]]) -> dict[str, Any]:
    filter_fail = Counter[str]()
    quality_skip = Counter[str]()
    plan_skip = Counter[str]()
    execute_fail = Counter[str]()
    exit_reasons = Counter[str]()
    for row in candidate_rows:
        stage = str(row.get("decision_stage", "") or "").strip().lower()
        decision = str(row.get("decision", "") or "").strip().lower()
        reason = str(row.get("reason", "") or "").strip().lower() or "unknown"
        if stage == "filter_fail":
            filter_fail[reason] += 1
        elif stage == "quality_gate" and decision == "skip":
            quality_skip[reason] += 1
    for row in trade_rows:
        stage = str(row.get("decision_stage", "") or "").strip().lower()
        decision = str(row.get("decision", "") or "").strip().lower()
        reason = str(row.get("reason", "") or "").strip().lower() or "unknown"
        if stage == "plan_trade" and decision == "skip":
            plan_skip[reason] += 1
        if decision == "fail" or reason in EXEC_ROUTE_FAIL_REASONS:
            execute_fail[reason] += 1
        if stage == "trade_close":
            exit_reasons[reason] += 1
    return {
        "filter_fail": _to_share_rows(filter_fail),
        "quality_skip": _to_share_rows(quality_skip),
        "plan_skip": _to_share_rows(plan_skip),
        "execute_fail": _to_share_rows(execute_fail),
        "exit": _to_share_rows(exit_reasons),
    }


def _exec_health_15m(trade_rows: list[dict[str, Any]]) -> dict[str, Any]:
    plan_attempts = 0
    plan_skips = 0
    opens = 0
    closes = 0
    route_fail = Counter[str]()
    buy_fail = Counter[str]()
    sell_fail = Counter[str]()
    close_pnl_pct: list[float] = []
    close_pnl_usd: list[float] = []
    for row in trade_rows:
        stage = str(row.get("decision_stage", "") or "").strip().lower()
        decision = str(row.get("decision", "") or "").strip().lower()
        reason = str(row.get("reason", "") or "").strip().lower() or "unknown"
        if stage == "plan_trade":
            plan_attempts += 1
            if decision == "skip":
                plan_skips += 1
            if reason in EXEC_ROUTE_FAIL_REASONS:
                route_fail[reason] += 1
        elif stage == "trade_open":
            if decision == "open":
                opens += 1
            elif decision == "fail":
                buy_fail[reason] += 1
        elif stage == "trade_close":
            if decision == "close":
                closes += 1
                try:
                    close_pnl_pct.append(float(row.get("pnl_percent", 0.0) or 0.0))
                except Exception:
                    pass
                try:
                    close_pnl_usd.append(float(row.get("pnl_usd", 0.0) or 0.0))
                except Exception:
                    pass
            elif decision == "fail":
                sell_fail[reason] += 1

    losses_pct = [x for x in close_pnl_pct if x < 0.0]
    win_count = sum(1 for x in close_pnl_usd if x > 0.0)
    loss_count = sum(1 for x in close_pnl_usd if x < 0.0)
    winrate = (float(win_count) / float(max(1, win_count + loss_count))) if (win_count + loss_count) > 0 else 0.0
    return {
        "plan_attempts": int(plan_attempts),
        "plan_skips": int(plan_skips),
        "opens": int(opens),
        "closes": int(closes),
        "open_rate": round(float(opens) / float(max(1, plan_attempts)), 6),
        "close_rate": round(float(closes) / float(max(1, opens)), 6),
        "winrate_closed": round(winrate, 6),
        "buy_fail": _to_share_rows(buy_fail, limit=6),
        "sell_fail": _to_share_rows(sell_fail, limit=6),
        "route_fail": _to_share_rows(route_fail, limit=6),
        "sell_quote_ok": "N/A",
        "roundtrip_loss_avg_pct": round(float(sum(losses_pct) / len(losses_pct)), 6) if losses_pct else "N/A",
        "roundtrip_loss_median_pct": round(float(_median(losses_pct) or 0.0), 6) if losses_pct else "N/A",
    }


def _exit_mix_60m(trade_rows: list[dict[str, Any]]) -> dict[str, Any]:
    close_reasons = Counter[str]()
    pnl_sum = 0.0
    close_pnl_usd: list[float] = []
    for row in trade_rows:
        stage = str(row.get("decision_stage", "") or "").strip().lower()
        if stage != "trade_close":
            continue
        reason = str(row.get("reason", "") or "").strip().upper() or "UNKNOWN"
        close_reasons[reason] += 1
        try:
            pnl = float(row.get("pnl_usd", 0.0) or 0.0)
            pnl_sum += pnl
            close_pnl_usd.append(pnl)
        except Exception:
            pass
    total = int(sum(int(v) for v in close_reasons.values()))
    out: list[dict[str, Any]] = []
    for reason, count in close_reasons.most_common(12):
        c = int(count)
        share = (float(c) / float(total)) if total > 0 else 0.0
        out.append({"reason": str(reason), "count": c, "share": round(share, 6)})
    wins = [x for x in close_pnl_usd if x > 0.0]
    losses = [x for x in close_pnl_usd if x < 0.0]
    largest_loss_usd = min(losses) if losses else None
    median_win_usd = _median(wins) if wins else None
    tail_loss_ratio = None
    if largest_loss_usd is not None and median_win_usd is not None and float(median_win_usd) > 1e-9:
        tail_loss_ratio = abs(float(largest_loss_usd)) / float(median_win_usd)
    return {
        "total": int(total),
        "pnl_usd_sum": round(float(pnl_sum), 6),
        "distribution": out,
        "largest_loss_usd": round(float(largest_loss_usd), 6) if largest_loss_usd is not None else "N/A",
        "median_win_usd": round(float(median_win_usd), 6) if median_win_usd is not None else "N/A",
        "tail_loss_ratio": round(float(tail_loss_ratio), 6) if tail_loss_ratio is not None else "N/A",
    }


def _symbol_churn_15m(trade_rows: list[dict[str, Any]]) -> dict[str, Any]:
    opens = Counter[str]()
    closes = Counter[str]()
    flat_closes = Counter[str]()
    churn_reason_closes = Counter[str]()
    loss_closes = Counter[str]()
    pnl_sum_by_symbol: dict[str, float] = {}

    for row in trade_rows:
        stage = str(row.get("decision_stage", "") or "").strip().lower()
        decision = str(row.get("decision", "") or "").strip().lower()
        symbol = str(row.get("symbol", "") or "").strip().upper()
        if not symbol:
            continue
        if stage == "trade_open" and decision == "open":
            opens[symbol] += 1
            continue
        if stage != "trade_close" or decision != "close":
            continue
        closes[symbol] += 1
        reason = str(row.get("reason", "") or "").strip().upper()
        pnl_usd = _safe_float(row.get("pnl_usd", 0.0), 0.0)
        pnl_sum_by_symbol[symbol] = float(pnl_sum_by_symbol.get(symbol, 0.0)) + float(pnl_usd)
        if reason in CHURN_CLOSE_REASONS:
            churn_reason_closes[symbol] += 1
        if abs(float(pnl_usd)) <= float(CHURN_FLAT_PNL_ABS_USD):
            flat_closes[symbol] += 1
        if float(pnl_usd) < 0.0:
            loss_closes[symbol] += 1

    open_total = int(sum(int(v) for v in opens.values()))
    close_total = int(sum(int(v) for v in closes.values()))
    top_opens = [
        {
            "symbol": str(sym),
            "count": int(count),
            "share": round(float(count) / float(max(1, open_total)), 6),
        }
        for sym, count in opens.most_common(6)
    ]
    top_closes = [
        {
            "symbol": str(sym),
            "count": int(count),
            "share": round(float(count) / float(max(1, close_total)), 6),
        }
        for sym, count in closes.most_common(6)
    ]
    top_symbol = str(opens.most_common(1)[0][0]) if opens else ""
    open_count = int(opens.get(top_symbol, 0))
    close_count = int(closes.get(top_symbol, 0))
    open_share = float(open_count) / float(max(1, open_total))
    flat_close_count = int(flat_closes.get(top_symbol, 0))
    flat_close_share = float(flat_close_count) / float(max(1, close_count))
    churn_reason_share = float(churn_reason_closes.get(top_symbol, 0)) / float(max(1, close_count))
    loss_close_share = float(loss_closes.get(top_symbol, 0)) / float(max(1, close_count))
    net_pnl_usd = float(pnl_sum_by_symbol.get(top_symbol, 0.0))

    detected = (
        bool(top_symbol)
        and open_count >= int(CHURN_MIN_OPENS)
        and close_count >= int(CHURN_MIN_CLOSES)
        and open_share >= float(CHURN_MIN_OPEN_SHARE)
        and flat_close_share >= float(CHURN_MIN_FLAT_CLOSE_SHARE)
        and churn_reason_share >= 0.50
    )
    if detected and close_count >= 4 and open_share >= 0.75 and flat_close_share >= 0.70:
        ttl_seconds = 1800
    elif detected:
        ttl_seconds = 1200
    else:
        ttl_seconds = 0

    return {
        "detected": bool(detected),
        "symbol": str(top_symbol),
        "open_count": int(open_count),
        "close_count": int(close_count),
        "open_share": round(float(open_share), 6),
        "flat_close_count": int(flat_close_count),
        "flat_close_share": round(float(flat_close_share), 6),
        "churn_reason_share": round(float(churn_reason_share), 6),
        "loss_close_share": round(float(loss_close_share), 6),
        "net_pnl_usd": round(float(net_pnl_usd), 6),
        "ttl_seconds": int(ttl_seconds),
        "top_open_symbols": top_opens,
        "top_close_symbols": top_closes,
    }


def _blacklist_forensics(
    *,
    candidate_rows_15m: list[dict[str, Any]],
    trade_rows_15m: list[dict[str, Any]],
    trade_rows_60m: list[dict[str, Any]],
    metrics: WindowMetrics | None = None,
) -> dict[str, Any]:
    plan_attempts_15m = 0
    plan_blacklist_15m = 0
    plan_blacklist_60m = 0
    filter_blacklist_15m = 0
    unique_tokens_15m: set[str] = set()
    token_hits_15m = Counter[str]()
    detail_counter_15m = Counter[str]()
    detail_class_counter_15m = Counter[str]()
    for row in candidate_rows_15m:
        stage = str(row.get("decision_stage", "") or "").strip().lower()
        reason = str(row.get("reason", "") or "").strip().lower()
        if stage == "filter_fail" and reason == "blacklist":
            filter_blacklist_15m += 1
    for row in trade_rows_15m:
        stage = str(row.get("decision_stage", "") or "").strip().lower()
        decision = str(row.get("decision", "") or "").strip().lower()
        reason = str(row.get("reason", "") or "").strip().lower()
        if stage == "plan_trade":
            plan_attempts_15m += 1
            if decision == "skip" and reason == "blacklist":
                plan_blacklist_15m += 1
                token = str(row.get("token_address", "") or row.get("symbol", "") or "").strip().lower()
                if token:
                    unique_tokens_15m.add(token)
                    token_hits_15m[token] += 1
                detail = _normalize_blacklist_detail(str(row.get("detail", "") or ""))
                detail_counter_15m[detail] += 1
    for row in trade_rows_60m:
        stage = str(row.get("decision_stage", "") or "").strip().lower()
        decision = str(row.get("decision", "") or "").strip().lower()
        reason = str(row.get("reason", "") or "").strip().lower()
        if stage == "plan_trade" and decision == "skip" and reason == "blacklist":
            plan_blacklist_60m += 1
    if metrics is not None:
        detail_counter_15m.update(metrics.blacklist_detail_reasons)
    detail_class_counter_15m = Counter[str]()
    for detail, count in detail_counter_15m.items():
        detail_class_counter_15m[_blacklist_detail_class(detail)] += int(count or 0)
    detail_total = int(sum(int(v or 0) for v in detail_class_counter_15m.values()))
    unknown_count = int(detail_class_counter_15m.get("unknown", 0) or 0)
    honeypot_count = int(detail_class_counter_15m.get("honeypot", 0) or 0)
    unknown_share = float(unknown_count) / float(max(1, detail_total))
    honeypot_share = float(honeypot_count) / float(max(1, detail_total))
    share_15m = float(plan_blacklist_15m) / float(max(1, plan_attempts_15m))
    top_tokens = [{"token": str(k), "count": int(v)} for k, v in token_hits_15m.most_common(8)]
    return {
        "plan_attempts_15m": int(plan_attempts_15m),
        "plan_skip_blacklist_15m": int(plan_blacklist_15m),
        "plan_skip_blacklist_share_15m": round(share_15m, 6),
        "plan_skip_blacklist_60m": int(plan_blacklist_60m),
        "filter_fail_blacklist_15m": int(filter_blacklist_15m),
        "unique_blacklist_tokens_15m": int(len(unique_tokens_15m)),
        "top_blacklist_tokens_15m": top_tokens,
        "top_blacklist_details_15m": _to_share_rows(detail_counter_15m, limit=8),
        "top_blacklist_detail_classes_15m": _to_share_rows(detail_class_counter_15m, limit=6),
        "plan_skip_blacklist_unknown_15m": int(unknown_count),
        "plan_skip_blacklist_honeypot_15m": int(honeypot_count),
        "plan_skip_blacklist_unknown_share_15m": round(float(unknown_share), 6),
        "plan_skip_blacklist_honeypot_share_15m": round(float(honeypot_share), 6),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _target_policy_from_args(args: argparse.Namespace) -> TargetPolicy:
    payload: dict[str, Any] = {}
    raw_path = str(getattr(args, "target_policy_file", "") or "").strip()
    if raw_path:
        try:
            data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                payload = data
        except Exception as exc:
            raise SystemExit(f"Invalid target policy file '{raw_path}': {exc}") from exc

    def _pick(name: str, cli_default: Any) -> Any:
        return payload.get(name, getattr(args, name, cli_default))

    return TargetPolicy(
        target_trades_per_hour=max(0.0, _safe_float(_pick("target_trades_per_hour", 12.0), 12.0)),
        target_pnl_per_hour_usd=_safe_float(_pick("target_pnl_per_hour_usd", 0.05), 0.05),
        min_open_rate_15m=max(0.0, min(1.0, _safe_float(_pick("min_open_rate_15m", 0.04), 0.04))),
        min_selected_15m=max(1, int(_safe_float(_pick("min_selected_15m", 16), 16))),
        min_closed_for_risk_checks=max(1, int(_safe_float(_pick("min_closed_for_risk_checks", 6), 6))),
        min_winrate_closed_15m=max(0.0, min(1.0, _safe_float(_pick("min_winrate_closed_15m", 0.35), 0.35))),
        max_blacklist_share_15m=max(0.0, min(1.0, _safe_float(_pick("max_blacklist_share_15m", 0.45), 0.45))),
        max_blacklist_added_15m=max(1, int(_safe_float(_pick("max_blacklist_added_15m", 80), 80))),
        rollback_degrade_streak=max(1, int(_safe_float(_pick("rollback_degrade_streak", 3), 3))),
        hold_hysteresis_open_rate=max(0.0, min(1.0, _safe_float(_pick("hold_hysteresis_open_rate", 0.07), 0.07))),
        hold_hysteresis_trades_per_hour=max(0.0, _safe_float(_pick("hold_hysteresis_trades_per_hour", 6.0), 6.0)),
        pre_risk_min_plan_attempts_15m=max(
            1,
            int(_safe_float(_pick("pre_risk_min_plan_attempts_15m", 8), 8)),
        ),
        pre_risk_route_fail_rate_15m=max(
            0.0,
            min(1.0, _safe_float(_pick("pre_risk_route_fail_rate_15m", 0.35), 0.35)),
        ),
        pre_risk_buy_fail_rate_15m=max(
            0.0,
            min(1.0, _safe_float(_pick("pre_risk_buy_fail_rate_15m", 0.35), 0.35)),
        ),
        pre_risk_sell_fail_rate_15m=max(
            0.0,
            min(1.0, _safe_float(_pick("pre_risk_sell_fail_rate_15m", 0.30), 0.30)),
        ),
        pre_risk_roundtrip_loss_median_pct_15m=min(
            -0.05,
            _safe_float(_pick("pre_risk_roundtrip_loss_median_pct_15m", -1.2), -1.2),
        ),
        tail_loss_min_closes_60m=max(
            1,
            int(_safe_float(_pick("tail_loss_min_closes_60m", 6), 6)),
        ),
        tail_loss_ratio_max=max(
            1.0,
            _safe_float(_pick("tail_loss_ratio_max", 8.0), 8.0),
        ),
        diversity_min_buys_15m=max(
            1,
            int(_safe_float(_pick("diversity_min_buys_15m", 4), 4)),
        ),
        diversity_min_unique_symbols_15m=max(
            1,
            int(_safe_float(_pick("diversity_min_unique_symbols_15m", 2), 2)),
        ),
        diversity_max_top1_open_share_15m=max(
            0.20,
            min(1.0, _safe_float(_pick("diversity_max_top1_open_share_15m", 0.72), 0.72)),
        ),
        adaptive_target_enabled=str(_pick("adaptive_target_enabled", "true")).strip().lower()
        in {"1", "true", "yes", "y", "on"},
        adaptive_target_floor_trades_per_hour=max(
            0.0,
            _safe_float(_pick("adaptive_target_floor_trades_per_hour", 4.0), 4.0),
        ),
        adaptive_target_step_up_trades_per_hour=max(
            0.1,
            _safe_float(_pick("adaptive_target_step_up_trades_per_hour", 1.5), 1.5),
        ),
        adaptive_target_step_down_trades_per_hour=max(
            0.1,
            _safe_float(_pick("adaptive_target_step_down_trades_per_hour", 3.0), 3.0),
        ),
        adaptive_target_headroom_mult=max(
            1.0,
            _safe_float(_pick("adaptive_target_headroom_mult", 1.35), 1.35),
        ),
        adaptive_target_headroom_add_trades_per_hour=max(
            0.0,
            _safe_float(_pick("adaptive_target_headroom_add_trades_per_hour", 2.0), 2.0),
        ),
        adaptive_target_stable_ticks_for_step_up=max(
            1,
            int(_safe_float(_pick("adaptive_target_stable_ticks_for_step_up", 2), 2)),
        ),
        adaptive_target_fail_ticks_for_step_down=max(
            1,
            int(_safe_float(_pick("adaptive_target_fail_ticks_for_step_down", 2), 2)),
        ),
    )


def _target_floor_tph(*, requested_tph: float, configured_floor: float) -> float:
    requested = max(0.0, float(requested_tph))
    if requested <= 0.0:
        return 0.0
    return min(requested, max(1.0, float(configured_floor)))


def _resolve_effective_target_trades_per_hour(
    *,
    requested_tph: float,
    target: TargetPolicy,
    state: RuntimeState,
    throughput_est: float,
) -> tuple[float, dict[str, Any]]:
    requested = max(0.0, float(requested_tph))
    observed = max(0.0, float(throughput_est))
    if requested <= 0.0:
        return 0.0, {
            "enabled": bool(target.adaptive_target_enabled),
            "requested_tph": 0.0,
            "effective_tph": 0.0,
            "floor_tph": 0.0,
            "headroom_cap_tph": 0.0,
            "reason": "requested_zero",
        }
    floor_tph = _target_floor_tph(
        requested_tph=requested,
        configured_floor=float(target.adaptive_target_floor_trades_per_hour),
    )
    if not bool(target.adaptive_target_enabled):
        return requested, {
            "enabled": False,
            "requested_tph": round(float(requested), 6),
            "effective_tph": round(float(requested), 6),
            "floor_tph": round(float(floor_tph), 6),
            "headroom_cap_tph": round(float(requested), 6),
            "reason": "adaptive_disabled",
        }

    headroom_cap = min(
        float(requested),
        max(
            float(floor_tph),
            (float(observed) * float(target.adaptive_target_headroom_mult))
            + float(target.adaptive_target_headroom_add_trades_per_hour),
        ),
    )
    current_effective = float(state.effective_target_trades_per_hour or 0.0)
    reason = "state_keep"
    if current_effective <= 0.0:
        bootstrap = min(
            float(headroom_cap),
            max(
                float(floor_tph),
                float(observed) + float(target.adaptive_target_step_up_trades_per_hour),
            ),
        )
        current_effective = bootstrap
        reason = "bootstrap"
    current_effective = _clamp_float(float(current_effective), float(floor_tph), float(requested))
    if current_effective > float(headroom_cap):
        current_effective = max(float(floor_tph), float(headroom_cap))
        reason = "headroom_cap"

    return float(current_effective), {
        "enabled": True,
        "requested_tph": round(float(requested), 6),
        "effective_tph": round(float(current_effective), 6),
        "floor_tph": round(float(floor_tph), 6),
        "headroom_cap_tph": round(float(headroom_cap), 6),
        "reason": reason,
    }


def _update_effective_target_state(
    *,
    state: RuntimeState,
    target: TargetPolicy,
    decision: PolicyDecision,
    metrics: WindowMetrics,
) -> dict[str, Any]:
    requested = max(0.0, float(target.target_trades_per_hour))
    observed = max(0.0, float(decision.throughput_est_trades_h))
    if requested <= 0.0:
        state.effective_target_trades_per_hour = 0.0
        state.effective_target_stable_ticks = 0
        state.effective_target_fail_ticks = 0
        state.effective_target_last_reason = "requested_zero"
        return {
            "enabled": bool(target.adaptive_target_enabled),
            "requested_tph": 0.0,
            "effective_tph_before": 0.0,
            "effective_tph_after": 0.0,
            "stable_ticks": 0,
            "fail_ticks": 0,
            "adjustment": "requested_zero",
        }
    if not bool(target.adaptive_target_enabled):
        state.effective_target_trades_per_hour = float(requested)
        state.effective_target_stable_ticks = 0
        state.effective_target_fail_ticks = 0
        state.effective_target_last_reason = "adaptive_disabled"
        return {
            "enabled": False,
            "requested_tph": round(float(requested), 6),
            "effective_tph_before": round(float(requested), 6),
            "effective_tph_after": round(float(requested), 6),
            "stable_ticks": 0,
            "fail_ticks": 0,
            "adjustment": "disabled_noop",
        }

    floor_tph = _target_floor_tph(
        requested_tph=requested,
        configured_floor=float(target.adaptive_target_floor_trades_per_hour),
    )
    headroom_cap = min(
        float(requested),
        max(
            float(floor_tph),
            (float(observed) * float(target.adaptive_target_headroom_mult))
            + float(target.adaptive_target_headroom_add_trades_per_hour),
        ),
    )
    before = _clamp_float(
        float(state.effective_target_trades_per_hour or 0.0),
        float(floor_tph),
        float(requested),
    )
    if before <= 0.0:
        before = float(floor_tph)
    after = before

    stable_now = (
        (not bool(decision.flow_fail))
        and (not bool(decision.risk_fail))
        and (not bool(decision.pre_risk_fail))
        and (not bool(decision.blacklist_fail))
        and (not bool(decision.diversity_fail))
        and float(decision.open_rate_15m) >= float(target.min_open_rate_15m)
        and int(metrics.opened_from_batch) >= 1
    )
    fail_now = bool(
        decision.flow_fail
        or decision.risk_fail
        or decision.pre_risk_fail
        or decision.blacklist_fail
        or decision.diversity_fail
    )
    if stable_now:
        state.effective_target_stable_ticks = int(max(0, state.effective_target_stable_ticks) + 1)
    else:
        state.effective_target_stable_ticks = 0
    if fail_now:
        state.effective_target_fail_ticks = int(max(0, state.effective_target_fail_ticks) + 1)
    else:
        state.effective_target_fail_ticks = 0

    adjustment = "hold"
    if (
        int(state.effective_target_stable_ticks) >= int(target.adaptive_target_stable_ticks_for_step_up)
        and float(after) < float(headroom_cap)
        and float(after) < float(requested)
    ):
        after = min(
            float(headroom_cap),
            float(requested),
            float(after) + float(target.adaptive_target_step_up_trades_per_hour),
        )
        adjustment = "step_up"
        state.effective_target_stable_ticks = 0
    elif int(state.effective_target_fail_ticks) >= int(target.adaptive_target_fail_ticks_for_step_down):
        down_target = min(float(after), max(float(floor_tph), float(headroom_cap)))
        down_target = min(down_target, float(after) - float(target.adaptive_target_step_down_trades_per_hour))
        after = max(float(floor_tph), float(down_target))
        adjustment = "step_down"
        state.effective_target_fail_ticks = 0

    after = _clamp_float(float(after), float(floor_tph), float(requested))
    state.effective_target_trades_per_hour = float(after)
    state.effective_target_last_reason = str(
        f"{adjustment} observed={observed:.2f} headroom={headroom_cap:.2f} "
        f"stable={int(state.effective_target_stable_ticks)} fail={int(state.effective_target_fail_ticks)}"
    )
    return {
        "enabled": True,
        "requested_tph": round(float(requested), 6),
        "effective_tph_before": round(float(before), 6),
        "effective_tph_after": round(float(after), 6),
        "floor_tph": round(float(floor_tph), 6),
        "headroom_cap_tph": round(float(headroom_cap), 6),
        "stable_ticks": int(state.effective_target_stable_ticks),
        "fail_ticks": int(state.effective_target_fail_ticks),
        "adjustment": adjustment,
    }


def _resolve_policy_phase(
    *,
    metrics: WindowMetrics,
    telemetry: dict[str, Any],
    target: TargetPolicy,
    state: RuntimeState,
    forced_phase: str,
    effective_target_trades_per_hour: float | None = None,
) -> PolicyDecision:
    funnel = telemetry.get("funnel_15m", {}) if isinstance(telemetry, dict) else {}
    exec_health = telemetry.get("exec_health_15m", {}) if isinstance(telemetry, dict) else {}
    exit_mix = telemetry.get("exit_mix_60m", {}) if isinstance(telemetry, dict) else {}
    blacklist = telemetry.get("blacklist_forensics_15m", {}) if isinstance(telemetry, dict) else {}
    buy_15m = int(_safe_float((funnel or {}).get("buy", 0), 0.0))
    throughput_est = float(buy_15m) * 4.0
    throughput_target_requested = max(0.0, float(target.target_trades_per_hour))
    if effective_target_trades_per_hour is None:
        throughput_target_effective = float(throughput_target_requested)
    else:
        throughput_target_effective = _clamp_float(
            float(effective_target_trades_per_hour),
            0.0,
            max(float(throughput_target_requested), float(effective_target_trades_per_hour)),
        )
    open_rate = _safe_float((exec_health or {}).get("open_rate", 0.0), 0.0)
    closes = int(_safe_float((exec_health or {}).get("closes", 0), 0.0))
    winrate = _safe_float((exec_health or {}).get("winrate_closed", 0.0), 0.0)
    pnl_hour = _safe_float((exit_mix or {}).get("pnl_usd_sum", 0.0), 0.0)
    plan_attempts = int(_safe_float((exec_health or {}).get("plan_attempts", 0), 0.0))
    opens = int(_safe_float((exec_health or {}).get("opens", 0), 0.0))
    buy_total_15m = int(metrics.autobuy_total)
    buy_unique_15m = int(metrics.unique_buy_symbols)
    buy_top_share_15m = float(metrics.top_buy_symbol_share)

    def _rows_count(rows: Any) -> int:
        total = 0
        if not isinstance(rows, list):
            return 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                total += int(row.get("count", 0) or 0)
            except Exception:
                continue
        return int(total)

    def _optional_float(raw: Any) -> float | None:
        try:
            return float(raw)
        except Exception:
            return None

    route_fail_count = _rows_count((exec_health or {}).get("route_fail", []))
    buy_fail_count = _rows_count((exec_health or {}).get("buy_fail", []))
    sell_fail_count = _rows_count((exec_health or {}).get("sell_fail", []))
    route_fail_rate = float(route_fail_count) / float(max(1, plan_attempts))
    buy_fail_rate = float(buy_fail_count) / float(max(1, opens + buy_fail_count))
    sell_fail_rate = float(sell_fail_count) / float(max(1, closes + sell_fail_count))
    roundtrip_loss_median_pct = _optional_float((exec_health or {}).get("roundtrip_loss_median_pct"))

    exit_total_60m = int(_safe_float((exit_mix or {}).get("total", 0), 0.0))
    tail_loss_ratio = _optional_float((exit_mix or {}).get("tail_loss_ratio"))
    largest_loss_usd = _optional_float((exit_mix or {}).get("largest_loss_usd"))
    median_win_usd = _optional_float((exit_mix or {}).get("median_win_usd"))

    blacklist_added = int(_safe_float((blacklist or {}).get("plan_skip_blacklist_15m", 0), 0.0))
    blacklist_share = _safe_float((blacklist or {}).get("plan_skip_blacklist_share_15m", 0.0), 0.0)
    blacklist_unique = int(_safe_float((blacklist or {}).get("unique_blacklist_tokens_15m", 0), 0.0))
    blacklist_unknown_share = _safe_float((blacklist or {}).get("plan_skip_blacklist_unknown_share_15m", 0.0), 0.0)
    blacklist_honeypot_share = _safe_float((blacklist or {}).get("plan_skip_blacklist_honeypot_share_15m", 0.0), 0.0)
    blacklist_share_min_attempts = max(8, int(target.pre_risk_min_plan_attempts_15m))
    top_blacklist_rows = (blacklist or {}).get("top_blacklist_tokens_15m", [])
    top_blacklist_total = 0
    if isinstance(top_blacklist_rows, list):
        for row in top_blacklist_rows[:2]:
            if not isinstance(row, dict):
                continue
            try:
                top_blacklist_total += int(row.get("count", 0) or 0)
            except Exception:
                continue
    blacklist_top_share = float(top_blacklist_total) / float(max(1, blacklist_added)) if blacklist_added > 0 else 0.0

    reasons: list[str] = []
    if abs(float(throughput_target_effective) - float(throughput_target_requested)) > 1e-9:
        reasons.append(
            "adaptive_target "
            f"tph_effective={throughput_target_effective:.2f} tph_requested={throughput_target_requested:.2f}"
        )
    flow_fail = (
        int(metrics.selected_from_batch) >= int(target.min_selected_15m)
        and (
            open_rate < float(target.min_open_rate_15m)
            or throughput_est < float(throughput_target_effective) * 0.40
        )
    )
    pre_risk_fail = (
        plan_attempts >= int(target.pre_risk_min_plan_attempts_15m)
        and (
            route_fail_rate >= float(target.pre_risk_route_fail_rate_15m)
            or buy_fail_rate >= float(target.pre_risk_buy_fail_rate_15m)
            or sell_fail_rate >= float(target.pre_risk_sell_fail_rate_15m)
            or (
                roundtrip_loss_median_pct is not None
                and float(roundtrip_loss_median_pct) <= float(target.pre_risk_roundtrip_loss_median_pct_15m)
            )
        )
    )
    diversity_fail = (
        buy_total_15m >= int(target.diversity_min_buys_15m)
        and (
            buy_unique_15m < int(target.diversity_min_unique_symbols_15m)
            or buy_top_share_15m > float(target.diversity_max_top1_open_share_15m)
        )
    )
    risk_fail_core = (
        closes >= int(target.min_closed_for_risk_checks)
        and (winrate < float(target.min_winrate_closed_15m) or pnl_hour < float(target.target_pnl_per_hour_usd) * -1.0)
    )
    tail_loss_fail = (
        exit_total_60m >= int(target.tail_loss_min_closes_60m)
        and tail_loss_ratio is not None
        and float(tail_loss_ratio) >= float(target.tail_loss_ratio_max)
    )
    risk_fail = bool(risk_fail_core or tail_loss_fail)
    blacklist_fail_raw = (
        blacklist_added >= int(target.max_blacklist_added_15m)
        or (
            plan_attempts >= int(blacklist_share_min_attempts)
            and blacklist_share >= float(target.max_blacklist_share_15m)
        )
    )
    concentrated_single_token_pressure = (
        blacklist_fail_raw
        and blacklist_unique <= 1
        and blacklist_share >= 0.30
        and blacklist_added >= 12
    )
    concentrated_two_token_pressure = (
        blacklist_fail_raw
        and blacklist_unique <= 2
        and blacklist_share >= 0.30
        and blacklist_added >= 12
        and blacklist_top_share >= 0.90
    )
    unknown_dominated_pressure = (
        blacklist_fail_raw
        and blacklist_added >= 12
        and blacklist_unknown_share >= 0.75
        and blacklist_honeypot_share <= 0.10
    )
    blacklist_fail = bool(
        blacklist_fail_raw
        and (not concentrated_single_token_pressure)
        and (not concentrated_two_token_pressure)
        and (not unknown_dominated_pressure)
    )

    if flow_fail:
        reasons.append(
            f"flow_fail open_rate={open_rate:.3f} tph_est={throughput_est:.2f} selected={metrics.selected_from_batch}"
        )
    if pre_risk_fail:
        reasons.append(
            "pre_risk_fail "
            f"route_fail_rate={route_fail_rate:.3f} buy_fail_rate={buy_fail_rate:.3f} "
            f"sell_fail_rate={sell_fail_rate:.3f} roundtrip_loss_median={roundtrip_loss_median_pct}"
        )
    if diversity_fail:
        reasons.append(
            "diversity_fail "
            f"buys15={buy_total_15m} unique15={buy_unique_15m} top1_share15={buy_top_share_15m:.3f} "
            f"limits(min_unique={int(target.diversity_min_unique_symbols_15m)} "
            f"max_top1={float(target.diversity_max_top1_open_share_15m):.3f})"
        )
    if risk_fail:
        if tail_loss_fail:
            reasons.append(
                "tail_loss_fail "
                f"tail_loss_ratio={tail_loss_ratio} largest_loss={largest_loss_usd} median_win={median_win_usd}"
            )
        if risk_fail_core:
            reasons.append(f"risk_fail winrate={winrate:.3f} closes={closes} pnl_h={pnl_hour:.4f}")
    if blacklist_fail:
        reasons.append(
            f"blacklist_fail added15={blacklist_added} share15={blacklist_share:.3f} unique15={blacklist_unique}"
        )
    elif concentrated_single_token_pressure or concentrated_two_token_pressure:
        reasons.append(
            "blacklist_concentrated_ignore "
            f"added15={blacklist_added} share15={blacklist_share:.3f} "
            f"unique15={blacklist_unique} top2_share={blacklist_top_share:.3f}"
        )
    elif unknown_dominated_pressure:
        reasons.append(
            "blacklist_unknown_dominated_ignore "
            f"added15={blacklist_added} share15={blacklist_share:.3f} "
            f"unknown_share={blacklist_unknown_share:.3f} honeypot_share={blacklist_honeypot_share:.3f}"
        )

    prev_phase = str(state.last_phase or "expand")
    selected_now = int(metrics.selected_from_batch)
    opened_now = int(metrics.opened_from_batch)
    open_positions_now = int(metrics.open_positions)
    recovery_min_selected = max(8, int(target.min_selected_15m) // 2)
    risk_fail_recovery_unlock = (
        bool(risk_fail)
        and (not bool(blacklist_fail))
        and int(state.degrade_streak) >= int(target.rollback_degrade_streak)
        and prev_phase in {"tighten", "expand"}
        and open_positions_now <= 0
        and opened_now <= 0
        and selected_now >= int(recovery_min_selected)
    )
    phase = "hold"
    if str(forced_phase or "auto") != "auto":
        phase = str(forced_phase)
        reasons.append(f"phase_forced={phase}")
    elif risk_fail_recovery_unlock:
        phase = "expand"
        reasons.append(
            "risk_fail_recovery_unlock "
            f"degrade={int(state.degrade_streak)} selected={selected_now} opened={opened_now}"
        )
    elif pre_risk_fail or risk_fail or blacklist_fail:
        phase = "tighten"
    elif diversity_fail:
        phase = "expand"
        reasons.append("diversity_rebalance_expand")
    elif flow_fail:
        phase = "expand"
    else:
        # Hysteresis: do not oscillate from tighten/expand to hold instantly.
        if prev_phase == "tighten":
            if open_rate >= float(target.hold_hysteresis_open_rate) and throughput_est >= float(target.hold_hysteresis_trades_per_hour):
                phase = "hold"
            else:
                phase = "tighten"
                reasons.append("hysteresis_keep_tighten")
        elif prev_phase == "expand":
            if open_rate >= float(target.hold_hysteresis_open_rate):
                phase = "hold"
            else:
                phase = "expand"
                reasons.append("hysteresis_keep_expand")
        else:
            phase = "hold"

    return PolicyDecision(
        phase=phase,
        reasons=reasons,
        target_trades_per_hour_effective=round(float(throughput_target_effective), 6),
        target_trades_per_hour_requested=round(float(throughput_target_requested), 6),
        throughput_est_trades_h=round(float(throughput_est), 6),
        pnl_hour_usd=round(float(pnl_hour), 6),
        blacklist_added_15m=int(blacklist_added),
        blacklist_share_15m=round(float(blacklist_share), 6),
        open_rate_15m=round(float(open_rate), 6),
        risk_fail=bool(risk_fail),
        flow_fail=bool(flow_fail),
        blacklist_fail=bool(blacklist_fail),
        pre_risk_fail=bool(pre_risk_fail),
        diversity_fail=bool(diversity_fail),
    )


def _action_family(action: Action) -> str:
    reason = str(action.reason or "").strip().lower()
    key = str(action.key or "").strip().upper()
    if reason in FORCE_ACTION_REASONS:
        return "force"
    if "tighten" in reason:
        return "tighten"
    expand_signals = (
        "score_min_dominant",
        "safe_volume_dominant",
        "ev_net_low",
        "ev_net_low_gate",
        "ev_net_low_probe_tolerance",
        "cooldown_left",
        "cooldown_dominant_flow_expand",
        "min_trade_size",
        "low_throughput_expand_topn",
        "dedup_pressure",
        "route_pressure",
        "route_floor_normalize",
        "duplicate_choke",
        "anti_concentration",
        "quality_gate_symbol_concentration",
        "quality_source_budget_choke",
        "quality_duplicate_choke",
        "excluded_symbol_rotation",
        "feed_starvation",
        "edge_deadlock_recovery",
        "source_qos_cap_rebalance",
        "diversity_kpi_penalty",
    )
    if any(sig in reason for sig in expand_signals):
        return "expand"
    if key in {"V2_UNIVERSE_NOVELTY_MIN_SHARE", "SYMBOL_CONCENTRATION_MAX_SHARE"}:
        return "expand"
    return "neutral"


def _is_numeric_pair(old_value: str, new_value: str) -> bool:
    try:
        float(str(old_value).strip())
        float(str(new_value).strip())
        return True
    except Exception:
        return False


def _cap_numeric_value(*, old_value: str, new_value: str, limit: float, integer: bool) -> str:
    old_f = float(str(old_value).strip())
    new_f = float(str(new_value).strip())
    delta = new_f - old_f
    capped = old_f + max(-float(limit), min(float(limit), delta))
    if integer:
        return _to_int_str(int(round(capped)))
    return _to_float_str(capped)


def _cap_source_caps_delta(old_value: str, new_value: str, per_source_limit: int) -> str:
    old_caps = _get_source_cap_map({"k": old_value}, "k")
    new_caps = _get_source_cap_map({"k": new_value}, "k")
    if not new_caps:
        return str(new_value).strip()
    out = dict(new_caps)
    for source, target_val in new_caps.items():
        old_val = int(old_caps.get(source, target_val))
        step = int(target_val) - int(old_val)
        capped_val = int(old_val) + max(-int(per_source_limit), min(int(per_source_limit), step))
        out[source] = int(capped_val)
    return _format_source_cap_map(out)


def _apply_action_delta_caps(actions: list[Action], phase: str) -> tuple[list[Action], list[dict[str, Any]]]:
    limits: dict[str, tuple[float, bool]] = {
        "MARKET_MODE_STRICT_SCORE": (2.0, True),
        "MARKET_MODE_SOFT_SCORE": (2.0, True),
        "SAFE_MIN_VOLUME_5M_USD": (3.0, False),
        "MIN_EXPECTED_EDGE_PERCENT": (0.10, False),
        "MIN_EXPECTED_EDGE_USD": (0.0015, False),
        "V2_ROLLING_EDGE_MIN_PERCENT": (0.08, False),
        "V2_ROLLING_EDGE_MIN_USD": (0.0015, False),
        "V2_CALIBRATION_EDGE_USD_MIN": (0.0015, False),
        "V2_CALIBRATION_VOLUME_MIN": (12.0, False),
        "EV_FIRST_ENTRY_MIN_NET_USD": (0.0010, False),
        "EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD": (0.0010, False),
        "MAX_TOKEN_COOLDOWN_SECONDS": (40.0, True),
        "MIN_TRADE_USD": (0.10, False),
        "PAPER_TRADE_SIZE_MIN_USD": (0.10, False),
        "PAPER_TRADE_SIZE_MAX_USD": (0.20, False),
        "ENTRY_A_CORE_MIN_TRADE_USD": (0.10, False),
        "AUTO_TRADE_TOP_N": (6.0, True),
        "HEAVY_CHECK_DEDUP_TTL_SECONDS": (15.0, True),
        "PLAN_MAX_SINGLE_SOURCE_SHARE": (0.06, False),
        "PLAN_MAX_WATCHLIST_SHARE": (0.06, False),
        "PLAN_MIN_NON_WATCHLIST_PER_BATCH": (1.0, True),
        "SOURCE_ROUTER_MIN_TRADES": (2.0, True),
        "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY": (0.06, False),
        "SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY": (0.06, False),
        "TOKEN_EV_MEMORY_MIN_TRADES": (2.0, True),
        "TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY": (0.06, False),
        "TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY": (0.06, False),
        "V2_UNIVERSE_NOVELTY_MIN_SHARE": (0.06, False),
        "SYMBOL_CONCENTRATION_MAX_SHARE": (0.06, False),
        "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE": (1.0, True),
        "V2_SOURCE_QOS_TOPK_PER_CYCLE": (24.0, True),
        "V2_QUALITY_SYMBOL_MAX_SHARE": (0.03, False),
        "V2_QUALITY_SYMBOL_MIN_ABS_CAP": (1.0, True),
        "V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS": (60.0, True),
        "TOKEN_AGE_MAX": (1200.0, True),
        "SEEN_TOKEN_TTL": (1800.0, True),
        "WATCHLIST_REFRESH_SECONDS": (300.0, True),
        "WATCHLIST_MAX_TOKENS": (12.0, True),
        "WATCHLIST_MIN_LIQUIDITY_USD": (50000.0, False),
        "WATCHLIST_MIN_VOLUME_24H_USD": (150000.0, False),
        "PAPER_WATCHLIST_MIN_SCORE": (4.0, True),
        "PAPER_WATCHLIST_MIN_LIQUIDITY_USD": (30000.0, False),
        "PAPER_WATCHLIST_MIN_VOLUME_5M_USD": (200.0, False),
    }
    capped_rows: list[dict[str, Any]] = []
    per_source_cap_limit = 60
    if phase == "hold":
        per_source_cap_limit = 30
        limits = {k: (v[0] / 2.0, v[1]) for k, v in limits.items()}
    out: list[Action] = []
    for action in actions:
        key = str(action.key or "").strip().upper()
        old_value = str(action.old_value or "").strip()
        new_value = str(action.new_value or "").strip()
        updated = new_value
        if key in {"V2_SOURCE_QOS_SOURCE_CAPS", "V2_UNIVERSE_SOURCE_CAPS"}:
            updated = _cap_source_caps_delta(old_value, new_value, per_source_limit=per_source_cap_limit)
        elif key in limits and _is_numeric_pair(old_value, new_value):
            limit, integer = limits[key]
            updated = _cap_numeric_value(old_value=old_value, new_value=new_value, limit=limit, integer=bool(integer))
        if updated != new_value:
            capped_rows.append(
                {
                    "key": key,
                    "old": old_value,
                    "requested": new_value,
                    "applied": updated,
                }
            )
        out.append(Action(key=action.key, old_value=action.old_value, new_value=updated, reason=action.reason))
    return out, capped_rows


def _filter_actions_by_phase(actions: list[Action], phase: str) -> tuple[list[Action], list[dict[str, Any]]]:
    if phase not in PHASE_CHOICES:
        return actions, []
    blocked: list[dict[str, Any]] = []
    out: list[Action] = []
    for action in actions:
        family = _action_family(action)
        reason = str(action.reason or "")
        reason_low = reason.strip().lower()
        allow = False
        if family == "force":
            allow = True
        elif phase == "expand":
            allow = family in {"expand", "neutral"}
        elif phase == "tighten":
            allow = family in {"tighten", "neutral"}
            if (not allow) and family == "expand":
                # Escape hatch: when flow is dead, allow low-risk de-choke adjustments even in tighten.
                if any(sig in reason_low for sig in TIGHTEN_FLOW_ESCAPE_SIGNALS):
                    allow = True
        elif phase == "hold":
            allow = family == "neutral"
        if allow:
            out.append(action)
        else:
            blocked.append(
                {
                    "key": str(action.key),
                    "reason": reason,
                    "blocked_by": f"phase={phase}",
                    "family": family,
                }
            )
    return out, blocked


def _enforce_mutable_action_keys(
    *,
    actions: list[Action],
    protected_keys: set[str],
) -> tuple[list[Action], list[dict[str, Any]]]:
    out: list[Action] = []
    blocked: list[dict[str, Any]] = []
    for action in actions:
        key = str(action.key or "").strip()
        if not key:
            continue
        if key in protected_keys:
            blocked.append(
                {
                    "key": key,
                    "reason": str(action.reason or ""),
                    "blocked_by": "protected_key",
                }
            )
            continue
        if key not in TUNER_MUTABLE_KEYS:
            blocked.append(
                {
                    "key": key,
                    "reason": str(action.reason or ""),
                    "blocked_by": "mutable_whitelist",
                }
            )
            continue
        out.append(action)
    return out, blocked


def _build_rollback_actions(
    *,
    current: dict[str, str],
    stable: dict[str, str],
) -> list[Action]:
    if not stable:
        return []
    actions: list[Action] = []
    for key in sorted(TUNER_MUTABLE_KEYS):
        if key not in stable:
            continue
        cur_val = str(current.get(key, "")).strip()
        target = str(stable.get(key, "")).strip()
        if cur_val == target:
            continue
        actions.append(
            Action(
                key=key,
                old_value=cur_val,
                new_value=target,
                reason="auto_rollback_degradation",
            )
        )
    return actions

def _overrides_hash(overrides: dict[str, str]) -> str:
    normalized = {str(k): str(v) for k, v in overrides.items()}
    payload = json.dumps({k: normalized[k] for k in sorted(normalized.keys())}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _latest_jsonl_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except Exception:
            continue
        if isinstance(row, dict):
            return row
    return {}


def _git_commit_hash(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if proc.returncode == 0:
            return str(proc.stdout or "").strip()
    except Exception:
        return ""
    return ""


def _pick_run_tag(candidate_rows: list[dict[str, Any]], trade_rows: list[dict[str, Any]], profile_id: str) -> str:
    for row in reversed(trade_rows):
        tag = str(row.get("run_tag", "") or "").strip()
        if tag:
            return tag
    for row in reversed(candidate_rows):
        tag = str(row.get("run_tag", "") or "").strip()
        if tag:
            return tag
    return profile_id


def _collect_telemetry_v2(
    *,
    root: Path,
    profile_id: str,
    config_before: dict[str, str],
    config_after: dict[str, str],
    actions: list[Action],
    metrics: WindowMetrics | None = None,
) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    cutoff_15m = now_utc - timedelta(minutes=15)
    cutoff_60m = now_utc - timedelta(minutes=60)
    candidate_log = root / "logs" / "matrix" / profile_id / "candidates.jsonl"
    trade_log = root / "logs" / "matrix" / profile_id / "trade_decisions.jsonl"
    candidate_rows_15m = _read_jsonl_window(candidate_log, cutoff=cutoff_15m)
    trade_rows_15m = _read_jsonl_window(trade_log, cutoff=cutoff_15m)
    trade_rows_60m = _read_jsonl_window(trade_log, cutoff=cutoff_60m)
    last_row = _latest_jsonl_row(_runtime_log_path(root, profile_id))
    last_hash = str((last_row.get("telemetry_v2") or {}).get("config_hash_after", "") or "")
    run_tag = _pick_run_tag(candidate_rows_15m, trade_rows_15m, profile_id)
    blacklist_forensics = _blacklist_forensics(
        candidate_rows_15m=candidate_rows_15m,
        trade_rows_15m=trade_rows_15m,
        trade_rows_60m=trade_rows_60m,
        metrics=metrics,
    )
    return {
        "ts_utc": now_utc.isoformat(),
        "run_tag": run_tag,
        "commit_hash": _git_commit_hash(root),
        "log_paths": {
            "candidate_log_file": str(candidate_log),
            "trade_decisions_log_file": str(trade_log),
        },
        "funnel_15m": _funnel_15m(candidate_rows_15m, trade_rows_15m),
        "top_reasons_15m": _top_reasons_15m(candidate_rows_15m, trade_rows_15m),
        "exec_health_15m": _exec_health_15m(trade_rows_15m),
        "symbol_churn_15m": _symbol_churn_15m(trade_rows_15m),
        "exit_mix_60m": _exit_mix_60m(trade_rows_60m),
        "blacklist_forensics_15m": blacklist_forensics,
        "config_hash_before": _overrides_hash(config_before),
        "config_hash_after": _overrides_hash(config_after),
        "previous_config_hash_after": last_hash,
        "config_diff": [
            {
                "key": str(a.key),
                "old": str(a.old_value),
                "new": str(a.new_value),
                "reason": str(a.reason),
            }
            for a in actions
        ],
    }


def _load_runtime_rows(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    if limit > 0 and len(rows) > limit:
        return rows[-limit:]
    return rows


def _summarize_runtime_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    apply_states = Counter[str]()
    action_keys = Counter[str]()
    action_reasons = Counter[str]()
    rule_hits = Counter[str]()
    top_plan_skip = Counter[str]()
    policy_phases = Counter[str]()
    blocked_action_keys = Counter[str]()
    rollback_ticks = 0
    effective_ticks = 0
    blacklist_pressure_ticks = 0
    raw_values: list[float] = []
    buy_values: list[float] = []
    for row in rows:
        apply_states[str(row.get("apply_state", "") or "unknown")] += 1
        policy_phases[str(row.get("policy_phase", "") or "unknown")] += 1
        if bool(row.get("rollback_triggered")):
            rollback_ticks += 1
        t2_row = row.get("telemetry_v2") or {}
        if isinstance(t2_row, dict):
            if bool(t2_row.get("tuner_effective")):
                effective_ticks += 1
            bf = t2_row.get("blacklist_forensics_15m") or {}
            if isinstance(bf, dict):
                if _safe_float(bf.get("plan_skip_blacklist_share_15m", 0.0), 0.0) >= 0.40:
                    blacklist_pressure_ticks += 1
        for item in row.get("blocked_actions", []) or []:
            if not isinstance(item, dict):
                continue
            blocked_action_keys[str(item.get("key", "") or "unknown")] += 1
        for action in row.get("actions", []) or []:
            if not isinstance(action, dict):
                continue
            action_keys[str(action.get("key", "") or "unknown")] += 1
            action_reasons[str(action.get("reason", "") or "unknown")] += 1
        meta = row.get("decision_meta") or {}
        if isinstance(meta, dict):
            hits = meta.get("rule_hits") or {}
            if isinstance(hits, dict):
                for key, value in hits.items():
                    try:
                        rule_hits[str(key)] += int(value)
                    except Exception:
                        continue
        t2 = row.get("telemetry_v2") or {}
        if isinstance(t2, dict):
            funnel = t2.get("funnel_15m") or {}
            if isinstance(funnel, dict):
                try:
                    raw_values.append(float(funnel.get("raw", 0.0) or 0.0))
                except Exception:
                    pass
                try:
                    buy_values.append(float(funnel.get("buy", 0.0) or 0.0))
                except Exception:
                    pass
            top = t2.get("top_reasons_15m") or {}
            if isinstance(top, dict):
                for item in (top.get("plan_skip", []) or [])[:3]:
                    if not isinstance(item, dict):
                        continue
                    reason = str(item.get("reason_code", "") or "unknown")
                    try:
                        count = int(item.get("count", 0) or 0)
                    except Exception:
                        count = 0
                    top_plan_skip[reason] += count
    ticks = int(len(rows))
    action_ticks = sum(1 for row in rows if (row.get("actions") or []))
    avg_raw = (sum(raw_values) / len(raw_values)) if raw_values else 0.0
    avg_buy = (sum(buy_values) / len(buy_values)) if buy_values else 0.0
    return {
        "ticks": ticks,
        "action_ticks": int(action_ticks),
        "apply_states": _counter_top(apply_states, limit=10),
        "top_action_keys": _counter_top(action_keys, limit=10),
        "top_action_reasons": _counter_top(action_reasons, limit=10),
        "policy_phases": _counter_top(policy_phases, limit=6),
        "blocked_action_keys": _counter_top(blocked_action_keys, limit=10),
        "rule_hits": _counter_top(rule_hits, limit=10),
        "top_plan_skip_reasons": _counter_top(top_plan_skip, limit=8),
        "funnel_15m_avg_raw": round(float(avg_raw), 4),
        "funnel_15m_avg_buy": round(float(avg_buy), 4),
        "rollback_ticks": int(rollback_ticks),
        "effective_ticks": int(effective_ticks),
        "blacklist_pressure_ticks": int(blacklist_pressure_ticks),
    }


def _load_preset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_preset(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _to_bool_str(value: bool) -> str:
    return "true" if bool(value) else "false"


def _to_int_str(value: int) -> str:
    return str(int(value))


def _to_float_str(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _get_int(overrides: dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(str(overrides.get(key, default)).strip()))
    except Exception:
        return int(default)


def _get_float(overrides: dict[str, str], key: str, default: float) -> float:
    try:
        return float(str(overrides.get(key, default)).strip())
    except Exception:
        return float(default)


def _get_bool(overrides: dict[str, str], key: str, default: bool) -> bool:
    raw = str(overrides.get(key, _to_bool_str(default))).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _get_source_cap_map(overrides: dict[str, str], key: str) -> dict[str, int]:
    raw = str(overrides.get(key, "")).strip()
    out: dict[str, int] = {}
    if not raw:
        return out
    for chunk in raw.split(","):
        part = str(chunk or "").strip()
        if not part or ":" not in part:
            continue
        source_raw, value_raw = part.split(":", 1)
        source = str(source_raw or "").strip().lower()
        if not source:
            continue
        try:
            out[source] = int(float(str(value_raw or "").strip()))
        except Exception:
            continue
    return out


def _format_source_cap_map(caps: dict[str, int]) -> str:
    if not caps:
        return ""
    ordered_keys = [k for k in SOURCE_CAP_ORDER if k in caps]
    tail = sorted([k for k in caps.keys() if k not in SOURCE_CAP_ORDER])
    ordered_keys.extend(tail)
    return ",".join(f"{k}:{int(caps[k])}" for k in ordered_keys)


def _parse_symbol_csv(value: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in str(value or "").split(","):
        sym = str(raw or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _format_symbol_csv(symbols: list[str]) -> str:
    return ",".join(_parse_symbol_csv(",".join(symbols)))


def _rebalance_source_caps(
    caps: dict[str, int],
    *,
    watchlist_scale: float,
    min_value: int,
    max_value: int,
    watchlist_min: int | None = None,
) -> dict[str, int]:
    if not caps:
        return {}
    staged = {str(k).strip().lower(): int(v) for k, v in caps.items()}
    if "watchlist" not in staged:
        return staged
    old_watch = int(staged.get("watchlist", 0))
    if old_watch <= 0:
        return staged
    watch_floor = int(min_value)
    if watchlist_min is not None:
        watch_floor = max(int(min_value), int(watchlist_min))
    watch_floor = min(int(max_value), max(0, int(watch_floor)))
    scaled = int(round(float(old_watch) * float(watchlist_scale)))
    new_watch = _clamp_int(scaled, min_value, max_value)
    if watch_floor > 0:
        new_watch = max(int(new_watch), int(watch_floor))
    delta = max(0, old_watch - new_watch)
    staged["watchlist"] = new_watch
    if delta <= 0:
        return staged
    non_watch = [k for k in staged.keys() if k != "watchlist"]
    if not non_watch:
        return staged
    share = delta // len(non_watch)
    remainder = delta % len(non_watch)
    for idx, key in enumerate(sorted(non_watch)):
        inc = int(share + (1 if idx < remainder else 0))
        staged[key] = _clamp_int(int(staged.get(key, 0)) + inc, min_value, max_value)
    return staged


def _is_low_throughput(metrics: WindowMetrics) -> bool:
    selected = int(metrics.selected_from_batch)
    opened = int(metrics.opened_from_batch)
    open_rate = float(metrics.open_rate)
    if selected <= 0:
        return True
    if selected >= 20 and open_rate < 0.06:
        return True
    if selected >= 8 and opened <= 2 and open_rate < 0.10:
        return True
    if selected >= 4 and opened <= 0:
        return True
    return False


def _adaptive_bounds(
    metrics: WindowMetrics,
    mode: ModeSpec,
    *,
    allow_zero_cooldown: bool = False,
) -> AdaptiveBounds:
    bounds = AdaptiveBounds(
        strict_floor=int(mode.strict_floor),
        soft_floor=int(mode.soft_floor),
        strict_ceiling=int(mode.strict_ceiling),
        soft_ceiling=int(mode.soft_ceiling),
        volume_floor=float(mode.volume_floor),
        volume_ceiling=float(mode.volume_ceiling),
        edge_floor=float(mode.edge_floor),
        edge_ceiling=float(mode.edge_ceiling),
        edge_usd_floor=float(mode.edge_usd_floor),
        edge_usd_ceiling=float(mode.edge_usd_ceiling),
        cooldown_floor=int(mode.cooldown_floor),
        cooldown_ceiling=int(mode.cooldown_ceiling),
    )
    low_throughput = _is_low_throughput(metrics)
    if low_throughput:
        bounds.strict_floor = max(15, int(mode.strict_floor) - 10)
        bounds.soft_floor = max(10, int(mode.soft_floor) - 10)
        bounds.volume_floor = max(5.0, float(mode.volume_floor) - 3.0)
        bounds.edge_floor = max(0.05, float(mode.edge_floor) - 0.15)
        bounds.edge_usd_floor = max(0.0005, float(mode.edge_usd_floor) - 0.0015)
        if bool(allow_zero_cooldown):
            bounds.cooldown_floor = max(0, int(mode.cooldown_floor) - 10)
        else:
            bounds.cooldown_floor = max(0, int(mode.cooldown_floor))
    risk_stress = (
        int(metrics.total_closed) >= 10
        and (
            (metrics.winrate_total is not None and float(metrics.winrate_total) < 35.0)
            or (metrics.realized_total is not None and float(metrics.realized_total) < 0.0)
        )
    )
    if risk_stress and not low_throughput:
        bounds.strict_floor = max(bounds.strict_floor, int(mode.strict_floor))
        bounds.soft_floor = max(bounds.soft_floor, int(mode.soft_floor))
        bounds.edge_floor = max(bounds.edge_floor, min(bounds.edge_ceiling, float(mode.edge_floor) + 0.05))
        bounds.edge_usd_floor = max(
            bounds.edge_usd_floor,
            min(bounds.edge_usd_ceiling, float(mode.edge_usd_floor) + 0.0005),
        )
    return bounds


def _apply_action(
    overrides: dict[str, str],
    actions: list[Action],
    key: str,
    new_value: str,
    reason: str,
) -> None:
    old_value = str(overrides.get(key, "")).strip()
    new_text = str(new_value).strip()
    if old_value == new_text:
        return
    # Avoid format-only churn like "0.20" -> "0.2".
    try:
        if abs(float(old_value) - float(new_text)) < 1e-12:
            return
    except Exception:
        pass
    if old_value.lower() in {"true", "false"} and new_text.lower() in {"true", "false"}:
        if old_value.lower() == new_text.lower():
            return
    if old_value.lower() in {"1", "0"} and new_text.lower() in {"true", "false"}:
        if (old_value == "1" and new_text.lower() == "true") or (old_value == "0" and new_text.lower() == "false"):
            return
    if new_text.lower() in {"1", "0"} and old_value.lower() in {"true", "false"}:
        if (new_text == "1" and old_value.lower() == "true") or (new_text == "0" and old_value.lower() == "false"):
            return
    if old_value == str(new_value):
        return
    actions.append(Action(key=key, old_value=old_value, new_value=new_text, reason=reason))
    overrides[key] = new_text


def _coalesce_actions(actions: list[Action]) -> tuple[list[Action], int]:
    """Collapse multiple mutations of the same key into one final action per key."""
    if len(actions) <= 1:
        return list(actions), 0
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for idx, action in enumerate(actions):
        key = str(action.key or "").strip()
        if not key:
            continue
        slot = by_key.get(key)
        reason = str(action.reason or "").strip()
        if slot is None:
            by_key[key] = {
                "old": str(action.old_value),
                "new": str(action.new_value),
                "reasons": [reason] if reason else [],
                "idx": int(idx),
            }
            order.append(key)
            continue
        slot["new"] = str(action.new_value)
        slot["idx"] = int(idx)
        if reason and reason not in slot["reasons"]:
            slot["reasons"].append(reason)
    out: list[Action] = []
    # Keep relative order by last mutation index so later decisions have precedence.
    for key in sorted(order, key=lambda k: int(by_key[k]["idx"])):
        slot = by_key[key]
        old_value = str(slot.get("old", "") or "")
        new_value = str(slot.get("new", "") or "")
        if old_value == new_value:
            continue
        reasons = [str(x).strip() for x in (slot.get("reasons") or []) if str(x).strip()]
        reason_joined = " | ".join(reasons[:4]) if reasons else ""
        out.append(Action(key=key, old_value=old_value, new_value=new_value, reason=reason_joined))
    collapsed = max(0, len(actions) - len(out))
    return out, int(collapsed)


def _action_priority(action: Action) -> int:
    key = str(action.key or "").strip().upper()
    reason = str(action.reason or "").strip().lower()
    high = {
        "HEAVY_CHECK_DEDUP_TTL_SECONDS",
        "MAX_TOKEN_COOLDOWN_SECONDS",
        "MIN_TRADE_USD",
        "PAPER_TRADE_SIZE_MIN_USD",
        "PAPER_TRADE_SIZE_MAX_USD",
        "ENTRY_A_CORE_MIN_TRADE_USD",
        "MARKET_MODE_STRICT_SCORE",
        "MARKET_MODE_SOFT_SCORE",
        "SAFE_MIN_VOLUME_5M_USD",
        "MIN_EXPECTED_EDGE_PERCENT",
        "MIN_EXPECTED_EDGE_USD",
        "AUTO_TRADE_TOP_N",
        "AUTO_TRADE_EXCLUDED_SYMBOLS",
        "SOURCE_ROUTER_MIN_TRADES",
        "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY",
        "SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY",
        "TOKEN_EV_MEMORY_MIN_TRADES",
        "TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY",
        "TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY",
        "TOKEN_AGE_MAX",
        "SEEN_TOKEN_TTL",
        "WATCHLIST_REFRESH_SECONDS",
        "WATCHLIST_MAX_TOKENS",
        "V2_CALIBRATION_ENABLED",
    }
    medium = {
        "PLAN_MAX_WATCHLIST_SHARE",
        "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
        "PLAN_MAX_SINGLE_SOURCE_SHARE",
        "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
        "V2_SOURCE_QOS_TOPK_PER_CYCLE",
        "V2_UNIVERSE_NOVELTY_MIN_SHARE",
        "SYMBOL_CONCENTRATION_MAX_SHARE",
        "V2_QUALITY_SYMBOL_MAX_SHARE",
        "V2_QUALITY_SYMBOL_MIN_ABS_CAP",
        "V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS",
        "WATCHLIST_MIN_LIQUIDITY_USD",
        "WATCHLIST_MIN_VOLUME_24H_USD",
        "PAPER_WATCHLIST_MIN_SCORE",
        "PAPER_WATCHLIST_MIN_LIQUIDITY_USD",
        "PAPER_WATCHLIST_MIN_VOLUME_5M_USD",
        "V2_QUALITY_SOURCE_BUDGET_ENABLED",
        "V2_ROLLING_EDGE_MIN_USD",
        "V2_ROLLING_EDGE_MIN_PERCENT",
        "V2_CALIBRATION_EDGE_USD_MIN",
        "V2_CALIBRATION_VOLUME_MIN",
        "V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW",
        "V2_SOURCE_QOS_SOURCE_CAPS",
        "V2_UNIVERSE_SOURCE_CAPS",
    }
    low = set()
    if key in high:
        return 0
    if key in medium:
        return 1
    if key in low:
        return 2
    if "recover_tighten" in reason:
        return 3
    return 4


def _build_actions(
    *,
    metrics: WindowMetrics,
    overrides: dict[str, str],
    mode: ModeSpec,
    telemetry: dict[str, Any] | None = None,
    allow_zero_cooldown: bool = False,
) -> list[Action]:
    actions, _, _ = _build_action_plan(
        metrics=metrics,
        overrides=overrides,
        mode=mode,
        telemetry=telemetry,
        allow_zero_cooldown=allow_zero_cooldown,
    )
    return actions


def _build_action_plan(
    *,
    metrics: WindowMetrics,
    overrides: dict[str, str],
    mode: ModeSpec,
    telemetry: dict[str, Any] | None = None,
    runtime_state: RuntimeState | None = None,
    now_ts: float | None = None,
    allow_zero_cooldown: bool = False,
) -> tuple[list[Action], list[str], dict[str, Any]]:
    staged = dict(overrides)
    actions: list[Action] = []
    trace: list[str] = []
    rule_hits: Counter[str] = Counter()
    bounds = _adaptive_bounds(metrics, mode, allow_zero_cooldown=allow_zero_cooldown)
    tick_ts = float(now_ts) if now_ts is not None else float(time.time())

    strict = _get_int(staged, "MARKET_MODE_STRICT_SCORE", 60)
    soft = _get_int(staged, "MARKET_MODE_SOFT_SCORE", 52)
    min_vol = _get_float(staged, "SAFE_MIN_VOLUME_5M_USD", 20.0)
    min_edge_pct = _get_float(staged, "MIN_EXPECTED_EDGE_PERCENT", 0.8)
    min_edge_usd = _get_float(staged, "MIN_EXPECTED_EDGE_USD", 0.010)
    ev_first_min_net_usd = _get_float(staged, "EV_FIRST_ENTRY_MIN_NET_USD", 0.0016)
    ev_probe_tolerance_usd = _get_float(staged, "EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD", 0.0030)
    cooldown = _get_int(staged, "MAX_TOKEN_COOLDOWN_SECONDS", 90)
    trade_min_paper = _get_float(staged, "PAPER_TRADE_SIZE_MIN_USD", 0.45)
    trade_min_gate = _get_float(staged, "MIN_TRADE_USD", trade_min_paper)
    trade_max_paper = _get_float(staged, "PAPER_TRADE_SIZE_MAX_USD", 1.0)
    trade_min_a_core = _get_float(staged, "ENTRY_A_CORE_MIN_TRADE_USD", 0.45)
    pe_enabled = _get_bool(staged, "PROFIT_ENGINE_ENABLED", True)
    novelty_share = _get_float(staged, "V2_UNIVERSE_NOVELTY_MIN_SHARE", 0.20)
    symbol_share_cap = _get_float(staged, "SYMBOL_CONCENTRATION_MAX_SHARE", 0.35)
    per_symbol_cycle_cap = _get_int(staged, "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE", 3)
    quality_symbol_max_share = _get_float(staged, "V2_QUALITY_SYMBOL_MAX_SHARE", 0.10)
    quality_symbol_min_abs_cap = _get_int(staged, "V2_QUALITY_SYMBOL_MIN_ABS_CAP", 2)
    quality_symbol_reentry_min_seconds = _get_int(staged, "V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS", 240)
    dedup_ttl_seconds = _get_int(staged, "HEAVY_CHECK_DEDUP_TTL_SECONDS", 20)
    top_n = _get_int(staged, "AUTO_TRADE_TOP_N", 40)
    plan_watchlist_share = _get_float(staged, "PLAN_MAX_WATCHLIST_SHARE", 0.30)
    plan_min_non_watchlist = _get_int(staged, "PLAN_MIN_NON_WATCHLIST_PER_BATCH", 1)
    plan_single_source_share = _get_float(staged, "PLAN_MAX_SINGLE_SOURCE_SHARE", 0.50)
    qos_caps = _get_source_cap_map(staged, "V2_SOURCE_QOS_SOURCE_CAPS")
    universe_caps = _get_source_cap_map(staged, "V2_UNIVERSE_SOURCE_CAPS")
    source_router_min_trades = _get_int(staged, "SOURCE_ROUTER_MIN_TRADES", 8)
    source_router_bad_prob = _get_float(staged, "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY", 0.55)
    source_router_severe_prob = _get_float(staged, "SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY", 0.35)
    token_ev_min_trades = _get_int(staged, "TOKEN_EV_MEMORY_MIN_TRADES", 4)
    token_ev_bad_prob = _get_float(staged, "TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY", 0.72)
    token_ev_severe_prob = _get_float(staged, "TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY", 0.42)
    rolling_edge_min_pct = _get_float(staged, "V2_ROLLING_EDGE_MIN_PERCENT", 0.35)
    rolling_edge_min_usd = _get_float(staged, "V2_ROLLING_EDGE_MIN_USD", 0.008)
    calibration_enabled = _get_bool(staged, "V2_CALIBRATION_ENABLED", True)
    calibration_no_tighten = _get_bool(staged, "V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW", True)
    calibration_edge_usd_min = _get_float(staged, "V2_CALIBRATION_EDGE_USD_MIN", 0.010)
    calibration_volume_min = _get_float(staged, "V2_CALIBRATION_VOLUME_MIN", 20.0)
    source_qos_topk_per_cycle = _get_int(staged, "V2_SOURCE_QOS_TOPK_PER_CYCLE", 220)
    token_age_max = _get_int(staged, "TOKEN_AGE_MAX", 3600)
    seen_token_ttl = _get_int(staged, "SEEN_TOKEN_TTL", 21600)
    watchlist_refresh_seconds = _get_int(staged, "WATCHLIST_REFRESH_SECONDS", 3600)
    watchlist_max_tokens = _get_int(staged, "WATCHLIST_MAX_TOKENS", 30)
    watchlist_min_liquidity = _get_float(staged, "WATCHLIST_MIN_LIQUIDITY_USD", 200000.0)
    watchlist_min_volume_24h = _get_float(staged, "WATCHLIST_MIN_VOLUME_24H_USD", 500000.0)
    paper_watchlist_min_score = _get_int(staged, "PAPER_WATCHLIST_MIN_SCORE", 90)
    paper_watchlist_min_liquidity = _get_float(staged, "PAPER_WATCHLIST_MIN_LIQUIDITY_USD", 150000.0)
    paper_watchlist_min_volume_5m = _get_float(staged, "PAPER_WATCHLIST_MIN_VOLUME_5M_USD", 500.0)
    quality_source_budget_enabled = _get_bool(staged, "V2_QUALITY_SOURCE_BUDGET_ENABLED", True)
    excluded_symbols = _parse_symbol_csv(staged.get("AUTO_TRADE_EXCLUDED_SYMBOLS", ""))
    excluded_symbol_set = {str(x).strip().upper() for x in excluded_symbols}

    score_min_hits = int(metrics.filter_fail_reasons.get("score_min", 0))
    safe_volume_hits = int(metrics.filter_fail_reasons.get("safe_volume", 0))
    safe_age_hits = int(metrics.filter_fail_reasons.get("safe_age", 0))
    heavy_dedup_hits = int(metrics.filter_fail_reasons.get("heavy_dedup_ttl", 0))
    ev_low_hits = int(metrics.autotrade_skip_reasons.get("ev_net_low", 0))
    edge_low_hits = int(metrics.autotrade_skip_reasons.get("edge_low", 0))
    edge_usd_low_hits = int(metrics.autotrade_skip_reasons.get("edge_usd_low", 0))
    negative_edge_hits = int(metrics.autotrade_skip_reasons.get("negative_edge", 0))
    min_trade_hits = int(metrics.autotrade_skip_reasons.get("min_trade_size", 0))
    symbol_concentration_skip_hits = int(metrics.autotrade_skip_reasons.get("symbol_concentration", 0))
    source_route_prob_hits = int(metrics.autotrade_skip_reasons.get("source_route_prob", 0))
    token_ev_prob_hits = int(metrics.autotrade_skip_reasons.get("token_ev_memory_prob", 0))
    blacklist_hits = int(metrics.autotrade_skip_reasons.get("blacklist", 0))
    address_duplicate_hits = int(metrics.autotrade_skip_reasons.get("address_or_duplicate", 0))
    excluded_symbol_hits = int(metrics.autotrade_skip_reasons.get("excluded_symbol", 0))
    cooldown_hits = int(metrics.autotrade_skip_reasons.get("cooldown", 0))
    cooldown_hits += int(metrics.autotrade_skip_reasons.get("symbol_cooldown", 0))
    cooldown_hits += sum(
        v
        for k, v in metrics.autotrade_skip_reasons.items()
        if k.startswith("cooldown_left_") or k.startswith("symbol_cooldown_left_")
    )
    cold_start_hits = int(metrics.pe_reasons.get("cold_start", 0))
    buy_top_share = float(metrics.top_buy_symbol_share)
    buy_unique = int(metrics.unique_buy_symbols)
    buy_total = int(metrics.autobuy_total)
    concentration_drop_hits = int(metrics.symbol_concentration_drop_hits)
    open_rate = float(metrics.open_rate)
    low_throughput = _is_low_throughput(metrics)
    # Flow guardrails: prevent tuner from self-choking entry funnel on weak windows.
    flow_watchlist_share_floor = 0.08 if low_throughput else 0.02
    flow_watchlist_share_ceiling = 0.60
    flow_non_watchlist_ceiling = 3 if low_throughput else 6
    cooldown_dominant_early = bool(
        low_throughput and cooldown_hits >= max(12, int(metrics.selected_from_batch) // 2)
    )
    flow_per_symbol_cap_floor = 1
    flow_qos_watchlist_cap_floor = 8 if low_throughput else 2
    flow_universe_watchlist_cap_floor = 4 if low_throughput else 2
    source_qos_cap_hits = int(metrics.filter_fail_reasons.get("source_qos_cap", 0))
    source_qos_symbol_cap_hits = int(metrics.filter_fail_reasons.get("source_qos_symbol_cap", 0))
    edge_pressure_hits = int(ev_low_hits + edge_low_hits + edge_usd_low_hits + negative_edge_hits)

    def _reason_count(rows: list[dict[str, Any]], reason_code: str) -> int:
        target = str(reason_code or "").strip().lower()
        if not target:
            return 0
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            code = str(row.get("reason_code", "") or "").strip().lower()
            if code != target:
                continue
            try:
                return int(row.get("count", 0) or 0)
            except Exception:
                return 0
        return 0

    quality_skip_rows: list[dict[str, Any]] = []
    plan_skip_rows: list[dict[str, Any]] = []
    funnel_15m: dict[str, Any] = {}
    churn_15m: dict[str, Any] = {}
    if isinstance(telemetry, dict):
        top_reasons = telemetry.get("top_reasons_15m", {}) or {}
        if isinstance(top_reasons, dict):
            qrows = top_reasons.get("quality_skip", []) or []
            prows = top_reasons.get("plan_skip", []) or []
            if isinstance(qrows, list):
                quality_skip_rows = [x for x in qrows if isinstance(x, dict)]
            if isinstance(prows, list):
                plan_skip_rows = [x for x in prows if isinstance(x, dict)]
        funnel_raw = telemetry.get("funnel_15m", {}) or {}
        if isinstance(funnel_raw, dict):
            funnel_15m = funnel_raw
        churn_raw = telemetry.get("symbol_churn_15m", {}) or {}
        if isinstance(churn_raw, dict):
            churn_15m = churn_raw

    quality_skip_total = 0
    for row in quality_skip_rows:
        try:
            quality_skip_total += max(0, int(row.get("count", 0) or 0))
        except Exception:
            continue
    quality_duplicate_hits = _reason_count(quality_skip_rows, "duplicate_address")
    quality_source_budget_hits = _reason_count(quality_skip_rows, "source_budget")
    quality_symbol_concentration_hits = _reason_count(quality_skip_rows, "symbol_concentration")
    plan_ev_low_hits = _reason_count(plan_skip_rows, "ev_net_low")
    plan_excluded_hits = _reason_count(plan_skip_rows, "excluded_symbol")
    funnel_raw_count = int(_safe_float((funnel_15m or {}).get("raw", 0), 0.0))
    funnel_pre_count = int(_safe_float((funnel_15m or {}).get("pre", 0), 0.0))
    funnel_buy_count = int(_safe_float((funnel_15m or {}).get("buy", 0), 0.0))
    churn_detected = bool((churn_15m or {}).get("detected", False))
    churn_symbol = str((churn_15m or {}).get("symbol", "") or "").strip().upper()
    churn_open_share = _safe_float((churn_15m or {}).get("open_share", 0.0), 0.0)
    churn_flat_share = _safe_float((churn_15m or {}).get("flat_close_share", 0.0), 0.0)
    churn_ttl_seconds = _clamp_int(int(_safe_float((churn_15m or {}).get("ttl_seconds", 0), 0.0)), 120, 3600)
    churn_lock_active = False
    churn_lock_symbol = ""
    churn_lock_remaining = 0

    address_duplicate_hits = max(address_duplicate_hits, quality_duplicate_hits)
    source_route_prob_hits = max(source_route_prob_hits, int(quality_source_budget_hits // 2))
    symbol_concentration_skip_hits = max(symbol_concentration_skip_hits, quality_symbol_concentration_hits)
    concentration_drop_hits = max(concentration_drop_hits, quality_symbol_concentration_hits)
    ev_low_hits = max(ev_low_hits, plan_ev_low_hits)
    excluded_symbol_hits = max(excluded_symbol_hits, plan_excluded_hits)
    quality_source_budget_share = (
        float(quality_source_budget_hits) / float(max(1, quality_skip_total))
        if quality_skip_total > 0
        else 0.0
    )

    if mode.name in {"fast", "conveyor"} and cold_start_hits > 0 and pe_enabled:
        pe_enabled = False
        rule_hits["pe_cold_start_guard"] += 1
        trace.append(f"pe_cold_start_guard cold_start={cold_start_hits}")
        _apply_action(
            staged,
            actions,
            "PROFIT_ENGINE_ENABLED",
            _to_bool_str(pe_enabled),
            "pe_cold_start_detected",
        )

    if metrics.selected_from_batch <= 0 and metrics.scanned <= 0 and metrics.lines_seen <= 0:
        trace.append("flow_guard no_metrics")
        limited = actions[: mode.max_actions_per_tick]
        return limited, trace, {"rule_hits": dict(rule_hits)}

    route_pressure = source_route_prob_hits >= max(4, int(metrics.opened_from_batch) + 2)
    normalized_route = False
    normalized_source_caps = False
    if plan_watchlist_share < flow_watchlist_share_floor:
        plan_watchlist_share = flow_watchlist_share_floor
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_WATCHLIST_SHARE",
            _to_float_str(plan_watchlist_share),
            "route_floor_normalize watchlist_share",
        )
        normalized_route = True
    if plan_min_non_watchlist > flow_non_watchlist_ceiling:
        plan_min_non_watchlist = flow_non_watchlist_ceiling
        _apply_action(
            staged,
            actions,
            "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
            _to_int_str(plan_min_non_watchlist),
            "route_floor_normalize non_watch_min",
        )
        normalized_route = True
    if plan_single_source_share < 0.20:
        plan_single_source_share = 0.20
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_SINGLE_SOURCE_SHARE",
            _to_float_str(plan_single_source_share),
            "route_floor_normalize single_source_share",
        )
        normalized_route = True
    if qos_caps:
        watch_cap_qos = int(qos_caps.get("watchlist", 0) or 0)
        if watch_cap_qos > 0 and watch_cap_qos < flow_qos_watchlist_cap_floor:
            qos_caps["watchlist"] = int(flow_qos_watchlist_cap_floor)
            _apply_action(
                staged,
                actions,
                "V2_SOURCE_QOS_SOURCE_CAPS",
                _format_source_cap_map(qos_caps),
                "route_floor_normalize source_qos_watchlist_cap",
            )
            normalized_source_caps = True
    if universe_caps:
        watch_cap_universe = int(universe_caps.get("watchlist", 0) or 0)
        if watch_cap_universe > 0 and watch_cap_universe < flow_universe_watchlist_cap_floor:
            universe_caps["watchlist"] = int(flow_universe_watchlist_cap_floor)
            _apply_action(
                staged,
                actions,
                "V2_UNIVERSE_SOURCE_CAPS",
                _format_source_cap_map(universe_caps),
                "route_floor_normalize universe_watchlist_cap",
            )
            normalized_source_caps = True
    if normalized_route:
        rule_hits["route_floor_normalize"] += 1
    if normalized_source_caps:
        rule_hits["source_cap_floor_normalize"] += 1

    active_lock_until = 0.0
    active_lock_symbol = ""
    if runtime_state is not None:
        active_lock_symbol = str(runtime_state.churn_lock_symbol or "").strip().upper()
        active_lock_until = float(runtime_state.churn_lock_until_ts or 0.0)
        if active_lock_symbol and tick_ts >= active_lock_until:
            if active_lock_symbol in excluded_symbol_set:
                excluded_symbols = [sym for sym in excluded_symbols if sym != active_lock_symbol]
                excluded_symbol_set.discard(active_lock_symbol)
                _apply_action(
                    staged,
                    actions,
                    "AUTO_TRADE_EXCLUDED_SYMBOLS",
                    _format_symbol_csv(excluded_symbols),
                    "churn_lock_release ttl_expired",
                )
            rule_hits["churn_lock_release"] += 1
            trace.append(f"churn_lock_release symbol={active_lock_symbol} reason=ttl_expired")
            runtime_state.churn_lock_symbol = ""
            runtime_state.churn_lock_until_ts = 0.0
            runtime_state.churn_lock_last_reason = "ttl_expired"
            active_lock_symbol = ""
            active_lock_until = 0.0
        if churn_detected and churn_symbol:
            ttl_seconds = max(600, int(churn_ttl_seconds or 0))
            lock_until = tick_ts + float(ttl_seconds)
            if active_lock_symbol == churn_symbol:
                active_lock_until = max(float(active_lock_until), float(lock_until))
            else:
                active_lock_symbol = churn_symbol
                active_lock_until = float(lock_until)
            runtime_state.churn_lock_symbol = str(churn_symbol)
            runtime_state.churn_lock_until_ts = float(active_lock_until)
            runtime_state.churn_lock_last_reason = (
                f"open_share={churn_open_share:.2f} flat_share={churn_flat_share:.2f}"
            )
            rule_hits["churn_lock_activate"] += 1
        release_ready = (
            bool(active_lock_symbol)
            and int(metrics.opened_from_batch) >= 3
            and int(buy_unique) >= 3
            and float(buy_top_share) <= 0.55
            and (not churn_detected)
        )
        if release_ready:
            if active_lock_symbol in excluded_symbol_set:
                excluded_symbols = [sym for sym in excluded_symbols if sym != active_lock_symbol]
                excluded_symbol_set.discard(active_lock_symbol)
                _apply_action(
                    staged,
                    actions,
                    "AUTO_TRADE_EXCLUDED_SYMBOLS",
                    _format_symbol_csv(excluded_symbols),
                    "churn_lock_release diversity_recovered",
                )
            rule_hits["churn_lock_release"] += 1
            trace.append(
                f"churn_lock_release symbol={active_lock_symbol} reason=diversity_recovered uniq={buy_unique} top={buy_top_share:.2f}"
            )
            runtime_state.churn_lock_symbol = ""
            runtime_state.churn_lock_until_ts = 0.0
            runtime_state.churn_lock_last_reason = "diversity_recovered"
            active_lock_symbol = ""
            active_lock_until = 0.0
    if runtime_state is None and churn_detected and churn_symbol:
        active_lock_symbol = churn_symbol
        active_lock_until = tick_ts + float(max(600, int(churn_ttl_seconds or 0)))

    if active_lock_symbol and tick_ts < active_lock_until:
        churn_lock_active = True
        churn_lock_symbol = str(active_lock_symbol)
        churn_lock_remaining = int(max(0.0, float(active_lock_until) - float(tick_ts)))
        if churn_lock_symbol not in excluded_symbol_set:
            excluded_symbols.append(churn_lock_symbol)
            excluded_symbol_set.add(churn_lock_symbol)
            _apply_action(
                staged,
                actions,
                "AUTO_TRADE_EXCLUDED_SYMBOLS",
                _format_symbol_csv(excluded_symbols),
                f"churn_lock exclude_symbol={churn_lock_symbol}",
            )
        rule_hits["churn_lock_active"] += 1
        trace.append(
            "churn_lock "
            f"symbol={churn_lock_symbol} ttl={churn_lock_remaining}s open_share={churn_open_share:.2f} flat_share={churn_flat_share:.2f}"
        )
        novelty_floor = 0.34 if churn_detected else 0.26
        symbol_cap_ceiling = 0.26 if churn_detected else 0.30
        per_symbol_cap_ceiling = 2 if churn_detected else 3
        quality_reentry_floor = 360 if churn_detected else 300
        quality_share_ceiling = 0.12 if churn_detected else 0.14
        cooldown_floor = 120 if churn_detected else 90
        token_ev_min_trades_floor = 6 if churn_detected else 5
        token_ev_bad_prob_ceiling = 0.60 if churn_detected else 0.65
        token_ev_severe_prob_ceiling = 0.32 if churn_detected else 0.36
        source_router_bad_prob_ceiling = 0.48 if churn_detected else 0.52
        source_router_severe_prob_ceiling = 0.28 if churn_detected else 0.32
        novelty_share = _clamp_float(max(novelty_share, novelty_floor), 0.05, 0.90)
        symbol_share_cap = _clamp_float(min(symbol_share_cap, symbol_cap_ceiling), 0.05, 0.50)
        per_symbol_cycle_cap = _clamp_int(min(per_symbol_cycle_cap, per_symbol_cap_ceiling), 1, 20)
        quality_symbol_reentry_min_seconds = _clamp_int(
            max(quality_symbol_reentry_min_seconds, quality_reentry_floor),
            0,
            3600,
        )
        quality_symbol_max_share = _clamp_float(min(quality_symbol_max_share, quality_share_ceiling), 0.05, 0.50)
        cooldown = _clamp_int(max(cooldown, cooldown_floor), bounds.cooldown_floor, bounds.cooldown_ceiling)
        plan_min_non_watchlist = _clamp_int(max(plan_min_non_watchlist, 2), 0, 8)
        plan_single_source_share = _clamp_float(min(plan_single_source_share, 0.45), 0.20, 1.0)
        token_ev_min_trades = _clamp_int(max(token_ev_min_trades, token_ev_min_trades_floor), 2, 24)
        token_ev_bad_prob = _clamp_float(min(token_ev_bad_prob, token_ev_bad_prob_ceiling), 0.35, 0.95)
        token_ev_severe_prob = _clamp_float(min(token_ev_severe_prob, token_ev_severe_prob_ceiling), 0.20, 0.90)
        source_router_bad_prob = _clamp_float(min(source_router_bad_prob, source_router_bad_prob_ceiling), 0.35, 0.95)
        source_router_severe_prob = _clamp_float(
            min(source_router_severe_prob, source_router_severe_prob_ceiling),
            0.20,
            0.90,
        )
        _apply_action(
            staged,
            actions,
            "V2_UNIVERSE_NOVELTY_MIN_SHARE",
            _to_float_str(novelty_share),
            f"churn_lock diversity_reserve symbol={churn_lock_symbol}",
        )
        _apply_action(
            staged,
            actions,
            "SYMBOL_CONCENTRATION_MAX_SHARE",
            _to_float_str(symbol_share_cap),
            f"churn_lock symbol_share symbol={churn_lock_symbol}",
        )
        _apply_action(
            staged,
            actions,
            "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
            _to_int_str(per_symbol_cycle_cap),
            "churn_lock per_symbol_cap",
        )
        _apply_action(
            staged,
            actions,
            "V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS",
            _to_int_str(quality_symbol_reentry_min_seconds),
            "churn_lock quality_reentry",
        )
        _apply_action(
            staged,
            actions,
            "V2_QUALITY_SYMBOL_MAX_SHARE",
            _to_float_str(quality_symbol_max_share),
            "churn_lock quality_share",
        )
        _apply_action(
            staged,
            actions,
            "MAX_TOKEN_COOLDOWN_SECONDS",
            _to_int_str(cooldown),
            "churn_lock cooldown",
        )
        _apply_action(
            staged,
            actions,
            "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
            _to_int_str(plan_min_non_watchlist),
            "churn_lock diversity_non_watch_min",
        )
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_SINGLE_SOURCE_SHARE",
            _to_float_str(plan_single_source_share),
            "churn_lock diversity_single_source_share",
        )
        _apply_action(
            staged,
            actions,
            "TOKEN_EV_MEMORY_MIN_TRADES",
            _to_int_str(token_ev_min_trades),
            "churn_lock token_ev_min_trades",
        )
        _apply_action(
            staged,
            actions,
            "TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY",
            _to_float_str(token_ev_bad_prob),
            "churn_lock token_ev_bad_prob",
        )
        _apply_action(
            staged,
            actions,
            "TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY",
            _to_float_str(token_ev_severe_prob),
            "churn_lock token_ev_severe_prob",
        )
        _apply_action(
            staged,
            actions,
            "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY",
            _to_float_str(source_router_bad_prob),
            "churn_lock source_router_bad_prob",
        )
        _apply_action(
            staged,
            actions,
            "SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY",
            _to_float_str(source_router_severe_prob),
            "churn_lock source_router_severe_prob",
        )

    if route_pressure:
        plan_watchlist_share = _clamp_float(
            plan_watchlist_share - 0.05,
            flow_watchlist_share_floor,
            flow_watchlist_share_ceiling,
        )
        plan_min_non_watchlist = _clamp_int(plan_min_non_watchlist + 1, 0, flow_non_watchlist_ceiling)
        plan_single_source_share = _clamp_float(plan_single_source_share - 0.04, 0.20, 1.0)
        rule_hits["route_pressure_rebalance"] += 1
        trace.append(f"route_pressure source_route_prob={source_route_prob_hits} open_rate={open_rate:.3f}")
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_WATCHLIST_SHARE",
            _to_float_str(plan_watchlist_share),
            "route_pressure watchlist_share",
        )
        _apply_action(
            staged,
            actions,
            "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
            _to_int_str(plan_min_non_watchlist),
            "route_pressure non_watch_min",
        )
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_SINGLE_SOURCE_SHARE",
            _to_float_str(plan_single_source_share),
            "route_pressure single_source_share",
        )
        source_router_min_trades = _clamp_int(source_router_min_trades + 2, 4, 20)
        source_router_bad_prob = _clamp_float(source_router_bad_prob + 0.05, 0.35, 0.95)
        source_router_severe_prob = _clamp_float(source_router_severe_prob + 0.07, 0.20, 0.90)
        _apply_action(
            staged,
            actions,
            "SOURCE_ROUTER_MIN_TRADES",
            _to_int_str(source_router_min_trades),
            "route_pressure source_router_min_trades",
        )
        _apply_action(
            staged,
            actions,
            "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY",
            _to_float_str(source_router_bad_prob),
            "route_pressure source_router_bad_prob",
        )
        _apply_action(
            staged,
            actions,
            "SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY",
            _to_float_str(source_router_severe_prob),
            "route_pressure source_router_severe_prob",
        )
        if token_ev_prob_hits >= 3:
            token_ev_min_trades = _clamp_int(token_ev_min_trades + 2, 2, 18)
            token_ev_bad_prob = _clamp_float(token_ev_bad_prob + 0.05, 0.35, 0.95)
            token_ev_severe_prob = _clamp_float(token_ev_severe_prob + 0.07, 0.20, 0.90)
            _apply_action(
                staged,
                actions,
                "TOKEN_EV_MEMORY_MIN_TRADES",
                _to_int_str(token_ev_min_trades),
                "route_pressure token_ev_min_trades",
            )
            _apply_action(
                staged,
                actions,
                "TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY",
                _to_float_str(token_ev_bad_prob),
                "route_pressure token_ev_bad_prob",
            )
            _apply_action(
                staged,
                actions,
                "TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY",
                _to_float_str(token_ev_severe_prob),
                "route_pressure token_ev_severe_prob",
            )
        if qos_caps:
            qos_caps = _rebalance_source_caps(
                qos_caps,
                watchlist_scale=0.80,
                min_value=1,
                max_value=600,
                watchlist_min=2,
            )
            _apply_action(
                staged,
                actions,
                "V2_SOURCE_QOS_SOURCE_CAPS",
                _format_source_cap_map(qos_caps),
                "route_pressure source_qos_caps",
            )
        if universe_caps and source_route_prob_hits >= 12:
            universe_caps = _rebalance_source_caps(
                universe_caps,
                watchlist_scale=0.80,
                min_value=1,
                max_value=600,
                watchlist_min=2,
            )
            _apply_action(
                staged,
                actions,
                "V2_UNIVERSE_SOURCE_CAPS",
                _format_source_cap_map(universe_caps),
                "route_pressure universe_caps",
            )

    quality_source_budget_choke = (
        low_throughput
        and quality_source_budget_hits >= max(24, int(metrics.selected_from_batch) // 5)
        and quality_source_budget_share >= 0.22
    )
    if quality_source_budget_choke:
        rule_hits["quality_source_budget_choke"] += 1
        trace.append(
            "quality_source_budget_choke "
            f"hits={quality_source_budget_hits} share={quality_source_budget_share:.2f}"
        )
        if quality_source_budget_enabled:
            quality_source_budget_enabled = False
            _apply_action(
                staged,
                actions,
                "V2_QUALITY_SOURCE_BUDGET_ENABLED",
                _to_bool_str(False),
                "quality_source_budget_choke disable_source_budget",
            )
        plan_watchlist_share = _clamp_float(
            plan_watchlist_share - 0.03,
            flow_watchlist_share_floor,
            flow_watchlist_share_ceiling,
        )
        plan_min_non_watchlist = _clamp_int(plan_min_non_watchlist + 1, 0, flow_non_watchlist_ceiling)
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_WATCHLIST_SHARE",
            _to_float_str(plan_watchlist_share),
            "quality_source_budget_choke watchlist_share",
        )
        _apply_action(
            staged,
            actions,
            "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
            _to_int_str(plan_min_non_watchlist),
            "quality_source_budget_choke non_watch_min",
        )

    excluded_rotation_needed = (
        low_throughput
        and excluded_symbol_hits >= max(8, int(metrics.selected_from_batch) // 6)
    )
    if excluded_rotation_needed:
        rule_hits["excluded_symbol_rotation"] += 1
        trace.append(
            "excluded_symbol_rotation "
            f"hits={excluded_symbol_hits} selected={metrics.selected_from_batch} buy15m={funnel_buy_count}"
        )
        plan_watchlist_share = _clamp_float(
            plan_watchlist_share - 0.04,
            flow_watchlist_share_floor,
            flow_watchlist_share_ceiling,
        )
        plan_min_non_watchlist = _clamp_int(plan_min_non_watchlist + 1, 0, flow_non_watchlist_ceiling)
        watchlist_refresh_seconds = _clamp_int(watchlist_refresh_seconds - 120, 60, 3600)
        watchlist_max_tokens = _clamp_int(watchlist_max_tokens + 8, 5, 120)
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_WATCHLIST_SHARE",
            _to_float_str(plan_watchlist_share),
            "excluded_symbol_rotation watchlist_share",
        )
        _apply_action(
            staged,
            actions,
            "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
            _to_int_str(plan_min_non_watchlist),
            "excluded_symbol_rotation non_watch_min",
        )
        _apply_action(
            staged,
            actions,
            "WATCHLIST_REFRESH_SECONDS",
            _to_int_str(watchlist_refresh_seconds),
            "excluded_symbol_rotation refresh",
        )
        _apply_action(
            staged,
            actions,
            "WATCHLIST_MAX_TOKENS",
            _to_int_str(watchlist_max_tokens),
            "excluded_symbol_rotation max_tokens",
        )

    feed_starvation = (
        low_throughput
        and funnel_raw_count > 0
        and funnel_pre_count > 0
        and funnel_raw_count < 260
        and (funnel_buy_count <= 1 or open_rate < 0.03)
    )
    if feed_starvation:
        rule_hits["feed_starvation"] += 1
        trace.append(
            "feed_starvation "
            f"raw15={funnel_raw_count} pre15={funnel_pre_count} buy15={funnel_buy_count} open_rate={open_rate:.3f}"
        )
        token_age_max = _clamp_int(token_age_max + 900, 1800, 21600)
        seen_token_ttl = _clamp_int(seen_token_ttl - 900, 300, 21600)
        watchlist_refresh_seconds = _clamp_int(watchlist_refresh_seconds - 120, 60, 3600)
        watchlist_max_tokens = _clamp_int(watchlist_max_tokens + 10, 5, 120)
        watchlist_min_liquidity = _clamp_float(watchlist_min_liquidity - 20000.0, 5000.0, 1000000.0)
        watchlist_min_volume_24h = _clamp_float(watchlist_min_volume_24h - 60000.0, 10000.0, 5000000.0)
        paper_watchlist_min_liquidity = _clamp_float(paper_watchlist_min_liquidity - 12000.0, 10000.0, 500000.0)
        paper_watchlist_min_volume_5m = _clamp_float(paper_watchlist_min_volume_5m - 40.0, 10.0, 20000.0)
        paper_watchlist_min_score = _clamp_int(paper_watchlist_min_score - 2, 70, 98)
        _apply_action(staged, actions, "TOKEN_AGE_MAX", _to_int_str(token_age_max), "feed_starvation token_age_max")
        _apply_action(staged, actions, "SEEN_TOKEN_TTL", _to_int_str(seen_token_ttl), "feed_starvation seen_token_ttl")
        _apply_action(
            staged,
            actions,
            "WATCHLIST_REFRESH_SECONDS",
            _to_int_str(watchlist_refresh_seconds),
            "feed_starvation watch_refresh",
        )
        _apply_action(staged, actions, "WATCHLIST_MAX_TOKENS", _to_int_str(watchlist_max_tokens), "feed_starvation watch_max")
        _apply_action(
            staged,
            actions,
            "WATCHLIST_MIN_LIQUIDITY_USD",
            _to_float_str(watchlist_min_liquidity),
            "feed_starvation watch_min_liq",
        )
        _apply_action(
            staged,
            actions,
            "WATCHLIST_MIN_VOLUME_24H_USD",
            _to_float_str(watchlist_min_volume_24h),
            "feed_starvation watch_min_vol24h",
        )
        _apply_action(
            staged,
            actions,
            "PAPER_WATCHLIST_MIN_LIQUIDITY_USD",
            _to_float_str(paper_watchlist_min_liquidity),
            "feed_starvation paper_watch_min_liq",
        )
        _apply_action(
            staged,
            actions,
            "PAPER_WATCHLIST_MIN_VOLUME_5M_USD",
            _to_float_str(paper_watchlist_min_volume_5m),
            "feed_starvation paper_watch_min_vol5m",
        )
        _apply_action(
            staged,
            actions,
            "PAPER_WATCHLIST_MIN_SCORE",
            _to_int_str(paper_watchlist_min_score),
            "feed_starvation paper_watch_min_score",
        )

    source_qos_cap_choke = low_throughput and (
        source_qos_cap_hits >= max(10, int(metrics.selected_from_batch) // 8)
        or source_qos_symbol_cap_hits >= max(6, int(metrics.selected_from_batch) // 10)
    )
    if source_qos_cap_choke:
        rule_hits["relax_source_qos_cap"] += 1
        trace.append(
            "source_qos_cap_rebalance "
            f"hits={source_qos_cap_hits} symbol_hits={source_qos_symbol_cap_hits} "
            f"selected={metrics.selected_from_batch} topk={source_qos_topk_per_cycle}"
        )
        if qos_caps:
            watch_cap = int(qos_caps.get("watchlist", 0) or 0)
            if watch_cap > 0:
                qos_caps["watchlist"] = _clamp_int(watch_cap + 4, 2, 600)
            for src in ("onchain", "onchain+market", "dexscreener", "geckoterminal"):
                prev = int(qos_caps.get(src, 0) or 0)
                if prev > 0:
                    qos_caps[src] = _clamp_int(prev + 3, 1, 600)
            _apply_action(
                staged,
                actions,
                "V2_SOURCE_QOS_SOURCE_CAPS",
                _format_source_cap_map(qos_caps),
                "source_qos_cap_rebalance caps",
            )
        source_qos_topk_per_cycle = _clamp_int(source_qos_topk_per_cycle + 36, 20, 600)
        _apply_action(
            staged,
            actions,
            "V2_SOURCE_QOS_TOPK_PER_CYCLE",
            _to_int_str(source_qos_topk_per_cycle),
            "source_qos_cap_rebalance topk",
        )
        if source_qos_symbol_cap_hits >= max(6, int(metrics.selected_from_batch) // 10):
            per_symbol_cycle_cap = _clamp_int(per_symbol_cycle_cap + 1, flow_per_symbol_cap_floor, 20)
            _apply_action(
                staged,
                actions,
                "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
                _to_int_str(per_symbol_cycle_cap),
                "source_qos_cap_rebalance symbol_cap",
            )
        source_qos_watch_target = max(0.14, flow_watchlist_share_floor + 0.04)
        if plan_watchlist_share < source_qos_watch_target:
            plan_watchlist_share = _clamp_float(
                plan_watchlist_share + 0.05,
                flow_watchlist_share_floor,
                1.0,
            )
            _apply_action(
                staged,
                actions,
                "PLAN_MAX_WATCHLIST_SHARE",
                _to_float_str(plan_watchlist_share),
                "source_qos_cap_rebalance watchlist_share",
            )
        if plan_min_non_watchlist > 0:
            plan_min_non_watchlist = _clamp_int(plan_min_non_watchlist - 1, 0, flow_non_watchlist_ceiling)
            _apply_action(
                staged,
                actions,
                "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
                _to_int_str(plan_min_non_watchlist),
                "source_qos_cap_rebalance non_watch_min",
            )

    edge_deadlock = (
        low_throughput
        and int(metrics.opened_from_batch) <= 0
        and edge_pressure_hits >= max(24, int(metrics.selected_from_batch) // 3)
    )
    edge_deadlock_risk_block = (
        int(metrics.total_closed) >= 10
        and (
            (metrics.winrate_total is not None and float(metrics.winrate_total) < 30.0)
            or (metrics.realized_total is not None and float(metrics.realized_total) < -0.10)
        )
    )
    if edge_deadlock and (not edge_deadlock_risk_block):
        rule_hits["edge_deadlock_recovery"] += 1
        trace.append(
            "edge_deadlock_recovery "
            f"edge_hits={edge_pressure_hits} selected={metrics.selected_from_batch} "
            f"roll_usd={rolling_edge_min_usd:.4f} cal_usd={calibration_edge_usd_min:.4f} vol_min={calibration_volume_min:.1f}"
        )
        rolling_edge_min_pct = _clamp_float(rolling_edge_min_pct - 0.05, 0.05, 3.0)
        rolling_edge_min_usd = _clamp_float(rolling_edge_min_usd - 0.0010, 0.0005, 0.05)
        calibration_edge_usd_min = _clamp_float(calibration_edge_usd_min - 0.0010, 0.0005, 0.05)
        calibration_volume_min = _clamp_float(calibration_volume_min - 8.0, 5.0, 450.0)
        _apply_action(
            staged,
            actions,
            "V2_ROLLING_EDGE_MIN_PERCENT",
            _to_float_str(rolling_edge_min_pct),
            "edge_deadlock_recovery rolling_pct",
        )
        _apply_action(
            staged,
            actions,
            "V2_ROLLING_EDGE_MIN_USD",
            _to_float_str(rolling_edge_min_usd),
            "edge_deadlock_recovery rolling_usd",
        )
        _apply_action(
            staged,
            actions,
            "V2_CALIBRATION_EDGE_USD_MIN",
            _to_float_str(calibration_edge_usd_min),
            "edge_deadlock_recovery calibration_edge_usd_min",
        )
        _apply_action(
            staged,
            actions,
            "V2_CALIBRATION_VOLUME_MIN",
            _to_float_str(calibration_volume_min),
            "edge_deadlock_recovery calibration_volume_min",
        )
        ev_first_min_net_usd = _clamp_float(ev_first_min_net_usd - 0.0005, -0.0020, 0.02)
        ev_probe_tolerance_usd = _clamp_float(ev_probe_tolerance_usd + 0.0008, 0.001, 0.01)
        _apply_action(
            staged,
            actions,
            "EV_FIRST_ENTRY_MIN_NET_USD",
            _to_float_str(ev_first_min_net_usd),
            "edge_deadlock_recovery ev_min_net",
        )
        _apply_action(
            staged,
            actions,
            "EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD",
            _to_float_str(ev_probe_tolerance_usd),
            "edge_deadlock_recovery ev_probe_tolerance",
        )
        if not calibration_no_tighten:
            calibration_no_tighten = True
            _apply_action(
                staged,
                actions,
                "V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW",
                _to_bool_str(calibration_no_tighten),
                "edge_deadlock_recovery calibration_no_tighten",
            )
        severe_deadlock = (
            edge_pressure_hits >= max(160, int(metrics.selected_from_batch) * 2)
            and int(metrics.opened_from_batch) <= 0
        )
        if severe_deadlock and calibration_enabled:
            calibration_enabled = False
            _apply_action(
                staged,
                actions,
                "V2_CALIBRATION_ENABLED",
                _to_bool_str(calibration_enabled),
                "edge_deadlock_recovery disable_calibration",
            )
    elif edge_deadlock_risk_block:
        rule_hits["edge_deadlock_blocked_risk"] += 1
        trace.append(
            "edge_deadlock_blocked_risk "
            f"closed={metrics.total_closed} winrate={metrics.winrate_total} realized={metrics.realized_total}"
        )

    calibration_reenable_ready = (
        (not calibration_enabled)
        and int(metrics.opened_from_batch) >= 3
        and float(open_rate) >= 0.08
        and int(metrics.total_closed) >= 6
        and ((metrics.winrate_total is None) or float(metrics.winrate_total) >= 35.0)
    )
    if calibration_reenable_ready:
        calibration_enabled = True
        rule_hits["calibration_reenable"] += 1
        trace.append(
            "calibration_reenable "
            f"opened={metrics.opened_from_batch} open_rate={open_rate:.3f} closed={metrics.total_closed} winrate={metrics.winrate_total}"
        )
        _apply_action(
            staged,
            actions,
            "V2_CALIBRATION_ENABLED",
            _to_bool_str(calibration_enabled),
            "calibration_reenable",
        )

    if metrics.opened_from_batch <= 0 or low_throughput:
        if score_min_hits > 0:
            strict = _clamp_int(strict - 2, bounds.strict_floor, bounds.strict_ceiling)
            soft = _clamp_int(soft - 2, bounds.soft_floor, bounds.soft_ceiling)
            if soft > strict:
                soft = strict
            rule_hits["relax_score_min"] += 1
            trace.append(f"relax_score_min hits={score_min_hits}")
            _apply_action(staged, actions, "MARKET_MODE_STRICT_SCORE", _to_int_str(strict), "score_min_dominant")
            _apply_action(staged, actions, "MARKET_MODE_SOFT_SCORE", _to_int_str(soft), "score_min_dominant")
        if safe_volume_hits > 0:
            min_vol = _clamp_float(min_vol - 1.0, bounds.volume_floor, bounds.volume_ceiling)
            rule_hits["relax_safe_volume"] += 1
            trace.append(f"relax_safe_volume hits={safe_volume_hits}")
            _apply_action(staged, actions, "SAFE_MIN_VOLUME_5M_USD", _to_float_str(min_vol), "safe_volume_dominant")
        if ev_low_hits > 0:
            min_edge_pct = _clamp_float(min_edge_pct - 0.05, bounds.edge_floor, bounds.edge_ceiling)
            min_edge_usd = _clamp_float(min_edge_usd - 0.001, bounds.edge_usd_floor, bounds.edge_usd_ceiling)
            rule_hits["relax_ev_low"] += 1
            trace.append(f"relax_ev_low hits={ev_low_hits}")
            _apply_action(staged, actions, "MIN_EXPECTED_EDGE_PERCENT", _to_float_str(min_edge_pct), "ev_net_low")
            _apply_action(staged, actions, "MIN_EXPECTED_EDGE_USD", _to_float_str(min_edge_usd), "ev_net_low")
            if ev_low_hits >= 6 or int(metrics.selected_from_batch) <= 8:
                ev_first_min_net_usd = _clamp_float(ev_first_min_net_usd - 0.0004, -0.0020, 0.02)
                ev_probe_tolerance_usd = _clamp_float(ev_probe_tolerance_usd + 0.0005, 0.001, 0.01)
                _apply_action(
                    staged,
                    actions,
                    "EV_FIRST_ENTRY_MIN_NET_USD",
                    _to_float_str(ev_first_min_net_usd),
                    "ev_net_low_gate",
                )
                _apply_action(
                    staged,
                    actions,
                    "EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD",
                    _to_float_str(ev_probe_tolerance_usd),
                    "ev_net_low_probe_tolerance",
                )
        if cooldown_hits > 0:
            next_cooldown = _clamp_int(cooldown - 15, bounds.cooldown_floor, bounds.cooldown_ceiling)
            if next_cooldown < cooldown:
                cooldown = next_cooldown
                rule_hits["relax_cooldown"] += 1
                trace.append(f"relax_cooldown hits={cooldown_hits}")
                _apply_action(staged, actions, "MAX_TOKEN_COOLDOWN_SECONDS", _to_int_str(cooldown), "cooldown_left")
            else:
                rule_hits["cooldown_floor_reached"] += 1
                trace.append(
                    "cooldown_floor_reached "
                    f"hits={cooldown_hits} current={cooldown} floor={bounds.cooldown_floor}"
                )
        if min_trade_hits > 0:
            trade_min_gate = _clamp_float(trade_min_gate - 0.05, 0.1, 2.0)
            trade_min_paper = _clamp_float(trade_min_paper - 0.05, 0.1, 2.0)
            trade_min_a_core = _clamp_float(trade_min_a_core - 0.05, 0.1, 2.5)
            trade_max_paper = _clamp_float(
                trade_max_paper + (0.20 if min_trade_hits >= 8 else 0.10),
                max(0.3, trade_min_paper),
                2.5,
            )
            rule_hits["relax_trade_size_min"] += 1
            trace.append(f"relax_trade_size_min hits={min_trade_hits}")
            _apply_action(staged, actions, "MIN_TRADE_USD", _to_float_str(trade_min_gate), "min_trade_size_gate")
            _apply_action(staged, actions, "PAPER_TRADE_SIZE_MIN_USD", _to_float_str(trade_min_paper), "min_trade_size")
            _apply_action(staged, actions, "PAPER_TRADE_SIZE_MAX_USD", _to_float_str(trade_max_paper), "min_trade_size_max")
            _apply_action(staged, actions, "ENTRY_A_CORE_MIN_TRADE_USD", _to_float_str(trade_min_a_core), "min_trade_size_a_core")
        if low_throughput and (safe_age_hits > 0 or safe_volume_hits > 0 or min_trade_hits > 0):
            top_n = _clamp_int(top_n + 4, 8, 80)
            rule_hits["low_throughput_expand_topn"] += 1
            trace.append(f"low_throughput_expand_topn open_rate={open_rate:.3f} top_n={top_n}")
            _apply_action(staged, actions, "AUTO_TRADE_TOP_N", _to_int_str(top_n), "low_throughput_expand_topn")

    cooldown_dominant_flow = (
        low_throughput
        and cooldown_hits >= max(18, int(metrics.selected_from_batch) // 2)
        and cooldown_hits >= max(8, ev_low_hits + edge_low_hits)
    )
    if cooldown_dominant_flow:
        rule_hits["cooldown_dominant_flow_expand"] += 1
        trace.append(
            "cooldown_dominant_flow_expand "
            f"cooldown_hits={cooldown_hits} ev_low={ev_low_hits} edge_low={edge_low_hits} "
            f"selected={metrics.selected_from_batch}"
        )
        top_n = _clamp_int(top_n + 4, 8, 80)
        novelty_share = _clamp_float(novelty_share + 0.04, 0.05, 0.90)
        symbol_share_cap = _clamp_float(symbol_share_cap - 0.02, 0.08, 0.50)
        plan_watchlist_share = _clamp_float(plan_watchlist_share + 0.04, flow_watchlist_share_floor, 0.35)
        plan_single_source_share = _clamp_float(plan_single_source_share - 0.04, 0.20, 1.0)
        per_symbol_cycle_cap = _clamp_int(per_symbol_cycle_cap - 1, flow_per_symbol_cap_floor, 20)
        _apply_action(staged, actions, "AUTO_TRADE_TOP_N", _to_int_str(top_n), "cooldown_dominant_flow_expand topn")
        _apply_action(
            staged,
            actions,
            "V2_UNIVERSE_NOVELTY_MIN_SHARE",
            _to_float_str(novelty_share),
            "cooldown_dominant_flow_expand novelty",
        )
        _apply_action(
            staged,
            actions,
            "SYMBOL_CONCENTRATION_MAX_SHARE",
            _to_float_str(symbol_share_cap),
            "cooldown_dominant_flow_expand symbol_share",
        )
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_WATCHLIST_SHARE",
            _to_float_str(plan_watchlist_share),
            "cooldown_dominant_flow_expand watchlist_share",
        )
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_SINGLE_SOURCE_SHARE",
            _to_float_str(plan_single_source_share),
            "cooldown_dominant_flow_expand single_source_share",
        )
        _apply_action(
            staged,
            actions,
            "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
            _to_int_str(per_symbol_cycle_cap),
            "cooldown_dominant_flow_expand symbol_cap",
        )

    dedup_dominant = heavy_dedup_hits >= max(
        6,
        safe_age_hits + safe_volume_hits + score_min_hits,
    )
    dedup_pressure = heavy_dedup_hits >= max(8, int(metrics.selected_from_batch) // 3) or dedup_dominant
    if dedup_pressure and (low_throughput or metrics.opened_from_batch <= 0):
        hard_dedup_choke = heavy_dedup_hits >= max(24, int(metrics.selected_from_batch) // 2)
        if hard_dedup_choke:
            dedup_ttl_seconds = 0
        else:
            dedup_ttl_seconds = _clamp_int(dedup_ttl_seconds - 5, 0, 120)
        per_symbol_cycle_cap = _clamp_int(per_symbol_cycle_cap + 1, 1, 20)
        top_n = _clamp_int(top_n + 2, 8, 80)
        rule_hits["dedup_pressure_relax"] += 1
        trace.append(
            f"dedup_pressure_relax heavy_dedup={heavy_dedup_hits} open_rate={open_rate:.3f} hard={hard_dedup_choke}"
        )
        _apply_action(
            staged,
            actions,
            "HEAVY_CHECK_DEDUP_TTL_SECONDS",
            _to_int_str(dedup_ttl_seconds),
            "dedup_pressure_ttl",
        )
        _apply_action(
            staged,
            actions,
            "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
            _to_int_str(per_symbol_cycle_cap),
            "dedup_pressure_symbol_cap",
        )
        _apply_action(staged, actions, "AUTO_TRADE_TOP_N", _to_int_str(top_n), "dedup_pressure_topn")

    # Quality-gate choke case: many candidates are dropped by symbol concentration
    # before opens happen. In this case, relax V2 quality concentration guard first.
    quality_concentration_choke = (
        metrics.opened_from_batch <= 0
        and concentration_drop_hits >= 12
        and metrics.selected_from_batch >= 8
    )
    if quality_concentration_choke:
        quality_share_ceiling = 0.24 if mode.name in {"fast", "conveyor"} else 0.20
        quality_symbol_max_share = _clamp_float(
            quality_symbol_max_share + 0.02,
            0.10,
            quality_share_ceiling,
        )
        quality_symbol_min_abs_cap = _clamp_int(quality_symbol_min_abs_cap + 1, 2, 6)
        if quality_symbol_reentry_min_seconds > 0:
            quality_symbol_reentry_min_seconds = _clamp_int(quality_symbol_reentry_min_seconds - 30, 60, 3600)
        rule_hits["relax_quality_concentration"] += 1
        trace.append(
            "relax_quality_concentration "
            f"sym_drop={concentration_drop_hits} selected={metrics.selected_from_batch} opened={metrics.opened_from_batch}"
        )
        _apply_action(
            staged,
            actions,
            "V2_QUALITY_SYMBOL_MAX_SHARE",
            _to_float_str(quality_symbol_max_share),
            (
                "quality_gate_symbol_concentration "
                f"drop_hits={concentration_drop_hits} selected={metrics.selected_from_batch}"
            ),
        )
        _apply_action(
            staged,
            actions,
            "V2_QUALITY_SYMBOL_MIN_ABS_CAP",
            _to_int_str(quality_symbol_min_abs_cap),
            "quality_gate_symbol_concentration min_abs_cap",
        )
        _apply_action(
            staged,
            actions,
            "V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS",
            _to_int_str(quality_symbol_reentry_min_seconds),
            "quality_gate_symbol_concentration reentry",
        )

    # Recovery path: if flow exists but quality concentration/size pressure is high,
    # tighten key thresholds back to avoid over-relaxed drift.
    tighten_signals = (
        symbol_concentration_skip_hits >= 8
        or min_trade_hits >= 8
        or (
            metrics.total_closed >= 8
            and metrics.winrate_total is not None
            and float(metrics.winrate_total) < 50.0
        )
        or (
            metrics.total_closed >= 8
            and metrics.realized_total is not None
            and float(metrics.realized_total) < 0.0
        )
    )
    recover_activity_ready = (
        int(metrics.selected_from_batch) >= 24
        and int(metrics.opened_from_batch) >= 3
        and open_rate >= 0.08
    )
    recover_diversity_ready = buy_total <= 0 or buy_unique >= 2 or buy_top_share <= 0.75
    over_relaxed_candidate = recover_activity_ready and recover_diversity_ready and tighten_signals
    tighten_blocked = over_relaxed_candidate and (
        low_throughput
        or source_route_prob_hits >= max(4, int(metrics.selected_from_batch) // 8)
        or min_trade_hits >= max(12, int(metrics.selected_from_batch) // 4)
        or address_duplicate_hits >= max(6, int(metrics.selected_from_batch) // 5)
    )
    if tighten_signals and not recover_activity_ready:
        rule_hits["recover_tighten_not_ready"] += 1
        trace.append(
            "recover_tighten_not_ready "
            f"selected={metrics.selected_from_batch} opened={metrics.opened_from_batch} open_rate={open_rate:.3f}"
        )
    if tighten_blocked:
        rule_hits["recover_tighten_blocked"] += 1
        trace.append(
            "recover_tighten_blocked "
            f"open_rate={open_rate:.3f} route={source_route_prob_hits} min_trade={min_trade_hits} duplicate={address_duplicate_hits}"
        )
    if over_relaxed_candidate and not tighten_blocked:
        if not quality_source_budget_enabled:
            quality_source_budget_enabled = True
            _apply_action(
                staged,
                actions,
                "V2_QUALITY_SOURCE_BUDGET_ENABLED",
                _to_bool_str(True),
                "recover_tighten enable_source_budget",
            )
        strict = _clamp_int(strict + 1, bounds.strict_floor, bounds.strict_ceiling)
        soft = _clamp_int(soft + 1, bounds.soft_floor, bounds.soft_ceiling)
        if soft > strict:
            soft = strict
        min_vol = _clamp_float(min_vol + 1.0, bounds.volume_floor, bounds.volume_ceiling)
        min_edge_pct = _clamp_float(min_edge_pct + 0.03, bounds.edge_floor, bounds.edge_ceiling)
        min_edge_usd = _clamp_float(min_edge_usd + 0.0005, bounds.edge_usd_floor, bounds.edge_usd_ceiling)
        rule_hits["recover_tighten"] += 1
        trace.append(
            "recover_tighten "
            f"symbol_concentration={symbol_concentration_skip_hits} min_trade_size={min_trade_hits} "
            f"closed={metrics.total_closed} winrate={metrics.winrate_total} realized={metrics.realized_total}"
        )
        _apply_action(
            staged,
            actions,
            "MARKET_MODE_STRICT_SCORE",
            _to_int_str(strict),
            (
                "recover_tighten "
                f"symbol_concentration={symbol_concentration_skip_hits} min_trade_size={min_trade_hits}"
            ),
        )
        _apply_action(
            staged,
            actions,
            "MARKET_MODE_SOFT_SCORE",
            _to_int_str(soft),
            (
                "recover_tighten "
                f"symbol_concentration={symbol_concentration_skip_hits} min_trade_size={min_trade_hits}"
            ),
        )
        _apply_action(
            staged,
            actions,
            "SAFE_MIN_VOLUME_5M_USD",
            _to_float_str(min_vol),
            "recover_tighten safe_volume",
        )
        _apply_action(
            staged,
            actions,
            "MIN_EXPECTED_EDGE_PERCENT",
            _to_float_str(min_edge_pct),
            "recover_tighten edge_percent",
        )
        _apply_action(
            staged,
            actions,
            "MIN_EXPECTED_EDGE_USD",
            _to_float_str(min_edge_usd),
            "recover_tighten edge_usd",
        )

    diversity_kpi_penalty = (
        buy_total >= 4
        and (
            buy_unique <= 1
            or buy_top_share >= 0.78
        )
    )
    if diversity_kpi_penalty:
        rule_hits["diversity_kpi_penalty"] += 1
        trace.append(
            "diversity_kpi_penalty "
            f"buys={buy_total} unique={buy_unique} top_share={buy_top_share:.2f} open_rate={open_rate:.3f}"
        )
        plan_watchlist_share = _clamp_float(
            plan_watchlist_share - 0.05,
            flow_watchlist_share_floor,
            flow_watchlist_share_ceiling,
        )
        plan_min_non_watchlist = _clamp_int(plan_min_non_watchlist + 1, 0, flow_non_watchlist_ceiling)
        plan_single_source_share = _clamp_float(plan_single_source_share - 0.06, 0.20, 1.0)
        novelty_share = _clamp_float(novelty_share + 0.06, 0.10, 0.90)
        symbol_share_cap = _clamp_float(symbol_share_cap - 0.05, 0.08, 0.50)
        per_symbol_cycle_cap = _clamp_int(per_symbol_cycle_cap - 1, flow_per_symbol_cap_floor, 20)
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_WATCHLIST_SHARE",
            _to_float_str(plan_watchlist_share),
            "diversity_kpi_penalty watchlist_share",
        )
        _apply_action(
            staged,
            actions,
            "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
            _to_int_str(plan_min_non_watchlist),
            "diversity_kpi_penalty non_watch_min",
        )
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_SINGLE_SOURCE_SHARE",
            _to_float_str(plan_single_source_share),
            "diversity_kpi_penalty single_source_share",
        )
        _apply_action(
            staged,
            actions,
            "V2_UNIVERSE_NOVELTY_MIN_SHARE",
            _to_float_str(novelty_share),
            "diversity_kpi_penalty novelty_share",
        )
        _apply_action(
            staged,
            actions,
            "SYMBOL_CONCENTRATION_MAX_SHARE",
            _to_float_str(symbol_share_cap),
            "diversity_kpi_penalty symbol_share_cap",
        )
        _apply_action(
            staged,
            actions,
            "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
            _to_int_str(per_symbol_cycle_cap),
            "diversity_kpi_penalty per_symbol_cap",
        )
        if not quality_source_budget_enabled:
            quality_source_budget_enabled = True
            _apply_action(
                staged,
                actions,
                "V2_QUALITY_SOURCE_BUDGET_ENABLED",
                _to_bool_str(True),
                "diversity_kpi_penalty enable_source_budget",
            )

    # When opens are concentrated in one symbol, force diversity before full lock-in.
    # This keeps the conveyor alive when one temporary leader dominates.
    duplicate_choke_pattern = (
        address_duplicate_hits >= max(6, int(metrics.selected_from_batch) // 3)
        and (low_throughput or metrics.opened_from_batch <= 1)
    )
    repeated_symbol_pattern = buy_total >= 3 and buy_unique <= 2 and buy_top_share >= 0.65
    drop_concentration_pattern = (
        concentration_drop_hits >= 10
        and metrics.selected_from_batch >= 20
        and metrics.opened_from_batch > 0
    )
    concentration_signal = repeated_symbol_pattern or drop_concentration_pattern or duplicate_choke_pattern
    if concentration_signal:
        rule_hits["anti_concentration"] += 1
        trace.append(
            "anti_concentration "
            f"top_share={buy_top_share:.2f} unique={buy_unique} buys={buy_total} sym_drop={concentration_drop_hits}"
        )
        novelty_step = 0.04
        symbol_cap_step = 0.03
        cooldown_step = 20
        watchlist_scale = 0.85
        if duplicate_choke_pattern:
            novelty_step = 0.08
            symbol_cap_step = 0.05
            cooldown_step = 30
            watchlist_scale = 0.60
            top_n = _clamp_int(top_n + 4, 8, 80)
            plan_watchlist_share = _clamp_float(
                plan_watchlist_share - 0.06,
                flow_watchlist_share_floor,
                flow_watchlist_share_ceiling,
            )
            plan_min_non_watchlist = _clamp_int(plan_min_non_watchlist + 1, 0, flow_non_watchlist_ceiling)
            plan_single_source_share = _clamp_float(plan_single_source_share - 0.05, 0.20, 1.0)
            source_router_min_trades = _clamp_int(source_router_min_trades - 2, 4, 20)
            source_router_bad_prob = _clamp_float(source_router_bad_prob + 0.04, 0.35, 0.95)
            source_router_severe_prob = _clamp_float(source_router_severe_prob + 0.05, 0.20, 0.90)
            trace.append(
                "duplicate_choke_relax "
                f"duplicate={address_duplicate_hits} selected={metrics.selected_from_batch} open_rate={open_rate:.3f}"
            )
            _apply_action(
                staged,
                actions,
                "PLAN_MAX_WATCHLIST_SHARE",
                _to_float_str(plan_watchlist_share),
                "duplicate_choke watchlist_share",
            )
            _apply_action(
                staged,
                actions,
                "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
                _to_int_str(plan_min_non_watchlist),
                "duplicate_choke non_watch_min",
            )
            _apply_action(
                staged,
                actions,
                "PLAN_MAX_SINGLE_SOURCE_SHARE",
                _to_float_str(plan_single_source_share),
                "duplicate_choke single_source_share",
            )
            _apply_action(staged, actions, "AUTO_TRADE_TOP_N", _to_int_str(top_n), "duplicate_choke_topn")
            _apply_action(
                staged,
                actions,
                "SOURCE_ROUTER_MIN_TRADES",
                _to_int_str(source_router_min_trades),
                "duplicate_choke source_router_min_trades",
            )
            _apply_action(
                staged,
                actions,
                "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY",
                _to_float_str(source_router_bad_prob),
                "duplicate_choke source_router_bad_prob",
            )
            _apply_action(
                staged,
                actions,
                "SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY",
                _to_float_str(source_router_severe_prob),
                "duplicate_choke source_router_severe_prob",
            )
            if qos_caps:
                qos_caps = _rebalance_source_caps(
                    qos_caps,
                    watchlist_scale=watchlist_scale,
                    min_value=1,
                    max_value=600,
                    watchlist_min=5,
                )
                _apply_action(
                    staged,
                    actions,
                    "V2_SOURCE_QOS_SOURCE_CAPS",
                    _format_source_cap_map(qos_caps),
                    "duplicate_choke source_qos_caps",
                )
            if universe_caps:
                universe_caps = _rebalance_source_caps(
                    universe_caps,
                    watchlist_scale=watchlist_scale,
                    min_value=1,
                    max_value=600,
                    watchlist_min=5,
                )
                _apply_action(
                    staged,
                    actions,
                    "V2_UNIVERSE_SOURCE_CAPS",
                    _format_source_cap_map(universe_caps),
                    "duplicate_choke universe_caps",
                )
        novelty_floor = 0.10
        novelty_ceiling = 0.85
        symbol_cap_floor = 0.12
        symbol_cap_ceiling = 0.50
        novelty_share = _clamp_float(novelty_share + novelty_step, novelty_floor, novelty_ceiling)
        symbol_share_cap = _clamp_float(symbol_share_cap - symbol_cap_step, symbol_cap_floor, symbol_cap_ceiling)
        per_symbol_cycle_cap = _clamp_int(per_symbol_cycle_cap - 1, flow_per_symbol_cap_floor, 20)
        _apply_action(
            staged,
            actions,
            "V2_UNIVERSE_NOVELTY_MIN_SHARE",
            _to_float_str(novelty_share),
            (
                f"anti_concentration top_share={buy_top_share:.2f} unique={buy_unique} "
                f"buys={buy_total} sym_drop={concentration_drop_hits}"
            ),
        )
        _apply_action(
            staged,
            actions,
            "SYMBOL_CONCENTRATION_MAX_SHARE",
            _to_float_str(symbol_share_cap),
            f"anti_concentration top_share={buy_top_share:.2f} unique={buy_unique} sym_drop={concentration_drop_hits}",
        )
        _apply_action(
            staged,
            actions,
            "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
            _to_int_str(per_symbol_cycle_cap),
            "anti_concentration per_symbol_cap",
        )
        if cooldown_hits <= 0:
            cooldown = _clamp_int(cooldown + cooldown_step, bounds.cooldown_floor, bounds.cooldown_ceiling)
            _apply_action(
                staged,
                actions,
                "MAX_TOKEN_COOLDOWN_SECONDS",
                _to_int_str(cooldown),
                "anti_concentration cooldown",
            )

    # Final low-throughput floor guards: keep flow knobs within survivable band.
    if low_throughput:
        if plan_watchlist_share < flow_watchlist_share_floor:
            plan_watchlist_share = flow_watchlist_share_floor
            _apply_action(
                staged,
                actions,
                "PLAN_MAX_WATCHLIST_SHARE",
                _to_float_str(plan_watchlist_share),
                "flow_floor_guard watchlist_share",
            )
        if plan_min_non_watchlist > flow_non_watchlist_ceiling:
            plan_min_non_watchlist = flow_non_watchlist_ceiling
            _apply_action(
                staged,
                actions,
                "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
                _to_int_str(plan_min_non_watchlist),
                "flow_floor_guard non_watch_min",
            )
        if per_symbol_cycle_cap < flow_per_symbol_cap_floor:
            per_symbol_cycle_cap = flow_per_symbol_cap_floor
            _apply_action(
                staged,
                actions,
                "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
                _to_int_str(per_symbol_cycle_cap),
                "flow_floor_guard per_symbol_cap",
            )

    # Guard relation from safe-contract.
    strict = _get_int(staged, "MARKET_MODE_STRICT_SCORE", strict)
    soft = _get_int(staged, "MARKET_MODE_SOFT_SCORE", soft)
    if soft > strict:
        rule_hits["soft_leq_strict_guard"] += 1
        trace.append(f"soft_leq_strict_guard soft={soft} strict={strict}")
        _apply_action(
            staged,
            actions,
            "MARKET_MODE_SOFT_SCORE",
            _to_int_str(strict),
            "soft_must_be_leq_strict",
        )

    if mode.name in {"calm", "sniper"} and metrics.total_closed >= 3 and metrics.winrate_total is not None:
        if metrics.winrate_total < 45.0:
            strict = _clamp_int(
                _get_int(staged, "MARKET_MODE_STRICT_SCORE", strict) + 1,
                bounds.strict_floor,
                bounds.strict_ceiling,
            )
            soft = _clamp_int(
                _get_int(staged, "MARKET_MODE_SOFT_SCORE", soft) + 1,
                bounds.soft_floor,
                bounds.soft_ceiling,
            )
            if soft > strict:
                soft = strict
            rule_hits["low_winrate_tighten"] += 1
            trace.append(f"low_winrate_tighten winrate={metrics.winrate_total}")
            _apply_action(staged, actions, "MARKET_MODE_STRICT_SCORE", _to_int_str(strict), "low_winrate_tighten")
            _apply_action(staged, actions, "MARKET_MODE_SOFT_SCORE", _to_int_str(soft), "low_winrate_tighten")

    if route_pressure and (blacklist_hits > 0 or address_duplicate_hits > 0) and low_throughput:
        rule_hits["route_guard_hold"] += 1
        trace.append(
            "route_guard_hold "
            f"blacklist={blacklist_hits} duplicate={address_duplicate_hits} open_rate={open_rate:.3f}"
        )

    coalesced_actions, coalesced_count = _coalesce_actions(actions)
    prioritized = sorted(coalesced_actions, key=_action_priority)
    limited = prioritized[: mode.max_actions_per_tick]
    trimmed = max(0, len(prioritized) - len(limited))
    meta: dict[str, Any] = {"rule_hits": dict(rule_hits)}
    meta["churn_lock"] = {
        "active": bool(churn_lock_active),
        "symbol": str(churn_lock_symbol),
        "remaining_seconds": int(churn_lock_remaining),
        "detected_15m": bool(churn_detected),
        "open_share_15m": round(float(churn_open_share), 6),
        "flat_close_share_15m": round(float(churn_flat_share), 6),
    }
    if trimmed > 0:
        meta["trimmed_actions"] = int(trimmed)
    if coalesced_count > 0:
        meta["coalesced_actions"] = int(coalesced_count)
    return limited, trace, meta


def _validate_overrides(root: Path, overrides: dict[str, str]) -> list[str]:
    contract = load_contract(root)
    return validate_overrides(overrides, contract)


def _append_runtime_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_runtime_patch_overrides(
    *,
    root: Path,
    profile_id: str,
    overrides: dict[str, str],
) -> tuple[bool, str]:
    hot: dict[str, str] = {}
    for key in sorted(TUNER_HOT_APPLY_KEYS):
        if key in overrides:
            hot[str(key)] = str(overrides.get(key, ""))
    payload = {
        "ts": _now_iso(),
        "profile_id": str(profile_id),
        "overrides": hot,
    }
    path = _runtime_patch_path(root, profile_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return True, f"runtime_patch_synced:{path}"
    except Exception as exc:
        return False, f"runtime_patch_write_failed:{exc}"


def _restart_required_changed_keys(actions: list[Action]) -> list[str]:
    changed = {str(a.key or "").strip() for a in (actions or []) if str(a.key or "").strip()}
    if not changed:
        return []
    required: list[str] = []
    for key in sorted(changed):
        if key in TUNER_RESTART_REQUIRED_KEYS:
            required.append(key)
    return required


def _partition_pending_diff_keys(diff_keys: list[str]) -> tuple[list[str], list[str]]:
    restart_required: list[str] = []
    hot_only: list[str] = []
    for key in diff_keys:
        item = str(key or "").strip()
        if not item:
            continue
        if item in TUNER_RESTART_REQUIRED_KEYS:
            restart_required.append(item)
        else:
            hot_only.append(item)
    return restart_required, hot_only


def _run_launcher(root: Path, profile_id: str, *, timeout_seconds: int = 90) -> tuple[bool, str]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(root / "tools" / "matrix_paper_launcher.ps1"),
        "-ProfileIds",
        profile_id,
        "-Run",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            check=False,
            timeout=max(15, int(timeout_seconds)),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return False, f"launcher_timeout after {max(15, int(timeout_seconds))}s"
    ok = proc.returncode == 0
    return ok, f"launcher_exit_code={proc.returncode}"


def _python_executable(root: Path) -> str:
    if os.name == "nt":
        py = root / ".venv" / "Scripts" / "python.exe"
    else:
        py = root / ".venv" / "bin" / "python"
    if py.exists():
        return str(py)
    return "python"


def _active_matrix_payload(root: Path) -> dict[str, Any]:
    path = _active_matrix_path(root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_active_matrix_payload(root: Path, payload: dict[str, Any]) -> bool:
    path = _active_matrix_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception:
        return False


def _update_active_matrix_overrides(root: Path, profile_id: str, overrides: dict[str, str]) -> bool:
    payload = _active_matrix_payload(root)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    changed = False
    for row in items:
        if not isinstance(row, dict):
            continue
        if str(row.get("id", "")).strip() != str(profile_id):
            continue
        row["overrides"] = {str(k): str(v) for k, v in (overrides or {}).items()}
        changed = True
    if not changed:
        return False
    payload["items"] = items
    payload["updated_at"] = _now_iso()
    return _write_active_matrix_payload(root, payload)


def _active_item_for_profile(root: Path, profile_id: str) -> dict[str, Any]:
    payload = _active_matrix_payload(root)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    for row in items:
        if not isinstance(row, dict):
            continue
        if str(row.get("id", "")).strip() == str(profile_id):
            return row
    return {}


def _profile_env_file(root: Path, profile_id: str) -> str:
    row = _active_item_for_profile(root, profile_id)
    env_file = str(row.get("env_file", "") or "").strip()
    if env_file:
        return env_file
    fallback = root / "data" / "matrix" / "env" / f"{profile_id}.env"
    return str(fallback)


def _clear_profile_graceful_stop(root: Path, profile_id: str) -> None:
    row = _active_item_for_profile(root, profile_id)
    rel = str(row.get("graceful_stop_file", "") or "").strip()
    if not rel:
        return
    path = Path(rel)
    abs_path = path if path.is_absolute() else (root / path)
    try:
        if abs_path.exists():
            abs_path.unlink()
    except Exception:
        pass


def _profile_graceful_stop_path(root: Path, profile_id: str) -> Path | None:
    row = _active_item_for_profile(root, profile_id)
    rel = str(row.get("graceful_stop_file", "") or "").strip()
    if not rel:
        return None
    path = Path(rel)
    return path if path.is_absolute() else (root / path)


def _profile_pid(root: Path, profile_id: str) -> int:
    row = _active_item_for_profile(root, profile_id)
    try:
        return int(row.get("pid", 0) or 0)
    except Exception:
        return 0


def _request_profile_graceful_stop(root: Path, profile_id: str) -> tuple[bool, str]:
    path = _profile_graceful_stop_path(root, profile_id)
    if path is None:
        return False, "graceful_stop_path_missing"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": float(time.time()),
            "timestamp": _now_iso(),
            "source": "runtime_tuner",
            "reason": "runtime_tuner_restart",
            "actor": "matrix_runtime_tuner.py",
            "profile_id": str(profile_id),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return True, f"graceful_stop_requested:{path}"
    except Exception as exc:
        return False, f"graceful_stop_write_failed:{exc}"


def _stop_profile_process(
    root: Path, profile_id: str, *, graceful_timeout_seconds: int = 18
) -> tuple[bool, str]:
    pid = _profile_pid(root, profile_id)
    if pid <= 0 or not _pid_is_running(pid):
        _clear_profile_graceful_stop(root, profile_id)
        _update_active_matrix_pid(root, profile_id, 0, "stopped")
        return True, "already_stopped"
    requested, stop_detail = _request_profile_graceful_stop(root, profile_id)
    deadline = float(time.time()) + float(max(2, int(graceful_timeout_seconds)))
    while float(time.time()) < deadline:
        if not _pid_is_running(pid):
            break
        time.sleep(0.25)
    if _pid_is_running(pid):
        forced = _terminate_pid(pid, timeout_seconds=LOCK_STALE_TERMINATE_TIMEOUT_SECONDS)
        if not forced:
            _clear_profile_graceful_stop(root, profile_id)
            return False, f"stop_failed pid={pid}"
        suffix = "forced_kill"
        stop_detail = f"{stop_detail};{suffix}" if stop_detail else suffix
    elif not requested and not stop_detail:
        stop_detail = "stopped_without_graceful_signal"
    _clear_profile_graceful_stop(root, profile_id)
    _update_active_matrix_pid(root, profile_id, 0, "stopped")
    return True, f"stopped_pid={pid} detail={stop_detail}"


def _spawn_profile_process(root: Path, profile_id: str) -> tuple[bool, str, int]:
    env_file = _profile_env_file(root, profile_id)
    try:
        env_path = Path(env_file)
    except Exception:
        env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = (root / env_path).resolve()
    if not env_path.exists():
        return False, f"env_missing:{env_path}", 0
    py = _python_executable(root)
    try:
        env = os.environ.copy()
        env["BOT_ENV_FILE"] = str(env_path)
        env["PYTHONIOENCODING"] = "utf-8"
        creationflags = 0
        if os.name == "nt":
            creationflags = int(getattr(subprocess, "DETACHED_PROCESS", 0)) | int(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        proc = subprocess.Popen(
            [py, "main_local.py"],
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
    except Exception as exc:
        return False, f"spawn_error:{exc}", 0
    pid = int(getattr(proc, "pid", 0) or 0)
    if pid <= 0:
        return False, "spawn_pid_zero", 0
    for _ in range(24):
        if _pid_is_running(pid):
            return True, f"spawned_pid={pid}", pid
        time.sleep(0.5)
    return False, f"spawned_not_alive pid={pid}", pid


def _update_active_matrix_pid(root: Path, profile_id: str, pid: int, status: str) -> bool:
    payload = _active_matrix_payload(root)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    changed = False
    alive_count = 0
    for row in items:
        if not isinstance(row, dict):
            continue
        row_pid = int(row.get("pid", 0) or 0)
        row_alive = row_pid > 0 and _pid_is_running(row_pid)
        if str(row.get("id", "")).strip() == str(profile_id):
            row["pid"] = int(pid) if int(pid) > 0 else None
            row["status"] = str(status or "unknown")
            row_alive = int(pid) > 0 and _pid_is_running(int(pid))
            changed = True
        if row_alive:
            alive_count += 1
    if not changed:
        return False
    payload["updated_at"] = _now_iso()
    payload["running"] = bool(alive_count > 0)
    payload["alive_count"] = int(alive_count)
    payload["items"] = items
    return _write_active_matrix_payload(root, payload)


def _run_dead_profile_recovery(root: Path, profile_id: str) -> tuple[bool, str]:
    _clear_profile_graceful_stop(root, profile_id)
    ok, detail, pid = _spawn_profile_process(root, profile_id)
    if not ok:
        return False, detail
    _update_active_matrix_pid(root, profile_id, pid, "running")
    for _ in range(10):
        if _profile_running(root, profile_id):
            return True, detail
        time.sleep(0.4)
    return False, f"{detail} profile_check_failed"


def _restart_profile_process(root: Path, profile_id: str) -> tuple[bool, str]:
    stopped_ok, stopped_detail = _stop_profile_process(root, profile_id)
    if not stopped_ok:
        hb_recent = _heartbeat_recent(
            root, profile_id, max_age_seconds=DEAD_PROFILE_HEARTBEAT_MAX_AGE_SECONDS
        )
        pid = _profile_pid(root, profile_id)
        pid_running = _pid_is_running(pid) if pid > 0 else False
        session = _latest_session_log(root, profile_id)
        session_age = _session_age_seconds(session) if session is not None else float("inf")
        stale_profile = (
            (not hb_recent)
            and (
                (not pid_running)
                or (session_age > float(DEAD_PROFILE_HEARTBEAT_MAX_AGE_SECONDS))
            )
        )
        if not stale_profile:
            return False, stopped_detail
        _clear_profile_graceful_stop(root, profile_id)
        _update_active_matrix_pid(root, profile_id, 0, "stopped")
        started_ok, started_detail = _run_dead_profile_recovery(root, profile_id)
        if not started_ok:
            return False, f"{stopped_detail};stale_recovery_failed:{started_detail}"
        return True, f"{stopped_detail};stale_recovery:{started_detail}"
    started_ok, started_detail = _run_dead_profile_recovery(root, profile_id)
    if not started_ok:
        return False, f"{stopped_detail};{started_detail}"
    return True, f"{stopped_detail};{started_detail}"


def _load_env_entries(path: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    if not path.exists():
        return [], {}
    rows: list[tuple[str, str]] = []
    env_map: dict[str, str] = {}
    seen: set[str] = set()
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = str(raw_line)
            stripped = line.strip()
            if (not stripped) or stripped.startswith("#") or ("=" not in line):
                rows.append(("raw", line))
                continue
            key_raw, value_raw = line.split("=", 1)
            key = str(key_raw).strip()
            if not key:
                rows.append(("raw", line))
                continue
            rows.append(("kv", key))
            if key not in seen:
                seen.add(key)
                env_map[key] = str(value_raw)
        return rows, env_map
    except Exception:
        return [], {}


def _write_env_entries(path: Path, rows: list[tuple[str, str]], env_map: dict[str, str]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        out_lines: list[str] = []
        rendered: set[str] = set()
        for row_type, row_value in rows:
            if row_type == "raw":
                out_lines.append(str(row_value))
                continue
            key = str(row_value)
            if key in rendered:
                continue
            out_lines.append(f"{key}={str(env_map.get(key, ''))}")
            rendered.add(key)
        for key in env_map.keys():
            if key in rendered:
                continue
            out_lines.append(f"{key}={str(env_map.get(key, ''))}")
        data = "\n".join(out_lines) + "\n"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception:
        return False


def _sync_profile_env_overrides(
    root: Path,
    profile_id: str,
    overrides: dict[str, str],
) -> tuple[bool, str]:
    env_file = _profile_env_file(root, profile_id)
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = (root / env_path).resolve()
    if not env_path.exists():
        return False, f"env_missing:{env_path}"
    rows, env_map = _load_env_entries(env_path)
    if not env_map and not rows:
        return False, f"env_read_failed:{env_path}"
    changed = False
    for key, value in (overrides or {}).items():
        k = str(key)
        v = str(value)
        old = str(env_map.get(k, ""))
        if old != v:
            env_map[k] = v
            changed = True
    if not changed:
        return True, "env_sync_noop"
    ok = _write_env_entries(env_path, rows, env_map)
    if not ok:
        return False, f"env_write_failed:{env_path}"
    return True, f"env_synced:{env_path}"


def _profile_running(root: Path, profile_id: str) -> bool:
    path = _active_matrix_path(root)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for item in payload.get("items", []) or []:
        if str(item.get("id", "")).strip() != profile_id:
            continue
        try:
            pid = int(item.get("pid", 0) or 0)
        except Exception:
            pid = 0
        if pid > 0 and _pid_is_running(pid):
            return True
        status = str(item.get("status", "")).strip().lower()
        if status == "running" and pid <= 0:
            return True
        return False
    return False


def _active_override_subset(root: Path, profile_id: str, keys: set[str]) -> dict[str, str]:
    path = _active_matrix_path(root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    for item in payload.get("items", []) or []:
        if str(item.get("id", "")).strip() != profile_id:
            continue
        raw = item.get("overrides", {})
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key in keys:
            if key in raw:
                out[str(key)] = str(raw.get(key))
        return out
    return {}


def _diff_override_keys(current: dict[str, str], target: dict[str, str]) -> list[str]:
    diff: list[str] = []
    for key, value in (target or {}).items():
        cur = str((current or {}).get(str(key), "")).strip()
        want = str(value).strip()
        if cur != want:
            diff.append(str(key))
    return diff


def _tick(
    *,
    root: Path,
    profile_id: str,
    mode: ModeSpec,
    target_policy: TargetPolicy,
    forced_phase: str,
    window_minutes: int,
    dry_run: bool,
    restart_cooldown_seconds: int,
    restart_max_per_hour: int,
    allow_zero_cooldown: bool,
    last_restart_ts: float,
) -> tuple[dict[str, Any], float]:
    session = _latest_session_log(root, profile_id)
    if session is None:
        raise SystemExit(f"No session logs for profile '{profile_id}'")
    metrics = _read_window_metrics(session, window_minutes=window_minutes)
    session_age_seconds = _session_age_seconds(session)
    # Prevent restart churn on fresh sessions: let runtime collect enough data first.
    warmup_guard_active = bool(
        session_age_seconds < 180.0
        and (int(metrics.lines_seen) < 120 or int(metrics.scanned) < 30)
    )
    preset_file = _preset_path(root, profile_id)
    if not preset_file.exists():
        raise SystemExit(
            f"User preset not found: {preset_file}. Runtime tuner only works with user presets."
        )
    preset = _load_preset(preset_file)
    overrides_raw = preset.get("overrides", {})
    if not isinstance(overrides_raw, dict):
        raise SystemExit(f"Invalid preset format (overrides object missing): {preset_file}")
    overrides: dict[str, str] = {str(k): str(v) for k, v in overrides_raw.items()}
    contract = load_contract(root)
    protected_keys = {str(x).strip() for x in (contract.get("protected_keys") or []) if str(x).strip()}
    state = _load_runtime_state(root, profile_id)
    tick_now_ts = float(time.time())
    restart_history = _prune_restart_history(
        list(state.restart_history_ts or []),
        now_ts=tick_now_ts,
        window_seconds=3600.0,
    )
    if float(last_restart_ts or 0.0) > 0.0:
        restart_history.append(float(last_restart_ts))
        restart_history = _prune_restart_history(
            restart_history,
            now_ts=tick_now_ts,
            window_seconds=3600.0,
        )
    state.restart_history_ts = list(restart_history)

    def _can_restart(now_ts: float) -> tuple[bool, float, int]:
        nonlocal restart_history
        restart_history = _prune_restart_history(
            restart_history,
            now_ts=now_ts,
            window_seconds=3600.0,
        )
        return _restart_gate(
            now_ts=now_ts,
            history=restart_history,
            restart_cooldown_seconds=restart_cooldown_seconds,
            restart_max_per_hour=restart_max_per_hour,
        )

    def _record_restart(now_ts: float) -> None:
        nonlocal restart_history, last_restart_ts
        restart_history.append(float(now_ts))
        restart_history = _prune_restart_history(
            restart_history,
            now_ts=now_ts,
            window_seconds=3600.0,
        )
        state.restart_history_ts = list(restart_history)
        last_restart_ts = float(now_ts)
    runtime_subset = _active_override_subset(root, profile_id, TUNER_RUNTIME_FALLBACK_KEYS)
    for key, value in runtime_subset.items():
        if key not in overrides or str(overrides.get(key, "")).strip() == "":
            overrides[key] = str(value)

    telemetry_before = _collect_telemetry_v2(
        root=root,
        profile_id=profile_id,
        config_before=overrides,
        config_after=overrides,
        actions=[],
        metrics=metrics,
    )
    pre_funnel = telemetry_before.get("funnel_15m", {}) if isinstance(telemetry_before, dict) else {}
    pre_buy_15m = int(_safe_float((pre_funnel or {}).get("buy", 0), 0.0))
    effective_target_tph, effective_target_meta_pre = _resolve_effective_target_trades_per_hour(
        requested_tph=float(target_policy.target_trades_per_hour),
        target=target_policy,
        state=state,
        throughput_est=float(pre_buy_15m) * 4.0,
    )
    policy_decision = _resolve_policy_phase(
        metrics=metrics,
        telemetry=telemetry_before,
        target=target_policy,
        state=state,
        forced_phase=forced_phase,
        effective_target_trades_per_hour=float(effective_target_tph),
    )

    actions, decision_trace, decision_meta = _build_action_plan(
        metrics=metrics,
        overrides=overrides,
        mode=mode,
        telemetry=telemetry_before,
        runtime_state=state,
        now_ts=tick_now_ts,
        allow_zero_cooldown=bool(allow_zero_cooldown),
    )
    if warmup_guard_active:
        actions = []
        decision_trace.insert(
            0,
            (
                "warmup_guard "
                f"session_age={session_age_seconds:.1f}s lines={int(metrics.lines_seen)} "
                f"scanned={int(metrics.scanned)}"
            ),
        )
        decision_meta = dict(decision_meta or {})
        decision_meta["warmup_guard"] = {
            "active": True,
            "session_age_seconds": round(float(session_age_seconds), 2),
            "lines_seen": int(metrics.lines_seen),
            "scanned": int(metrics.scanned),
        }
    actions, blocked_mutations = _enforce_mutable_action_keys(actions=actions, protected_keys=protected_keys)
    actions, blocked_phase = _filter_actions_by_phase(actions=actions, phase=policy_decision.phase)
    actions, capped_actions = _apply_action_delta_caps(actions=actions, phase=policy_decision.phase)

    blocked_actions = blocked_mutations + blocked_phase
    rollback_triggered = False
    projected_degrade_streak = (
        state.degrade_streak + 1
        if (
            policy_decision.risk_fail
            or policy_decision.pre_risk_fail
            or policy_decision.blacklist_fail
            or policy_decision.diversity_fail
        )
        else 0
    )
    current_mutable = {
        str(k): str(v)
        for k, v in overrides.items()
        if str(k) in TUNER_MUTABLE_KEYS and str(k) not in TUNER_EPHEMERAL_KEYS
    }
    current_hash = _overrides_hash(current_mutable)
    if (
        policy_decision.phase == "tighten"
        and projected_degrade_streak >= int(target_policy.rollback_degrade_streak)
        and bool(state.stable_overrides)
        and str(state.stable_hash or "")
        and str(state.stable_hash or "") != current_hash
        and (not warmup_guard_active)
    ):
        rollback_actions = _build_rollback_actions(current=overrides, stable=state.stable_overrides)
        rollback_actions, rollback_blocked = _enforce_mutable_action_keys(
            actions=rollback_actions,
            protected_keys=protected_keys,
        )
        rollback_actions, rollback_capped = _apply_action_delta_caps(actions=rollback_actions, phase="tighten")
        if rollback_actions:
            rollback_triggered = True
            actions = rollback_actions
            blocked_actions.extend(rollback_blocked)
            capped_actions.extend(rollback_capped)
            decision_trace.insert(0, f"rollback_triggered degrade_streak={projected_degrade_streak}")
            decision_meta = dict(decision_meta or {})
            decision_meta["rollback_target_hash"] = str(state.stable_hash or "")

    staged_overrides: dict[str, str] = {str(k): str(v) for k, v in overrides.items()}
    for action in actions:
        staged_overrides[action.key] = action.new_value
    cooldown_floor = int(max(0, int(mode.cooldown_floor)))
    if not bool(allow_zero_cooldown):
        cooldown_now = _get_int(staged_overrides, "MAX_TOKEN_COOLDOWN_SECONDS", cooldown_floor)
        if cooldown_now < cooldown_floor:
            old_value = str(staged_overrides.get("MAX_TOKEN_COOLDOWN_SECONDS", str(cooldown_now)))
            new_value = _to_int_str(cooldown_floor)
            staged_overrides["MAX_TOKEN_COOLDOWN_SECONDS"] = new_value
            if old_value.strip() != new_value:
                actions.append(
                    Action(
                        key="MAX_TOKEN_COOLDOWN_SECONDS",
                        old_value=old_value,
                        new_value=new_value,
                        reason="cooldown_floor_guard",
                    )
                )
                decision_trace.append(f"cooldown_floor_guard {old_value}->{new_value}")
    issues: list[str] = []
    restarted = False
    restart_tail = ""
    apply_state = "noop"
    target_overrides: dict[str, str] = dict(overrides)
    pending_runtime_diff_keys: list[str] = []
    pending_restart_diff_keys: list[str] = []
    pending_hot_apply_diff_keys: list[str] = []
    changed_restart_required_keys: list[str] = []
    env_sync_ok = True
    env_sync_detail = ""
    runtime_patch_sync_ok = True
    runtime_patch_sync_detail = ""

    if actions:
        changed_restart_required_keys = _restart_required_changed_keys(actions)
        issues = _validate_overrides(root, staged_overrides)
        if issues:
            apply_state = "validation_failed"
        elif dry_run:
            apply_state = "dry_run"
        else:
            preset["overrides"] = staged_overrides
            preset["updated_at"] = _now_iso()
            _write_preset(preset_file, preset)
            apply_state = "written"
            target_overrides = dict(staged_overrides)
            env_sync_ok, env_sync_detail = _sync_profile_env_overrides(root, profile_id, target_overrides)
            if not env_sync_ok:
                issues.append(str(env_sync_detail))
                apply_state = "written_env_sync_failed"
            elif env_sync_ok:
                runtime_patch_sync_ok, runtime_patch_sync_detail = _write_runtime_patch_overrides(
                    root=root,
                    profile_id=profile_id,
                    overrides=target_overrides,
                )
                if not runtime_patch_sync_ok:
                    issues.append(str(runtime_patch_sync_detail))
                    apply_state = "written_runtime_patch_sync_failed"

                if changed_restart_required_keys:
                    now_ts = time.time()
                    restart_allowed, restart_cooldown_left, restart_budget_left = _can_restart(now_ts)
                    can_restart = (
                        (not warmup_guard_active)
                        and (
                            metrics.open_positions <= 0
                            and restart_allowed
                        )
                    )
                    if can_restart:
                        ok, restart_tail = _restart_profile_process(root, profile_id)
                        restarted = ok and _profile_running(root, profile_id)
                        apply_state = "written_restarted" if restarted else "written_restart_failed"
                        if restarted:
                            _record_restart(now_ts)
                    else:
                        apply_state = "written_restart_deferred"
                        decision_trace.append(
                            "restart_deferred "
                            f"cooldown_left={round(float(restart_cooldown_left), 2)} "
                            f"restart_budget_left={int(restart_budget_left)} "
                            f"open_positions={int(metrics.open_positions)} warmup={bool(warmup_guard_active)}"
                        )
                elif runtime_patch_sync_ok:
                    apply_state = "written_hot_applied"
    elif not dry_run:
        target_overrides = dict(overrides)
        env_sync_ok, env_sync_detail = _sync_profile_env_overrides(root, profile_id, target_overrides)
        if not env_sync_ok:
            issues.append(str(env_sync_detail))
            apply_state = "env_sync_failed"
        else:
            runtime_patch_sync_ok, runtime_patch_sync_detail = _write_runtime_patch_overrides(
                root=root,
                profile_id=profile_id,
                overrides=target_overrides,
            )
            if not runtime_patch_sync_ok:
                issues.append(str(runtime_patch_sync_detail))
                if apply_state == "noop":
                    apply_state = "runtime_patch_sync_failed"

    if not dry_run and apply_state in {
        "noop",
        "written_restart_deferred",
        "written_hot_applied",
        "runtime_patch_sync_failed",
        "written_runtime_patch_sync_failed",
    }:
        active_now = _active_override_subset(root, profile_id, set(target_overrides.keys()))
        diff_keys = _diff_override_keys(active_now, target_overrides)
        pending_runtime_diff_keys = list(diff_keys)
        pending_restart_diff_keys, pending_hot_apply_diff_keys = _partition_pending_diff_keys(pending_runtime_diff_keys)
        if pending_hot_apply_diff_keys:
            decision_trace.append(
                "pending_hot_apply "
                f"diff_keys={len(pending_hot_apply_diff_keys)}"
            )
        if pending_restart_diff_keys:
            now_ts = time.time()
            restart_allowed, restart_cooldown_left, restart_budget_left = _can_restart(now_ts)
            can_restart = (
                (not warmup_guard_active)
                and (
                    metrics.open_positions <= 0
                    and restart_allowed
                )
            )
            if can_restart:
                ok, restart_tail = _restart_profile_process(root, profile_id)
                restarted = ok and _profile_running(root, profile_id)
                apply_state = "restart_applied" if restarted else "restart_failed"
                if restarted:
                    _record_restart(now_ts)
                    decision_trace.append(
                        f"pending_restart_applied diff_keys={len(pending_restart_diff_keys)}"
                    )
            else:
                if apply_state == "noop":
                    apply_state = "restart_pending_deferred"
                decision_trace.append(
                    "pending_restart_deferred "
                    f"diff_keys={len(pending_restart_diff_keys)} "
                    f"open_positions={metrics.open_positions} "
                    f"cooldown_left={round(float(restart_cooldown_left), 2)} "
                    f"restart_budget_left={int(restart_budget_left)} "
                    f"warmup={bool(warmup_guard_active)}"
                )
    if not dry_run and metrics.open_positions <= 0:
        alive = _profile_running(root, profile_id)
        hb_recent = _heartbeat_recent(root, profile_id, max_age_seconds=90.0)
        if (not alive) and (not hb_recent):
            now_ts = time.time()
            can_restart, restart_cooldown_left, restart_budget_left = _can_restart(now_ts)
            if can_restart:
                ok, restart_tail = _run_dead_profile_recovery(root, profile_id)
                restarted = ok and _profile_running(root, profile_id)
                if restarted:
                    _record_restart(now_ts)
                    if apply_state == "noop":
                        apply_state = "restart_applied_dead_profile"
                else:
                    if apply_state == "noop":
                        apply_state = "restart_failed_dead_profile"
            else:
                if apply_state == "noop":
                    apply_state = "restart_dead_profile_deferred"
            decision_trace.append(
                "dead_profile_recovery "
                f"alive={alive} hb_recent={hb_recent} can_restart={can_restart} "
                f"cooldown_left={round(float(restart_cooldown_left), 2)} "
                f"restart_budget_left={int(restart_budget_left)}"
            )

    if not dry_run and restarted:
        if not _update_active_matrix_overrides(root, profile_id, target_overrides):
            issues.append("active_matrix_overrides_sync_failed")

    telemetry_v2 = _collect_telemetry_v2(
        root=root,
        profile_id=profile_id,
        config_before=overrides,
        config_after=target_overrides,
        actions=actions,
        metrics=metrics,
    )
    if not dry_run:
        active_now = _active_override_subset(root, profile_id, set(target_overrides.keys()))
        pending_runtime_diff_keys = _diff_override_keys(active_now, target_overrides)
        pending_restart_diff_keys, pending_hot_apply_diff_keys = _partition_pending_diff_keys(
            pending_runtime_diff_keys
        )
    tuner_effective = (
        bool(actions)
        and not bool(dry_run)
        and _overrides_hash(overrides) != _overrides_hash(target_overrides)
        and bool(env_sync_ok)
        and bool(runtime_patch_sync_ok)
        and len(pending_restart_diff_keys) == 0
    )
    telemetry_v2["tuner_effective"] = bool(tuner_effective)
    telemetry_v2["pending_runtime_diff_keys"] = [str(x) for x in pending_runtime_diff_keys[:64]]
    telemetry_v2["pending_restart_diff_keys"] = [str(x) for x in pending_restart_diff_keys[:64]]
    telemetry_v2["pending_hot_apply_diff_keys"] = [str(x) for x in pending_hot_apply_diff_keys[:64]]
    telemetry_v2["changed_restart_required_keys"] = [str(x) for x in changed_restart_required_keys[:64]]
    restart_guard_ok, restart_cooldown_left_now, restart_budget_left_now = _can_restart(float(time.time()))
    telemetry_v2["restart_guard"] = {
        "allowed_now": bool(restart_guard_ok),
        "cooldown_left_seconds": round(float(restart_cooldown_left_now), 3),
        "restart_budget_left_hour": int(restart_budget_left_now),
        "restart_history_len_hour": int(len(restart_history)),
        "restart_cooldown_seconds": int(max(0, restart_cooldown_seconds)),
        "restart_max_per_hour": int(max(1, restart_max_per_hour)),
    }
    telemetry_v2["runtime_patch_sync_ok"] = bool(runtime_patch_sync_ok)
    if runtime_patch_sync_detail:
        telemetry_v2["runtime_patch_sync_detail"] = str(runtime_patch_sync_detail)
    telemetry_v2["policy_phase"] = str(policy_decision.phase)
    telemetry_v2["policy_snapshot"] = {
        "target_trades_per_hour": float(target_policy.target_trades_per_hour),
        "target_trades_per_hour_requested": float(policy_decision.target_trades_per_hour_requested),
        "target_trades_per_hour_effective": float(policy_decision.target_trades_per_hour_effective),
        "target_pnl_per_hour_usd": float(target_policy.target_pnl_per_hour_usd),
        "throughput_est_trades_h": float(policy_decision.throughput_est_trades_h),
        "open_rate_15m": float(policy_decision.open_rate_15m),
        "pnl_hour_usd": float(policy_decision.pnl_hour_usd),
        "pre_risk_fail": bool(policy_decision.pre_risk_fail),
        "diversity_fail": bool(policy_decision.diversity_fail),
        "diversity_buys_15m": int(metrics.autobuy_total),
        "diversity_unique_symbols_15m": int(metrics.unique_buy_symbols),
        "diversity_top1_open_share_15m": float(metrics.top_buy_symbol_share),
        "blacklist_added_15m": int(policy_decision.blacklist_added_15m),
        "blacklist_share_15m": float(policy_decision.blacklist_share_15m),
        "adaptive_target_pre": effective_target_meta_pre,
    }

    state.last_phase = str(policy_decision.phase)
    if (
        policy_decision.risk_fail
        or policy_decision.pre_risk_fail
        or policy_decision.blacklist_fail
        or policy_decision.diversity_fail
    ):
        state.degrade_streak = max(0, int(state.degrade_streak) + 1)
    else:
        state.degrade_streak = 0
    if bool(tuner_effective):
        state.last_effective_hash = _overrides_hash(target_overrides)
        state.last_effective_at = _now_iso()
    effective_target_meta_post = _update_effective_target_state(
        state=state,
        target=target_policy,
        decision=policy_decision,
        metrics=metrics,
    )
    policy_snapshot = telemetry_v2.get("policy_snapshot", {})
    if isinstance(policy_snapshot, dict):
        policy_snapshot["adaptive_target_post"] = effective_target_meta_post
        telemetry_v2["policy_snapshot"] = policy_snapshot
    healthy_snapshot = (
        policy_decision.phase == "hold"
        and not policy_decision.risk_fail
        and not policy_decision.pre_risk_fail
        and not policy_decision.blacklist_fail
        and not policy_decision.diversity_fail
        and int(metrics.selected_from_batch) >= int(target_policy.min_selected_15m)
        and int(metrics.opened_from_batch) >= 1
    )
    if healthy_snapshot:
        stable_source = dict(target_overrides if tuner_effective else overrides)
        state.stable_overrides = {
            str(k): str(v)
            for k, v in stable_source.items()
            if str(k) in TUNER_MUTABLE_KEYS and str(k) not in TUNER_EPHEMERAL_KEYS
        }
        state.stable_hash = _overrides_hash(state.stable_overrides)
    state.restart_history_ts = list(restart_history)
    _save_runtime_state(root, profile_id, state)

    if policy_decision.reasons:
        decision_trace.extend(policy_decision.reasons)
    if blocked_actions:
        decision_trace.append(f"blocked_actions={len(blocked_actions)}")
    if capped_actions:
        decision_trace.append(f"delta_capped_actions={len(capped_actions)}")
    decision_meta = dict(decision_meta or {})
    decision_meta["policy_phase"] = str(policy_decision.phase)
    decision_meta["policy"] = {
        "flow_fail": bool(policy_decision.flow_fail),
        "pre_risk_fail": bool(policy_decision.pre_risk_fail),
        "risk_fail": bool(policy_decision.risk_fail),
        "blacklist_fail": bool(policy_decision.blacklist_fail),
        "diversity_fail": bool(policy_decision.diversity_fail),
        "degrade_streak": int(state.degrade_streak),
        "rollback_degrade_streak": int(target_policy.rollback_degrade_streak),
        "target_trades_per_hour_requested": float(policy_decision.target_trades_per_hour_requested),
        "target_trades_per_hour_effective": float(policy_decision.target_trades_per_hour_effective),
        "adaptive_target": effective_target_meta_post,
    }
    decision_meta["runtime_apply"] = {
        "env_sync_ok": bool(env_sync_ok),
        "env_sync_detail": str(env_sync_detail or ""),
        "runtime_patch_sync_ok": bool(runtime_patch_sync_ok),
        "runtime_patch_sync_detail": str(runtime_patch_sync_detail or ""),
        "restart_required_changed_keys": [str(x) for x in changed_restart_required_keys[:64]],
        "pending_restart_diff_keys": [str(x) for x in pending_restart_diff_keys[:64]],
        "pending_hot_apply_diff_keys": [str(x) for x in pending_hot_apply_diff_keys[:64]],
    }
    decision_meta["blocked_actions"] = int(len(blocked_actions))
    decision_meta["delta_capped_actions"] = int(len(capped_actions))
    if rollback_triggered:
        decision_meta["rollback_triggered"] = True

    report = {
        "ts": _now_iso(),
        "profile_id": profile_id,
        "mode": mode.name,
        "policy_phase": policy_decision.phase,
        "session": str(session),
        "window_minutes": int(window_minutes),
        "metrics": {
            "lines_seen": metrics.lines_seen,
            "scanned": metrics.scanned,
            "trade_candidates": metrics.trade_candidates,
            "opened_from_summary": metrics.opened_from_summary,
            "selected_from_batch": metrics.selected_from_batch,
            "opened_from_batch": metrics.opened_from_batch,
            "open_positions": metrics.open_positions,
            "total_closed": metrics.total_closed,
            "winrate_total": metrics.winrate_total,
            "realized_total": metrics.realized_total,
            "autobuy_total": metrics.autobuy_total,
            "unique_buy_symbols": metrics.unique_buy_symbols,
            "top_buy_symbol": metrics.top_buy_symbol,
            "top_buy_symbol_count": metrics.top_buy_symbol_count,
            "top_buy_symbol_share": round(metrics.top_buy_symbol_share, 6),
            "top_buy_symbols": metrics.buy_symbol_counts.most_common(6),
            "symbol_concentration_drop_hits": metrics.symbol_concentration_drop_hits,
            "top_filter_fail": metrics.filter_fail_reasons.most_common(6),
            "top_autotrade_skip": metrics.autotrade_skip_reasons.most_common(6),
            "top_blacklist_details": metrics.blacklist_detail_reasons.most_common(6),
            "pe_reasons": metrics.pe_reasons.most_common(4),
        },
        "actions": [
            {
                "key": a.key,
                "old": a.old_value,
                "new": a.new_value,
                "reason": a.reason,
            }
            for a in actions
        ],
        "blocked_actions": blocked_actions,
        "delta_capped_actions": capped_actions,
        "rollback_triggered": bool(rollback_triggered),
        "apply_state": apply_state,
        "issues": issues,
        "restarted": bool(restarted),
        "restart_tail": restart_tail,
        "decision_trace": decision_trace,
        "decision_meta": decision_meta,
        "telemetry_v2": telemetry_v2,
    }
    _append_runtime_log(_runtime_log_path(root, profile_id), report)
    return report, last_restart_ts


def _print_tick(report: dict[str, Any]) -> None:
    m = report.get("metrics", {})
    t2 = report.get("telemetry_v2", {}) or {}
    ps = t2.get("policy_snapshot", {}) or {}
    target_eff = ps.get("target_trades_per_hour_effective")
    target_req = ps.get("target_trades_per_hour_requested")
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(
        f"[{ts}] [{report.get('profile_id')}] "
        f"mode={report.get('mode')} phase={report.get('policy_phase')} state={report.get('apply_state')}"
    )
    print(
        "  flow15 "
        f"scanned={m.get('scanned')} candidates={m.get('trade_candidates')} "
        f"selected={m.get('selected_from_batch')} opened={m.get('opened_from_batch')} "
        f"open_positions={m.get('open_positions')} closed_total={m.get('total_closed')}"
    )
    print(
        "  diversity15 "
        f"buy_total={m.get('autobuy_total')} unique_buy={m.get('unique_buy_symbols')} "
        f"top_symbol={m.get('top_buy_symbol')} top_share={m.get('top_buy_symbol_share')}"
    )
    print(f"  target_tph effective={target_eff} requested={target_req}")
    actions = report.get("actions", [])
    if actions:
        print("  actions")
        for row in actions:
            print(f"  - {row['key']}: {row['old']} -> {row['new']} ({row['reason']})")
    issues = report.get("issues") or []
    if issues:
        print("  issues")
        for issue in issues:
            print(f"  ! {issue}")
    funnel = t2.get("funnel_15m", {}) or {}
    if funnel:
        print(
            "  telemetry15 "
            f"raw={funnel.get('raw')} src={funnel.get('source')} pre={funnel.get('pre')} "
            f"thr={funnel.get('thr')} q={funnel.get('quarantine')} exec={funnel.get('exec')} "
            f"buy={funnel.get('buy')}"
        )
    top_plan = ((t2.get("top_reasons_15m", {}) or {}).get("plan_skip", []) or [])[:1]
    if top_plan:
        row = top_plan[0]
        print(f"  top_plan_skip {row.get('reason_code')} count={row.get('count')}")
    trace = report.get("decision_trace") or []
    if trace:
        print(f"  decision {trace[0]}")
    blocked = report.get("blocked_actions") or []
    if blocked:
        print(f"  blocked_actions={len(blocked)}")
    capped = report.get("delta_capped_actions") or []
    if capped:
        print(f"  delta_capped={len(capped)}")


def cmd_once(args: argparse.Namespace) -> int:
    root = _project_root(args.root)
    ok, lock_msg = _acquire_runtime_lock(
        root=root,
        profile_id=args.profile_id,
        owner_label=f"runtime_tuner once {args.profile_id}",
    )
    if not ok:
        print(lock_msg)
        return 2
    mode = MODE_SPECS[args.mode]
    target_policy = _target_policy_from_args(args)
    try:
        report, _ = _tick(
            root=root,
            profile_id=args.profile_id,
            mode=mode,
            target_policy=target_policy,
            forced_phase=str(args.policy_phase),
            window_minutes=args.window_minutes,
            dry_run=bool(args.dry_run),
            restart_cooldown_seconds=args.restart_cooldown_seconds,
            restart_max_per_hour=args.restart_max_per_hour,
            allow_zero_cooldown=bool(args.allow_zero_cooldown),
            last_restart_ts=0.0,
        )
        _print_tick(report)
        return 0
    finally:
        _release_runtime_lock(root=root, profile_id=args.profile_id)


def cmd_run(args: argparse.Namespace) -> int:
    root = _project_root(args.root)
    ok, lock_msg = _acquire_runtime_lock(
        root=root,
        profile_id=args.profile_id,
        owner_label=f"runtime_tuner run {args.profile_id}",
    )
    if not ok:
        print(lock_msg)
        return 2
    mode = MODE_SPECS[args.mode]
    target_policy = _target_policy_from_args(args)
    duration_seconds = max(60, int(args.duration_minutes) * 60)
    interval_seconds = max(30, int(args.interval_seconds))
    deadline = time.time() + float(duration_seconds)
    last_restart_ts = 0.0
    try:
        while time.time() < deadline:
            report, last_restart_ts = _tick(
                root=root,
                profile_id=args.profile_id,
                mode=mode,
                target_policy=target_policy,
                forced_phase=str(args.policy_phase),
                window_minutes=args.window_minutes,
                dry_run=bool(args.dry_run),
                restart_cooldown_seconds=args.restart_cooldown_seconds,
                restart_max_per_hour=args.restart_max_per_hour,
                allow_zero_cooldown=bool(args.allow_zero_cooldown),
                last_restart_ts=last_restart_ts,
            )
            _print_tick(report)
            if time.time() >= deadline:
                break
            time.sleep(float(interval_seconds))
        return 0
    finally:
        _release_runtime_lock(root=root, profile_id=args.profile_id)


def cmd_replay(args: argparse.Namespace) -> int:
    root = _project_root(args.root)
    path = _runtime_log_path(root, args.profile_id)
    rows = _load_runtime_rows(path, limit=max(1, int(args.limit)))
    if not rows:
        print(f"No runtime rows found: {path}")
        return 1
    summary = _summarize_runtime_rows(rows)
    if bool(args.json):
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    print(
        f"[{args.profile_id}] replay rows={summary.get('ticks')} action_ticks={summary.get('action_ticks')} "
        f"avg_raw15={summary.get('funnel_15m_avg_raw')} avg_buy15={summary.get('funnel_15m_avg_buy')}"
    )
    for key, value in summary.get("apply_states", [])[:6]:
        print(f"  apply_state {key}: {value}")
    for key, value in summary.get("policy_phases", [])[:4]:
        print(f"  policy_phase {key}: {value}")
    for key, value in summary.get("top_action_keys", [])[:6]:
        print(f"  action_key {key}: {value}")
    for key, value in summary.get("blocked_action_keys", [])[:6]:
        print(f"  blocked_key {key}: {value}")
    for key, value in summary.get("rule_hits", [])[:6]:
        print(f"  rule_hit {key}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Matrix runtime preset tuner (safe sidecar for pre-live calibration)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--profile-id", required=True, help="Matrix profile id (must be a user preset).")
        sp.add_argument("--mode", choices=MODE_CHOICES, default="conveyor")
        sp.add_argument("--policy-phase", choices=STATE_MACHINE_CHOICES, default="auto")
        sp.add_argument("--window-minutes", type=int, default=12, help="Metrics lookback window.")
        sp.add_argument("--restart-cooldown-seconds", type=int, default=180, help="Min seconds between restarts.")
        sp.add_argument(
            "--restart-max-per-hour",
            type=int,
            default=6,
            help="Hard cap for restarts in rolling 1h window.",
        )
        sp.add_argument(
            "--allow-zero-cooldown",
            action="store_true",
            help="Allow MAX_TOKEN_COOLDOWN_SECONDS=0 (disabled by default for anti-churn safety).",
        )
        sp.add_argument("--target-policy-file", default="", help="Optional JSON file with target policy overrides.")
        sp.add_argument("--target-trades-per-hour", type=float, default=12.0)
        sp.add_argument("--target-pnl-per-hour-usd", type=float, default=0.05)
        sp.add_argument("--min-open-rate-15m", type=float, default=0.04)
        sp.add_argument("--min-selected-15m", type=int, default=16)
        sp.add_argument("--min-closed-for-risk-checks", type=int, default=6)
        sp.add_argument("--min-winrate-closed-15m", type=float, default=0.35)
        sp.add_argument("--max-blacklist-share-15m", type=float, default=0.45)
        sp.add_argument("--max-blacklist-added-15m", type=int, default=80)
        sp.add_argument("--pre-risk-min-plan-attempts-15m", type=int, default=8)
        sp.add_argument("--pre-risk-route-fail-rate-15m", type=float, default=0.35)
        sp.add_argument("--pre-risk-buy-fail-rate-15m", type=float, default=0.35)
        sp.add_argument("--pre-risk-sell-fail-rate-15m", type=float, default=0.30)
        sp.add_argument("--pre-risk-roundtrip-loss-median-pct-15m", type=float, default=-1.2)
        sp.add_argument("--tail-loss-min-closes-60m", type=int, default=6)
        sp.add_argument("--tail-loss-ratio-max", type=float, default=8.0)
        sp.add_argument("--rollback-degrade-streak", type=int, default=3)
        sp.add_argument("--hold-hysteresis-open-rate", type=float, default=0.07)
        sp.add_argument("--hold-hysteresis-trades-per-hour", type=float, default=6.0)
        sp.add_argument(
            "--adaptive-target-enabled",
            default="true",
            help="Enable adaptive effective trades/hour target (true|false).",
        )
        sp.add_argument("--adaptive-target-floor-trades-per-hour", type=float, default=4.0)
        sp.add_argument("--adaptive-target-step-up-trades-per-hour", type=float, default=1.5)
        sp.add_argument("--adaptive-target-step-down-trades-per-hour", type=float, default=3.0)
        sp.add_argument("--adaptive-target-headroom-mult", type=float, default=1.35)
        sp.add_argument("--adaptive-target-headroom-add-trades-per-hour", type=float, default=2.0)
        sp.add_argument("--adaptive-target-stable-ticks-for-step-up", type=int, default=2)
        sp.add_argument("--adaptive-target-fail-ticks-for-step-down", type=int, default=2)
        sp.add_argument("--dry-run", action="store_true", help="Analyze and propose actions without writing preset.")
        sp.add_argument("--root", default="", help="Project root (auto-detected by default).")

    p_once = sub.add_parser("once", help="Run one tuning tick.")
    add_common(p_once)
    p_once.set_defaults(fn=cmd_once)

    p_run = sub.add_parser("run", help="Run tuning loop for fixed duration.")
    add_common(p_run)
    p_run.add_argument("--duration-minutes", type=int, default=60, help="Controller runtime.")
    p_run.add_argument("--interval-seconds", type=int, default=120, help="Tick interval.")
    p_run.set_defaults(fn=cmd_run)

    p_replay = sub.add_parser("replay", help="Replay runtime_tuner.jsonl and print aggregate stats.")
    p_replay.add_argument("--profile-id", required=True, help="Matrix profile id.")
    p_replay.add_argument("--limit", type=int, default=240, help="Last N rows to read from runtime log.")
    p_replay.add_argument("--json", action="store_true", help="Print summary JSON.")
    p_replay.add_argument("--root", default="", help="Project root (auto-detected by default).")
    p_replay.set_defaults(fn=cmd_replay)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
