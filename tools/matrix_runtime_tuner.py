from __future__ import annotations

import argparse
import hashlib
import json
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
TUNER_RUNTIME_FALLBACK_KEYS = {
    "MARKET_MODE_STRICT_SCORE",
    "MARKET_MODE_SOFT_SCORE",
    "SAFE_MIN_VOLUME_5M_USD",
    "MIN_EXPECTED_EDGE_PERCENT",
    "MIN_EXPECTED_EDGE_USD",
    "EV_FIRST_ENTRY_MIN_NET_USD",
    "EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD",
    "MAX_TOKEN_COOLDOWN_SECONDS",
    "MIN_TRADE_USD",
    "PAPER_TRADE_SIZE_MIN_USD",
    "PAPER_TRADE_SIZE_MAX_USD",
    "ENTRY_A_CORE_MIN_TRADE_USD",
    "AUTO_TRADE_TOP_N",
    "HEAVY_CHECK_DEDUP_TTL_SECONDS",
    "PLAN_MAX_SINGLE_SOURCE_SHARE",
    "PLAN_MAX_WATCHLIST_SHARE",
    "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
    "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
    "V2_SOURCE_QOS_SOURCE_CAPS",
    "V2_UNIVERSE_SOURCE_CAPS",
    "SOURCE_ROUTER_MIN_TRADES",
    "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY",
    "SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY",
    "TOKEN_EV_MEMORY_MIN_TRADES",
    "TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY",
    "TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY",
}
TUNER_MUTABLE_KEYS = {
    "AUTO_TRADE_TOP_N",
    "ENTRY_A_CORE_MIN_TRADE_USD",
    "EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD",
    "EV_FIRST_ENTRY_MIN_NET_USD",
    "HEAVY_CHECK_DEDUP_TTL_SECONDS",
    "MARKET_MODE_SOFT_SCORE",
    "MARKET_MODE_STRICT_SCORE",
    "MAX_TOKEN_COOLDOWN_SECONDS",
    "MIN_EXPECTED_EDGE_PERCENT",
    "MIN_EXPECTED_EDGE_USD",
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
    "V2_QUALITY_SYMBOL_MAX_SHARE",
    "V2_QUALITY_SYMBOL_MIN_ABS_CAP",
    "V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS",
    "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
    "V2_SOURCE_QOS_SOURCE_CAPS",
    "V2_UNIVERSE_NOVELTY_MIN_SHARE",
    "V2_UNIVERSE_SOURCE_CAPS",
}
FORCE_ACTION_REASONS = {"soft_must_be_leq_strict"}
PHASE_CHOICES = ("expand", "hold", "tighten")
STATE_MACHINE_CHOICES = ("auto",) + PHASE_CHOICES


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


@dataclass
class PolicyDecision:
    phase: str
    reasons: list[str]
    throughput_est_trades_h: float
    pnl_hour_usd: float
    blacklist_added_15m: int
    blacklist_share_15m: float
    open_rate_15m: float
    risk_fail: bool
    flow_fail: bool
    blacklist_fail: bool


@dataclass
class RuntimeState:
    degrade_streak: int = 0
    stable_hash: str = ""
    stable_overrides: dict[str, str] = field(default_factory=dict)
    last_phase: str = "expand"
    last_effective_hash: str = ""
    last_effective_at: str = ""



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
    return RuntimeState(
        degrade_streak=int(payload.get("degrade_streak", 0) or 0),
        stable_hash=str(payload.get("stable_hash", "") or ""),
        stable_overrides=stable_map,
        last_phase=str(payload.get("last_phase", "expand") or "expand"),
        last_effective_hash=str(payload.get("last_effective_hash", "") or ""),
        last_effective_at=str(payload.get("last_effective_at", "") or ""),
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
        "updated_at": _now_iso(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    for row in trade_rows:
        stage = str(row.get("decision_stage", "") or "").strip().lower()
        if stage != "trade_close":
            continue
        reason = str(row.get("reason", "") or "").strip().upper() or "UNKNOWN"
        close_reasons[reason] += 1
        try:
            pnl_sum += float(row.get("pnl_usd", 0.0) or 0.0)
        except Exception:
            pass
    total = int(sum(int(v) for v in close_reasons.values()))
    out: list[dict[str, Any]] = []
    for reason, count in close_reasons.most_common(12):
        c = int(count)
        share = (float(c) / float(total)) if total > 0 else 0.0
        out.append({"reason": str(reason), "count": c, "share": round(share, 6)})
    return {"total": int(total), "pnl_usd_sum": round(float(pnl_sum), 6), "distribution": out}


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
    )


def _resolve_policy_phase(
    *,
    metrics: WindowMetrics,
    telemetry: dict[str, Any],
    target: TargetPolicy,
    state: RuntimeState,
    forced_phase: str,
) -> PolicyDecision:
    funnel = telemetry.get("funnel_15m", {}) if isinstance(telemetry, dict) else {}
    exec_health = telemetry.get("exec_health_15m", {}) if isinstance(telemetry, dict) else {}
    exit_mix = telemetry.get("exit_mix_60m", {}) if isinstance(telemetry, dict) else {}
    blacklist = telemetry.get("blacklist_forensics_15m", {}) if isinstance(telemetry, dict) else {}
    buy_15m = int(_safe_float((funnel or {}).get("buy", 0), 0.0))
    throughput_est = float(buy_15m) * 4.0
    open_rate = _safe_float((exec_health or {}).get("open_rate", 0.0), 0.0)
    closes = int(_safe_float((exec_health or {}).get("closes", 0), 0.0))
    winrate = _safe_float((exec_health or {}).get("winrate_closed", 0.0), 0.0)
    pnl_hour = _safe_float((exit_mix or {}).get("pnl_usd_sum", 0.0), 0.0)
    blacklist_added = int(_safe_float((blacklist or {}).get("plan_skip_blacklist_15m", 0), 0.0))
    blacklist_share = _safe_float((blacklist or {}).get("plan_skip_blacklist_share_15m", 0.0), 0.0)

    reasons: list[str] = []
    flow_fail = (
        int(metrics.selected_from_batch) >= int(target.min_selected_15m)
        and (
            open_rate < float(target.min_open_rate_15m)
            or throughput_est < float(target.target_trades_per_hour) * 0.40
        )
    )
    risk_fail = (
        closes >= int(target.min_closed_for_risk_checks)
        and (winrate < float(target.min_winrate_closed_15m) or pnl_hour < float(target.target_pnl_per_hour_usd) * -1.0)
    )
    blacklist_fail = (
        blacklist_added >= int(target.max_blacklist_added_15m)
        or blacklist_share >= float(target.max_blacklist_share_15m)
    )

    if flow_fail:
        reasons.append(
            f"flow_fail open_rate={open_rate:.3f} tph_est={throughput_est:.2f} selected={metrics.selected_from_batch}"
        )
    if risk_fail:
        reasons.append(f"risk_fail winrate={winrate:.3f} closes={closes} pnl_h={pnl_hour:.4f}")
    if blacklist_fail:
        reasons.append(
            f"blacklist_fail added15={blacklist_added} share15={blacklist_share:.3f}"
        )

    prev_phase = str(state.last_phase or "expand")
    phase = "hold"
    if str(forced_phase or "auto") != "auto":
        phase = str(forced_phase)
        reasons.append(f"phase_forced={phase}")
    elif risk_fail or blacklist_fail:
        phase = "tighten"
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
        throughput_est_trades_h=round(float(throughput_est), 6),
        pnl_hour_usd=round(float(pnl_hour), 6),
        blacklist_added_15m=int(blacklist_added),
        blacklist_share_15m=round(float(blacklist_share), 6),
        open_rate_15m=round(float(open_rate), 6),
        risk_fail=bool(risk_fail),
        flow_fail=bool(flow_fail),
        blacklist_fail=bool(blacklist_fail),
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
        "min_trade_size",
        "low_throughput_expand_topn",
        "dedup_pressure",
        "route_pressure",
        "route_floor_normalize",
        "duplicate_choke",
        "anti_concentration",
        "quality_gate_symbol_concentration",
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
        "V2_QUALITY_SYMBOL_MAX_SHARE": (0.03, False),
        "V2_QUALITY_SYMBOL_MIN_ABS_CAP": (1.0, True),
        "V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS": (60.0, True),
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
        allow = False
        if family == "force":
            allow = True
        elif phase == "expand":
            allow = family in {"expand", "neutral"}
        elif phase == "tighten":
            allow = family in {"tighten", "neutral"}
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


def _adaptive_bounds(metrics: WindowMetrics, mode: ModeSpec) -> AdaptiveBounds:
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
        bounds.cooldown_floor = max(0, int(mode.cooldown_floor) - 10)
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
        "SOURCE_ROUTER_MIN_TRADES",
        "SOURCE_ROUTER_BAD_ENTRY_PROBABILITY",
        "SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY",
        "TOKEN_EV_MEMORY_MIN_TRADES",
        "TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY",
        "TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY",
    }
    medium = {
        "PLAN_MAX_WATCHLIST_SHARE",
        "PLAN_MIN_NON_WATCHLIST_PER_BATCH",
        "PLAN_MAX_SINGLE_SOURCE_SHARE",
        "V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE",
        "V2_UNIVERSE_NOVELTY_MIN_SHARE",
        "SYMBOL_CONCENTRATION_MAX_SHARE",
        "V2_QUALITY_SYMBOL_MAX_SHARE",
        "V2_QUALITY_SYMBOL_MIN_ABS_CAP",
        "V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS",
    }
    low = {
        "V2_SOURCE_QOS_SOURCE_CAPS",
        "V2_UNIVERSE_SOURCE_CAPS",
    }
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
) -> list[Action]:
    actions, _, _ = _build_action_plan(metrics=metrics, overrides=overrides, mode=mode)
    return actions


def _build_action_plan(
    *,
    metrics: WindowMetrics,
    overrides: dict[str, str],
    mode: ModeSpec,
) -> tuple[list[Action], list[str], dict[str, Any]]:
    staged = dict(overrides)
    actions: list[Action] = []
    trace: list[str] = []
    rule_hits: Counter[str] = Counter()
    bounds = _adaptive_bounds(metrics, mode)

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

    score_min_hits = int(metrics.filter_fail_reasons.get("score_min", 0))
    safe_volume_hits = int(metrics.filter_fail_reasons.get("safe_volume", 0))
    safe_age_hits = int(metrics.filter_fail_reasons.get("safe_age", 0))
    heavy_dedup_hits = int(metrics.filter_fail_reasons.get("heavy_dedup_ttl", 0))
    ev_low_hits = int(metrics.autotrade_skip_reasons.get("ev_net_low", 0))
    min_trade_hits = int(metrics.autotrade_skip_reasons.get("min_trade_size", 0))
    symbol_concentration_skip_hits = int(metrics.autotrade_skip_reasons.get("symbol_concentration", 0))
    source_route_prob_hits = int(metrics.autotrade_skip_reasons.get("source_route_prob", 0))
    token_ev_prob_hits = int(metrics.autotrade_skip_reasons.get("token_ev_memory_prob", 0))
    blacklist_hits = int(metrics.autotrade_skip_reasons.get("blacklist", 0))
    address_duplicate_hits = int(metrics.autotrade_skip_reasons.get("address_or_duplicate", 0))
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
    if plan_watchlist_share < 0.08:
        plan_watchlist_share = 0.08
        _apply_action(
            staged,
            actions,
            "PLAN_MAX_WATCHLIST_SHARE",
            _to_float_str(plan_watchlist_share),
            "route_floor_normalize watchlist_share",
        )
        normalized_route = True
    if plan_min_non_watchlist > 6:
        plan_min_non_watchlist = 6
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
        if watch_cap_qos > 0 and watch_cap_qos < 6:
            qos_caps["watchlist"] = 6
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
        if watch_cap_universe > 0 and watch_cap_universe < 6:
            universe_caps["watchlist"] = 6
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

    if route_pressure:
        plan_watchlist_share = _clamp_float(plan_watchlist_share - 0.05, 0.08, 0.60)
        plan_min_non_watchlist = _clamp_int(plan_min_non_watchlist + 1, 0, 6)
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
                watchlist_min=6,
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
                watchlist_min=6,
            )
            _apply_action(
                staged,
                actions,
                "V2_UNIVERSE_SOURCE_CAPS",
                _format_source_cap_map(universe_caps),
                "route_pressure universe_caps",
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
            cooldown = _clamp_int(cooldown - 15, bounds.cooldown_floor, bounds.cooldown_ceiling)
            rule_hits["relax_cooldown"] += 1
            trace.append(f"relax_cooldown hits={cooldown_hits}")
            _apply_action(staged, actions, "MAX_TOKEN_COOLDOWN_SECONDS", _to_int_str(cooldown), "cooldown_left")
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
            plan_watchlist_share = _clamp_float(plan_watchlist_share - 0.06, 0.08, 0.60)
            plan_min_non_watchlist = _clamp_int(plan_min_non_watchlist + 1, 0, 6)
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
                    watchlist_min=6,
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
                    watchlist_min=6,
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
        per_symbol_cycle_cap = _clamp_int(per_symbol_cycle_cap - 1, 1, 20)
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

    prioritized = sorted(actions, key=_action_priority)
    limited = prioritized[: mode.max_actions_per_tick]
    trimmed = max(0, len(prioritized) - len(limited))
    meta: dict[str, Any] = {"rule_hits": dict(rule_hits)}
    if trimmed > 0:
        meta["trimmed_actions"] = int(trimmed)
    return limited, trace, meta


def _validate_overrides(root: Path, overrides: dict[str, str]) -> list[str]:
    contract = load_contract(root)
    return validate_overrides(overrides, contract)


def _append_runtime_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def _profile_running(root: Path, profile_id: str) -> bool:
    path = _active_matrix_path(root)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not bool(payload.get("running")):
        return False
    for item in payload.get("items", []) or []:
        if str(item.get("id", "")).strip() == profile_id and str(item.get("status", "")).strip().lower() == "running":
            return True
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
    last_restart_ts: float,
) -> tuple[dict[str, Any], float]:
    session = _latest_session_log(root, profile_id)
    if session is None:
        raise SystemExit(f"No session logs for profile '{profile_id}'")
    metrics = _read_window_metrics(session, window_minutes=window_minutes)
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
    policy_decision = _resolve_policy_phase(
        metrics=metrics,
        telemetry=telemetry_before,
        target=target_policy,
        state=state,
        forced_phase=forced_phase,
    )

    actions, decision_trace, decision_meta = _build_action_plan(metrics=metrics, overrides=overrides, mode=mode)
    actions, blocked_mutations = _enforce_mutable_action_keys(actions=actions, protected_keys=protected_keys)
    actions, blocked_phase = _filter_actions_by_phase(actions=actions, phase=policy_decision.phase)
    actions, capped_actions = _apply_action_delta_caps(actions=actions, phase=policy_decision.phase)

    blocked_actions = blocked_mutations + blocked_phase
    rollback_triggered = False
    projected_degrade_streak = state.degrade_streak + 1 if (policy_decision.risk_fail or policy_decision.blacklist_fail) else 0
    current_mutable = {str(k): str(v) for k, v in overrides.items() if str(k) in TUNER_MUTABLE_KEYS}
    current_hash = _overrides_hash(current_mutable)
    if (
        policy_decision.phase == "tighten"
        and projected_degrade_streak >= int(target_policy.rollback_degrade_streak)
        and bool(state.stable_overrides)
        and str(state.stable_hash or "")
        and str(state.stable_hash or "") != current_hash
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
    issues: list[str] = []
    restarted = False
    restart_tail = ""
    apply_state = "noop"
    target_overrides: dict[str, str] = dict(overrides)
    pending_runtime_diff_keys: list[str] = []

    if actions:
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
            now_ts = time.time()
            can_restart = (
                metrics.open_positions <= 0
                and (now_ts - float(last_restart_ts)) >= float(max(0, restart_cooldown_seconds))
            )
            if can_restart:
                ok, restart_tail = _run_launcher(root, profile_id)
                restarted = ok and _profile_running(root, profile_id)
                apply_state = "written_restarted" if restarted else "written_restart_failed"
                if restarted:
                    last_restart_ts = now_ts
            else:
                apply_state = "written_restart_deferred"
    elif not dry_run:
        target_overrides = dict(overrides)

    if not dry_run and apply_state in {"noop", "written_restart_deferred"}:
        active_now = _active_override_subset(root, profile_id, set(target_overrides.keys()))
        diff_keys = _diff_override_keys(active_now, target_overrides)
        if diff_keys:
            now_ts = time.time()
            can_restart = (
                metrics.open_positions <= 0
                and (now_ts - float(last_restart_ts)) >= float(max(0, restart_cooldown_seconds))
            )
            if can_restart:
                ok, restart_tail = _run_launcher(root, profile_id)
                restarted = ok and _profile_running(root, profile_id)
                apply_state = "restart_applied" if restarted else "restart_failed"
                if restarted:
                    last_restart_ts = now_ts
                    decision_trace.append(f"pending_restart_applied diff_keys={len(diff_keys)}")
            else:
                if apply_state == "noop":
                    apply_state = "restart_pending_deferred"
                decision_trace.append(
                    "pending_restart_deferred "
                    f"diff_keys={len(diff_keys)} open_positions={metrics.open_positions}"
                )
                pending_runtime_diff_keys = diff_keys

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
    tuner_effective = (
        bool(actions)
        and not bool(dry_run)
        and _overrides_hash(overrides) != _overrides_hash(target_overrides)
        and len(pending_runtime_diff_keys) == 0
    )
    telemetry_v2["tuner_effective"] = bool(tuner_effective)
    telemetry_v2["pending_runtime_diff_keys"] = [str(x) for x in pending_runtime_diff_keys[:64]]
    telemetry_v2["policy_phase"] = str(policy_decision.phase)
    telemetry_v2["policy_snapshot"] = {
        "target_trades_per_hour": float(target_policy.target_trades_per_hour),
        "target_pnl_per_hour_usd": float(target_policy.target_pnl_per_hour_usd),
        "throughput_est_trades_h": float(policy_decision.throughput_est_trades_h),
        "open_rate_15m": float(policy_decision.open_rate_15m),
        "pnl_hour_usd": float(policy_decision.pnl_hour_usd),
        "blacklist_added_15m": int(policy_decision.blacklist_added_15m),
        "blacklist_share_15m": float(policy_decision.blacklist_share_15m),
    }

    state.last_phase = str(policy_decision.phase)
    if policy_decision.risk_fail or policy_decision.blacklist_fail:
        state.degrade_streak = max(0, int(state.degrade_streak) + 1)
    else:
        state.degrade_streak = 0
    if bool(tuner_effective):
        state.last_effective_hash = _overrides_hash(target_overrides)
        state.last_effective_at = _now_iso()
    healthy_snapshot = (
        policy_decision.phase == "hold"
        and not policy_decision.risk_fail
        and not policy_decision.blacklist_fail
        and int(metrics.selected_from_batch) >= int(target_policy.min_selected_15m)
        and int(metrics.opened_from_batch) >= 1
    )
    if healthy_snapshot:
        stable_source = dict(target_overrides if tuner_effective else overrides)
        state.stable_overrides = {str(k): str(v) for k, v in stable_source.items() if str(k) in TUNER_MUTABLE_KEYS}
        state.stable_hash = _overrides_hash(state.stable_overrides)
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
        "risk_fail": bool(policy_decision.risk_fail),
        "blacklist_fail": bool(policy_decision.blacklist_fail),
        "degrade_streak": int(state.degrade_streak),
        "rollback_degrade_streak": int(target_policy.rollback_degrade_streak),
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
    print(
        f"[{report.get('profile_id')}] mode={report.get('mode')} "
        f"phase={report.get('policy_phase')} "
        f"scan={m.get('scanned')} cand={m.get('trade_candidates')} "
        f"selected={m.get('selected_from_batch')} open={m.get('opened_from_batch')} "
        f"buy_total={m.get('autobuy_total')} uniq_buy={m.get('unique_buy_symbols')} "
        f"top={m.get('top_buy_symbol')}@{m.get('top_buy_symbol_share')} "
        f"state={report.get('apply_state')}"
    )
    actions = report.get("actions", [])
    if actions:
        for row in actions:
            print(f"  - {row['key']}: {row['old']} -> {row['new']} ({row['reason']})")
    issues = report.get("issues") or []
    if issues:
        for issue in issues:
            print(f"  ! {issue}")
    t2 = report.get("telemetry_v2", {}) or {}
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
    mode = MODE_SPECS[args.mode]
    target_policy = _target_policy_from_args(args)
    report, _ = _tick(
        root=root,
        profile_id=args.profile_id,
        mode=mode,
        target_policy=target_policy,
        forced_phase=str(args.policy_phase),
        window_minutes=args.window_minutes,
        dry_run=bool(args.dry_run),
        restart_cooldown_seconds=args.restart_cooldown_seconds,
        last_restart_ts=0.0,
    )
    _print_tick(report)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = _project_root(args.root)
    mode = MODE_SPECS[args.mode]
    target_policy = _target_policy_from_args(args)
    duration_seconds = max(60, int(args.duration_minutes) * 60)
    interval_seconds = max(30, int(args.interval_seconds))
    deadline = time.time() + float(duration_seconds)
    last_restart_ts = 0.0
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
            last_restart_ts=last_restart_ts,
        )
        _print_tick(report)
        if time.time() >= deadline:
            break
        time.sleep(float(interval_seconds))
    return 0


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
        sp.add_argument("--target-policy-file", default="", help="Optional JSON file with target policy overrides.")
        sp.add_argument("--target-trades-per-hour", type=float, default=12.0)
        sp.add_argument("--target-pnl-per-hour-usd", type=float, default=0.05)
        sp.add_argument("--min-open-rate-15m", type=float, default=0.04)
        sp.add_argument("--min-selected-15m", type=int, default=16)
        sp.add_argument("--min-closed-for-risk-checks", type=int, default=6)
        sp.add_argument("--min-winrate-closed-15m", type=float, default=0.35)
        sp.add_argument("--max-blacklist-share-15m", type=float, default=0.45)
        sp.add_argument("--max-blacklist-added-15m", type=int, default=80)
        sp.add_argument("--rollback-degrade-streak", type=int, default=3)
        sp.add_argument("--hold-hysteresis-open-rate", type=float, default=0.07)
        sp.add_argument("--hold-hysteresis-trades-per-hour", type=float, default=6.0)
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
