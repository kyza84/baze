"""Local fast runner without Telegram polling/webhooks."""

from __future__ import annotations

import asyncio
import atexit
import ctypes
import hashlib
import json
import logging
import os
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

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
from trading import runtime_policy
from trading.v2_runtime import (
    DualEntryController,
    MatrixChampionGuard,
    PolicyEntryRouter,
    RollingEdgeGovernor,
    RuntimeKpiLoop,
    SafetyBudgetController,
    SourceQosController,
    UnifiedCalibrator,
    UniverseFlowController,
    UniverseQualityGateController,
)
from utils.log_contracts import candidate_decision_event
from utils.addressing import normalize_address

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_RAW_INSTANCE_ID = str(getattr(config, "BOT_INSTANCE_ID", "") or "").strip()
RUN_TAG = str(getattr(config, "RUN_TAG", _RAW_INSTANCE_ID) or _RAW_INSTANCE_ID or "single").strip()
INSTANCE_ID = _RAW_INSTANCE_ID or (RUN_TAG if RUN_TAG and RUN_TAG.lower() != "single" else "")
WINDOWS_MUTEX_NAME = (
    f"Global\\solana_alert_bot_main_local_{INSTANCE_ID}"
    if INSTANCE_ID
    else "Global\\solana_alert_bot_main_local_single_instance"
)
PID_FILE = os.path.join(PROJECT_ROOT, "bot.pid")
HEARTBEAT_FILE = os.path.join(LOG_DIR, "heartbeat.json")
_RUNTIME_TUNER_LOCK_STATE: dict[str, float | int | bool] = {
    "checked_at": 0.0,
    "active": False,
    "pid": 0,
}
_RUNTIME_TUNER_PATCH_STATE: dict[str, Any] = {
    "checked_mtime": 0.0,
    "applied_hash": "",
    "last_error": "",
    "applied_keys": [],
}
_RUNTIME_TUNER_RELOAD_PREFIX_MAP: dict[str, str] = {
    "V2_UNIVERSE_": "universe",
    "V2_SOURCE_QOS_": "source_qos",
    "V2_QUALITY_": "quality_gate",
    "V2_SAFETY_BUDGET_": "safety_budget",
    "V2_CALIBRATION_": "calibrator",
    "V2_ROLLING_EDGE_": "rolling_edge",
    "V2_ANTI_SELF_TIGHTEN_": "rolling_edge",
    "V2_ENTRY_": "dual_entry",
    "V2_POLICY_": "policy_router",
    "V2_KPI_": "kpi_loop",
}
_RUNTIME_TUNER_RELOAD_EXACT_MAP: dict[str, str] = {
    "PROFIT_ENGINE_ENABLED": "kpi_loop",
}


def _should_manage_single_pid_file() -> bool:
    # Matrix workers use per-instance metadata, not shared bot.pid.
    return not str(INSTANCE_ID or "").strip().lower().startswith("mx")


def _write_single_pid_file() -> None:
    if not _should_manage_single_pid_file():
        return
    try:
        with open(PID_FILE, "w", encoding="ascii") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass


def _clear_single_pid_file() -> None:
    if not _should_manage_single_pid_file():
        return
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE, "r", encoding="ascii") as f:
                raw = str(f.read() or "").strip()
            if raw.isdigit() and int(raw) == os.getpid():
                os.remove(PID_FILE)
    except Exception:
        pass


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


def _read_graceful_stop_meta() -> dict[str, str]:
    path = _graceful_stop_file_path()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = str(f.read() or "").strip()
    except Exception:
        return {}
    if not raw:
        return {}

    def _to_text(value: Any) -> str:
        return str(value or "").strip()

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            source = _to_text(payload.get("source")) or _to_text(payload.get("event")) or "unknown"
            reason = _to_text(payload.get("reason")) or "unknown"
            actor = _to_text(payload.get("actor")) or "unknown"
            ts = _to_text(payload.get("timestamp")) or _to_text(payload.get("ts"))
            return {
                "source": source,
                "reason": reason,
                "actor": actor,
                "timestamp": ts,
                "raw": "",
            }
    except Exception:
        pass

    first = raw.splitlines()[0].strip() if raw else ""
    tokens = first.split(maxsplit=1) if first else []
    legacy_ts = tokens[0] if len(tokens) >= 1 else ""
    legacy_reason = tokens[1] if len(tokens) >= 2 else (first or "legacy_signal")
    return {
        "source": "legacy_signal_file",
        "reason": legacy_reason or "legacy_signal",
        "actor": "unknown",
        "timestamp": legacy_ts,
        "raw": first[:180],
    }


def _clear_graceful_stop_flag() -> None:
    try:
        path = _graceful_stop_file_path()
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _write_heartbeat(*, stage: str, cycle_index: int, open_trades: int = 0, detail: str = "") -> None:
    try:
        payload = {
            "run_tag": RUN_TAG,
            "pid": int(os.getpid()),
            "ts": float(time.time()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": str(stage or "").strip() or "loop",
            "cycle_index": int(cycle_index),
            "open_trades": int(max(0, int(open_trades))),
            "detail": str(detail or "").strip(),
        }
        os.makedirs(os.path.dirname(HEARTBEAT_FILE), exist_ok=True)
        tmp = HEARTBEAT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, HEARTBEAT_FILE)
    except Exception:
        # Heartbeat must never break runtime loop.
        pass


def _pid_is_alive(pid: int) -> bool:
    pid = int(pid or 0)
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            probe = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                creationflags=creationflags,
            )
            line = str(probe.stdout or "").strip().lower()
            if not line or "no tasks are running" in line:
                return False
            return (str(pid) in line) and ("python.exe" in line)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _runtime_tuner_lock_file(run_tag: str) -> str:
    safe_tag = str(run_tag or "").strip()
    if not safe_tag:
        return ""
    return os.path.join(PROJECT_ROOT, "logs", "matrix", safe_tag, "runtime_tuner.lock.json")


def _runtime_tuner_patch_file(run_tag: str) -> str:
    safe_tag = str(run_tag or "").strip()
    if not safe_tag:
        return ""
    return os.path.join(PROJECT_ROOT, "logs", "matrix", safe_tag, "runtime_tuner_runtime_overrides.json")


def _runtime_tuner_applied_keys() -> list[str]:
    raw = _RUNTIME_TUNER_PATCH_STATE.get("applied_keys", [])
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        key = str(item or "").strip().upper()
        if key:
            out.append(key)
    return out


def _runtime_tuner_reload_targets(applied_keys: list[str]) -> set[str]:
    targets: set[str] = set()
    for key in applied_keys:
        normalized = str(key or "").strip().upper()
        if not normalized:
            continue
        exact = _RUNTIME_TUNER_RELOAD_EXACT_MAP.get(normalized)
        if exact:
            targets.add(exact)
        for prefix, target in _RUNTIME_TUNER_RELOAD_PREFIX_MAP.items():
            if normalized.startswith(prefix):
                targets.add(target)
                break
    return targets


def _enforce_source_qos_dual_entry_guard() -> None:
    if (
        bool(getattr(config, "V2_SOURCE_QOS_FORCE_DUAL_ENTRY", True))
        and not bool(getattr(config, "V2_ENTRY_DUAL_CHANNEL_ENABLED", False))
    ):
        setattr(config, "V2_ENTRY_DUAL_CHANNEL_ENABLED", True)
        if float(getattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE", 0.0) or 0.0) <= 0.0:
            setattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE", 0.18)
        logger.warning(
            "V2_SOURCE_QOS forced dual-entry lanes enabled=%s explore_share=%.2f",
            bool(getattr(config, "V2_ENTRY_DUAL_CHANNEL_ENABLED", False)),
            float(getattr(config, "V2_ENTRY_EXPLORE_MAX_SHARE", 0.0) or 0.0),
        )


def _coerce_runtime_override_value(*, key: str, raw: Any) -> Any:
    text = str(raw).strip() if raw is not None else ""
    current = getattr(config, key, None)

    if isinstance(current, bool):
        low = text.lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
        return bool(current)
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(round(float(text)))
        except Exception:
            return int(current)
    if isinstance(current, float):
        try:
            return float(text)
        except Exception:
            return float(current)
    if isinstance(current, (list, tuple, set)):
        if isinstance(raw, (list, tuple, set)):
            parts = [str(x).strip() for x in raw if str(x).strip()]
        else:
            parts = [p.strip() for p in text.split(",") if p.strip()]
        if isinstance(current, tuple):
            return tuple(parts)
        if isinstance(current, set):
            return set(parts)
        return parts
    if isinstance(current, str):
        return text

    # Fallback for unknown attribute types.
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        return text


def _runtime_tuner_apply_runtime_overrides(*, run_tag: str) -> tuple[bool, int, str]:
    if not bool(getattr(config, "RUNTIME_TUNER_HOT_APPLY_ENABLED", True)):
        _RUNTIME_TUNER_PATCH_STATE["applied_keys"] = []
        return False, 0, "disabled"
    patch_path = _runtime_tuner_patch_file(run_tag)
    if not patch_path or not os.path.exists(patch_path):
        _RUNTIME_TUNER_PATCH_STATE["applied_keys"] = []
        return False, 0, "missing"

    try:
        mtime = float(os.path.getmtime(patch_path))
    except Exception as exc:
        _RUNTIME_TUNER_PATCH_STATE["applied_keys"] = []
        return False, 0, f"error:stat_failed:{exc}"

    checked_mtime = float(_RUNTIME_TUNER_PATCH_STATE.get("checked_mtime", 0.0) or 0.0)
    if mtime <= checked_mtime:
        _RUNTIME_TUNER_PATCH_STATE["applied_keys"] = []
        return False, 0, "unchanged"

    try:
        with open(patch_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        _RUNTIME_TUNER_PATCH_STATE["applied_keys"] = []
        return False, 0, f"error:read_failed:{exc}"

    overrides_raw = payload.get("overrides", {}) if isinstance(payload, dict) else {}
    if not isinstance(overrides_raw, dict):
        _RUNTIME_TUNER_PATCH_STATE["applied_keys"] = []
        return False, 0, "error:invalid_payload"

    normalized = {str(k): str(v) for k, v in overrides_raw.items()}
    payload_hash = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    if payload_hash == str(_RUNTIME_TUNER_PATCH_STATE.get("applied_hash", "") or ""):
        _RUNTIME_TUNER_PATCH_STATE["checked_mtime"] = mtime
        _RUNTIME_TUNER_PATCH_STATE["applied_keys"] = []
        return False, 0, "unchanged_hash"

    applied_keys: list[str] = []
    skipped_unknown = 0
    for key in sorted(normalized.keys()):
        if not key or (not hasattr(config, key)):
            skipped_unknown += 1
            continue
        current = getattr(config, key)
        value = _coerce_runtime_override_value(key=key, raw=normalized[key])
        if current == value:
            continue
        setattr(config, key, value)
        applied_keys.append(key)

    _RUNTIME_TUNER_PATCH_STATE["checked_mtime"] = mtime
    _RUNTIME_TUNER_PATCH_STATE["applied_hash"] = payload_hash
    _RUNTIME_TUNER_PATCH_STATE["last_error"] = ""
    _RUNTIME_TUNER_PATCH_STATE["applied_keys"] = sorted(applied_keys)
    if applied_keys:
        detail = ",".join(applied_keys[:8])
        if len(applied_keys) > 8:
            detail += ",..."
        if skipped_unknown > 0:
            detail += f" skipped_unknown={skipped_unknown}"
        return True, len(applied_keys), detail
    if skipped_unknown > 0:
        return False, 0, f"no_changes skipped_unknown={skipped_unknown}"
    return False, 0, "no_changes"


def _runtime_tuner_control_active(*, run_tag: str) -> bool:
    if not bool(getattr(config, "RUNTIME_TUNER_LOCK_LOCAL_CONTROLLERS", True)):
        return False

    interval = max(1.0, float(getattr(config, "RUNTIME_TUNER_LOCK_CHECK_INTERVAL_SECONDS", 3) or 3))
    now_ts = float(time.time())
    last_checked = float(_RUNTIME_TUNER_LOCK_STATE.get("checked_at", 0.0) or 0.0)
    if (now_ts - last_checked) < interval:
        return bool(_RUNTIME_TUNER_LOCK_STATE.get("active", False))

    active = False
    pid = 0
    lock_path = _runtime_tuner_lock_file(run_tag)
    if lock_path and os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            pid = int(payload.get("pid", 0) or 0)
            active = _pid_is_alive(pid)
            if (not active) and bool(getattr(config, "RUNTIME_TUNER_PRUNE_STALE_LOCK", True)):
                stale_after = max(
                    10.0,
                    float(getattr(config, "RUNTIME_TUNER_STALE_LOCK_SECONDS", 120) or 120),
                )
                lock_mtime = float(os.path.getmtime(lock_path))
                lock_age = max(0.0, now_ts - lock_mtime)
                if lock_age >= stale_after:
                    try:
                        os.remove(lock_path)
                        logger.warning(
                            "RUNTIME_TUNER_CONTROL pruned stale lock run_tag=%s pid=%s age=%.1fs path=%s",
                            run_tag,
                            int(pid or 0),
                            float(lock_age),
                            lock_path,
                        )
                    except Exception:
                        pass
        except Exception:
            active = False
            pid = 0

    _RUNTIME_TUNER_LOCK_STATE["checked_at"] = now_ts
    _RUNTIME_TUNER_LOCK_STATE["active"] = bool(active)
    _RUNTIME_TUNER_LOCK_STATE["pid"] = int(pid or 0)
    return bool(active)


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
    return runtime_policy.policy_state(source_stats=source_stats, safety_stats=safety_stats)


def _apply_policy_hysteresis(
    *,
    raw_mode: str,
    raw_reason: str,
    current_mode: str,
    bad_streak: int,
    good_streak: int,
) -> tuple[str, str, int, int]:
    return runtime_policy.apply_policy_hysteresis(
        raw_mode=raw_mode,
        raw_reason=raw_reason,
        current_mode=current_mode,
        bad_streak=bad_streak,
        good_streak=good_streak,
    )


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


def _source_name_key(raw: Any) -> str:
    return str(raw or "unknown").strip().lower() or "unknown"


def _count_sources(tokens: list[dict[str, Any]] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for token in (tokens or []):
        if not isinstance(token, dict):
            continue
        key = _source_name_key(token.get("source", "unknown"))
        out[key] = int(out.get(key, 0)) + 1
    return out


def _count_candidate_sources(candidates: list[tuple[dict[str, Any], dict[str, Any]]] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for token_data, _ in (candidates or []):
        key = _source_name_key((token_data or {}).get("source", "unknown"))
        out[key] = int(out.get(key, 0)) + 1
    return out


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


def _format_source_qos_brief(meta: dict[str, Any], snapshot: dict[str, Any]) -> str:
    in_total = int((meta or {}).get("in_total", 0) or 0)
    out_total = int((meta or {}).get("out_total", in_total) or in_total)
    drop_counts_raw = dict((meta or {}).get("drop_counts", {}) or {})
    drop_counts: dict[str, int] = {}
    for key, value in drop_counts_raw.items():
        try:
            cnt = int(value)
        except Exception:
            continue
        if cnt > 0:
            drop_counts[str(key)] = cnt
    active_cooldowns = len(dict((snapshot or {}).get("active_cooldowns", {}) or {}))
    dropped_total = max(0, in_total - out_total)
    return (
        f"enabled={1 if bool((meta or {}).get('enabled', False)) else 0}"
        f"/in={in_total}"
        f"/out={out_total}"
        f"/dropped={dropped_total}"
        f"/active_cd={active_cooldowns}"
        f"/drops={_format_top_counts(drop_counts, limit=3)}"
    )


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
    return runtime_policy.detect_market_regime(
        policy_state_now=policy_state,
        source_stats=source_stats,
        safety_stats=safety_stats,
        avg_candidates_recent=avg_candidates_recent,
    )


def _market_mode_entry_profile(market_mode: str) -> dict[str, float | int | bool]:
    return runtime_policy.market_mode_entry_profile(market_mode=market_mode)


def _apply_market_mode_hysteresis(
    *,
    raw_mode: str,
    raw_reason: str,
    current_mode: str,
    risk_streak: int,
    recover_streak: int,
) -> tuple[str, str, int, int]:
    return runtime_policy.apply_market_mode_hysteresis(
        raw_mode=raw_mode,
        raw_reason=raw_reason,
        current_mode=current_mode,
        risk_streak=risk_streak,
        recover_streak=recover_streak,
    )


def _excluded_trade_addresses() -> set[str]:
    def _iter_values(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            return [x.strip() for x in raw.split(",") if x.strip()]
        if isinstance(raw, (list, tuple, set)):
            return [str(x).strip() for x in raw if str(x).strip()]
        text = str(raw).strip()
        return [text] if text else []

    out: set[str] = set()
    weth = normalize_address(getattr(config, "WETH_ADDRESS", ""))
    if weth:
        out.add(weth)
    for addr in _iter_values(getattr(config, "AUTO_TRADE_EXCLUDED_ADDRESSES", [])):
        n = normalize_address(str(addr))
        if n:
            out.add(n)
    return out


def _is_placeholder_trade_address(address: str) -> bool:
    addr = normalize_address(address)
    if not addr:
        return True
    return addr == "0x0000000000000000000000000000000000000000"


def _excluded_trade_symbols() -> set[str]:
    raw = getattr(config, "AUTO_TRADE_EXCLUDED_SYMBOLS", []) or []
    if isinstance(raw, str):
        items = [x.strip() for x in raw.split(",") if x.strip()]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(x).strip() for x in raw if str(x).strip()]
    else:
        text = str(raw).strip()
        items = [text] if text else []
    return {
        str(x).strip().upper()
        for x in items
        if str(x).strip()
    }


def _excluded_trade_symbol_keywords() -> list[str]:
    raw = getattr(config, "AUTO_TRADE_EXCLUDED_SYMBOL_KEYWORDS", []) or []
    if isinstance(raw, str):
        items = [x.strip() for x in raw.split(",") if x.strip()]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(x).strip() for x in raw if str(x).strip()]
    else:
        text = str(raw).strip()
        items = [text] if text else []
    return [
        str(x).strip().upper()
        for x in items
        if str(x).strip()
    ]


def _excluded_symbol_reason(
    symbol: str,
    *,
    excluded_symbols: set[str] | None = None,
    excluded_keywords: list[str] | None = None,
) -> str:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return ""
    symbols = excluded_symbols if excluded_symbols is not None else _excluded_trade_symbols()
    if sym in symbols:
        return "excluded_symbol"
    keywords = excluded_keywords if excluded_keywords is not None else _excluded_trade_symbol_keywords()
    for keyword in keywords:
        if keyword and keyword in sym:
            return "excluded_symbol_keyword"
    return ""


def _merge_token_streams(*groups: list[dict]) -> list[dict]:
    def _is_watchlist_source(token: dict) -> bool:
        src = str((token or {}).get("source", "") or "").strip().lower()
        return src.startswith("watchlist")

    def _merge_enrichment(primary: dict, secondary: dict) -> dict:
        # Keep the primary token identity/source, but backfill missing market fields.
        out = dict(primary or {})
        for key, value in (secondary or {}).items():
            if key == "source":
                continue
            current = out.get(key)
            if current in (None, "", 0, 0.0):
                out[key] = value
                continue
            if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                if float(value) > float(current):
                    out[key] = value
        return out

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
            existing_watch = _is_watchlist_source(existing)
            token_watch = _is_watchlist_source(token)
            if (not existing_watch) and token_watch:
                merged[address] = _merge_enrichment(existing, token)
                continue
            if existing_watch and (not token_watch):
                merged[address] = _merge_enrichment(token, existing)
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
        if self.mode == "observe":
            # Backward-compatible alias used by matrix presets.
            self.mode = "dry_run"
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
        self.market_mode_owns_strictness = bool(getattr(config, "MARKET_MODE_OWNS_STRICTNESS", True))
        self.orchestrator_lock_adaptive_filters = bool(
            getattr(config, "STRATEGY_ORCHESTRATOR_LOCK_ADAPTIVE_FILTERS", True)
        )
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
        fallback = max(0.0, float(fallback_abs))
        if pct <= 0.0:
            return fallback
        step = max(0.0, float(value)) * (pct / 100.0)
        return step if step > 0.0 else fallback

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
        if (
            self.orchestrator_lock_adaptive_filters
            and bool(getattr(config, "STRATEGY_ORCHESTRATOR_ENABLED", False))
            and str(getattr(config, "STRATEGY_ORCHESTRATOR_MODE", "off") or "off").strip().lower() in {"dry_run", "apply"}
        ):
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
        skip_ev_net_low = int(self.window_skip_reasons.get("ev_net_low", 0))
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
        edge_negative_blocked = (skip_negative_edge + skip_edge_low + skip_ev_net_low) > 0 and edge_mode in {
            "percent",
            "both",
        }
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
                f"neg_edge={skip_negative_edge + skip_edge_low} ev_net={skip_ev_net_low}"
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

        if self.market_mode_owns_strictness:
            strictness_changed = (
                (score_next != score_now)
                or (abs(volume_next - volume_now) >= 0.5)
                or (ttl_next != ttl_now)
                or (abs(edge_next - edge_now) >= 0.01)
            )
            score_next = score_now
            volume_next = volume_now
            ttl_next = ttl_now
            edge_next = edge_now
            if strictness_changed and action != "hold":
                action = "hold_market_mode_owned"
                reason = "strictness_locked_to_market_mode"

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
            "ADAPTIVE_FILTERS mode=%s action=%s changed=%s reason=%s regime=%s avg_cand=%.2f avg_opened=%.2f closed_delta=%s realized_delta=$%.2f pnl_signal=%s pressure=%s score=%s->%s volume=%.0f->%.0f dedup_ttl=%s->%s dyn_dedup(p=%.1f,target=%s,range=%s-%s,samples=%s) edge=%.2f->%.2f lock_floor=%.2f->%.2f no_mom_pnl=%.2f->%.2f weakness_pnl=%.2f->%.2f fails(score=%s,volume=%s,dedup=%s) skips(neg_edge=%s,ev_net=%s,edge_usd=%s,min_trade=%s) zero_open=%s/%s",
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
            skip_ev_net_low,
            skip_edge_usd_low,
            skip_min_trade,
            self.zero_open_streak,
            self.zero_open_windows_before_reset,
        )

        self._reset_window(now_ts=now_ts, realized_usd=realized_now, closed=closed_now)


class AutonomousControlController:
    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "AUTONOMOUS_CONTROL_ENABLED", False))
        self.mode = str(getattr(config, "AUTONOMOUS_CONTROL_MODE", "dry_run") or "dry_run").strip().lower()
        if self.mode == "observe":
            self.mode = "dry_run"
        self.paper_only = bool(getattr(config, "AUTONOMOUS_CONTROL_PAPER_ONLY", True))
        self.interval_seconds = int(getattr(config, "AUTONOMOUS_CONTROL_INTERVAL_SECONDS", 300) or 300)
        self.min_window_cycles = int(getattr(config, "AUTONOMOUS_CONTROL_MIN_WINDOW_CYCLES", 3) or 3)
        self.cooldown_windows = int(getattr(config, "AUTONOMOUS_CONTROL_COOLDOWN_WINDOWS", 1) or 1)

        self.target_candidates_min = float(getattr(config, "AUTONOMOUS_CONTROL_TARGET_CANDIDATES_MIN", 2.0) or 2.0)
        self.target_candidates_high = float(getattr(config, "AUTONOMOUS_CONTROL_TARGET_CANDIDATES_HIGH", 8.0) or 8.0)
        self.target_opened_min = float(getattr(config, "AUTONOMOUS_CONTROL_TARGET_OPENED_MIN", 0.18) or 0.18)

        self.neg_realized_trigger = float(getattr(config, "AUTONOMOUS_CONTROL_NEG_REALIZED_TRIGGER_USD", 0.08) or 0.08)
        self.pos_realized_trigger = float(getattr(config, "AUTONOMOUS_CONTROL_POS_REALIZED_TRIGGER_USD", 0.08) or 0.08)
        self.max_loss_streak_trigger = int(getattr(config, "AUTONOMOUS_CONTROL_MAX_LOSS_STREAK_TRIGGER", 3) or 3)

        self.step_open_trades = int(getattr(config, "AUTONOMOUS_CONTROL_STEP_OPEN_TRADES", 1) or 1)
        self.step_top_n = int(getattr(config, "AUTONOMOUS_CONTROL_STEP_TOP_N", 1) or 1)
        self.step_max_buys_per_hour = int(getattr(config, "AUTONOMOUS_CONTROL_STEP_MAX_BUYS_PER_HOUR", 6) or 6)
        self.step_trade_size_max = float(getattr(config, "AUTONOMOUS_CONTROL_STEP_TRADE_SIZE_MAX_USD", 0.05) or 0.05)

        self.max_open_min_bound = int(getattr(config, "AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MIN", 1) or 1)
        self.max_open_max_bound = int(getattr(config, "AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MAX", 6) or 6)
        self.top_n_min_bound = int(getattr(config, "AUTONOMOUS_CONTROL_TOP_N_MIN", 1) or 1)
        self.top_n_max_bound = int(getattr(config, "AUTONOMOUS_CONTROL_TOP_N_MAX", 20) or 20)
        self.max_buys_min_bound = int(getattr(config, "AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN", 6) or 6)
        self.max_buys_max_bound = int(getattr(config, "AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MAX", 96) or 96)
        self.size_max_min_bound = float(getattr(config, "AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MIN", 0.25) or 0.25)
        self.size_max_max_bound = float(getattr(config, "AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MAX", 2.0) or 2.0)

        self.risk_off_open_cap = int(getattr(config, "AUTONOMOUS_CONTROL_RISK_OFF_OPEN_TRADES_CAP", 2) or 2)
        self.risk_off_top_n_cap = int(getattr(config, "AUTONOMOUS_CONTROL_RISK_OFF_TOP_N_CAP", 8) or 8)
        self.risk_off_buys_cap = int(getattr(config, "AUTONOMOUS_CONTROL_RISK_OFF_MAX_BUYS_PER_HOUR_CAP", 24) or 24)
        self.risk_off_size_cap = float(getattr(config, "AUTONOMOUS_CONTROL_RISK_OFF_TRADE_SIZE_MAX_CAP", 0.70) or 0.70)
        self.anti_stall_enabled = bool(getattr(config, "AUTONOMOUS_CONTROL_ANTI_STALL_ENABLED", True))
        self.anti_stall_min_candidates = float(getattr(config, "AUTONOMOUS_CONTROL_ANTI_STALL_MIN_CANDIDATES", 1.0) or 1.0)
        self.anti_stall_min_utilization = float(getattr(config, "AUTONOMOUS_CONTROL_ANTI_STALL_MIN_UTILIZATION", 0.85) or 0.85)
        self.anti_stall_limit_skip_min = int(getattr(config, "AUTONOMOUS_CONTROL_ANTI_STALL_LIMIT_SKIP_MIN", 1) or 1)
        self.anti_stall_expand_mult = float(getattr(config, "AUTONOMOUS_CONTROL_ANTI_STALL_EXPAND_MULT", 1.0) or 1.0)
        self.recovery_enabled = bool(getattr(config, "AUTONOMOUS_CONTROL_RECOVERY_ENABLED", True))
        self.recovery_min_candidates = float(getattr(config, "AUTONOMOUS_CONTROL_RECOVERY_MIN_CANDIDATES", 0.8) or 0.8)
        self.recovery_expand_mult = float(getattr(config, "AUTONOMOUS_CONTROL_RECOVERY_EXPAND_MULT", 1.2) or 1.2)
        self.fragile_tighten_enabled = bool(getattr(config, "AUTONOMOUS_CONTROL_FRAGILE_TIGHTEN_ENABLED", True))
        self.market_mode_owns_strictness = bool(getattr(config, "MARKET_MODE_OWNS_STRICTNESS", True))
        self.orchestrator_lock_controls = bool(
            getattr(config, "STRATEGY_ORCHESTRATOR_LOCK_AUTONOMY_CONTROLS", True)
        )

        self.decisions_log_enabled = bool(getattr(config, "AUTONOMOUS_CONTROL_DECISIONS_LOG_ENABLED", True))
        raw_log = str(
            getattr(
                config,
                "AUTONOMOUS_CONTROL_DECISIONS_LOG_FILE",
                os.path.join("logs", "autonomy_decisions.jsonl"),
            )
            or ""
        ).strip()
        if not raw_log:
            raw_log = os.path.join("logs", "autonomy_decisions.jsonl")
        self.decisions_log_file = raw_log if os.path.isabs(raw_log) else os.path.abspath(os.path.join(PROJECT_ROOT, raw_log))

        self.last_eval_ts = time.time()
        self.cooldown_until_ts = 0.0
        self.window_cycles = 0
        self.window_candidates = 0
        self.window_opened = 0
        self.window_skip_reasons: dict[str, int] = {}
        self.window_regime_counts: dict[str, int] = {}
        self.window_policy_counts: dict[str, int] = {}
        self.prev_closed = 0
        self.prev_wins = 0
        self.prev_losses = 0
        self.prev_realized_usd = 0.0

        self.max_open_min_bound = max(1, int(self.max_open_min_bound))
        self.max_open_max_bound = max(self.max_open_min_bound, int(self.max_open_max_bound))
        self.top_n_min_bound = max(1, int(self.top_n_min_bound))
        self.top_n_max_bound = max(self.top_n_min_bound, int(self.top_n_max_bound))
        self.max_buys_min_bound = max(1, int(self.max_buys_min_bound))
        self.max_buys_max_bound = max(self.max_buys_min_bound, int(self.max_buys_max_bound))
        self.size_max_min_bound = max(0.05, float(self.size_max_min_bound))
        self.size_max_max_bound = max(self.size_max_min_bound, float(self.size_max_max_bound))
        self.step_open_trades = max(1, int(self.step_open_trades))
        self.step_top_n = max(1, int(self.step_top_n))
        self.step_max_buys_per_hour = max(1, int(self.step_max_buys_per_hour))
        self.step_trade_size_max = max(0.01, float(self.step_trade_size_max))
        self.max_loss_streak_trigger = max(1, int(self.max_loss_streak_trigger))
        self.anti_stall_min_candidates = max(0.0, float(self.anti_stall_min_candidates))
        self.anti_stall_min_utilization = max(0.5, min(1.0, float(self.anti_stall_min_utilization)))
        self.anti_stall_limit_skip_min = max(0, int(self.anti_stall_limit_skip_min))
        self.anti_stall_expand_mult = max(0.5, min(3.0, float(self.anti_stall_expand_mult)))
        self.recovery_min_candidates = max(0.0, float(self.recovery_min_candidates))
        self.recovery_expand_mult = max(0.5, min(3.0, float(self.recovery_expand_mult)))

        base_open, base_top_n, base_buys, base_size = self._snapshot_controls()
        self.base_max_open = self._clamp_int(base_open, self.max_open_min_bound, self.max_open_max_bound)
        self.base_top_n = self._clamp_int(base_top_n, self.top_n_min_bound, self.top_n_max_bound)
        self.base_buys_per_hour = self._clamp_int(base_buys, self.max_buys_min_bound, self.max_buys_max_bound)
        self.base_trade_size_max = self._clamp_float(base_size, self.size_max_min_bound, self.size_max_max_bound)

    @staticmethod
    def _clamp_int(value: int, lower: int, upper: int) -> int:
        return max(lower, min(upper, int(value)))

    @staticmethod
    def _clamp_float(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

    def _snapshot_controls(self) -> tuple[int, int, int, float]:
        max_open = int(getattr(config, "MAX_OPEN_TRADES", 1) or 1)
        if max_open <= 0:
            max_open = int(self.max_open_max_bound)
        top_n = int(getattr(config, "AUTO_TRADE_TOP_N", 1) or 1)
        max_buys = int(getattr(config, "MAX_BUYS_PER_HOUR", 1) or 1)
        if max_buys <= 0:
            max_buys = int(self.max_buys_max_bound)
        size_max = float(getattr(config, "PAPER_TRADE_SIZE_MAX_USD", 1.0) or 1.0)
        return max(1, max_open), max(1, top_n), max(1, max_buys), max(0.05, size_max)

    def _apply_controls(self, *, max_open: int, top_n: int, max_buys_per_hour: int, trade_size_max_usd: float) -> None:
        setattr(config, "MAX_OPEN_TRADES", int(max_open))
        setattr(config, "AUTO_TRADE_TOP_N", int(top_n))
        setattr(config, "MAX_BUYS_PER_HOUR", int(max_buys_per_hour))
        setattr(config, "MAX_TRADES_PER_HOUR", int(max_buys_per_hour))
        setattr(config, "PAPER_TRADE_SIZE_MAX_USD", float(trade_size_max_usd))

    def _write_decision(self, event: dict[str, object]) -> None:
        if not self.decisions_log_enabled:
            return
        try:
            payload = dict(event or {})
            payload.setdefault("run_tag", RUN_TAG)
            os.makedirs(os.path.dirname(self.decisions_log_file) or ".", exist_ok=True)
            with open(self.decisions_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=False) + "\n")
        except Exception:
            logger.exception("AUTONOMY decision log write failed")

    def prime(self, *, auto_stats: dict[str, float | int]) -> None:
        self.prev_closed = int(auto_stats.get("closed", 0) or 0)
        self.prev_wins = int(auto_stats.get("wins", 0) or 0)
        self.prev_losses = int(auto_stats.get("losses", 0) or 0)
        self.prev_realized_usd = float(auto_stats.get("realized_pnl_usd", 0.0) or 0.0)

    def record_cycle(
        self,
        *,
        candidates: int,
        opened: int,
        policy_state: str,
        market_regime: str,
        skip_reasons_cycle: dict[str, int] | None = None,
    ) -> None:
        self.window_cycles += 1
        self.window_candidates += int(candidates)
        self.window_opened += int(opened)
        for reason, count in (skip_reasons_cycle or {}).items():
            key = str(reason or "unknown").strip().lower() or "unknown"
            self.window_skip_reasons[key] = int(self.window_skip_reasons.get(key, 0)) + int(count or 0)
        pkey = str(policy_state or "UNKNOWN").strip().upper() or "UNKNOWN"
        rkey = str(market_regime or "UNKNOWN").strip().upper() or "UNKNOWN"
        self.window_policy_counts[pkey] = int(self.window_policy_counts.get(pkey, 0)) + 1
        self.window_regime_counts[rkey] = int(self.window_regime_counts.get(rkey, 0)) + 1

    def _reset_window(self, *, now_ts: float, auto_stats: dict[str, float | int]) -> None:
        self.last_eval_ts = float(now_ts)
        self.window_cycles = 0
        self.window_candidates = 0
        self.window_opened = 0
        self.window_skip_reasons = {}
        self.window_policy_counts = {}
        self.window_regime_counts = {}
        self.prev_closed = int(auto_stats.get("closed", 0) or 0)
        self.prev_wins = int(auto_stats.get("wins", 0) or 0)
        self.prev_losses = int(auto_stats.get("losses", 0) or 0)
        self.prev_realized_usd = float(auto_stats.get("realized_pnl_usd", 0.0) or 0.0)

    def maybe_adapt(self, *, policy_state: str, market_regime: str, auto_stats: dict[str, float | int]) -> None:
        if not self.enabled:
            return
        if self.mode not in {"dry_run", "apply"}:
            return
        if (
            self.orchestrator_lock_controls
            and bool(getattr(config, "STRATEGY_ORCHESTRATOR_ENABLED", False))
            and str(getattr(config, "STRATEGY_ORCHESTRATOR_MODE", "off") or "off").strip().lower() in {"dry_run", "apply"}
        ):
            return
        if self.paper_only and not bool(getattr(config, "AUTO_TRADE_PAPER", False)):
            return

        now_ts = time.time()
        if (now_ts - float(self.last_eval_ts)) < float(self.interval_seconds):
            return
        if int(self.window_cycles) < int(self.min_window_cycles):
            return

        avg_candidates = float(self.window_candidates) / max(1, int(self.window_cycles))
        avg_opened = float(self.window_opened) / max(1, int(self.window_cycles))
        closed_now = int(auto_stats.get("closed", 0) or 0)
        wins_now = int(auto_stats.get("wins", 0) or 0)
        losses_now = int(auto_stats.get("losses", 0) or 0)
        realized_now = float(auto_stats.get("realized_pnl_usd", 0.0) or 0.0)
        loss_streak = int(auto_stats.get("loss_streak", 0) or 0)
        open_now = int(auto_stats.get("open_trades", 0) or 0)
        risk_block_reason = str(auto_stats.get("risk_block_reason", "") or "").strip()
        trades_last_hour = int(auto_stats.get("trades_last_hour", 0) or 0)
        skip_limits = int(self.window_skip_reasons.get("disabled_or_limits", 0))
        skip_cooldown = int(self.window_skip_reasons.get("cooldown", 0))

        closed_delta = int(closed_now - int(self.prev_closed))
        wins_delta = int(wins_now - int(self.prev_wins))
        losses_delta = int(losses_now - int(self.prev_losses))
        realized_delta = float(realized_now - float(self.prev_realized_usd))
        policy_now = str(policy_state or "UNKNOWN").strip().upper() or "UNKNOWN"
        regime_now = str(market_regime or "UNKNOWN").strip().upper() or "UNKNOWN"

        max_open_now, top_n_now, buys_now, size_max_now = self._snapshot_controls()
        hourly_utilization = float(trades_last_hour) / float(max(1, int(buys_now)))
        max_open_next, top_n_next, buys_next, size_max_next = max_open_now, top_n_now, buys_now, size_max_now
        below_baseline = (
            max_open_now < int(self.base_max_open)
            or top_n_now < int(self.base_top_n)
            or buys_now < int(self.base_buys_per_hour)
            or (size_max_now + 1e-9) < float(self.base_trade_size_max)
        )
        action = "hold"
        reason = "steady_state"

        if self.market_mode_owns_strictness:
            action = "hold_market_mode_owned"
            reason = "controls_locked_to_market_mode"
        elif now_ts < float(self.cooldown_until_ts):
            remaining = max(0.0, float(self.cooldown_until_ts) - now_ts)
            action = "hold"
            reason = f"cooldown_active {remaining:.1f}s"
        elif (
            policy_now != "OK"
            or regime_now in {"RED", "YELLOW", "RISK_OFF", "CAUTION"}
            or bool(risk_block_reason)
        ):
            max_open_next = min(max_open_next, int(self.risk_off_open_cap))
            top_n_next = min(top_n_next, int(self.risk_off_top_n_cap))
            buys_next = min(buys_next, int(self.risk_off_buys_cap))
            size_max_next = min(size_max_next, float(self.risk_off_size_cap))
            action = "risk_tighten"
            reason = (
                f"policy={policy_now} regime={regime_now}"
                + (f" risk_block={risk_block_reason}" if risk_block_reason else "")
            )
        elif self.fragile_tighten_enabled and regime_now in {"RED", "FRAGILE"}:
            max_open_next -= int(self.step_open_trades)
            top_n_next -= int(self.step_top_n)
            buys_next -= int(self.step_max_buys_per_hour)
            size_max_next -= float(self.step_trade_size_max) * 0.5
            action = "fragile_tighten"
            reason = f"regime={regime_now} mild_step_down"
        elif (
            closed_delta > 0
            and (
                realized_delta <= -abs(float(self.neg_realized_trigger))
                or losses_delta > wins_delta
                or loss_streak >= int(self.max_loss_streak_trigger)
            )
        ):
            max_open_next -= int(self.step_open_trades)
            top_n_next -= int(self.step_top_n)
            buys_next -= int(self.step_max_buys_per_hour)
            size_max_next -= float(self.step_trade_size_max)
            action = "loss_tighten"
            reason = (
                f"realized_delta=${realized_delta:.3f} losses_delta={losses_delta} "
                f"wins_delta={wins_delta} loss_streak={loss_streak}"
            )
        elif (
            avg_candidates >= float(self.target_candidates_high)
            and avg_opened < float(self.target_opened_min)
            and realized_delta >= (-abs(float(self.neg_realized_trigger)) * 0.5)
            and loss_streak <= 1
        ):
            max_open_next += int(self.step_open_trades)
            top_n_next += int(self.step_top_n)
            buys_next += int(self.step_max_buys_per_hour)
            if realized_delta >= float(self.pos_realized_trigger):
                size_max_next += float(self.step_trade_size_max)
            action = "flow_expand"
            reason = f"high_flow_no_fill avg_cand={avg_candidates:.2f} avg_opened={avg_opened:.2f}"
        elif (
            closed_delta > 0
            and realized_delta >= abs(float(self.pos_realized_trigger))
            and losses_delta <= wins_delta
            and avg_candidates >= float(self.target_candidates_min)
        ):
            top_n_next += int(self.step_top_n)
            buys_next += int(self.step_max_buys_per_hour)
            action = "profit_expand"
            reason = (
                f"realized_delta=${realized_delta:.3f} wins_delta={wins_delta} "
                f"losses_delta={losses_delta}"
            )
        elif (
            self.anti_stall_enabled
            and policy_now == "OK"
            and regime_now not in {"RED", "RISK_OFF", "CAUTION"}
            and avg_candidates >= max(float(self.anti_stall_min_candidates), float(self.target_candidates_min) * 0.60)
            and avg_opened <= max(0.02, float(self.target_opened_min) * 0.35)
            and (
                hourly_utilization >= float(self.anti_stall_min_utilization)
                or skip_limits >= int(self.anti_stall_limit_skip_min)
                or skip_cooldown >= int(self.anti_stall_limit_skip_min)
            )
        ):
            buy_step = max(1, int(round(float(self.step_max_buys_per_hour) * float(self.anti_stall_expand_mult))))
            top_step = max(1, int(round(float(self.step_top_n) * max(1.0, float(self.anti_stall_expand_mult)))))
            buys_next += buy_step
            top_n_next += top_step
            if open_now >= max_open_now and max_open_now < self.max_open_max_bound:
                max_open_next += int(self.step_open_trades)
            action = "anti_stall_expand"
            reason = (
                f"open_stall util={hourly_utilization:.2f} limits={skip_limits} cooldown={skip_cooldown} "
                f"avg_cand={avg_candidates:.2f} avg_opened={avg_opened:.2f}"
            )
        elif (
            self.recovery_enabled
            and policy_now == "OK"
            and regime_now not in {"RED", "RISK_OFF", "CAUTION"}
            and below_baseline
            and avg_candidates >= max(float(self.recovery_min_candidates), float(self.target_candidates_min) * 0.50)
            and avg_opened <= max(0.08, float(self.target_opened_min))
            and losses_delta <= (wins_delta + 1)
        ):
            buy_step = max(1, int(round(float(self.step_max_buys_per_hour) * float(self.recovery_expand_mult))))
            top_step = max(1, int(round(float(self.step_top_n) * max(1.0, float(self.recovery_expand_mult)))))
            max_open_next = min(int(self.base_max_open), max_open_now + int(self.step_open_trades))
            top_n_next = min(int(self.base_top_n), top_n_now + top_step)
            buys_next = min(int(self.base_buys_per_hour), buys_now + buy_step)
            if realized_delta >= (-abs(float(self.neg_realized_trigger)) * 0.5):
                size_max_next = min(float(self.base_trade_size_max), size_max_now + float(self.step_trade_size_max))
            action = "recovery_expand"
            reason = (
                f"recover baseline={self.base_max_open}/{self.base_top_n}/{self.base_buys_per_hour} "
                f"now={max_open_now}/{top_n_now}/{buys_now} avg_cand={avg_candidates:.2f}"
            )
        elif avg_candidates < float(self.target_candidates_min) and avg_opened <= 0.01:
            top_n_next -= int(self.step_top_n)
            buys_next -= int(self.step_max_buys_per_hour)
            action = "thin_market_conserve"
            reason = f"low_flow avg_cand={avg_candidates:.2f} avg_opened={avg_opened:.2f}"

        max_open_next = self._clamp_int(max_open_next, self.max_open_min_bound, self.max_open_max_bound)
        top_n_next = self._clamp_int(top_n_next, self.top_n_min_bound, self.top_n_max_bound)
        buys_next = self._clamp_int(buys_next, self.max_buys_min_bound, self.max_buys_max_bound)
        size_min_now = max(0.01, float(getattr(config, "PAPER_TRADE_SIZE_MIN_USD", 0.0) or 0.0))
        size_lower_bound = max(self.size_max_min_bound, size_min_now)
        size_max_next = self._clamp_float(size_max_next, size_lower_bound, self.size_max_max_bound)

        changed = (
            max_open_next != max_open_now
            or top_n_next != top_n_now
            or buys_next != buys_now
            or abs(size_max_next - size_max_now) >= 0.01
        )
        if self.mode == "apply" and changed and now_ts >= float(self.cooldown_until_ts):
            self._apply_controls(
                max_open=max_open_next,
                top_n=top_n_next,
                max_buys_per_hour=buys_next,
                trade_size_max_usd=size_max_next,
            )
            self.cooldown_until_ts = now_ts + (max(0, self.cooldown_windows) * max(1, self.interval_seconds))

        logger.warning(
            "AUTONOMY mode=%s action=%s changed=%s reason=%s regime_top=%s policy_top=%s avg_cand=%.2f avg_opened=%.2f util=%.2f skip_limits=%s skip_cooldown=%s open_now=%s closed_delta=%s wins_delta=%s losses_delta=%s realized_delta=$%.3f loss_streak=%s controls(open=%s->%s top_n=%s->%s buys_h=%s->%s size_max=%.2f->%.2f)",
            self.mode,
            action,
            changed,
            reason,
            _format_top_counts(self.window_regime_counts, limit=1),
            _format_top_counts(self.window_policy_counts, limit=1),
            avg_candidates,
            avg_opened,
            hourly_utilization,
            skip_limits,
            skip_cooldown,
            open_now,
            closed_delta,
            wins_delta,
            losses_delta,
            realized_delta,
            loss_streak,
            max_open_now,
            max_open_next,
            top_n_now,
            top_n_next,
            buys_now,
            buys_next,
            size_max_now,
            size_max_next,
        )
        self._write_decision(
            {
                "ts": time.time(),
                "mode": self.mode,
                "action": action,
                "changed": bool(changed),
                "reason": reason,
                "market_regime_top": _format_top_counts(self.window_regime_counts, limit=1),
                "policy_top": _format_top_counts(self.window_policy_counts, limit=1),
                "avg_candidates": round(avg_candidates, 4),
                "avg_opened": round(avg_opened, 4),
                "hourly_utilization": round(hourly_utilization, 4),
                "skip_limits": int(skip_limits),
                "skip_cooldown": int(skip_cooldown),
                "below_baseline": bool(below_baseline),
                "closed_delta": int(closed_delta),
                "wins_delta": int(wins_delta),
                "losses_delta": int(losses_delta),
                "realized_delta_usd": round(realized_delta, 6),
                "open_now": int(open_now),
                "loss_streak": int(loss_streak),
                "controls_baseline": {
                    "MAX_OPEN_TRADES": int(self.base_max_open),
                    "AUTO_TRADE_TOP_N": int(self.base_top_n),
                    "MAX_BUYS_PER_HOUR": int(self.base_buys_per_hour),
                    "PAPER_TRADE_SIZE_MAX_USD": round(float(self.base_trade_size_max), 6),
                },
                "controls_before": {
                    "MAX_OPEN_TRADES": int(max_open_now),
                    "AUTO_TRADE_TOP_N": int(top_n_now),
                    "MAX_BUYS_PER_HOUR": int(buys_now),
                    "PAPER_TRADE_SIZE_MAX_USD": round(size_max_now, 6),
                },
                "controls_after": {
                    "MAX_OPEN_TRADES": int(max_open_next),
                    "AUTO_TRADE_TOP_N": int(top_n_next),
                    "MAX_BUYS_PER_HOUR": int(buys_next),
                    "PAPER_TRADE_SIZE_MAX_USD": round(size_max_next, 6),
                },
            }
        )
        self._reset_window(now_ts=now_ts, auto_stats=auto_stats)


class StrategyOrchestrator:
    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "STRATEGY_ORCHESTRATOR_ENABLED", False))
        self.mode = str(getattr(config, "STRATEGY_ORCHESTRATOR_MODE", "dry_run") or "dry_run").strip().lower()
        self.interval_seconds = int(getattr(config, "STRATEGY_ORCHESTRATOR_INTERVAL_SECONDS", 300) or 300)
        self.min_window_cycles = int(getattr(config, "STRATEGY_ORCHESTRATOR_MIN_WINDOW_CYCLES", 2) or 2)
        self.cooldown_windows = int(getattr(config, "STRATEGY_ORCHESTRATOR_COOLDOWN_WINDOWS", 1) or 1)
        self.min_closed_delta = int(getattr(config, "STRATEGY_ORCHESTRATOR_MIN_CLOSED_DELTA", 6) or 6)
        self.defense_enter_streak = int(getattr(config, "STRATEGY_ORCHESTRATOR_DEFENSE_ENTER_STREAK", 2) or 2)
        self.harvest_enter_streak = int(getattr(config, "STRATEGY_ORCHESTRATOR_HARVEST_ENTER_STREAK", 2) or 2)
        self.defense_trigger_avg_pnl = float(
            getattr(config, "STRATEGY_ORCHESTRATOR_DEFENSE_TRIGGER_AVG_PNL_PER_TRADE_USD", -0.002) or -0.002
        )
        self.defense_trigger_loss_share = float(
            getattr(config, "STRATEGY_ORCHESTRATOR_DEFENSE_TRIGGER_LOSS_SHARE", 0.62) or 0.62
        )
        self.harvest_trigger_avg_pnl = float(
            getattr(config, "STRATEGY_ORCHESTRATOR_HARVEST_TRIGGER_AVG_PNL_PER_TRADE_USD", 0.001) or 0.001
        )
        self.harvest_trigger_loss_share = float(
            getattr(config, "STRATEGY_ORCHESTRATOR_HARVEST_TRIGGER_LOSS_SHARE", 0.45) or 0.45
        )
        self.red_force_defense_enabled = bool(
            getattr(config, "STRATEGY_ORCHESTRATOR_RED_FORCE_DEFENSE_ENABLED", True)
        )
        self.red_force_defense_realized_delta_usd = float(
            getattr(config, "STRATEGY_ORCHESTRATOR_RED_FORCE_DEFENSE_REALIZED_DELTA_USD", -0.02) or -0.02
        )
        initial_profile = str(getattr(config, "STRATEGY_ORCHESTRATOR_INITIAL_PROFILE", "harvest") or "harvest").strip().lower()
        self.current_profile = "defense" if initial_profile == "defense" else "harvest"
        self.bad_streak = 0
        self.good_streak = 0
        self.last_eval_ts = time.time()
        self.cooldown_until_ts = 0.0
        self.window_cycles = 0
        self.window_candidates = 0
        self.window_opened = 0
        self.window_regime_counts: dict[str, int] = {}
        self.prev_closed = 0
        self.prev_wins = 0
        self.prev_losses = 0
        self.prev_realized_usd = 0.0
        raw_log = str(
            getattr(config, "AUTONOMOUS_CONTROL_DECISIONS_LOG_FILE", os.path.join("logs", "autonomy_decisions.jsonl")) or ""
        ).strip()
        if not raw_log:
            raw_log = os.path.join("logs", "autonomy_decisions.jsonl")
        base = raw_log if os.path.isabs(raw_log) else os.path.abspath(os.path.join(PROJECT_ROOT, raw_log))
        self.decisions_log_file = os.path.abspath(
            os.path.join(os.path.dirname(base) or PROJECT_ROOT, f"{RUN_TAG}_orchestrator_decisions.jsonl")
        )
        self.profiles = {
            "harvest": self._load_profile("HARVEST"),
            "defense": self._load_profile("DEFENSE"),
        }
        self._apply_profile(self.current_profile)

    def _load_profile(self, prefix: str) -> dict[str, float]:
        p = str(prefix or "").strip().upper()
        return {
            "max_open": float(getattr(config, f"STRATEGY_ORCHESTRATOR_{p}_MAX_OPEN_TRADES", 3)),
            "top_n": float(getattr(config, f"STRATEGY_ORCHESTRATOR_{p}_TOP_N", 10)),
            "max_buys": float(getattr(config, f"STRATEGY_ORCHESTRATOR_{p}_MAX_BUYS_PER_HOUR", 48)),
            "size_max": float(getattr(config, f"STRATEGY_ORCHESTRATOR_{p}_TRADE_SIZE_MAX_USD", 1.0)),
            "hold_max": float(getattr(config, f"STRATEGY_ORCHESTRATOR_{p}_HOLD_MAX_SECONDS", 180)),
            "no_mom_age": float(getattr(config, f"STRATEGY_ORCHESTRATOR_{p}_NO_MOMENTUM_MIN_AGE_PERCENT", 12.0)),
            "no_mom_max_pnl": float(getattr(config, f"STRATEGY_ORCHESTRATOR_{p}_NO_MOMENTUM_MAX_PNL_PERCENT", 0.25)),
            "weak_age": float(getattr(config, f"STRATEGY_ORCHESTRATOR_{p}_WEAKNESS_MIN_AGE_PERCENT", 14.0)),
            "weak_pnl": float(getattr(config, f"STRATEGY_ORCHESTRATOR_{p}_WEAKNESS_PNL_PERCENT", -2.6)),
            "partial_tp_trigger": float(getattr(config, f"STRATEGY_ORCHESTRATOR_{p}_PARTIAL_TP_TRIGGER_PERCENT", 1.2)),
            "partial_tp_fraction": float(getattr(config, f"STRATEGY_ORCHESTRATOR_{p}_PARTIAL_TP_SELL_FRACTION", 0.4)),
        }

    def _apply_profile(self, profile: str) -> None:
        row = self.profiles.get(profile, self.profiles["harvest"])
        setattr(config, "MAX_OPEN_TRADES", max(1, int(round(float(row["max_open"])))))
        setattr(config, "AUTO_TRADE_TOP_N", max(1, int(round(float(row["top_n"])))))
        buys = max(1, int(round(float(row["max_buys"]))))
        setattr(config, "MAX_BUYS_PER_HOUR", buys)
        setattr(config, "MAX_TRADES_PER_HOUR", buys)
        size_max = max(0.05, float(row["size_max"]))
        setattr(config, "PAPER_TRADE_SIZE_MAX_USD", size_max)
        setattr(config, "PAPER_MAX_HOLD_SECONDS", max(30, int(round(float(row["hold_max"])))))
        setattr(config, "HOLD_MAX_SECONDS", max(int(getattr(config, "HOLD_MIN_SECONDS", 30) or 30), int(round(float(row["hold_max"])))))
        setattr(config, "NO_MOMENTUM_EXIT_MIN_AGE_PERCENT", max(1.0, float(row["no_mom_age"])))
        setattr(config, "NO_MOMENTUM_EXIT_MAX_PNL_PERCENT", float(row["no_mom_max_pnl"]))
        setattr(config, "WEAKNESS_EXIT_MIN_AGE_PERCENT", max(1.0, float(row["weak_age"])))
        setattr(config, "WEAKNESS_EXIT_PNL_PERCENT", float(row["weak_pnl"]))
        setattr(config, "PAPER_PARTIAL_TP_TRIGGER_PERCENT", max(0.1, float(row["partial_tp_trigger"])))
        setattr(config, "PAPER_PARTIAL_TP_SELL_FRACTION", max(0.05, min(0.95, float(row["partial_tp_fraction"]))))

    def _write_decision(self, event: dict[str, object]) -> None:
        try:
            payload = dict(event or {})
            payload.setdefault("run_tag", RUN_TAG)
            os.makedirs(os.path.dirname(self.decisions_log_file) or ".", exist_ok=True)
            with open(self.decisions_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=False) + "\n")
        except Exception:
            logger.exception("ORCHESTRATOR decision log write failed")

    def prime(self, *, auto_stats: dict[str, float | int]) -> None:
        self.prev_closed = int(auto_stats.get("closed", 0) or 0)
        self.prev_wins = int(auto_stats.get("wins", 0) or 0)
        self.prev_losses = int(auto_stats.get("losses", 0) or 0)
        self.prev_realized_usd = float(auto_stats.get("realized_pnl_usd", 0.0) or 0.0)

    def record_cycle(self, *, candidates: int, opened: int, market_regime: str) -> None:
        self.window_cycles += 1
        self.window_candidates += int(candidates)
        self.window_opened += int(opened)
        rkey = str(market_regime or "UNKNOWN").strip().upper() or "UNKNOWN"
        self.window_regime_counts[rkey] = int(self.window_regime_counts.get(rkey, 0)) + 1

    def _reset_window(self, *, now_ts: float, auto_stats: dict[str, float | int]) -> None:
        self.last_eval_ts = float(now_ts)
        self.window_cycles = 0
        self.window_candidates = 0
        self.window_opened = 0
        self.window_regime_counts = {}
        self.prev_closed = int(auto_stats.get("closed", 0) or 0)
        self.prev_wins = int(auto_stats.get("wins", 0) or 0)
        self.prev_losses = int(auto_stats.get("losses", 0) or 0)
        self.prev_realized_usd = float(auto_stats.get("realized_pnl_usd", 0.0) or 0.0)

    def maybe_adapt(self, *, policy_state: str, market_regime: str, auto_stats: dict[str, float | int]) -> None:
        if not self.enabled:
            return
        if self.mode not in {"dry_run", "apply"}:
            return
        now_ts = time.time()
        if (now_ts - float(self.last_eval_ts)) < float(self.interval_seconds):
            return
        if int(self.window_cycles) < int(self.min_window_cycles):
            return

        closed_now = int(auto_stats.get("closed", 0) or 0)
        wins_now = int(auto_stats.get("wins", 0) or 0)
        losses_now = int(auto_stats.get("losses", 0) or 0)
        realized_now = float(auto_stats.get("realized_pnl_usd", 0.0) or 0.0)
        closed_delta = int(closed_now - int(self.prev_closed))
        wins_delta = int(wins_now - int(self.prev_wins))
        losses_delta = int(losses_now - int(self.prev_losses))
        realized_delta = float(realized_now - float(self.prev_realized_usd))
        avg_pnl_trade = (realized_delta / float(closed_delta)) if closed_delta > 0 else 0.0
        loss_share = (float(losses_delta) / float(closed_delta)) if closed_delta > 0 else 0.0
        avg_candidates = float(self.window_candidates) / max(1, int(self.window_cycles))
        avg_opened = float(self.window_opened) / max(1, int(self.window_cycles))
        policy_now = str(policy_state or "UNKNOWN").strip().upper() or "UNKNOWN"
        regime_now = str(market_regime or "UNKNOWN").strip().upper() or "UNKNOWN"

        bad_signal = False
        good_signal = False
        if closed_delta >= int(self.min_closed_delta):
            bad_signal = (
                avg_pnl_trade <= float(self.defense_trigger_avg_pnl)
                or loss_share >= float(self.defense_trigger_loss_share)
            )
            good_signal = (
                avg_pnl_trade >= float(self.harvest_trigger_avg_pnl)
                and loss_share <= float(self.harvest_trigger_loss_share)
            )
        if policy_now != "OK":
            bad_signal = True
        elif self.red_force_defense_enabled and regime_now in {"RED", "RISK_OFF"}:
            # RED/RISK_OFF should not force DEFENSE when trade outcome remains positive.
            red_window_bad = (
                realized_delta <= float(self.red_force_defense_realized_delta_usd)
                or (closed_delta >= int(self.min_closed_delta) and avg_pnl_trade <= float(self.defense_trigger_avg_pnl))
                or (closed_delta >= int(self.min_closed_delta) and loss_share >= float(self.defense_trigger_loss_share))
            )
            if red_window_bad:
                bad_signal = True

        if self.current_profile == "harvest":
            self.bad_streak = self.bad_streak + 1 if bad_signal else 0
            self.good_streak = 0
        else:
            self.good_streak = self.good_streak + 1 if good_signal else 0
            self.bad_streak = 0

        target_profile = self.current_profile
        reason = "hold"
        if now_ts >= float(self.cooldown_until_ts):
            if self.current_profile == "harvest" and self.bad_streak >= int(self.defense_enter_streak):
                target_profile = "defense"
                reason = "bad_window_streak"
            elif self.current_profile == "defense" and self.good_streak >= int(self.harvest_enter_streak):
                target_profile = "harvest"
                reason = "good_window_streak"

        changed = target_profile != self.current_profile
        if self.mode == "apply" and changed:
            self.current_profile = target_profile
            self._apply_profile(self.current_profile)
            self.cooldown_until_ts = now_ts + (max(0, self.cooldown_windows) * max(1, self.interval_seconds))
            self.bad_streak = 0
            self.good_streak = 0

        logger.warning(
            "ORCHESTRATOR mode=%s profile=%s target=%s changed=%s reason=%s policy=%s regime=%s avg_cand=%.2f avg_opened=%.2f closed_delta=%s wins_delta=%s losses_delta=%s avg_pnl_trade=$%.4f loss_share=%.2f",
            self.mode,
            self.current_profile,
            target_profile,
            changed,
            reason,
            policy_now,
            regime_now,
            avg_candidates,
            avg_opened,
            closed_delta,
            wins_delta,
            losses_delta,
            avg_pnl_trade,
            loss_share,
        )
        self._write_decision(
            {
                "ts": time.time(),
                "mode": self.mode,
                "profile_now": self.current_profile,
                "profile_target": target_profile,
                "changed": bool(changed),
                "reason": reason,
                "policy": policy_now,
                "regime": regime_now,
                "avg_candidates": round(avg_candidates, 4),
                "avg_opened": round(avg_opened, 4),
                "closed_delta": int(closed_delta),
                "wins_delta": int(wins_delta),
                "losses_delta": int(losses_delta),
                "avg_pnl_trade_usd": round(avg_pnl_trade, 6),
                "loss_share": round(loss_share, 6),
                "regime_top": _format_top_counts(self.window_regime_counts, limit=1),
                "bad_streak": int(self.bad_streak),
                "good_streak": int(self.good_streak),
            }
        )
        self._reset_window(now_ts=now_ts, auto_stats=auto_stats)


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
            event = candidate_decision_event(dict(event), run_tag=RUN_TAG)
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, sort_keys=False) + "\n")
        except Exception:
            logger.exception("CANDIDATE_LOG write failed")


class ProfileAutoStopController:
    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "PROFILE_AUTOSTOP_ENABLED", False))
        self.notify_enabled = bool(getattr(config, "PROFILE_AUTOSTOP_NOTIFY_ENABLED", True))
        self.interval_seconds = int(getattr(config, "PROFILE_AUTOSTOP_EVAL_INTERVAL_SECONDS", 120) or 120)
        self.min_runtime_seconds = int(getattr(config, "PROFILE_AUTOSTOP_MIN_RUNTIME_SECONDS", 3600) or 3600)
        self.min_closed_trades = int(getattr(config, "PROFILE_AUTOSTOP_MIN_CLOSED_TRADES", 24) or 24)
        self.min_realized_pnl_usd = float(getattr(config, "PROFILE_AUTOSTOP_MIN_REALIZED_PNL_USD", 0.0) or 0.0)
        self.min_avg_pnl_per_trade_usd = float(
            getattr(config, "PROFILE_AUTOSTOP_MIN_AVG_PNL_PER_TRADE_USD", 0.0005) or 0.0005
        )
        self.max_loss_share = float(getattr(config, "PROFILE_AUTOSTOP_MAX_LOSS_SHARE", 0.58) or 0.58)
        self.max_drawdown_from_peak_usd = float(
            getattr(config, "PROFILE_AUTOSTOP_MAX_DRAWDOWN_FROM_PEAK_USD", 0.18) or 0.18
        )
        self.min_fail_signals = int(getattr(config, "PROFILE_AUTOSTOP_MIN_FAIL_SIGNALS", 2) or 2)
        self.window_minutes = int(getattr(config, "PROFILE_AUTOSTOP_WINDOW_MINUTES", 60) or 60)
        self.window_min_closed = int(getattr(config, "PROFILE_AUTOSTOP_WINDOW_MIN_CLOSED_TRADES", 10) or 10)
        self.min_pnl_per_hour_usd = float(getattr(config, "PROFILE_AUTOSTOP_MIN_PNL_PER_HOUR_USD", 0.0) or 0.0)
        self.max_sl_sum_window_usd = float(getattr(config, "PROFILE_AUTOSTOP_MAX_SL_SUM_WINDOW_USD", -999.0) or -999.0)
        self.event_tag = str(
            getattr(config, "PROFILE_AUTOSTOP_EVENT_TAG", "[AUTOSTOP][PROFILE_STOPPED]") or "[AUTOSTOP][PROFILE_STOPPED]"
        ).strip()
        self.started_ts = time.time()
        self.next_eval_ts = self.started_ts + max(30, int(self.interval_seconds))
        self.peak_realized_pnl_usd = 0.0
        self.triggered = False

    def maybe_stop(self, *, auto_stats: dict[str, float | int]) -> tuple[bool, dict[str, object] | None]:
        if not self.enabled or self.triggered:
            return False, None
        now_ts = time.time()
        if now_ts < float(self.next_eval_ts):
            return False, None
        self.next_eval_ts = now_ts + max(30, int(self.interval_seconds))

        runtime_seconds = max(0, int(now_ts - float(self.started_ts)))
        closed = int(auto_stats.get("closed", 0) or 0)
        wins = int(auto_stats.get("wins", 0) or 0)
        losses = int(auto_stats.get("losses", 0) or 0)
        realized = float(auto_stats.get("realized_pnl_usd", 0.0) or 0.0)
        self.peak_realized_pnl_usd = max(float(self.peak_realized_pnl_usd), float(realized))

        if runtime_seconds < int(self.min_runtime_seconds):
            return False, None
        if closed < int(self.min_closed_trades):
            return False, None

        avg_pnl_per_trade = (realized / float(closed)) if closed > 0 else 0.0
        loss_share = (float(losses) / float(closed)) if closed > 0 else 0.0
        drawdown_from_peak = float(realized - float(self.peak_realized_pnl_usd))
        window_closed = int(auto_stats.get("autostop_window_closed", 0) or 0)
        window_pnl_per_hour = float(auto_stats.get("autostop_window_pnl_per_hour_usd", 0.0) or 0.0)
        window_loss_share = float(auto_stats.get("autostop_window_loss_share", 0.0) or 0.0)
        window_sl_sum = float(auto_stats.get("autostop_window_sl_sum_usd", 0.0) or 0.0)
        window_avg_pnl = float(auto_stats.get("autostop_window_avg_pnl_per_trade_usd", 0.0) or 0.0)
        window_realized = float(auto_stats.get("autostop_window_realized_pnl_usd", 0.0) or 0.0)

        fail_reasons: list[str] = []
        if realized < float(self.min_realized_pnl_usd):
            fail_reasons.append(
                f"realized_pnl_usd {realized:.4f} < {float(self.min_realized_pnl_usd):.4f}"
            )
        if avg_pnl_per_trade < float(self.min_avg_pnl_per_trade_usd):
            fail_reasons.append(
                f"avg_pnl_trade {avg_pnl_per_trade:.5f} < {float(self.min_avg_pnl_per_trade_usd):.5f}"
            )
        if loss_share > float(self.max_loss_share):
            fail_reasons.append(f"loss_share {loss_share:.3f} > {float(self.max_loss_share):.3f}")
        if drawdown_from_peak <= -abs(float(self.max_drawdown_from_peak_usd)):
            fail_reasons.append(
                f"drawdown_from_peak {drawdown_from_peak:.4f} <= -{abs(float(self.max_drawdown_from_peak_usd)):.4f}"
            )
        if (
            window_closed >= int(self.window_min_closed)
            and float(self.min_pnl_per_hour_usd) > 0.0
            and window_pnl_per_hour < float(self.min_pnl_per_hour_usd)
        ):
            fail_reasons.append(
                f"pnl_per_hour {window_pnl_per_hour:.4f} < {float(self.min_pnl_per_hour_usd):.4f}"
            )
        if (
            window_closed >= int(self.window_min_closed)
            and float(self.max_sl_sum_window_usd) > -999.0
            and window_sl_sum < float(self.max_sl_sum_window_usd)
        ):
            fail_reasons.append(
                f"sl_sum_window {window_sl_sum:.4f} < {float(self.max_sl_sum_window_usd):.4f}"
            )

        if len(fail_reasons) < int(self.min_fail_signals):
            return False, None

        self.triggered = True
        summary = (
            f"profile={RUN_TAG} runtime={runtime_seconds}s closed={closed} wins={wins} losses={losses} "
            f"realized={realized:.4f} avg={avg_pnl_per_trade:.5f} loss_share={loss_share:.3f} "
            f"drawdown_from_peak={drawdown_from_peak:.4f} "
            f"window({int(self.window_minutes)}m): closed={window_closed} pnl={window_realized:.4f} "
            f"pnl_h={window_pnl_per_hour:.4f} avg={window_avg_pnl:.5f} loss_share={window_loss_share:.3f} sl_sum={window_sl_sum:.4f} "
            f"fail_signals={len(fail_reasons)}"
        )
        reason_text = "; ".join(fail_reasons)
        event_tag = str(self.event_tag or "[AUTOSTOP][PROFILE_STOPPED]").strip()
        tagged_reason = f"{event_tag} {reason_text}".strip()
        try:
            stop_file = _graceful_stop_file_path()
            os.makedirs(os.path.dirname(stop_file) or ".", exist_ok=True)
            with open(stop_file, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": now_ts,
                            "run_tag": RUN_TAG,
                            "event": "PROFILE_AUTOSTOP",
                            "reason": tagged_reason,
                            "summary": summary,
                        },
                        ensure_ascii=False,
                        sort_keys=False,
                    )
                )
        except Exception:
            logger.exception("PROFILE_AUTOSTOP failed to write graceful stop file")

        event = {
            "event_type": "PROFILE_AUTOSTOP",
            "symbol": "PROFILE_STOPPED",
            "score": 0,
            "recommendation": "STOP",
            "risk_level": "WARNING",
            "name": f"{event_tag} {RUN_TAG} stopped",
            "breakdown": {
                "run_tag": RUN_TAG,
                "reason": tagged_reason,
                "summary": summary,
                "fail_reasons": fail_reasons,
                "event_tag": event_tag,
                "window_minutes": int(self.window_minutes),
                "window_closed": int(window_closed),
                "window_realized_pnl_usd": float(window_realized),
                "window_pnl_per_hour_usd": float(window_pnl_per_hour),
                "window_avg_pnl_per_trade_usd": float(window_avg_pnl),
                "window_loss_share": float(window_loss_share),
                "window_sl_sum_usd": float(window_sl_sum),
            },
            "address": "",
            "liquidity": 0.0,
            "volume_5m": 0.0,
            "price_change_5m": 0.0,
            "warning_flags": len(fail_reasons),
        }
        return True, event


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

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(run_tag)s] %(name)s: %(message)s")
    prev_factory = logging.getLogRecordFactory()

    def _record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = prev_factory(*args, **kwargs)
        if not hasattr(record, "run_tag"):
            setattr(record, "run_tag", RUN_TAG)
        return record

    logging.setLogRecordFactory(_record_factory)

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
    safe_volume_probe_state: dict[str, dict[str, float | int]] = {}
    filter_fail_reasons_session: dict[str, int] = {}
    adaptive_filters = AdaptiveFilterController()
    adaptive_init_stats = auto_trader.get_stats()
    adaptive_filters.prev_closed = int(adaptive_init_stats.get("closed", 0) or 0)
    adaptive_filters.prev_realized_usd = float(adaptive_init_stats.get("realized_pnl_usd", 0.0) or 0.0)
    autonomy_controller = AutonomousControlController()
    autonomy_controller.prime(auto_stats=adaptive_init_stats)
    strategy_orchestrator = StrategyOrchestrator()
    strategy_orchestrator.prime(auto_stats=adaptive_init_stats)
    profile_autostop = ProfileAutoStopController()
    v2_universe = UniverseFlowController()
    v2_quality_gate = UniverseQualityGateController()
    v2_safety_budget = SafetyBudgetController()
    v2_calibrator = UnifiedCalibrator()
    v2_policy_router = PolicyEntryRouter()
    v2_dual_entry = DualEntryController()
    v2_rolling_edge = RollingEdgeGovernor()
    v2_kpi_loop = RuntimeKpiLoop()
    v2_champion_guard = MatrixChampionGuard(
        run_tag=RUN_TAG,
        paper_state_file=str(getattr(config, "PAPER_STATE_FILE", "")),
        graceful_stop_file=_graceful_stop_file_path(),
    )
    mini_analyzer = MiniAnalyzer()
    mini_analyzer.prime(
        closed=int(adaptive_init_stats.get("closed", 0) or 0),
        realized_usd=float(adaptive_init_stats.get("realized_pnl_usd", 0.0) or 0.0),
    )
    v2_source_qos = SourceQosController()
    _enforce_source_qos_dual_entry_guard()
    recent_candidate_counts: deque[int] = deque(
        maxlen=max(3, int(getattr(config, "MARKET_REGIME_WINDOW_CYCLES", 12) or 12))
    )
    market_regime_current = "YELLOW"
    market_regime_reason = "init"
    market_mode_risk_streak = 0
    market_mode_recover_streak = 0
    market_regime_prev = market_regime_current
    runtime_tuner_lock_prev: bool | None = None
    cycle_index = 0

    logger.info("Local mode started (Telegram disabled).")
    _write_heartbeat(stage="startup", cycle_index=cycle_index, open_trades=0)
    try:
        while True:
            cycle_index += 1
            _write_heartbeat(stage="cycle_begin", cycle_index=cycle_index, open_trades=len(auto_trader.open_positions))
            runtime_tuner_lock_active = _runtime_tuner_control_active(run_tag=RUN_TAG)
            if runtime_tuner_lock_prev is None or bool(runtime_tuner_lock_prev) != bool(runtime_tuner_lock_active):
                if runtime_tuner_lock_active:
                    logger.warning(
                        "RUNTIME_TUNER_CONTROL active run_tag=%s: local adaptive controllers locked (kpi/calibration/rolling/orchestrator).",
                        RUN_TAG,
                    )
                else:
                    logger.warning(
                        "RUNTIME_TUNER_CONTROL inactive run_tag=%s: local adaptive controllers unlocked.",
                        RUN_TAG,
                    )
                runtime_tuner_lock_prev = bool(runtime_tuner_lock_active)
            hot_applied, hot_count, hot_detail = _runtime_tuner_apply_runtime_overrides(run_tag=RUN_TAG)
            if hot_applied:
                logger.warning(
                    "RUNTIME_TUNER_HOT_APPLY run_tag=%s keys=%s details=%s",
                    RUN_TAG,
                    int(hot_count),
                    hot_detail,
                )
                applied_keys = _runtime_tuner_applied_keys()
                reload_targets = _runtime_tuner_reload_targets(applied_keys)
                reloaded: list[str] = []
                if "universe" in reload_targets:
                    v2_universe = UniverseFlowController()
                    reloaded.append("universe")
                if "source_qos" in reload_targets:
                    v2_source_qos = SourceQosController()
                    _enforce_source_qos_dual_entry_guard()
                    reloaded.append("source_qos")
                if "quality_gate" in reload_targets:
                    v2_quality_gate = UniverseQualityGateController()
                    reloaded.append("quality_gate")
                if "safety_budget" in reload_targets:
                    v2_safety_budget = SafetyBudgetController()
                    reloaded.append("safety_budget")
                if "calibrator" in reload_targets:
                    v2_calibrator = UnifiedCalibrator()
                    reloaded.append("calibrator")
                if "policy_router" in reload_targets:
                    v2_policy_router = PolicyEntryRouter()
                    reloaded.append("policy_router")
                if "dual_entry" in reload_targets:
                    v2_dual_entry = DualEntryController()
                    reloaded.append("dual_entry")
                if "rolling_edge" in reload_targets:
                    v2_rolling_edge = RollingEdgeGovernor()
                    reloaded.append("rolling_edge")
                if "kpi_loop" in reload_targets:
                    v2_kpi_loop = RuntimeKpiLoop()
                    reloaded.append("kpi_loop")
                if reloaded:
                    logger.warning(
                        "RUNTIME_TUNER_HOT_APPLY_REFRESH run_tag=%s targets=%s applied_keys=%s",
                        RUN_TAG,
                        ",".join(sorted(reloaded)),
                        ",".join(applied_keys[:8]) + (",..." if len(applied_keys) > 8 else ""),
                    )
            elif str(hot_detail).startswith("error:"):
                last_err = str(_RUNTIME_TUNER_PATCH_STATE.get("last_error", "") or "")
                if hot_detail != last_err:
                    _RUNTIME_TUNER_PATCH_STATE["last_error"] = hot_detail
                    logger.warning("RUNTIME_TUNER_HOT_APPLY run_tag=%s %s", RUN_TAG, hot_detail)
            if _graceful_stop_requested():
                stop_path = _graceful_stop_file_path()
                stop_meta = _read_graceful_stop_meta()
                if stop_meta:
                    logger.warning(
                        "GRACEFUL_STOP requested path=%s source=%s reason=%s actor=%s ts=%s raw=%s",
                        stop_path,
                        str(stop_meta.get("source", "unknown") or "unknown"),
                        str(stop_meta.get("reason", "unknown") or "unknown"),
                        str(stop_meta.get("actor", "unknown") or "unknown"),
                        str(stop_meta.get("timestamp", "") or ""),
                        str(stop_meta.get("raw", "") or ""),
                    )
                else:
                    logger.warning("GRACEFUL_STOP requested path=%s", stop_path)
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
                if tokens:
                    before_universe = len(tokens)
                    tokens = v2_universe.filter_tokens(tokens or [])
                    if len(tokens) != before_universe:
                        logger.info(
                            "V2_UNIVERSE cycle=%s before=%s after=%s",
                            cycle_index,
                            before_universe,
                            len(tokens),
                        )
                tokens_seen_total = len(tokens or [])
                seen_by_source = _count_sources(tokens)
                passed_by_source: dict[str, int] = {}
                planned_by_source: dict[str, int] = {}
                source_qos_meta: dict[str, Any] = {
                    "enabled": bool(getattr(config, "V2_SOURCE_QOS_ENABLED", True)),
                    "in_total": tokens_seen_total,
                    "out_total": tokens_seen_total,
                    "drop_counts": {},
                }
                source_qos_dropped: list[dict[str, Any]] = []
                source_flow_cycle: dict[str, dict[str, Any]] = {}
                source_qos_event: dict[str, Any] | None = None
                source_qos_snapshot: dict[str, Any] = {}
                if tokens:
                    tokens, source_qos_dropped, source_qos_meta = v2_source_qos.filter_tokens(tokens or [])
                    if int(source_qos_meta.get("in_total", 0) or 0) != int(source_qos_meta.get("out_total", 0) or 0):
                        logger.info(
                            "V2_SOURCE_QOS_FILTER in=%s out=%s drops=%s caps=%s",
                            int(source_qos_meta.get("in_total", 0) or 0),
                            int(source_qos_meta.get("out_total", 0) or 0),
                            dict(source_qos_meta.get("drop_counts", {}) or {}),
                            dict(source_qos_meta.get("source_caps_effective", {}) or {}),
                        )

                high_quality = 0
                alerts_sent = 0
                trade_candidates: list[tuple[dict, dict]] = []
                safety_checked = 0
                safety_fail_closed_hits = 0
                heavy_dedup_skipped = 0
                heavy_dedup_override = 0
                dedup_repeat_intervals_cycle: list[float] = []
                v2_safety_budget.reset_cycle()
                excluded_addresses = _excluded_trade_addresses()
                now_mono = time.monotonic()
                if heavy_seen_until:
                    expired = [a for a, ts in heavy_seen_until.items() if float(ts or 0.0) <= now_mono]
                    for a in expired:
                        heavy_seen_until.pop(a, None)
                        heavy_last_liquidity.pop(a, None)
                        heavy_last_volume_5m.pop(a, None)
                if safe_volume_probe_state:
                    expired_probe = [
                        a
                        for a, row in safe_volume_probe_state.items()
                        if float((row or {}).get("until", 0.0) or 0.0) <= now_mono
                    ]
                    for a in expired_probe:
                        safe_volume_probe_state.pop(a, None)
                if tokens or source_qos_dropped:
                    cycle_filter_fails: dict[str, int] = {}
                    mode_profile = _market_mode_entry_profile(market_regime_current)
                    regime_score_delta = int(mode_profile.get("score_delta", 0) or 0)
                    regime_volume_mult = float(mode_profile.get("volume_mult", 1.0) or 1.0)
                    regime_edge_mult = float(mode_profile.get("edge_mult", 1.0) or 1.0)
                    regime_size_mult = float(mode_profile.get("size_mult", 1.0) or 1.0)
                    regime_hold_mult = float(mode_profile.get("hold_mult", 1.0) or 1.0)
                    regime_partial_tp_trigger_mult = float(mode_profile.get("partial_tp_trigger_mult", 1.0) or 1.0)
                    regime_partial_tp_sell_mult = float(mode_profile.get("partial_tp_sell_mult", 1.0) or 1.0)
                    allow_soft = bool(mode_profile.get("allow_soft", True))
                    soft_cap = int(mode_profile.get("soft_cap", 0) or 0)
                    base_score_floor = int(getattr(config, "MIN_TOKEN_SCORE", 70) or 70)
                    strict_score_min = max(
                        base_score_floor,
                        int(getattr(config, "MARKET_MODE_STRICT_SCORE", 78) or 78) + int(regime_score_delta),
                    )
                    soft_score_base = int(getattr(config, "MARKET_MODE_SOFT_SCORE", 70) or 70)
                    soft_score_min = max(base_score_floor, min(strict_score_min, soft_score_base + int(regime_score_delta)))
                    soft_selected_cycle = 0
                    excluded_symbols_cycle = _excluded_trade_symbols()
                    excluded_keywords_cycle = _excluded_trade_symbol_keywords()

                    def _filter_fail(reason: str, token: dict, extra: str = "") -> None:
                        key = str(reason or "unknown").strip().lower() or "unknown"
                        quality = _candidate_quality_features(token, (token.get("score_data") or {}))
                        candidate_id = str(token.get("_candidate_id", "") or "")
                        filter_fail_reasons_session[key] = int(filter_fail_reasons_session.get(key, 0)) + 1
                        cycle_filter_fails[key] = int(cycle_filter_fails.get(key, 0)) + 1
                        candidate_writer.write(
                            {
                                "ts": time.time(),
                                "cycle_index": cycle_index,
                                "candidate_id": candidate_id,
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

                    if source_qos_dropped:
                        for drop_idx, drop_row in enumerate(source_qos_dropped):
                            row = dict(drop_row or {})
                            token_drop = dict(row.get("token") or {})
                            if not token_drop:
                                continue
                            if not str(token_drop.get("_candidate_id", "") or "").strip():
                                drop_addr = normalize_address(token_drop.get("address", ""))
                                token_drop["_candidate_id"] = f"{RUN_TAG}:{cycle_index}:qos:{drop_idx}:{drop_addr or 'noaddr'}"
                            _filter_fail(
                                str(row.get("reason", "source_qos_drop") or "source_qos_drop"),
                                token_drop,
                                str(row.get("extra", "") or ""),
                            )

                    cycle_seen_addresses: set[str] = set()
                    for token_idx, token in enumerate(tokens):
                        token_address = normalize_address(token.get("address", ""))
                        candidate_id = f"{RUN_TAG}:{cycle_index}:{token_idx}:{token_address or 'noaddr'}"
                        token["_candidate_id"] = candidate_id
                        if _is_placeholder_trade_address(token_address):
                            _filter_fail("invalid_address", token, f"address={token_address or 'missing'}")
                            continue
                        if token_address in cycle_seen_addresses:
                            _filter_fail("duplicate_address", token, "cycle_dedup")
                            continue
                        cycle_seen_addresses.add(token_address)
                        blk_blocked, blk_reason = auto_trader.blacklist_status(token_address)
                        if blk_blocked:
                            _filter_fail("blacklist", token, str(blk_reason or "blacklisted"))
                            continue
                        symbol_now = str(token.get("symbol", "") or "")
                        if auto_trader.is_hard_blocked(token_address, symbol=symbol_now):
                            auto_trader.add_hard_blocklist_entry(token_address)
                            _filter_fail("hard_blocklist", token, f"address={token_address}")
                            continue
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
                        symbol_block_reason = _excluded_symbol_reason(
                            str(token.get("symbol", "") or ""),
                            excluded_symbols=excluded_symbols_cycle,
                            excluded_keywords=excluded_keywords_cycle,
                        )
                        if symbol_block_reason:
                            _filter_fail(symbol_block_reason, token)
                            continue
                        if not _belongs_to_candidate_shard(token_address):
                            _filter_fail("shard_skip", token)
                            continue
                        score_data = scorer.calculate_score(token)
                        token["score_data"] = score_data
                        if int(score_data.get("score", 0)) >= 70:
                            high_quality += 1
                        score_now = int(score_data.get("score", 0) or 0)
                        if score_now >= strict_score_min:
                            entry_tier = "A"
                        elif score_now >= soft_score_min:
                            entry_tier = "B"
                        else:
                            entry_tier = ""

                        if config.AUTO_FILTER_ENABLED and not entry_tier:
                            _filter_fail(
                                "score_min",
                                token,
                                f"score={score_now} strict_min={strict_score_min} soft_min={soft_score_min} mode={market_regime_current}",
                            )
                            continue
                        if entry_tier == "B" and not allow_soft:
                            _filter_fail(
                                "tier_b_blocked_mode",
                                token,
                                f"score={score_now} mode={market_regime_current}",
                            )
                            continue
                        if entry_tier == "B" and soft_cap > 0 and soft_selected_cycle >= soft_cap:
                            _filter_fail(
                                "tier_b_cycle_cap",
                                token,
                                f"score={score_now} cap={soft_cap} mode={market_regime_current}",
                            )
                            continue

                        if config.SAFE_TEST_MODE:
                            if float(token.get("liquidity") or 0) < float(config.SAFE_MIN_LIQUIDITY_USD):
                                _filter_fail("safe_liquidity", token)
                                continue
                            safe_volume_floor = float(config.SAFE_MIN_VOLUME_5M_USD) * max(0.1, float(regime_volume_mult))
                            token_volume_5m = float(token.get("volume_5m") or 0.0)
                            if token_volume_5m < safe_volume_floor:
                                soft_enabled = bool(getattr(config, "SAFE_VOLUME_TWO_TIER_ENABLED", False))
                                soft_ratio = max(
                                    0.20,
                                    min(0.99, float(getattr(config, "SAFE_VOLUME_TWO_TIER_SOFT_RATIO", 0.80) or 0.80)),
                                )
                                soft_min_volume = float(safe_volume_floor) * float(soft_ratio)
                                if soft_enabled and token_address and token_volume_5m >= soft_min_volume:
                                    stable_cycles_required = max(
                                        1,
                                        int(getattr(config, "SAFE_VOLUME_TWO_TIER_STABLE_CYCLES", 2) or 2),
                                    )
                                    stability_ratio = max(
                                        0.50,
                                        min(
                                            1.20,
                                            float(
                                                getattr(
                                                    config,
                                                    "SAFE_VOLUME_TWO_TIER_STABILITY_RATIO",
                                                    0.85,
                                                )
                                                or 0.85
                                            ),
                                        ),
                                    )
                                    probe_ttl_seconds = max(
                                        60,
                                        int(getattr(config, "SAFE_VOLUME_TWO_TIER_TTL_SECONDS", 900) or 900),
                                    )
                                    prev_probe = dict(safe_volume_probe_state.get(token_address, {}) or {})
                                    prev_vol = float(prev_probe.get("vol", 0.0) or 0.0)
                                    prev_liq = float(prev_probe.get("liq", 0.0) or 0.0)
                                    prev_hits = int(prev_probe.get("hits", 0) or 0)
                                    token_liq_now = float(token.get("liquidity") or 0.0)
                                    stable_volume = (prev_vol <= 0.0) or (token_volume_5m >= (prev_vol * stability_ratio))
                                    stable_liquidity = (prev_liq <= 0.0) or (token_liq_now >= (prev_liq * stability_ratio))
                                    probe_hits = (prev_hits + 1) if (stable_volume and stable_liquidity) else 1
                                    safe_volume_probe_state[token_address] = {
                                        "hits": int(probe_hits),
                                        "vol": float(token_volume_5m),
                                        "liq": float(token_liq_now),
                                        "until": float(now_mono + float(probe_ttl_seconds)),
                                    }
                                    if probe_hits >= stable_cycles_required:
                                        token["_safe_volume_soft_pass"] = True
                                        token["_safe_volume_soft_detail"] = (
                                            f"vol5m={token_volume_5m:.0f}/{safe_volume_floor:.0f} "
                                            f"soft_min={soft_min_volume:.0f} hits={probe_hits}/{stable_cycles_required}"
                                        )
                                        logger.info(
                                            "SAFE_VOLUME_SOFT_PASS token=%s vol5m=%.0f min=%.0f soft_min=%.0f hits=%s/%s mode=%s",
                                            token.get("symbol", "N/A"),
                                            token_volume_5m,
                                            safe_volume_floor,
                                            soft_min_volume,
                                            probe_hits,
                                            stable_cycles_required,
                                            market_regime_current,
                                        )
                                    else:
                                        _filter_fail(
                                            "safe_volume_quarantine",
                                            token,
                                            (
                                                f"vol5m={token_volume_5m:.0f} min={safe_volume_floor:.0f} "
                                                f"soft_min={soft_min_volume:.0f} hits={probe_hits}/{stable_cycles_required} "
                                                f"regime={market_regime_current}"
                                            ),
                                        )
                                        continue
                                else:
                                    _filter_fail(
                                        "safe_volume",
                                        token,
                                        f"vol5m={token_volume_5m:.0f} min={safe_volume_floor:.0f} regime={market_regime_current}",
                                    )
                                    continue
                            elif token_address:
                                safe_volume_probe_state.pop(token_address, None)
                            if int(token.get("age_seconds") or 0) < int(config.SAFE_MIN_AGE_SECONDS):
                                _filter_fail("safe_age", token)
                                continue
                            if abs(float(token.get("price_change_5m") or 0)) > float(config.SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT):
                                _filter_fail("safe_change_5m", token)
                                continue

                        # Heavy safety-check dedup: avoid re-checking same token address too often.
                        heavy_ttl_raw = getattr(config, "HEAVY_CHECK_DEDUP_TTL_SECONDS", 900)
                        try:
                            heavy_ttl = int(heavy_ttl_raw)
                        except Exception:
                            heavy_ttl = 900
                        heavy_ttl = max(0, heavy_ttl)
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
                        if not v2_safety_budget.allow(token):
                            _filter_fail("safety_budget", token)
                            continue
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
                            if bool(getattr(config, "ENTRY_FAIL_CLOSED_ON_SAFETY_GAP", True)):
                                safety_source = str((safety or {}).get("source", "") or "").strip().lower()
                                allowed_sources = set(getattr(config, "ENTRY_ALLOWED_SAFETY_SOURCES", ["goplus"]) or ["goplus"])
                                if (not safety_source) or (safety_source not in allowed_sources):
                                    _filter_fail("safe_source", token, f"source={safety_source or 'missing'}")
                                    continue
                                fail_reason = str((safety or {}).get("fail_reason", "") or "").strip()
                                if fail_reason:
                                    is_transient_source = safety_source in {"transient_fallback", "cache_transient"}
                                    transient_mode = bool(getattr(config, "TOKEN_SAFETY_TRANSIENT_DEGRADED_ENABLED", True))
                                    if (not is_transient_source) or (not transient_mode):
                                        _filter_fail("safe_fail_reason", token, fail_reason)
                                        continue
                            if bool(getattr(config, "ENTRY_BLOCK_RISKY_CONTRACT_FLAGS", True)):
                                risky_flags = [
                                    str(x).strip().lower()
                                    for x in ((safety or {}).get("risky_flags") or [])
                                    if str(x).strip()
                                ]
                                hard_flags = set(getattr(config, "ENTRY_HARD_RISKY_FLAG_CODES", []) or [])
                                if any(flag in hard_flags for flag in risky_flags):
                                    _filter_fail("safe_risky_flags", token, ",".join(risky_flags))
                                    continue
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
                        token["_regime_size_mult"] = float(regime_size_mult)
                        token["_regime_hold_mult"] = float(regime_hold_mult)
                        token["_regime_partial_tp_trigger_mult"] = float(regime_partial_tp_trigger_mult)
                        token["_regime_partial_tp_sell_mult"] = float(regime_partial_tp_sell_mult)
                        token["_regime_name"] = str(market_regime_current)
                        token["_entry_tier"] = str(entry_tier)
                        source_name = str(token.get("source", "") or "").strip().lower()
                        if (
                            bool(getattr(config, "AUTO_TRADE_PAPER", False))
                            and source_name.startswith("watchlist")
                            and not bool(getattr(config, "PAPER_ALLOW_WATCHLIST_SOURCE", False))
                        ):
                            strict_guard_enabled = bool(getattr(config, "PAPER_WATCHLIST_STRICT_GUARD_ENABLED", True))
                            if not strict_guard_enabled:
                                _filter_fail("source_disabled", token, "source=watchlist paper_mode")
                                continue
                            watch_score = int(score_data.get("score", 0) or 0)
                            watch_liq = float(token.get("liquidity") or 0.0)
                            watch_vol = float(token.get("volume_5m") or 0.0)
                            watch_abs_chg = abs(float(token.get("price_change_5m") or 0.0))
                            min_score = int(getattr(config, "PAPER_WATCHLIST_MIN_SCORE", 90) or 90)
                            min_liq = float(getattr(config, "PAPER_WATCHLIST_MIN_LIQUIDITY_USD", 150000.0) or 150000.0)
                            min_vol = float(getattr(config, "PAPER_WATCHLIST_MIN_VOLUME_5M_USD", 500.0) or 500.0)
                            min_abs_chg = float(getattr(config, "PAPER_WATCHLIST_MIN_ABS_CHANGE_5M", 0.30) or 0.30)
                            if (
                                watch_score < min_score
                                or watch_liq < min_liq
                                or watch_vol < min_vol
                                or watch_abs_chg < min_abs_chg
                            ):
                                _filter_fail(
                                    "watchlist_strict_guard",
                                    token,
                                    (
                                        f"score={watch_score}/{min_score} "
                                        f"liq={watch_liq:.0f}/{min_liq:.0f} "
                                        f"vol5m={watch_vol:.0f}/{min_vol:.0f} "
                                        f"abs5m={watch_abs_chg:.2f}/{min_abs_chg:.2f}"
                                    ),
                                )
                                continue
                        if entry_tier == "B":
                            soft_selected_cycle += 1

                        logger.info(
                            "FILTER_PASS token=%s tier=%s score=%s mode=%s risk=%s liq=%.0f vol5m=%.0f",
                            token.get("symbol", "N/A"),
                            entry_tier,
                            score_data.get("score", 0),
                            market_regime_current,
                            token.get("risk_level", "N/A"),
                            float(token.get("liquidity") or 0),
                            float(token.get("volume_5m") or 0),
                        )
                        candidate_writer.write(
                            {
                                "ts": time.time(),
                                "cycle_index": cycle_index,
                                "candidate_id": candidate_id,
                                "source_mode": source_mode,
                                "decision_stage": "post_filters",
                                "decision": "candidate_pass",
                                "reason": "passed_all_filters",
                                "market_regime": market_regime_current,
                                "entry_tier": entry_tier,
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
                        v2_universe.record_candidate_pass(token_address, str(token.get("symbol", "") or ""))
                        # Optional: avoid alert spam from watchlist flow.
                        if str(token.get("source", "")).lower() == "watchlist" and not bool(
                            getattr(config, "WATCHLIST_ALERTS_ENABLED", False)
                        ):
                            pass
                        else:
                            alerts_sent += await local_alerter.send_alert(token, score_data, safety=safety)
                    passed_by_source = _count_candidate_sources(trade_candidates)
                else:
                    cycle_filter_fails = {}
                    passed_by_source = {}

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
                raw_market_mode, raw_market_reason = _detect_market_regime(
                    policy_state=policy_state,
                    source_stats=source_stats,
                    safety_stats=safety_stats,
                    avg_candidates_recent=avg_candidates_recent,
                )
                market_regime_current, market_regime_reason, market_mode_risk_streak, market_mode_recover_streak = (
                    _apply_market_mode_hysteresis(
                        raw_mode=raw_market_mode,
                        raw_reason=raw_market_reason,
                        current_mode=market_regime_current,
                        risk_streak=market_mode_risk_streak,
                        recover_streak=market_mode_recover_streak,
                    )
                )
                policy_mode_current = policy_state
                candidates_pre_route = list(trade_candidates)
                candidate_lookup: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
                for token_data, score_data in candidates_pre_route:
                    cid = str((token_data or {}).get("_candidate_id", "") or "")
                    if cid:
                        candidate_lookup[cid] = (token_data, score_data)
                quality_gate_meta: dict[str, object] = {
                    "enabled": bool(getattr(config, "V2_QUALITY_GATE_ENABLED", False)),
                    "in_total": len(candidates_pre_route),
                    "out_total": len(candidates_pre_route),
                    "core_out": 0,
                    "explore_out": 0,
                    "probe_out": 0,
                    "drop_counts": {},
                }
                candidates_quality = list(candidates_pre_route)
                if candidates_pre_route:
                    candidates_quality, quality_gate_meta = v2_quality_gate.filter_candidates(
                        candidates=candidates_pre_route,
                        auto_trader=auto_trader,
                        market_mode=market_regime_current,
                    )
                    logger.info(
                        (
                            "V2_QUALITY_GATE enabled=%s regime=%s in=%s out=%s core_out=%s explore_out=%s probe_out=%s "
                            "quota=%s max_symbol_share=%.3f drop=%s sources_used=%s"
                        ),
                        bool(quality_gate_meta.get("enabled", False)),
                        str(quality_gate_meta.get("regime", market_regime_current)),
                        int(quality_gate_meta.get("in_total", 0) or 0),
                        int(quality_gate_meta.get("out_total", 0) or 0),
                        int(quality_gate_meta.get("core_out", 0) or 0),
                        int(quality_gate_meta.get("explore_out", 0) or 0),
                        int(quality_gate_meta.get("probe_out", 0) or 0),
                        int(quality_gate_meta.get("explore_quota", 0) or 0),
                        float(quality_gate_meta.get("max_symbol_share_out", 0.0) or 0.0),
                        dict(quality_gate_meta.get("drop_counts", {}) or {}),
                        dict(quality_gate_meta.get("source_used", {}) or {}),
                    )
                quality_dropped = list(quality_gate_meta.get("dropped", []) or [])
                if quality_dropped:
                    for drop_row in quality_dropped:
                        cid = str((drop_row or {}).get("candidate_id", "") or "")
                        token_data, score_data = candidate_lookup.get(cid, ({}, {}))
                        candidate_writer.write(
                            {
                                "ts": time.time(),
                                "cycle_index": cycle_index,
                                "candidate_id": cid,
                                "source_mode": source_mode,
                                "decision_stage": "quality_gate",
                                "decision": "skip",
                                "reason": str((drop_row or {}).get("reason", "quality_gate")),
                                "quality_bucket": str((drop_row or {}).get("bucket", "")),
                                "market_regime": market_regime_current,
                                "market_regime_reason": market_regime_reason,
                                "address": normalize_address((token_data or {}).get("address", "")),
                                "symbol": str((token_data or {}).get("symbol", (drop_row or {}).get("symbol", "N/A"))),
                                "score": int((score_data or {}).get("score", 0)),
                                "recommendation": str((score_data or {}).get("recommendation", "")),
                                "quality": _candidate_quality_features(token_data or {}, score_data or {}),
                            }
                        )
                candidates_symbols_cycle = [
                    str((token_data or {}).get("symbol", "") or "").strip().upper()
                    for token_data, _ in candidates_quality
                    if str((token_data or {}).get("symbol", "") or "").strip()
                ]
                dual_route_meta: dict[str, object] = {
                    "enabled": bool(getattr(config, "V2_ENTRY_DUAL_CHANNEL_ENABLED", False)),
                    "in_total": len(candidates_quality),
                    "out_total": len(candidates_quality),
                    "core_out": len(candidates_quality),
                    "explore_out": 0,
                    "regime": market_regime_current,
                }
                candidates_dual = list(candidates_quality)
                if candidates_quality:
                    candidates_dual, dual_route_meta = v2_dual_entry.allocate(
                        candidates=candidates_quality,
                        market_mode=market_regime_current,
                    )
                    if int(dual_route_meta.get("in_total", 0) or 0) != int(dual_route_meta.get("out_total", 0) or 0) or bool(
                        dual_route_meta.get("enabled", False)
                    ):
                        logger.info(
                            (
                                "V2_ENTRY_ROUTE enabled=%s regime=%s in=%s out=%s core_out=%s explore_out=%s "
                                "quota=%s allow_explore=%s"
                            ),
                            bool(dual_route_meta.get("enabled", False)),
                            str(dual_route_meta.get("regime", market_regime_current)),
                            int(dual_route_meta.get("in_total", 0) or 0),
                            int(dual_route_meta.get("out_total", 0) or 0),
                            int(dual_route_meta.get("core_out", 0) or 0),
                            int(dual_route_meta.get("explore_out", 0) or 0),
                            int(dual_route_meta.get("explore_quota", 0) or 0),
                            bool(dual_route_meta.get("allow_explore", False)),
                        )

                candidates_routed = list(candidates_dual)
                policy_route_meta: dict[str, object] = {
                    "enabled": bool(getattr(config, "V2_POLICY_ROUTER_ENABLED", False)),
                    "policy_state": policy_state,
                    "effective_mode": ("OK" if str(policy_state).upper() == "OK" else "BLOCKED"),
                    "action": ("allow_all" if str(policy_state).upper() == "OK" else "legacy_block"),
                    "in_total": len(candidates_dual),
                    "out_total": len(candidates_dual) if str(policy_state).upper() == "OK" else 0,
                    "reason": policy_reason,
                }
                if candidates_dual:
                    candidates_routed, policy_route_meta = v2_policy_router.route(
                        candidates=candidates_dual,
                        policy_state=policy_state,
                        policy_reason=policy_reason,
                        market_mode=market_regime_current,
                    )

                policy_effective_mode = str(policy_route_meta.get("effective_mode", policy_state) or policy_state).strip().upper()
                policy_effective_reason = (
                    f"{policy_reason} | route={str(policy_route_meta.get('action', 'n/a'))}"
                )
                auto_trader.set_data_policy(policy_effective_mode, policy_effective_reason)
                if policy_effective_mode == "BLOCKED":
                    logger.warning(
                        "AUTO_POLICY mode=%s action=block reason=%s candidates_in=%s",
                        policy_state,
                        policy_effective_reason,
                        len(candidates_dual),
                    )
                elif policy_effective_mode == "LIMITED":
                    logger.warning(
                        (
                            "AUTO_POLICY mode=%s action=limited reason=%s candidates_in=%s allowed=%s "
                            "strict_out=%s soft_out=%s"
                        ),
                        policy_state,
                        policy_effective_reason,
                        int(policy_route_meta.get("in_total", 0) or 0),
                        int(policy_route_meta.get("out_total", 0) or 0),
                        int(policy_route_meta.get("strict_out", 0) or 0),
                        int(policy_route_meta.get("soft_out", 0) or 0),
                    )
                logger.info(
                    "MARKET_REGIME mode=%s reason=%s avg_cand_recent=%.2f",
                    market_regime_current,
                    market_regime_reason,
                    avg_candidates_recent,
                )
                if market_regime_current != market_regime_prev:
                    try:
                        await local_alerter.send_event(
                            {
                                "event_type": "MARKET_MODE_CHANGE",
                                "symbol": "MARKET_MODE",
                                "recommendation": market_regime_current,
                                "risk_level": "INFO",
                                "name": f"{RUN_TAG} market mode {market_regime_prev}->{market_regime_current}",
                                "breakdown": {
                                    "previous_mode": market_regime_prev,
                                    "new_mode": market_regime_current,
                                    "reason": market_regime_reason,
                                    "avg_candidates_recent": round(avg_candidates_recent, 4),
                                },
                            }
                        )
                    except Exception:
                        logger.exception("MARKET_MODE_CHANGE event write failed")
                market_regime_prev = market_regime_current

                opened_trades = 0
                planned_by_source = _count_candidate_sources(candidates_routed)
                if candidates_routed:
                    opened_trades = await auto_trader.plan_batch(candidates_routed)

                routed_candidate_ids = {
                    str((token_data or {}).get("_candidate_id", "") or "")
                    for token_data, _ in candidates_routed
                }
                if candidates_dual and len(routed_candidate_ids) < len(candidates_dual):
                    policy_skip_reason = f"policy_{str(policy_effective_mode).lower()}"
                    for token_data, score_data in candidates_dual:
                        candidate_id = str(token_data.get("_candidate_id", "") or "")
                        if candidate_id in routed_candidate_ids:
                            continue
                        candidate_writer.write(
                            {
                                "ts": time.time(),
                                "cycle_index": cycle_index,
                                "candidate_id": candidate_id,
                                "source_mode": source_mode,
                                "decision_stage": "policy_gate",
                                "decision": "skip",
                                "reason": policy_skip_reason,
                                "policy_reason": str(policy_effective_reason),
                                "market_regime": market_regime_current,
                                "market_regime_reason": market_regime_reason,
                                "address": normalize_address(token_data.get("address", "")),
                                "symbol": str(token_data.get("symbol", "N/A")),
                                "score": int(score_data.get("score", 0)),
                                "recommendation": str(score_data.get("recommendation", "")),
                                "quality": _candidate_quality_features(token_data, score_data),
                            }
                        )
                await auto_trader.process_open_positions(bot=None)
                auto_stats = auto_trader.get_stats()
                skip_reasons_cycle = auto_trader.pop_skip_reason_counts_window()
                source_flow_cycle = auto_trader.pop_source_flow_window()
                candidate_count_cycle = len(candidates_quality)
                adaptive_filters.record_cycle(
                    candidates=candidate_count_cycle,
                    opened=opened_trades,
                    filter_fails_cycle=cycle_filter_fails,
                    skip_reasons_cycle=skip_reasons_cycle,
                    market_regime=market_regime_current,
                    dedup_repeat_intervals_cycle=dedup_repeat_intervals_cycle,
                )
                autonomy_controller.record_cycle(
                    candidates=candidate_count_cycle,
                    opened=opened_trades,
                    policy_state=policy_state,
                    market_regime=market_regime_current,
                    skip_reasons_cycle=skip_reasons_cycle,
                )
                strategy_orchestrator.record_cycle(
                    candidates=candidate_count_cycle,
                    opened=opened_trades,
                    market_regime=market_regime_current,
                )
                mini_analyzer.record(
                    scanned=int(tokens_seen_total),
                    candidates=candidate_count_cycle,
                    opened=opened_trades,
                    filter_fails_cycle=cycle_filter_fails,
                    skip_reasons_cycle=skip_reasons_cycle,
                    policy_state=policy_state,
                    market_regime=market_regime_current,
                )
                source_qos_event = v2_source_qos.record_cycle(
                    seen_by_source=seen_by_source,
                    passed_by_source=passed_by_source,
                    planned_by_source=planned_by_source,
                    source_flow=source_flow_cycle,
                )
                source_qos_snapshot = v2_source_qos.snapshot()
                v2_rolling_edge.record_cycle(
                    candidates=candidate_count_cycle,
                    opened=opened_trades,
                    skip_reasons_cycle=skip_reasons_cycle,
                )
                v2_kpi_loop.record_cycle(
                    candidates=candidate_count_cycle,
                    opened=opened_trades,
                    policy_state=policy_effective_mode,
                    symbols=candidates_symbols_cycle,
                    skip_reasons_cycle=skip_reasons_cycle,
                )
                if not runtime_tuner_lock_active:
                    adaptive_filters.maybe_adapt(
                        policy_state=policy_state,
                        auto_stats=auto_stats,
                        source_stats=source_stats,
                    )
                    autonomy_controller.maybe_adapt(
                        policy_state=policy_state,
                        market_regime=market_regime_current,
                        auto_stats=auto_stats,
                    )
                    strategy_orchestrator.maybe_adapt(
                        policy_state=policy_state,
                        market_regime=market_regime_current,
                        auto_stats=auto_stats,
                    )
                if isinstance(source_qos_event, dict):
                    try:
                        await local_alerter.send_event(
                            {
                                "event_type": "V2_SOURCE_QOS_APPLIED",
                                "symbol": "V2_SOURCE_QOS",
                                "recommendation": "INFO",
                                "risk_level": "INFO",
                                "name": f"{RUN_TAG} source qos updated",
                                "breakdown": source_qos_event,
                            }
                        )
                    except Exception:
                        logger.exception("V2_SOURCE_QOS event write failed")
                kpi_event = None if runtime_tuner_lock_active else v2_kpi_loop.maybe_apply(auto_trader=auto_trader)
                if isinstance(kpi_event, dict):
                    try:
                        await local_alerter.send_event(
                            {
                                "event_type": "V2_KPI_LOOP_APPLIED",
                                "symbol": "V2_KPI_LOOP",
                                "recommendation": str(kpi_event.get("action", "INFO")).upper(),
                                "risk_level": "INFO",
                                "name": f"{RUN_TAG} KPI loop adjusted",
                                "breakdown": kpi_event,
                            }
                        )
                    except Exception:
                        logger.exception("V2_KPI_LOOP event write failed")
                calibration_event = None if runtime_tuner_lock_active else v2_calibrator.maybe_apply()
                if isinstance(calibration_event, dict):
                    try:
                        await local_alerter.send_event(
                            {
                                "event_type": "V2_CALIBRATION_APPLIED",
                                "symbol": "V2_CALIBRATION",
                                "recommendation": "INFO",
                                "risk_level": "INFO",
                                "name": f"{RUN_TAG} calibration updated",
                                "breakdown": calibration_event,
                            }
                        )
                    except Exception:
                        logger.exception("V2_CALIBRATION event write failed")
                rolling_edge_event = (
                    None
                    if runtime_tuner_lock_active
                    else v2_rolling_edge.maybe_apply(auto_trader=auto_trader, auto_stats=auto_stats)
                )
                if isinstance(rolling_edge_event, dict):
                    try:
                        await local_alerter.send_event(
                            {
                                "event_type": "V2_ROLLING_EDGE_APPLIED",
                                "symbol": "V2_ROLLING_EDGE",
                                "recommendation": str(rolling_edge_event.get("action", "INFO")).upper(),
                                "risk_level": "INFO",
                                "name": f"{RUN_TAG} rolling edge updated",
                                "breakdown": rolling_edge_event,
                            }
                        )
                    except Exception:
                        logger.exception("V2_ROLLING_EDGE event write failed")
                stop_now, stop_event = profile_autostop.maybe_stop(auto_stats=auto_stats)
                if stop_now:
                    reason = ""
                    stop_tag = ""
                    if isinstance(stop_event, dict):
                        reason = str((stop_event.get("breakdown") or {}).get("reason", ""))
                        stop_tag = str((stop_event.get("breakdown") or {}).get("event_tag", ""))
                    logger.error(
                        "PROFILE_AUTOSTOP_TRIGGERED tag=%s run_tag=%s reason=%s",
                        stop_tag or "[AUTOSTOP][PROFILE_STOPPED]",
                        RUN_TAG,
                        reason,
                    )
                    if bool(profile_autostop.notify_enabled) and isinstance(stop_event, dict):
                        try:
                            await local_alerter.send_event(stop_event)
                        except Exception:
                            logger.exception("PROFILE_AUTOSTOP notification write failed")
                    break
                champ_stop, champ_event = v2_champion_guard.maybe_stop(auto_stats=auto_stats)
                if champ_stop:
                    reason = ""
                    if isinstance(champ_event, dict):
                        reason = str((champ_event.get("breakdown") or {}).get("reason", ""))
                    logger.error(
                        "V2_CHAMPION_GUARD_TRIGGERED run_tag=%s reason=%s",
                        RUN_TAG,
                        reason,
                    )
                    if isinstance(champ_event, dict):
                        try:
                            await local_alerter.send_event(champ_event)
                        except Exception:
                            logger.exception("V2_CHAMPION_GUARD event write failed")
                    break
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
                v2_budget_snapshot = v2_safety_budget.snapshot()

                logger.info(
                    (
                        "Scanned %s tokens | High quality: %s | Alerts sent: %s | Trade candidates: %s | "
                        "Quality out/core/explore/probe: %s/%s/%s/%s | Opened: %s | "
                        "Mode: local | Source: %s | Policy: raw=%s/effective=%s(%s) | Regime: %s(%s) | "
                        "Safety: checked=%s fail_closed=%s reasons=%s | FiltersTop(session): %s | Dedup: heavy_skip=%s override=%s | "
                        "V2Budget: %s/%s | SourceQoS: %s | Ingest: %s | Sources: %s | Tasks: %s | RSS: %.1fMB | CycleAvg: %.2fs"
                    ),
                    int(tokens_seen_total),
                    high_quality,
                    alerts_sent,
                    len(trade_candidates),
                    int(quality_gate_meta.get("out_total", 0) or 0),
                    int(quality_gate_meta.get("core_out", 0) or 0),
                    int(quality_gate_meta.get("explore_out", 0) or 0),
                    int(quality_gate_meta.get("probe_out", 0) or 0),
                    opened_trades,
                    source_mode,
                    policy_state,
                    policy_effective_mode,
                    policy_effective_reason,
                    market_regime_current,
                    market_regime_reason,
                    safety_checked,
                    safety_fail_closed_hits,
                    _format_safety_reasons_brief(safety_stats),
                    _format_top_filter_reasons(filter_fail_reasons_session),
                    heavy_dedup_skipped,
                    heavy_dedup_override,
                    int(v2_budget_snapshot.get("used_total", 0) or 0),
                    int(v2_budget_snapshot.get("max_total", 0) or 0),
                    _format_source_qos_brief(source_qos_meta, source_qos_snapshot),
                    _format_ingest_stats_brief(gecko_ingest_stats),
                    _format_source_stats_brief(source_stats),
                    active_tasks,
                    rss_mb,
                    (sum(cycle_times) / len(cycle_times)) if cycle_times else 0.0,
                )
            except Exception:
                _write_heartbeat(stage="cycle_error", cycle_index=cycle_index, open_trades=len(auto_trader.open_positions))
                logger.exception("Local monitoring loop error")
            finally:
                cycle_times.append(max(0.0, time.perf_counter() - cycle_started))
                _write_heartbeat(stage="cycle_end", cycle_index=cycle_index, open_trades=len(auto_trader.open_positions))

            sleep_seconds = (
                max(1, int(config.ONCHAIN_POLL_INTERVAL_SECONDS))
                if str(config.SIGNAL_SOURCE).lower() == "onchain"
                else int(SCAN_INTERVAL)
            )
            await asyncio.sleep(sleep_seconds)
    finally:
        _write_heartbeat(stage="shutdown", cycle_index=cycle_index, open_trades=len(auto_trader.open_positions))
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
    _write_single_pid_file()
    configure_logging()
    _clear_graceful_stop_flag()
    try:
        asyncio.run(run_local_loop())
    finally:
        _clear_single_pid_file()
        lock.release()


if __name__ == "__main__":
    main()
