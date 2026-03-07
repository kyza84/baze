from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow direct `python tools/matrix_watchdog.py ...` execution to import project modules.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.state_file import atomic_write_json, state_file_lock


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with state_file_lock(str(path), timeout_seconds=2.5, poll_seconds=0.05):
        atomic_write_json(
            str(path),
            payload,
            encoding="utf-8",
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )


def _try_write_json(path: Path, payload: dict[str, Any]) -> bool:
    try:
        _write_json(path, payload)
        return True
    except Exception as exc:
        try:
            print(f"[WATCHDOG] active_matrix_merge_failed err={exc}")
        except Exception:
            pass
        return False


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def _pid_cmdline_windows(pid: int) -> str:
    if int(pid) <= 0 or os.name != "nt":
        return ""
    try:
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        cmd = (
            f"$p=Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\" -ErrorAction SilentlyContinue; "
            'if($p){ [string]$p.CommandLine }'
        )
        probe = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=creationflags,
        )
        return str(probe.stdout or "").strip()
    except Exception:
        return ""


def _pid_alive(pid: int, *, require_substring: str | None = None) -> bool:
    if pid <= 0:
        return False
    needle = str(require_substring or "").strip().lower()
    if os.name == "nt":
        try:
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            probe = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=creationflags,
            )
            line = str(probe.stdout or "").strip()
            if not line:
                return False
            # tasklist returns a localized INFO line (non-CSV) when PID is missing.
            if not line.startswith('"'):
                return False
            row = next(csv.reader([line]), [])
            if len(row) < 2:
                return False
            pid_cell = str(row[1] or "").replace(",", "").strip()
            if str(int(pid)) != pid_cell:
                return False
            if needle:
                cmdline = _pid_cmdline_windows(int(pid)).lower()
                return bool(cmdline and needle in cmdline)
            return True
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        alive = True
    except PermissionError:
        alive = True
    except OSError:
        alive = False
    except Exception:
        alive = False
    if not alive:
        return False
    if needle:
        try:
            proc_cmdline = Path(f"/proc/{int(pid)}/cmdline").read_text(encoding="utf-8", errors="ignore").lower()
            return needle in proc_cmdline
        except Exception:
            return False
    return True


def _kill_pid(pid: int) -> None:
    if pid <= 0:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=20)
            return
        os.kill(pid, signal.SIGKILL)
    except Exception:
        return


def _python_path(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _restart_item(root: Path, item: dict[str, Any]) -> tuple[bool, str, int]:
    py = _python_path(root)
    if not py.exists():
        return False, f"python not found: {py}", 0
    env_file = str(item.get("env_file", "") or "").strip()
    if not env_file:
        return False, "missing env_file", 0
    try:
        env = os.environ.copy()
        env["BOT_ENV_FILE"] = env_file
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            [str(py), "main_local.py"],
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0,
            close_fds=True,
        )
        pid = int(proc.pid or 0)
        if pid <= 0:
            return False, "spawn failed", 0
        return True, "restarted", pid
    except Exception as exc:
        return False, str(exc), 0


def _heartbeat_age_seconds(root: Path, item: dict[str, Any]) -> float | None:
    rel_log_dir = str(item.get("log_dir", "") or "").strip()
    if not rel_log_dir:
        return None
    hb_path = root / rel_log_dir / "heartbeat.json"
    hb = _read_json(hb_path)
    ts = hb.get("ts")
    try:
        hb_ts = float(ts)
    except Exception:
        return None
    return max(0.0, time.time() - hb_ts)


def _runtime_activity_age_seconds(root: Path, item: dict[str, Any]) -> float | None:
    rel_log_dir = str(item.get("log_dir", "") or "").strip()
    if not rel_log_dir:
        return None
    log_dir = root / rel_log_dir
    mtimes: list[float] = []
    for path in (
        log_dir / "heartbeat.json",
        log_dir / "app.log",
        log_dir / "out.log",
    ):
        try:
            if path.exists():
                mtimes.append(float(path.stat().st_mtime))
        except Exception:
            continue
    if not mtimes:
        return None
    return max(0.0, float(time.time()) - max(mtimes))


def _file_age_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        mtime = float(path.stat().st_mtime)
    except Exception:
        return None
    return max(0.0, time.time() - mtime)


def _runtime_tuner_lock_pid(root: Path, item: dict[str, Any]) -> int:
    rel_log_dir = str(item.get("log_dir", "") or "").strip()
    if not rel_log_dir:
        return 0
    lock_path = root / rel_log_dir / "runtime_tuner.lock.json"
    payload = _read_json(lock_path)
    try:
        return int(payload.get("pid", 0) or 0)
    except Exception:
        return 0


def _runtime_tuner_log_dir(root: Path, profile_id: str, rel_log_dir: str = "") -> Path:
    rel = str(rel_log_dir or "").strip()
    if rel:
        return root / rel
    return root / "logs" / "matrix" / str(profile_id or "").strip()


def _runtime_tuner_lock_path(root: Path, profile_id: str, rel_log_dir: str = "") -> Path:
    return _runtime_tuner_log_dir(root, profile_id, rel_log_dir) / "runtime_tuner.lock.json"


def _runtime_tuner_jsonl_path(root: Path, profile_id: str, rel_log_dir: str = "") -> Path:
    return _runtime_tuner_log_dir(root, profile_id, rel_log_dir) / "runtime_tuner.jsonl"


def _runtime_tuner_log_age_seconds(root: Path, item: dict[str, Any]) -> float | None:
    rel_log_dir = str(item.get("log_dir", "") or "").strip()
    if not rel_log_dir:
        return None
    log_path = root / rel_log_dir / "runtime_tuner.jsonl"
    return _file_age_seconds(log_path)


def _runtime_tuner_known(root: Path, item: dict[str, Any]) -> bool:
    rel_log_dir = str(item.get("log_dir", "") or "").strip()
    if not rel_log_dir:
        return False
    log_dir = root / rel_log_dir
    return bool((log_dir / "runtime_tuner.jsonl").exists() or (log_dir / "runtime_tuner.lock.json").exists())


def _restart_tuner(root: Path, profile_id: str, rel_log_dir: str = "") -> tuple[bool, str, int]:
    py = _python_path(root)
    if not py.exists():
        return False, f"python not found: {py}", 0
    script = root / "tools" / "matrix_runtime_tuner.py"
    if not script.exists():
        return False, f"script not found: {script}", 0
    log_dir_rel = str(rel_log_dir or "").strip() or f"logs/matrix/{profile_id}"

    def _verify_tuner_started(spawn_pid: int) -> tuple[bool, str]:
        if int(spawn_pid) <= 0:
            return False, "spawn_pid_invalid"
        lock_path = _runtime_tuner_lock_path(root, profile_id, log_dir_rel)
        log_path = _runtime_tuner_jsonl_path(root, profile_id, log_dir_rel)
        # Guard against false-positive spawn: process must stay alive and acquire profile lock.
        deadline = float(time.time()) + 18.0
        while float(time.time()) < deadline:
            alive = _pid_alive(int(spawn_pid), require_substring="matrix_runtime_tuner.py")
            if not alive:
                return False, "spawned_not_alive"
            lock_payload = _read_json(lock_path)
            try:
                lock_pid = int(lock_payload.get("pid", 0) or 0)
            except Exception:
                lock_pid = 0
            if lock_pid == int(spawn_pid):
                log_age = _file_age_seconds(log_path)
                detail = "lock_acquired"
                if log_age is not None:
                    detail = f"{detail} log_age={round(float(log_age), 2)}"
                return True, detail
            time.sleep(0.6)
        return False, "lock_not_acquired"

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            [
                str(py),
                "-u",
                str(script),
                "run",
                "--profile-id",
                str(profile_id),
                "--root",
                str(root),
                "--mode",
                "conveyor",
                "--policy-phase",
                "auto",
                "--window-minutes",
                "12",
                "--duration-minutes",
                "600",
                "--interval-seconds",
                "120",
                "--lock-acquire-timeout-seconds",
                "45",
                "--lock-acquire-poll-seconds",
                "1.0",
            ],
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0,
            close_fds=True,
        )
        pid = int(proc.pid or 0)
        if pid <= 0:
            return False, "spawn failed", 0
        ok, verify_msg = _verify_tuner_started(pid)
        if not ok:
            return False, verify_msg, pid
        return True, f"restarted {verify_msg}", pid
    except Exception as exc:
        return False, str(exc), 0


def _meta_path(root: Path) -> Path:
    return root / "data" / "matrix" / "runs" / "active_matrix.json"


def _state_path(root: Path) -> Path:
    return root / "data" / "matrix" / "runs" / "watchdog_state.json"


def _events_path(root: Path) -> Path:
    return root / "data" / "matrix" / "runs" / "watchdog_events.jsonl"


def _lock_path(root: Path) -> Path:
    return root / "data" / "matrix" / "runs" / "watchdog.lock.json"


def _merge_runtime_meta_update(root: Path, items_snapshot: list[dict[str, Any]], alive_count: int) -> bool:
    path = _meta_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    updates: dict[str, dict[str, Any]] = {}
    for row in items_snapshot:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id", "") or "").strip()
        if not rid:
            continue
        updates[rid] = {
            "pid": row.get("pid", None),
            "status": str(row.get("status", "") or "").strip() or "unknown",
        }
    if not updates:
        return False
    try:
        with state_file_lock(str(path), timeout_seconds=2.5, poll_seconds=0.05):
            payload: dict[str, Any] = {}
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        payload = raw
                except Exception:
                    payload = {}
            items = payload.get("items") if isinstance(payload.get("items"), list) else []
            changed = False
            for row in items:
                if not isinstance(row, dict):
                    continue
                rid = str(row.get("id", "") or "").strip()
                if not rid or rid not in updates:
                    continue
                next_pid = updates[rid].get("pid", None)
                next_status = updates[rid].get("status", "unknown")
                if row.get("pid", None) != next_pid:
                    row["pid"] = next_pid
                    changed = True
                if str(row.get("status", "") or "") != str(next_status):
                    row["status"] = str(next_status)
                    changed = True
            running_next = bool(int(alive_count) > 0)
            if bool(payload.get("running", False)) != running_next:
                payload["running"] = running_next
                changed = True
            if int(payload.get("alive_count", -1) or -1) != int(alive_count):
                payload["alive_count"] = int(alive_count)
                changed = True
            if not changed:
                return False
            payload["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            payload["items"] = items
            atomic_write_json(
                str(path),
                payload,
                encoding="utf-8",
                ensure_ascii=False,
                indent=2,
                sort_keys=False,
            )
            return True
    except Exception:
        return False


def _acquire_follow_lock(root: Path) -> tuple[bool, str]:
    path = _lock_path(root)
    me = int(os.getpid())
    prev = _read_json(path)
    prev_pid = 0
    try:
        prev_pid = int(prev.get("pid", 0) or 0)
    except Exception:
        prev_pid = 0
    if prev_pid > 0 and prev_pid != me and _pid_alive(prev_pid, require_substring="matrix_watchdog.py"):
        return False, f"watchdog_lock_busy pid={prev_pid}"
    payload = {
        "pid": me,
        "owner": "matrix_watchdog.py",
        "started_at": _now_iso(),
        "root": str(root),
    }
    if not _try_write_json(path, payload):
        return False, f"watchdog_lock_write_failed path={path}"
    return True, f"watchdog_lock_acquired pid={me}"


def _release_follow_lock(root: Path) -> None:
    path = _lock_path(root)
    me = int(os.getpid())
    prev = _read_json(path)
    prev_pid = 0
    try:
        prev_pid = int(prev.get("pid", 0) or 0)
    except Exception:
        prev_pid = 0
    if prev_pid > 0 and prev_pid != me:
        return
    try:
        if path.exists():
            path.unlink()
    except Exception:
        _try_write_json(
            path,
            {
                "pid": 0,
                "owner": "matrix_watchdog.py",
                "released_at": _now_iso(),
            },
        )


def _scan_once(
    root: Path,
    stale_seconds: int,
    restart_cooldown_seconds: int,
    *,
    watch_tuner: bool = False,
    tuner_stale_seconds: int = 420,
    tuner_restart_cooldown_seconds: int = 300,
) -> dict[str, Any]:
    meta_path = _meta_path(root)
    state_path = _state_path(root)
    meta = _read_json(meta_path)
    items = meta.get("items") if isinstance(meta.get("items"), list) else []
    requested_run = bool(meta.get("requested_run", False))
    state = _read_json(state_path)
    last_restarts = state.get("last_restarts", {}) if isinstance(state.get("last_restarts"), dict) else {}
    last_tuner_restarts = (
        state.get("last_tuner_restarts", {}) if isinstance(state.get("last_tuner_restarts"), dict) else {}
    )
    events: list[dict[str, Any]] = []
    changed = False
    now_iso = _now_iso()

    for row in items:
        if not isinstance(row, dict):
            continue
        profile_id = str(row.get("id", "") or "").strip() or "unknown"
        pid = int(row.get("pid", 0) or 0)
        alive = _pid_alive(pid, require_substring="main_local.py")
        hb_age = _heartbeat_age_seconds(root, row)
        activity_age = _runtime_activity_age_seconds(root, row)
        stale = (
            alive
            and (hb_age is not None)
            and (hb_age >= float(stale_seconds))
            and (activity_age is None or activity_age >= float(stale_seconds))
        )
        dead = not alive
        now_ts = time.time()
        last_ts = float(last_restarts.get(profile_id, 0.0) or 0.0)
        cooldown_left = float(restart_cooldown_seconds) - (now_ts - last_ts)

        restart_reason = ""
        if requested_run and stale:
            restart_reason = "stale_heartbeat"
        elif requested_run and dead:
            restart_reason = "pid_dead_or_missing"

        if restart_reason and cooldown_left <= 0.0:
            if alive:
                _kill_pid(pid)
            ok, msg, new_pid = _restart_item(root, row)
            evt = {
                "ts": now_ts,
                "timestamp": now_iso,
                "id": profile_id,
                "reason": restart_reason,
                "heartbeat_age_sec": round(float(hb_age or 0.0), 2),
                "activity_age_sec": round(float(activity_age or 0.0), 2),
                "old_pid": pid,
                "restart_ok": bool(ok),
                "message": msg,
                "new_pid": int(new_pid),
            }
            events.append(evt)
            _append_jsonl(_events_path(root), evt)
            last_restarts[profile_id] = now_ts
            changed = True
            if ok:
                row["pid"] = int(new_pid)
                row["status"] = "running"
            else:
                row["status"] = "restart_failed"

        if watch_tuner and requested_run and _runtime_tuner_known(root, row):
            tuner_pid = _runtime_tuner_lock_pid(root, row)
            tuner_alive = _pid_alive(tuner_pid, require_substring="matrix_runtime_tuner.py") if tuner_pid > 0 else False
            tuner_age = _runtime_tuner_log_age_seconds(root, row)
            tuner_stale = (tuner_age is None) or (float(tuner_age) >= float(tuner_stale_seconds))
            tuner_dead = (tuner_pid > 0) and (not tuner_alive)
            tuner_reason = ""
            if tuner_dead:
                tuner_reason = "tuner_pid_dead"
            elif tuner_stale:
                tuner_reason = "tuner_stale_jsonl"
            if tuner_reason:
                tuner_last_ts = float(last_tuner_restarts.get(profile_id, 0.0) or 0.0)
                tuner_cooldown_left = float(tuner_restart_cooldown_seconds) - (now_ts - tuner_last_ts)
                if tuner_cooldown_left <= 0.0:
                    if tuner_pid > 0 and tuner_alive:
                        _kill_pid(tuner_pid)
                    ok, msg, new_pid = _restart_tuner(root, profile_id, str(row.get("log_dir", "") or ""))
                    evt = {
                        "ts": now_ts,
                        "timestamp": now_iso,
                        "component": "runtime_tuner",
                        "id": profile_id,
                        "reason": tuner_reason,
                        "tuner_log_age_sec": round(float(tuner_age or 0.0), 2),
                        "old_pid": int(tuner_pid),
                        "restart_ok": bool(ok),
                        "message": msg,
                        "new_pid": int(new_pid),
                    }
                    events.append(evt)
                    _append_jsonl(_events_path(root), evt)
                    last_tuner_restarts[profile_id] = now_ts
                    changed = True

    alive_count = 0
    for row in items:
        if isinstance(row, dict):
            pid = int(row.get("pid", 0) or 0)
            if _pid_alive(pid, require_substring="main_local.py"):
                alive_count += 1
    meta_running = bool(alive_count > 0)
    meta_alive_count_prev = int(meta.get("alive_count", -1) or -1)
    meta_running_prev = bool(meta.get("running", False))
    meta_needs_update = bool(changed) or (meta_running_prev != meta_running) or (meta_alive_count_prev != int(alive_count))
    if meta_needs_update:
        _merge_runtime_meta_update(root, items, alive_count)

    prev_events_tail = state.get("events_tail", []) if isinstance(state.get("events_tail"), list) else []
    state_payload = {
        "updated_at": now_iso,
        "last_scan_ts": float(time.time()),
        "last_restarts": last_restarts,
        "last_tuner_restarts": last_tuner_restarts,
        "events_tail": (prev_events_tail + events)[-80:],
    }
    _try_write_json(state_path, state_payload)

    return {
        "checked_items": len(items),
        "events": events,
        "changed": bool(changed or meta_needs_update),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Matrix watchdog: restart stale profile workers.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--stale-seconds", type=int, default=360)
    parser.add_argument("--restart-cooldown-seconds", type=int, default=120)
    parser.add_argument("--watch-tuner", action="store_true")
    parser.add_argument("--tuner-stale-seconds", type=int, default=420)
    parser.add_argument("--tuner-restart-cooldown-seconds", type=int, default=300)
    parser.add_argument("--loop-seconds", type=int, default=20)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.follow:
        locked, lock_msg = _acquire_follow_lock(root)
        if not locked:
            print(f"[WATCHDOG] {lock_msg}")
            return 0
        try:
            while True:
                res = _scan_once(
                    root=root,
                    stale_seconds=max(60, int(args.stale_seconds)),
                    restart_cooldown_seconds=max(30, int(args.restart_cooldown_seconds)),
                    watch_tuner=bool(args.watch_tuner),
                    tuner_stale_seconds=max(120, int(args.tuner_stale_seconds)),
                    tuner_restart_cooldown_seconds=max(60, int(args.tuner_restart_cooldown_seconds)),
                )
                for evt in res.get("events", []) or []:
                    print(
                        f"[WATCHDOG] component={evt.get('component', 'main_local')} id={evt.get('id')} "
                        f"reason={evt.get('reason')} old_pid={evt.get('old_pid')} "
                        f"new_pid={evt.get('new_pid')} ok={evt.get('restart_ok')} "
                        f"age={evt.get('heartbeat_age_sec', evt.get('tuner_log_age_sec', 'N/A'))}s"
                    )
                time.sleep(max(10, int(args.loop_seconds)))
        finally:
            _release_follow_lock(root)
    else:
        res = _scan_once(
            root=root,
            stale_seconds=max(60, int(args.stale_seconds)),
            restart_cooldown_seconds=max(30, int(args.restart_cooldown_seconds)),
            watch_tuner=bool(args.watch_tuner),
            tuner_stale_seconds=max(120, int(args.tuner_stale_seconds)),
            tuner_restart_cooldown_seconds=max(60, int(args.tuner_restart_cooldown_seconds)),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
