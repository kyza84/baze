from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            txt = (proc.stdout or "").lower()
            return proc.returncode == 0 and "python.exe" in txt and str(pid) in txt
        os.kill(pid, 0)
        return True
    except Exception:
        return False


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


def _meta_path(root: Path) -> Path:
    return root / "data" / "matrix" / "runs" / "active_matrix.json"


def _state_path(root: Path) -> Path:
    return root / "data" / "matrix" / "runs" / "watchdog_state.json"


def _events_path(root: Path) -> Path:
    return root / "data" / "matrix" / "runs" / "watchdog_events.jsonl"


def _scan_once(root: Path, stale_seconds: int, restart_cooldown_seconds: int) -> dict[str, Any]:
    meta_path = _meta_path(root)
    state_path = _state_path(root)
    meta = _read_json(meta_path)
    items = meta.get("items") if isinstance(meta.get("items"), list) else []
    state = _read_json(state_path)
    last_restarts = state.get("last_restarts", {}) if isinstance(state.get("last_restarts"), dict) else {}
    events: list[dict[str, Any]] = []
    changed = False

    for row in items:
        if not isinstance(row, dict):
            continue
        profile_id = str(row.get("id", "") or "").strip() or "unknown"
        pid = int(row.get("pid", 0) or 0)
        alive = _pid_alive(pid)
        hb_age = _heartbeat_age_seconds(root, row)
        stale = alive and (hb_age is not None) and (hb_age >= float(stale_seconds))
        now_ts = time.time()
        last_ts = float(last_restarts.get(profile_id, 0.0) or 0.0)
        cooldown_left = float(restart_cooldown_seconds) - (now_ts - last_ts)

        if stale and cooldown_left <= 0.0:
            _kill_pid(pid)
            ok, msg, new_pid = _restart_item(root, row)
            evt = {
                "ts": now_ts,
                "timestamp": _now_iso(),
                "id": profile_id,
                "reason": "stale_heartbeat",
                "heartbeat_age_sec": round(float(hb_age or 0.0), 2),
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

    if changed:
        alive_count = 0
        for row in items:
            if isinstance(row, dict):
                pid = int(row.get("pid", 0) or 0)
                if _pid_alive(pid):
                    alive_count += 1
        meta["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        meta["running"] = bool(alive_count > 0)
        meta["alive_count"] = int(alive_count)
        meta["items"] = items
        _write_json(meta_path, meta)
        _write_json(
            state_path,
            {
                "updated_at": _now_iso(),
                "last_restarts": last_restarts,
                "events_tail": (state.get("events_tail", []) if isinstance(state.get("events_tail"), list) else [])[-80:] + events,
            },
        )

    return {
        "checked_items": len(items),
        "events": events,
        "changed": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Matrix watchdog: restart stale profile workers.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--stale-seconds", type=int, default=180)
    parser.add_argument("--restart-cooldown-seconds", type=int, default=120)
    parser.add_argument("--loop-seconds", type=int, default=20)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.follow:
        while True:
            res = _scan_once(
                root=root,
                stale_seconds=max(60, int(args.stale_seconds)),
                restart_cooldown_seconds=max(30, int(args.restart_cooldown_seconds)),
            )
            for evt in res.get("events", []) or []:
                print(
                    f"[WATCHDOG] id={evt.get('id')} reason={evt.get('reason')} old_pid={evt.get('old_pid')} "
                    f"new_pid={evt.get('new_pid')} ok={evt.get('restart_ok')} age={evt.get('heartbeat_age_sec')}s"
                )
            time.sleep(max(10, int(args.loop_seconds)))
    else:
        res = _scan_once(
            root=root,
            stale_seconds=max(60, int(args.stale_seconds)),
            restart_cooldown_seconds=max(30, int(args.restart_cooldown_seconds)),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
