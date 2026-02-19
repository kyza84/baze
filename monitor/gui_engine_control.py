"""Process and script control helpers for launcher GUI."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EngineContext:
    project_root: str
    python_path: str
    pid_file: str
    log_dir: str
    out_log: str
    err_log: str


def run_powershell_script(
    context: EngineContext,
    script_path: str,
    args: list[str] | None = None,
    *,
    timeout_seconds: int = 90,
) -> tuple[bool, str]:
    if not os.path.exists(script_path):
        return False, f"Script not found: {script_path}"
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path]
    if args:
        cmd.extend(args)
    try:
        proc = subprocess.run(
            cmd,
            cwd=context.project_root,
            capture_output=True,
            text=True,
            timeout=max(10, int(timeout_seconds)),
        )
    except Exception as exc:
        return False, str(exc)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    msg = out or err or f"exit={proc.returncode}"
    if proc.returncode != 0:
        return False, msg
    return True, msg


def start_bot(
    context: EngineContext,
    *,
    matrix_alive_count: Callable[[], int],
    list_main_local_pids: Callable[[], list[int]],
    is_main_local_process: Callable[[int], bool],
    read_pid: Callable[[], int | None],
    is_running: Callable[[int | None], bool],
    clear_graceful_stop_flag: Callable[[], None],
) -> tuple[bool, str]:
    if not os.path.exists(context.python_path):
        return False, f"Python not found: {context.python_path}"
    if matrix_alive_count() > 0:
        return False, "Matrix instances are running. Stop Matrix mode before starting single mode."
    clear_graceful_stop_flag()

    existing = sorted(set(list_main_local_pids()))
    if existing:
        keep_pid = existing[0]
        for pid in existing[1:]:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        with open(context.pid_file, "w", encoding="ascii") as f:
            f.write(str(keep_pid))
        if len(existing) > 1:
            return True, f"Already running main_local.py (PID {keep_pid}), duplicates removed: {len(existing) - 1}"
        return True, f"Already running main_local.py (PID {keep_pid})"

    pid = read_pid()
    if pid and is_main_local_process(pid):
        return True, f"Already running main_local.py (PID {pid})"
    if pid and is_running(pid):
        try:
            os.remove(context.pid_file)
        except OSError:
            pass

    os.makedirs(context.log_dir, exist_ok=True)
    out_file = open(context.out_log, "a", encoding="utf-8")
    err_file = open(context.err_log, "a", encoding="utf-8")

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        [context.python_path, "main_local.py"],
        cwd=context.project_root,
        stdout=out_file,
        stderr=err_file,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )
    with open(context.pid_file, "w", encoding="ascii") as f:
        f.write(str(proc.pid))
    return True, f"Started main_local.py (PID {proc.pid})"


def stop_bot(
    context: EngineContext,
    *,
    list_main_local_pids: Callable[[], list[int]],
    is_main_local_process: Callable[[int], bool],
    read_pid: Callable[[], int | None],
    signal_graceful_stop: Callable[[], tuple[bool, str]],
    resolve_graceful_timeout: Callable[[], int],
    clear_graceful_stop_flag: Callable[[], None],
) -> tuple[bool, str]:
    pids = sorted(set(list_main_local_pids()))
    if not pids:
        pid = read_pid()
        if pid and is_main_local_process(pid):
            pids = [pid]
        else:
            try:
                os.remove(context.pid_file)
            except OSError:
                pass
            return True, "Already stopped"

    graceful_ok, graceful_msg = signal_graceful_stop()
    timeout_sec = resolve_graceful_timeout()
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        alive = [pid for pid in pids if is_main_local_process(pid)]
        if not alive:
            try:
                os.remove(context.pid_file)
            except OSError:
                pass
            clear_graceful_stop_flag()
            return True, f"Graceful stop completed in <= {timeout_sec}s."
        time.sleep(0.25)

    failed: list[int] = []
    hard_stopped: list[int] = []
    for pid in pids:
        if not is_main_local_process(pid):
            continue
        result = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        if result.returncode != 0 and is_main_local_process(pid):
            failed.append(pid)
        else:
            hard_stopped.append(pid)
    try:
        os.remove(context.pid_file)
    except OSError:
        pass
    if failed:
        return False, f"Graceful stop timeout={timeout_sec}s; hard-kill failed for PID(s): {', '.join(str(pid) for pid in failed)}"
    clear_graceful_stop_flag()
    if graceful_ok:
        return True, f"Graceful stop timeout={timeout_sec}s; hard-kill fallback for PID(s): {', '.join(str(pid) for pid in hard_stopped)}"
    return True, f"Graceful signal failed ({graceful_msg}); hard-kill fallback for PID(s): {', '.join(str(pid) for pid in hard_stopped)}"
