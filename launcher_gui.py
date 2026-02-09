"""Small desktop launcher for bot process control (Windows)."""

import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_PATH = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
PID_FILE = os.path.join(PROJECT_ROOT, "bot.pid")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUT_LOG = os.path.join(LOG_DIR, "bot.out.log")
ERR_LOG = os.path.join(LOG_DIR, "bot.err.log")
APP_LOG = os.path.join(LOG_DIR, "app.log")


def read_pid() -> int | None:
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r", encoding="ascii") as f:
            return int(f.read().strip())
    except Exception:
        return None


def is_running(pid: int) -> bool:
    if not pid:
        return False
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def start_bot() -> tuple[bool, str]:
    if not os.path.exists(PYTHON_PATH):
        return False, f"Python not found: {PYTHON_PATH}"

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
        [PYTHON_PATH, "main.py"],
        cwd=PROJECT_ROOT,
        stdout=out_file,
        stderr=err_file,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )

    with open(PID_FILE, "w", encoding="ascii") as f:
        f.write(str(proc.pid))

    return True, f"Started (PID {proc.pid})"


def stop_bot() -> tuple[bool, str]:
    pid = read_pid()
    if not pid:
        return True, "Already stopped"

    if not is_running(pid):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass
        return True, f"Stopped (stale PID {pid} removed)"

    result = subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
    )
    try:
        os.remove(PID_FILE)
    except OSError:
        pass

    if result.returncode == 0:
        return True, f"Stopped (PID {pid})"
    return False, result.stderr.strip() or "Failed to stop process"


def tail_logs(max_lines: int = 80) -> str:
    chunks = []
    for label, path in (("APP", APP_LOG), ("OUT", OUT_LOG), ("ERR", ERR_LOG)):
        chunks.append(f"===== {label} LOG: {path} =====")
        if not os.path.exists(path):
            chunks.append("<no file>")
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-max_lines:]
        chunks.extend([line.rstrip("\n") for line in lines] or ["<empty>"])
    return "\n".join(chunks)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Solana Alert Bot Control")
        self.geometry("900x620")

        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Status: unknown")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT)

        ttk.Button(top, text="Start", command=self.on_start).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="Stop", command=self.on_stop).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="Refresh", command=self.refresh).pack(side=tk.RIGHT, padx=5)

        self.log_text = tk.Text(self, wrap="none")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.after(200, self.refresh)
        self.after(2000, self.auto_refresh)

    def refresh(self) -> None:
        pid = read_pid()
        if pid and is_running(pid):
            self.status_var.set(f"Status: RUNNING (PID {pid})")
        else:
            self.status_var.set("Status: STOPPED")

        logs = tail_logs()
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, logs)

    def auto_refresh(self) -> None:
        self.refresh()
        self.after(2000, self.auto_refresh)

    def on_start(self) -> None:
        ok, msg = start_bot()
        if not ok:
            messagebox.showerror("Start error", msg)
        self.refresh()

    def on_stop(self) -> None:
        ok, msg = stop_bot()
        if not ok:
            messagebox.showerror("Stop error", msg)
        self.refresh()


if __name__ == "__main__":
    App().mainloop()
