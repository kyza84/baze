"""Shared helpers for safe state-file locking and atomic JSON writes."""

from __future__ import annotations

import json
import os
import errno
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Iterator

try:  # pragma: no cover - platform specific
    import msvcrt
except Exception:  # pragma: no cover - platform specific
    msvcrt = None  # type: ignore[assignment]

try:  # pragma: no cover - platform specific
    import fcntl
except Exception:  # pragma: no cover - platform specific
    fcntl = None  # type: ignore[assignment]

E_STATE_LOCKED = "E_STATE_LOCKED"
E_JSON_CORRUPT = "E_JSON_CORRUPT"
E_STATE_IO = "E_STATE_IO"

_TRANSIENT_REPLACE_WINERRORS = {5, 32, 33}
_TRANSIENT_REPLACE_ERRNOS = {errno.EACCES, errno.EBUSY, errno.EPERM}


class StateFileLockError(RuntimeError):
    """Raised when the state-file lock cannot be acquired in time."""

    code = E_STATE_LOCKED


def _ensure_lock_byte(handle: Any) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    handle.seek(0)


def _try_lock(handle: Any) -> None:
    if os.name == "nt" and msvcrt is not None:  # pragma: no cover - windows-only runtime path
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            raise BlockingIOError(str(exc)) from exc
    if fcntl is not None:  # pragma: no cover - unix-only runtime path
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as exc:
            raise BlockingIOError(str(exc)) from exc
    # No lock primitive available: fallback is no-op.
    return


def _unlock(handle: Any) -> None:
    if os.name == "nt" and msvcrt is not None:  # pragma: no cover - windows-only runtime path
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    if fcntl is not None:  # pragma: no cover - unix-only runtime path
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def state_file_lock(
    target_path: str,
    *,
    timeout_seconds: float = 2.0,
    poll_seconds: float = 0.05,
) -> Iterator[None]:
    """Acquire an inter-process lock for a state file using `<state>.lock`."""

    lock_path = f"{str(target_path)}.lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    timeout = max(0.05, float(timeout_seconds))
    poll = max(0.01, float(poll_seconds))
    deadline = time.monotonic() + timeout

    handle = open(lock_path, "a+b")
    locked = False
    try:
        _ensure_lock_byte(handle)
        while True:
            try:
                _try_lock(handle)
                locked = True
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise StateFileLockError(
                        f"{E_STATE_LOCKED}: state lock timeout path={target_path}"
                    ) from exc
                time.sleep(poll)
        yield
    finally:
        if locked:
            try:
                _unlock(handle)
            except OSError:
                pass
        try:
            handle.close()
        except OSError:
            pass


def atomic_write_json(
    path: str,
    payload: Any,
    *,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """Write JSON atomically via temp file + replace in the same directory."""

    abs_path = str(path)
    state_dir = os.path.dirname(abs_path) or "."
    os.makedirs(state_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{os.path.basename(abs_path)}.",
        suffix=".tmp",
        dir=state_dir,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            json.dump(payload, f, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        replace_retries = max(0, int(os.getenv("STATE_ATOMIC_REPLACE_RETRIES", "8") or 8))
        replace_base_delay = max(0.01, float(os.getenv("STATE_ATOMIC_REPLACE_BASE_DELAY_SECONDS", "0.03") or 0.03))
        for attempt in range(replace_retries + 1):
            try:
                os.replace(tmp_path, abs_path)
                break
            except OSError as exc:
                winerror = int(getattr(exc, "winerror", 0) or 0)
                err_no = int(getattr(exc, "errno", 0) or 0)
                transient = (winerror in _TRANSIENT_REPLACE_WINERRORS) or (err_no in _TRANSIENT_REPLACE_ERRNOS)
                if (not transient) or attempt >= replace_retries:
                    raise
                time.sleep(replace_base_delay * (1.5**attempt))
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def write_json_atomic_locked(
    path: str,
    payload: Any,
    *,
    timeout_seconds: float = 2.0,
    poll_seconds: float = 0.05,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """Acquire file lock and write JSON atomically."""

    with state_file_lock(path, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds):
        atomic_write_json(
            path,
            payload,
            encoding=encoding,
            ensure_ascii=ensure_ascii,
            indent=indent,
            sort_keys=sort_keys,
        )


def read_json_locked(
    path: str,
    *,
    timeout_seconds: float = 2.0,
    poll_seconds: float = 0.05,
    encoding: str = "utf-8-sig",
) -> Any:
    """Acquire file lock and read JSON payload."""

    with state_file_lock(path, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds):
        with open(path, "r", encoding=encoding) as f:
            return json.load(f)
