from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import matrix_watchdog as mwd  # noqa: E402


class _FakeProc:
    def __init__(self, pid: int) -> None:
        self.pid = int(pid)


class MatrixWatchdogTests(unittest.TestCase):
    def test_restart_tuner_fails_when_spawn_exits_before_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            py = root / ".venv" / "Scripts" / "python.exe"
            script = root / "tools" / "matrix_runtime_tuner.py"
            py.parent.mkdir(parents=True, exist_ok=True)
            script.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("", encoding="utf-8")
            script.write_text("print('ok')\n", encoding="utf-8")

            with (
                mock.patch.object(mwd.subprocess, "Popen", return_value=_FakeProc(11111)),
                mock.patch.object(
                    mwd,
                    "_pid_alive",
                    side_effect=lambda pid, require_substring=None: False,
                ),
            ):
                ok, msg, pid = mwd._restart_tuner(root, "u_test", "logs/matrix/u_test")

        self.assertFalse(ok)
        self.assertEqual(int(pid), 11111)
        self.assertIn("spawned_not_alive", msg)

    def test_restart_tuner_requires_lock_acquire_before_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            py = root / ".venv" / "Scripts" / "python.exe"
            script = root / "tools" / "matrix_runtime_tuner.py"
            log_dir = root / "logs" / "matrix" / "u_test"
            lock_path = log_dir / "runtime_tuner.lock.json"
            log_path = log_dir / "runtime_tuner.jsonl"
            py.parent.mkdir(parents=True, exist_ok=True)
            script.parent.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            py.write_text("", encoding="utf-8")
            script.write_text("print('ok')\n", encoding="utf-8")
            lock_path.write_text(json.dumps({"pid": 22222}), encoding="utf-8")
            log_path.write_text("", encoding="utf-8")

            with (
                mock.patch.object(mwd.subprocess, "Popen", return_value=_FakeProc(22222)),
                mock.patch.object(
                    mwd,
                    "_pid_alive",
                    side_effect=lambda pid, require_substring=None: int(pid) == 22222,
                ),
            ):
                ok, msg, pid = mwd._restart_tuner(root, "u_test", "logs/matrix/u_test")

        self.assertTrue(ok)
        self.assertEqual(int(pid), 22222)
        self.assertIn("lock_acquired", msg)


if __name__ == "__main__":
    unittest.main()
