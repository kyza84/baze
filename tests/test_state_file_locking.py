from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
import unittest

from utils.state_file import StateFileLockError, state_file_lock, write_json_atomic_locked


def _hold_lock_worker(path: str, ready: multiprocessing.Event, release: multiprocessing.Event) -> None:
    with state_file_lock(path, timeout_seconds=2.0, poll_seconds=0.01):
        ready.set()
        release.wait(2.0)


class StateFileLockingTests(unittest.TestCase):
    def test_atomic_locked_write_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = os.path.join(tmp_dir, "state.json")
            payload = {"paper_balance_usd": 7.0, "realized_pnl_usd": 0.0, "open_positions": []}
            write_json_atomic_locked(
                state_path,
                payload,
                timeout_seconds=0.5,
                poll_seconds=0.01,
                encoding="utf-8",
                ensure_ascii=False,
                indent=2,
            )
            with open(state_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, payload)

    def test_state_lock_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = os.path.join(tmp_dir, "state.json")
            ctx = multiprocessing.get_context("spawn")
            ready = ctx.Event()
            release = ctx.Event()
            proc = ctx.Process(target=_hold_lock_worker, args=(state_path, ready, release))
            proc.start()
            try:
                self.assertTrue(ready.wait(2.0), "worker did not acquire state lock in time")
                with self.assertRaises(StateFileLockError):
                    with state_file_lock(state_path, timeout_seconds=0.08, poll_seconds=0.01):
                        pass
            finally:
                release.set()
                proc.join(2.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(1.0)
            self.assertEqual(proc.exitcode, 0)
