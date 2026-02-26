from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import matrix_promote_live as mpl  # noqa: E402


class MatrixPromoteLiveTests(unittest.TestCase):
    def test_build_canary_overrides_caps_runtime_limits(self) -> None:
        src = {
            "MAX_OPEN_TRADES": "5",
            "MAX_BUYS_PER_HOUR": "42",
            "AUTO_TRADE_TOP_N": "20",
            "PAPER_TRADE_SIZE_MAX_USD": "1.50",
            "PAPER_TRADE_SIZE_MIN_USD": "0.80",
        }
        out = mpl._build_canary_overrides(
            target_map=src,
            max_open_trades=1,
            max_buys_per_hour=8,
            top_n=8,
            size_max_usd=0.45,
            ttl_minutes=60,
        )
        self.assertEqual(out["MAX_OPEN_TRADES"], "1")
        self.assertEqual(out["MAX_BUYS_PER_HOUR"], "8")
        self.assertEqual(out["AUTO_TRADE_TOP_N"], "8")
        self.assertEqual(out["PAPER_TRADE_SIZE_MAX_USD"], "0.45")
        self.assertEqual(out["PAPER_TRADE_SIZE_MIN_USD"], "0.45")
        self.assertEqual(out["LIVE_CANARY_ENABLED"], "true")
        self.assertIn("LIVE_CANARY_UNTIL_TS", out)

    def test_write_switch_snapshot_and_pick_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = mpl._write_switch_snapshot(
                root=str(root),
                profile_id="p1",
                source_env="src.env",
                target_env="dst.env",
                canary=False,
                previous_dotenv={"BOT_ENV_FILE": "old.env"},
                next_dotenv={"BOT_ENV_FILE": "new.env"},
            )
            second = mpl._write_switch_snapshot(
                root=str(root),
                profile_id="p2",
                source_env="src2.env",
                target_env="dst2.env",
                canary=True,
                previous_dotenv={"BOT_ENV_FILE": "old2.env"},
                next_dotenv={"BOT_ENV_FILE": "new2.env"},
            )
            latest = mpl._latest_switch_snapshot(str(root))
            self.assertTrue(latest in {first, second})
            payload = json.loads(Path(latest).read_text(encoding="utf-8"))
            self.assertIn("previous_env", payload)
            self.assertIn("next_env", payload)

    def test_profile_open_positions_from_active_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_file = root / "trading" / "paper_state.mx_test.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(
                json.dumps({"open_positions": [{"id": 1}, {"id": 2}], "closed_positions": []}),
                encoding="utf-8",
            )
            active_path = root / "data" / "matrix" / "runs" / "active_matrix.json"
            active_path.parent.mkdir(parents=True, exist_ok=True)
            active_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "mx_test",
                                "paper_state_file": str(state_file),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            open_count = mpl._profile_open_positions(str(root), "mx_test")
            self.assertEqual(open_count, 2)


if __name__ == "__main__":
    unittest.main()
