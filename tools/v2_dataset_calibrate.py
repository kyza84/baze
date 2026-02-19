"""Generate and persist V2 runtime calibration from unified dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trading.v2_runtime import UnifiedCalibrator


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V2 calibration snapshot from unified dataset.")
    parser.add_argument("--apply", action="store_true", help="Apply overrides to in-memory config during this run.")
    args = parser.parse_args()

    cal = UnifiedCalibrator()
    cal.enabled = True
    payload = cal._build_payload()
    if payload is None:
        print(json.dumps({"ok": False, "reason": "not_enough_data_or_db_missing"}, ensure_ascii=False))
        return 1
    cal._write_payload(payload)
    if args.apply:
        cal._apply_config_overrides(payload.get("applied", {}))
    print(json.dumps({"ok": True, **payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
