from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402
from monitor.dexscreener import DexScreenerMonitor  # noqa: E402
from monitor.watchlist import WatchlistMonitor  # noqa: E402
from utils.addressing import normalize_address  # noqa: E402


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _norm_row(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    addr = normalize_address(row.get("address"))
    if not addr:
        return None
    return {
        "ts": time.time(),
        "source": source,
        "address": addr,
        "symbol": str(row.get("symbol") or "N/A"),
        "name": str(row.get("name") or ""),
        "liquidity_usd": _safe_float(row.get("liquidity")),
        "volume_5m_usd": _safe_float(row.get("volume_5m")),
        "age_seconds": _safe_int(row.get("age_seconds")),
        "price_usd": _safe_float(row.get("price_usd")),
        "price_change_5m": _safe_float(row.get("price_change_5m")),
        "risk_level": str(row.get("risk_level") or ""),
        "warning_flags": _safe_int(row.get("warning_flags")),
        "is_contract_safe": bool(row.get("is_contract_safe", False)),
    }


async def run_capture(*, cycles: int, interval_seconds: int, out_path: str) -> dict[str, Any]:
    dex = DexScreenerMonitor()
    watch = WatchlistMonitor()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    total_rows = 0
    total_unique = set()
    source_counts: Counter[str] = Counter()

    try:
        with open(out_path, "a", encoding="utf-8") as f:
            for cycle in range(1, cycles + 1):
                merged: dict[str, dict[str, Any]] = {}

                dex_rows = await dex.fetch_new_tokens()
                for row in dex_rows or []:
                    item = _norm_row(row, source="dex_new")
                    if not item:
                        continue
                    merged[item["address"]] = item

                # Use private refresh intentionally: for dataset capture we want fresh snapshot every cycle.
                watch_rows = await watch._refresh()  # type: ignore[attr-defined]
                for row in watch_rows or []:
                    item = _norm_row(row, source="watchlist")
                    if not item:
                        continue
                    prev = merged.get(item["address"])
                    if prev is None or item["liquidity_usd"] > prev["liquidity_usd"]:
                        merged[item["address"]] = item

                for item in merged.values():
                    item["cycle"] = cycle
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_rows += 1
                    total_unique.add(str(item["address"]))
                    source_counts[str(item["source"])] += 1

                await asyncio.sleep(max(1, int(interval_seconds)))
    finally:
        await dex.close()
        await watch.close()

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "cycles": cycles,
        "interval_seconds": interval_seconds,
        "rows_written": total_rows,
        "unique_addresses": len(total_unique),
        "source_counts": dict(source_counts),
        "output_file": out_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture open-source market dataset into local JSONL.")
    parser.add_argument("--cycles", type=int, default=30, help="Number of fetch cycles")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between cycles")
    parser.add_argument(
        "--out",
        default=os.path.join("data", "external", "open_market_dataset.jsonl"),
        help="Output JSONL file",
    )
    args = parser.parse_args()

    out_path = args.out if os.path.isabs(args.out) else os.path.abspath(os.path.join(PROJECT_ROOT, args.out))
    summary = asyncio.run(
        run_capture(
            cycles=max(1, int(args.cycles)),
            interval_seconds=max(1, int(args.interval)),
            out_path=out_path,
        )
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
