from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter
from datetime import datetime


def _bucket_volume(v: float) -> str:
    if v < 100:
        return "<100"
    if v < 500:
        return "100-499"
    if v < 1_000:
        return "500-999"
    if v < 5_000:
        return "1k-4.9k"
    if v < 20_000:
        return "5k-19k"
    return "20k+"


def _bucket_liquidity(v: float) -> str:
    if v < 5_000:
        return "<5k"
    if v < 20_000:
        return "5k-19k"
    if v < 100_000:
        return "20k-99k"
    if v < 500_000:
        return "100k-499k"
    return "500k+"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze captured open market dataset JSONL.")
    parser.add_argument(
        "--file",
        default=os.path.join("data", "external", "open_market_dataset.jsonl"),
        help="Input JSONL",
    )
    args = parser.parse_args()
    path = args.file
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        print(json.dumps({"error": "file_not_found", "file": path}, ensure_ascii=False))
        return 1

    rows = 0
    unique_addr = set()
    src = Counter()
    vol_bucket = Counter()
    liq_bucket = Counter()
    vol_vals = []
    liq_vals = []
    age_vals = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            rows += 1
            addr = str(row.get("address") or "").lower()
            if addr:
                unique_addr.add(addr)
            source = str(row.get("source") or "unknown")
            src[source] += 1
            vol = float(row.get("volume_5m_usd") or 0.0)
            liq = float(row.get("liquidity_usd") or 0.0)
            age = float(row.get("age_seconds") or 0.0)
            vol_vals.append(vol)
            liq_vals.append(liq)
            age_vals.append(age)
            vol_bucket[_bucket_volume(vol)] += 1
            liq_bucket[_bucket_liquidity(liq)] += 1

    def _med(vals: list[float]) -> float:
        return round(statistics.median(vals), 3) if vals else 0.0

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "file": path,
        "rows": rows,
        "unique_addresses": len(unique_addr),
        "source_counts": dict(src),
        "median_volume_5m_usd": _med(vol_vals),
        "median_liquidity_usd": _med(liq_vals),
        "median_age_seconds": _med(age_vals),
        "volume_buckets": dict(vol_bucket),
        "liquidity_buckets": dict(liq_bucket),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
