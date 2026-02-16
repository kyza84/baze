"""Summarize autonomous control decisions for active matrix profiles."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_env(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                key = k.strip()
                if not key:
                    continue
                out[key] = v.strip()
    except Exception:
        return {}
    return out


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return rows
    return rows


def _resolve_items(root: str) -> list[dict[str, str]]:
    active_path = os.path.join(root, "data", "matrix", "runs", "active_matrix.json")
    active = _read_json(active_path)
    items = active.get("items")
    out: list[dict[str, str]] = []
    if not isinstance(items, list):
        return out
    for row in items:
        if not isinstance(row, dict):
            continue
        profile_id = str(row.get("id", "") or "").strip()
        env_file = str(row.get("env_file", "") or "").strip()
        if not profile_id:
            continue
        if env_file and not os.path.isabs(env_file):
            env_file = os.path.join(root, env_file)
        out.append({"id": profile_id, "env_file": env_file})
    return out


def _summarize_profile(root: str, profile_id: str, env_file: str, tail: int) -> dict[str, Any]:
    env_map = _read_env(env_file) if env_file else {}
    log_file = str(env_map.get("AUTONOMOUS_CONTROL_DECISIONS_LOG_FILE", "") or "").strip()
    if not log_file:
        log_file = os.path.join("logs", "matrix", profile_id, "autonomy_decisions.jsonl")
    if not os.path.isabs(log_file):
        log_file = os.path.join(root, log_file)
    rows = _read_jsonl(log_file)
    last_rows = rows[-max(1, int(tail)) :]
    latest = last_rows[-1] if last_rows else {}

    return {
        "profile_id": profile_id,
        "env_file": env_file,
        "decisions_log_file": log_file,
        "decisions_total": len(rows),
        "latest_action": str(latest.get("action", "") or ""),
        "latest_changed": bool(latest.get("changed", False)),
        "latest_reason": str(latest.get("reason", "") or ""),
        "latest_market_regime_top": str(latest.get("market_regime_top", "") or ""),
        "latest_policy_top": str(latest.get("policy_top", "") or ""),
        "latest_avg_candidates": float(latest.get("avg_candidates", 0.0) or 0.0),
        "latest_avg_opened": float(latest.get("avg_opened", 0.0) or 0.0),
        "latest_realized_delta_usd": float(latest.get("realized_delta_usd", 0.0) or 0.0),
        "latest_controls_after": dict(latest.get("controls_after") or {}),
        "latest_ts": latest.get("ts"),
        "tail": last_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize matrix autonomous-control decisions.")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--tail", type=int, default=3, help="How many latest decisions per profile to include")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    items = _resolve_items(root)
    if not items:
        print("No active matrix items found.")
        return 1

    profiles: list[dict[str, Any]] = []
    for row in items:
        profile_id = str(row.get("id", "")).strip()
        env_file = str(row.get("env_file", "")).strip()
        profiles.append(_summarize_profile(root, profile_id, env_file, int(args.tail)))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles,
    }
    report_dir = os.path.join(root, "data", "matrix", "reports")
    os.makedirs(report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"matrix_autonomy_summary_{stamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"AUTONOMY_REPORT {report_path}")
    for row in profiles:
        print(
            f"{row['profile_id']:<20} decisions={row['decisions_total']:>3} "
            f"action={row['latest_action'] or '-':<18} changed={str(row['latest_changed']).lower():<5} "
            f"avg_cand={row['latest_avg_candidates']:.2f} avg_opened={row['latest_avg_opened']:.2f} "
            f"delta=${row['latest_realized_delta_usd']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

