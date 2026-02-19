"""Promote a matrix profile into live with strict config parity."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

LIVE_APPLY_CONFIRM_PHRASE = "CONFIRM_LIVE_SWITCH"
_ALLOWED_PROMOTION_DIFF_KEYS = {
    "BOT_INSTANCE_ID",
    "RUN_TAG",
    "WALLET_MODE",
    "AUTO_TRADE_ENABLED",
    "AUTO_TRADE_PAPER",
    "PAPER_RESET_ON_START",
    "CANDIDATE_SHARD_MOD",
    "CANDIDATE_SHARD_SLOT",
    "CANDIDATE_DECISIONS_LOG_FILE",
    "AUTONOMOUS_CONTROL_DECISIONS_LOG_FILE",
    "PAPER_STATE_FILE",
}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except Exception:
            return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except Exception:
        return out
    return out


def _load_env(path: str) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    data: dict[str, str] = {}
    if not os.path.exists(path):
        return order, data
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, value = s.split("=", 1)
            key = key.strip()
            if not key:
                continue
            if key not in data:
                order.append(key)
            data[key] = value.strip()
    return order, data


def _write_env(path: str, order: list[str], data: dict[str, str]) -> None:
    seen = set(order)
    all_keys = list(order) + [k for k in data.keys() if k not in seen]
    lines = [f"{k}={data[k]}" for k in all_keys if k in data]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _load_active_items(root: str) -> list[dict[str, str]]:
    active_meta_path = os.path.join(root, "data", "matrix", "runs", "active_matrix.json")
    active = _read_json(active_meta_path)
    items = active.get("items")
    out: list[dict[str, str]] = []
    if not isinstance(items, list):
        return out
    for row in items:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id", "") or "").strip()
        env_file = str(row.get("env_file", "") or "").strip()
        state_file = str(row.get("paper_state_file", "") or "").strip()
        if not rid:
            continue
        if env_file and not os.path.isabs(env_file):
            env_file = os.path.join(root, env_file)
        if state_file and not os.path.isabs(state_file):
            state_file = os.path.join(root, state_file)
        out.append({"id": rid, "env_file": env_file, "state_file": state_file})
    return out


def _window_rank_winner(root: str, *, lookback_hours: float, min_closed: int) -> tuple[str, dict[str, Any]]:
    items = _load_active_items(root)
    if not items:
        raise RuntimeError("No active matrix items found for window ranking.")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(0.25, float(lookback_hours)))
    ranked: list[dict[str, Any]] = []
    for row in items:
        rid = str(row.get("id", "")).strip()
        state_file = str(row.get("state_file", "")).strip()
        payload = _read_json(state_file) if state_file else {}
        closed = payload.get("closed_positions") or []
        open_rows = payload.get("open_positions") or []
        if not isinstance(closed, list):
            closed = []
        if not isinstance(open_rows, list):
            open_rows = []
        closed = [x for x in closed if isinstance(x, dict)]
        open_rows = [x for x in open_rows if isinstance(x, dict)]

        closed_window = [x for x in closed if (_parse_ts(x.get("closed_at")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
        entries_window = sum(
            1
            for x in (open_rows + closed)
            if (_parse_ts((x or {}).get("opened_at")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff
        )

        wins = sum(1 for x in closed_window if _safe_float(x.get("pnl_usd"), 0.0) > 0.0)
        losses = sum(1 for x in closed_window if _safe_float(x.get("pnl_usd"), 0.0) < 0.0)
        loss_share = (float(losses) / float(max(1, wins + losses))) if (wins + losses) > 0 else 0.0
        winrate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0
        pnl = float(sum(_safe_float(x.get("pnl_usd"), 0.0) for x in closed_window))
        gains = float(sum(_safe_float(x.get("pnl_usd"), 0.0) for x in closed_window if _safe_float(x.get("pnl_usd"), 0.0) > 0))
        losses_abs = abs(float(sum(_safe_float(x.get("pnl_usd"), 0.0) for x in closed_window if _safe_float(x.get("pnl_usd"), 0.0) < 0)))
        pf = (gains / losses_abs) if losses_abs > 0 else (float("inf") if gains > 0 else 0.0)
        worst = min((_safe_float(x.get("pnl_usd"), 0.0) for x in closed_window), default=0.0)

        score = 0.0
        score += pnl * 160.0
        score += (winrate - 50.0) * 0.8
        if pf == float("inf"):
            score += 12.0
        else:
            score += max(-16.0, min(16.0, (pf - 1.0) * 10.0))
        score += min(14.0, float(entries_window + len(closed_window)) * 0.35)
        score += min(14.0, len(closed_window) * 0.55)
        score -= max(0.0, (abs(worst) - 0.08) * 110.0)
        score -= max(0.0, (loss_share - 0.56) * 40.0)
        if len(closed_window) < int(min_closed):
            score -= float(int(min_closed) - len(closed_window)) * 2.5

        ranked.append(
            {
                "profile_id": rid,
                "state_file": state_file,
                "entries_window": entries_window,
                "closed_window": len(closed_window),
                "wins_window": wins,
                "losses_window": losses,
                "loss_share_window": loss_share,
                "pnl_window_usd": pnl,
                "winrate_window_pct": winrate,
                "profit_factor_window": pf,
                "worst_trade_window_usd": worst,
                "score_total": score,
            }
        )

    if not ranked:
        raise RuntimeError("Window ranking produced no candidates.")

    ranked.sort(
        key=lambda r: (
            float(r.get("score_total", 0.0)),
            float(r.get("pnl_window_usd", 0.0)),
            float(r.get("winrate_window_pct", 0.0)),
            int(r.get("closed_window", 0)),
        ),
        reverse=True,
    )
    winner_id = str(ranked[0]["profile_id"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_hours": float(lookback_hours),
        "cutoff_utc": cutoff.isoformat(),
        "min_closed": int(min_closed),
        "winner_id": winner_id,
        "profiles_ranked": ranked,
    }
    report_dir = os.path.join(root, "data", "matrix", "reports")
    os.makedirs(report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"matrix_live_window_pick_{stamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return winner_id, {"report_path": report_path, "report": report}


def _resolve_profile_env(
    root: str,
    profile_id: str | None,
    *,
    lookback_hours: float,
    min_closed: int,
) -> tuple[str, str, dict[str, Any]]:
    active_items = _load_active_items(root)
    by_id = {str(x.get("id", "")): x for x in active_items}
    meta: dict[str, Any] = {"selection_mode": "", "window_pick_report": ""}

    if profile_id:
        hit = by_id.get(profile_id)
        if hit:
            env_file = str(hit.get("env_file", "") or "").strip()
            if env_file:
                meta["selection_mode"] = "manual_active"
                return profile_id, env_file, meta
        env_guess = os.path.join(root, "data", "matrix", "env", f"{profile_id}.env")
        meta["selection_mode"] = "manual_fallback"
        return profile_id, env_guess, meta

    # Prefer explicit recent window ranking if active matrix is available.
    if active_items:
        winner_id, details = _window_rank_winner(root, lookback_hours=lookback_hours, min_closed=min_closed)
        hit = by_id.get(winner_id, {})
        env_file = str(hit.get("env_file", "") or "").strip() or os.path.join(root, "data", "matrix", "env", f"{winner_id}.env")
        meta["selection_mode"] = "window_rank"
        meta["window_pick_report"] = str(details.get("report_path", ""))
        return winner_id, env_file, meta

    # Fallback to latest regular rank report.
    reports_dir = os.path.join(root, "data", "matrix", "reports")
    latest_rank = ""
    if os.path.isdir(reports_dir):
        candidates = [
            os.path.join(reports_dir, x)
            for x in os.listdir(reports_dir)
            if x.startswith("matrix_rank_") and x.endswith(".json")
        ]
        if candidates:
            latest_rank = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)[0]
    if latest_rank:
        payload = _read_json(latest_rank)
        winner_id = str(payload.get("winner_id", "") or "").strip()
        if winner_id:
            env_guess = os.path.join(root, "data", "matrix", "env", f"{winner_id}.env")
            meta["selection_mode"] = "latest_rank_report"
            return winner_id, env_guess, meta

    raise RuntimeError("Cannot resolve source profile/env. Run matrix first or pass --profile-id.")


def _build_live_overrides(_source: dict[str, str], profile_id: str) -> dict[str, str]:
    # Keep strategy mechanics untouched; only switch runtime/execution context to live.
    return {
        "BOT_INSTANCE_ID": f"live_{profile_id}",
        "RUN_TAG": f"live_{profile_id}",
        "WALLET_MODE": "live",
        "AUTO_TRADE_ENABLED": "true",
        "AUTO_TRADE_PAPER": "false",
        "PAPER_RESET_ON_START": "false",
        "CANDIDATE_SHARD_MOD": "1",
        "CANDIDATE_SHARD_SLOT": "0",
        "CANDIDATE_DECISIONS_LOG_FILE": f"logs/live/{profile_id}/candidates.jsonl",
        "AUTONOMOUS_CONTROL_DECISIONS_LOG_FILE": f"logs/live/{profile_id}/autonomy_decisions.jsonl",
        "PAPER_STATE_FILE": f"trading/paper_state.live_{profile_id}.json",
    }


def _window_controls(
    root: str,
    *,
    profile_id: str,
    source_env: dict[str, str],
    lookback_hours: float,
    mode: str,
) -> dict[str, str]:
    mode = str(mode or "median").strip().lower()
    if mode not in {"off", "last", "median"}:
        mode = "median"
    if mode == "off":
        return {}

    path = str(source_env.get("AUTONOMOUS_CONTROL_DECISIONS_LOG_FILE", "") or "").strip()
    if not path:
        path = os.path.join("logs", "matrix", profile_id, "autonomy_decisions.jsonl")
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    rows = _load_jsonl(path)
    if not rows:
        return {}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(0.25, float(lookback_hours)))
    valid: list[dict[str, Any]] = []
    for row in rows:
        ts = _parse_ts(row.get("ts"))
        if ts is not None and ts < cutoff:
            continue
        controls = row.get("controls_after")
        if isinstance(controls, dict):
            valid.append(controls)
    if not valid:
        return {}

    chosen: dict[str, Any] = {}
    keys = ("MAX_OPEN_TRADES", "AUTO_TRADE_TOP_N", "MAX_BUYS_PER_HOUR", "PAPER_TRADE_SIZE_MAX_USD")
    if mode == "last":
        src = valid[-1]
        for key in keys:
            if key in src:
                chosen[key] = src[key]
    else:
        # Median-ish robust pick from recent runtime controls.
        for key in keys:
            values: list[float] = []
            for row in valid:
                if key not in row:
                    continue
                values.append(_safe_float(row.get(key), 0.0))
            if not values:
                continue
            values.sort()
            chosen[key] = values[len(values) // 2]

    out: dict[str, str] = {}
    if "MAX_OPEN_TRADES" in chosen:
        out["MAX_OPEN_TRADES"] = str(max(1, _safe_int(chosen["MAX_OPEN_TRADES"], 1)))
    if "AUTO_TRADE_TOP_N" in chosen:
        out["AUTO_TRADE_TOP_N"] = str(max(1, _safe_int(chosen["AUTO_TRADE_TOP_N"], 1)))
    if "MAX_BUYS_PER_HOUR" in chosen:
        out["MAX_BUYS_PER_HOUR"] = str(max(1, _safe_int(chosen["MAX_BUYS_PER_HOUR"], 1)))
    if "PAPER_TRADE_SIZE_MAX_USD" in chosen:
        size_max = max(0.05, _safe_float(chosen["PAPER_TRADE_SIZE_MAX_USD"], 0.60))
        out["PAPER_TRADE_SIZE_MAX_USD"] = f"{size_max:.2f}"
        size_min = max(0.05, min(size_max, _safe_float(source_env.get("PAPER_TRADE_SIZE_MIN_USD"), 0.45)))
        out["PAPER_TRADE_SIZE_MIN_USD"] = f"{size_min:.2f}"
    return out


def _diff_env(before: dict[str, str], after: dict[str, str]) -> list[tuple[str, str, str]]:
    keys = sorted(set(before.keys()) | set(after.keys()))
    out: list[tuple[str, str, str]] = []
    for key in keys:
        prev = str(before.get(key, ""))
        cur = str(after.get(key, ""))
        if prev != cur:
            out.append((key, prev, cur))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote matrix profile into live env with strict parity.")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--profile-id", default="", help="Profile id to promote (optional)")
    parser.add_argument("--lookback-hours", type=float, default=4.0, help="Window for auto winner pick and runtime controls")
    parser.add_argument("--min-closed", type=int, default=10, help="Min closed trades for window winner confidence")
    parser.add_argument(
        "--window-controls",
        default="off",
        choices=["off", "last", "median"],
        help="Optional fold of recent autonomy controls (default: off for strict parity)",
    )
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="Allow non-runtime parameter drift from source matrix profile (not recommended).",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help=f"Required with --apply. Must equal: {LIVE_APPLY_CONFIRM_PHRASE}",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply BOT_ENV_FILE + mode flags into .env for next bot start",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    requested_profile = args.profile_id.strip()
    if args.apply and not requested_profile:
        print("LIVE_APPLY_BLOCKED profile id is required when using --apply.")
        print("Example: --profile-id mx3_flow_compound --apply --confirm CONFIRM_LIVE_SWITCH")
        return 2
    if args.apply and args.confirm.strip() != LIVE_APPLY_CONFIRM_PHRASE:
        print("LIVE_APPLY_BLOCKED invalid --confirm phrase.")
        print("Re-run with: --apply --confirm CONFIRM_LIVE_SWITCH")
        return 2

    profile_id, source_env_path, select_meta = _resolve_profile_env(
        root,
        requested_profile or None,
        lookback_hours=float(args.lookback_hours),
        min_closed=max(1, int(args.min_closed)),
    )
    if not os.path.exists(source_env_path):
        raise RuntimeError(f"Source env not found: {source_env_path}")

    order, source_map = _load_env(source_env_path)
    if not source_map:
        raise RuntimeError(f"Source env is empty/unreadable: {source_env_path}")

    target_map = dict(source_map)
    target_map.update(_build_live_overrides(source_map, profile_id))
    runtime_controls = _window_controls(
        root,
        profile_id=profile_id,
        source_env=source_map,
        lookback_hours=float(args.lookback_hours),
        mode=str(args.window_controls),
    )
    if runtime_controls:
        target_map.update(runtime_controls)

    diffs = _diff_env(source_map, target_map)
    unexpected = [d for d in diffs if d[0] not in _ALLOWED_PROMOTION_DIFF_KEYS]
    if unexpected and not args.allow_drift:
        print("PROMOTION_BLOCKED unexpected parameter drift detected (strict parity).")
        for key, prev, cur in unexpected[:40]:
            print(f"DRIFT {key}: {prev} -> {cur}")
        print("Re-run with --allow-drift to bypass, or disable drift sources (e.g. --window-controls off).")
        return 3

    target_env = os.path.join(root, "data", "matrix", "env", f"live_from_{profile_id}.env")
    _write_env(target_env, order, target_map)

    print(f"SOURCE_PROFILE {profile_id}")
    print(f"SOURCE_ENV {source_env_path}")
    print(f"SELECTION_MODE {select_meta.get('selection_mode', '')}")
    if str(select_meta.get("window_pick_report", "")).strip():
        print(f"WINDOW_PICK_REPORT {select_meta['window_pick_report']}")
    print(f"WINDOW_CONTROLS_MODE {args.window_controls}")
    print(f"PROMOTION_DIFFS {len(diffs)}")
    if diffs:
        for key, prev, cur in diffs:
            print(f"DIFF {key}: {prev} -> {cur}")
    if unexpected:
        print(f"UNEXPECTED_DIFFS {len(unexpected)}")
    print(f"LIVE_ENV {target_env}")

    if args.apply:
        dot_env = os.path.join(root, ".env")
        env_order, env_map = _load_env(dot_env)
        rel_target = os.path.relpath(target_env, root).replace("\\", "/")
        env_map["BOT_ENV_FILE"] = rel_target
        env_map["WALLET_MODE"] = "live"
        env_map["AUTO_TRADE_ENABLED"] = "true"
        env_map["AUTO_TRADE_PAPER"] = "false"
        env_map["PAPER_RESET_ON_START"] = "false"
        _write_env(dot_env, env_order, env_map)
        print(f"APPLIED_TO_DOTENV {dot_env}")
        print(f"BOT_ENV_FILE={rel_target}")
    else:
        print("DRY_RUN only. Use --apply to update .env.")

    print(f"STAMP {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
