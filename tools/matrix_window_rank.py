"""Rank active matrix paper profiles using a recent fixed window (e.g. last 4h)."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
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
                key, value = s.split("=", 1)
                k = key.strip()
                if not k:
                    continue
                out[k] = value.strip()
    except Exception:
        return {}
    return out


def _resolve_candidate_log(
    *,
    root: str,
    profile_id: str,
    item: dict[str, Any] | None = None,
) -> str:
    row = item or {}
    raw = str(row.get("candidate_log_file", "") or "").strip()
    if raw:
        return raw if os.path.isabs(raw) else os.path.join(root, raw)
    env_file = str(row.get("env_file", "") or "").strip()
    if env_file and os.path.exists(env_file):
        env_map = _read_env(env_file)
        cand_env = str(env_map.get("CANDIDATE_DECISIONS_LOG_FILE", "") or "").strip()
        if cand_env:
            return cand_env if os.path.isabs(cand_env) else os.path.join(root, cand_env)
    log_dir = str(row.get("log_dir", "") or "").strip()
    if log_dir:
        base = log_dir if os.path.isabs(log_dir) else os.path.join(root, log_dir)
        return os.path.join(base, "candidates.jsonl")
    return os.path.join(root, "logs", "matrix", profile_id, "candidates.jsonl")


def _window_blocked_counts(candidate_log_file: str, *, cutoff: datetime) -> dict[str, int]:
    out = {
        "blocked_by_safe_source": 0,
        "blocked_by_watchlist_guard": 0,
        "blocked_by_safety_budget": 0,
    }
    if not os.path.exists(candidate_log_file):
        return out
    try:
        with open(candidate_log_file, "r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                stage = str(row.get("decision_stage", "") or "").strip().lower()
                if stage != "filter_fail":
                    continue
                ts = _parse_ts(row.get("timestamp"))
                if ts is None:
                    ts = _parse_ts(row.get("ts"))
                if ts is None or ts < cutoff:
                    continue
                reason = str(row.get("reason", "") or "").strip().lower()
                if reason == "safe_source":
                    out["blocked_by_safe_source"] += 1
                elif reason == "watchlist_strict_guard":
                    out["blocked_by_watchlist_guard"] += 1
                elif reason == "safety_budget":
                    out["blocked_by_safety_budget"] += 1
    except Exception:
        return out
    return out


def _load_items(root: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    active_path = os.path.join(root, "data", "matrix", "runs", "active_matrix.json")
    active = _read_json(active_path)
    items = active.get("items")
    if isinstance(items, list):
        for row in items:
            if not isinstance(row, dict):
                continue
            profile_id = str(row.get("id", "") or "").strip()
            state_file = str(row.get("paper_state_file", "") or "").strip()
            if not profile_id or not state_file:
                continue
            if not os.path.isabs(state_file):
                state_file = os.path.join(root, state_file)
            env_file = str(row.get("env_file", "") or "").strip()
            if env_file and not os.path.isabs(env_file):
                env_file = os.path.join(root, env_file)
            out.append(
                {
                    "id": profile_id,
                    "state_file": state_file,
                    "env_file": env_file,
                    "log_dir": str(row.get("log_dir", "") or ""),
                }
            )
    if out:
        return out

    trading_dir = os.path.join(root, "trading")
    if not os.path.isdir(trading_dir):
        return []
    for name in sorted(os.listdir(trading_dir)):
        if not name.startswith("paper_state.mx") or not name.endswith(".json"):
            continue
        profile_id = name.replace("paper_state.", "").replace(".json", "")
        out.append(
            {
                "id": profile_id,
                "state_file": os.path.join(trading_dir, name),
                "env_file": "",
                "log_dir": os.path.join("logs", "matrix", profile_id),
            }
        )
    return out


@dataclass
class ProfileWindowStats:
    profile_id: str
    state_file: str
    candidate_log_file: str
    open_now: int
    entries_window: int
    exits_window: int
    closed_window: int
    wins_window: int
    losses_window: int
    loss_share_window: float
    breakeven_window: int
    pnl_window_usd: float
    pnl_window_clipped_usd: float
    winrate_window_pct: float
    profit_factor_window: float
    worst_trade_window_usd: float
    best_trade_window_usd: float
    top_symbol_share_window: float
    blocked_by_safe_source_window: int
    blocked_by_watchlist_guard_window: int
    blocked_by_safety_budget_window: int
    score_total: float
    score_breakdown: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        pf = self.profit_factor_window
        return {
            "profile_id": self.profile_id,
            "state_file": self.state_file,
            "open_now": self.open_now,
            "entries_window": self.entries_window,
            "exits_window": self.exits_window,
            "closed_window": self.closed_window,
            "wins_window": self.wins_window,
            "losses_window": self.losses_window,
            "loss_share_window": round(self.loss_share_window, 4),
            "breakeven_window": self.breakeven_window,
            "pnl_window_usd": round(self.pnl_window_usd, 6),
            "pnl_window_clipped_usd": round(self.pnl_window_clipped_usd, 6),
            "winrate_window_pct": round(self.winrate_window_pct, 2),
            "profit_factor_window": round(pf, 4) if pf != float("inf") else "inf",
            "worst_trade_window_usd": round(self.worst_trade_window_usd, 6),
            "best_trade_window_usd": round(self.best_trade_window_usd, 6),
            "top_symbol_share_window": round(self.top_symbol_share_window, 4),
            "blocked_by_safe_source_window": int(self.blocked_by_safe_source_window),
            "blocked_by_watchlist_guard_window": int(self.blocked_by_watchlist_guard_window),
            "blocked_by_safety_budget_window": int(self.blocked_by_safety_budget_window),
            "blocked_total_window": int(
                self.blocked_by_safe_source_window
                + self.blocked_by_watchlist_guard_window
                + self.blocked_by_safety_budget_window
            ),
            "candidate_log_file": self.candidate_log_file,
            "score_total": round(self.score_total, 4),
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
        }


def _build_stats(
    profile_id: str,
    state_file: str,
    candidate_log_file: str,
    *,
    cutoff: datetime,
    min_closed: int,
) -> ProfileWindowStats:
    payload = _read_json(state_file)
    open_rows = payload.get("open_positions") or []
    closed_rows = payload.get("closed_positions") or []
    if not isinstance(open_rows, list):
        open_rows = []
    if not isinstance(closed_rows, list):
        closed_rows = []

    open_rows = [x for x in open_rows if isinstance(x, dict)]
    closed_rows = [x for x in closed_rows if isinstance(x, dict)]

    open_window = [x for x in open_rows if (_parse_ts(x.get("opened_at")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
    closed_window = [x for x in closed_rows if (_parse_ts(x.get("closed_at")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
    clipped_window = [x for x in closed_window if abs(_safe_float(x.get("pnl_percent"), 0.0)) < 100.0]
    base_window = clipped_window if clipped_window else closed_window

    entries_window = len(open_window)
    for row in closed_rows:
        opened_at = _parse_ts(row.get("opened_at"))
        if opened_at is not None and opened_at >= cutoff:
            entries_window += 1

    exits_window = len(closed_window)
    wins = sum(1 for row in base_window if _safe_float(row.get("pnl_usd"), 0.0) > 0.0)
    losses = sum(1 for row in base_window if _safe_float(row.get("pnl_usd"), 0.0) < 0.0)
    breakeven = max(0, len(base_window) - wins - losses)
    pnl_window = float(sum(_safe_float(row.get("pnl_usd"), 0.0) for row in closed_window))
    pnl_window_clipped = float(sum(_safe_float(row.get("pnl_usd"), 0.0) for row in base_window))
    loss_share = (float(losses) / float(max(1, wins + losses))) if (wins + losses) > 0 else 0.0

    winrate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0
    gains = float(sum(_safe_float(row.get("pnl_usd"), 0.0) for row in base_window if _safe_float(row.get("pnl_usd"), 0.0) > 0.0))
    losses_abs = abs(float(sum(_safe_float(row.get("pnl_usd"), 0.0) for row in base_window if _safe_float(row.get("pnl_usd"), 0.0) < 0.0)))
    if losses_abs > 0:
        pf = gains / losses_abs
    elif gains > 0:
        pf = float("inf")
    else:
        pf = 0.0

    worst = min((_safe_float(row.get("pnl_usd"), 0.0) for row in base_window), default=0.0)
    best = max((_safe_float(row.get("pnl_usd"), 0.0) for row in base_window), default=0.0)
    symbol_counts: dict[str, int] = {}
    for row in base_window:
        symbol = str(row.get("symbol", "") or "").strip().upper() or "UNKNOWN"
        symbol_counts[symbol] = int(symbol_counts.get(symbol, 0)) + 1
    top_symbol_share = 0.0
    if base_window and symbol_counts:
        top_symbol_share = max(symbol_counts.values()) / float(len(base_window))
    blocked = _window_blocked_counts(candidate_log_file, cutoff=cutoff)

    # Keep activity as a small tie-breaker, not a primary driver.
    activity_score = min(14.0, float(entries_window + exits_window) * 0.35)
    pnl_score = pnl_window_clipped * 160.0
    quality_score = (winrate - 50.0) * 0.8
    if pf == float("inf"):
        quality_score += 12.0
    else:
        quality_score += max(-16.0, min(16.0, (pf - 1.0) * 10.0))
    sample_score = min(14.0, len(closed_window) * 0.55)

    risk_penalty = 0.0
    risk_penalty += max(0.0, (abs(worst) - 0.08) * 110.0)
    risk_penalty += max(0.0, (loss_share - 0.56) * 40.0)
    risk_penalty += max(0.0, (float(top_symbol_share) - 0.38) * 42.0)
    if len(closed_window) < int(min_closed):
        risk_penalty += float(int(min_closed) - len(closed_window)) * 2.5

    total = activity_score + pnl_score + quality_score + sample_score - risk_penalty
    breakdown = {
        "activity": activity_score,
        "pnl": pnl_score,
        "quality": quality_score,
        "sample": sample_score,
        "risk_penalty": risk_penalty,
    }

    return ProfileWindowStats(
        profile_id=profile_id,
        state_file=state_file,
        candidate_log_file=candidate_log_file,
        open_now=len(open_rows),
        entries_window=entries_window,
        exits_window=exits_window,
        closed_window=len(closed_window),
        wins_window=wins,
        losses_window=losses,
        loss_share_window=loss_share,
        breakeven_window=breakeven,
        pnl_window_usd=pnl_window,
        pnl_window_clipped_usd=pnl_window_clipped,
        winrate_window_pct=winrate,
        profit_factor_window=pf,
        worst_trade_window_usd=worst,
        best_trade_window_usd=best,
        top_symbol_share_window=top_symbol_share,
        blocked_by_safe_source_window=int(blocked.get("blocked_by_safe_source", 0) or 0),
        blocked_by_watchlist_guard_window=int(blocked.get("blocked_by_watchlist_guard", 0) or 0),
        blocked_by_safety_budget_window=int(blocked.get("blocked_by_safety_budget", 0) or 0),
        score_total=total,
        score_breakdown=breakdown,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank active matrix profiles in a fixed recent window.")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--lookback-hours", type=float, default=4.0, help="Window size in hours")
    parser.add_argument("--min-closed", type=int, default=10, help="Minimum closed trades for confidence")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    items = _load_items(root)
    if not items:
        print("No matrix profiles found.")
        return 1

    lookback_hours = max(0.25, float(args.lookback_hours))
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    stats: list[ProfileWindowStats] = []
    for row in items:
        profile_id = str(row.get("id", "")).strip()
        state_file = str(row.get("state_file", "")).strip()
        candidate_log_file = _resolve_candidate_log(root=root, profile_id=profile_id, item=row)
        if not profile_id or not state_file:
            continue
        stats.append(
            _build_stats(
                profile_id,
                state_file,
                candidate_log_file,
                cutoff=cutoff,
                min_closed=max(1, int(args.min_closed)),
            )
        )
    if not stats:
        print("No profile stats collected.")
        return 1

    stats.sort(
        key=lambda s: (
            float(s.score_total),
            float(s.pnl_window_usd),
            float(s.winrate_window_pct),
            int(s.closed_window),
            int(s.entries_window + s.exits_window),
        ),
        reverse=True,
    )
    winner = stats[0]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_hours": lookback_hours,
        "cutoff_utc": cutoff.isoformat(),
        "min_closed": int(args.min_closed),
        "winner_id": winner.profile_id,
        "profiles_ranked": [s.to_dict() for s in stats],
    }

    report_dir = os.path.join(root, "data", "matrix", "reports")
    os.makedirs(report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(report_dir, f"matrix_window_rank_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"WINDOW_RANK_REPORT {out_path}")
    for idx, row in enumerate(stats, start=1):
        pf = "inf" if row.profit_factor_window == float("inf") else f"{row.profit_factor_window:.2f}"
        print(
            f"{idx}. {row.profile_id:<20} score={row.score_total:>7.2f} "
            f"pnl_4h=${row.pnl_window_usd:+.4f} clipped=${row.pnl_window_clipped_usd:+.4f} "
            f"closed_4h={row.closed_window:>3} top_sym={row.top_symbol_share_window:>4.2f} "
            f"wr_4h={row.winrate_window_pct:>5.1f}% pf_4h={pf} "
            f"entries_4h={row.entries_window:>3} "
            f"blocked(safe_source/watchlist/safety_budget)="
            f"{row.blocked_by_safe_source_window}/{row.blocked_by_watchlist_guard_window}/{row.blocked_by_safety_budget_window}"
        )
    print(f"WINNER {winner.profile_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
