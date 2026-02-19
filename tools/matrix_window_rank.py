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
            out.append({"id": profile_id, "state_file": state_file})
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
            }
        )
    return out


@dataclass
class ProfileWindowStats:
    profile_id: str
    state_file: str
    open_now: int
    entries_window: int
    exits_window: int
    closed_window: int
    wins_window: int
    losses_window: int
    loss_share_window: float
    breakeven_window: int
    pnl_window_usd: float
    winrate_window_pct: float
    profit_factor_window: float
    worst_trade_window_usd: float
    best_trade_window_usd: float
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
            "winrate_window_pct": round(self.winrate_window_pct, 2),
            "profit_factor_window": round(pf, 4) if pf != float("inf") else "inf",
            "worst_trade_window_usd": round(self.worst_trade_window_usd, 6),
            "best_trade_window_usd": round(self.best_trade_window_usd, 6),
            "score_total": round(self.score_total, 4),
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
        }


def _build_stats(profile_id: str, state_file: str, *, cutoff: datetime, min_closed: int) -> ProfileWindowStats:
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

    entries_window = len(open_window)
    for row in closed_rows:
        opened_at = _parse_ts(row.get("opened_at"))
        if opened_at is not None and opened_at >= cutoff:
            entries_window += 1

    exits_window = len(closed_window)
    wins = sum(1 for row in closed_window if _safe_float(row.get("pnl_usd"), 0.0) > 0.0)
    losses = sum(1 for row in closed_window if _safe_float(row.get("pnl_usd"), 0.0) < 0.0)
    breakeven = max(0, len(closed_window) - wins - losses)
    pnl_window = float(sum(_safe_float(row.get("pnl_usd"), 0.0) for row in closed_window))
    loss_share = (float(losses) / float(max(1, wins + losses))) if (wins + losses) > 0 else 0.0

    winrate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0
    gains = float(sum(_safe_float(row.get("pnl_usd"), 0.0) for row in closed_window if _safe_float(row.get("pnl_usd"), 0.0) > 0.0))
    losses_abs = abs(float(sum(_safe_float(row.get("pnl_usd"), 0.0) for row in closed_window if _safe_float(row.get("pnl_usd"), 0.0) < 0.0)))
    if losses_abs > 0:
        pf = gains / losses_abs
    elif gains > 0:
        pf = float("inf")
    else:
        pf = 0.0

    worst = min((_safe_float(row.get("pnl_usd"), 0.0) for row in closed_window), default=0.0)
    best = max((_safe_float(row.get("pnl_usd"), 0.0) for row in closed_window), default=0.0)

    # Keep activity as a small tie-breaker, not a primary driver.
    activity_score = min(14.0, float(entries_window + exits_window) * 0.35)
    pnl_score = pnl_window * 160.0
    quality_score = (winrate - 50.0) * 0.8
    if pf == float("inf"):
        quality_score += 12.0
    else:
        quality_score += max(-16.0, min(16.0, (pf - 1.0) * 10.0))
    sample_score = min(14.0, len(closed_window) * 0.55)

    risk_penalty = 0.0
    risk_penalty += max(0.0, (abs(worst) - 0.08) * 110.0)
    risk_penalty += max(0.0, (loss_share - 0.56) * 40.0)
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
        open_now=len(open_rows),
        entries_window=entries_window,
        exits_window=exits_window,
        closed_window=len(closed_window),
        wins_window=wins,
        losses_window=losses,
        loss_share_window=loss_share,
        breakeven_window=breakeven,
        pnl_window_usd=pnl_window,
        winrate_window_pct=winrate,
        profit_factor_window=pf,
        worst_trade_window_usd=worst,
        best_trade_window_usd=best,
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
        if not profile_id or not state_file:
            continue
        stats.append(_build_stats(profile_id, state_file, cutoff=cutoff, min_closed=max(1, int(args.min_closed))))
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
            f"{idx}. {row.profile_id:<20} score={row.score_total:>7.2f} pnl_4h=${row.pnl_window_usd:+.4f} "
            f"closed_4h={row.closed_window:>3} wr_4h={row.winrate_window_pct:>5.1f}% pf_4h={pf} "
            f"entries_4h={row.entries_window:>3}"
        )
    print(f"WINNER {winner.profile_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
