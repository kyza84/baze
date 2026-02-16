"""Rank active matrix paper profiles and pick a winner for live promotion."""

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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_ts(raw: Any) -> datetime | None:
    if not raw:
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


@dataclass
class ProfileStats:
    profile_id: str
    state_file: str
    open_now: int
    closed_total: int
    wins: int
    losses: int
    breakeven: int
    realized_pnl_usd: float
    winrate_pct: float
    profit_factor: float
    worst_trade_usd: float
    current_loss_streak: int
    entries_last_window: int
    exits_last_window: int
    score_total: float
    score_breakdown: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        out = {
            "profile_id": self.profile_id,
            "state_file": self.state_file,
            "open_now": self.open_now,
            "closed_total": self.closed_total,
            "wins": self.wins,
            "losses": self.losses,
            "breakeven": self.breakeven,
            "realized_pnl_usd": round(self.realized_pnl_usd, 6),
            "winrate_pct": round(self.winrate_pct, 2),
            "profit_factor": round(self.profit_factor, 4) if self.profit_factor != float("inf") else "inf",
            "worst_trade_usd": round(self.worst_trade_usd, 6),
            "current_loss_streak": self.current_loss_streak,
            "entries_last_window": self.entries_last_window,
            "exits_last_window": self.exits_last_window,
            "score_total": round(self.score_total, 4),
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
        }
        return out


def _load_items(project_root: str) -> list[dict[str, str]]:
    active_path = os.path.join(project_root, "data", "matrix", "runs", "active_matrix.json")
    payload = _read_json(active_path)
    out: list[dict[str, str]] = []
    items = payload.get("items")
    if isinstance(items, list):
        for row in items:
            if not isinstance(row, dict):
                continue
            profile_id = str(row.get("id", "") or "").strip()
            state_file = str(row.get("paper_state_file", "") or "").strip()
            if not profile_id or not state_file:
                continue
            state_abs = state_file if os.path.isabs(state_file) else os.path.join(project_root, state_file)
            out.append({"id": profile_id, "state_file": state_abs})
    if out:
        return out

    # Fallback to local state files if active matrix meta is absent.
    trading_dir = os.path.join(project_root, "trading")
    if not os.path.isdir(trading_dir):
        return out
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


def _current_loss_streak(closed_rows: list[dict[str, Any]]) -> int:
    sorted_rows = sorted(
        closed_rows,
        key=lambda x: _parse_ts(x.get("closed_at")) or datetime.min.replace(tzinfo=timezone.utc),
    )
    streak = 0
    for row in reversed(sorted_rows):
        pnl = _safe_float((row or {}).get("pnl_usd"), 0.0)
        if pnl < 0:
            streak += 1
            continue
        break
    return streak


def _build_stats(
    profile_id: str,
    state_file: str,
    *,
    lookback_hours: float,
    min_closed: int,
) -> ProfileStats:
    state = _read_json(state_file)
    open_rows = state.get("open_positions") or []
    closed_rows = state.get("closed_positions") or []
    if not isinstance(open_rows, list):
        open_rows = []
    if not isinstance(closed_rows, list):
        closed_rows = []
    open_rows = [x for x in open_rows if isinstance(x, dict)]
    closed_rows = [x for x in closed_rows if isinstance(x, dict)]

    wins = sum(1 for row in closed_rows if _safe_float(row.get("pnl_usd"), 0.0) > 0.0)
    losses = sum(1 for row in closed_rows if _safe_float(row.get("pnl_usd"), 0.0) < 0.0)
    breakeven = max(0, len(closed_rows) - wins - losses)
    realized = _safe_float(state.get("realized_pnl_usd"), 0.0)
    if realized == 0.0 and closed_rows:
        realized = float(sum(_safe_float(row.get("pnl_usd"), 0.0) for row in closed_rows))

    closed_non_be = max(1, wins + losses)
    winrate_pct = (wins / closed_non_be) * 100.0 if (wins + losses) > 0 else 0.0

    gain_sum = float(sum(_safe_float(row.get("pnl_usd"), 0.0) for row in closed_rows if _safe_float(row.get("pnl_usd"), 0.0) > 0))
    loss_sum_abs = float(
        abs(sum(_safe_float(row.get("pnl_usd"), 0.0) for row in closed_rows if _safe_float(row.get("pnl_usd"), 0.0) < 0))
    )
    if loss_sum_abs > 0:
        profit_factor = gain_sum / loss_sum_abs
    elif gain_sum > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    worst_trade_usd = 0.0
    if closed_rows:
        worst_trade_usd = float(min(_safe_float(row.get("pnl_usd"), 0.0) for row in closed_rows))
    current_loss_streak = _current_loss_streak(closed_rows)

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=max(0.5, float(lookback_hours)))

    entries_last = 0
    exits_last = 0
    for row in open_rows:
        opened_at = _parse_ts(row.get("opened_at"))
        if opened_at is not None and opened_at >= cutoff:
            entries_last += 1

    for row in closed_rows:
        opened_at = _parse_ts(row.get("opened_at"))
        closed_at = _parse_ts(row.get("closed_at"))
        if opened_at is not None and opened_at >= cutoff:
            entries_last += 1
        if closed_at is not None and closed_at >= cutoff:
            exits_last += 1

    # Composite score: prioritize realized pnl + activity, keep risk penalty explicit.
    activity_score = min(40.0, float(entries_last + exits_last) * 2.5)
    pnl_score = float(realized) * 45.0
    quality_score = ((winrate_pct - 50.0) * 0.35)
    if profit_factor == float("inf"):
        quality_score += 15.0
    else:
        quality_score += max(-15.0, min(15.0, (float(profit_factor) - 1.0) * 12.0))
    sample_score = min(12.0, len(closed_rows) * 0.4)

    risk_penalty = 0.0
    risk_penalty += max(0.0, (abs(worst_trade_usd) - 0.70) * 20.0)
    risk_penalty += max(0.0, (float(current_loss_streak) - 2.0) * 4.0)
    if len(closed_rows) < int(min_closed):
        risk_penalty += float(int(min_closed) - len(closed_rows)) * 2.0

    total = activity_score + pnl_score + quality_score + sample_score - risk_penalty
    breakdown = {
        "activity": activity_score,
        "pnl": pnl_score,
        "quality": quality_score,
        "sample": sample_score,
        "risk_penalty": risk_penalty,
    }

    return ProfileStats(
        profile_id=profile_id,
        state_file=state_file,
        open_now=len(open_rows),
        closed_total=len(closed_rows),
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        realized_pnl_usd=float(realized),
        winrate_pct=float(winrate_pct),
        profit_factor=float(profit_factor),
        worst_trade_usd=float(worst_trade_usd),
        current_loss_streak=current_loss_streak,
        entries_last_window=entries_last,
        exits_last_window=exits_last,
        score_total=float(total),
        score_breakdown=breakdown,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank matrix paper profiles and select winner.")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--lookback-hours", type=float, default=6.0, help="Window for activity scoring")
    parser.add_argument("--min-closed", type=int, default=8, help="Minimum closed trades before full confidence")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    items = _load_items(root)
    if not items:
        print("No matrix profiles found.")
        return 1

    stats: list[ProfileStats] = []
    for row in items:
        profile_id = str(row.get("id", "")).strip()
        state_file = str(row.get("state_file", "")).strip()
        if not profile_id or not state_file:
            continue
        stats.append(
            _build_stats(
                profile_id=profile_id,
                state_file=state_file,
                lookback_hours=float(args.lookback_hours),
                min_closed=max(1, int(args.min_closed)),
            )
        )
    if not stats:
        print("No profile stats collected.")
        return 1

    stats.sort(
        key=lambda s: (
            float(s.score_total),
            float(s.realized_pnl_usd),
            int(s.exits_last_window + s.entries_last_window),
            int(s.closed_total),
        ),
        reverse=True,
    )
    winner = stats[0]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_hours": float(args.lookback_hours),
        "min_closed": int(args.min_closed),
        "winner_id": winner.profile_id,
        "profiles_ranked": [s.to_dict() for s in stats],
    }
    report_dir = os.path.join(root, "data", "matrix", "reports")
    os.makedirs(report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(report_dir, f"matrix_rank_{stamp}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"RANK_REPORT {out_json}")
    for idx, s in enumerate(stats, start=1):
        pf = "inf" if s.profit_factor == float("inf") else f"{s.profit_factor:.2f}"
        print(
            f"{idx}. {s.profile_id:<20} score={s.score_total:>7.2f} "
            f"realized=${s.realized_pnl_usd:+.4f} closed={s.closed_total:>3} "
            f"W/L/BE={s.wins}/{s.losses}/{s.breakeven} wr={s.winrate_pct:>5.1f}% "
            f"pf={pf} act={s.entries_last_window + s.exits_last_window}"
        )
    print(f"WINNER {winner.profile_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
