import argparse
import collections
import datetime as dt
import json
import os
import statistics
from typing import Any


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


def _profile_from_path(path: str) -> str:
    base = os.path.basename(path).lower()
    if base.startswith("paper_state.") and base.endswith(".json"):
        return base.replace("paper_state.", "").replace(".json", "")
    if "mx1_" in base:
        return "mx1_refine"
    if "mx2_" in base:
        return "mx2_explore_wide"
    if "mx3_" in base:
        return "mx3_timeout_cut"
    if "baseline" in base:
        return "mx1_baseline"
    return "single_or_other"


def _find_state_files(root: str) -> list[str]:
    out: list[str] = []
    for base in ("trading", os.path.join("data", "matrix", "backups")):
        base_path = os.path.join(root, base)
        if not os.path.isdir(base_path):
            continue
        for dirpath, _, files in os.walk(base_path):
            for name in files:
                lower = name.lower()
                if not lower.endswith(".json") and ".json." not in lower:
                    continue
                if "paper_state" not in lower:
                    continue
                out.append(os.path.join(dirpath, name))
    return sorted(set(out))


def _trade_key(t: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(t.get("token_address", "")).lower(),
        str(t.get("symbol", "")),
        str(t.get("opened_at", "")),
        str(t.get("closed_at", "")),
        round(_safe_float(t.get("position_size_usd")), 6),
        round(_safe_float(t.get("entry_price_usd")), 12),
        str(t.get("close_reason", "")),
    )


def _score_bin(score: int) -> str:
    if score < 60:
        return "<60"
    if score < 70:
        return "60-69"
    if score < 80:
        return "70-79"
    if score < 90:
        return "80-89"
    return "90+"


def _hold_bin(seconds: int) -> str:
    if seconds < 240:
        return "<4m"
    if seconds < 360:
        return "4-6m"
    if seconds < 480:
        return "6-8m"
    if seconds < 600:
        return "8-10m"
    return "10m+"


def build_report(root: str) -> dict[str, Any]:
    files = _find_state_files(root)
    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_files = 0
    parse_errors = 0

    for path in files:
        source_files += 1
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                payload = json.load(f)
        except Exception:
            parse_errors += 1
            continue

        closed = payload.get("closed_positions") or []
        if not isinstance(closed, list):
            continue

        profile = _profile_from_path(path)
        for t in closed:
            if not isinstance(t, dict):
                continue
            key = _trade_key(t)
            row = dict(t)
            row["_profile"] = profile
            row["_source"] = path
            dedup[key] = row

    trades = list(dedup.values())
    pnls = [_safe_float(t.get("pnl_usd")) for t in trades]
    reasons = collections.Counter(str(t.get("close_reason", "UNKNOWN")) for t in trades)
    profiles = collections.Counter(str(t.get("_profile", "unknown")) for t in trades)
    score_bins = collections.Counter(_score_bin(_safe_int(t.get("score"))) for t in trades)
    hold_bins = collections.Counter(_hold_bin(_safe_int(t.get("max_hold_seconds"))) for t in trades)

    by_reason_pnl: dict[str, list[float]] = collections.defaultdict(list)
    by_profile_pnl: dict[str, list[float]] = collections.defaultdict(list)
    by_score_pnl: dict[str, list[float]] = collections.defaultdict(list)

    for t in trades:
        pnl = _safe_float(t.get("pnl_usd"))
        by_reason_pnl[str(t.get("close_reason", "UNKNOWN"))].append(pnl)
        by_profile_pnl[str(t.get("_profile", "unknown"))].append(pnl)
        by_score_pnl[_score_bin(_safe_int(t.get("score")))].append(pnl)

    def _pack(group: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for k, vals in group.items():
            if not vals:
                continue
            out[k] = {
                "count": len(vals),
                "sum_usd": round(sum(vals), 6),
                "avg_usd": round(sum(vals) / len(vals), 6),
                "median_usd": round(statistics.median(vals), 6),
            }
        return out

    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_files_found": len(files),
        "source_files_loaded": source_files - parse_errors,
        "source_files_parse_errors": parse_errors,
        "closed_trades_unique": len(trades),
        "pnl": {
            "total_usd": round(sum(pnls), 6),
            "avg_usd": round((sum(pnls) / len(pnls)), 6) if pnls else 0.0,
            "median_usd": round(statistics.median(pnls), 6) if pnls else 0.0,
            "best_usd": round(max(pnls), 6) if pnls else 0.0,
            "worst_usd": round(min(pnls), 6) if pnls else 0.0,
            "wins": sum(1 for x in pnls if x > 0),
            "losses": sum(1 for x in pnls if x < 0),
            "breakeven": sum(1 for x in pnls if x == 0),
            "winrate_pct": round((sum(1 for x in pnls if x > 0) / len(pnls) * 100.0), 2) if pnls else 0.0,
        },
        "close_reasons": dict(reasons.most_common()),
        "profiles": dict(profiles.most_common()),
        "score_bins": dict(score_bins.most_common()),
        "hold_bins": dict(hold_bins.most_common()),
        "pnl_by_reason": _pack(by_reason_pnl),
        "pnl_by_profile": _pack(by_profile_pnl),
        "pnl_by_score_bin": _pack(by_score_pnl),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze accumulated paper_state dataset.")
    parser.add_argument("--root", default=".", help="Project root")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    report = build_report(root)

    out_dir = os.path.join(root, "data", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"paper_dataset_report_{stamp}.json")
    md_path = os.path.join(out_dir, f"paper_dataset_report_{stamp}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Paper Dataset Report",
        f"- Generated: {report['generated_at']}",
        f"- Source files loaded: {report['source_files_loaded']} / {report['source_files_found']}",
        f"- Unique closed trades: {report['closed_trades_unique']}",
        "",
        "## PnL",
        f"- Total USD: {report['pnl']['total_usd']}",
        f"- Winrate: {report['pnl']['winrate_pct']}% ({report['pnl']['wins']}W / {report['pnl']['losses']}L)",
        f"- Avg/Median: {report['pnl']['avg_usd']} / {report['pnl']['median_usd']}",
        f"- Best/Worst: {report['pnl']['best_usd']} / {report['pnl']['worst_usd']}",
        "",
        "## Close Reasons",
    ]
    for k, v in report["close_reasons"].items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## PnL By Reason"]
    for k, v in sorted(report["pnl_by_reason"].items()):
        lines.append(f"- {k}: count={v['count']} sum={v['sum_usd']} avg={v['avg_usd']} median={v['median_usd']}")

    lines += ["", "## PnL By Profile"]
    for k, v in sorted(report["pnl_by_profile"].items()):
        lines.append(f"- {k}: count={v['count']} sum={v['sum_usd']} avg={v['avg_usd']} median={v['median_usd']}")

    lines += ["", "## PnL By Score Bin"]
    for k, v in sorted(report["pnl_by_score_bin"].items()):
        lines.append(f"- {k}: count={v['count']} sum={v['sum_usd']} avg={v['avg_usd']} median={v['median_usd']}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json_path)
    print(md_path)
    print(json.dumps(report["pnl"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
