from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_builtin_profiles(launcher_path: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not launcher_path.exists():
        return out
    patt = re.compile(r"name='([^']+)'\s*;\s*(?:base='([^']+)'\s*;\s*)?overrides=@\{")
    for line in launcher_path.read_text(encoding="utf-8").splitlines():
        m = patt.search(line)
        if not m:
            continue
        out.append(
            {
                "name": m.group(1).strip(),
                "base": (m.group(2) or "").strip(),
                "kind": "builtin",
            }
        )
    return out


def _read_user_presets(preset_dir: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not preset_dir.exists():
        return out
    for path in sorted(preset_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        name = str(data.get("name", "") or "").strip()
        base = str(data.get("base", "") or "").strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "base": base,
                "kind": "user",
                "updated_at": str(data.get("updated_at", "") or data.get("created_at", "") or "").strip(),
            }
        )
    return out


def _read_recent_report_winners(report_dir: Path, limit: int) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    if not report_dir.exists():
        return winners
    files = sorted(report_dir.glob("matrix_*_rank_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    seen: set[tuple[str, str]] = set()
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        winner = str(data.get("winner_id", "") or "").strip()
        if not winner:
            continue
        key = (path.name.split("_")[1], winner)
        if key in seen:
            continue
        seen.add(key)
        winners.append(
            {
                "report": path.name,
                "winner_id": winner,
                "generated_at": str(data.get("generated_at", "") or "").strip(),
            }
        )
        if len(winners) >= max(1, int(limit)):
            break
    return winners


def main() -> int:
    parser = argparse.ArgumentParser(description="Show matrix profile catalog (builtin + user + recent winners).")
    parser.add_argument("--root", default=".")
    parser.add_argument("--top-winners", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    launcher = root / "tools" / "matrix_paper_launcher.ps1"
    user_dir = root / "data" / "matrix" / "user_presets"
    report_dir = root / "data" / "matrix" / "reports"

    builtins = _parse_builtin_profiles(launcher)
    users = _read_user_presets(user_dir)
    winners = _read_recent_report_winners(report_dir, limit=max(1, int(args.top_winners)))

    payload = {
        "root": str(root),
        "builtin_count": len(builtins),
        "user_count": len(users),
        "profiles": sorted([*builtins, *users], key=lambda x: (x.get("kind", ""), x.get("name", ""))),
        "recent_winners": winners,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Built-in profiles: {len(builtins)}")
    print(f"User presets: {len(users)}")
    if users:
        print("User presets list:")
        for row in users:
            name = row.get("name", "")
            base = row.get("base", "")
            upd = row.get("updated_at", "")
            print(f"- {name} (base={base}, updated={upd})")
    print("Recent winners:")
    if winners:
        for row in winners:
            print(f"- {row['winner_id']} via {row['report']} at {row['generated_at']}")
    else:
        print("- none")
    print("Pinned working references:")
    for fixed in ("mx30_guarded_balanced", "mx31_guarded_aggressive"):
        print(f"- {fixed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

