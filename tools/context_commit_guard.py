from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQUIRED = {
    "docs/PROJECT_STATE.md",
    "docs/CHAT_FIRST_MESSAGE.md",
}

IGNORED_PREFIXES = (
    "logs/",
    "data/",
    "snapshots/",
    "collected_info/",
    "__pycache__/",
)


def _git(args: list[str]) -> list[str]:
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr or p.stdout)
        raise SystemExit(p.returncode)
    return [x.strip().replace("\\", "/") for x in (p.stdout or "").splitlines() if x.strip()]


def main() -> int:
    staged = _git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    if not staged:
        return 0

    staged_set = set(staged)
    material = []
    for path in staged:
        low = path.lower()
        if any(low.startswith(pref) for pref in IGNORED_PREFIXES):
            continue
        if path in REQUIRED:
            continue
        material.append(path)

    if not material:
        return 0

    missing = [p for p in sorted(REQUIRED) if p not in staged_set]
    if missing:
        sys.stderr.write(
            "\n[context-guard] Commit blocked: context docs must be updated for material changes.\n"
        )
        sys.stderr.write("[context-guard] Material staged files:\n")
        for m in material[:20]:
            sys.stderr.write(f"  - {m}\n")
        if len(material) > 20:
            sys.stderr.write(f"  ... and {len(material)-20} more\n")
        sys.stderr.write("[context-guard] Missing required staged files:\n")
        for miss in missing:
            sys.stderr.write(f"  - {miss}\n")
        sys.stderr.write(
            "\nUpdate and stage context docs, then commit again.\n"
            "See docs/CONTEXT_UPDATE_RULES.md\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
