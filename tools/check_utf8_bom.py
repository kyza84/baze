"""Check tracked repository files for UTF-8 BOM prefixes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BOM = b"\xef\xbb\xbf"
ALWAYS_TEXT_FILENAMES = {
    ".env.example",
    ".gitignore",
}
TEXT_SUFFIXES = {
    ".env",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FALLBACK_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "collected_info",
    "data",
    "logs",
    "snapshots",
}


def _looks_text(path: Path) -> bool:
    return path.name in ALWAYS_TEXT_FILENAMES or path.suffix.lower() in TEXT_SUFFIXES


def _git_tracked_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    out: list[Path] = []
    for raw in result.stdout.splitlines():
        rel = raw.strip()
        if not rel:
            continue
        candidate = (root / rel).resolve()
        if candidate.is_file() and _looks_text(candidate):
            out.append(candidate)
    return out


def _fallback_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in FALLBACK_SKIP_DIRS for part in path.parts):
            continue
        if _looks_text(path):
            out.append(path)
    return out


def _target_files(root: Path) -> list[Path]:
    tracked = _git_tracked_files(root)
    if tracked:
        return tracked
    return _fallback_files(root)


def _has_bom(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(3) == BOM
    except OSError:
        return False


def _strip_bom(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if not data.startswith(BOM):
        return False
    path.write_bytes(data[len(BOM) :])
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit repository files for UTF-8 BOM prefixes.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove BOM prefixes in-place instead of failing.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    hits: list[Path] = [path for path in _target_files(root) if _has_bom(path)]

    if not hits:
        print("OK: no UTF-8 BOM prefixes found.")
        return 0

    if args.fix:
        fixed = 0
        for path in hits:
            if _strip_bom(path):
                fixed += 1
                print(f"FIXED {path.relative_to(root).as_posix()}")
        print(f"DONE: removed BOM from {fixed} file(s).")
        return 0

    print("FAIL: UTF-8 BOM prefixes detected in:")
    for path in hits:
        print(f" - {path.relative_to(root).as_posix()}")
    print("Use: python tools/check_utf8_bom.py --fix")
    return 1


if __name__ == "__main__":
    sys.exit(main())
