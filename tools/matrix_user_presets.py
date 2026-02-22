from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matrix_preset_guard import load_contract, validate_overrides


NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _preset_dir(root: Path) -> Path:
    return root / "data" / "matrix" / "user_presets"


def _launcher_path(root: Path) -> Path:
    return root / "tools" / "matrix_paper_launcher.ps1"


def _builtin_profile_names(root: Path) -> set[str]:
    out: set[str] = set()
    path = _launcher_path(root)
    if not path.exists():
        return out
    patt = re.compile(r"name='([^']+)'\s*;\s*(?:base='([^']+)'\s*;\s*)?overrides=@\{")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = patt.search(line)
        if m:
            out.add(m.group(1).strip())
    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    name = str(value or "").strip()
    if not name or not NAME_RE.match(name):
        raise ValueError("Invalid preset name. Allowed: A-Z a-z 0-9 . _ -")
    return name


def _parse_sets(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in items:
        s = str(raw or "").strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"Invalid --set '{s}'. Expected KEY=VALUE.")
        k, v = s.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid --set '{s}'. Empty key.")
        out[k] = v.strip()
    return out


def _load_overrides_file(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    data = _read_json(Path(path))
    if not isinstance(data, dict):
        raise ValueError("Overrides file must be a JSON object.")
    out: dict[str, str] = {}
    for k, v in data.items():
        kk = str(k).strip()
        if not kk:
            continue
        out[kk] = str(v)
    return out


def _list_user_presets(root: Path) -> list[dict[str, Any]]:
    base = _preset_dir(root)
    if not base.exists():
        return []
    out: list[dict[str, Any]] = []
    for f in sorted(base.glob("*.json")):
        try:
            row = _read_json(f)
            if isinstance(row, dict):
                row["_file"] = str(f)
                out.append(row)
        except Exception:
            continue
    return out


def cmd_list(args: argparse.Namespace) -> int:
    root = _project_root()
    builtins = _builtin_profile_names(root)
    users = _list_user_presets(root)
    print(f"Built-in profiles: {len(builtins)}")
    if users:
        print(f"User presets: {len(users)}")
        for row in users:
            name = str(row.get("name", "") or "").strip()
            base = str(row.get("base", "") or "").strip()
            updated = str(row.get("updated_at", "") or row.get("created_at", "") or "").strip()
            note = str(row.get("note", "") or "").strip()
            note_part = f" note={note}" if note else ""
            print(f"- {name} (base={base}, updated={updated}){note_part}")
    else:
        print("User presets: 0")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    root = _project_root()
    name = _safe_name(args.name)
    path = _preset_dir(root) / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"Preset not found: {name}")
    text = path.read_text(encoding="utf-8")
    if bool(args.json_only):
        print(text)
    else:
        print(path)
        print(text)
    return 0


def _resolve_source_preset(root: Path, source: str) -> tuple[str, dict[str, str], str]:
    src = _safe_name(source)
    path = _preset_dir(root) / f"{src}.json"
    if path.exists():
        row = _read_json(path)
        base = str(row.get("base", "") or "").strip() or src
        overrides_raw = row.get("overrides", {})
        overrides: dict[str, str] = {}
        if isinstance(overrides_raw, dict):
            for k, v in overrides_raw.items():
                kk = str(k).strip()
                if kk:
                    overrides[kk] = str(v)
        return src, overrides, base
    builtins = _builtin_profile_names(root)
    if src not in builtins:
        raise SystemExit(f"Unknown source preset: {src}")
    return src, {}, src


def cmd_create(args: argparse.Namespace) -> int:
    root = _project_root()
    name = _safe_name(args.name)
    base = _safe_name(args.base)
    builtins = _builtin_profile_names(root)
    if name in builtins:
        raise SystemExit(f"Preset '{name}' conflicts with built-in profile name.")
    if base not in builtins and not (_preset_dir(root) / f"{base}.json").exists():
        raise SystemExit(f"Base profile not found: {base}")

    path = _preset_dir(root) / f"{name}.json"
    if path.exists() and not bool(args.force):
        raise SystemExit(f"Preset already exists: {name}. Use --force to overwrite.")

    overrides = _load_overrides_file(args.overrides_file)
    overrides.update(_parse_sets(args.sets or []))
    contract = load_contract(root)
    issues = validate_overrides(overrides, contract)
    if issues:
        raise SystemExit("Preset validation failed:\n- " + "\n- ".join(issues))
    now = _now_iso()
    payload = {
        "schema_version": "matrix.user_preset.v1",
        "name": name,
        "base": base,
        "overrides": overrides,
        "note": str(args.note or "").strip(),
        "created_at": now,
        "updated_at": now,
    }
    _write_json(path, payload)
    print(f"Created preset: {path}")
    return 0


def cmd_clone(args: argparse.Namespace) -> int:
    root = _project_root()
    name = _safe_name(args.name)
    builtins = _builtin_profile_names(root)
    if name in builtins:
        raise SystemExit(f"Preset '{name}' conflicts with built-in profile name.")
    path = _preset_dir(root) / f"{name}.json"
    if path.exists() and not bool(args.force):
        raise SystemExit(f"Preset already exists: {name}. Use --force to overwrite.")

    source_name, source_overrides, source_base = _resolve_source_preset(root, args.source)
    base = source_name
    # Keep clone shallow (base = source) so future source fixes propagate.
    # Optional additional overrides stay explicit on top.
    overrides = dict(source_overrides)
    overrides.update(_load_overrides_file(args.overrides_file))
    overrides.update(_parse_sets(args.sets or []))
    contract = load_contract(root)
    issues = validate_overrides(overrides, contract)
    if issues:
        raise SystemExit("Preset validation failed:\n- " + "\n- ".join(issues))

    now = _now_iso()
    payload = {
        "schema_version": "matrix.user_preset.v1",
        "name": name,
        "base": base,
        "overrides": overrides,
        "note": str(args.note or f"clone_of={source_name} base_hint={source_base}").strip(),
        "created_at": now,
        "updated_at": now,
    }
    _write_json(path, payload)
    print(f"Cloned preset: {path}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    root = _project_root()
    name = _safe_name(args.name)
    path = _preset_dir(root) / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"Preset not found: {name}")
    path.unlink()
    print(f"Deleted preset: {name}")
    return 0


def cmd_allowed(args: argparse.Namespace) -> int:
    root = _project_root()
    contract = load_contract(root)
    allow = contract.get("allowed_keys", {}) or {}
    if bool(args.json):
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return 0
    print(f"Safe tuning keys: {len(allow)}")
    for key in sorted(allow.keys()):
        spec = allow.get(key, {})
        kind = str(spec.get("type", "-"))
        lo = spec.get("min")
        hi = spec.get("max")
        sens = str(spec.get("sensitivity", "-"))
        imp = str(spec.get("importance", "-"))
        rng = ""
        if lo is not None or hi is not None:
            rng = f" range=[{lo},{hi}]"
        print(f"- {key} type={kind}{rng} sensitivity={sens} importance={imp}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manage matrix user presets.")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_list = sp.add_parser("list", help="List built-in profiles and user presets.")
    p_list.set_defaults(fn=cmd_list)

    p_show = sp.add_parser("show", help="Show user preset JSON.")
    p_show.add_argument("--name", required=True)
    p_show.add_argument("--json-only", action="store_true")
    p_show.set_defaults(fn=cmd_show)

    p_create = sp.add_parser("create", help="Create user preset from a base profile.")
    p_create.add_argument("--name", required=True)
    p_create.add_argument("--base", required=True)
    p_create.add_argument("--overrides-file", default="")
    p_create.add_argument("--set", dest="sets", action="append", default=[])
    p_create.add_argument("--note", default="")
    p_create.add_argument("--force", action="store_true")
    p_create.set_defaults(fn=cmd_create)

    p_clone = sp.add_parser("clone", help="Clone built-in or user preset into user preset.")
    p_clone.add_argument("--source", required=True)
    p_clone.add_argument("--name", required=True)
    p_clone.add_argument("--overrides-file", default="")
    p_clone.add_argument("--set", dest="sets", action="append", default=[])
    p_clone.add_argument("--note", default="")
    p_clone.add_argument("--force", action="store_true")
    p_clone.set_defaults(fn=cmd_clone)

    p_del = sp.add_parser("delete", help="Delete user preset.")
    p_del.add_argument("--name", required=True)
    p_del.set_defaults(fn=cmd_delete)

    p_allowed = sp.add_parser("allowed", help="Print safe tuning contract keys.")
    p_allowed.add_argument("--json", action="store_true")
    p_allowed.set_defaults(fn=cmd_allowed)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
