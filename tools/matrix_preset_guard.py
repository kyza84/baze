from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BOOL_TRUE = {"1", "true", "yes", "on"}
BOOL_FALSE = {"0", "false", "no", "off"}
SYMBOL_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SLUG_RE = re.compile(r"^[A-Za-z0-9._:+-]+$")


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _contract_path(root: Path) -> Path:
    return root / "tools" / "matrix_safe_tuning_contract.json"


def load_contract(root: Path) -> dict[str, Any]:
    path = _contract_path(root)
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any) -> float:
    return float(str(value).strip())


def _to_int(value: Any) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"[-+]?\d+", text):
        raise ValueError("must be integer")
    return int(text)


def _check_bool(value: Any) -> None:
    text = str(value).strip().lower()
    if text not in BOOL_TRUE and text not in BOOL_FALSE:
        raise ValueError("must be boolean true/false")


def _check_csv_symbol_list(value: Any, max_items: int) -> None:
    text = str(value or "").strip()
    if not text:
        return
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) > max_items:
        raise ValueError(f"too many items (max={max_items})")
    for sym in parts:
        if not SYMBOL_RE.match(sym):
            raise ValueError(f"invalid symbol '{sym}'")


def _check_csv_slug_list(value: Any, max_items: int) -> None:
    text = str(value or "").strip()
    if not text:
        return
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    if len(parts) > max_items:
        raise ValueError(f"too many items (max={max_items})")
    for item in parts:
        if not SLUG_RE.match(item):
            raise ValueError(f"invalid slug '{item}'")


def _check_source_cap_map(value: Any, max_items: int, min_value: int, max_value: int) -> None:
    text = str(value or "").strip()
    if not text:
        raise ValueError("must not be empty")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) > max_items:
        raise ValueError(f"too many pairs (max={max_items})")
    seen: set[str] = set()
    for chunk in parts:
        if ":" not in chunk:
            raise ValueError(f"invalid pair '{chunk}' (expected key:value)")
        raw_key, raw_val = chunk.split(":", 1)
        key = raw_key.strip().lower()
        if not key:
            raise ValueError("empty source key")
        if key in seen:
            raise ValueError(f"duplicate source key '{key}'")
        if not SLUG_RE.match(key):
            raise ValueError(f"invalid source key '{key}'")
        seen.add(key)
        val = _to_int(raw_val)
        if val < min_value:
            raise ValueError(f"source cap '{key}' below min {min_value}")
        if val > max_value:
            raise ValueError(f"source cap '{key}' above max {max_value}")


def _validate_scalar(name: str, value: Any, spec: dict[str, Any]) -> str | None:
    kind = str(spec.get("type", "")).strip()
    try:
        if kind == "float":
            val = _to_float(value)
            lo = spec.get("min")
            hi = spec.get("max")
            if lo is not None and val < float(lo):
                return f"{name}={value} below min {lo}"
            if hi is not None and val > float(hi):
                return f"{name}={value} above max {hi}"
            return None
        if kind == "int":
            val = _to_int(value)
            lo = spec.get("min")
            hi = spec.get("max")
            if lo is not None and val < int(lo):
                return f"{name}={value} below min {lo}"
            if hi is not None and val > int(hi):
                return f"{name}={value} above max {hi}"
            return None
        if kind == "bool":
            _check_bool(value)
            return None
        if kind == "enum":
            opts = {str(x) for x in (spec.get("values") or [])}
            if str(value) not in opts:
                return f"{name}={value} not in {sorted(opts)}"
            return None
        if kind == "csv_symbol_list":
            max_items = int(spec.get("max_items", 50) or 50)
            _check_csv_symbol_list(value, max_items=max_items)
            return None
        if kind == "csv_slug_list":
            max_items = int(spec.get("max_items", 50) or 50)
            _check_csv_slug_list(value, max_items=max_items)
            return None
        if kind == "source_cap_map":
            max_items = int(spec.get("max_items", 16) or 16)
            min_value = int(spec.get("min_value", 1) or 1)
            max_value = int(spec.get("max_value", 999) or 999)
            _check_source_cap_map(
                value,
                max_items=max_items,
                min_value=min_value,
                max_value=max_value,
            )
            return None
        return f"{name}: unsupported spec type '{kind}'"
    except ValueError as exc:
        return f"{name}={value} invalid: {exc}"


def _validate_global_rules(overrides: dict[str, str], rules: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for rule in rules:
        left = str(rule.get("left", "")).strip()
        right = str(rule.get("right", "")).strip()
        op = str(rule.get("op", "")).strip()
        rid = str(rule.get("id", "")).strip() or "rule"
        if not left or not right or op != "<=":
            continue
        if left not in overrides or right not in overrides:
            continue
        try:
            lv = _to_float(overrides[left])
            rv = _to_float(overrides[right])
        except Exception:
            out.append(f"{rid}: cannot compare {left} and {right}")
            continue
        if lv > rv:
            out.append(f"{rid}: {left} ({lv}) must be <= {right} ({rv})")
    return out


def validate_overrides(overrides: dict[str, str], contract: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    allow = contract.get("allowed_keys", {}) or {}
    protected = {str(k).strip() for k in (contract.get("protected_keys") or []) if str(k).strip()}
    blocked_prefixes = [str(x).strip() for x in (contract.get("blocked_prefixes") or []) if str(x).strip()]

    for k, v in overrides.items():
        key = str(k or "").strip()
        if not key:
            issues.append("Empty override key is not allowed.")
            continue
        if key in protected:
            issues.append(f"{key}: protected key, editing is blocked")
            continue
        if any(key.startswith(pref) for pref in blocked_prefixes):
            issues.append(f"{key}: blocked by prefix policy")
            continue
        spec = allow.get(key)
        if not isinstance(spec, dict):
            issues.append(f"{key}: key is outside safe allow-list")
            continue
        scalar_issue = _validate_scalar(key, v, spec)
        if scalar_issue:
            issues.append(scalar_issue)

    issues.extend(_validate_global_rules(overrides, list(contract.get("global_rules") or [])))
    return issues


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    raise ValueError("JSON root must be an object")


def _result_json(ok: bool, issues: list[str], source: str) -> str:
    return json.dumps({"ok": ok, "source": source, "issues": issues}, ensure_ascii=False, indent=2)


def cmd_describe(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    contract = load_contract(root)
    if args.json:
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return 0
    allow = contract.get("allowed_keys", {}) or {}
    print(f"Safe tuning keys: {len(allow)}")
    for key in sorted(allow.keys()):
        spec = allow[key]
        kind = spec.get("type", "-")
        lo = spec.get("min")
        hi = spec.get("max")
        sensitivity = spec.get("sensitivity", "-")
        importance = spec.get("importance", "-")
        rng = ""
        if lo is not None or hi is not None:
            rng = f" range=[{lo},{hi}]"
        print(f"- {key} type={kind}{rng} sensitivity={sensitivity} importance={importance}")
    return 0


def cmd_validate_overrides(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    contract = load_contract(root)
    payload = json.loads(args.overrides_json or "{}")
    if not isinstance(payload, dict):
        raise SystemExit("overrides_json must be a JSON object")
    overrides = {str(k): str(v) for k, v in payload.items()}
    issues = validate_overrides(overrides, contract)
    ok = len(issues) == 0
    if args.json:
        print(_result_json(ok, issues, "overrides_json"))
    elif not args.quiet:
        if ok:
            print("OK")
        else:
            for row in issues:
                print(f"- {row}")
    return 0 if ok else 2


def cmd_validate_preset_file(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    contract = load_contract(root)
    path = Path(args.path).resolve()
    payload = _read_json(path)
    raw = payload.get("overrides", {})
    if not isinstance(raw, dict):
        raise SystemExit("Preset file has invalid 'overrides' field.")
    overrides = {str(k): str(v) for k, v in raw.items()}
    issues = validate_overrides(overrides, contract)
    ok = len(issues) == 0
    if args.json:
        print(_result_json(ok, issues, str(path)))
    elif not args.quiet:
        if ok:
            print(f"OK: {path}")
        else:
            print(f"FAILED: {path}")
            for row in issues:
                print(f"- {row}")
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Guardrail validator for matrix user preset tuning.")
    p.add_argument("--root", default=".")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_desc = sp.add_parser("describe")
    p_desc.add_argument("--json", action="store_true")
    p_desc.set_defaults(fn=cmd_describe)

    p_ovr = sp.add_parser("validate-overrides")
    p_ovr.add_argument("--root", default=".")
    p_ovr.add_argument("--overrides-json", required=True)
    p_ovr.add_argument("--json", action="store_true")
    p_ovr.add_argument("--quiet", action="store_true")
    p_ovr.set_defaults(fn=cmd_validate_overrides)

    p_file = sp.add_parser("validate-preset-file")
    p_file.add_argument("--root", default=".")
    p_file.add_argument("--path", required=True)
    p_file.add_argument("--json", action="store_true")
    p_file.add_argument("--quiet", action="store_true")
    p_file.set_defaults(fn=cmd_validate_preset_file)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
