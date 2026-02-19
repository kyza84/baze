"""Preflight checks for live mode readiness (no trading actions)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from eth_account import Account
from web3 import HTTPProvider, Web3


ROUTER_WETH_ABI: list[dict[str, Any]] = [
    {
        "name": "WETH",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
    }
]


@dataclass
class CheckEvent:
    level: str
    code: str
    message: str


@dataclass
class RpcHealth:
    url: str
    ok: bool = False
    latency_ms: float = 0.0
    chain_id: int | None = None
    block_number: int | None = None
    gas_price_gwei: float | None = None
    error: str = ""


@dataclass
class Report:
    ok: bool = True
    errors: list[CheckEvent] = field(default_factory=list)
    warnings: list[CheckEvent] = field(default_factory=list)
    infos: list[CheckEvent] = field(default_factory=list)
    rpc_nodes: list[RpcHealth] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, code: str, message: str) -> None:
        event = CheckEvent(level=level, code=code, message=message)
        if level == "error":
            self.ok = False
            self.errors.append(event)
        elif level == "warning":
            self.warnings.append(event)
        else:
            self.infos.append(event)


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _fmt_node(url: str) -> str:
    trimmed = url.strip()
    if len(trimmed) <= 42:
        return trimmed
    return f"{trimmed[:20]}...{trimmed[-16:]}"


def _rpc_probe(url: str, timeout_s: float) -> RpcHealth:
    out = RpcHealth(url=url)
    started = time.perf_counter()
    try:
        w3 = Web3(HTTPProvider(url, request_kwargs={"timeout": timeout_s}))
        connected = bool(w3.is_connected())
        elapsed = (time.perf_counter() - started) * 1000.0
        out.latency_ms = round(elapsed, 1)
        if not connected:
            out.error = "not_connected"
            return out
        out.chain_id = int(w3.eth.chain_id)
        out.block_number = int(w3.eth.block_number)
        try:
            out.gas_price_gwei = float(w3.from_wei(int(w3.eth.gas_price), "gwei"))
        except Exception:
            out.gas_price_gwei = None
        out.ok = True
        return out
    except Exception as exc:
        out.latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        out.error = str(exc)
        return out


def run_checks(env_file: Path, rpc_timeout_s: float, max_block_drift: int) -> Report:
    report = Report()

    if not env_file.exists():
        report.add("error", "env_missing", f".env file not found: {env_file}")
        return report

    env = {str(k): str(v) for k, v in dotenv_values(env_file).items() if k is not None and v is not None}
    env_runtime = dict(os.environ)
    for k, v in env.items():
        env_runtime.setdefault(k, v)

    auto_enabled = _truthy(env_runtime.get("AUTO_TRADE_ENABLED", "false"))
    auto_paper = _truthy(env_runtime.get("AUTO_TRADE_PAPER", "true"))
    wallet_mode = str(env_runtime.get("WALLET_MODE", "") or "").strip().lower()
    if not auto_enabled:
        report.add("error", "mode_auto_disabled", "AUTO_TRADE_ENABLED must be true for live readiness.")
    if auto_paper:
        report.add("error", "mode_paper_enabled", "AUTO_TRADE_PAPER must be false for live readiness.")
    if wallet_mode and wallet_mode != "live":
        report.add("warning", "wallet_mode_not_live", f"WALLET_MODE={wallet_mode!r}; expected 'live'.")

    required_non_empty = [
        "LIVE_PRIVATE_KEY",
        "LIVE_WALLET_ADDRESS",
        "WETH_ADDRESS",
    ]
    for key in required_non_empty:
        if not str(env_runtime.get(key, "") or "").strip():
            report.add("error", "missing_key", f"Required key is empty: {key}")

    routers = []
    live_router = str(env_runtime.get("LIVE_ROUTER_ADDRESS", "") or "").strip()
    if live_router:
        routers.append(live_router)
    routers.extend(_split_csv(env_runtime.get("LIVE_ROUTER_ADDRESSES", "")))
    if not routers:
        report.add("error", "router_missing", "LIVE_ROUTER_ADDRESS/LIVE_ROUTER_ADDRESSES is empty.")

    rpc_urls = []
    primary = str(env_runtime.get("RPC_PRIMARY", "") or "").strip()
    secondary = str(env_runtime.get("RPC_SECONDARY", "") or "").strip()
    if primary:
        rpc_urls.append(primary)
    if secondary and secondary not in rpc_urls:
        rpc_urls.append(secondary)
    if not rpc_urls:
        report.add("error", "rpc_missing", "RPC_PRIMARY/RPC_SECONDARY is empty.")

    wallet_raw = str(env_runtime.get("LIVE_WALLET_ADDRESS", "") or "").strip()
    weth_raw = str(env_runtime.get("WETH_ADDRESS", "") or "").strip()
    if wallet_raw and not Web3.is_address(wallet_raw):
        report.add("error", "wallet_invalid", "LIVE_WALLET_ADDRESS is not a valid EVM address.")
    if weth_raw and not Web3.is_address(weth_raw):
        report.add("error", "weth_invalid", "WETH_ADDRESS is not a valid EVM address.")
    for idx, router in enumerate(routers):
        if not Web3.is_address(router):
            report.add("error", "router_invalid", f"Router[{idx}] is not a valid EVM address: {router}")

    private_key = str(env_runtime.get("LIVE_PRIVATE_KEY", "") or "").strip()
    if private_key and wallet_raw:
        try:
            account = Account.from_key(private_key)
            if account.address.lower() != wallet_raw.lower():
                report.add(
                    "error",
                    "wallet_key_mismatch",
                    "LIVE_WALLET_ADDRESS does not match LIVE_PRIVATE_KEY derived address.",
                )
        except Exception as exc:
            report.add("error", "private_key_invalid", f"LIVE_PRIVATE_KEY parse failed: {exc}")

    live_chain_id = _to_int(env_runtime.get("LIVE_CHAIN_ID", env_runtime.get("EVM_CHAIN_ID", "8453")), 8453)
    min_gas_reserve_eth = _to_float(env_runtime.get("LIVE_MIN_GAS_RESERVE_ETH", "0.0007"), 0.0007)
    max_gas_gwei = _to_float(env_runtime.get("LIVE_MAX_GAS_GWEI", "2.0"), 2.0)

    for url in rpc_urls:
        report.rpc_nodes.append(_rpc_probe(url, timeout_s=rpc_timeout_s))

    healthy = [x for x in report.rpc_nodes if x.ok]
    if not healthy:
        report.add("error", "rpc_all_failed", "No healthy RPC nodes.")
    else:
        for node in healthy:
            if node.chain_id != live_chain_id:
                report.add(
                    "error",
                    "rpc_chain_mismatch",
                    f"RPC { _fmt_node(node.url) } chain_id={node.chain_id}, expected {live_chain_id}.",
                )

        blocks = [x.block_number for x in healthy if x.block_number is not None]
        if blocks:
            drift = int(max(blocks) - min(blocks))
            if drift > max(0, max_block_drift):
                report.add("warning", "rpc_block_drift", f"RPC block drift is high: {drift} blocks.")

        latencies = [x.latency_ms for x in healthy]
        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = max(latencies) if latencies else 0.0
        report.summary["rpc_latency_ms_p50"] = round(float(p50), 1)
        report.summary["rpc_latency_ms_p95"] = round(float(p95), 1)

        primary_node = sorted(healthy, key=lambda x: x.latency_ms)[0]
        try:
            w3 = Web3(HTTPProvider(primary_node.url, request_kwargs={"timeout": rpc_timeout_s}))
            wallet = w3.to_checksum_address(wallet_raw)
            balance_wei = int(w3.eth.get_balance(wallet))
            balance_eth = float(w3.from_wei(balance_wei, "ether"))
            report.summary["wallet_balance_eth"] = balance_eth
            report.summary["wallet_address"] = wallet
            if balance_eth < min_gas_reserve_eth:
                report.add(
                    "error",
                    "wallet_low_gas_reserve",
                    f"Wallet balance {balance_eth:.6f} ETH below LIVE_MIN_GAS_RESERVE_ETH={min_gas_reserve_eth:.6f}.",
                )
            else:
                report.add(
                    "info",
                    "wallet_balance_ok",
                    f"Wallet balance {balance_eth:.6f} ETH (reserve floor {min_gas_reserve_eth:.6f}).",
                )

            nonce = int(w3.eth.get_transaction_count(wallet))
            report.summary["wallet_nonce"] = nonce
            report.add("info", "wallet_nonce", f"Wallet nonce: {nonce}.")

            if max_gas_gwei > 0:
                try:
                    current_gas_gwei = float(w3.from_wei(int(w3.eth.gas_price), "gwei"))
                    report.summary["network_gas_gwei"] = current_gas_gwei
                    if current_gas_gwei > (max_gas_gwei * 1.5):
                        report.add(
                            "warning",
                            "gas_above_limit",
                            f"Network gas {current_gas_gwei:.3f} gwei is above LIVE_MAX_GAS_GWEI={max_gas_gwei:.3f}.",
                        )
                except Exception:
                    pass

            for idx, router in enumerate(routers):
                try:
                    router_addr = w3.to_checksum_address(router)
                    contract = w3.eth.contract(address=router_addr, abi=ROUTER_WETH_ABI)
                    weth_out = str(contract.functions.WETH().call())
                    if not Web3.is_address(weth_out):
                        raise ValueError("router WETH() returned non-address")
                    report.add("info", "router_probe_ok", f"Router[{idx}] WETH() probe ok: {router_addr}")
                except Exception as exc:
                    report.add("error", "router_probe_failed", f"Router[{idx}] probe failed: {exc}")
        except Exception as exc:
            report.add("error", "wallet_probe_failed", f"Wallet probe failed: {exc}")

    report.summary["env_file"] = str(env_file)
    report.summary["mode"] = {
        "AUTO_TRADE_ENABLED": auto_enabled,
        "AUTO_TRADE_PAPER": auto_paper,
        "WALLET_MODE": wallet_mode or "<empty>",
    }
    report.summary["rpc_total"] = len(report.rpc_nodes)
    report.summary["rpc_healthy"] = len(healthy)
    return report


def _print_report(report: Report) -> None:
    print("=== LIVE PREFLIGHT CHECK ===")
    print(f"status: {'PASS' if report.ok else 'FAIL'}")
    print("")
    if report.summary:
        print("Summary:")
        for k, v in report.summary.items():
            print(f"- {k}: {v}")
        print("")
    if report.rpc_nodes:
        print("RPC nodes:")
        for node in report.rpc_nodes:
            status = "ok" if node.ok else "fail"
            cid = "-" if node.chain_id is None else str(node.chain_id)
            block = "-" if node.block_number is None else str(node.block_number)
            gas = "-" if node.gas_price_gwei is None else f"{node.gas_price_gwei:.3f}"
            err = f" | err={node.error}" if node.error else ""
            print(
                f"- {status} | { _fmt_node(node.url) } | latency={node.latency_ms:.1f}ms | chain={cid} | block={block} | gas={gas}{err}"
            )
        print("")

    def _emit(title: str, items: list[CheckEvent]) -> None:
        if not items:
            return
        print(f"{title}:")
        for item in items:
            print(f"- [{item.code}] {item.message}")
        print("")

    _emit("Errors", report.errors)
    _emit("Warnings", report.warnings)
    _emit("Info", report.infos)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for live mode readiness.")
    parser.add_argument("--env-file", default=".env", help="Path to env file (default: .env)")
    parser.add_argument("--rpc-timeout", type=float, default=8.0, help="RPC timeout seconds (default: 8)")
    parser.add_argument("--max-block-drift", type=int, default=5, help="Warn if RPC nodes diverge more than this many blocks")
    parser.add_argument("--json-out", default="", help="Optional path to write JSON report")
    args = parser.parse_args()

    report = run_checks(
        env_file=Path(args.env_file),
        rpc_timeout_s=max(1.0, float(args.rpc_timeout)),
        max_block_drift=max(0, int(args.max_block_drift)),
    )
    _print_report(report)

    if args.json_out:
        payload = {
            "ok": report.ok,
            "summary": report.summary,
            "rpc_nodes": [x.__dict__ for x in report.rpc_nodes],
            "errors": [x.__dict__ for x in report.errors],
            "warnings": [x.__dict__ for x in report.warnings],
            "infos": [x.__dict__ for x in report.infos],
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"json_report: {out_path}")

    return 0 if report.ok else 2


if __name__ == "__main__":
    sys.exit(main())

