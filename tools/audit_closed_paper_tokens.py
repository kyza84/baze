import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from web3 import Web3

BASE_CHAIN_ID_HP = "8453"

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
]

ROUTER_ABI = [
    {
        "name": "getAmountsOut",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "inputs": [{"name": "amountIn", "type": "uint256"}, {"name": "path", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function",
    }
]


@dataclass
class TokenAudit:
    symbol: str
    address: str
    dex_best_liq_usd: float | None
    dex_best_vol5m_usd: float | None
    dex_best_pair: str | None
    dex_flagged: bool | None
    hp_is_honeypot: bool | None
    hp_buy_tax: float | None
    hp_sell_tax: float | None
    hp_status: str
    onchain_buy_ok: bool
    onchain_sell_ok: bool
    roundtrip_ratio: float | None
    notes: list[str]


def _http_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"accept": "application/json", "user-agent": "earnforme-bot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _load_env_if_needed(repo: Path) -> None:
    # Keep the script runnable from a terminal without manual env setup.
    env_path = repo / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k and k not in os.environ:
            os.environ[k] = v


def honeypot_check(address: str) -> tuple[bool | None, float | None, float | None, str]:
    qs = urllib.parse.urlencode({"address": address, "chainID": BASE_CHAIN_ID_HP})
    url = "https://api.honeypot.is/v2/IsHoneypot?" + qs
    try:
        data = _http_json(url)
        hp = data.get("honeypotResult", {}) if isinstance(data, dict) else {}
        tax = data.get("taxResult", {}) if isinstance(data, dict) else {}
        is_hp = hp.get("isHoneypot")
        buy_tax = tax.get("buyTax")
        sell_tax = tax.get("sellTax")
        if is_hp is True:
            status = "HONEYPOT"
        elif is_hp is False:
            status = "not_honeypot"
        else:
            status = "unknown"
        return (
            is_hp if isinstance(is_hp, bool) else None,
            buy_tax if isinstance(buy_tax, (int, float)) else None,
            sell_tax if isinstance(sell_tax, (int, float)) else None,
            status,
        )
    except Exception as e:
        return None, None, None, f"check_failed:{e}"


def dexscreener_best_pair(address: str) -> tuple[float | None, float | None, str | None, bool | None, str]:
    url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
    try:
        data = _http_json(url)
        pairs = data.get("pairs", []) if isinstance(data, dict) else []
        best = None
        for p in pairs:
            if not isinstance(p, dict):
                continue
            if str(p.get("chainId", "")).lower() != "base":
                continue
            liq_usd = ((p.get("liquidity") or {}) or {}).get("usd")
            if not isinstance(liq_usd, (int, float)):
                continue
            cand = {
                "liq_usd": float(liq_usd),
                "vol5m_usd": float((((p.get("volume") or {}) or {}).get("m5")) or 0.0),
                "pair": str(p.get("pairAddress", "")) or None,
                "flagged": bool(p.get("flagged")) if p.get("flagged") is not None else None,
            }
            if best is None or cand["liq_usd"] > best["liq_usd"]:
                best = cand
        if best is None:
            return None, None, None, None, "no_base_pairs"
        return best["liq_usd"], best["vol5m_usd"], best["pair"], best["flagged"], "ok"
    except Exception as e:
        return None, None, None, None, f"check_failed:{e}"


def onchain_roundtrip_quote(
    w3: Web3, router_addr: str, weth_addr: str, token_addr: str, buy_amount_wei: int
) -> tuple[bool, bool, float | None, list[str]]:
    notes: list[str] = []
    router = w3.eth.contract(address=w3.to_checksum_address(router_addr), abi=ROUTER_ABI)
    token = w3.eth.contract(address=w3.to_checksum_address(token_addr), abi=ERC20_ABI)

    try:
        decimals = int(token.functions.decimals().call())
    except Exception as e:
        notes.append(f"decimals_call_failed:{e}")
        decimals = 18

    # Buy quote (WETH -> token)
    try:
        out = router.functions.getAmountsOut(
            int(buy_amount_wei),
            [w3.to_checksum_address(weth_addr), w3.to_checksum_address(token_addr)],
        ).call()
        token_out = int(out[-1])
        onchain_buy_ok = token_out > 0
        if not onchain_buy_ok:
            notes.append("buy_quote_zero")
            return False, False, None, notes
    except Exception as e:
        notes.append(f"buy_quote_failed:{e}")
        return False, False, None, notes

    # Sell quote (token -> WETH) using 10% of buy output.
    sell_in = max(1, token_out // 10)
    min_reasonable = 10 ** max(0, min(6, decimals))  # 1e6 units (or less if decimals < 6)
    if sell_in < min_reasonable:
        sell_in = min_reasonable
        notes.append("sell_amount_bumped")

    try:
        out2 = router.functions.getAmountsOut(
            int(sell_in),
            [w3.to_checksum_address(token_addr), w3.to_checksum_address(weth_addr)],
        ).call()
        weth_out = int(out2[-1])
        onchain_sell_ok = weth_out > 0
        ratio = None
        if onchain_sell_ok and token_out > 0:
            expected_in = float(buy_amount_wei) * (float(sell_in) / float(token_out))
            if expected_in > 0:
                ratio = float(weth_out) / expected_in
        else:
            notes.append("sell_quote_zero")
        return onchain_buy_ok, onchain_sell_ok, ratio, notes
    except Exception as e:
        notes.append(f"sell_quote_failed:{e}")
        return True, False, None, notes


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    _load_env_if_needed(repo)

    state_path = repo / "trading" / "paper_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    closed = state.get("closed_positions", [])

    rpc = os.getenv("RPC_PRIMARY") or ""
    router = os.getenv("LIVE_ROUTER_ADDRESS") or ""
    weth = os.getenv("WETH_ADDRESS") or ""

    print(f"paper_state={state_path}")
    print(f"closed_positions={len(closed)}")
    if not rpc:
        print("rpc_missing: RPC_PRIMARY")
        return 2
    if not (router and weth):
        print("env_missing: LIVE_ROUTER_ADDRESS/WETH_ADDRESS")
        return 2

    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 20}))
    if not w3.is_connected():
        print("rpc_connect_failed")
        return 2

    buy_amount_wei = int(Web3.to_wei(0.00005, "ether"))

    audits: list[TokenAudit] = []
    for p in closed:
        addr = str(p.get("token_address", "")).strip()
        sym = str(p.get("symbol", "N/A"))
        if not addr or addr.lower() == weth.lower():
            continue

        dex_liq, dex_vol5m, dex_pair, dex_flagged, dex_status = dexscreener_best_pair(addr)
        hp_is, hp_buy, hp_sell, hp_status = honeypot_check(addr)
        on_buy, on_sell, rr, notes = onchain_roundtrip_quote(w3, router, weth, addr, buy_amount_wei)
        if dex_status != "ok":
            notes.append(f"dexscreener:{dex_status}")

        audits.append(
            TokenAudit(
                symbol=sym,
                address=addr,
                dex_best_liq_usd=dex_liq,
                dex_best_vol5m_usd=dex_vol5m,
                dex_best_pair=dex_pair,
                dex_flagged=dex_flagged,
                hp_is_honeypot=hp_is,
                hp_buy_tax=hp_buy,
                hp_sell_tax=hp_sell,
                hp_status=hp_status,
                onchain_buy_ok=on_buy,
                onchain_sell_ok=on_sell,
                roundtrip_ratio=rr,
                notes=notes,
            )
        )
        time.sleep(0.2)

    print("--- audit ---")
    for a in audits:
        liq = f"{a.dex_best_liq_usd:.0f}" if isinstance(a.dex_best_liq_usd, (int, float)) else "-"
        vol5 = f"{a.dex_best_vol5m_usd:.0f}" if isinstance(a.dex_best_vol5m_usd, (int, float)) else "-"
        rr = f"{a.roundtrip_ratio:.2f}" if isinstance(a.roundtrip_ratio, float) else "-"
        notes = ";".join(a.notes) if a.notes else "-"
        print(
            f"- {a.symbol} {a.address} dex_liq=${liq} vol5m=${vol5} flagged={a.dex_flagged} "
            f"hp={a.hp_status} buyTax={a.hp_buy_tax} sellTax={a.hp_sell_tax} "
            f"onchain_buy={a.onchain_buy_ok} onchain_sell={a.onchain_sell_ok} rr={rr} notes={notes}"
        )

    print("--- manual links ---")
    for a in audits:
        print(f"- {a.symbol}")
        print(f"  BaseScan: https://basescan.org/token/{a.address}")
        print(f"  DexScreener: https://dexscreener.com/base/{a.address}")
        if a.dex_best_pair:
            print(f"  Pair: https://dexscreener.com/base/{a.dex_best_pair}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
