import json
import urllib.parse
import urllib.request
from pathlib import Path


BASE_CHAIN_ID = "8453"
WETH_BASE = "0x4200000000000000000000000000000000000006"


def honeypot_check(address: str) -> dict:
    qs = urllib.parse.urlencode({"address": address, "chainID": BASE_CHAIN_ID})
    url = "https://api.honeypot.is/v2/IsHoneypot?" + qs
    req = urllib.request.Request(
        url,
        headers={"accept": "application/json", "user-agent": "earnforme-bot/1.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    state_path = Path(__file__).resolve().parents[1] / "trading" / "paper_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    closed = state.get("closed_positions", [])

    print(f"paper_state={state_path}")
    print(f"closed_positions={len(closed)}")
    addrs: list[tuple[str, str]] = []
    for p in closed:
        sym = str(p.get("symbol", "N/A"))
        addr = str(p.get("token_address", "")).strip()
        reason = str(p.get("close_reason", ""))
        pnl = float(p.get("pnl_usd", 0.0) or 0.0)
        print(f"- {sym} {addr} close_reason={reason} pnl_usd={pnl:.4f}")
        if addr and addr.lower() != WETH_BASE.lower():
            addrs.append((sym, addr))

    if not addrs:
        print("no non-WETH tokens to check")
        return 0

    print("--- honeypot.is (Base 8453) ---")
    for sym, addr in addrs:
        try:
            data = honeypot_check(addr)
            hp = data.get("honeypotResult", {}) if isinstance(data, dict) else {}
            sim = data.get("simulationResult", {}) if isinstance(data, dict) else {}
            tax = data.get("taxResult", {}) if isinstance(data, dict) else {}

            flags = []
            if hp.get("isHoneypot") is True:
                flags.append("HONEYPOT")
            if hp.get("honeypotReason"):
                flags.append(str(hp.get("honeypotReason")))

            buy_tax = tax.get("buyTax")
            sell_tax = tax.get("sellTax")
            max_buy = sim.get("maxBuy")
            max_sell = sim.get("maxSell")

            print(
                f"- {sym} {addr} flags={flags or ['ok?']} "
                f"buyTax={buy_tax} sellTax={sell_tax} maxBuy={max_buy} maxSell={max_sell}"
            )
        except Exception as e:
            print(f"- {sym} {addr} honeypot_check_failed={e}")

    print("--- manual links ---")
    for sym, addr in addrs:
        print(f"- {sym}")
        print(f"  BaseScan: https://basescan.org/token/{addr}")
        print(f"  DexScreener: https://dexscreener.com/base/{addr}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

