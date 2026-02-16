"""On-chain live swap executor for Base (UniswapV2-compatible router)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from eth_account import Account
from web3 import HTTPProvider, Web3
from web3.contract import Contract

import config


ERC20_ABI: list[dict[str, Any]] = [
    {
        "name": "decimals",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "owner", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "allowance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "approve",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "spender", "type": "address"}, {"name": "value", "type": "uint256"}],
        "outputs": [{"name": "", "type": "bool"}],
    },
]


ROUTER_ABI: list[dict[str, Any]] = [
    {
        "name": "WETH",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
    },
    {
        "name": "getAmountsOut",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "amountIn", "type": "uint256"}, {"name": "path", "type": "address[]"}],
        "outputs": [{"name": "amounts", "type": "uint256[]"}],
    },
    {
        "name": "swapExactETHForTokensSupportingFeeOnTransferTokens",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "name": "swapExactTokensForETHSupportingFeeOnTransferTokens",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [],
    },
]
ERC20_TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex()


@dataclass
class LiveBuyResult:
    tx_hash: str
    token_amount_raw: int
    spent_eth: float
    balance_before_raw: int = 0
    balance_after_raw: int = 0


@dataclass
class LiveSellResult:
    tx_hash: str
    received_eth: float


class LiveExecutor:
    def __init__(self) -> None:
        if not config.LIVE_PRIVATE_KEY:
            raise ValueError("LIVE_PRIVATE_KEY is empty")
        if not config.LIVE_WALLET_ADDRESS:
            raise ValueError("LIVE_WALLET_ADDRESS is empty")
        if not config.LIVE_ROUTER_ADDRESS:
            raise ValueError("LIVE_ROUTER_ADDRESS is empty")

        rpc = (config.RPC_PRIMARY or "").strip() or (config.RPC_SECONDARY or "").strip()
        if not rpc:
            raise ValueError("RPC_PRIMARY/RPC_SECONDARY is empty")

        self.w3 = Web3(HTTPProvider(rpc, request_kwargs={"timeout": config.RPC_TIMEOUT_SECONDS}))
        if not self.w3.is_connected():
            raise ValueError("Web3 not connected")

        self.account = Account.from_key(config.LIVE_PRIVATE_KEY)
        self.wallet = self.w3.to_checksum_address(config.LIVE_WALLET_ADDRESS)
        if self.account.address.lower() != self.wallet.lower():
            raise ValueError("LIVE_WALLET_ADDRESS does not match LIVE_PRIVATE_KEY")

        self.routers = self._load_routers()
        if not self.routers:
            raise ValueError("No usable LIVE router configured")
        self.router_address, self.router = self.routers[0]
        self.weth = self.w3.to_checksum_address(config.WETH_ADDRESS or self.router.functions.WETH().call())
        self.route_intermediates = self._load_route_intermediates()

    def _load_routers(self) -> list[tuple[str, Contract]]:
        out: list[tuple[str, Contract]] = []
        seen: set[str] = set()
        raw_list = [str(config.LIVE_ROUTER_ADDRESS or "").strip()]
        extra = str(getattr(config, "LIVE_ROUTER_ADDRESSES", "") or "").strip()
        if extra:
            raw_list.extend([x.strip() for x in extra.split(",") if x.strip()])
        for raw in raw_list:
            if not raw:
                continue
            try:
                addr = self.w3.to_checksum_address(raw)
            except Exception:
                continue
            low = addr.lower()
            if low in seen:
                continue
            seen.add(low)
            try:
                router = self.w3.eth.contract(address=addr, abi=ROUTER_ABI)
                # Probe to ensure ABI compatibility.
                _ = router.functions.WETH().call()
                out.append((addr, router))
            except Exception:
                continue
        return out

    def _load_route_intermediates(self) -> list[str]:
        raw = str(getattr(config, "LIVE_ROUTE_INTERMEDIATE_ADDRESSES", "") or "").strip()
        if not raw:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for part in raw.split(","):
            s = str(part or "").strip()
            if not s:
                continue
            try:
                addr = self.w3.to_checksum_address(s)
            except Exception:
                continue
            low = addr.lower()
            if low == self.weth.lower() or low in seen:
                continue
            seen.add(low)
            out.append(addr)
        return out

    def _buy_paths(self, token: str) -> list[list[str]]:
        paths: list[list[str]] = [[self.weth, token]]
        for mid in self.route_intermediates:
            if mid.lower() == token.lower():
                continue
            paths.append([self.weth, mid, token])
        return paths

    def _sell_paths(self, token: str) -> list[list[str]]:
        paths: list[list[str]] = [[token, self.weth]]
        for mid in self.route_intermediates:
            if mid.lower() == token.lower():
                continue
            paths.append([token, mid, self.weth])
        return paths

    def _best_quote(self, amount_in: int, paths: list[list[str]]) -> tuple[bool, str, str, Contract | None, list[str], int]:
        best_out = 0
        best_router_addr = ""
        best_router: Contract | None = None
        best_path: list[str] = []
        last_err = ""
        for router_addr, router in self.routers:
            for path in paths:
                try:
                    amounts = router.functions.getAmountsOut(int(amount_in), path).call()
                    if not isinstance(amounts, (list, tuple)) or len(amounts) < 2:
                        continue
                    out = int(amounts[-1])
                    if out > best_out:
                        best_out = out
                        best_router_addr = router_addr
                        best_router = router
                        best_path = list(path)
                except Exception as exc:
                    last_err = str(exc)
                    continue
        if best_path and best_out > 0:
            return True, "ok", best_router_addr, best_router, best_path, best_out
        if last_err:
            return False, f"quote_failed:{last_err}", "", None, [], 0
        return False, "quote_zero", "", None, [], 0

    def native_balance_eth(self) -> float:
        wei = self.w3.eth.get_balance(self.wallet)
        return float(self.w3.from_wei(wei, "ether"))

    def token_balance_raw(self, token_address: str) -> int:
        token = self.w3.to_checksum_address(token_address)
        token_contract = self.w3.eth.contract(address=token, abi=ERC20_ABI)
        return int(token_contract.functions.balanceOf(self.wallet).call())

    def is_buy_route_supported(self, token_address: str, spend_eth: float) -> tuple[bool, str]:
        try:
            token = self.w3.to_checksum_address(token_address)
        except Exception as exc:
            return False, f"invalid_token_address:{exc}"

        amount_in = int(self.w3.to_wei(max(0.0, float(spend_eth)), "ether"))
        if amount_in <= 0:
            amount_in = 1

        ok, reason, router_addr, _, path, _ = self._best_quote(int(amount_in), self._buy_paths(token))
        if not ok:
            return False, reason
        return True, f"ok:router={router_addr} path={'->'.join(path)}"

    def token_decimals(self, token_address: str) -> int:
        """Best-effort ERC20 decimals() with a safe fallback."""
        try:
            token = self.w3.to_checksum_address(token_address)
        except Exception:
            return 18
        try:
            token_contract = self.w3.eth.contract(address=token, abi=ERC20_ABI)
            dec = int(token_contract.functions.decimals().call())
            if 0 <= dec <= 255:
                return dec
        except Exception:
            pass
        return 18

    def is_sell_route_supported(self, token_address: str, amount_tokens: float = 1.0) -> tuple[bool, str]:
        """Sanity-check that router can quote token->WETH (helps avoid unsellable/no-quote tokens)."""
        try:
            token = self.w3.to_checksum_address(token_address)
        except Exception as exc:
            return False, f"invalid_token_address:{exc}"

        decimals = self.token_decimals(token_address)
        try:
            amt = float(amount_tokens)
        except Exception:
            amt = 1.0
        if amt <= 0:
            amt = 1.0
        # Clamp decimals to avoid huge pow in pathological cases.
        decimals = int(max(0, min(36, int(decimals))))
        amount_in = int(amt * (10**decimals))
        if amount_in <= 0:
            amount_in = 1

        ok, reason, router_addr, _, path, _ = self._best_quote(int(amount_in), self._sell_paths(token))
        if not ok:
            return False, reason
        return True, f"ok:router={router_addr} path={'->'.join(path)}"

    def roundtrip_quote(self, token_address: str, spend_eth: float) -> tuple[bool, str, float]:
        """
        Quote WETH->token for the intended spend size, then quote token->WETH for a fraction of that output.
        This helps filter out ultra-thin liquidity where selling is effectively impossible for our size.
        Note: this cannot detect transfer taxes/honeypots reliably (use honeypot guard for that).
        """
        try:
            token = self.w3.to_checksum_address(token_address)
        except Exception as exc:
            return False, f"invalid_token_address:{exc}", 0.0

        try:
            amount_in_wei = int(self.w3.to_wei(max(0.0, float(spend_eth)), "ether"))
        except Exception:
            amount_in_wei = 0
        if amount_in_wei <= 0:
            return False, "invalid_amount_in", 0.0

        try:
            sell_fraction = float(getattr(config, "LIVE_ROUNDTRIP_SELL_FRACTION", 0.25) or 0.25)
        except Exception:
            sell_fraction = 0.25
        sell_fraction = max(0.01, min(1.0, sell_fraction))

        try:
            buy_ok, buy_reason, buy_router_addr, _, buy_path, token_out = self._best_quote(int(amount_in_wei), self._buy_paths(token))
            if not buy_ok:
                return False, f"buy_{buy_reason}", 0.0

            sell_amount_in = int(max(1, int(token_out * sell_fraction)))
            sell_ok, sell_reason, sell_router_addr, _, sell_path, weth_out = self._best_quote(int(sell_amount_in), self._sell_paths(token))
            if not sell_ok:
                return False, f"sell_{sell_reason}", 0.0

            # Compare to the proportional input size for the fraction we try to sell back.
            denom = float(amount_in_wei) * float(sell_fraction)
            ratio = float(weth_out) / denom if denom > 0 else 0.0
            if ratio <= 0:
                return False, "roundtrip_ratio_zero", 0.0
            return True, (
                f"ok:buy_router={buy_router_addr} buy={'->'.join(buy_path)} "
                f"sell_router={sell_router_addr} sell={'->'.join(sell_path)}"
            ), ratio
        except Exception as exc:
            return False, f"roundtrip_quote_failed:{exc}", 0.0

    def quote_sell_eth(self, token_address: str, token_amount_raw: int) -> float:
        """Best-effort quote for token->WETH output in ETH units."""
        try:
            token = self.w3.to_checksum_address(token_address)
        except Exception:
            return 0.0
        amount_in = int(token_amount_raw or 0)
        if amount_in <= 0:
            return 0.0
        ok, _reason, _router_addr, _router, _path, out_wei = self._best_quote(amount_in, self._sell_paths(token))
        if not ok or int(out_wei or 0) <= 0:
            return 0.0
        try:
            return float(self.w3.from_wei(int(out_wei), "ether"))
        except Exception:
            return 0.0

    def buy_token(self, token_address: str, spend_eth: float) -> LiveBuyResult:
        token = self.w3.to_checksum_address(token_address)
        amount_in = int(self.w3.to_wei(spend_eth, "ether"))
        if amount_in <= 0:
            raise ValueError("amount_in is zero")
        ok_path, path_reason, router_addr, router, path, _ = self._best_quote(int(amount_in), self._buy_paths(token))
        if (not ok_path) or router is None:
            raise RuntimeError(f"unsupported_route:{path_reason}")
        amount_out_min = self._estimate_amount_out_min(router, amount_in, path)
        token_contract = self.w3.eth.contract(address=token, abi=ERC20_ABI)
        balance_before = self._read_token_balance_raw(token_contract)

        tx = router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
            amount_out_min,
            path,
            self.wallet,
            self._deadline(),
        ).build_transaction(self._tx_params(value_wei=amount_in))
        tx_hash, receipt = self._send_and_wait_with_receipt(tx)

        balance_after = self._read_token_balance_raw(token_contract)
        if balance_after <= balance_before:
            balance_after = self._wait_buy_balance_after(token_contract, balance_before)
        bought_raw = max(0, balance_after - balance_before)
        if bought_raw <= 0:
            # Fallback: recover bought amount from ERC20 Transfer logs when balanceOf
            # is stale/inconsistent right after the swap.
            bought_raw = self._extract_received_from_receipt(receipt, token)
        if bought_raw <= 0 and balance_after <= balance_before:
            # Final retry after receipt parse, some RPC nodes lag just a little longer.
            balance_after = self._wait_buy_balance_after(token_contract, balance_before)
            bought_raw = max(0, balance_after - balance_before)
        return LiveBuyResult(
            tx_hash=tx_hash,
            token_amount_raw=bought_raw,
            spent_eth=spend_eth,
            balance_before_raw=int(balance_before),
            balance_after_raw=int(balance_after),
        )

    def sell_token(self, token_address: str, token_amount_raw: int) -> LiveSellResult:
        token = self.w3.to_checksum_address(token_address)
        if token_amount_raw <= 0:
            raise ValueError("token_amount_raw is zero")

        token_contract = self.w3.eth.contract(address=token, abi=ERC20_ABI)
        ok_path, path_reason, router_addr, router, path, _ = self._best_quote(int(token_amount_raw), self._sell_paths(token))
        if (not ok_path) or router is None:
            raise RuntimeError(f"unsupported_sell_route:{path_reason}")
        self._ensure_allowance(token_contract, router_addr, token_amount_raw)
        amount_out_min = self._estimate_amount_out_min(router, token_amount_raw, path)
        eth_before = int(self.w3.eth.get_balance(self.wallet))

        tx = router.functions.swapExactTokensForETHSupportingFeeOnTransferTokens(
            int(token_amount_raw),
            amount_out_min,
            path,
            self.wallet,
            self._deadline(),
        ).build_transaction(self._tx_params())
        tx_hash, receipt = self._send_and_wait_with_receipt(tx)

        eth_after = int(self.w3.eth.get_balance(self.wallet))
        received_wei_net = max(0, eth_after - eth_before)
        gas_paid_wei = self._receipt_gas_paid_wei(receipt)
        # Wallet delta is net of gas. For PnL we need gross swap proceeds.
        received_wei = max(0, received_wei_net + gas_paid_wei)
        received_eth = float(self.w3.from_wei(int(received_wei), "ether"))
        return LiveSellResult(tx_hash=tx_hash, received_eth=received_eth)

    def _ensure_allowance(self, token_contract: Contract, router_address: str, required_amount: int) -> None:
        allowance = int(token_contract.functions.allowance(self.wallet, router_address).call())
        if allowance >= required_amount:
            return
        approve_tx = token_contract.functions.approve(router_address, (2**256) - 1).build_transaction(
            self._tx_params()
        )
        self._send_and_wait(approve_tx)

    def _estimate_amount_out_min(self, router: Contract, amount_in: int, path: list[str]) -> int:
        try:
            amounts = router.functions.getAmountsOut(int(amount_in), path).call()
            quoted_out = int(amounts[-1])
            slip = max(1, int(config.LIVE_SLIPPAGE_BPS))
            out_min = int(quoted_out * (10_000 - slip) / 10_000)
            return max(1, out_min)
        except Exception:
            return 1

    def _deadline(self) -> int:
        return int(time.time()) + int(config.LIVE_SWAP_DEADLINE_SECONDS)

    def _tx_params(self, value_wei: int = 0) -> dict[str, Any]:
        pending_nonce = self.w3.eth.get_transaction_count(self.wallet, "pending")
        latest = self.w3.eth.get_block("latest")
        base_fee = int(latest.get("baseFeePerGas") or 0)
        priority = int(self.w3.to_wei(max(0.0, float(config.LIVE_PRIORITY_FEE_GWEI)), "gwei"))
        cap = int(self.w3.to_wei(max(0.0, float(config.LIVE_MAX_GAS_GWEI)), "gwei"))
        if cap <= 0:
            # Hard fail-safe: never send a live tx with an unbounded fee cap.
            cap = int(self.w3.to_wei(1, "gwei"))

        # Some RPCs/networks can behave weirdly with baseFeePerGas. Always sanity-check
        # current suggested gas price and refuse to send if it exceeds the configured cap.
        observed_gas_price = int(self.w3.eth.gas_price or 0)
        if observed_gas_price > cap:
            obs_gwei = float(self.w3.from_wei(observed_gas_price, "gwei"))
            cap_gwei = float(self.w3.from_wei(cap, "gwei"))
            raise RuntimeError(f"gas_price_too_high observed_gwei={obs_gwei:.3f} cap_gwei={cap_gwei:.3f}")

        # EIP-1559 style: compute a max fee, but keep it <= cap.
        # Also keep it >= observed gas price so the tx isn't immediately underpriced.
        max_fee = min(cap, max(observed_gas_price, (base_fee * 2) + priority))
        if max_fee <= 0:
            max_fee = min(cap, int(self.w3.to_wei(1, "gwei")))

        return {
            "from": self.wallet,
            "chainId": int(config.LIVE_CHAIN_ID),
            "nonce": pending_nonce,
            "value": int(value_wei),
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": min(priority, max_fee),
            "type": 2,
        }

    def _send_and_wait(self, tx: dict[str, Any]) -> str:
        tx_hash, _ = self._send_and_wait_with_receipt(tx)
        return tx_hash

    def _send_and_wait_with_receipt(self, tx: dict[str, Any]) -> tuple[str, Any]:
        gas = self.w3.eth.estimate_gas(tx)
        gas_cap = int(getattr(config, "LIVE_MAX_SWAP_GAS", 0) or 0)
        gas_limit = int(gas * 1.15)
        if gas_cap > 0 and int(gas_limit) > gas_cap:
            raise RuntimeError(f"gas_estimate_too_high gas={int(gas_limit)} cap={gas_cap}")
        tx["gas"] = int(gas_limit)

        # Preflight: ensure we can afford worst-case maxFeePerGas * gas_limit + value.
        # Without this, the node will reject with "insufficient funds for gas * price + value".
        bal = int(self.w3.eth.get_balance(self.wallet))
        max_fee = int(tx.get("maxFeePerGas") or 0)
        value = int(tx.get("value") or 0)
        worst_cost = (int(tx["gas"]) * max_fee) + value
        # Base has additional L1 data fee; include a small buffer so we fail-safe.
        buffered = int(worst_cost * 1.20)
        if buffered > bal:
            have_eth = float(self.w3.from_wei(bal, "ether"))
            want_eth = float(self.w3.from_wei(buffered, "ether"))
            fee_gwei = float(self.w3.from_wei(max_fee, "gwei")) if max_fee > 0 else 0.0
            raise RuntimeError(
                f"insufficient_balance_for_tx have_eth={have_eth:.8f} want_eth={want_eth:.8f} "
                f"gas={int(tx['gas'])} maxFee_gwei={fee_gwei:.3f} value_wei={value}"
            )
        signed = self.account.sign_transaction(tx)
        raw_tx = getattr(signed, "raw_transaction", None)
        if raw_tx is None:
            raw_tx = getattr(signed, "rawTransaction", None)
        if raw_tx is None:
            raise RuntimeError("signed_tx_missing_raw_bytes")
        tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=int(config.LIVE_TX_TIMEOUT_SECONDS))
        if int(receipt.status) != 1:
            raise RuntimeError(f"tx_failed hash={tx_hash.hex()}")
        return tx_hash.hex(), receipt

    @staticmethod
    def _receipt_gas_paid_wei(receipt: Any) -> int:
        """Return actual paid gas in wei from transaction receipt."""
        def _as_int(value: Any) -> int:
            if value is None:
                return 0
            try:
                if isinstance(value, str):
                    return int(value, 0)
                return int(value)
            except Exception:
                return 0

        try:
            gas_used = _as_int(getattr(receipt, "gasUsed", 0) or 0)
        except Exception:
            gas_used = 0
        if gas_used <= 0 and isinstance(receipt, dict):
            try:
                gas_used = _as_int(receipt.get("gasUsed") or 0)
            except Exception:
                gas_used = 0

        try:
            eff_price = _as_int(getattr(receipt, "effectiveGasPrice", 0) or 0)
        except Exception:
            eff_price = 0
        if eff_price <= 0 and isinstance(receipt, dict):
            try:
                eff_price = _as_int(receipt.get("effectiveGasPrice") or 0)
            except Exception:
                eff_price = 0
        if eff_price <= 0:
            try:
                eff_price = _as_int(getattr(receipt, "gasPrice", 0) or 0)
            except Exception:
                eff_price = 0
        if eff_price <= 0 and isinstance(receipt, dict):
            try:
                eff_price = _as_int(receipt.get("gasPrice") or 0)
            except Exception:
                eff_price = 0

        if gas_used <= 0 or eff_price <= 0:
            return 0
        return int(gas_used * eff_price)

    @staticmethod
    def _to_hex_str(value: Any) -> str:
        try:
            out = Web3.to_hex(value)
            if isinstance(out, str):
                return out.lower()
        except Exception:
            pass
        text = str(value or "").strip().lower()
        return text if text.startswith("0x") else ""

    def _read_token_balance_raw(self, token_contract: Contract) -> int:
        try:
            return int(token_contract.functions.balanceOf(self.wallet).call())
        except Exception:
            return 0

    def _wait_buy_balance_after(self, token_contract: Contract, balance_before: int) -> int:
        attempts = int(getattr(config, "LIVE_BUY_BALANCE_RECHECK_ATTEMPTS", 8) or 8)
        delay = float(getattr(config, "LIVE_BUY_BALANCE_RECHECK_DELAY_SECONDS", 0.30) or 0.30)
        attempts = max(1, attempts)
        delay = max(0.05, delay)
        latest = int(balance_before or 0)
        for idx in range(attempts):
            latest = self._read_token_balance_raw(token_contract)
            if latest > int(balance_before):
                return latest
            if idx < (attempts - 1):
                time.sleep(delay)
        return latest

    def _extract_received_from_receipt(self, receipt: Any, token_address: str) -> int:
        """Best-effort parse of ERC20 Transfer logs to wallet for a target token."""
        try:
            token_low = str(token_address or "").lower()
            wallet_topic = "0x" + ("0" * 24) + self.wallet.lower().replace("0x", "")
            logs: list[Any] = []
            if isinstance(receipt, dict):
                logs = list(receipt.get("logs") or [])
            else:
                logs = list(getattr(receipt, "logs", []) or [])
            total = 0
            for lg in logs:
                if isinstance(lg, dict):
                    addr = str(lg.get("address", "") or "").lower()
                    topics = [self._to_hex_str(x) for x in (lg.get("topics") or [])]
                    data_hex = self._to_hex_str(lg.get("data", "0x0")) or "0x0"
                else:
                    addr = str(getattr(lg, "address", "") or "").lower()
                    topics = [self._to_hex_str(x) for x in (getattr(lg, "topics", []) or [])]
                    data_hex = self._to_hex_str(getattr(lg, "data", "0x0")) or "0x0"
                if addr != token_low or len(topics) < 3:
                    continue
                if str(topics[0]).lower() != ERC20_TRANSFER_TOPIC.lower():
                    continue
                if str(topics[2]).lower() != wallet_topic.lower():
                    continue
                try:
                    val = int(data_hex, 16)
                except Exception:
                    val = 0
                if val > 0:
                    total += int(val)
            return int(max(0, total))
        except Exception:
            return 0
