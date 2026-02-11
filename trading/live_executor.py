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


@dataclass
class LiveBuyResult:
    tx_hash: str
    token_amount_raw: int
    spent_eth: float


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

        self.router_address = self.w3.to_checksum_address(config.LIVE_ROUTER_ADDRESS)
        self.router: Contract = self.w3.eth.contract(address=self.router_address, abi=ROUTER_ABI)
        self.weth = self.w3.to_checksum_address(config.WETH_ADDRESS or self.router.functions.WETH().call())

    def native_balance_eth(self) -> float:
        wei = self.w3.eth.get_balance(self.wallet)
        return float(self.w3.from_wei(wei, "ether"))

    def is_buy_route_supported(self, token_address: str, spend_eth: float) -> tuple[bool, str]:
        try:
            token = self.w3.to_checksum_address(token_address)
        except Exception as exc:
            return False, f"invalid_token_address:{exc}"

        amount_in = int(self.w3.to_wei(max(0.0, float(spend_eth)), "ether"))
        if amount_in <= 0:
            amount_in = 1

        path = [self.weth, token]
        try:
            amounts = self.router.functions.getAmountsOut(int(amount_in), path).call()
            if not isinstance(amounts, (list, tuple)) or len(amounts) < 2:
                return False, "quote_empty"
            if int(amounts[-1]) <= 0:
                return False, "quote_zero"
            return True, "ok"
        except Exception as exc:
            return False, f"quote_failed:{exc}"

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

        path = [token, self.weth]
        try:
            amounts = self.router.functions.getAmountsOut(int(amount_in), path).call()
            if not isinstance(amounts, (list, tuple)) or len(amounts) < 2:
                return False, "quote_empty"
            if int(amounts[-1]) <= 0:
                return False, "quote_zero"
            return True, "ok"
        except Exception as exc:
            return False, f"quote_failed:{exc}"

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

        path_buy = [self.weth, token]
        path_sell = [token, self.weth]
        try:
            buy_amounts = self.router.functions.getAmountsOut(int(amount_in_wei), path_buy).call()
            if not isinstance(buy_amounts, (list, tuple)) or len(buy_amounts) < 2:
                return False, "buy_quote_empty", 0.0
            token_out = int(buy_amounts[-1])
            if token_out <= 0:
                return False, "buy_quote_zero", 0.0

            sell_amount_in = int(max(1, int(token_out * sell_fraction)))
            sell_amounts = self.router.functions.getAmountsOut(int(sell_amount_in), path_sell).call()
            if not isinstance(sell_amounts, (list, tuple)) or len(sell_amounts) < 2:
                return False, "sell_quote_empty", 0.0
            weth_out = int(sell_amounts[-1])
            if weth_out <= 0:
                return False, "sell_quote_zero", 0.0

            # Compare to the proportional input size for the fraction we try to sell back.
            denom = float(amount_in_wei) * float(sell_fraction)
            ratio = float(weth_out) / denom if denom > 0 else 0.0
            if ratio <= 0:
                return False, "roundtrip_ratio_zero", 0.0
            return True, "ok", ratio
        except Exception as exc:
            return False, f"roundtrip_quote_failed:{exc}", 0.0

    def buy_token(self, token_address: str, spend_eth: float) -> LiveBuyResult:
        token = self.w3.to_checksum_address(token_address)
        amount_in = int(self.w3.to_wei(spend_eth, "ether"))
        if amount_in <= 0:
            raise ValueError("amount_in is zero")

        route_ok, route_reason = self.is_buy_route_supported(token_address, spend_eth)
        if not route_ok:
            raise RuntimeError(f"unsupported_route:{route_reason}")

        path = [self.weth, token]
        amount_out_min = self._estimate_amount_out_min(amount_in, path)
        token_contract = self.w3.eth.contract(address=token, abi=ERC20_ABI)
        balance_before = int(token_contract.functions.balanceOf(self.wallet).call())

        tx = self.router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
            amount_out_min,
            path,
            self.wallet,
            self._deadline(),
        ).build_transaction(self._tx_params(value_wei=amount_in))
        tx_hash = self._send_and_wait(tx)

        balance_after = int(token_contract.functions.balanceOf(self.wallet).call())
        bought_raw = max(0, balance_after - balance_before)
        return LiveBuyResult(tx_hash=tx_hash, token_amount_raw=bought_raw, spent_eth=spend_eth)

    def sell_token(self, token_address: str, token_amount_raw: int) -> LiveSellResult:
        token = self.w3.to_checksum_address(token_address)
        if token_amount_raw <= 0:
            raise ValueError("token_amount_raw is zero")

        token_contract = self.w3.eth.contract(address=token, abi=ERC20_ABI)
        self._ensure_allowance(token_contract, token_amount_raw)

        path = [token, self.weth]
        amount_out_min = self._estimate_amount_out_min(token_amount_raw, path)
        eth_before = int(self.w3.eth.get_balance(self.wallet))

        tx = self.router.functions.swapExactTokensForETHSupportingFeeOnTransferTokens(
            int(token_amount_raw),
            amount_out_min,
            path,
            self.wallet,
            self._deadline(),
        ).build_transaction(self._tx_params())
        tx_hash = self._send_and_wait(tx)

        eth_after = int(self.w3.eth.get_balance(self.wallet))
        received_wei = max(0, eth_after - eth_before)
        received_eth = float(self.w3.from_wei(received_wei, "ether"))
        return LiveSellResult(tx_hash=tx_hash, received_eth=received_eth)

    def _ensure_allowance(self, token_contract: Contract, required_amount: int) -> None:
        allowance = int(token_contract.functions.allowance(self.wallet, self.router_address).call())
        if allowance >= required_amount:
            return
        approve_tx = token_contract.functions.approve(self.router_address, (2**256) - 1).build_transaction(
            self._tx_params()
        )
        self._send_and_wait(approve_tx)

    def _estimate_amount_out_min(self, amount_in: int, path: list[str]) -> int:
        try:
            amounts = self.router.functions.getAmountsOut(int(amount_in), path).call()
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
        return tx_hash.hex()
