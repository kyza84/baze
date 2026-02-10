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

    def buy_token(self, token_address: str, spend_eth: float) -> LiveBuyResult:
        token = self.w3.to_checksum_address(token_address)
        amount_in = int(self.w3.to_wei(spend_eth, "ether"))
        if amount_in <= 0:
            raise ValueError("amount_in is zero")

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
        max_fee = (base_fee * 2) + priority
        cap = int(self.w3.to_wei(max(0.1, float(config.LIVE_MAX_GAS_GWEI)), "gwei"))
        if max_fee > cap:
            max_fee = cap
        if max_fee <= 0:
            max_fee = int(self.w3.to_wei(1, "gwei"))

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
        tx["gas"] = int(gas * 1.15)
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=int(config.LIVE_TX_TIMEOUT_SECONDS))
        if int(receipt.status) != 1:
            raise RuntimeError(f"tx_failed hash={tx_hash.hex()}")
        return tx_hash.hex()
