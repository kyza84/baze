"""On-chain PairCreated monitor for Base factory contracts."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import aiohttp
from web3 import HTTPProvider, Web3

import config

logger = logging.getLogger(__name__)


class OnChainRPCError(RuntimeError):
    """Raised when RPC operations fail after retries."""


@dataclass
class PairCandidate:
    chain: str
    dex: str
    pair_address: str
    token0: str
    token1: str
    base_token: str
    created_block: int
    detected_ts: str
    factory_address: str


class OnChainFactoryMonitor:
    def __init__(self) -> None:
        self.weth_address = str(config.WETH_ADDRESS or "").lower()
        self.v2_factory_address = str(config.BASE_FACTORY_ADDRESS or "").lower()
        self.v2_pair_created_topic = str(config.PAIR_CREATED_TOPIC or "").lower()
        self.enable_uniswap_v3 = bool(config.ONCHAIN_ENABLE_UNISWAP_V3)
        self.v3_factory_address = str(config.UNISWAP_V3_FACTORY_ADDRESS or "").lower()
        self.v3_pool_created_topic = str(config.UNISWAP_V3_POOL_CREATED_TOPIC or "").lower()
        self.last_block_file = os.path.abspath(config.ONCHAIN_LAST_BLOCK_FILE)
        self.seen_pairs_file = os.path.abspath(config.ONCHAIN_SEEN_PAIRS_FILE)
        self.seen_pair_ttl_seconds = int(config.ONCHAIN_SEEN_PAIR_TTL_SECONDS)
        self.finality_blocks = int(config.ONCHAIN_FINALITY_BLOCKS)
        self.log_sources = self._build_log_sources()
        self.providers = [p for p in [config.RPC_PRIMARY, config.RPC_SECONDARY] if p]
        self.provider_index = 0
        self.web3 = self._build_web3()
        self.last_processed_block = self._load_last_block()
        self.seen_pairs = self._load_seen_pairs()

        self._weth_price_usd = 0.0
        self._weth_price_ts = 0.0

    def _build_web3(self) -> Web3:
        if not self.providers:
            raise OnChainRPCError("RPC_PRIMARY/RPC_SECONDARY are not configured for on-chain source.")
        provider = self.providers[self.provider_index]
        return Web3(
            HTTPProvider(
                provider,
                request_kwargs={"timeout": config.RPC_TIMEOUT_SECONDS},
            )
        )

    def _rotate_provider(self) -> None:
        if len(self.providers) <= 1:
            return
        self.provider_index = (self.provider_index + 1) % len(self.providers)
        self.web3 = self._build_web3()

    def _load_last_block(self) -> int | None:
        if not os.path.exists(self.last_block_file):
            return None
        try:
            with open(self.last_block_file, "r", encoding="ascii") as f:
                raw = f.read().strip()
            return int(raw)
        except Exception:
            return None

    def _save_last_block(self, block_number: int) -> None:
        directory = os.path.dirname(self.last_block_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.last_block_file, "w", encoding="ascii") as f:
            f.write(str(int(block_number)))

    def _load_seen_pairs(self) -> dict[str, float]:
        if not os.path.exists(self.seen_pairs_file):
            return {}
        try:
            with open(self.seen_pairs_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            out: dict[str, float] = {}
            for key, value in (payload or {}).items():
                try:
                    out[str(key).lower()] = float(value)
                except (TypeError, ValueError):
                    continue
            return out
        except Exception:
            return {}

    def _save_seen_pairs(self) -> None:
        directory = os.path.dirname(self.seen_pairs_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.seen_pairs_file, "w", encoding="utf-8") as f:
            json.dump(self.seen_pairs, f, ensure_ascii=False)

    def _prune_seen_pairs(self) -> None:
        now_ts = datetime.now(timezone.utc).timestamp()
        ttl = max(60, int(self.seen_pair_ttl_seconds))
        self.seen_pairs = {
            pair: ts
            for pair, ts in self.seen_pairs.items()
            if (now_ts - float(ts)) <= ttl
        }

    def _is_seen_pair(self, pair_address: str) -> bool:
        self._prune_seen_pairs()
        return pair_address in self.seen_pairs

    def _mark_pair_seen(self, pair_address: str) -> None:
        self._prune_seen_pairs()
        self.seen_pairs[pair_address] = datetime.now(timezone.utc).timestamp()

    async def _rpc_with_backoff(self, call: Callable[[], Any], op_name: str) -> Any:
        delays = [1, 2, 4]
        last_error: Exception | None = None
        for attempt, delay in enumerate(delays, start=1):
            try:
                return await asyncio.to_thread(call)
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                last_error = exc
                if attempt < len(delays):
                    self._rotate_provider()
                    await asyncio.sleep(delay)
        raise OnChainRPCError(f"{op_name} failed after retries: {last_error}")

    @staticmethod
    def _topic_to_address(topic: Any) -> str:
        hex_topic = topic.hex() if hasattr(topic, "hex") else str(topic)
        clean = hex_topic.lower().replace("0x", "").rjust(64, "0")
        return f"0x{clean[-40:]}"

    @staticmethod
    def _parse_pair_address_from_data(data: Any) -> str:
        data_hex = data.hex() if hasattr(data, "hex") else str(data)
        clean = data_hex.lower().replace("0x", "")
        if len(clean) < 64:
            return ""
        first_slot = clean[:64]
        return f"0x{first_slot[-40:]}"

    @staticmethod
    def _parse_slot_address(clean_data_hex: str, slot_index: int) -> str:
        start = slot_index * 64
        end = start + 64
        if len(clean_data_hex) < end:
            return ""
        slot = clean_data_hex[start:end]
        address = f"0x{slot[-40:]}"
        if address == "0x0000000000000000000000000000000000000000":
            return ""
        return address

    def _build_log_sources(self) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        if self.v2_factory_address and self.v2_pair_created_topic:
            sources.append(
                {
                    "name": "uniswap_v2",
                    "dex": "uniswap_v2",
                    "factory": self.v2_factory_address,
                    "topic": self.v2_pair_created_topic,
                    "kind": "v2",
                }
            )
        if self.enable_uniswap_v3 and self.v3_factory_address and self.v3_pool_created_topic:
            sources.append(
                {
                    "name": "uniswap_v3",
                    "dex": "uniswap_v3",
                    "factory": self.v3_factory_address,
                    "topic": self.v3_pool_created_topic,
                    "kind": "v3",
                }
            )
        return sources

    def _parse_v2_pair_created_log(self, row: Any, source: dict[str, str]) -> PairCandidate | None:
        topics = list(getattr(row, "topics", []) or row.get("topics", []))
        if len(topics) < 3:
            return None
        topic0 = topics[0].hex().lower() if hasattr(topics[0], "hex") else str(topics[0]).lower()
        if topic0 != source["topic"]:
            return None

        token0 = self._topic_to_address(topics[1]).lower()
        token1 = self._topic_to_address(topics[2]).lower()
        pair_address = self._parse_pair_address_from_data(getattr(row, "data", row.get("data", ""))).lower()
        if not pair_address:
            return None
        if self.weth_address not in {token0, token1}:
            return None

        block_number = int(getattr(row, "blockNumber", row.get("blockNumber", 0)))
        detected_ts = datetime.now(timezone.utc).isoformat()
        factory_address = str(getattr(row, "address", row.get("address", source["factory"]))).lower()

        return PairCandidate(
            chain="base",
            dex=source["dex"],
            pair_address=pair_address,
            token0=token0,
            token1=token1,
            base_token="WETH",
            created_block=block_number,
            detected_ts=detected_ts,
            factory_address=factory_address,
        )

    def _parse_v3_pool_created_log(self, row: Any, source: dict[str, str]) -> PairCandidate | None:
        topics = list(getattr(row, "topics", []) or row.get("topics", []))
        if len(topics) < 3:
            return None
        topic0 = topics[0].hex().lower() if hasattr(topics[0], "hex") else str(topics[0]).lower()
        if topic0 != source["topic"]:
            return None

        token0 = self._topic_to_address(topics[1]).lower()
        token1 = self._topic_to_address(topics[2]).lower()
        if self.weth_address not in {token0, token1}:
            return None

        data_hex = (getattr(row, "data", row.get("data", ""))).hex() if hasattr(getattr(row, "data", None), "hex") else str(getattr(row, "data", row.get("data", "")))
        clean = data_hex.lower().replace("0x", "")
        # Uniswap v3 PoolCreated data:
        # slot 0 => tickSpacing (int24), slot 1 => pool (address)
        pair_address = self._parse_slot_address(clean, 1).lower()
        if not pair_address:
            pair_address = self._parse_slot_address(clean, 0).lower()
        if not pair_address:
            return None

        block_number = int(getattr(row, "blockNumber", row.get("blockNumber", 0)))
        detected_ts = datetime.now(timezone.utc).isoformat()
        factory_address = str(getattr(row, "address", row.get("address", source["factory"]))).lower()
        return PairCandidate(
            chain="base",
            dex=source["dex"],
            pair_address=pair_address,
            token0=token0,
            token1=token1,
            base_token="WETH",
            created_block=block_number,
            detected_ts=detected_ts,
            factory_address=factory_address,
        )

    async def advance_cursor_only(self) -> None:
        latest_raw = int(await self._rpc_with_backoff(lambda: self.web3.eth.block_number, "eth_blockNumber"))
        latest_finalized = max(0, latest_raw - self.finality_blocks)
        if self.last_processed_block is None or latest_finalized > self.last_processed_block:
            self.last_processed_block = latest_finalized
            self._save_last_block(self.last_processed_block)
        self._prune_seen_pairs()
        self._save_seen_pairs()

    async def poll_pair_candidates_once(self) -> list[PairCandidate]:
        if not self.weth_address:
            raise OnChainRPCError(
                "WETH_ADDRESS must be set for on-chain source."
            )
        if not self.log_sources:
            raise OnChainRPCError(
                "No on-chain log sources configured. Set BASE_FACTORY_ADDRESS/PAIR_CREATED_TOPIC or enable UNISWAP_V3 source."
            )

        latest_raw = int(await self._rpc_with_backoff(lambda: self.web3.eth.block_number, "eth_blockNumber"))
        latest_block = max(0, latest_raw - self.finality_blocks)
        if self.last_processed_block is None:
            # Start from latest-1 to avoid replaying old history on first launch.
            self.last_processed_block = max(0, latest_block - 1)
            self._save_last_block(self.last_processed_block)
            return []
        if latest_block <= self.last_processed_block:
            return []

        out: list[PairCandidate] = []
        from_block = self.last_processed_block + 1
        to_block = latest_block
        chunk = int(config.ONCHAIN_BLOCK_CHUNK)

        for start in range(from_block, to_block + 1, chunk):
            end = min(to_block, start + chunk - 1)
            for source in self.log_sources:
                params = {
                    "address": Web3.to_checksum_address(source["factory"]),
                    "fromBlock": start,
                    "toBlock": end,
                    "topics": [source["topic"]],
                }
                rows = await self._rpc_with_backoff(
                    lambda p=params: self.web3.eth.get_logs(p),
                    f"eth_getLogs[{source['name']}:{start}-{end}]",
                )
                for row in rows:
                    if source["kind"] == "v3":
                        candidate = self._parse_v3_pool_created_log(row, source)
                    else:
                        candidate = self._parse_v2_pair_created_log(row, source)
                    if not candidate:
                        continue
                    if candidate.factory_address and candidate.factory_address != source["factory"]:
                        continue
                    if self._is_seen_pair(candidate.pair_address):
                        continue
                    self._mark_pair_seen(candidate.pair_address)
                    logger.info(
                        "PAIR_DETECTED source=%s dex=%s pair=%s token0=%s token1=%s block=%s factory=%s",
                        source["name"],
                        candidate.dex,
                        candidate.pair_address,
                        candidate.token0,
                        candidate.token1,
                        candidate.created_block,
                        candidate.factory_address or source["factory"],
                    )
                    out.append(candidate)

            self.last_processed_block = end
            self._save_last_block(self.last_processed_block)
            self._save_seen_pairs()

        return out

    async def _fetch_weth_price_usd(self) -> float:
        now_ts = datetime.now(timezone.utc).timestamp()
        if self._weth_price_usd > 0 and (now_ts - self._weth_price_ts) < 60:
            return self._weth_price_usd
        if not self.weth_address:
            return 0.0

        url = f"{config.DEXSCREENER_API}/tokens/{self.weth_address}"
        timeout = aiohttp.ClientTimeout(total=config.DEX_TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return self._weth_price_usd
                    payload = await response.json()
        except Exception:
            return self._weth_price_usd

        best_liq = -1.0
        best_price = 0.0
        for pair in payload.get("pairs", []) or []:
            if str(pair.get("chainId", "")).lower() != str(config.CHAIN_ID).lower():
                continue
            liq = float((pair.get("liquidity") or {}).get("usd") or 0)
            price = float(pair.get("priceUsd") or 0)
            if price > 0 and liq > best_liq:
                best_liq = liq
                best_price = price
        if best_price > 0:
            self._weth_price_usd = best_price
            self._weth_price_ts = now_ts
        return self._weth_price_usd

    async def _candidate_to_token(self, candidate: PairCandidate) -> dict[str, Any] | None:
        token_address = candidate.token0 if candidate.token1 == self.weth_address else candidate.token1
        url = f"{config.DEXSCREENER_API}/tokens/{token_address}"
        timeout = aiohttp.ClientTimeout(total=config.DEX_TIMEOUT)

        best_pair: dict[str, Any] | None = None
        retries = int(config.ONCHAIN_ENRICH_RETRIES)
        retry_delay = int(config.ONCHAIN_ENRICH_RETRY_DELAY_SECONDS)
        for attempt in range(1, retries + 1):
            payload: dict[str, Any] | None = None
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            payload = await response.json()
            except Exception:
                payload = None

            best_liq = -1.0
            pairs = (payload or {}).get("pairs", []) or []
            for pair in pairs:
                if str(pair.get("chainId", "")).lower() != str(config.CHAIN_ID).lower():
                    continue
                pair_address = str(pair.get("pairAddress", "") or "").lower()
                liq = float((pair.get("liquidity") or {}).get("usd") or 0)
                if pair_address and pair_address == candidate.pair_address:
                    best_pair = pair
                    break
                if liq > best_liq:
                    best_liq = liq
                    best_pair = pair

            if best_pair:
                break
            if attempt < retries:
                logger.info(
                    "ENRICH_RETRY token=%s pair=%s attempt=%s/%s delay=%ss",
                    token_address,
                    candidate.pair_address,
                    attempt,
                    retries,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)

        now = datetime.now(timezone.utc)
        if not best_pair:
            weth_price_usd = await self._fetch_weth_price_usd()
            return {
                "name": "Unknown",
                "symbol": "N/A",
                "address": token_address,
                "liquidity": 0.0,
                "volume_5m": 0.0,
                "price_change_5m": 0.0,
                "price_usd": 0.0,
                "weth_price_usd": weth_price_usd,
                "chain": candidate.chain,
                "dex": candidate.dex,
                "pair_address": candidate.pair_address,
                "token0": candidate.token0,
                "token1": candidate.token1,
                "base_token": candidate.base_token,
                "created_block": candidate.created_block,
                "detected_ts": candidate.detected_ts,
                "created_at": now,
                "age_seconds": 0,
                "age_minutes": 0,
            }

        base_token = best_pair.get("baseToken") or {}
        quote_token = best_pair.get("quoteToken") or {}
        token_symbol = str(base_token.get("symbol") or "N/A")
        token_name = str(base_token.get("name") or "Unknown")
        if str(base_token.get("address", "")).lower() != token_address:
            token_symbol = str(quote_token.get("symbol") or "N/A")
            token_name = str(quote_token.get("name") or "Unknown")

        created_ms = int(best_pair.get("pairCreatedAt") or int(now.timestamp() * 1000))
        created_at = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
        age_seconds = max(0, int((now - created_at).total_seconds()))

        weth_price_usd = 0.0
        price_native = float(best_pair.get("priceNative") or 0)
        price_usd = float(best_pair.get("priceUsd") or 0)
        quote_addr = str(quote_token.get("address") or "").lower()
        base_addr = str(base_token.get("address") or "").lower()
        if price_native > 0 and price_usd > 0 and quote_addr == self.weth_address:
            weth_price_usd = price_usd / price_native
        elif base_addr == self.weth_address and price_usd > 0:
            weth_price_usd = price_usd
        if weth_price_usd <= 0:
            weth_price_usd = await self._fetch_weth_price_usd()

        return {
            "name": token_name,
            "symbol": token_symbol,
            "address": token_address,
            "liquidity": float((best_pair.get("liquidity") or {}).get("usd") or 0),
            "volume_5m": float((best_pair.get("volume") or {}).get("m5") or 0),
            "price_change_5m": float((best_pair.get("priceChange") or {}).get("m5") or 0),
            "price_usd": price_usd,
            "weth_price_usd": weth_price_usd,
            "dexscreener_url": best_pair.get("url", ""),
            "chain": candidate.chain,
            "dex": candidate.dex,
            "pair_address": candidate.pair_address,
            "token0": candidate.token0,
            "token1": candidate.token1,
            "base_token": candidate.base_token,
            "created_block": candidate.created_block,
            "detected_ts": candidate.detected_ts,
            "created_at": created_at,
            "age_seconds": age_seconds,
            "age_minutes": int(round(age_seconds / 60)),
        }

    async def fetch_new_tokens(self) -> list[dict[str, Any]]:
        candidates = await self.poll_pair_candidates_once()
        if not candidates:
            return []
        enriched = await asyncio.gather(*[self._candidate_to_token(candidate) for candidate in candidates])
        return [row for row in enriched if row]


async def _run_once() -> int:
    try:
        monitor = OnChainFactoryMonitor()
        rows = await monitor.poll_pair_candidates_once()
        print(f"On-chain pools detected: {len(rows)}")
        for row in rows[:3]:
            print(
                "dex=%s pair=%s token0=%s token1=%s block=%s factory=%s"
                % (
                    row.dex,
                    row.pair_address,
                    row.token0,
                    row.token1,
                    row.created_block,
                    row.factory_address,
                )
            )
    except Exception as exc:
        print(f"PairCreated detected: 0 (error: {exc})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="On-chain PairCreated monitor")
    parser.add_argument("--once", action="store_true", help="Run one pass and exit")
    args = parser.parse_args()

    if args.once:
        return asyncio.run(_run_once())

    async def _loop() -> int:
        monitor = OnChainFactoryMonitor()
        while True:
            rows = await monitor.poll_pair_candidates_once()
            print(f"PairCreated detected: {len(rows)}")
            await asyncio.sleep(max(1, int(config.ONCHAIN_POLL_INTERVAL_SECONDS)))

    return asyncio.run(_loop())


if __name__ == "__main__":
    raise SystemExit(main())
