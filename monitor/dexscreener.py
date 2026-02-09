"""Token monitor with DexScreener primary source and GeckoTerminal fallback."""

import asyncio
from datetime import datetime, timezone
from typing import Any

import aiohttp

from config import DEXSCREENER_API, DEX_RETRIES, DEX_TIMEOUT, MIN_LIQUIDITY, SEEN_TOKEN_TTL, TOKEN_AGE_MAX

GECKO_NEW_POOLS_URL = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools?page=1&include=base_token"


class DexScreenerMonitor:
    def __init__(self) -> None:
        self.seen_tokens: dict[str, float] = {}
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

    async def fetch_new_tokens(self) -> list[dict[str, Any]]:
        self._prune_seen_tokens()

        tokens = await self._fetch_from_dexscreener()
        if tokens:
            return tokens

        return await self._fetch_from_geckoterminal()

    async def _fetch_from_dexscreener(self) -> list[dict[str, Any]]:
        url = f"{DEXSCREENER_API}/search?q=solana"
        timeout = aiohttp.ClientTimeout(total=DEX_TIMEOUT)

        for attempt in range(1, DEX_RETRIES + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, headers=self._headers) as response:
                        if response.status != 200:
                            if attempt < DEX_RETRIES:
                                await asyncio.sleep(attempt)
                                continue
                            return []
                        data = await response.json()
                        pairs = data.get("pairs", [])
                        return self._filter_dex_pairs(pairs)
            except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ValueError):
                if attempt < DEX_RETRIES:
                    await asyncio.sleep(attempt)
                    continue
                return []

        return []

    async def _fetch_from_geckoterminal(self) -> list[dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(total=DEX_TIMEOUT)

        for attempt in range(1, DEX_RETRIES + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(GECKO_NEW_POOLS_URL, headers=self._headers) as response:
                        if response.status != 200:
                            if attempt < DEX_RETRIES:
                                await asyncio.sleep(attempt)
                                continue
                            return []

                        data = await response.json()
                        pools = data.get("data", [])
                        included = data.get("included", [])
                        return self._filter_gecko_pools(pools, included)
            except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ValueError):
                if attempt < DEX_RETRIES:
                    await asyncio.sleep(attempt)
                    continue
                return []

        return []

    def _filter_dex_pairs(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []

        for pair in pairs:
            if pair.get("chainId") != "solana":
                continue

            base_token = pair.get("baseToken", {})
            address = base_token.get("address")
            if not address:
                continue

            liquidity_usd = float((pair.get("liquidity") or {}).get("usd") or 0)
            created_ms = pair.get("pairCreatedAt")
            if not created_ms:
                continue

            created_at = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
            if not self._passes_filters(address, liquidity_usd, created_at):
                continue

            filtered.append(self._format_dex_token_data(pair, created_at))

        return filtered

    def _filter_gecko_pools(self, pools: list[dict[str, Any]], included: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        token_map = self._build_gecko_token_map(included)

        for pool in pools:
            attrs = pool.get("attributes", {})
            rel = pool.get("relationships", {})
            base_data = (rel.get("base_token") or {}).get("data") or {}
            base_id = base_data.get("id", "")
            base_token = token_map.get(base_id, {})
            address = base_token.get("address", "")
            if not address:
                continue

            liquidity_usd = float(attrs.get("reserve_in_usd") or 0)
            created_at_str = attrs.get("pool_created_at")
            if not created_at_str:
                continue
            created_at = self._parse_rfc3339(created_at_str)
            if not created_at:
                continue

            if not self._passes_filters(address, liquidity_usd, created_at):
                continue

            filtered.append(
                {
                    "name": base_token.get("name") or (attrs.get("name", "Unknown").split("/")[0].strip()),
                    "symbol": base_token.get("symbol") or "N/A",
                    "address": address,
                    "liquidity": liquidity_usd,
                    "volume_5m": float((attrs.get("volume_usd") or {}).get("m5") or 0),
                    "price_change_5m": float((attrs.get("price_change_percentage") or {}).get("m5") or 0),
                    "dexscreener_url": f"https://dexscreener.com/solana/{address}",
                    "pumpfun_url": f"https://pump.fun/{address}",
                    "created_at": created_at,
                }
            )

        return filtered

    def _passes_filters(self, address: str, liquidity_usd: float, created_at: datetime) -> bool:
        if address in self.seen_tokens:
            return False

        if liquidity_usd < MIN_LIQUIDITY:
            return False

        now = datetime.now(timezone.utc)
        age_seconds = (now - created_at).total_seconds()
        if age_seconds > TOKEN_AGE_MAX:
            return False

        self.seen_tokens[address] = now.timestamp()
        return True

    def _format_dex_token_data(self, pair: dict[str, Any], created_at: datetime) -> dict[str, Any]:
        base = pair.get("baseToken", {})
        liquidity = float((pair.get("liquidity") or {}).get("usd") or 0)
        volume_5m = float((pair.get("volume") or {}).get("m5") or 0)
        price_change_5m = float((pair.get("priceChange") or {}).get("m5") or 0)

        address = base.get("address", "")
        dexscreener_url = pair.get("url") or f"https://dexscreener.com/solana/{address}"

        return {
            "name": base.get("name", "Unknown"),
            "symbol": base.get("symbol", "N/A"),
            "address": address,
            "liquidity": liquidity,
            "volume_5m": volume_5m,
            "price_change_5m": price_change_5m,
            "dexscreener_url": dexscreener_url,
            "pumpfun_url": f"https://pump.fun/{address}" if address else "https://pump.fun",
            "created_at": created_at,
        }

    def _build_gecko_token_map(self, included: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for item in included:
            if item.get("type") != "token":
                continue
            out[item.get("id", "")] = item.get("attributes", {})
        return out

    @staticmethod
    def _parse_rfc3339(value: str) -> datetime | None:
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None

    def _prune_seen_tokens(self) -> None:
        now_ts = datetime.now(timezone.utc).timestamp()
        expired = [token for token, ts in self.seen_tokens.items() if (now_ts - ts) > SEEN_TOKEN_TTL]
        for token in expired:
            self.seen_tokens.pop(token, None)
