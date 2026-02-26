"""Dynamic watchlist (vetted tokens) monitor for stability-first trading.

This is NOT a "new pairs" feed. It refreshes a list of established/liquid pools and emits
only those that currently show activity (volume/price change), so the bot can trade without
living entirely in the scam-heavy newest-pairs universe.
"""

from __future__ import annotations

import json
import os
import time
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import config
from utils.addressing import normalize_address
from utils.http_client import ResilientHttpClient

logger = logging.getLogger(__name__)


class WatchlistMonitor:
    def __init__(self) -> None:
        self._last_refresh_ts = 0.0
        self._cached_tokens: list[dict[str, Any]] = []
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
        self._http = ResilientHttpClient(
            timeout_seconds=float(config.DEX_TIMEOUT),
            headers=self._headers,
            source_limits={
                "watchlist_dex": 8,
                "watchlist_gecko": 5,
            },
        )

    async def close(self) -> None:
        await self._http.close()

    def runtime_stats(self, reset: bool = False) -> dict[str, dict[str, int | float]]:
        return self._http.snapshot_stats(reset=reset)

    async def _fetch_json(self, url: str, source: str, retries: int | None = None) -> Any | None:
        result = await self._http.get_json(
            url,
            source=source,
            max_attempts=(retries if retries is not None else int(config.DEX_RETRIES)),
        )
        if result.ok:
            return result.data
        if result.status == 429:
            logger.warning("RATE_LIMIT source=%s status=429 url=%s", source, url)
            return None
        return None

    async def fetch_tokens(self) -> list[dict[str, Any]]:
        if not bool(getattr(config, "WATCHLIST_ENABLED", False)):
            return []
        now = time.time()
        refresh = float(getattr(config, "WATCHLIST_REFRESH_SECONDS", 3600) or 3600)
        if self._cached_tokens and (now - self._last_refresh_ts) < refresh:
            return list(self._cached_tokens)
        tokens = await self._refresh()
        self._cached_tokens = list(tokens)
        self._last_refresh_ts = now
        self._save_cache(tokens)
        return list(tokens)

    async def _refresh(self) -> list[dict[str, Any]]:
        # Build a real watchlist universe from DexScreener search (older + liquid) and optionally
        # GeckoTerminal pools as a supplement.
        tokens: list[dict[str, Any]] = []
        seen_addr: set[str] = set()

        for row in await self._fetch_dexscreener_watchlist():
            addr = normalize_address(row.get("address"))
            if not addr or addr in seen_addr:
                continue
            seen_addr.add(addr)
            tokens.append(row)

        trending_pages = int(getattr(config, "WATCHLIST_GECKO_TRENDING_PAGES", 2) or 2)
        pools_pages = int(getattr(config, "WATCHLIST_GECKO_POOLS_PAGES", 2) or 2)
        trending_pages = max(1, min(5, trending_pages))
        pools_pages = max(0, min(5, pools_pages))

        trending_tasks = [self._fetch_gecko_pool_page(kind="trending_pools", page=page) for page in range(1, trending_pages + 1)]
        trending_rows = await asyncio.gather(*trending_tasks, return_exceptions=True) if trending_tasks else []
        for rows in trending_rows:
            if isinstance(rows, Exception):
                logger.warning("Watchlist trending page failed: %s", rows)
                continue
            for row in rows:
                addr = normalize_address(row.get("address"))
                if not addr or addr in seen_addr:
                    continue
                seen_addr.add(addr)
                tokens.append(row)

        pool_tasks = [self._fetch_gecko_pool_page(kind="pools", page=page) for page in range(1, pools_pages + 1)]
        pool_rows = await asyncio.gather(*pool_tasks, return_exceptions=True) if pool_tasks else []
        for rows in pool_rows:
            if isinstance(rows, Exception):
                logger.warning("Watchlist pools page failed: %s", rows)
                continue
            for row in rows:
                addr = normalize_address(row.get("address"))
                if not addr or addr in seen_addr:
                    continue
                seen_addr.add(addr)
                tokens.append(row)

        # Hard cap total output; filters downstream will pick the best anyway.
        max_tokens = int(getattr(config, "WATCHLIST_MAX_TOKENS", 30) or 30)
        if max_tokens > 0:
            tokens = tokens[: max(1, max_tokens)]
        return tokens

    async def _fetch_dexscreener_watchlist(self) -> list[dict[str, Any]]:
        queries = getattr(config, "DEX_SEARCH_QUERIES", None) or [getattr(config, "DEX_SEARCH_QUERY", "base")]
        chain_id = normalize_address(getattr(config, "CHAIN_ID", "base") or "base")
        weth = normalize_address(getattr(config, "WETH_ADDRESS", "") or "")
        allow = set(getattr(config, "WATCHLIST_DEX_ALLOWLIST", []) or [])
        require_weth_quote = bool(getattr(config, "WATCHLIST_REQUIRE_WETH_QUOTE", True))

        min_liq = float(getattr(config, "WATCHLIST_MIN_LIQUIDITY_USD", 200000) or 200000)
        min_vol_h24 = float(getattr(config, "WATCHLIST_MIN_VOLUME_24H_USD", 500000) or 500000)

        out: list[dict[str, Any]] = []
        active_queries = [str(q or "").strip() for q in list(queries)[:20] if str(q or "").strip()]
        if not active_queries:
            return []

        urls = [f"{config.DEXSCREENER_API}/search?q={query}" for query in active_queries]
        payloads = await asyncio.gather(
            *[self._fetch_json(url, source="watchlist_dex") for url in urls],
            return_exceptions=True,
        )
        for payload in payloads:
            if isinstance(payload, Exception):
                logger.warning("Watchlist dex payload failed: %s", payload)
                continue
            if not isinstance(payload, dict):
                continue

            pairs = payload.get("pairs", []) or []
            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                if normalize_address(pair.get("chainId", "")) != chain_id:
                    continue
                dex_id = str(pair.get("dexId") or "").lower()
                if allow and dex_id and (dex_id not in allow):
                    continue
                quote = pair.get("quoteToken") or {}
                quote_addr = normalize_address((quote or {}).get("address", ""))
                if require_weth_quote and weth and quote_addr and quote_addr != weth:
                    continue

                base = pair.get("baseToken") or {}
                address = normalize_address((base or {}).get("address", ""))
                if not address:
                    continue

                try:
                    liq = float((pair.get("liquidity") or {}).get("usd") or 0.0)
                except Exception:
                    liq = 0.0
                if liq < min_liq:
                    continue

                vol = pair.get("volume") or {}
                try:
                    vol_h24 = float((vol or {}).get("h24") or 0.0)
                except Exception:
                    vol_h24 = 0.0
                if vol_h24 < min_vol_h24:
                    continue

                created_ms = pair.get("pairCreatedAt")
                created_at = None
                if created_ms:
                    try:
                        created_at = datetime.fromtimestamp(float(created_ms) / 1000, tz=timezone.utc)
                    except Exception:
                        created_at = None
                if created_at is None:
                    # Some DexScreener search rows miss pairCreatedAt even for mature pools.
                    # Treat missing timestamp as aged watchlist data instead of age=0 to avoid false safe_age drops.
                    fallback_age_seconds = int(
                        getattr(config, "WATCHLIST_MISSING_PAIR_AGE_FALLBACK_SECONDS", 86400) or 86400
                    )
                    age_seconds = max(0, fallback_age_seconds)
                    created_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
                else:
                    age_seconds = int((datetime.now(timezone.utc) - created_at).total_seconds())

                try:
                    vol_5m = float((vol or {}).get("m5") or 0.0)
                except Exception:
                    vol_5m = 0.0
                pc = pair.get("priceChange") or {}
                try:
                    pc_5m = float((pc or {}).get("m5") or 0.0)
                except Exception:
                    pc_5m = 0.0
                try:
                    price_usd = float(pair.get("priceUsd") or 0.0)
                except Exception:
                    price_usd = 0.0

                out.append(
                    {
                        "name": str((base or {}).get("name") or "Unknown"),
                        "symbol": str((base or {}).get("symbol") or "N/A"),
                        "address": address,
                        "liquidity": float(liq),
                        "volume_5m": float(vol_5m),
                        "price_change_5m": float(pc_5m),
                        "price_usd": float(price_usd),
                        "dexscreener_url": str(pair.get("url") or ""),
                        "source": "watchlist",
                        "dex": dex_id.replace("-", "_"),
                        "dex_labels": [dex_id] if dex_id else [],
                        "pair_address": str(pair.get("pairAddress") or ""),
                        "created_at": created_at,
                        "age_seconds": max(0, int(age_seconds)),
                        "age_minutes": max(0, int(round(age_seconds / 60))),
                    }
                )

        # Highest liquidity first to stabilize selection, downstream filters will cut further.
        out.sort(key=lambda x: float(x.get("liquidity") or 0.0), reverse=True)
        return out

    async def _fetch_gecko_pool_page(self, kind: str, page: int = 1) -> list[dict[str, Any]]:
        network = str(getattr(config, "GECKO_NETWORK", "base") or "base").strip()
        include = "base_token,quote_token,dex"
        url = f"https://api.geckoterminal.com/api/v2/networks/{network}/{kind}?page={int(page)}&include={include}"
        data = await self._fetch_json(url, source="watchlist_gecko")
        if not isinstance(data, dict):
            return []

        pools = data.get("data", []) or []
        included = data.get("included", []) or []
        token_map: dict[str, dict[str, Any]] = {}
        dex_map: dict[str, dict[str, Any]] = {}
        for item in included:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "token":
                token_map[str(item.get("id", ""))] = item.get("attributes", {}) or {}
            elif item.get("type") == "dex":
                dex_map[str(item.get("id", ""))] = item.get("attributes", {}) or {}

        weth = normalize_address(getattr(config, "WETH_ADDRESS", "") or "")
        require_weth_quote = bool(getattr(config, "WATCHLIST_REQUIRE_WETH_QUOTE", True))
        min_liq = float(getattr(config, "WATCHLIST_MIN_LIQUIDITY_USD", 200000) or 200000)
        min_vol_h24 = float(getattr(config, "WATCHLIST_MIN_VOLUME_24H_USD", 500000) or 500000)
        min_vol_5m = float(getattr(config, "WATCHLIST_MIN_VOLUME_5M_USD", 0.0) or 0.0)
        min_pc = float(getattr(config, "WATCHLIST_MIN_PRICE_CHANGE_5M_ABS_PERCENT", 0.0) or 0.0)

        out: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for pool in pools:
            if not isinstance(pool, dict):
                continue
            attrs = pool.get("attributes", {}) or {}
            rel = pool.get("relationships", {}) or {}
            base_id = str(((rel.get("base_token") or {}).get("data") or {}).get("id") or "")
            quote_id = str(((rel.get("quote_token") or {}).get("data") or {}).get("id") or "")
            dex_id = normalize_address(((rel.get("dex") or {}).get("data") or {}).get("id") or "")
            dex_name = str((dex_map.get(dex_id) or {}).get("name") or dex_id).lower()

            base_token = token_map.get(base_id, {}) or {}
            quote_token = token_map.get(quote_id, {}) or {}
            address = normalize_address(base_token.get("address") or "")
            quote_addr = normalize_address(quote_token.get("address") or "")
            if not address:
                continue
            # Ensure pool quote is WETH for our router.
            if require_weth_quote and weth and quote_addr and quote_addr != weth:
                continue
            # Exclude obvious v3 pools from watchlist flow (we route on v2 compatible router).
            if "v3" in dex_id or "v3" in dex_name:
                continue

            try:
                liquidity_usd = float(attrs.get("reserve_in_usd") or 0.0)
            except Exception:
                liquidity_usd = 0.0
            if liquidity_usd < min_liq:
                continue

            volume = attrs.get("volume_usd") or {}
            chg = attrs.get("price_change_percentage") or {}
            try:
                vol_5m = float((volume or {}).get("m5") or 0.0)
            except Exception:
                vol_5m = 0.0
            try:
                vol_h24 = float((volume or {}).get("h24") or 0.0)
            except Exception:
                vol_h24 = 0.0
            if vol_h24 < min_vol_h24 or (min_vol_5m > 0 and vol_5m < min_vol_5m):
                continue

            try:
                pc_5m = float((chg or {}).get("m5") or 0.0)
            except Exception:
                pc_5m = 0.0
            if min_pc > 0 and abs(pc_5m) < min_pc:
                continue

            created_at = self._parse_rfc3339(str(attrs.get("pool_created_at") or "")) or now
            age_seconds = int((now - created_at).total_seconds())
            price_usd = float(attrs.get("base_token_price_usd") or 0.0)

            out.append(
                {
                    "name": str(base_token.get("name") or (str(attrs.get("name") or "Unknown").split("/")[0].strip())),
                    "symbol": str(base_token.get("symbol") or "N/A"),
                    "address": address,
                    "liquidity": float(liquidity_usd),
                    "volume_5m": float(vol_5m),
                    "price_change_5m": float(pc_5m),
                    "price_usd": float(price_usd),
                    "dexscreener_url": "",  # not available in Gecko payload
                    "source": "watchlist",
                    "dex": dex_id.replace("-", "_"),
                    "dex_labels": [],
                    "pair_address": str(attrs.get("address") or pool.get("id") or ""),
                    "created_at": created_at,
                    "age_seconds": max(0, age_seconds),
                    "age_minutes": max(0, int(round(age_seconds / 60))),
                }
            )
        return out

    @staticmethod
    def _parse_rfc3339(value: str) -> datetime | None:
        try:
            if not value:
                return None
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _save_cache(self, tokens: list[dict[str, Any]]) -> None:
        path = str(getattr(config, "WATCHLIST_CACHE_FILE", "") or "").strip()
        if not path:
            return
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            payload = {
                "updated_ts": time.time(),
                "chain_id": str(getattr(config, "CHAIN_ID", "")),
                "count": len(tokens),
                "tokens": [
                    {
                        "address": str(t.get("address", "")),
                        "symbol": str(t.get("symbol", "")),
                        "liquidity": float(t.get("liquidity") or 0),
                        "volume_5m": float(t.get("volume_5m") or 0),
                        "price_change_5m": float(t.get("price_change_5m") or 0),
                    }
                    for t in tokens
                ],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            return
