"""Token monitor with DexScreener primary source and GeckoTerminal fallback."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from config import (
    CHAIN_ID,
    CHAIN_NAME,
    DEX_BOOSTS_MAX_TOKENS,
    DEX_BOOSTS_SOURCE_ENABLED,
    DEXSCREENER_API,
    DEXSCREENER_TOKEN_URL_TEMPLATE,
    DEX_RETRIES,
    DEX_SEARCH_QUERY,
    DEX_SEARCH_QUERIES,
    DEX_TIMEOUT,
    GECKO_NETWORK,
    GECKO_INGEST_DEDUP_TTL_SECONDS,
    GECKO_NEW_POOLS_DRAIN_MAX_PER_CYCLE,
    GECKO_NEW_POOLS_INGEST_INTERVAL_SECONDS,
    GECKO_NEW_POOLS_PAGES,
    GECKO_NEW_POOLS_QUEUE_MAX,
    MIN_LIQUIDITY,
    SEEN_TOKEN_TTL,
    TOKEN_AGE_MAX,
)
from utils.addressing import normalize_address
from utils.http_client import ResilientHttpClient

logger = logging.getLogger(__name__)


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
        self._http = ResilientHttpClient(
            timeout_seconds=float(DEX_TIMEOUT),
            headers=self._headers,
            source_limits={
                "dexscreener": 8,
                "geckoterminal": 5,
                "dex_boosts": 5,
            },
        )
        self._gecko_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=int(GECKO_NEW_POOLS_QUEUE_MAX))
        self._gecko_ingest_task: asyncio.Task | None = None
        self._gecko_stop = asyncio.Event()
        self._gecko_start_lock = asyncio.Lock()
        self._gecko_enqueued_count = 0
        self._gecko_drained_count = 0
        self._gecko_dropped_oldest_count = 0
        self._gecko_ingest_dedup_skipped = 0
        self._gecko_ingest_seen_until: dict[str, float] = {}

    async def close(self) -> None:
        self._gecko_stop.set()
        task = self._gecko_ingest_task
        self._gecko_ingest_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Gecko ingest task stopped with error", exc_info=True)
        await self._http.close()

    def runtime_stats(self, reset: bool = False) -> dict[str, dict[str, int | float]]:
        stats = self._http.snapshot_stats(reset=reset)
        qsize = int(self._gecko_queue.qsize())
        oldest_age = 0.0
        if qsize > 0:
            try:
                oldest = self._gecko_queue._queue[0]  # type: ignore[attr-defined]
                oldest_ts = float(oldest.get("_enqueued_monotonic", 0.0) or 0.0)
                if oldest_ts > 0:
                    oldest_age = max(0.0, time.monotonic() - oldest_ts)
            except Exception:
                oldest_age = 0.0
        stats["gecko_ingest"] = {
            "queue_size": qsize,
            "oldest_item_age_sec": round(oldest_age, 2),
            "dropped_oldest_count": int(self._gecko_dropped_oldest_count),
            "ingest_dedup_skipped": int(self._gecko_ingest_dedup_skipped),
            "enqueued_count": int(self._gecko_enqueued_count),
            "drained_count": int(self._gecko_drained_count),
            "ingest_seen_keys": int(len(self._gecko_ingest_seen_until)),
        }
        if reset:
            self._gecko_enqueued_count = 0
            self._gecko_drained_count = 0
            self._gecko_dropped_oldest_count = 0
            self._gecko_ingest_dedup_skipped = 0
        return stats

    async def _fetch_json(self, url: str, source: str, retries: int | None = None) -> Any | None:
        result = await self._http.get_json(
            url,
            source=source,
            max_attempts=(retries if retries is not None else DEX_RETRIES),
        )
        if result.ok:
            return result.data
        if result.status == 429:
            logger.warning("RATE_LIMIT source=%s status=429 url=%s", source, url)
        return None

    async def fetch_new_tokens(self) -> list[dict[str, Any]]:
        self._prune_seen_tokens()
        await self._ensure_gecko_ingest_started()

        dex_task = asyncio.create_task(self._fetch_from_dexscreener())
        gecko_task = asyncio.create_task(self._drain_gecko_queue())
        tasks = [dex_task, gecko_task]
        if DEX_BOOSTS_SOURCE_ENABLED and DEX_BOOSTS_MAX_TOKENS > 0:
            tasks.append(asyncio.create_task(self._fetch_from_dex_boosts()))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        dex_tokens = results[0] if len(results) > 0 else []
        gecko_tokens = results[1] if len(results) > 1 else []
        boost_tokens = results[2] if len(results) > 2 else []
        if isinstance(dex_tokens, Exception):
            logger.warning("DexScreener source failed: %s", dex_tokens)
            dex_tokens = []
        if isinstance(gecko_tokens, Exception):
            logger.warning("Gecko source failed: %s", gecko_tokens)
            gecko_tokens = []
        if isinstance(boost_tokens, Exception):
            logger.warning("Boost source failed: %s", boost_tokens)
            boost_tokens = []
        return self._merge_tokens(dex_tokens, gecko_tokens, boost_tokens)

    async def _fetch_from_dexscreener(self) -> list[dict[str, Any]]:
        queries = DEX_SEARCH_QUERIES or [DEX_SEARCH_QUERY]
        tasks = [self._fetch_dex_query(query) for query in queries]
        result_sets = await asyncio.gather(*tasks, return_exceptions=True)
        merged: list[dict[str, Any]] = []
        for rows in result_sets:
            if isinstance(rows, Exception):
                logger.warning("Dex query failed: %s", rows)
                continue
            merged.extend(rows)
        return merged

    async def _fetch_dex_query(self, query: str) -> list[dict[str, Any]]:
        url = f"{DEXSCREENER_API}/search?q={query}"
        data = await self._fetch_json(url, source="dexscreener")
        if not isinstance(data, dict):
            return []
        pairs = data.get("pairs", [])
        return self._filter_dex_pairs(pairs)

    async def _ensure_gecko_ingest_started(self) -> None:
        task = self._gecko_ingest_task
        if task is not None and not task.done():
            return
        async with self._gecko_start_lock:
            task = self._gecko_ingest_task
            if task is not None and not task.done():
                return
            self._gecko_stop.clear()
            # Prime queue once before background ingest starts.
            await self._ingest_gecko_once()
            self._gecko_ingest_task = asyncio.create_task(self._gecko_ingest_loop(), name="gecko_new_pools_ingest")

    async def _gecko_ingest_loop(self) -> None:
        interval = max(15, int(GECKO_NEW_POOLS_INGEST_INTERVAL_SECONDS))
        while not self._gecko_stop.is_set():
            try:
                await self._ingest_gecko_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Gecko ingest loop failed: %s", exc)
            try:
                await asyncio.wait_for(self._gecko_stop.wait(), timeout=float(interval))
            except asyncio.TimeoutError:
                continue

    async def _ingest_gecko_once(self) -> None:
        rows = await self._fetch_from_geckoterminal_api()
        if not rows:
            return
        for row in rows:
            await self._queue_gecko_token(row)

    async def _queue_gecko_token(self, row: dict[str, Any]) -> None:
        now_mono = time.monotonic()
        self._prune_ingest_seen(now_mono)
        ingest_key = self._ingest_key(row)
        if ingest_key:
            until = float(self._gecko_ingest_seen_until.get(ingest_key, 0.0) or 0.0)
            if until > now_mono:
                self._gecko_ingest_dedup_skipped += 1
                return
            self._gecko_ingest_seen_until[ingest_key] = now_mono + float(GECKO_INGEST_DEDUP_TTL_SECONDS)

        payload = dict(row)
        payload["_enqueued_monotonic"] = now_mono
        try:
            self._gecko_queue.put_nowait(payload)
            self._gecko_enqueued_count += 1
        except asyncio.QueueFull:
            try:
                _ = self._gecko_queue.get_nowait()
                self._gecko_dropped_oldest_count += 1
            except asyncio.QueueEmpty:
                pass
            try:
                self._gecko_queue.put_nowait(payload)
                self._gecko_enqueued_count += 1
            except asyncio.QueueFull:
                return

    def _ingest_key(self, row: dict[str, Any]) -> str:
        source = str(row.get("source") or "geckoterminal").strip().lower()
        dex = str(row.get("dex") or "").strip().lower()
        pair_address = str(row.get("pair_address") or "").strip().lower()
        token_address = normalize_address(str(row.get("address") or ""))
        if pair_address:
            return f"{dex}:{pair_address}"
        if token_address:
            return f"{source}:{token_address}"
        return ""

    def _prune_ingest_seen(self, now_mono: float) -> None:
        if not self._gecko_ingest_seen_until:
            return
        if len(self._gecko_ingest_seen_until) < 2000:
            return
        expired = [k for k, v in self._gecko_ingest_seen_until.items() if float(v or 0.0) <= now_mono]
        for k in expired:
            self._gecko_ingest_seen_until.pop(k, None)

    async def _drain_gecko_queue(self) -> list[dict[str, Any]]:
        max_items = max(10, int(GECKO_NEW_POOLS_DRAIN_MAX_PER_CYCLE))
        out: list[dict[str, Any]] = []
        seen_addr: set[str] = set()
        for _ in range(max_items):
            try:
                row = self._gecko_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            address = normalize_address(str(row.get("address") or ""))
            if not address or address in seen_addr:
                continue
            row.pop("_enqueued_monotonic", None)
            seen_addr.add(address)
            out.append(row)
            self._gecko_drained_count += 1
        return out

    async def _fetch_from_geckoterminal_api(self) -> list[dict[str, Any]]:
        all_tokens: list[dict[str, Any]] = []
        for page in range(1, GECKO_NEW_POOLS_PAGES + 1):
            page_tokens = await self._fetch_gecko_page(page)
            if not page_tokens:
                if page == 1:
                    return []
                break
            all_tokens.extend(page_tokens)
        return all_tokens

    async def _fetch_gecko_page(self, page: int) -> list[dict[str, Any]]:
        gecko_url = (
            f"https://api.geckoterminal.com/api/v2/networks/{GECKO_NETWORK}/new_pools"
            f"?page={page}&include=base_token"
        )
        data = await self._fetch_json(gecko_url, source="geckoterminal")
        if not isinstance(data, dict):
            return []
        pools = data.get("data", [])
        included = data.get("included", [])
        return self._filter_gecko_pools(pools, included)

    async def _fetch_from_dex_boosts(self) -> list[dict[str, Any]]:
        url = "https://api.dexscreener.com/token-boosts/latest/v1"
        data = await self._fetch_json(url, source="dex_boosts", retries=1)
        if data is None:
            return []

        if not isinstance(data, list):
            return []

        token_addresses: list[str] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            chain = str(row.get("chainId", "")).lower()
            if chain and chain != CHAIN_ID.lower():
                continue
            addr = str(row.get("tokenAddress", "")).strip()
            if addr:
                token_addresses.append(addr)
            if len(token_addresses) >= DEX_BOOSTS_MAX_TOKENS:
                break
        if not token_addresses:
            return []

        tasks = [self._fetch_token_by_address(addr) for addr in token_addresses]
        rows = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, Exception):
                logger.warning("Boost token resolve failed: %s", row)
                continue
            if row:
                out.append(row)
        return out

    async def _fetch_token_by_address(self, token_address: str) -> dict[str, Any] | None:
        token_address = normalize_address(token_address)
        if not token_address:
            return None
        url = f"{DEXSCREENER_API}/tokens/{token_address}"
        data = await self._fetch_json(url, source="dexscreener", retries=1)
        if not isinstance(data, dict):
            return None

        pairs = data.get("pairs", []) or []
        best_pair = None
        best_liq = -1.0
        for pair in pairs:
            if str(pair.get("chainId", "")).lower() != CHAIN_ID.lower():
                continue
            liq = float((pair.get("liquidity") or {}).get("usd") or 0)
            if liq > best_liq:
                best_liq = liq
                best_pair = pair
        if not isinstance(best_pair, dict):
            return None

        created_ms = best_pair.get("pairCreatedAt")
        if not created_ms:
            return None
        created_at = datetime.fromtimestamp(float(created_ms) / 1000, tz=timezone.utc)
        address = normalize_address(str((best_pair.get("baseToken") or {}).get("address") or ""))
        liquidity_usd = float((best_pair.get("liquidity") or {}).get("usd") or 0)
        if not self._passes_filters(address, liquidity_usd, created_at):
            return None
        return self._format_token_data(best_pair, created_at)

    @staticmethod
    def _merge_tokens(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_address: dict[str, dict[str, Any]] = {}
        for tokens in groups:
            for token in tokens:
                address = normalize_address(str(token.get("address", "")))
                if not address:
                    continue
                existing = by_address.get(address)
                if not existing:
                    by_address[address] = token
                    continue
                if float(token.get("liquidity", 0)) > float(existing.get("liquidity", 0)):
                    by_address[address] = token
        return list(by_address.values())

    def _filter_dex_pairs(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []

        for pair in pairs:
            if str(pair.get("chainId", "")).lower() != CHAIN_ID.lower():
                continue

            base_token = pair.get("baseToken", {})
            address = normalize_address(base_token.get("address"))
            if not address:
                continue

            liquidity_usd = float((pair.get("liquidity") or {}).get("usd") or 0)
            created_ms = pair.get("pairCreatedAt")
            if not created_ms:
                continue

            created_at = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
            if not self._passes_filters(address, liquidity_usd, created_at):
                continue

            filtered.append(self._format_token_data(pair, created_at))

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
            address = normalize_address(base_token.get("address", ""))
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

            now = datetime.now(timezone.utc)
            age_seconds = int((now - created_at).total_seconds())
            filtered.append(
                {
                    "name": base_token.get("name") or (attrs.get("name", "Unknown").split("/")[0].strip()),
                    "symbol": base_token.get("symbol") or "N/A",
                    "address": address,
                    "liquidity": liquidity_usd,
                    "volume_5m": float((attrs.get("volume_usd") or {}).get("m5") or 0),
                    "price_change_5m": float((attrs.get("price_change_percentage") or {}).get("m5") or 0),
                    "price_usd": float(attrs.get("base_token_price_usd") or 0),
                    "dexscreener_url": DEXSCREENER_TOKEN_URL_TEMPLATE.format(
                        chain=CHAIN_NAME,
                        token_address=address,
                    ),
                    "source": "geckoterminal",
                    "dex": str(attrs.get("dex_id") or attrs.get("dex_name") or "geckoterminal").lower(),
                    "dex_labels": [],
                    "pair_address": str(attrs.get("address") or pool.get("id") or ""),
                    "created_at": created_at,
                    "age_seconds": age_seconds,
                    "age_minutes": int(round(age_seconds / 60)),
                }
            )

        return filtered

    def _passes_filters(self, address: str, liquidity_usd: float, created_at: datetime) -> bool:
        address = normalize_address(address)
        if not address:
            return False
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

    def _format_token_data(self, pair: dict[str, Any], created_at: datetime) -> dict[str, Any]:
        base = pair.get("baseToken", {})
        liquidity = float((pair.get("liquidity") or {}).get("usd") or 0)
        volume_5m = float((pair.get("volume") or {}).get("m5") or 0)
        price_change_5m = float((pair.get("priceChange") or {}).get("m5") or 0)
        price_usd = float(pair.get("priceUsd") or 0)
        dex_id = str(pair.get("dexId") or "").lower()
        pair_address = str(pair.get("pairAddress") or "")
        dex_labels_raw = pair.get("labels") or []
        dex_labels = [str(label) for label in dex_labels_raw] if isinstance(dex_labels_raw, list) else []

        address = normalize_address(base.get("address", ""))
        dexscreener_url = pair.get("url") or DEXSCREENER_TOKEN_URL_TEMPLATE.format(
            chain=CHAIN_NAME,
            token_address=address,
        )
        current_time = datetime.now(timezone.utc)
        created_ms = pair.get("pairCreatedAt") or int(created_at.timestamp() * 1000)
        token_age_seconds = current_time.timestamp() - (created_ms / 1000)

        return {
            "name": base.get("name", "Unknown"),
            "symbol": base.get("symbol", "N/A"),
            "address": address,
            "liquidity": liquidity,
            "volume_5m": volume_5m,
            "price_change_5m": price_change_5m,
            "price_usd": price_usd,
            "dexscreener_url": dexscreener_url,
            "source": "dexscreener",
            "dex": dex_id,
            "dex_labels": dex_labels,
            "pair_address": pair_address,
            "created_at": created_at,
            "age_seconds": max(0, int(token_age_seconds)),
            "age_minutes": max(0, int(round(token_age_seconds / 60))),
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
