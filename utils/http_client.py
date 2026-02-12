"""Resilient shared HTTP client with retry/backoff and per-source limits."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import aiohttp

import config

logger = logging.getLogger(__name__)


@dataclass
class HttpResult:
    ok: bool
    status: int
    data: Any | None
    error: str = ""


@dataclass
class HttpSourceStats:
    ok: int = 0
    fail: int = 0
    rate_limited: int = 0
    limiter_waits: int = 0
    cooldown_waits: int = 0
    retries: int = 0
    latency_total_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_count: int = 0


class ResilientHttpClient:
    def __init__(
        self,
        timeout_seconds: float,
        headers: dict[str, str] | None = None,
        source_limits: dict[str, int] | None = None,
    ) -> None:
        self._timeout = aiohttp.ClientTimeout(total=max(1.0, float(timeout_seconds)))
        self._headers = dict(headers or {})
        self._source_limits = dict(source_limits or {})
        self._session: aiohttp.ClientSession | None = None
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._stats: dict[str, HttpSourceStats] = {}
        self._rate_windows: dict[str, deque[float]] = {}
        self._rate_locks: dict[str, asyncio.Lock] = {}
        self._cooldown_until: dict[str, float] = {}

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector_limit = max(1, int(getattr(config, "HTTP_CONNECTOR_LIMIT", 30) or 30))
            connector = aiohttp.TCPConnector(limit=connector_limit)
            self._session = aiohttp.ClientSession(timeout=self._timeout, connector=connector)
        return self._session

    def _get_semaphore(self, source: str) -> asyncio.Semaphore:
        key = str(source or "default").strip().lower() or "default"
        sem = self._semaphores.get(key)
        if sem is not None:
            return sem
        default_limit = max(1, int(getattr(config, "HTTP_DEFAULT_CONCURRENCY", 8) or 8))
        limit = max(1, int(self._source_limits.get(key, default_limit)))
        sem = asyncio.Semaphore(limit)
        self._semaphores[key] = sem
        return sem

    def _stats_row(self, source: str) -> HttpSourceStats:
        key = str(source or "default").strip().lower() or "default"
        row = self._stats.get(key)
        if row is None:
            row = HttpSourceStats()
            self._stats[key] = row
        return row

    @staticmethod
    def _source_key(source: str) -> str:
        return str(source or "default").strip().lower() or "default"

    def _rate_limit_config(self, source_key: str) -> tuple[int, float]:
        raw = getattr(config, "HTTP_SOURCE_RATE_LIMITS", {}) or {}
        limit_data = raw.get(source_key)
        if isinstance(limit_data, tuple) and len(limit_data) == 2:
            try:
                count = max(1, int(limit_data[0]))
                window_seconds = max(1.0, float(limit_data[1]))
                return count, window_seconds
            except Exception:
                pass
        return (1_000_000, 1.0)

    def _source_429_cooldown_seconds(self, source_key: str) -> float:
        per_source = getattr(config, "HTTP_SOURCE_429_COOLDOWNS", {}) or {}
        if source_key in per_source:
            try:
                return max(0.0, float(per_source[source_key]))
            except Exception:
                pass
        return max(0.0, float(getattr(config, "HTTP_429_COOLDOWN_SECONDS", 90.0) or 90.0))

    def _get_rate_lock(self, source_key: str) -> asyncio.Lock:
        lock = self._rate_locks.get(source_key)
        if lock is None:
            lock = asyncio.Lock()
            self._rate_locks[source_key] = lock
        return lock

    async def _wait_rate_slot(self, source_key: str, stats: HttpSourceStats, url: str) -> None:
        max_calls, window_seconds = self._rate_limit_config(source_key)
        if max_calls >= 1_000_000:
            return
        lock = self._get_rate_lock(source_key)
        while True:
            wait_for = 0.0
            async with lock:
                now = time.monotonic()
                window = self._rate_windows.get(source_key)
                if window is None:
                    window = deque()
                    self._rate_windows[source_key] = window
                cutoff = now - window_seconds
                while window and window[0] <= cutoff:
                    window.popleft()
                if len(window) < max_calls:
                    window.append(now)
                    return
                wait_for = max(0.01, (window[0] + window_seconds) - now)
            stats.limiter_waits += 1
            logger.debug(
                "HTTP_RATE_WAIT source=%s wait=%.2fs window=%ss max_calls=%s url=%s",
                source_key,
                wait_for,
                window_seconds,
                max_calls,
                url,
            )
            await asyncio.sleep(wait_for)

    async def _wait_cooldown(self, source_key: str, stats: HttpSourceStats, url: str) -> None:
        while True:
            now = time.monotonic()
            until = float(self._cooldown_until.get(source_key, 0.0) or 0.0)
            if until <= now:
                return
            wait_for = max(0.01, until - now)
            stats.cooldown_waits += 1
            logger.debug("HTTP_COOLDOWN_WAIT source=%s wait=%.2fs url=%s", source_key, wait_for, url)
            await asyncio.sleep(wait_for)

    def _apply_source_cooldown(self, source_key: str, response: aiohttp.ClientResponse) -> None:
        retry_after_raw = (response.headers or {}).get("Retry-After", "")
        retry_after = 0.0
        if retry_after_raw:
            try:
                retry_after = max(0.0, float(retry_after_raw))
            except Exception:
                retry_after = 0.0
        cooldown_seconds = max(self._source_429_cooldown_seconds(source_key), retry_after)
        if cooldown_seconds <= 0:
            return
        now = time.monotonic()
        until = now + cooldown_seconds
        prev = float(self._cooldown_until.get(source_key, 0.0) or 0.0)
        self._cooldown_until[source_key] = max(prev, until)

    def snapshot_stats(self, reset: bool = False) -> dict[str, dict[str, int | float]]:
        out: dict[str, dict[str, int | float]] = {}
        now = time.monotonic()
        all_sources = set(self._stats.keys()) | set(self._cooldown_until.keys())
        for source in all_sources:
            row = self._stats.get(source)
            if row is None:
                row = HttpSourceStats()
                self._stats[source] = row
            total = int(row.ok + row.fail)
            err_pct = (float(row.fail) / total * 100.0) if total > 0 else 0.0
            cooldown_until = float(self._cooldown_until.get(source, 0.0) or 0.0)
            cooldown_remaining = max(0.0, cooldown_until - now)
            out[source] = {
                "ok": int(row.ok),
                "fail": int(row.fail),
                "total": total,
                "rate_limited": int(row.rate_limited),
                "limiter_waits": int(row.limiter_waits),
                "cooldown_waits": int(row.cooldown_waits),
                "cooldown_active": 1 if cooldown_remaining > 0 else 0,
                "cooldown_remaining_sec": round(cooldown_remaining, 2),
                "cooldown_until_monotonic": round(cooldown_until, 3),
                "retries": int(row.retries),
                "error_percent": round(err_pct, 2),
                "latency_avg_ms": round((row.latency_total_ms / row.latency_count), 2) if row.latency_count > 0 else 0.0,
                "latency_max_ms": round(float(row.latency_max_ms), 2),
            }
        if reset:
            self._stats = {}
        return out

    @staticmethod
    def _compute_delay(attempt: int, status: int) -> float:
        base = max(0.05, float(getattr(config, "HTTP_BACKOFF_BASE_SECONDS", 0.5) or 0.5))
        cap = max(base, float(getattr(config, "HTTP_BACKOFF_MAX_SECONDS", 8.0) or 8.0))
        jitter = max(0.0, float(getattr(config, "HTTP_JITTER_SECONDS", 0.25) or 0.25))
        rate_limit_bias = max(0.0, float(getattr(config, "HTTP_RATE_LIMIT_DELAY_SECONDS", 2.0) or 2.0))

        exp = min(cap, base * (2 ** max(0, attempt - 1)))
        if status == 429:
            exp = min(cap, exp + rate_limit_bias)
        return max(0.01, exp + random.uniform(0.0, jitter))

    async def get_json(
        self,
        url: str,
        *,
        source: str = "default",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_attempts: int | None = None,
    ) -> HttpResult:
        attempts = max(1, int(max_attempts or int(getattr(config, "HTTP_RETRY_ATTEMPTS", 3) or 3)))
        req_headers = dict(self._headers)
        if headers:
            req_headers.update(headers)

        source_key = self._source_key(source)
        sem = self._get_semaphore(source_key)
        stats = self._stats_row(source_key)
        for attempt in range(1, attempts + 1):
            status = 0
            await self._wait_cooldown(source_key, stats, url)
            await self._wait_rate_slot(source_key, stats, url)
            async with sem:
                started = time.perf_counter()
                try:
                    session = await self._get_session()
                    async with session.get(url, params=params, headers=req_headers) as response:
                        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
                        stats.latency_total_ms += elapsed_ms
                        stats.latency_count += 1
                        stats.latency_max_ms = max(stats.latency_max_ms, elapsed_ms)
                        status = int(response.status or 0)
                        if status == 200:
                            payload = await response.json()
                            stats.ok += 1
                            return HttpResult(ok=True, status=status, data=payload)

                        retryable = status == 429 or (500 <= status <= 599)
                        if status == 429:
                            stats.rate_limited += 1
                            self._apply_source_cooldown(source_key, response)
                        if not retryable or attempt >= attempts:
                            stats.fail += 1
                            return HttpResult(ok=False, status=status, data=None, error=f"http_status_{status}")
                except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ValueError) as exc:
                    elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
                    stats.latency_total_ms += elapsed_ms
                    stats.latency_count += 1
                    stats.latency_max_ms = max(stats.latency_max_ms, elapsed_ms)
                    if attempt >= attempts:
                        stats.fail += 1
                        return HttpResult(ok=False, status=status, data=None, error=f"http_error:{exc}")
                except Exception as exc:  # pragma: no cover - defensive
                    elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
                    stats.latency_total_ms += elapsed_ms
                    stats.latency_count += 1
                    stats.latency_max_ms = max(stats.latency_max_ms, elapsed_ms)
                    if attempt >= attempts:
                        stats.fail += 1
                        return HttpResult(ok=False, status=status, data=None, error=f"unexpected_http_error:{exc}")

            stats.retries += 1
            delay = self._compute_delay(attempt=attempt, status=status)
            logger.debug(
                "HTTP_RETRY source=%s attempt=%s/%s status=%s delay=%.2fs url=%s",
                source,
                attempt,
                attempts,
                status,
                delay,
                url,
            )
            await asyncio.sleep(delay)

        return HttpResult(ok=False, status=0, data=None, error="http_exhausted")
