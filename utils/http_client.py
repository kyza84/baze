"""Resilient shared HTTP client with retry/backoff and per-source limits."""

from __future__ import annotations

import asyncio
import logging
import random
import time
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

    def snapshot_stats(self, reset: bool = False) -> dict[str, dict[str, int | float]]:
        out: dict[str, dict[str, int | float]] = {}
        for source, row in self._stats.items():
            total = int(row.ok + row.fail)
            err_pct = (float(row.fail) / total * 100.0) if total > 0 else 0.0
            out[source] = {
                "ok": int(row.ok),
                "fail": int(row.fail),
                "total": total,
                "rate_limited": int(row.rate_limited),
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

        sem = self._get_semaphore(source)
        stats = self._stats_row(source)
        for attempt in range(1, attempts + 1):
            status = 0
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
