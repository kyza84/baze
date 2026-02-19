"""Token safety checks with optional GoPlus integration and fallback heuristics."""

from __future__ import annotations

import logging
import time
from typing import Any

import config
from config import CHAIN_ID, EVM_CHAIN_ID, GOPLUS_ACCESS_TOKEN, GOPLUS_EVM_API, GOPLUS_SOLANA_API
from utils.http_client import ResilientHttpClient

logger = logging.getLogger(__name__)


class TokenChecker:
    def __init__(self) -> None:
        self._http = ResilientHttpClient(
            timeout_seconds=10.0,
            source_limits={"goplus": 4},
        )
        self._checks_total = 0
        self._api_ok = 0
        self._api_fail = 0
        self._api_fail_total = 0
        self._api_fail_transient = 0
        self._fail_closed = 0
        self._api_fail_reasons: dict[str, int] = {}
        self._safety_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    async def close(self) -> None:
        await self._http.close()

    def runtime_stats(self, reset: bool = False) -> dict[str, int | float]:
        total = int(self._checks_total)
        fail = int(self._api_fail)
        fail_total = int(self._api_fail_total)
        fail_transient = int(self._api_fail_transient)
        err_pct = (float(fail) / total * 100.0) if total > 0 else 0.0
        err_pct_total = (float(fail_total) / total * 100.0) if total > 0 else 0.0
        top_reason = "none"
        top_count = 0
        if self._api_fail_reasons:
            top_reason, top_count = max(self._api_fail_reasons.items(), key=lambda kv: int(kv[1]))
        out = {
            "checks_total": total,
            "api_ok": int(self._api_ok),
            "api_fail": fail,
            "api_fail_total": fail_total,
            "api_fail_transient": fail_transient,
            "fail_closed": int(self._fail_closed),
            "api_error_percent": round(err_pct, 2),
            "api_error_percent_total": round(err_pct_total, 2),
            "fail_reason_top": top_reason,
            "fail_reason_top_count": int(top_count),
            "fail_reason_counts": dict(self._api_fail_reasons),
        }
        if reset:
            self._checks_total = 0
            self._api_ok = 0
            self._api_fail = 0
            self._api_fail_total = 0
            self._api_fail_transient = 0
            self._fail_closed = 0
            self._api_fail_reasons = {}
        return out

    def _mark_fail_reason(self, reason: str | None) -> str:
        key = str(reason or "unknown").strip().lower() or "unknown"
        self._api_fail_reasons[key] = int(self._api_fail_reasons.get(key, 0)) + 1
        return key

    @staticmethod
    def _cache_key(token_address: str) -> str:
        return str(token_address or "").strip().lower()

    def _remember_safety(self, token_address: str, result: dict[str, Any]) -> None:
        key = self._cache_key(token_address)
        if not key:
            return
        self._safety_cache[key] = (time.time(), dict(result or {}))
        # Keep cache bounded to avoid uncontrolled growth on long 24/7 runs.
        if len(self._safety_cache) > 5000:
            try:
                oldest_key = min(self._safety_cache.items(), key=lambda kv: float(kv[1][0]))[0]
                self._safety_cache.pop(oldest_key, None)
            except Exception:
                self._safety_cache.clear()

    def _get_cached_safety(self, token_address: str) -> dict[str, Any] | None:
        key = self._cache_key(token_address)
        if not key:
            return None
        entry = self._safety_cache.get(key)
        if not entry:
            return None
        ts, cached = entry
        ttl = max(60, int(getattr(config, "V2_SAFETY_CACHE_TTL_SECONDS", 3600) or 3600))
        if (time.time() - float(ts)) > float(ttl):
            self._safety_cache.pop(key, None)
            return None
        return dict(cached or {})

    def _is_transient_fail_reason(self, fail_reason: str | None) -> bool:
        reason = str(fail_reason or "").strip().lower()
        if not reason:
            return False
        transient_reasons = [str(x).strip().lower() for x in getattr(config, "TOKEN_SAFETY_TRANSIENT_REASONS", [])]
        if reason in transient_reasons:
            return True
        # Keep 4029 as transient by default even if env override is missing.
        if reason == "api_code_4029":
            return True
        return False

    @staticmethod
    def _fallback_assessment(token_address: str, liquidity: float | None = None) -> dict[str, Any]:
        warnings: list[str] = []
        risk_level = "MEDIUM"
        is_safe = True

        if liquidity is not None and liquidity < 15000:
            risk_level = "HIGH"
            is_safe = False
            warnings.append("Low liquidity")

        top_10_holders = 85 if (liquidity is not None and liquidity < 15000) else 70
        if top_10_holders > 80:
            risk_level = "HIGH"
            is_safe = False
            warnings.append("Concentrated ownership")

        return {
            "token_address": token_address,
            "is_safe": is_safe,
            "risk_level": risk_level,
            "warnings": warnings,
            "warning_flags": len(warnings),
            "source": "fallback",
        }

    async def check_token_safety(self, token_address: str, liquidity: float | None = None) -> dict[str, Any]:
        self._checks_total += 1
        api_result, fail_reason = await self._check_with_goplus(token_address)
        if api_result:
            self._api_ok += 1
            self._remember_safety(token_address, api_result)
            return api_result

        # Stability-first mode: if we cannot validate safety with an external API, do not trade.
        if bool(getattr(config, "TOKEN_SAFETY_FAIL_CLOSED", False)):
            fail_key = self._mark_fail_reason(fail_reason)
            self._api_fail_total += 1
            if bool(getattr(config, "TOKEN_SAFETY_TRANSIENT_DEGRADED_ENABLED", True)) and self._is_transient_fail_reason(
                fail_key
            ):
                self._api_fail_transient += 1
                if bool(getattr(config, "TOKEN_SAFETY_TRANSIENT_USE_CACHE", True)):
                    cached = self._get_cached_safety(token_address)
                    if isinstance(cached, dict) and cached:
                        cached_out = dict(cached)
                        cached_out["source"] = "cache_transient"
                        cached_out["fail_reason"] = fail_key
                        cached_out["warning_flags"] = int(cached_out.get("warning_flags", 0) or 0)
                        return cached_out
                if bool(getattr(config, "TOKEN_SAFETY_TRANSIENT_USE_FALLBACK", True)):
                    fallback = self._fallback_assessment(token_address, liquidity=liquidity)
                    fallback["source"] = "transient_fallback"
                    fallback["fail_reason"] = fail_key
                    return fallback
            self._api_fail += 1
            self._fail_closed += 1
            return {
                "token_address": token_address,
                "is_safe": False,
                "risk_level": "HIGH",
                "warnings": [f"Safety API unavailable (fail-closed): {fail_key}"],
                "warning_flags": 1,
                "source": "fail_closed",
                "fail_reason": fail_key,
            }

        # Fallback placeholder logic if external API is unavailable.
        self._api_fail += 1
        self._api_fail_total += 1
        self._mark_fail_reason(fail_reason)
        return self._fallback_assessment(token_address, liquidity=liquidity)

    async def _check_with_goplus(self, token_address: str) -> tuple[dict[str, Any] | None, str | None]:
        if not token_address:
            return None, "empty_token_address"

        headers = {}
        if GOPLUS_ACCESS_TOKEN:
            headers["Authorization"] = f"Bearer {GOPLUS_ACCESS_TOKEN}"

        params = {"contract_addresses": token_address}
        if CHAIN_ID.lower() != "solana":
            params["chain_id"] = EVM_CHAIN_ID

        try:
            url = GOPLUS_SOLANA_API if CHAIN_ID.lower() == "solana" else GOPLUS_EVM_API.format(chain_id=EVM_CHAIN_ID)
            result = await self._http.get_json(
                url,
                source="goplus",
                params=params,
                headers=headers,
                max_attempts=int(getattr(config, "HTTP_RETRY_ATTEMPTS", 3)),
            )
            if not result.ok or not isinstance(result.data, dict):
                if int(result.status or 0) == 429:
                    logger.warning("RATE_LIMIT source=goplus status=429")
                status = int(result.status or 0)
                if status == 429:
                    return None, "http_429"
                if 500 <= status <= 599:
                    return None, f"http_{status}"
                if 400 <= status <= 499:
                    return None, f"http_{status}"
                return None, str(result.error or "http_error")
            data = result.data
        except Exception as exc:
            return None, f"exception:{exc.__class__.__name__}"

        code = str(data.get("code", "")).strip()
        if code and code not in {"1", "200", "ok", "OK"}:
            return None, f"api_code_{code}"

        result_map = data.get("result") or {}
        if not isinstance(result_map, dict):
            return None, "bad_result_map"
        result = (
            result_map.get(token_address)
            or result_map.get(token_address.lower())
            or result_map.get(token_address.upper())
        )
        if not isinstance(result, dict):
            return None, "no_token_entry"

        warnings: list[str] = []
        risk_points = 0

        def _flag(name: str, points: int = 1) -> None:
            nonlocal risk_points
            warnings.append(name)
            risk_points += points

        if result.get("is_mintable") in ("1", 1, True):
            _flag("Mint is enabled", 1)

        if result.get("is_freezeable") in ("1", 1, True):
            _flag("Freeze authority enabled", 1)

        if result.get("is_blacklisted") in ("1", 1, True):
            _flag("Blacklist flag present", 2)

        if result.get("is_open_source") in ("0", 0, False):
            _flag("Contract not open source", 1)

        if result.get("can_take_back_ownership") in ("1", 1, True):
            _flag("Ownership can be reclaimed", 2)

        if result.get("owner_change_balance") in ("1", 1, True):
            _flag("Owner can change balances", 2)

        if result.get("trading_cooldown") in ("1", 1, True):
            _flag("Trading cooldown enabled", 1)

        if result.get("is_proxy") in ("1", 1, True):
            _flag("Proxy contract", 1)

        try:
            buy_tax = float(result.get("buy_tax") or 0)
            if buy_tax >= 10:
                _flag(f"High buy tax {buy_tax:.1f}%", 2)
            elif buy_tax > 3:
                _flag(f"Buy tax {buy_tax:.1f}%", 1)
        except (TypeError, ValueError):
            pass

        try:
            sell_tax = float(result.get("sell_tax") or 0)
            if sell_tax >= 10:
                _flag(f"High sell tax {sell_tax:.1f}%", 2)
            elif sell_tax > 3:
                _flag(f"Sell tax {sell_tax:.1f}%", 1)
        except (TypeError, ValueError):
            pass

        if risk_points >= 3:
            risk_level = "HIGH"
            is_safe = False
        elif risk_points >= 1:
            risk_level = "MEDIUM"
            is_safe = True
        else:
            risk_level = "LOW"
            is_safe = True

        return {
            "token_address": token_address,
            "is_safe": is_safe,
            "risk_level": risk_level,
            "warnings": warnings,
            "warning_flags": len(warnings),
            "source": "goplus",
        }, None
