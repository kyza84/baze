"""Token safety checks with optional GoPlus integration and fallback heuristics."""

from __future__ import annotations

import logging
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
        self._fail_closed = 0
        self._api_fail_reasons: dict[str, int] = {}

    async def close(self) -> None:
        await self._http.close()

    def runtime_stats(self, reset: bool = False) -> dict[str, int | float]:
        total = int(self._checks_total)
        fail = int(self._api_fail)
        err_pct = (float(fail) / total * 100.0) if total > 0 else 0.0
        top_reason = "none"
        top_count = 0
        if self._api_fail_reasons:
            top_reason, top_count = max(self._api_fail_reasons.items(), key=lambda kv: int(kv[1]))
        out = {
            "checks_total": total,
            "api_ok": int(self._api_ok),
            "api_fail": fail,
            "fail_closed": int(self._fail_closed),
            "api_error_percent": round(err_pct, 2),
            "fail_reason_top": top_reason,
            "fail_reason_top_count": int(top_count),
            "fail_reason_counts": dict(self._api_fail_reasons),
        }
        if reset:
            self._checks_total = 0
            self._api_ok = 0
            self._api_fail = 0
            self._fail_closed = 0
            self._api_fail_reasons = {}
        return out

    def _mark_fail_reason(self, reason: str | None) -> str:
        key = str(reason or "unknown").strip().lower() or "unknown"
        self._api_fail_reasons[key] = int(self._api_fail_reasons.get(key, 0)) + 1
        return key

    async def check_token_safety(self, token_address: str, liquidity: float | None = None) -> dict[str, Any]:
        self._checks_total += 1
        api_result, fail_reason = await self._check_with_goplus(token_address)
        if api_result:
            self._api_ok += 1
            return api_result

        # Stability-first mode: if we cannot validate safety with an external API, do not trade.
        if bool(getattr(config, "TOKEN_SAFETY_FAIL_CLOSED", False)):
            self._api_fail += 1
            self._fail_closed += 1
            fail_key = self._mark_fail_reason(fail_reason)
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
        self._mark_fail_reason(fail_reason)
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
