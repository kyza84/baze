"""Token safety checks with optional GoPlus integration and fallback heuristics."""

from typing import Any

import aiohttp

from config import CHAIN_ID, EVM_CHAIN_ID, GOPLUS_ACCESS_TOKEN, GOPLUS_EVM_API, GOPLUS_SOLANA_API
import config


class TokenChecker:
    async def check_token_safety(self, token_address: str, liquidity: float | None = None) -> dict[str, Any]:
        api_result = await self._check_with_goplus(token_address)
        if api_result:
            return api_result

        # Stability-first mode: if we cannot validate safety with an external API, do not trade.
        if bool(getattr(config, "TOKEN_SAFETY_FAIL_CLOSED", False)):
            return {
                "token_address": token_address,
                "is_safe": False,
                "risk_level": "HIGH",
                "warnings": ["Safety API unavailable (fail-closed)"],
                "warning_flags": 1,
                "source": "fail_closed",
            }

        # Fallback placeholder logic if external API is unavailable.
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

    async def _check_with_goplus(self, token_address: str) -> dict[str, Any] | None:
        if not token_address:
            return None

        headers = {}
        if GOPLUS_ACCESS_TOKEN:
            headers["Authorization"] = f"Bearer {GOPLUS_ACCESS_TOKEN}"

        params = {"contract_addresses": token_address}
        if CHAIN_ID.lower() != "solana":
            params["chain_id"] = EVM_CHAIN_ID

        try:
            url = GOPLUS_SOLANA_API if CHAIN_ID.lower() == "solana" else GOPLUS_EVM_API.format(chain_id=EVM_CHAIN_ID)
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        return None
                    data = await response.json()
        except Exception:
            return None

        result_map = data.get("result") or {}
        result = (
            result_map.get(token_address)
            or result_map.get(token_address.lower())
            or result_map.get(token_address.upper())
        )
        if not isinstance(result, dict):
            return None

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
        }
