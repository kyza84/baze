"""Token safety checks with optional GoPlus integration and fallback heuristics."""

from typing import Any

import aiohttp

from config import CHAIN_ID, EVM_CHAIN_ID, GOPLUS_ACCESS_TOKEN, GOPLUS_EVM_API, GOPLUS_SOLANA_API


class TokenChecker:
    async def check_token_safety(self, token_address: str, liquidity: float | None = None) -> dict[str, Any]:
        api_result = await self._check_with_goplus(token_address)
        if api_result:
            return api_result

        # Fallback placeholder logic if external API is unavailable.
        warnings: list[str] = []
        risk_level = "MEDIUM"
        is_safe = True

        if liquidity is not None and liquidity < 10000:
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

        if result.get("is_mintable") in ("1", 1, True):
            warnings.append("Mint is enabled")
            risk_points += 1

        if result.get("is_freezeable") in ("1", 1, True):
            warnings.append("Freeze authority enabled")
            risk_points += 1

        if result.get("is_blacklisted") in ("1", 1, True):
            warnings.append("Blacklist flag present")
            risk_points += 2

        if result.get("is_open_source") in ("0", 0, False):
            warnings.append("Contract not open source")
            risk_points += 1

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
            "source": "goplus",
        }
