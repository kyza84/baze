"""Local desktop alert sink (no Telegram)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from monitor.token_checker import TokenChecker
from utils.log_contracts import local_alert_event

logger = logging.getLogger(__name__)


class LocalAlerter:
    def __init__(self, alerts_file: str) -> None:
        self.alerts_file = alerts_file
        self.token_checker = TokenChecker()
        os.makedirs(os.path.dirname(self.alerts_file), exist_ok=True)

    @staticmethod
    def _run_tag() -> str:
        run_tag = str(os.getenv("RUN_TAG", "") or "").strip()
        if run_tag:
            return run_tag
        return str(os.getenv("BOT_INSTANCE_ID", "") or "").strip()

    async def close(self) -> None:
        await self.token_checker.close()

    async def send_alert(
        self,
        token_data: dict[str, Any],
        score_data: dict[str, Any],
        safety: dict[str, Any] | None = None,
    ) -> int:
        if safety is None:
            safety = await self.token_checker.check_token_safety(
                token_data.get("address", ""),
                token_data.get("liquidity", 0),
            )
        risk_level = (safety or {}).get("risk_level", "HIGH")
        warning_flags = len((safety or {}).get("warnings") or [])
        now = datetime.now(timezone.utc)

        record = {
            "timestamp": now.isoformat(),
            "name": str(token_data.get("name", "Unknown")),
            "symbol": str(token_data.get("symbol", "N/A")),
            "address": str(token_data.get("address", "")),
            "liquidity": float(token_data.get("liquidity") or 0),
            "volume_5m": float(token_data.get("volume_5m") or 0),
            "price_change_5m": float(token_data.get("price_change_5m") or 0),
            "age_minutes": int(token_data.get("age_minutes") or 0),
            "score": int(score_data.get("score", 0)),
            "recommendation": str(score_data.get("recommendation", "SKIP")),
            "risk_level": str(risk_level),
            "warning_flags": warning_flags,
            "breakdown": score_data.get("breakdown", {}),
        }
        record = local_alert_event(record, run_tag=self._run_tag())

        with open(self.alerts_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "Local alert token=%s address=%s score=%s reco=%s risk=%s warnings=%s liq=%.0f vol5m=%.0f",
            record["symbol"],
            record["address"],
            record["score"],
            record["recommendation"],
            record["risk_level"],
            record["warning_flags"],
            record["liquidity"],
            record["volume_5m"],
        )
        return 1

    async def send_event(self, event: dict[str, Any]) -> int:
        now = datetime.now(timezone.utc)
        payload = dict(event or {})
        payload.setdefault("timestamp", now.isoformat())
        payload.setdefault("name", "SYSTEM")
        payload.setdefault("symbol", str(payload.get("symbol", "SYSTEM")))
        payload.setdefault("address", "")
        payload.setdefault("liquidity", 0.0)
        payload.setdefault("volume_5m", 0.0)
        payload.setdefault("price_change_5m", 0.0)
        payload.setdefault("age_minutes", 0)
        payload.setdefault("score", int(payload.get("score", 0) or 0))
        payload.setdefault("recommendation", str(payload.get("recommendation", "INFO")))
        payload.setdefault("risk_level", str(payload.get("risk_level", "INFO")))
        payload.setdefault("warning_flags", int(payload.get("warning_flags", 0) or 0))
        payload.setdefault("breakdown", {})
        payload = local_alert_event(payload, run_tag=self._run_tag())
        with open(self.alerts_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        logger.warning(
            "Local event symbol=%s reco=%s risk=%s",
            payload.get("symbol", "SYSTEM"),
            payload.get("recommendation", "INFO"),
            payload.get("risk_level", "INFO"),
        )
        return 1
