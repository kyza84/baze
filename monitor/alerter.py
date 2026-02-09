"""Alert delivery for new tokens."""

import asyncio
import logging
from datetime import datetime, timezone
from html import escape
from typing import Any, Iterable

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from monitor.token_checker import TokenChecker

logger = logging.getLogger(__name__)


class TokenAlerter:
    def __init__(self, max_concurrency: int = 20) -> None:
        self.token_checker = TokenChecker()
        self.max_concurrency = max_concurrency
        self.last_alert_sent_at: dict[int, datetime] = {}

    async def send_alert(self, bot, token_data: dict[str, Any], users: Iterable[Any]) -> None:
        safety = await self.token_checker.check_token_safety(
            token_data.get("address", ""),
            token_data.get("liquidity", 0),
        )
        token_data = {**token_data, "risk": safety}

        message = self._format_alert_message(token_data)
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("DexScreener", url=token_data.get("dexscreener_url", "https://dexscreener.com")),
                    InlineKeyboardButton("Pump.fun", url=token_data.get("pumpfun_url", "https://pump.fun")),
                ]
            ]
        )

        filtered_users = [u for u in users if self._should_notify_user(u, token_data)]
        if not filtered_users:
            return

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _send(u: Any) -> str:
            async with semaphore:
                chat_id = int(getattr(u, "telegram_id", 0))
                now = datetime.now(timezone.utc)
                settings = getattr(u, "settings", {}) or {}
                cooldown_seconds = int(settings.get("alert_cooldown_seconds", 30))
                last_sent_at = self.last_alert_sent_at.get(chat_id)
                if last_sent_at and (now - last_sent_at).total_seconds() < cooldown_seconds:
                    return "skipped"
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode="HTML",
                        reply_markup=keyboard,
                        disable_web_page_preview=True,
                    )
                    self.last_alert_sent_at[chat_id] = now
                    return "sent"
                except Exception as exc:
                    logger.warning("Alert send failed for chat_id=%s: %s", getattr(u, "telegram_id", "unknown"), exc)
                    return "failed"

        results = await asyncio.gather(*[_send(u) for u in filtered_users])
        success = sum(1 for r in results if r == "sent")
        failed = sum(1 for r in results if r == "failed")
        skipped = sum(1 for r in results if r == "skipped")
        logger.info(
            "Alert dispatch token=%s users=%s success=%s failed=%s skipped=%s risk=%s",
            token_data.get("symbol", "N/A"),
            len(filtered_users),
            success,
            failed,
            skipped,
            (token_data.get("risk") or {}).get("risk_level", "N/A"),
        )

    def _should_notify_user(self, user: Any, token_data: dict[str, Any]) -> bool:
        settings = getattr(user, "settings", {}) or {}
        if not settings.get("notify_enabled", True):
            return False

        user_min_liq = float(settings.get("min_liquidity", 5000))
        if float(token_data.get("liquidity", 0)) < user_min_liq:
            return False

        max_age_minutes = int(settings.get("max_age_minutes", 60))
        created_at = token_data.get("created_at")
        if isinstance(created_at, datetime):
            now = datetime.now(timezone.utc)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_minutes = (now - created_at).total_seconds() / 60
            if age_minutes > max_age_minutes:
                return False

        return True

    def _format_alert_message(self, token_data: dict[str, Any]) -> str:
        created_at = token_data.get("created_at")
        minutes_ago = self._minutes_ago(created_at)

        risk = token_data.get("risk", {})
        risk_level = risk.get("risk_level", "MEDIUM")
        warnings = risk.get("warnings", [])
        warning_text = ", ".join(warnings) if warnings else "No immediate flags"

        name = escape(str(token_data.get("name", "Unknown")))
        symbol = escape(str(token_data.get("symbol", "N/A")))
        address = escape(str(token_data.get("address", "")))

        return (
            "🚀 <b>New Solana Token Alert</b>\n\n"
            f"<b>{name}</b> ({symbol})\n"
            f"<code>{address}</code>\n\n"
            f"💰 <b>Liquidity:</b> ${token_data.get('liquidity', 0):,.2f}\n"
            f"📊 <b>5m Volume:</b> ${token_data.get('volume_5m', 0):,.2f}\n"
            f"📈 <b>5m Change:</b> {token_data.get('price_change_5m', 0):.2f}%\n"
            f"⏰ <b>Created:</b> {minutes_ago} minutes ago\n"
            f"⚠️ <b>Risk:</b> {risk_level} ({escape(warning_text)})\n\n"
            "🔗 Use the buttons below for quick access."
        )

    @staticmethod
    def _minutes_ago(created_at: datetime | None) -> int:
        if not isinstance(created_at, datetime):
            return 0
        now = datetime.now(timezone.utc)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        delta = now - created_at
        return max(0, int(delta.total_seconds() // 60))
