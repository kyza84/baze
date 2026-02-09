"""Alert delivery for new tokens."""

import asyncio
import logging
from datetime import datetime, timezone
from html import escape
from typing import Any, Iterable

import config
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from monitor.personal_stats import record_alert
from monitor.token_checker import TokenChecker
from monitor.token_scorer import TokenScorer

logger = logging.getLogger(__name__)


class TokenAlerter:
    def __init__(self, max_concurrency: int = 20) -> None:
        self.token_checker = TokenChecker()
        self.token_scorer = TokenScorer()
        self.max_concurrency = max_concurrency
        self.last_alert_sent_at: dict[int, datetime] = {}

    async def send_alert(
        self,
        bot,
        token_data: dict[str, Any],
        users: Iterable[Any],
        score_data: dict[str, Any] | None = None,
    ) -> int:
        safety = await self.token_checker.check_token_safety(
            token_data.get("address", ""),
            token_data.get("liquidity", 0),
        )
        risk_level = (safety or {}).get("risk_level", "HIGH")
        token_data = {**token_data, "risk": safety, "risk_level": risk_level}

        score_data = score_data or self.token_scorer.calculate_score(token_data)
        token_data["score_data"] = score_data

        if config.PERSONAL_MODE and int(score_data.get("score", 0)) < int(config.MIN_TOKEN_SCORE):
            return 0

        message = self._format_alert_message(token_data)
        keyboard = self._build_keyboard(token_data)

        if config.PERSONAL_MODE:
            personal_id = int(config.PERSONAL_TELEGRAM_ID or 0)
            filtered_users = [u for u in users if int(getattr(u, "telegram_id", 0)) == personal_id]
        else:
            filtered_users = [u for u in users if self._should_notify_user(u, token_data)]

        if not filtered_users:
            return 0

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
                    record_alert(chat_id, int(score_data.get("score", 0)))
                    return "sent"
                except Exception as exc:
                    logger.warning("Alert send failed for chat_id=%s: %s", getattr(u, "telegram_id", "unknown"), exc)
                    return "failed"

        results = await asyncio.gather(*[_send(u) for u in filtered_users])
        success = sum(1 for r in results if r == "sent")
        failed = sum(1 for r in results if r == "failed")
        skipped = sum(1 for r in results if r == "skipped")
        logger.info(
            "Alert dispatch token=%s users=%s success=%s failed=%s skipped=%s risk=%s score=%s",
            token_data.get("symbol", "N/A"),
            len(filtered_users),
            success,
            failed,
            skipped,
            (token_data.get("risk") or {}).get("risk_level", "N/A"),
            score_data.get("score", 0),
        )
        return success

    def _should_notify_user(self, user: Any, token_data: dict[str, Any]) -> bool:
        settings = getattr(user, "settings", {}) or {}
        if not settings.get("notify_enabled", True):
            return False

        user_min_liq = float(settings.get("min_liquidity", 5000))
        if float(token_data.get("liquidity", 0)) < user_min_liq:
            return False

        max_age_minutes = int(settings.get("max_age_minutes", 60))
        age_minutes = int(token_data.get("age_minutes") or self._minutes_ago(token_data.get("created_at")))
        if age_minutes > max_age_minutes:
            return False

        return True

    def _build_keyboard(self, token_data: dict[str, Any]) -> InlineKeyboardMarkup:
        token_address = str(token_data.get("address", ""))
        buy_url = config.SWAP_URL_TEMPLATE.format(
            token_address=token_address,
            amount=config.BUY_AMOUNT_ETH,
            chain=config.CHAIN_NAME,
        )
        chart_url = token_data.get("dexscreener_url") or "https://dexscreener.com"
        details_url = (
            config.EXPLORER_TOKEN_URL_TEMPLATE.format(token_address=token_address)
            if token_address
            else "https://basescan.org"
        )

        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(f"\U0001F680 Buy {config.BUY_AMOUNT_ETH} ETH", url=buy_url),
                    InlineKeyboardButton("\U0001F4CA Chart", url=chart_url),
                ],
                [
                    InlineKeyboardButton("\U0001F50D Details", url=details_url),
                    InlineKeyboardButton("\u274C Skip", callback_data=f"skip_{token_address}"),
                ],
            ]
        )

    def _format_alert_message(self, token_data: dict[str, Any]) -> str:
        score_data = token_data.get("score_data", {}) or {}
        breakdown = score_data.get("breakdown", {}) or {}

        name = escape(str(token_data.get("name", "Unknown")))
        symbol = escape(str(token_data.get("symbol", "N/A")))
        address = escape(str(token_data.get("address", "")))
        liquidity = float(token_data.get("liquidity", 0) or 0)
        volume_5m = float(token_data.get("volume_5m", 0) or 0)
        age_minutes = int(token_data.get("age_minutes") or self._minutes_ago(token_data.get("created_at")))
        price_change_5m = float(token_data.get("price_change_5m", 0) or 0)
        risk_level = escape(str((token_data.get("risk") or {}).get("risk_level", "MEDIUM")))

        return (
            "\U0001F680 NEW TOKEN ALERT\n\n"
            f"Name: {name}\n"
            f"Symbol: {symbol}\n"
            f"Address: <code>{address}</code>\n\n"
            f"\U0001F4B0 Liquidity: ${liquidity:,.0f}\n"
            f"\U0001F4CA Volume 5m: ${volume_5m:,.0f}\n"
            f"\u23F0 Age: {age_minutes} minutes old\n"
            f"\U0001F4C8 Price change 5m: {price_change_5m:+.1f}%\n\n"
            f"\u2B50 SCORE: {int(score_data.get('score', 0))}/100 - {escape(str(score_data.get('recommendation', 'SKIP')))}\n\n"
            "Breakdown:\n\n"
            f"- Liquidity: {int(breakdown.get('liquidity_score', 0))}/30\n\n"
            f"- Volume: {int(breakdown.get('volume_score', 0))}/30\n\n"
            f"- Risk: {int(breakdown.get('risk_score', 0))}/25\n\n"
            f"- Freshness: {int(breakdown.get('age_score', 0))}/15\n\n"
            f"\u26A0\uFE0F Risk: {risk_level}"
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
