"""CryptoBot payment integration."""

import logging
from typing import Any, Optional

from aiocryptopay import AioCryptoPay, Networks

from config import CRYPTOBOT_TOKEN, PLAN_DURATIONS_DAYS, PRICES
from database.db import create_invoice_record, get_invoice, mark_invoice_paid, update_user_subscription

logger = logging.getLogger(__name__)


class CryptoPayment:
    def __init__(self) -> None:
        self.client = AioCryptoPay(token=CRYPTOBOT_TOKEN, network=Networks.MAIN_NET)

    async def create_invoice(self, user_id: int, plan_type: str) -> Optional[dict[str, Any]]:
        if plan_type not in PLAN_DURATIONS_DAYS:
            return None

        payload = f"{user_id}:{plan_type}"
        try:
            invoice = await self.client.create_invoice(
                asset="USDT",
                amount=PRICES[plan_type],
                description=f"Solana Alert Bot - {plan_type}",
                payload=payload,
            )
            invoice_id = getattr(invoice, "invoice_id", None)
            if invoice_id is not None:
                create_invoice_record(
                    telegram_id=user_id,
                    invoice_id=int(invoice_id),
                    plan_type=plan_type,
                    amount=float(PRICES[plan_type]),
                    payload=payload,
                )
            return {
                "invoice_url": getattr(invoice, "bot_invoice_url", None),
                "invoice_id": invoice_id,
            }
        except Exception:
            logger.exception("Failed to create invoice")
            return None

    async def check_payment(self, invoice_id: int) -> bool:
        try:
            invoices = await self.client.get_invoices(invoice_ids=invoice_id)
            items = getattr(invoices, "items", [])
            if not items:
                return False
            status = str(getattr(items[0], "status", "")).lower()
            return status == "paid"
        except Exception:
            logger.exception("Failed to check payment for invoice_id=%s", invoice_id)
            return False

    async def handle_payment_webhook(self, update: dict[str, Any]) -> bool:
        try:
            invoice_data = self._extract_invoice_payload(update)
            if not invoice_data:
                return False

            status = str(invoice_data.get("status", "")).lower()
            if status != "paid":
                return False

            invoice_id = int(invoice_data.get("invoice_id"))
            invoice = get_invoice(invoice_id)
            if not invoice:
                logger.warning("Webhook ignored for unknown invoice_id=%s", invoice_id)
                return False

            # Idempotency: already processed means success, but no duplicate extension.
            if not mark_invoice_paid(invoice_id, update):
                return True

            days = PLAN_DURATIONS_DAYS.get(invoice.plan_type)
            if not days:
                logger.error("Unknown plan_type in invoice %s: %s", invoice_id, invoice.plan_type)
                return False

            user = update_user_subscription(invoice.telegram_id, days)
            return user is not None
        except Exception:
            logger.exception("Failed to handle payment webhook")
            return False

    def _extract_invoice_payload(self, update: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not isinstance(update, dict):
            return None

        if "invoice_id" in update:
            return update

        payload = update.get("payload")
        if isinstance(payload, dict) and "invoice_id" in payload:
            return payload

        data = update.get("data")
        if isinstance(data, dict) and "invoice_id" in data:
            return data

        return None

    async def close(self) -> None:
        await self.client.close()
