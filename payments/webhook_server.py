"""CryptoBot webhook HTTP server."""

import logging

from aiohttp import web

from config import CRYPTOBOT_WEBHOOK_PATH, CRYPTOBOT_WEBHOOK_SECRET, WEBHOOK_HOST, WEBHOOK_PORT
from payments.cryptobot import CryptoPayment

logger = logging.getLogger(__name__)


class CryptoWebhookServer:
    def __init__(self, payment: CryptoPayment) -> None:
        self.payment = payment
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

    async def start(self) -> None:
        if not CRYPTOBOT_WEBHOOK_SECRET:
            raise RuntimeError("CRYPTOBOT_WEBHOOK_SECRET is required for secure payment webhooks")

        app = web.Application()
        app.router.add_post(CRYPTOBOT_WEBHOOK_PATH, self._handle_webhook)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, WEBHOOK_HOST, WEBHOOK_PORT)
        await self.site.start()
        logger.info("Crypto webhook server listening on %s:%s%s", WEBHOOK_HOST, WEBHOOK_PORT, CRYPTOBOT_WEBHOOK_PATH)

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        header_secret = request.headers.get("X-Webhook-Secret", "")
        query_secret = request.query.get("secret", "")
        if header_secret != CRYPTOBOT_WEBHOOK_SECRET and query_secret != CRYPTOBOT_WEBHOOK_SECRET:
            return web.json_response({"ok": False, "error": "unauthorized"}, status=401)

        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

        processed = await self.payment.handle_payment_webhook(payload)
        return web.json_response({"ok": processed})
