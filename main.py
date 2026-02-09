"""Entry point for Solana Alert Bot."""

import asyncio
import logging
import os
from logging.handlers import RotatingFileHandler

from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from bot.handlers import (
    admin_approve_card_command,
    admin_grant_command,
    admin_pending_cards_command,
    admin_reject_card_command,
    admin_stats_command,
    demo_command,
    handle_admin_card_callback,
    handle_text_menu,
    handle_menu_callback,
    handle_payment_callback,
    payment_client,
    settings_command,
    start_command,
    status_command,
    subscribe_command,
    test_alert_command,
)
from config import APP_LOG_FILE, LOG_LEVEL, LOG_DIR, SCAN_INTERVAL, TELEGRAM_BOT_TOKEN
from database.db import get_all_subscribed_users, init_db
from monitor.alerter import TokenAlerter
from monitor.dexscreener import DexScreenerMonitor
from payments.webhook_server import CryptoWebhookServer


def configure_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = RotatingFileHandler(APP_LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Avoid leaking bot token in verbose transport logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)


logger = logging.getLogger(__name__)


async def monitoring_loop(application: Application) -> None:
    monitor = DexScreenerMonitor()
    alerter = TokenAlerter()

    while True:
        try:
            tokens = await monitor.fetch_new_tokens()
            if tokens:
                users = get_all_subscribed_users()
                if users:
                    for token in tokens:
                        await alerter.send_alert(application.bot, token, users)
        except Exception:
            logger.exception("Monitoring loop error")

        await asyncio.sleep(SCAN_INTERVAL)


async def post_init(application: Application) -> None:
    application.bot_data["monitor_task"] = asyncio.create_task(monitoring_loop(application))

    webhook_server = CryptoWebhookServer(payment_client)
    await webhook_server.start()
    application.bot_data["webhook_server"] = webhook_server


async def post_shutdown(application: Application) -> None:
    task = application.bot_data.get("monitor_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    webhook_server: CryptoWebhookServer | None = application.bot_data.get("webhook_server")
    if webhook_server:
        await webhook_server.stop()

    await payment_client.close()


def main() -> None:
    configure_logging()
    init_db()

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).post_shutdown(post_shutdown).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("demo", demo_command))
    app.add_handler(CommandHandler("subscribe", subscribe_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("testalert", test_alert_command))
    app.add_handler(CommandHandler("admin_stats", admin_stats_command))
    app.add_handler(CommandHandler("admin_grant", admin_grant_command))
    app.add_handler(CommandHandler("admin_pending_cards", admin_pending_cards_command))
    app.add_handler(CommandHandler("admin_approve_card", admin_approve_card_command))
    app.add_handler(CommandHandler("admin_reject_card", admin_reject_card_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_menu))

    app.add_handler(CallbackQueryHandler(handle_admin_card_callback, pattern=r"^(approve_card_|reject_card_)"))
    app.add_handler(CallbackQueryHandler(handle_payment_callback, pattern=r"^(pay_|card_)"))
    app.add_handler(
        CallbackQueryHandler(
            handle_menu_callback,
            pattern=r"^(demo|subscribe|status|settings|main_menu|toggle_notify|set_liquidity.*|set_max_age.*|set_interval.*)$",
        )
    )

    app.run_polling()


if __name__ == "__main__":
    main()
