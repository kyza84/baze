"""Telegram handlers."""

from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from bot.keyboards import (
    BTN_SETTINGS,
    BTN_STATUS,
    BTN_SUBSCRIBE,
    BTN_TRIAL,
    interval_options_keyboard,
    liquidity_options_keyboard,
    max_age_options_keyboard,
    main_menu_keyboard,
    settings_keyboard,
    subscription_keyboard,
)
from bot.messages import DEMO_ACTIVATED, SETTINGS_INFO, STATUS_TEMPLATE, SUBSCRIPTION_INFO, WELCOME_MESSAGE
from config import ADMIN_IDS, CARD_PAYMENT_DETAILS, PLAN_DURATIONS_DAYS, PRICES
from database.db import (
    create_manual_payment_request,
    get_admin_stats,
    get_db,
    get_manual_payment_request,
    get_or_create_user,
    get_user,
    grant_user_subscription,
    list_pending_manual_payment_requests,
    set_manual_payment_status,
    update_user_settings,
)
from database.models import User
from monitor.alerter import TokenAlerter
from payments.cryptobot import CryptoPayment

payment_client = CryptoPayment()


def _format_time_left(subscription_until) -> str:
    if not subscription_until:
        return "0m"
    delta = subscription_until - datetime.utcnow()
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return "0m"

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _is_admin(user_id: int | None) -> bool:
    return bool(user_id and user_id in ADMIN_IDS)


def _is_paid_subscription(user: User) -> bool:
    return user.is_subscribed() and not user.is_demo


def _admin_card_actions_keyboard(request_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Approve", callback_data=f"approve_card_{request_id}"),
                InlineKeyboardButton("❌ Reject", callback_data=f"reject_card_{request_id}"),
            ]
        ]
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not update.message:
        return

    get_or_create_user(user.id, user.username)
    await update.message.reply_text(
        WELCOME_MESSAGE,
        parse_mode="HTML",
        reply_markup=main_menu_keyboard(),
    )


async def demo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_user = update.effective_user
    if not tg_user:
        return

    get_or_create_user(tg_user.id, tg_user.username)
    db = get_db()
    try:
        db_user = db.query(User).filter(User.telegram_id == tg_user.id).first()
        if not db_user:
            return

        # Do not overwrite active paid subscriptions with trial.
        if db_user.is_subscribed() and not db_user.is_demo:
            target = update.message or (update.callback_query.message if update.callback_query else None)
            if target:
                await target.reply_text("You already have an active paid subscription.")
            return

        if db_user.trial_used:
            target = update.message or (update.callback_query.message if update.callback_query else None)
            if target:
                await target.reply_text("Free trial is available only once per user.")
            return

        db_user.activate_demo()
        db.commit()
    finally:
        db.close()

    target = update.message or (update.callback_query.message if update.callback_query else None)
    if target:
        await target.reply_text(DEMO_ACTIVATED, parse_mode="HTML", reply_markup=main_menu_keyboard())


async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    target = update.message or (update.callback_query.message if update.callback_query else None)
    if not target:
        return

    await target.reply_text(
        SUBSCRIPTION_INFO,
        parse_mode="HTML",
        reply_markup=subscription_keyboard(),
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_user = update.effective_user
    if not tg_user:
        return

    user = get_or_create_user(tg_user.id, tg_user.username)
    time_left = _format_time_left(user.subscription_until)

    if user.is_subscribed():
        plan_type = "Trial" if user.is_demo else "Paid"
    else:
        plan_type = "Inactive"

    settings_text = (
        f"\n\nSettings:\n"
        f"- Min liquidity: ${user.settings.get('min_liquidity', 5000)}\n"
        f"- Max age: {user.settings.get('max_age_minutes', 60)} min\n"
        f"- Alert interval: {user.settings.get('alert_cooldown_seconds', 30)} sec\n"
        f"- Notifications: {'ON' if user.settings.get('notify_enabled', True) else 'OFF'}"
    )

    message = STATUS_TEMPLATE.format(time_left=time_left, plan_type=plan_type) + settings_text
    target = update.message or (update.callback_query.message if update.callback_query else None)
    if target:
        await target.reply_text(message, parse_mode="HTML", reply_markup=main_menu_keyboard())


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_user = update.effective_user
    if not tg_user:
        return

    user = get_or_create_user(tg_user.id, tg_user.username)
    target = update.message or (update.callback_query.message if update.callback_query else None)
    paid_enabled = _is_paid_subscription(user)
    if target:
        await target.reply_text(
            SETTINGS_INFO
            + ("\n\n🔒 Advanced settings are available for paid subscriptions only." if not paid_enabled else ""),
            parse_mode="HTML",
            reply_markup=settings_keyboard(user.settings or {}, paid_enabled=paid_enabled),
        )


async def test_alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_user = update.effective_user
    if not tg_user or not update.message:
        return

    user = get_or_create_user(tg_user.id, tg_user.username)
    alerter = TokenAlerter()
    token_data = {
        "name": "Test Token",
        "symbol": "TST",
        "address": "So11111111111111111111111111111111111111112",
        "liquidity": 12500.0,
        "volume_5m": 3400.0,
        "price_change_5m": 12.7,
        "dexscreener_url": "https://dexscreener.com/solana",
        "pumpfun_url": "https://pump.fun",
        "created_at": datetime.utcnow(),
    }
    await alerter.send_alert(context.bot, token_data, [user])
    await update.message.reply_text("Test alert sent.")


async def admin_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_user = update.effective_user
    target = update.message
    if not tg_user or not target:
        return

    if not _is_admin(tg_user.id):
        await target.reply_text("Access denied.")
        return

    stats = get_admin_stats()
    await target.reply_text(
        (
            "📈 <b>Admin Stats</b>\n\n"
            f"Total users: <b>{stats['total_users']}</b>\n"
            f"Active subscriptions: <b>{stats['active_subscriptions']}</b>\n"
            f"Active trial: <b>{stats['active_demo']}</b>\n"
            f"Active paid: <b>{stats['paid_active']}</b>\n"
            f"Paid invoices: <b>{stats['paid_invoices']}</b>\n"
            f"Pending invoices: <b>{stats['pending_invoices']}</b>\n"
            f"Pending card requests: <b>{stats['pending_manual_requests']}</b>"
        ),
        parse_mode="HTML",
    )


async def admin_grant_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_user = update.effective_user
    target = update.message
    if not tg_user or not target:
        return

    if not _is_admin(tg_user.id):
        await target.reply_text("Access denied.")
        return

    if len(context.args) < 2:
        await target.reply_text("Usage: /admin_grant <telegram_id> <days>")
        return

    try:
        telegram_id = int(context.args[0])
        days = float(context.args[1])
    except ValueError:
        await target.reply_text("Invalid arguments. Example: /admin_grant 123456789 7")
        return

    user = grant_user_subscription(telegram_id=telegram_id, username=None, days=days)
    await target.reply_text(f"Granted {days} days to user {user.telegram_id}.")


async def admin_pending_cards_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_user = update.effective_user
    target = update.message
    if not tg_user or not target:
        return
    if not _is_admin(tg_user.id):
        await target.reply_text("Access denied.")
        return

    items = list_pending_manual_payment_requests(limit=30)
    if not items:
        await target.reply_text("No pending card payment requests.")
        return

    for item in items:
        await target.reply_text(
            (
                f"🧾 <b>Request #{item.id}</b>\n"
                f"User: <b>{item.telegram_id}</b> @{item.username or '-'}\n"
                f"Plan: <b>{item.plan_type}</b>\n"
                f"Amount: <b>${item.amount}</b>\n"
                f"Created: <b>{item.created_at:%Y-%m-%d %H:%M}</b>"
            ),
            parse_mode="HTML",
            reply_markup=_admin_card_actions_keyboard(item.id),
        )


async def admin_approve_card_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_user = update.effective_user
    target = update.message
    if not tg_user or not target:
        return
    if not _is_admin(tg_user.id):
        await target.reply_text("Access denied.")
        return
    if len(context.args) < 1:
        await target.reply_text("Usage: /admin_approve_card <request_id>")
        return

    try:
        request_id = int(context.args[0])
    except ValueError:
        await target.reply_text("request_id must be integer.")
        return

    req = get_manual_payment_request(request_id)
    if not req:
        await target.reply_text("Request not found.")
        return
    if req.status != "pending":
        await target.reply_text(f"Request already processed with status: {req.status}")
        return

    days = PLAN_DURATIONS_DAYS.get(req.plan_type)
    if not days:
        await target.reply_text("Unknown plan in request.")
        return

    grant_user_subscription(telegram_id=req.telegram_id, username=req.username, days=days)
    set_manual_payment_status(request_id, "approved", tg_user.id)
    await target.reply_text(f"Request #{request_id} approved. Subscription activated for user {req.telegram_id}.")

    try:
        await context.bot.send_message(
            chat_id=req.telegram_id,
            text=f"✅ Your card payment request #{request_id} was approved. Subscription activated.",
        )
    except Exception:
        pass


async def admin_reject_card_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_user = update.effective_user
    target = update.message
    if not tg_user or not target:
        return
    if not _is_admin(tg_user.id):
        await target.reply_text("Access denied.")
        return
    if len(context.args) < 1:
        await target.reply_text("Usage: /admin_reject_card <request_id> [reason]")
        return

    try:
        request_id = int(context.args[0])
    except ValueError:
        await target.reply_text("request_id must be integer.")
        return

    req = get_manual_payment_request(request_id)
    if not req:
        await target.reply_text("Request not found.")
        return
    if req.status != "pending":
        await target.reply_text(f"Request already processed with status: {req.status}")
        return

    reason = " ".join(context.args[1:]).strip() if len(context.args) > 1 else None
    set_manual_payment_status(request_id, "rejected", tg_user.id, comment=reason)
    await target.reply_text(f"Request #{request_id} rejected.")

    try:
        msg = f"❌ Your card payment request #{request_id} was rejected."
        if reason:
            msg += f"\nReason: {reason}"
        await context.bot.send_message(chat_id=req.telegram_id, text=msg)
    except Exception:
        pass


async def handle_admin_card_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    admin = update.effective_user
    if not query or not query.data or not admin:
        return

    await query.answer()
    if not _is_admin(admin.id):
        await query.message.reply_text("Access denied.")
        return

    action = "approve" if query.data.startswith("approve_card_") else "reject"
    request_id_raw = query.data.replace("approve_card_", "").replace("reject_card_", "")
    if not request_id_raw.isdigit():
        await query.message.reply_text("Invalid request ID.")
        return
    request_id = int(request_id_raw)

    req = get_manual_payment_request(request_id)
    if not req:
        await query.message.reply_text("Request not found.")
        return
    if req.status != "pending":
        await query.message.reply_text(f"Request already processed: {req.status}")
        return

    if action == "approve":
        days = PLAN_DURATIONS_DAYS.get(req.plan_type)
        if not days:
            await query.message.reply_text("Unknown plan in request.")
            return
        grant_user_subscription(telegram_id=req.telegram_id, username=req.username, days=days)
        set_manual_payment_status(request_id, "approved", admin.id)
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(f"✅ Request #{request_id} approved.")
        try:
            await context.bot.send_message(
                chat_id=req.telegram_id,
                text=f"✅ Your card payment request #{request_id} was approved. Subscription activated.",
            )
        except Exception:
            pass
        return

    set_manual_payment_status(request_id, "rejected", admin.id, comment="Rejected by admin button")
    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text(f"❌ Request #{request_id} rejected.")
    try:
        await context.bot.send_message(
            chat_id=req.telegram_id,
            text=f"❌ Your card payment request #{request_id} was rejected.",
        )
    except Exception:
        pass


async def handle_payment_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data or not update.effective_user:
        return

    await query.answer()
    callback = query.data
    user = update.effective_user

    if callback.startswith("card_"):
        plan_type = callback.replace("card_", "")
        if plan_type not in PRICES:
            await query.message.reply_text("Unknown plan.")
            return

        req = create_manual_payment_request(
            telegram_id=user.id,
            username=user.username,
            plan_type=plan_type,
            amount=float(PRICES[plan_type]),
        )
        await query.message.reply_text(
            (
                f"🏦 <b>Card payment request created</b>\n\n"
                f"Request ID: <b>{req.id}</b>\n"
                f"Plan: <b>{plan_type}</b>\n"
                f"Amount: <b>${PRICES[plan_type]}</b>\n\n"
                f"Transfer details:\n<code>{CARD_PAYMENT_DETAILS}</code>\n\n"
                "After transfer, wait for admin approval."
            ),
            parse_mode="HTML",
        )

        for admin_id in ADMIN_IDS:
            try:
                await context.bot.send_message(
                    chat_id=admin_id,
                    text=(
                        f"🧾 New card payment request\n"
                        f"ID: {req.id}\n"
                        f"User: {user.id} @{user.username or '-'}\n"
                        f"Plan: {plan_type}\n"
                        f"Amount: ${PRICES[plan_type]}"
                    ),
                    reply_markup=_admin_card_actions_keyboard(req.id),
                )
            except Exception:
                continue
        return

    plan_type = callback.replace("pay_", "")
    invoice = await payment_client.create_invoice(user.id, plan_type)
    if not invoice or not invoice.get("invoice_url"):
        await query.message.reply_text("Unable to create invoice right now. Please try again later.")
        return

    await query.message.reply_text(
        f"💳 Pay here: {invoice['invoice_url']}\n\nAfter payment, your subscription will be activated automatically.",
        disable_web_page_preview=True,
    )


async def handle_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    await query.answer()

    if query.data == "demo":
        await demo_command(update, context)
    elif query.data == "subscribe":
        await subscribe_command(update, context)
    elif query.data == "status":
        await status_command(update, context)
    elif query.data == "settings":
        await settings_command(update, context)
    elif query.data == "main_menu":
        await query.message.reply_text(
            "Main menu:",
            reply_markup=main_menu_keyboard(),
        )
    elif query.data == "toggle_notify" and update.effective_user:
        user = get_user(update.effective_user.id)
        if user:
            if not _is_paid_subscription(user):
                await query.message.reply_text("Premium setting. Please subscribe to unlock.")
                return
            current = bool((user.settings or {}).get("notify_enabled", True))
            updated = update_user_settings(update.effective_user.id, {"notify_enabled": not current})
            if updated:
                await query.message.reply_text(
                    "Notifications updated.",
                    reply_markup=settings_keyboard(updated.settings, paid_enabled=True),
                )
    elif query.data == "set_liquidity" and update.effective_user:
        user = get_user(update.effective_user.id)
        if user:
            if not _is_paid_subscription(user):
                await query.message.reply_text("Premium setting. Please subscribe to unlock.")
                return
            current = int((user.settings or {}).get("min_liquidity", 5000))
            cycle = [3000, 5000, 10000, 20000]
            next_value = cycle[(cycle.index(current) + 1) % len(cycle)] if current in cycle else cycle[0]
            updated = update_user_settings(update.effective_user.id, {"min_liquidity": next_value})
            if updated:
                await query.message.reply_text(
                    f"Min liquidity updated to ${next_value}.",
                    reply_markup=settings_keyboard(updated.settings, paid_enabled=True),
                )
    elif query.data == "set_liquidity_menu" and update.effective_user:
        user = get_user(update.effective_user.id)
        if user:
            if not _is_paid_subscription(user):
                await query.message.reply_text("Premium setting. Please subscribe to unlock.")
                return
            current = int((user.settings or {}).get("min_liquidity", 5000))
            await query.message.reply_text(
                "Choose minimum liquidity:",
                reply_markup=liquidity_options_keyboard(current),
            )
    elif query.data.startswith("set_liquidity_") and update.effective_user:
        user = get_user(update.effective_user.id)
        if user:
            if not _is_paid_subscription(user):
                await query.message.reply_text("Premium setting. Please subscribe to unlock.")
                return
            raw = query.data.replace("set_liquidity_", "")
            if raw.isdigit():
                next_value = int(raw)
                updated = update_user_settings(update.effective_user.id, {"min_liquidity": next_value})
                if updated:
                    await query.message.reply_text(
                        f"Min liquidity updated to ${next_value}.",
                        reply_markup=settings_keyboard(updated.settings, paid_enabled=True),
                    )
    elif query.data == "set_max_age" and update.effective_user:
        user = get_user(update.effective_user.id)
        if user:
            if not _is_paid_subscription(user):
                await query.message.reply_text("Premium setting. Please subscribe to unlock.")
                return
            current = int((user.settings or {}).get("max_age_minutes", 60))
            cycle = [15, 30, 60, 120]
            next_value = cycle[(cycle.index(current) + 1) % len(cycle)] if current in cycle else cycle[0]
            updated = update_user_settings(update.effective_user.id, {"max_age_minutes": next_value})
            if updated:
                await query.message.reply_text(
                    f"Max age updated to {next_value} minutes.",
                    reply_markup=settings_keyboard(updated.settings, paid_enabled=True),
                )
    elif query.data == "set_max_age_menu" and update.effective_user:
        user = get_user(update.effective_user.id)
        if user:
            if not _is_paid_subscription(user):
                await query.message.reply_text("Premium setting. Please subscribe to unlock.")
                return
            current = int((user.settings or {}).get("max_age_minutes", 60))
            await query.message.reply_text(
                "Choose maximum token age:",
                reply_markup=max_age_options_keyboard(current),
            )
    elif query.data.startswith("set_max_age_") and update.effective_user:
        user = get_user(update.effective_user.id)
        if user:
            if not _is_paid_subscription(user):
                await query.message.reply_text("Premium setting. Please subscribe to unlock.")
                return
            raw = query.data.replace("set_max_age_", "")
            if raw.isdigit():
                next_value = int(raw)
                updated = update_user_settings(update.effective_user.id, {"max_age_minutes": next_value})
                if updated:
                    await query.message.reply_text(
                        f"Max age updated to {next_value} minutes.",
                        reply_markup=settings_keyboard(updated.settings, paid_enabled=True),
                    )
    elif query.data == "set_interval" and update.effective_user:
        user = get_user(update.effective_user.id)
        if user:
            if not _is_paid_subscription(user):
                await query.message.reply_text("Premium setting. Please subscribe to unlock.")
                return
            current = int((user.settings or {}).get("alert_cooldown_seconds", 30))
            cycle = [15, 30, 60, 120]
            next_value = cycle[(cycle.index(current) + 1) % len(cycle)] if current in cycle else cycle[0]
            updated = update_user_settings(update.effective_user.id, {"alert_cooldown_seconds": next_value})
            if updated:
                await query.message.reply_text(
                    f"Alert interval updated to {next_value} seconds.",
                    reply_markup=settings_keyboard(updated.settings, paid_enabled=True),
                )
    elif query.data == "set_interval_menu" and update.effective_user:
        user = get_user(update.effective_user.id)
        if user:
            if not _is_paid_subscription(user):
                await query.message.reply_text("Premium setting. Please subscribe to unlock.")
                return
            current = int((user.settings or {}).get("alert_cooldown_seconds", 30))
            await query.message.reply_text(
                "Choose minimum interval between alerts:",
                reply_markup=interval_options_keyboard(current),
            )
    elif query.data.startswith("set_interval_") and update.effective_user:
        user = get_user(update.effective_user.id)
        if user:
            if not _is_paid_subscription(user):
                await query.message.reply_text("Premium setting. Please subscribe to unlock.")
                return
            raw = query.data.replace("set_interval_", "")
            if raw.isdigit():
                next_value = int(raw)
                updated = update_user_settings(update.effective_user.id, {"alert_cooldown_seconds": next_value})
                if updated:
                    await query.message.reply_text(
                        f"Alert interval updated to {next_value} seconds.",
                        reply_markup=settings_keyboard(updated.settings, paid_enabled=True),
                    )


async def handle_text_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    if text == BTN_TRIAL:
        await demo_command(update, context)
    elif text == BTN_SUBSCRIBE:
        await subscribe_command(update, context)
    elif text == BTN_STATUS:
        await status_command(update, context)
    elif text == BTN_SETTINGS:
        await settings_command(update, context)
