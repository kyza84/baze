"""Keyboards for bot navigation."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup

BTN_TRIAL = "🎁 Free Trial"
BTN_SUBSCRIBE = "💳 Subscribe"
BTN_STATUS = "📊 Status"
BTN_SETTINGS = "⚙️ Settings"


def main_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton(BTN_TRIAL), KeyboardButton(BTN_SUBSCRIBE)],
            [KeyboardButton(BTN_STATUS), KeyboardButton(BTN_SETTINGS)],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def subscription_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("1 Week - $5 (CryptoBot)", callback_data="pay_week"),
                InlineKeyboardButton("Card", callback_data="card_week"),
            ],
            [
                InlineKeyboardButton("2.5 Weeks - $10 (CryptoBot)", callback_data="pay_two_half_week"),
                InlineKeyboardButton("Card", callback_data="card_two_half_week"),
            ],
            [
                InlineKeyboardButton("1 Month - $15 (CryptoBot)", callback_data="pay_month"),
                InlineKeyboardButton("Card", callback_data="card_month"),
            ],
            [InlineKeyboardButton("< Back", callback_data="main_menu")],
        ]
    )


def settings_keyboard(
    current_settings: dict,
    paid_enabled: bool,
    show_personal_controls: bool = False,
    auto_trade_enabled: bool = False,
    auto_filter_enabled: bool = True,
) -> InlineKeyboardMarkup:
    min_liquidity = current_settings.get("min_liquidity", 5000)
    max_age_minutes = current_settings.get("max_age_minutes", 60)
    alert_cooldown = current_settings.get("alert_cooldown_seconds", 30)
    notify_enabled = current_settings.get("notify_enabled", True)

    if not paid_enabled:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🔒 Premium settings", callback_data="subscribe")],
                [InlineKeyboardButton("< Back", callback_data="main_menu")],
            ]
        )

    rows = [
        [InlineKeyboardButton(f"Min Liquidity: ${min_liquidity}", callback_data="set_liquidity_menu")],
        [InlineKeyboardButton(f"Max Age: {max_age_minutes}m", callback_data="set_max_age_menu")],
        [InlineKeyboardButton(f"Alert Interval: {alert_cooldown}s", callback_data="set_interval_menu")],
        [
            InlineKeyboardButton(
                f"Notifications: {'ON' if notify_enabled else 'OFF'}",
                callback_data="toggle_notify",
            )
        ],
    ]
    if show_personal_controls:
        rows.append(
            [
                InlineKeyboardButton(
                    f"Auto-trade: {'ON' if auto_trade_enabled else 'OFF'}",
                    callback_data="toggle_auto_trade",
                )
            ]
        )
        rows.append(
            [
                InlineKeyboardButton(
                    f"Auto-filter: {'ON' if auto_filter_enabled else 'OFF'}",
                    callback_data="toggle_auto_filter",
                )
            ]
        )
    rows.append([InlineKeyboardButton("< Back", callback_data="main_menu")])
    return InlineKeyboardMarkup(rows)


def liquidity_options_keyboard(current_value: int) -> InlineKeyboardMarkup:
    options = [3000, 5000, 10000, 20000]
    rows = []
    for value in options:
        label = f"{'✅ ' if value == current_value else ''}${value}"
        rows.append([InlineKeyboardButton(label, callback_data=f"set_liquidity_{value}")])
    rows.append([InlineKeyboardButton("< Back", callback_data="settings")])
    return InlineKeyboardMarkup(rows)


def max_age_options_keyboard(current_value: int) -> InlineKeyboardMarkup:
    options = [15, 30, 60, 120]
    rows = []
    for value in options:
        label = f"{'✅ ' if value == current_value else ''}{value} min"
        rows.append([InlineKeyboardButton(label, callback_data=f"set_max_age_{value}")])
    rows.append([InlineKeyboardButton("< Back", callback_data="settings")])
    return InlineKeyboardMarkup(rows)


def interval_options_keyboard(current_value: int) -> InlineKeyboardMarkup:
    options = [15, 30, 60, 120]
    rows = []
    for value in options:
        label = f"{'✅ ' if value == current_value else ''}{value} sec"
        rows.append([InlineKeyboardButton(label, callback_data=f"set_interval_{value}")])
    rows.append([InlineKeyboardButton("< Back", callback_data="settings")])
    return InlineKeyboardMarkup(rows)
