"""Message templates."""

WELCOME_MESSAGE = (
    "👋 <b>Welcome to Base Alert Bot</b>\n\n"
    "I scan fresh Base pairs and send instant token alerts based on liquidity and age filters.\n"
    "Start with a <b>free 6-hour trial</b> and test real-time notifications.\n\n"
    "Tap a menu button below to begin."
)

DEMO_ACTIVATED = (
    "🎁 <b>Trial Activated</b>\n\n"
    "Your free trial is active for <b>6 hours</b>.\n"
    "You will receive fresh Base token alerts during this period.\n\n"
    "Use /status anytime to check remaining access."
)

SUBSCRIPTION_INFO = (
    "💳 <b>Subscription Plans</b>\n\n"
    "- 1 Week: <b>$5</b>\n"
    "- 2.5 Weeks: <b>$10</b>\n"
    "- 1 Month: <b>$15</b>\n\n"
    "All plans include full token alerts, filters, and priority monitoring.\n"
    "Payment method: <b>CryptoBot (USDT)</b>."
)

STATUS_TEMPLATE = (
    "📊 <b>Your Status</b>\n\n"
    "Plan: <b>{plan_type}</b>\n"
    "Time left: <b>{time_left}</b>"
)

SETTINGS_INFO = (
    "⚙️ <b>Alert Settings</b>\n\n"
    "Adjust your minimum liquidity and toggle notifications with the buttons below."
)
