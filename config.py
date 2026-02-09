"""Application configuration."""

import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CRYPTOBOT_TOKEN = os.getenv("CRYPTOBOT_TOKEN", "")
GOPLUS_ACCESS_TOKEN = os.getenv("GOPLUS_ACCESS_TOKEN", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///bot.db")

PRICES = {
    "week": 5,
    "two_half_week": 10,
    "month": 15,
}

PLAN_DURATIONS_DAYS = {
    "week": 7,
    "two_half_week": 17.5,
    "month": 30,
}

TRIAL_HOURS = int(os.getenv("TRIAL_HOURS", "6"))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "30"))
MIN_LIQUIDITY = float(os.getenv("MIN_LIQUIDITY", "5000"))
TOKEN_AGE_MAX = int(os.getenv("TOKEN_AGE_MAX", "3600"))
DEXSCREENER_API = os.getenv("DEXSCREENER_API", "https://api.dexscreener.com/latest/dex")

DEX_TIMEOUT = int(os.getenv("DEX_TIMEOUT", "15"))
DEX_RETRIES = int(os.getenv("DEX_RETRIES", "3"))
SEEN_TOKEN_TTL = int(os.getenv("SEEN_TOKEN_TTL", "21600"))

WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "127.0.0.1")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8081"))
CRYPTOBOT_WEBHOOK_PATH = os.getenv("CRYPTOBOT_WEBHOOK_PATH", "/cryptobot/webhook")
CRYPTOBOT_WEBHOOK_SECRET = os.getenv("CRYPTOBOT_WEBHOOK_SECRET", "")
CARD_PAYMENT_DETAILS = os.getenv(
    "CARD_PAYMENT_DETAILS",
    "Card details not configured. Ask admin.",
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.getenv("LOG_DIR", "logs")
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")

ADMIN_IDS = {
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}
