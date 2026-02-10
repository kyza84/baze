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
RUN_MODE = os.getenv("RUN_MODE", "local").strip().lower()
SIGNAL_SOURCE = os.getenv("SIGNAL_SOURCE", "onchain").strip().lower()

CHAIN_NAME = os.getenv("CHAIN_NAME", "base")
CHAIN_ID = os.getenv("CHAIN_ID", "base")
EVM_CHAIN_ID = os.getenv("EVM_CHAIN_ID", "8453")
GECKO_NETWORK = os.getenv("GECKO_NETWORK", "base")

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "30"))
ONCHAIN_POLL_INTERVAL_SECONDS = max(1, int(os.getenv("ONCHAIN_POLL_INTERVAL_SECONDS", "2")))
MIN_LIQUIDITY = float(os.getenv("MIN_LIQUIDITY", "5000"))
TOKEN_AGE_MAX = int(os.getenv("TOKEN_AGE_MAX", "3600"))
DEXSCREENER_API = os.getenv("DEXSCREENER_API", "https://api.dexscreener.com/latest/dex")
DEX_SEARCH_QUERY = os.getenv("DEX_SEARCH_QUERY", CHAIN_NAME)
DEX_SEARCH_QUERIES = [
    q.strip()
    for q in os.getenv("DEX_SEARCH_QUERIES", DEX_SEARCH_QUERY).split(",")
    if q.strip()
]
if not DEX_SEARCH_QUERIES:
    DEX_SEARCH_QUERIES = [DEX_SEARCH_QUERY]
GECKO_NEW_POOLS_PAGES = max(1, int(os.getenv("GECKO_NEW_POOLS_PAGES", "1")))
DEXSCREENER_TOKEN_URL_TEMPLATE = os.getenv(
    "DEXSCREENER_TOKEN_URL_TEMPLATE",
    "https://dexscreener.com/{chain}/{token_address}",
)
EXPLORER_TOKEN_URL_TEMPLATE = os.getenv(
    "EXPLORER_TOKEN_URL_TEMPLATE",
    "https://basescan.org/token/{token_address}",
)
SWAP_URL_TEMPLATE = os.getenv(
    "SWAP_URL_TEMPLATE",
    "https://app.uniswap.org/#/swap?inputCurrency=ETH&outputCurrency={token_address}&chain=base",
)

DEX_TIMEOUT = int(os.getenv("DEX_TIMEOUT", "15"))
DEX_RETRIES = int(os.getenv("DEX_RETRIES", "3"))
SEEN_TOKEN_TTL = int(os.getenv("SEEN_TOKEN_TTL", "21600"))
RPC_TIMEOUT_SECONDS = max(3, int(os.getenv("RPC_TIMEOUT_SECONDS", "10")))
ONCHAIN_BLOCK_CHUNK = max(1, int(os.getenv("ONCHAIN_BLOCK_CHUNK", "500")))
ONCHAIN_FINALITY_BLOCKS = max(0, int(os.getenv("ONCHAIN_FINALITY_BLOCKS", "2")))
ONCHAIN_PARALLEL_MARKET_SOURCES = os.getenv("ONCHAIN_PARALLEL_MARKET_SOURCES", "true").lower() == "true"
ONCHAIN_ENRICH_RETRIES = max(1, int(os.getenv("ONCHAIN_ENRICH_RETRIES", "3")))
ONCHAIN_ENRICH_RETRY_DELAY_SECONDS = max(1, int(os.getenv("ONCHAIN_ENRICH_RETRY_DELAY_SECONDS", "8")))

# Base on-chain PairCreated source
RPC_PRIMARY = os.getenv("RPC_PRIMARY", "").strip()
RPC_SECONDARY = os.getenv("RPC_SECONDARY", "").strip()
BASE_FACTORY_ADDRESS = os.getenv("BASE_FACTORY_ADDRESS", "").strip().lower()
PAIR_CREATED_TOPIC = os.getenv(
    "PAIR_CREATED_TOPIC",
    # keccak256("PairCreated(address,address,address,uint256)")
    "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9",
).strip().lower()
# Optional Uniswap v3 PoolCreated source on Base.
ONCHAIN_ENABLE_UNISWAP_V3 = os.getenv("ONCHAIN_ENABLE_UNISWAP_V3", "true").lower() == "true"
UNISWAP_V3_FACTORY_ADDRESS = os.getenv(
    "UNISWAP_V3_FACTORY_ADDRESS",
    "0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
).strip().lower()
UNISWAP_V3_POOL_CREATED_TOPIC = os.getenv(
    "UNISWAP_V3_POOL_CREATED_TOPIC",
    # keccak256("PoolCreated(address,address,uint24,int24,address)")
    "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118",
).strip().lower()
WETH_ADDRESS = os.getenv("WETH_ADDRESS", "").strip().lower()
ONCHAIN_LAST_BLOCK_FILE = os.getenv("ONCHAIN_LAST_BLOCK_FILE", os.path.join("data", "last_block_base.txt"))
ONCHAIN_SEEN_PAIRS_FILE = os.getenv("ONCHAIN_SEEN_PAIRS_FILE", os.path.join("data", "seen_pairs_base.json"))
ONCHAIN_SEEN_PAIR_TTL_SECONDS = max(300, int(os.getenv("ONCHAIN_SEEN_PAIR_TTL_SECONDS", "7200")))

PERSONAL_MODE = os.getenv("PERSONAL_MODE", "true").lower() == "true"
PERSONAL_TELEGRAM_ID = int(os.getenv("PERSONAL_TELEGRAM_ID", "0"))
MIN_TOKEN_SCORE = int(os.getenv("MIN_TOKEN_SCORE", "70"))
AUTO_FILTER_ENABLED = os.getenv("AUTO_FILTER_ENABLED", "true").lower() == "true"
SAFE_TEST_MODE = os.getenv("SAFE_TEST_MODE", "true").lower() == "true"
SAFE_MIN_LIQUIDITY_USD = float(os.getenv("SAFE_MIN_LIQUIDITY_USD", "20000"))
SAFE_MIN_VOLUME_5M_USD = float(os.getenv("SAFE_MIN_VOLUME_5M_USD", "5000"))
SAFE_MIN_AGE_SECONDS = int(os.getenv("SAFE_MIN_AGE_SECONDS", "120"))
SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT = float(os.getenv("SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT", "18"))
SAFE_REQUIRE_CONTRACT_SAFE = os.getenv("SAFE_REQUIRE_CONTRACT_SAFE", "true").lower() == "true"
SAFE_REQUIRE_RISK_LEVEL = os.getenv("SAFE_REQUIRE_RISK_LEVEL", "MEDIUM").strip().upper()
SAFE_MAX_WARNING_FLAGS = int(os.getenv("SAFE_MAX_WARNING_FLAGS", "1"))

# Optional extra token flow sources
DEX_BOOSTS_SOURCE_ENABLED = os.getenv("DEX_BOOSTS_SOURCE_ENABLED", "true").lower() == "true"
DEX_BOOSTS_MAX_TOKENS = max(0, int(os.getenv("DEX_BOOSTS_MAX_TOKENS", "20")))

BUY_AMOUNT_ETH = float(os.getenv("BUY_AMOUNT_ETH", "0.0005"))
# Backward compatibility with previous variable naming.
BUY_AMOUNT_SOL = BUY_AMOUNT_ETH
PROFIT_TARGET_PERCENT = int(os.getenv("PROFIT_TARGET_PERCENT", "50"))
STOP_LOSS_PERCENT = int(os.getenv("STOP_LOSS_PERCENT", "30"))

AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"
AUTO_TRADE_PAPER = os.getenv("AUTO_TRADE_PAPER", "true").lower() == "true"
# AUTO_TRADE_ENTRY_MODE:
# - single: open only one best candidate per scan cycle
# - all: open every eligible candidate
# - top_n: open best N eligible candidates per scan cycle
AUTO_TRADE_ENTRY_MODE = os.getenv("AUTO_TRADE_ENTRY_MODE", "single").lower()
AUTO_TRADE_TOP_N = int(os.getenv("AUTO_TRADE_TOP_N", "10"))
# MAX_OPEN_TRADES:
# - 0 means unlimited open positions
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "1"))
MAX_BUY_AMOUNT = float(os.getenv("MAX_BUY_AMOUNT", "0.001"))
WETH_PRICE_FALLBACK_USD = float(os.getenv("WETH_PRICE_FALLBACK_USD", "3000"))
MAX_TRADES_PER_HOUR = max(0, int(os.getenv("MAX_TRADES_PER_HOUR", "3")))
KILL_SWITCH_FILE = os.getenv("KILL_SWITCH_FILE", os.path.join("data", "kill.txt"))
WALLET_BALANCE_USD = float(os.getenv("WALLET_BALANCE_USD", "2.75"))
PAPER_TRADE_SIZE_USD = float(os.getenv("PAPER_TRADE_SIZE_USD", "1.0"))
PAPER_MAX_HOLD_SECONDS = int(os.getenv("PAPER_MAX_HOLD_SECONDS", "1800"))

# Realistic paper simulation
PAPER_REALISM_ENABLED = os.getenv("PAPER_REALISM_ENABLED", "true").lower() == "true"
PAPER_GAS_PER_TX_USD = float(os.getenv("PAPER_GAS_PER_TX_USD", "0.03"))
PAPER_SWAP_FEE_BPS = float(os.getenv("PAPER_SWAP_FEE_BPS", "30"))
PAPER_BASE_SLIPPAGE_BPS = float(os.getenv("PAPER_BASE_SLIPPAGE_BPS", "80"))
PAPER_REALISM_CAP_ENABLED = os.getenv("PAPER_REALISM_CAP_ENABLED", "true").lower() == "true"
PAPER_REALISM_MAX_GAIN_PERCENT = float(os.getenv("PAPER_REALISM_MAX_GAIN_PERCENT", "600"))
PAPER_REALISM_MAX_LOSS_PERCENT = float(os.getenv("PAPER_REALISM_MAX_LOSS_PERCENT", "95"))

# Position sizing and edge filtering
DYNAMIC_POSITION_SIZING_ENABLED = os.getenv("DYNAMIC_POSITION_SIZING_ENABLED", "true").lower() == "true"
EDGE_FILTER_ENABLED = os.getenv("EDGE_FILTER_ENABLED", "true").lower() == "true"
MIN_EXPECTED_EDGE_PERCENT = float(os.getenv("MIN_EXPECTED_EDGE_PERCENT", "2.0"))
PAPER_TRADE_SIZE_MIN_USD = float(os.getenv("PAPER_TRADE_SIZE_MIN_USD", "0.25"))
PAPER_TRADE_SIZE_MAX_USD = float(os.getenv("PAPER_TRADE_SIZE_MAX_USD", str(PAPER_TRADE_SIZE_USD)))
CLOSED_TRADES_MAX_AGE_DAYS = max(0, int(os.getenv("CLOSED_TRADES_MAX_AGE_DAYS", "14")))
DYNAMIC_HOLD_ENABLED = os.getenv("DYNAMIC_HOLD_ENABLED", "true").lower() == "true"
HOLD_MIN_SECONDS = max(60, int(os.getenv("HOLD_MIN_SECONDS", "300")))
HOLD_MAX_SECONDS = max(HOLD_MIN_SECONDS, int(os.getenv("HOLD_MAX_SECONDS", str(PAPER_MAX_HOLD_SECONDS))))
PAPER_RESET_ON_START = os.getenv("PAPER_RESET_ON_START", "false").lower() == "true"
PAPER_PRICE_GUARD_ENABLED = os.getenv("PAPER_PRICE_GUARD_ENABLED", "true").lower() == "true"
PAPER_PRICE_GUARD_MAX_JUMP_PERCENT = float(os.getenv("PAPER_PRICE_GUARD_MAX_JUMP_PERCENT", "250"))
PAPER_PRICE_GUARD_CONFIRMATIONS = max(1, int(os.getenv("PAPER_PRICE_GUARD_CONFIRMATIONS", "2")))
PAPER_PRICE_GUARD_WINDOW_SECONDS = max(5, int(os.getenv("PAPER_PRICE_GUARD_WINDOW_SECONDS", "90")))
STAIR_STEP_ENABLED = os.getenv("STAIR_STEP_ENABLED", "false").lower() == "true"
STAIR_STEP_START_BALANCE_USD = float(os.getenv("STAIR_STEP_START_BALANCE_USD", "0"))
STAIR_STEP_SIZE_USD = float(os.getenv("STAIR_STEP_SIZE_USD", "5"))

# Trade risk controls
MAX_TOKEN_PRICE_CHANGE_5M_ABS_PERCENT = float(os.getenv("MAX_TOKEN_PRICE_CHANGE_5M_ABS_PERCENT", "35"))
MAX_TOKEN_COOLDOWN_SECONDS = max(0, int(os.getenv("MAX_TOKEN_COOLDOWN_SECONDS", "600")))
MAX_LOSS_PER_TRADE_PERCENT_BALANCE = float(os.getenv("MAX_LOSS_PER_TRADE_PERCENT_BALANCE", "1.2"))
DAILY_MAX_DRAWDOWN_PERCENT = float(os.getenv("DAILY_MAX_DRAWDOWN_PERCENT", "5.0"))
MAX_CONSECUTIVE_LOSSES = max(1, int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3")))
LOSS_STREAK_COOLDOWN_SECONDS = max(0, int(os.getenv("LOSS_STREAK_COOLDOWN_SECONDS", "1800")))

GOPLUS_EVM_API = os.getenv(
    "GOPLUS_EVM_API",
    "https://api.gopluslabs.io/api/v1/token_security/{chain_id}",
)
GOPLUS_SOLANA_API = os.getenv(
    "GOPLUS_SOLANA_API",
    "https://api.gopluslabs.io/api/v1/solana/token_security",
)

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
OUT_LOG_FILE = os.path.join(LOG_DIR, "out.log")

ADMIN_IDS = {
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}
