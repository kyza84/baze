"""Application configuration."""

import os
from typing import Dict, Tuple

from dotenv import load_dotenv

load_dotenv()


def _parse_source_rate_limits(raw: str) -> Dict[str, Tuple[int, float]]:
    out: Dict[str, Tuple[int, float]] = {}
    for chunk in str(raw or "").split(","):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        source_part, rate_part = item.split(":", 1)
        source = source_part.strip().lower()
        if not source or "/" not in rate_part:
            continue
        count_part, window_part = rate_part.split("/", 1)
        try:
            count = max(1, int(float(count_part.strip())))
            window_seconds = max(1.0, float(window_part.strip()))
        except Exception:
            continue
        out[source] = (count, window_seconds)
    return out


def _parse_source_float_map(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for chunk in str(raw or "").split(","):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        source_part, value_part = item.split(":", 1)
        source = source_part.strip().lower()
        if not source:
            continue
        try:
            out[source] = max(0.0, float(value_part.strip()))
        except Exception:
            continue
    return out

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
GECKO_NEW_POOLS_INGEST_INTERVAL_SECONDS = max(15, int(os.getenv("GECKO_NEW_POOLS_INGEST_INTERVAL_SECONDS", "75")))
GECKO_NEW_POOLS_QUEUE_MAX = max(50, int(os.getenv("GECKO_NEW_POOLS_QUEUE_MAX", "400")))
GECKO_NEW_POOLS_DRAIN_MAX_PER_CYCLE = max(10, int(os.getenv("GECKO_NEW_POOLS_DRAIN_MAX_PER_CYCLE", "120")))
GECKO_INGEST_DEDUP_TTL_SECONDS = max(60, int(os.getenv("GECKO_INGEST_DEDUP_TTL_SECONDS", "3600")))
HEAVY_CHECK_DEDUP_TTL_SECONDS = max(0, int(os.getenv("HEAVY_CHECK_DEDUP_TTL_SECONDS", "900")))
HEAVY_CHECK_OVERRIDE_LIQ_MULT = max(1.1, float(os.getenv("HEAVY_CHECK_OVERRIDE_LIQ_MULT", "2.0")))
HEAVY_CHECK_OVERRIDE_VOL_MULT = max(1.1, float(os.getenv("HEAVY_CHECK_OVERRIDE_VOL_MULT", "3.0")))
HEAVY_CHECK_OVERRIDE_VOL_MIN_ABS_USD = max(0.0, float(os.getenv("HEAVY_CHECK_OVERRIDE_VOL_MIN_ABS_USD", "500")))
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
HTTP_CONNECTOR_LIMIT = max(1, int(os.getenv("HTTP_CONNECTOR_LIMIT", "30")))
HTTP_DEFAULT_CONCURRENCY = max(1, int(os.getenv("HTTP_DEFAULT_CONCURRENCY", "8")))
HTTP_RETRY_ATTEMPTS = max(1, int(os.getenv("HTTP_RETRY_ATTEMPTS", "3")))
HTTP_BACKOFF_BASE_SECONDS = max(0.05, float(os.getenv("HTTP_BACKOFF_BASE_SECONDS", "0.50")))
HTTP_BACKOFF_MAX_SECONDS = max(0.10, float(os.getenv("HTTP_BACKOFF_MAX_SECONDS", "8.00")))
HTTP_JITTER_SECONDS = max(0.0, float(os.getenv("HTTP_JITTER_SECONDS", "0.25")))
HTTP_RATE_LIMIT_DELAY_SECONDS = max(0.0, float(os.getenv("HTTP_RATE_LIMIT_DELAY_SECONDS", "2.00")))
HTTP_429_COOLDOWN_SECONDS = max(1.0, float(os.getenv("HTTP_429_COOLDOWN_SECONDS", "90")))
HTTP_SOURCE_RATE_LIMITS = _parse_source_rate_limits(
    os.getenv(
        "HTTP_SOURCE_RATE_LIMITS",
        "geckoterminal:20/60,watchlist_gecko:20/60,dexscreener:120/60,watchlist_dex:120/60,dex_boosts:30/60,goplus:40/60",
    )
)
HTTP_SOURCE_429_COOLDOWNS = _parse_source_float_map(
    os.getenv(
        "HTTP_SOURCE_429_COOLDOWNS",
        "geckoterminal:120,watchlist_gecko:120,dexscreener:20,watchlist_dex:20,dex_boosts:20,goplus:60",
    )
)
DATA_POLICY_DEGRADED_ERROR_PERCENT = max(0.0, float(os.getenv("DATA_POLICY_DEGRADED_ERROR_PERCENT", "35")))
DATA_POLICY_FAIL_CLOSED_FAIL_CLOSED_RATIO = max(0.0, float(os.getenv("DATA_POLICY_FAIL_CLOSED_FAIL_CLOSED_RATIO", "60")))
DATA_POLICY_FAIL_CLOSED_API_ERROR_PERCENT = max(0.0, float(os.getenv("DATA_POLICY_FAIL_CLOSED_API_ERROR_PERCENT", "90")))
DATA_POLICY_ENTER_STREAK = max(1, int(os.getenv("DATA_POLICY_ENTER_STREAK", "1")))
DATA_POLICY_EXIT_STREAK = max(1, int(os.getenv("DATA_POLICY_EXIT_STREAK", "2")))
METRICS_RSS_LOG_SECONDS = max(60, int(os.getenv("METRICS_RSS_LOG_SECONDS", "900")))

# Safety API behavior
TOKEN_SAFETY_FAIL_CLOSED = os.getenv("TOKEN_SAFETY_FAIL_CLOSED", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)

# Dynamic watchlist (vetted tokens) for stability-first trading.
WATCHLIST_ENABLED = os.getenv("WATCHLIST_ENABLED", "false").strip().lower() in ("1", "true", "yes", "y", "on")
WATCHLIST_REFRESH_SECONDS = max(60, int(os.getenv("WATCHLIST_REFRESH_SECONDS", "3600")))
WATCHLIST_MAX_TOKENS = max(5, int(os.getenv("WATCHLIST_MAX_TOKENS", "30")))
WATCHLIST_MIN_LIQUIDITY_USD = float(os.getenv("WATCHLIST_MIN_LIQUIDITY_USD", "200000"))
WATCHLIST_MIN_VOLUME_24H_USD = float(os.getenv("WATCHLIST_MIN_VOLUME_24H_USD", "500000"))
WATCHLIST_MIN_VOLUME_5M_USD = float(os.getenv("WATCHLIST_MIN_VOLUME_5M_USD", "5000"))
WATCHLIST_MIN_PRICE_CHANGE_5M_ABS_PERCENT = float(os.getenv("WATCHLIST_MIN_PRICE_CHANGE_5M_ABS_PERCENT", "1.5"))
WATCHLIST_GECKO_TRENDING_PAGES = max(1, int(os.getenv("WATCHLIST_GECKO_TRENDING_PAGES", "2")))
WATCHLIST_GECKO_POOLS_PAGES = max(0, int(os.getenv("WATCHLIST_GECKO_POOLS_PAGES", "2")))
WATCHLIST_DEX_ALLOWLIST = [
    x.strip().lower()
    for x in os.getenv("WATCHLIST_DEX_ALLOWLIST", "").split(",")
    if x.strip()
]
WATCHLIST_REQUIRE_WETH_QUOTE = os.getenv("WATCHLIST_REQUIRE_WETH_QUOTE", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
WATCHLIST_ALERTS_ENABLED = os.getenv("WATCHLIST_ALERTS_ENABLED", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
WATCHLIST_CACHE_FILE = os.getenv("WATCHLIST_CACHE_FILE", os.path.join("data", "watchlist_cache.json"))
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
SAFE_MIN_LIQUIDITY_USD = float(os.getenv("SAFE_MIN_LIQUIDITY_USD", "5000"))
SAFE_MIN_VOLUME_5M_USD = float(os.getenv("SAFE_MIN_VOLUME_5M_USD", "1200"))
SAFE_MIN_AGE_SECONDS = int(os.getenv("SAFE_MIN_AGE_SECONDS", "300"))
SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT = float(os.getenv("SAFE_MAX_PRICE_CHANGE_5M_ABS_PERCENT", "18"))
SAFE_REQUIRE_CONTRACT_SAFE = os.getenv("SAFE_REQUIRE_CONTRACT_SAFE", "true").lower() == "true"
SAFE_REQUIRE_RISK_LEVEL = os.getenv("SAFE_REQUIRE_RISK_LEVEL", "MEDIUM").strip().upper()
SAFE_MAX_WARNING_FLAGS = int(os.getenv("SAFE_MAX_WARNING_FLAGS", "1"))
FILTER_THRESH_LOG_EVERY_CYCLES = max(1, int(os.getenv("FILTER_THRESH_LOG_EVERY_CYCLES", "30")))
# Adaptive filters (paper calibration)
ADAPTIVE_FILTERS_ENABLED = os.getenv("ADAPTIVE_FILTERS_ENABLED", "false").lower() == "true"
ADAPTIVE_FILTERS_MODE = os.getenv("ADAPTIVE_FILTERS_MODE", "dry_run").strip().lower()  # off | dry_run | apply
ADAPTIVE_FILTERS_PAPER_ONLY = os.getenv("ADAPTIVE_FILTERS_PAPER_ONLY", "true").lower() == "true"
ADAPTIVE_FILTERS_INTERVAL_SECONDS = max(60, int(os.getenv("ADAPTIVE_FILTERS_INTERVAL_SECONDS", "900")))
ADAPTIVE_FILTERS_MIN_WINDOW_CYCLES = max(1, int(os.getenv("ADAPTIVE_FILTERS_MIN_WINDOW_CYCLES", "5")))
ADAPTIVE_FILTERS_TARGET_CAND_MIN = float(os.getenv("ADAPTIVE_FILTERS_TARGET_CAND_MIN", "2.0"))
ADAPTIVE_FILTERS_TARGET_CAND_MAX = float(os.getenv("ADAPTIVE_FILTERS_TARGET_CAND_MAX", "12.0"))
ADAPTIVE_FILTERS_TARGET_OPEN_MIN = float(os.getenv("ADAPTIVE_FILTERS_TARGET_OPEN_MIN", "0.10"))
ADAPTIVE_FILTERS_NEG_REALIZED_TRIGGER_USD = float(os.getenv("ADAPTIVE_FILTERS_NEG_REALIZED_TRIGGER_USD", "0.60"))
ADAPTIVE_FILTERS_NEG_CLOSED_MIN = max(1, int(os.getenv("ADAPTIVE_FILTERS_NEG_CLOSED_MIN", "3")))
ADAPTIVE_SCORE_MIN = int(os.getenv("ADAPTIVE_SCORE_MIN", "60"))
ADAPTIVE_SCORE_MAX = int(os.getenv("ADAPTIVE_SCORE_MAX", "72"))
ADAPTIVE_SCORE_STEP = max(1, int(os.getenv("ADAPTIVE_SCORE_STEP", "1")))
ADAPTIVE_SAFE_VOLUME_MIN = float(os.getenv("ADAPTIVE_SAFE_VOLUME_MIN", "150"))
ADAPTIVE_SAFE_VOLUME_MAX = float(os.getenv("ADAPTIVE_SAFE_VOLUME_MAX", "1200"))
ADAPTIVE_SAFE_VOLUME_STEP = max(10.0, float(os.getenv("ADAPTIVE_SAFE_VOLUME_STEP", "50")))
ADAPTIVE_DEDUP_TTL_MIN = max(0, int(os.getenv("ADAPTIVE_DEDUP_TTL_MIN", "60")))
ADAPTIVE_DEDUP_TTL_MAX = max(0, int(os.getenv("ADAPTIVE_DEDUP_TTL_MAX", "900")))
ADAPTIVE_DEDUP_TTL_STEP = max(5, int(os.getenv("ADAPTIVE_DEDUP_TTL_STEP", "30")))

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
LIVE_WALLET_ADDRESS = os.getenv("LIVE_WALLET_ADDRESS", "").strip()
LIVE_PRIVATE_KEY = os.getenv("LIVE_PRIVATE_KEY", "").strip()
LIVE_CHAIN_ID = int(os.getenv("LIVE_CHAIN_ID", EVM_CHAIN_ID))
LIVE_ROUTER_ADDRESS = os.getenv("LIVE_ROUTER_ADDRESS", "").strip()
LIVE_SLIPPAGE_BPS = max(1, int(os.getenv("LIVE_SLIPPAGE_BPS", "200")))
LIVE_SWAP_DEADLINE_SECONDS = max(30, int(os.getenv("LIVE_SWAP_DEADLINE_SECONDS", "45")))
LIVE_TX_TIMEOUT_SECONDS = max(30, int(os.getenv("LIVE_TX_TIMEOUT_SECONDS", "180")))
LIVE_MAX_GAS_GWEI = float(os.getenv("LIVE_MAX_GAS_GWEI", "2.0"))
LIVE_PRIORITY_FEE_GWEI = float(os.getenv("LIVE_PRIORITY_FEE_GWEI", "0.02"))
# Keep some native ETH for gas so we can still SELL/approve after a BUY.
LIVE_MIN_GAS_RESERVE_ETH = float(os.getenv("LIVE_MIN_GAS_RESERVE_ETH", "0.0007"))
# Hard cap on estimated gas for swaps/approves. Abnormally high estimates are common on honeypots.
LIVE_MAX_SWAP_GAS = max(50_000, int(os.getenv("LIVE_MAX_SWAP_GAS", "450000")))
# Live honeypot guard (pre-buy simulation via honeypot.is). This is best-effort protection, not a guarantee.
HONEYPOT_API_ENABLED = os.getenv("HONEYPOT_API_ENABLED", "true").lower() == "true"
HONEYPOT_API_URL = os.getenv("HONEYPOT_API_URL", "https://api.honeypot.is/v2/IsHoneypot").strip()
HONEYPOT_API_TIMEOUT_SECONDS = max(3, int(os.getenv("HONEYPOT_API_TIMEOUT_SECONDS", "10")))
HONEYPOT_API_CACHE_TTL_SECONDS = max(60, int(os.getenv("HONEYPOT_API_CACHE_TTL_SECONDS", "1800")))
HONEYPOT_API_FAIL_CLOSED = os.getenv("HONEYPOT_API_FAIL_CLOSED", "true").lower() == "true"
HONEYPOT_MAX_BUY_TAX_PERCENT = float(os.getenv("HONEYPOT_MAX_BUY_TAX_PERCENT", "10"))
HONEYPOT_MAX_SELL_TAX_PERCENT = float(os.getenv("HONEYPOT_MAX_SELL_TAX_PERCENT", "10"))

# Live security (autotrade)
LIVE_SELLABILITY_CHECK_ENABLED = os.getenv("LIVE_SELLABILITY_CHECK_ENABLED", "true").lower() == "true"
# For getAmountsOut(token->WETH). Any positive value works; 1 token is a good sanity check.
LIVE_SELLABILITY_CHECK_AMOUNT_TOKENS = float(os.getenv("LIVE_SELLABILITY_CHECK_AMOUNT_TOKENS", "1.0"))
LIVE_ROUNDTRIP_CHECK_ENABLED = os.getenv("LIVE_ROUNDTRIP_CHECK_ENABLED", "true").lower() == "true"
# Quote WETH->token using spend size, then token->WETH for a fraction of the received tokens.
LIVE_ROUNDTRIP_SELL_FRACTION = float(os.getenv("LIVE_ROUNDTRIP_SELL_FRACTION", "0.25"))
# Minimum return ratio for the roundtrip quote (e.g. 0.70 means lose at most 30% to price impact/taxes).
LIVE_ROUNDTRIP_MIN_RETURN_RATIO = float(os.getenv("LIVE_ROUNDTRIP_MIN_RETURN_RATIO", "0.70"))

# Live session profit stop (USD). If > 0, the bot will stop opening new trades when
# wallet PnL since session start reaches the target.
LIVE_STOP_AFTER_PROFIT_USD = float(os.getenv("LIVE_STOP_AFTER_PROFIT_USD", "0.0"))
LIVE_SESSION_RESET_ON_START = os.getenv("LIVE_SESSION_RESET_ON_START", "true").lower() == "true"

# If enabled, a live position may be "abandoned" in local state when SELL cannot be executed
# (e.g. not enough gas, route unsupported, revert). This does NOT sell on-chain, it only
# unblocks the bot from being stuck forever with MAX_OPEN_TRADES=1.
LIVE_ABANDON_UNSELLABLE_POSITIONS = os.getenv("LIVE_ABANDON_UNSELLABLE_POSITIONS", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
RECOVERY_DISCOVERY_MAX_ADDRESSES = max(0, int(os.getenv("RECOVERY_DISCOVERY_MAX_ADDRESSES", "80")))
RECOVERY_ATTEMPT_INTERVAL_SECONDS = max(5, int(os.getenv("RECOVERY_ATTEMPT_INTERVAL_SECONDS", "30")))
RECOVERY_MAX_ATTEMPTS = max(1, int(os.getenv("RECOVERY_MAX_ATTEMPTS", "8")))

# Autotrade blacklist to avoid repeatedly touching tokens that already failed critical guards (route/honeypot/etc).
AUTOTRADE_BLACKLIST_ENABLED = os.getenv("AUTOTRADE_BLACKLIST_ENABLED", "true").lower() == "true"
AUTOTRADE_BLACKLIST_FILE = os.getenv(
    "AUTOTRADE_BLACKLIST_FILE",
    os.path.join("data", "autotrade_blacklist.json"),
)
AUTOTRADE_BLACKLIST_TTL_SECONDS = max(300, int(os.getenv("AUTOTRADE_BLACKLIST_TTL_SECONDS", "86400")))
AUTOTRADE_BLACKLIST_MAX_ENTRIES = max(100, int(os.getenv("AUTOTRADE_BLACKLIST_MAX_ENTRIES", "5000")))
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
MAX_BUYS_PER_HOUR = max(0, int(os.getenv("MAX_BUYS_PER_HOUR", str(MAX_TRADES_PER_HOUR))))
MIN_TRADE_USD = max(0.01, float(os.getenv("MIN_TRADE_USD", "0.25")))
MAX_TX_PER_DAY = max(0, int(os.getenv("MAX_TX_PER_DAY", "8")))
AUTO_TRADE_EXCLUDED_ADDRESSES = [
    x.strip().lower()
    for x in os.getenv("AUTO_TRADE_EXCLUDED_ADDRESSES", "").split(",")
    if x.strip()
]
KILL_SWITCH_FILE = os.getenv("KILL_SWITCH_FILE", os.path.join("data", "kill.txt"))
GRACEFUL_STOP_FILE = os.getenv("GRACEFUL_STOP_FILE", os.path.join("data", "graceful_stop.signal"))
GRACEFUL_STOP_TIMEOUT_SECONDS = max(2, int(os.getenv("GRACEFUL_STOP_TIMEOUT_SECONDS", "12")))
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
PROFIT_LOCK_ENABLED = os.getenv("PROFIT_LOCK_ENABLED", "true").lower() == "true"
PROFIT_LOCK_TRIGGER_PERCENT = float(os.getenv("PROFIT_LOCK_TRIGGER_PERCENT", "18"))
PROFIT_LOCK_FLOOR_PERCENT = float(os.getenv("PROFIT_LOCK_FLOOR_PERCENT", "4"))
WEAKNESS_EXIT_ENABLED = os.getenv("WEAKNESS_EXIT_ENABLED", "true").lower() == "true"
WEAKNESS_EXIT_MIN_AGE_PERCENT = float(os.getenv("WEAKNESS_EXIT_MIN_AGE_PERCENT", "45"))
WEAKNESS_EXIT_PNL_PERCENT = float(os.getenv("WEAKNESS_EXIT_PNL_PERCENT", "-9"))

# Position sizing and edge filtering
DYNAMIC_POSITION_SIZING_ENABLED = os.getenv("DYNAMIC_POSITION_SIZING_ENABLED", "true").lower() == "true"
EDGE_FILTER_ENABLED = os.getenv("EDGE_FILTER_ENABLED", "true").lower() == "true"
MIN_EXPECTED_EDGE_PERCENT = float(os.getenv("MIN_EXPECTED_EDGE_PERCENT", "2.0"))
EDGE_FILTER_MODE = os.getenv("EDGE_FILTER_MODE", "usd").strip().lower()
MIN_EXPECTED_EDGE_USD = float(os.getenv("MIN_EXPECTED_EDGE_USD", "0.10"))

# Position sizing quality multiplier (score/liquidity/volume/volatility).
POSITION_SIZE_QUALITY_ENABLED = os.getenv("POSITION_SIZE_QUALITY_ENABLED", "true").lower() == "true"
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
PAPER_METRICS_SUMMARY_SECONDS = max(60, int(os.getenv("PAPER_METRICS_SUMMARY_SECONDS", "900")))
STAIR_STEP_ENABLED = os.getenv("STAIR_STEP_ENABLED", "false").lower() == "true"
STAIR_STEP_START_BALANCE_USD = float(os.getenv("STAIR_STEP_START_BALANCE_USD", "0"))
STAIR_STEP_SIZE_USD = float(os.getenv("STAIR_STEP_SIZE_USD", "5"))
STAIR_STEP_TRADABLE_BUFFER_USD = float(os.getenv("STAIR_STEP_TRADABLE_BUFFER_USD", "0.35"))
AUTO_STOP_MIN_AVAILABLE_USD = float(os.getenv("AUTO_STOP_MIN_AVAILABLE_USD", "0.25"))
# If false, AutoTrader will not stop trading due to low available balance.
LOW_BALANCE_GUARD_ENABLED = os.getenv("LOW_BALANCE_GUARD_ENABLED", "true").lower() == "true"

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
