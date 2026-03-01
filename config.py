"""Application configuration."""

import os
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv


def _load_dotenv_safe(dotenv_path: str | None = None, *, override: bool = False) -> None:
    """Load dotenv using UTF-8-SIG so BOM-prefixed files don't break first key parsing."""
    try:
        load_dotenv(dotenv_path=dotenv_path, override=override, encoding="utf-8-sig")
    except TypeError:
        # Older python-dotenv versions may not expose the `encoding` argument.
        load_dotenv(dotenv_path=dotenv_path, override=override)


# Load base environment first, then optional per-instance override env file.
_load_dotenv_safe()
_BOT_ENV_FILE = os.getenv("BOT_ENV_FILE", "").strip()
if _BOT_ENV_FILE:
    _bot_env_path = Path(_BOT_ENV_FILE).expanduser()
    if not _bot_env_path.is_absolute():
        _bot_env_path = (Path.cwd() / _bot_env_path).resolve()
    if not _bot_env_path.exists():
        raise FileNotFoundError(f"BOT_ENV_FILE does not exist: {_bot_env_path}")
    if not _bot_env_path.is_file():
        raise IsADirectoryError(f"BOT_ENV_FILE is not a file: {_bot_env_path}")
    try:
        _load_dotenv_safe(str(_bot_env_path), override=True)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"Failed to load BOT_ENV_FILE '{_bot_env_path}': {exc}") from exc


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
BOT_INSTANCE_ID = os.getenv("BOT_INSTANCE_ID", "").strip()

CHAIN_NAME = os.getenv("CHAIN_NAME", "base")
CHAIN_ID = os.getenv("CHAIN_ID", "base")
EVM_CHAIN_ID = os.getenv("EVM_CHAIN_ID", "8453")
GECKO_NETWORK = os.getenv("GECKO_NETWORK", "base")
RUN_TAG = os.getenv("RUN_TAG", BOT_INSTANCE_ID).strip()

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
TOKEN_SAFETY_TRANSIENT_DEGRADED_ENABLED = os.getenv(
    "TOKEN_SAFETY_TRANSIENT_DEGRADED_ENABLED",
    "true",
).strip().lower() in ("1", "true", "yes", "y", "on")
TOKEN_SAFETY_TRANSIENT_USE_CACHE = os.getenv(
    "TOKEN_SAFETY_TRANSIENT_USE_CACHE",
    "true",
).strip().lower() in ("1", "true", "yes", "y", "on")
TOKEN_SAFETY_TRANSIENT_USE_FALLBACK = os.getenv(
    "TOKEN_SAFETY_TRANSIENT_USE_FALLBACK",
    "true",
).strip().lower() in ("1", "true", "yes", "y", "on")
TOKEN_SAFETY_TRANSIENT_REASONS = [
    x.strip().lower()
    for x in os.getenv(
        "TOKEN_SAFETY_TRANSIENT_REASONS",
        "api_code_4029,http_429,http_408,http_500,http_502,http_503,http_504",
    ).split(",")
    if x.strip()
]

# Dynamic watchlist (vetted tokens) for stability-first trading.
WATCHLIST_ENABLED = os.getenv("WATCHLIST_ENABLED", "false").strip().lower() in ("1", "true", "yes", "y", "on")
WATCHLIST_REFRESH_SECONDS = max(60, int(os.getenv("WATCHLIST_REFRESH_SECONDS", "3600")))
WATCHLIST_MAX_TOKENS = max(5, int(os.getenv("WATCHLIST_MAX_TOKENS", "30")))
WATCHLIST_MIN_LIQUIDITY_USD = float(os.getenv("WATCHLIST_MIN_LIQUIDITY_USD", "200000"))
WATCHLIST_MIN_VOLUME_24H_USD = float(os.getenv("WATCHLIST_MIN_VOLUME_24H_USD", "500000"))
WATCHLIST_MIN_VOLUME_5M_USD = float(os.getenv("WATCHLIST_MIN_VOLUME_5M_USD", "5000"))
WATCHLIST_MIN_PRICE_CHANGE_5M_ABS_PERCENT = float(os.getenv("WATCHLIST_MIN_PRICE_CHANGE_5M_ABS_PERCENT", "1.5"))
WATCHLIST_MISSING_PAIR_AGE_FALLBACK_SECONDS = max(
    0,
    int(os.getenv("WATCHLIST_MISSING_PAIR_AGE_FALLBACK_SECONDS", "86400")),
)
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
WATCHLIST_LOW_SUPPLY_MIN_COUNT = max(1, int(os.getenv("WATCHLIST_LOW_SUPPLY_MIN_COUNT", "12")))
WATCHLIST_LOW_SUPPLY_RELAX_FACTOR = max(
    0.05,
    min(1.0, float(os.getenv("WATCHLIST_LOW_SUPPLY_RELAX_FACTOR", "0.50"))),
)
WATCHLIST_LOW_SUPPLY_MIN_VOLUME_5M_USD = max(
    0.0,
    float(os.getenv("WATCHLIST_LOW_SUPPLY_MIN_VOLUME_5M_USD", "120")),
)
WATCHLIST_LOW_SUPPLY_RELAX_MIN_PRICE_CHANGE_5M_ABS_PERCENT = max(
    0.0,
    float(os.getenv("WATCHLIST_LOW_SUPPLY_RELAX_MIN_PRICE_CHANGE_5M_ABS_PERCENT", "0.0")),
)
WATCHLIST_LOW_SUPPLY_ALLOW_NON_WETH_IN_PAPER = os.getenv(
    "WATCHLIST_LOW_SUPPLY_ALLOW_NON_WETH_IN_PAPER",
    "true",
).strip().lower() in ("1", "true", "yes", "y", "on")
WATCHLIST_LOW_SUPPLY_RELAX_TRENDING_PAGES = max(
    0,
    int(os.getenv("WATCHLIST_LOW_SUPPLY_RELAX_TRENDING_PAGES", "1")),
)
WATCHLIST_LOW_SUPPLY_RELAX_POOLS_PAGES = max(
    0,
    int(os.getenv("WATCHLIST_LOW_SUPPLY_RELAX_POOLS_PAGES", "1")),
)
WATCHLIST_LOW_SUPPLY_FALLBACK_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("WATCHLIST_LOW_SUPPLY_FALLBACK_TIMEOUT_SECONDS", "8.0")),
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
SAFE_VOLUME_TWO_TIER_ENABLED = os.getenv("SAFE_VOLUME_TWO_TIER_ENABLED", "true").lower() == "true"
SAFE_VOLUME_TWO_TIER_SOFT_RATIO = max(
    0.20,
    min(0.99, float(os.getenv("SAFE_VOLUME_TWO_TIER_SOFT_RATIO", "0.80"))),
)
SAFE_VOLUME_TWO_TIER_STABLE_CYCLES = max(1, int(os.getenv("SAFE_VOLUME_TWO_TIER_STABLE_CYCLES", "2")))
SAFE_VOLUME_TWO_TIER_STABILITY_RATIO = max(
    0.50,
    min(1.20, float(os.getenv("SAFE_VOLUME_TWO_TIER_STABILITY_RATIO", "0.85"))),
)
SAFE_VOLUME_TWO_TIER_TTL_SECONDS = max(60, int(os.getenv("SAFE_VOLUME_TWO_TIER_TTL_SECONDS", "900")))
SAFE_REQUIRE_CONTRACT_SAFE = os.getenv("SAFE_REQUIRE_CONTRACT_SAFE", "true").lower() == "true"
SAFE_REQUIRE_RISK_LEVEL = os.getenv("SAFE_REQUIRE_RISK_LEVEL", "MEDIUM").strip().upper()
SAFE_MAX_WARNING_FLAGS = int(os.getenv("SAFE_MAX_WARNING_FLAGS", "1"))
SAFE_MIN_HOLDERS = max(0, int(os.getenv("SAFE_MIN_HOLDERS", "0")))
ENTRY_BLOCK_RISKY_CONTRACT_FLAGS = os.getenv("ENTRY_BLOCK_RISKY_CONTRACT_FLAGS", "true").lower() == "true"
ENTRY_HARD_RISKY_FLAG_CODES = [
    x.strip().lower()
    for x in os.getenv(
        "ENTRY_HARD_RISKY_FLAG_CODES",
        "mint,freeze,blacklist,proxy,owner_takeback,owner_change_balance",
    ).split(",")
    if x.strip()
]
ENTRY_FAIL_CLOSED_ON_SAFETY_GAP = os.getenv("ENTRY_FAIL_CLOSED_ON_SAFETY_GAP", "true").lower() == "true"
_entry_allow_transient_default = "false"
if (not str(GOPLUS_ACCESS_TOKEN or "").strip()) and str(os.getenv("WALLET_MODE", "paper") or "paper").strip().lower() == "paper":
    # Paper free-mode: allow transient/cache safety source to avoid total throughput stall.
    _entry_allow_transient_default = "true"
ENTRY_ALLOW_TRANSIENT_SAFETY_SOURCE = (
    os.getenv("ENTRY_ALLOW_TRANSIENT_SAFETY_SOURCE", _entry_allow_transient_default).lower() == "true"
)
ENTRY_TRANSIENT_SAFETY_COOLDOWN_SECONDS = max(
    0,
    int(os.getenv("ENTRY_TRANSIENT_SAFETY_COOLDOWN_SECONDS", "180")),
)
ENTRY_TRANSIENT_SAFETY_TO_BLACKLIST = os.getenv("ENTRY_TRANSIENT_SAFETY_TO_BLACKLIST", "false").lower() == "true"
_entry_allowed_safety_sources_default = "goplus"
if not str(GOPLUS_ACCESS_TOKEN or "").strip():
    # Free-mode default: no GoPlus token, accept transient/cache/fallback safety sources.
    _entry_allowed_safety_sources_default = "goplus,cache_transient,transient_fallback,fallback"
ENTRY_ALLOWED_SAFETY_SOURCES = [
    x.strip().lower()
    for x in os.getenv("ENTRY_ALLOWED_SAFETY_SOURCES", _entry_allowed_safety_sources_default).split(",")
    if x.strip()
]
FILTER_THRESH_LOG_EVERY_CYCLES = max(1, int(os.getenv("FILTER_THRESH_LOG_EVERY_CYCLES", "30")))
CANDIDATE_DECISIONS_LOG_ENABLED = os.getenv("CANDIDATE_DECISIONS_LOG_ENABLED", "true").lower() == "true"
CANDIDATE_DECISIONS_LOG_FILE = os.getenv("CANDIDATE_DECISIONS_LOG_FILE", os.path.join("logs", "candidates.jsonl"))
CANDIDATE_SHARD_MOD = max(1, int(os.getenv("CANDIDATE_SHARD_MOD", "1")))
CANDIDATE_SHARD_SLOT = max(0, int(os.getenv("CANDIDATE_SHARD_SLOT", "0")))
MINI_ANALYZER_ENABLED = os.getenv("MINI_ANALYZER_ENABLED", "true").lower() == "true"
MINI_ANALYZER_INTERVAL_SECONDS = max(60, int(os.getenv("MINI_ANALYZER_INTERVAL_SECONDS", "900")))
MARKET_REGIME_WINDOW_CYCLES = max(3, int(os.getenv("MARKET_REGIME_WINDOW_CYCLES", "12")))
MARKET_REGIME_MOMENTUM_CANDIDATES = float(os.getenv("MARKET_REGIME_MOMENTUM_CANDIDATES", "2.5"))
MARKET_REGIME_THIN_CANDIDATES = float(os.getenv("MARKET_REGIME_THIN_CANDIDATES", "0.8"))
MARKET_REGIME_FAIL_CLOSED_RATIO = float(os.getenv("MARKET_REGIME_FAIL_CLOSED_RATIO", "20.0"))
MARKET_REGIME_SOURCE_ERROR_PERCENT = float(os.getenv("MARKET_REGIME_SOURCE_ERROR_PERCENT", "20.0"))
MARKET_MODE_ENTER_STREAK = max(1, int(os.getenv("MARKET_MODE_ENTER_STREAK", "2")))
MARKET_MODE_EXIT_STREAK = max(1, int(os.getenv("MARKET_MODE_EXIT_STREAK", "3")))
MARKET_MODE_OWNS_STRICTNESS = os.getenv("MARKET_MODE_OWNS_STRICTNESS", "true").lower() == "true"
MARKET_MODE_STRICT_SCORE = max(0, int(os.getenv("MARKET_MODE_STRICT_SCORE", "78")))
MARKET_MODE_SOFT_SCORE = max(0, int(os.getenv("MARKET_MODE_SOFT_SCORE", "70")))
MARKET_MODE_YELLOW_SOFT_CAP_PER_CYCLE = max(0, int(os.getenv("MARKET_MODE_YELLOW_SOFT_CAP_PER_CYCLE", "2")))
MARKET_MODE_GREEN_SCORE_DELTA = int(os.getenv("MARKET_MODE_GREEN_SCORE_DELTA", "0"))
MARKET_MODE_YELLOW_SCORE_DELTA = int(os.getenv("MARKET_MODE_YELLOW_SCORE_DELTA", "1"))
MARKET_MODE_RED_SCORE_DELTA = int(os.getenv("MARKET_MODE_RED_SCORE_DELTA", "3"))
MARKET_MODE_GREEN_VOLUME_MULT = max(0.1, float(os.getenv("MARKET_MODE_GREEN_VOLUME_MULT", "1.00")))
MARKET_MODE_YELLOW_VOLUME_MULT = max(0.1, float(os.getenv("MARKET_MODE_YELLOW_VOLUME_MULT", "1.10")))
MARKET_MODE_RED_VOLUME_MULT = max(0.1, float(os.getenv("MARKET_MODE_RED_VOLUME_MULT", "1.30")))
MARKET_MODE_GREEN_EDGE_MULT = max(0.5, float(os.getenv("MARKET_MODE_GREEN_EDGE_MULT", "1.00")))
MARKET_MODE_YELLOW_EDGE_MULT = max(0.5, float(os.getenv("MARKET_MODE_YELLOW_EDGE_MULT", "1.08")))
MARKET_MODE_RED_EDGE_MULT = max(0.5, float(os.getenv("MARKET_MODE_RED_EDGE_MULT", "1.20")))
MARKET_MODE_GREEN_SIZE_MULT = max(0.1, float(os.getenv("MARKET_MODE_GREEN_SIZE_MULT", "1.00")))
MARKET_MODE_YELLOW_SIZE_MULT = max(0.1, float(os.getenv("MARKET_MODE_YELLOW_SIZE_MULT", "0.80")))
MARKET_MODE_RED_SIZE_MULT = max(0.1, float(os.getenv("MARKET_MODE_RED_SIZE_MULT", "0.55")))
MARKET_MODE_GREEN_HOLD_MULT = max(0.1, float(os.getenv("MARKET_MODE_GREEN_HOLD_MULT", "1.00")))
MARKET_MODE_YELLOW_HOLD_MULT = max(0.1, float(os.getenv("MARKET_MODE_YELLOW_HOLD_MULT", "0.80")))
MARKET_MODE_RED_HOLD_MULT = max(0.1, float(os.getenv("MARKET_MODE_RED_HOLD_MULT", "0.60")))
MARKET_MODE_GREEN_PARTIAL_TP_TRIGGER_MULT = max(0.2, float(os.getenv("MARKET_MODE_GREEN_PARTIAL_TP_TRIGGER_MULT", "1.00")))
MARKET_MODE_YELLOW_PARTIAL_TP_TRIGGER_MULT = max(0.2, float(os.getenv("MARKET_MODE_YELLOW_PARTIAL_TP_TRIGGER_MULT", "0.90")))
MARKET_MODE_RED_PARTIAL_TP_TRIGGER_MULT = max(0.2, float(os.getenv("MARKET_MODE_RED_PARTIAL_TP_TRIGGER_MULT", "0.75")))
MARKET_MODE_GREEN_PARTIAL_TP_SELL_MULT = max(0.2, float(os.getenv("MARKET_MODE_GREEN_PARTIAL_TP_SELL_MULT", "1.00")))
MARKET_MODE_YELLOW_PARTIAL_TP_SELL_MULT = max(0.2, float(os.getenv("MARKET_MODE_YELLOW_PARTIAL_TP_SELL_MULT", "1.20")))
MARKET_MODE_RED_PARTIAL_TP_SELL_MULT = max(0.2, float(os.getenv("MARKET_MODE_RED_PARTIAL_TP_SELL_MULT", "1.40")))
# Adaptive filters (paper calibration)
ADAPTIVE_FILTERS_ENABLED = os.getenv("ADAPTIVE_FILTERS_ENABLED", "false").lower() == "true"
ADAPTIVE_FILTERS_MODE = os.getenv("ADAPTIVE_FILTERS_MODE", "dry_run").strip().lower()  # off | dry_run | apply
ADAPTIVE_FILTERS_PAPER_ONLY = os.getenv("ADAPTIVE_FILTERS_PAPER_ONLY", "true").lower() == "true"
RUNTIME_TUNER_LOCK_LOCAL_CONTROLLERS = os.getenv(
    "RUNTIME_TUNER_LOCK_LOCAL_CONTROLLERS",
    "true",
).strip().lower() in ("1", "true", "yes", "y", "on")
RUNTIME_TUNER_LOCK_CHECK_INTERVAL_SECONDS = max(
    1,
    int(os.getenv("RUNTIME_TUNER_LOCK_CHECK_INTERVAL_SECONDS", "3")),
)
RUNTIME_TUNER_PRUNE_STALE_LOCK = os.getenv(
    "RUNTIME_TUNER_PRUNE_STALE_LOCK",
    "true",
).strip().lower() in ("1", "true", "yes", "y", "on")
RUNTIME_TUNER_STALE_LOCK_SECONDS = max(
    10,
    int(os.getenv("RUNTIME_TUNER_STALE_LOCK_SECONDS", "120")),
)
ADAPTIVE_FILTERS_INTERVAL_SECONDS = max(60, int(os.getenv("ADAPTIVE_FILTERS_INTERVAL_SECONDS", "900")))
ADAPTIVE_FILTERS_MIN_WINDOW_CYCLES = max(1, int(os.getenv("ADAPTIVE_FILTERS_MIN_WINDOW_CYCLES", "5")))
ADAPTIVE_FILTERS_COOLDOWN_WINDOWS = max(0, int(os.getenv("ADAPTIVE_FILTERS_COOLDOWN_WINDOWS", "1")))
ADAPTIVE_FILTERS_TARGET_CAND_MIN = float(os.getenv("ADAPTIVE_FILTERS_TARGET_CAND_MIN", "2.0"))
ADAPTIVE_FILTERS_TARGET_CAND_MAX = float(os.getenv("ADAPTIVE_FILTERS_TARGET_CAND_MAX", "12.0"))
ADAPTIVE_FILTERS_TARGET_OPEN_MIN = float(os.getenv("ADAPTIVE_FILTERS_TARGET_OPEN_MIN", "0.10"))
ADAPTIVE_ZERO_OPEN_RESET_ENABLED = os.getenv("ADAPTIVE_ZERO_OPEN_RESET_ENABLED", "true").lower() == "true"
ADAPTIVE_ZERO_OPEN_WINDOWS_BEFORE_RESET = max(1, int(os.getenv("ADAPTIVE_ZERO_OPEN_WINDOWS_BEFORE_RESET", "2")))
ADAPTIVE_ZERO_OPEN_MIN_CANDIDATES = float(os.getenv("ADAPTIVE_ZERO_OPEN_MIN_CANDIDATES", "1.0"))
ADAPTIVE_FILTERS_NEG_REALIZED_TRIGGER_USD = float(os.getenv("ADAPTIVE_FILTERS_NEG_REALIZED_TRIGGER_USD", "0.60"))
ADAPTIVE_FILTERS_NEG_CLOSED_MIN = max(1, int(os.getenv("ADAPTIVE_FILTERS_NEG_CLOSED_MIN", "3")))
ADAPTIVE_FILTERS_PNL_MIN_CLOSED = max(1, int(os.getenv("ADAPTIVE_FILTERS_PNL_MIN_CLOSED", "2")))
ADAPTIVE_SCORE_MIN = int(os.getenv("ADAPTIVE_SCORE_MIN", "60"))
ADAPTIVE_SCORE_MAX = int(os.getenv("ADAPTIVE_SCORE_MAX", "85"))
ADAPTIVE_SCORE_STEP = max(1, int(os.getenv("ADAPTIVE_SCORE_STEP", "1")))
ADAPTIVE_SAFE_VOLUME_MIN = float(os.getenv("ADAPTIVE_SAFE_VOLUME_MIN", "400"))
ADAPTIVE_SAFE_VOLUME_MAX = float(os.getenv("ADAPTIVE_SAFE_VOLUME_MAX", "5000"))
ADAPTIVE_SAFE_VOLUME_STEP = max(10.0, float(os.getenv("ADAPTIVE_SAFE_VOLUME_STEP", "50")))
ADAPTIVE_SAFE_VOLUME_STEP_PCT = max(0.0, float(os.getenv("ADAPTIVE_SAFE_VOLUME_STEP_PCT", "12")))
ADAPTIVE_DEDUP_TTL_MIN = max(0, int(os.getenv("ADAPTIVE_DEDUP_TTL_MIN", "120")))
ADAPTIVE_DEDUP_TTL_MAX = max(0, int(os.getenv("ADAPTIVE_DEDUP_TTL_MAX", "3600")))
ADAPTIVE_DEDUP_TTL_STEP = max(5, int(os.getenv("ADAPTIVE_DEDUP_TTL_STEP", "30")))
ADAPTIVE_DEDUP_TTL_STEP_PCT = max(0.0, float(os.getenv("ADAPTIVE_DEDUP_TTL_STEP_PCT", "25")))
ADAPTIVE_DEDUP_RELAX_ENABLED = os.getenv("ADAPTIVE_DEDUP_RELAX_ENABLED", "false").lower() == "true"
ADAPTIVE_DEDUP_DYNAMIC_ENABLED = os.getenv("ADAPTIVE_DEDUP_DYNAMIC_ENABLED", "false").lower() == "true"
ADAPTIVE_DEDUP_DYNAMIC_MIN = max(0, int(os.getenv("ADAPTIVE_DEDUP_DYNAMIC_MIN", "8")))
ADAPTIVE_DEDUP_DYNAMIC_MAX = max(0, int(os.getenv("ADAPTIVE_DEDUP_DYNAMIC_MAX", "30")))
ADAPTIVE_DEDUP_DYNAMIC_TARGET_PERCENTILE = max(
    50.0,
    min(99.0, float(os.getenv("ADAPTIVE_DEDUP_DYNAMIC_TARGET_PERCENTILE", "90"))),
)
ADAPTIVE_DEDUP_DYNAMIC_FACTOR = max(0.8, float(os.getenv("ADAPTIVE_DEDUP_DYNAMIC_FACTOR", "1.2")))
ADAPTIVE_DEDUP_DYNAMIC_MIN_SAMPLES = max(20, int(os.getenv("ADAPTIVE_DEDUP_DYNAMIC_MIN_SAMPLES", "200")))
ADAPTIVE_EDGE_ENABLED = os.getenv("ADAPTIVE_EDGE_ENABLED", "true").lower() == "true"
ADAPTIVE_EDGE_MIN = float(os.getenv("ADAPTIVE_EDGE_MIN", "1.5"))
ADAPTIVE_EDGE_MAX = float(os.getenv("ADAPTIVE_EDGE_MAX", "4.0"))
ADAPTIVE_EDGE_STEP = max(0.05, float(os.getenv("ADAPTIVE_EDGE_STEP", "0.25")))
ADAPTIVE_EDGE_STEP_PCT = max(0.0, float(os.getenv("ADAPTIVE_EDGE_STEP_PCT", "0")))
ADAPTIVE_EXIT_ENABLED = os.getenv("ADAPTIVE_EXIT_ENABLED", "true").lower() == "true"
ADAPTIVE_PROFIT_LOCK_FLOOR_MIN = float(os.getenv("ADAPTIVE_PROFIT_LOCK_FLOOR_MIN", "1.0"))
ADAPTIVE_PROFIT_LOCK_FLOOR_MAX = float(os.getenv("ADAPTIVE_PROFIT_LOCK_FLOOR_MAX", "8.0"))
ADAPTIVE_PROFIT_LOCK_FLOOR_STEP = max(0.1, float(os.getenv("ADAPTIVE_PROFIT_LOCK_FLOOR_STEP", "0.5")))
ADAPTIVE_NO_MOMENTUM_MAX_PNL_MIN = float(os.getenv("ADAPTIVE_NO_MOMENTUM_MAX_PNL_MIN", "-0.5"))
ADAPTIVE_NO_MOMENTUM_MAX_PNL_MAX = float(os.getenv("ADAPTIVE_NO_MOMENTUM_MAX_PNL_MAX", "1.5"))
ADAPTIVE_NO_MOMENTUM_MAX_PNL_STEP = max(0.05, float(os.getenv("ADAPTIVE_NO_MOMENTUM_MAX_PNL_STEP", "0.2")))
ADAPTIVE_WEAKNESS_PNL_MIN = float(os.getenv("ADAPTIVE_WEAKNESS_PNL_MIN", "-15.0"))
ADAPTIVE_WEAKNESS_PNL_MAX = float(os.getenv("ADAPTIVE_WEAKNESS_PNL_MAX", "-2.0"))
ADAPTIVE_WEAKNESS_PNL_STEP = max(0.1, float(os.getenv("ADAPTIVE_WEAKNESS_PNL_STEP", "0.5")))

# Autonomous controller (window-based throughput/risk balancing).
AUTONOMOUS_CONTROL_ENABLED = os.getenv("AUTONOMOUS_CONTROL_ENABLED", "false").lower() == "true"
AUTONOMOUS_CONTROL_MODE = os.getenv("AUTONOMOUS_CONTROL_MODE", "dry_run").strip().lower()  # off | dry_run | apply
AUTONOMOUS_CONTROL_PAPER_ONLY = os.getenv("AUTONOMOUS_CONTROL_PAPER_ONLY", "true").lower() == "true"
AUTONOMOUS_CONTROL_INTERVAL_SECONDS = max(60, int(os.getenv("AUTONOMOUS_CONTROL_INTERVAL_SECONDS", "300")))
AUTONOMOUS_CONTROL_MIN_WINDOW_CYCLES = max(1, int(os.getenv("AUTONOMOUS_CONTROL_MIN_WINDOW_CYCLES", "3")))
AUTONOMOUS_CONTROL_COOLDOWN_WINDOWS = max(0, int(os.getenv("AUTONOMOUS_CONTROL_COOLDOWN_WINDOWS", "1")))
AUTONOMOUS_CONTROL_TARGET_CANDIDATES_MIN = float(os.getenv("AUTONOMOUS_CONTROL_TARGET_CANDIDATES_MIN", "2.0"))
AUTONOMOUS_CONTROL_TARGET_CANDIDATES_HIGH = float(os.getenv("AUTONOMOUS_CONTROL_TARGET_CANDIDATES_HIGH", "8.0"))
AUTONOMOUS_CONTROL_TARGET_OPENED_MIN = float(os.getenv("AUTONOMOUS_CONTROL_TARGET_OPENED_MIN", "0.18"))
AUTONOMOUS_CONTROL_NEG_REALIZED_TRIGGER_USD = float(os.getenv("AUTONOMOUS_CONTROL_NEG_REALIZED_TRIGGER_USD", "0.08"))
AUTONOMOUS_CONTROL_POS_REALIZED_TRIGGER_USD = float(os.getenv("AUTONOMOUS_CONTROL_POS_REALIZED_TRIGGER_USD", "0.08"))
AUTONOMOUS_CONTROL_MAX_LOSS_STREAK_TRIGGER = max(1, int(os.getenv("AUTONOMOUS_CONTROL_MAX_LOSS_STREAK_TRIGGER", "3")))
AUTONOMOUS_CONTROL_STEP_OPEN_TRADES = max(1, int(os.getenv("AUTONOMOUS_CONTROL_STEP_OPEN_TRADES", "1")))
AUTONOMOUS_CONTROL_STEP_TOP_N = max(1, int(os.getenv("AUTONOMOUS_CONTROL_STEP_TOP_N", "1")))
AUTONOMOUS_CONTROL_STEP_MAX_BUYS_PER_HOUR = max(1, int(os.getenv("AUTONOMOUS_CONTROL_STEP_MAX_BUYS_PER_HOUR", "6")))
AUTONOMOUS_CONTROL_STEP_TRADE_SIZE_MAX_USD = max(0.01, float(os.getenv("AUTONOMOUS_CONTROL_STEP_TRADE_SIZE_MAX_USD", "0.05")))
AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MIN = max(1, int(os.getenv("AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MIN", "1")))
AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MAX = max(
    AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MIN,
    int(os.getenv("AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MAX", "6")),
)
AUTONOMOUS_CONTROL_TOP_N_MIN = max(1, int(os.getenv("AUTONOMOUS_CONTROL_TOP_N_MIN", "1")))
AUTONOMOUS_CONTROL_TOP_N_MAX = max(AUTONOMOUS_CONTROL_TOP_N_MIN, int(os.getenv("AUTONOMOUS_CONTROL_TOP_N_MAX", "20")))
AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN = max(1, int(os.getenv("AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN", "6")))
AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MAX = max(
    AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN,
    int(os.getenv("AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MAX", "96")),
)
AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MIN = max(0.05, float(os.getenv("AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MIN", "0.25")))
AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MAX = max(
    AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MIN,
    float(os.getenv("AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MAX", "2.00")),
)
AUTONOMOUS_CONTROL_RISK_OFF_OPEN_TRADES_CAP = max(1, int(os.getenv("AUTONOMOUS_CONTROL_RISK_OFF_OPEN_TRADES_CAP", "2")))
AUTONOMOUS_CONTROL_RISK_OFF_TOP_N_CAP = max(1, int(os.getenv("AUTONOMOUS_CONTROL_RISK_OFF_TOP_N_CAP", "8")))
AUTONOMOUS_CONTROL_RISK_OFF_MAX_BUYS_PER_HOUR_CAP = max(
    1,
    int(os.getenv("AUTONOMOUS_CONTROL_RISK_OFF_MAX_BUYS_PER_HOUR_CAP", "24")),
)
AUTONOMOUS_CONTROL_RISK_OFF_TRADE_SIZE_MAX_CAP = max(
    0.05,
    float(os.getenv("AUTONOMOUS_CONTROL_RISK_OFF_TRADE_SIZE_MAX_CAP", "0.70")),
)
AUTONOMOUS_CONTROL_ANTI_STALL_ENABLED = os.getenv("AUTONOMOUS_CONTROL_ANTI_STALL_ENABLED", "true").lower() == "true"
AUTONOMOUS_CONTROL_ANTI_STALL_MIN_CANDIDATES = max(
    0.0,
    float(os.getenv("AUTONOMOUS_CONTROL_ANTI_STALL_MIN_CANDIDATES", "1.0")),
)
AUTONOMOUS_CONTROL_ANTI_STALL_MIN_UTILIZATION = max(
    0.5,
    min(1.0, float(os.getenv("AUTONOMOUS_CONTROL_ANTI_STALL_MIN_UTILIZATION", "0.85"))),
)
AUTONOMOUS_CONTROL_ANTI_STALL_LIMIT_SKIP_MIN = max(
    0,
    int(os.getenv("AUTONOMOUS_CONTROL_ANTI_STALL_LIMIT_SKIP_MIN", "1")),
)
AUTONOMOUS_CONTROL_ANTI_STALL_EXPAND_MULT = max(
    0.5,
    min(3.0, float(os.getenv("AUTONOMOUS_CONTROL_ANTI_STALL_EXPAND_MULT", "1.0"))),
)
AUTONOMOUS_CONTROL_RECOVERY_ENABLED = os.getenv("AUTONOMOUS_CONTROL_RECOVERY_ENABLED", "true").lower() == "true"
AUTONOMOUS_CONTROL_RECOVERY_MIN_CANDIDATES = max(
    0.0,
    float(os.getenv("AUTONOMOUS_CONTROL_RECOVERY_MIN_CANDIDATES", "0.8")),
)
AUTONOMOUS_CONTROL_RECOVERY_EXPAND_MULT = max(
    0.5,
    min(3.0, float(os.getenv("AUTONOMOUS_CONTROL_RECOVERY_EXPAND_MULT", "1.2"))),
)
AUTONOMOUS_CONTROL_FRAGILE_TIGHTEN_ENABLED = os.getenv(
    "AUTONOMOUS_CONTROL_FRAGILE_TIGHTEN_ENABLED",
    "true",
).lower() == "true"
AUTONOMOUS_CONTROL_DECISIONS_LOG_ENABLED = os.getenv("AUTONOMOUS_CONTROL_DECISIONS_LOG_ENABLED", "true").lower() == "true"
AUTONOMOUS_CONTROL_DECISIONS_LOG_FILE = os.getenv(
    "AUTONOMOUS_CONTROL_DECISIONS_LOG_FILE",
    os.path.join("logs", "autonomy_decisions.jsonl"),
)
TRADE_DECISIONS_LOG_ENABLED = os.getenv("TRADE_DECISIONS_LOG_ENABLED", "true").lower() == "true"
TRADE_DECISIONS_LOG_FILE = os.getenv(
    "TRADE_DECISIONS_LOG_FILE",
    os.path.join("logs", "trade_decisions.jsonl"),
)

# Strategy orchestrator (HARVEST/DEFENSE profile switching over existing adaptives).
STRATEGY_ORCHESTRATOR_ENABLED = os.getenv("STRATEGY_ORCHESTRATOR_ENABLED", "false").lower() == "true"
STRATEGY_ORCHESTRATOR_MODE = os.getenv("STRATEGY_ORCHESTRATOR_MODE", "dry_run").strip().lower()  # off | dry_run | apply
STRATEGY_ORCHESTRATOR_LOCK_AUTONOMY_CONTROLS = (
    os.getenv("STRATEGY_ORCHESTRATOR_LOCK_AUTONOMY_CONTROLS", "true").lower() == "true"
)
STRATEGY_ORCHESTRATOR_LOCK_ADAPTIVE_FILTERS = (
    os.getenv("STRATEGY_ORCHESTRATOR_LOCK_ADAPTIVE_FILTERS", "true").lower() == "true"
)
STRATEGY_ORCHESTRATOR_INTERVAL_SECONDS = max(
    60,
    int(os.getenv("STRATEGY_ORCHESTRATOR_INTERVAL_SECONDS", "300")),
)
STRATEGY_ORCHESTRATOR_MIN_WINDOW_CYCLES = max(
    1,
    int(os.getenv("STRATEGY_ORCHESTRATOR_MIN_WINDOW_CYCLES", "2")),
)
STRATEGY_ORCHESTRATOR_COOLDOWN_WINDOWS = max(
    0,
    int(os.getenv("STRATEGY_ORCHESTRATOR_COOLDOWN_WINDOWS", "1")),
)
STRATEGY_ORCHESTRATOR_MIN_CLOSED_DELTA = max(
    1,
    int(os.getenv("STRATEGY_ORCHESTRATOR_MIN_CLOSED_DELTA", "6")),
)
STRATEGY_ORCHESTRATOR_DEFENSE_ENTER_STREAK = max(
    1,
    int(os.getenv("STRATEGY_ORCHESTRATOR_DEFENSE_ENTER_STREAK", "2")),
)
STRATEGY_ORCHESTRATOR_HARVEST_ENTER_STREAK = max(
    1,
    int(os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_ENTER_STREAK", "2")),
)
STRATEGY_ORCHESTRATOR_DEFENSE_TRIGGER_AVG_PNL_PER_TRADE_USD = float(
    os.getenv("STRATEGY_ORCHESTRATOR_DEFENSE_TRIGGER_AVG_PNL_PER_TRADE_USD", "-0.0020")
)
STRATEGY_ORCHESTRATOR_DEFENSE_TRIGGER_LOSS_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("STRATEGY_ORCHESTRATOR_DEFENSE_TRIGGER_LOSS_SHARE", "0.62"))),
)
STRATEGY_ORCHESTRATOR_HARVEST_TRIGGER_AVG_PNL_PER_TRADE_USD = float(
    os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_TRIGGER_AVG_PNL_PER_TRADE_USD", "0.0010")
)
STRATEGY_ORCHESTRATOR_HARVEST_TRIGGER_LOSS_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_TRIGGER_LOSS_SHARE", "0.45"))),
)
STRATEGY_ORCHESTRATOR_RED_FORCE_DEFENSE_ENABLED = (
    os.getenv("STRATEGY_ORCHESTRATOR_RED_FORCE_DEFENSE_ENABLED", "true").lower() == "true"
)
STRATEGY_ORCHESTRATOR_RED_FORCE_DEFENSE_REALIZED_DELTA_USD = float(
    os.getenv("STRATEGY_ORCHESTRATOR_RED_FORCE_DEFENSE_REALIZED_DELTA_USD", "-0.02")
)
STRATEGY_ORCHESTRATOR_INITIAL_PROFILE = os.getenv("STRATEGY_ORCHESTRATOR_INITIAL_PROFILE", "harvest").strip().lower()

# Per-profile auto-stop (mainly for matrix): stop weak profiles early with explicit reason.
PROFILE_AUTOSTOP_ENABLED = os.getenv("PROFILE_AUTOSTOP_ENABLED", "false").lower() == "true"
PROFILE_AUTOSTOP_NOTIFY_ENABLED = os.getenv("PROFILE_AUTOSTOP_NOTIFY_ENABLED", "true").lower() == "true"
PROFILE_AUTOSTOP_EVAL_INTERVAL_SECONDS = max(
    30,
    int(os.getenv("PROFILE_AUTOSTOP_EVAL_INTERVAL_SECONDS", "120")),
)
PROFILE_AUTOSTOP_MIN_RUNTIME_SECONDS = max(
    60,
    int(os.getenv("PROFILE_AUTOSTOP_MIN_RUNTIME_SECONDS", "3600")),
)
PROFILE_AUTOSTOP_MIN_CLOSED_TRADES = max(
    1,
    int(os.getenv("PROFILE_AUTOSTOP_MIN_CLOSED_TRADES", "24")),
)
PROFILE_AUTOSTOP_MIN_REALIZED_PNL_USD = float(
    os.getenv("PROFILE_AUTOSTOP_MIN_REALIZED_PNL_USD", "0.00")
)
PROFILE_AUTOSTOP_MIN_AVG_PNL_PER_TRADE_USD = float(
    os.getenv("PROFILE_AUTOSTOP_MIN_AVG_PNL_PER_TRADE_USD", "0.0005")
)
PROFILE_AUTOSTOP_MAX_LOSS_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("PROFILE_AUTOSTOP_MAX_LOSS_SHARE", "0.58"))),
)
PROFILE_AUTOSTOP_MAX_DRAWDOWN_FROM_PEAK_USD = max(
    0.0,
    float(os.getenv("PROFILE_AUTOSTOP_MAX_DRAWDOWN_FROM_PEAK_USD", "0.18")),
)
PROFILE_AUTOSTOP_MIN_FAIL_SIGNALS = max(
    1,
    int(os.getenv("PROFILE_AUTOSTOP_MIN_FAIL_SIGNALS", "2")),
)
PROFILE_AUTOSTOP_WINDOW_MINUTES = max(
    5,
    int(os.getenv("PROFILE_AUTOSTOP_WINDOW_MINUTES", "60")),
)
PROFILE_AUTOSTOP_WINDOW_MIN_CLOSED_TRADES = max(
    1,
    int(os.getenv("PROFILE_AUTOSTOP_WINDOW_MIN_CLOSED_TRADES", "10")),
)
PROFILE_AUTOSTOP_MIN_PNL_PER_HOUR_USD = float(
    os.getenv("PROFILE_AUTOSTOP_MIN_PNL_PER_HOUR_USD", "0.00")
)
PROFILE_AUTOSTOP_MAX_SL_SUM_WINDOW_USD = float(
    os.getenv("PROFILE_AUTOSTOP_MAX_SL_SUM_WINDOW_USD", "-999.0")
)
PROFILE_AUTOSTOP_EVENT_TAG = str(
    os.getenv("PROFILE_AUTOSTOP_EVENT_TAG", "[AUTOSTOP][PROFILE_STOPPED]") or "[AUTOSTOP][PROFILE_STOPPED]"
).strip()

# V2 runtime controls
V2_UNIVERSE_ENABLED = os.getenv("V2_UNIVERSE_ENABLED", "false").lower() == "true"
V2_UNIVERSE_MAX_TOTAL_PER_CYCLE = max(
    0,
    int(os.getenv("V2_UNIVERSE_MAX_TOTAL_PER_CYCLE", "0")),
)
V2_UNIVERSE_NOVELTY_WINDOW_SECONDS = max(
    60,
    int(os.getenv("V2_UNIVERSE_NOVELTY_WINDOW_SECONDS", "1800")),
)
V2_UNIVERSE_NOVELTY_MIN_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("V2_UNIVERSE_NOVELTY_MIN_SHARE", "0.30"))),
)
V2_UNIVERSE_NOVELTY_MIN_ABS = max(
    0,
    int(os.getenv("V2_UNIVERSE_NOVELTY_MIN_ABS", "2")),
)
V2_UNIVERSE_PASS_REPEAT_COOLDOWN_SECONDS = max(
    0,
    int(os.getenv("V2_UNIVERSE_PASS_REPEAT_COOLDOWN_SECONDS", "180")),
)
V2_UNIVERSE_PASS_REPEAT_OVERRIDE_VOL_MULT = max(
    1.0,
    float(os.getenv("V2_UNIVERSE_PASS_REPEAT_OVERRIDE_VOL_MULT", "2.2")),
)
V2_UNIVERSE_SOURCE_CAPS = os.getenv(
    "V2_UNIVERSE_SOURCE_CAPS",
    "onchain:120,onchain+market:120,dexscreener:100,geckoterminal:100,watchlist:35,dex_boosts:35",
).strip()
V2_UNIVERSE_SOURCE_WEIGHTS = os.getenv(
    "V2_UNIVERSE_SOURCE_WEIGHTS",
    "onchain:1.15,onchain+market:1.10,dexscreener:1.00,geckoterminal:1.05,watchlist:0.90,dex_boosts:1.10",
).strip()
V2_UNIVERSE_SYMBOL_REPEAT_WINDOW_SECONDS = max(
    60,
    int(os.getenv("V2_UNIVERSE_SYMBOL_REPEAT_WINDOW_SECONDS", "1800")),
)
V2_UNIVERSE_SYMBOL_REPEAT_SOFT_CAP = max(
    0,
    int(os.getenv("V2_UNIVERSE_SYMBOL_REPEAT_SOFT_CAP", "4")),
)
V2_UNIVERSE_SYMBOL_REPEAT_HARD_CAP = max(
    V2_UNIVERSE_SYMBOL_REPEAT_SOFT_CAP,
    int(os.getenv("V2_UNIVERSE_SYMBOL_REPEAT_HARD_CAP", "8")),
)
V2_UNIVERSE_SYMBOL_REPEAT_PENALTY_MULT = max(
    0.10,
    min(1.0, float(os.getenv("V2_UNIVERSE_SYMBOL_REPEAT_PENALTY_MULT", "0.72"))),
)
V2_UNIVERSE_SYMBOL_REPEAT_OVERRIDE_VOL_MULT = max(
    1.0,
    float(os.getenv("V2_UNIVERSE_SYMBOL_REPEAT_OVERRIDE_VOL_MULT", "2.5")),
)
V2_UNIVERSE_SURGE_ENABLED = os.getenv("V2_UNIVERSE_SURGE_ENABLED", "false").lower() == "true"
V2_UNIVERSE_SURGE_MIN_ABS_CHANGE_5M = max(
    0.0,
    float(os.getenv("V2_UNIVERSE_SURGE_MIN_ABS_CHANGE_5M", "4.0")),
)
V2_UNIVERSE_SURGE_MIN_VOLUME_5M_USD = max(
    0.0,
    float(os.getenv("V2_UNIVERSE_SURGE_MIN_VOLUME_5M_USD", "2500")),
)
V2_UNIVERSE_SURGE_MIN_VOL_TO_LIQ = max(
    0.0,
    float(os.getenv("V2_UNIVERSE_SURGE_MIN_VOL_TO_LIQ", "0.06")),
)
V2_UNIVERSE_SURGE_RESERVE_SHARE = max(
    0.0,
    min(0.50, float(os.getenv("V2_UNIVERSE_SURGE_RESERVE_SHARE", "0.22"))),
)
V2_UNIVERSE_SURGE_RESERVE_MIN_ABS = max(
    0,
    int(os.getenv("V2_UNIVERSE_SURGE_RESERVE_MIN_ABS", "2")),
)
V2_UNIVERSE_SURGE_SIZE_MULT = max(
    0.10,
    min(1.60, float(os.getenv("V2_UNIVERSE_SURGE_SIZE_MULT", "0.88"))),
)
V2_UNIVERSE_SURGE_HOLD_MULT = max(
    0.20,
    min(1.40, float(os.getenv("V2_UNIVERSE_SURGE_HOLD_MULT", "0.90"))),
)
V2_UNIVERSE_SURGE_EDGE_USD_MULT = max(
    0.05,
    min(1.80, float(os.getenv("V2_UNIVERSE_SURGE_EDGE_USD_MULT", "1.08"))),
)
V2_UNIVERSE_SURGE_EDGE_PERCENT_MULT = max(
    0.05,
    min(1.80, float(os.getenv("V2_UNIVERSE_SURGE_EDGE_PERCENT_MULT", "1.06"))),
)

# V2 quality gate: EV-first pool shaping (core/explore/cooldown) before entry routing.
V2_QUALITY_GATE_ENABLED = os.getenv("V2_QUALITY_GATE_ENABLED", "false").lower() == "true"
V2_QUALITY_REFRESH_SECONDS = max(
    120,
    int(os.getenv("V2_QUALITY_REFRESH_SECONDS", "1800")),
)
V2_QUALITY_WINDOW_SECONDS = max(
    600,
    int(os.getenv("V2_QUALITY_WINDOW_SECONDS", "14400")),
)
V2_QUALITY_MIN_SYMBOL_TRADES = max(
    2,
    int(os.getenv("V2_QUALITY_MIN_SYMBOL_TRADES", "12")),
)
V2_QUALITY_MIN_CLUSTER_TRADES = max(
    2,
    int(os.getenv("V2_QUALITY_MIN_CLUSTER_TRADES", "16")),
)
V2_QUALITY_MIN_AVG_PNL_USD = float(os.getenv("V2_QUALITY_MIN_AVG_PNL_USD", "0.0000"))
V2_QUALITY_MAX_LOSS_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("V2_QUALITY_MAX_LOSS_SHARE", "0.60"))),
)
V2_QUALITY_BAD_AVG_PNL_USD = float(os.getenv("V2_QUALITY_BAD_AVG_PNL_USD", "-0.0008"))
V2_QUALITY_BAD_LOSS_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("V2_QUALITY_BAD_LOSS_SHARE", "0.67"))),
)
V2_QUALITY_EXPLORE_MAX_SHARE = max(
    0.0,
    min(0.60, float(os.getenv("V2_QUALITY_EXPLORE_MAX_SHARE", "0.18"))),
)
V2_QUALITY_EXPLORE_MIN_ABS = max(
    0,
    int(os.getenv("V2_QUALITY_EXPLORE_MIN_ABS", "1")),
)
V2_QUALITY_EXPLORE_RESERVE_SHARE = max(
    0.0,
    min(0.60, float(os.getenv("V2_QUALITY_EXPLORE_RESERVE_SHARE", "0.18"))),
)
V2_QUALITY_COOLDOWN_PROBE_PROBABILITY = max(
    0.0,
    min(1.0, float(os.getenv("V2_QUALITY_COOLDOWN_PROBE_PROBABILITY", "0.15"))),
)
V2_QUALITY_COOLDOWN_SIZE_MULT = max(
    0.05,
    min(1.0, float(os.getenv("V2_QUALITY_COOLDOWN_SIZE_MULT", "0.55"))),
)
V2_QUALITY_COOLDOWN_HOLD_MULT = max(
    0.10,
    min(1.2, float(os.getenv("V2_QUALITY_COOLDOWN_HOLD_MULT", "0.75"))),
)
V2_QUALITY_COOLDOWN_EDGE_USD_MULT = max(
    0.05,
    float(os.getenv("V2_QUALITY_COOLDOWN_EDGE_USD_MULT", "1.35")),
)
V2_QUALITY_COOLDOWN_EDGE_PERCENT_MULT = max(
    0.05,
    float(os.getenv("V2_QUALITY_COOLDOWN_EDGE_PERCENT_MULT", "1.30")),
)
V2_QUALITY_SYMBOL_CONCENTRATION_WINDOW_SECONDS = max(
    300,
    int(os.getenv("V2_QUALITY_SYMBOL_CONCENTRATION_WINDOW_SECONDS", "3600")),
)
V2_QUALITY_SYMBOL_MAX_SHARE = max(
    0.02,
    min(0.50, float(os.getenv("V2_QUALITY_SYMBOL_MAX_SHARE", "0.10"))),
)
V2_QUALITY_SYMBOL_MIN_ABS_CAP = max(
    1,
    int(os.getenv("V2_QUALITY_SYMBOL_MIN_ABS_CAP", "2")),
)
V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS = max(
    0,
    int(os.getenv("V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS", "240")),
)
V2_QUALITY_SYMBOL_REENTRY_OVERRIDE_VOL_MULT = max(
    1.0,
    float(os.getenv("V2_QUALITY_SYMBOL_REENTRY_OVERRIDE_VOL_MULT", "2.3")),
)
V2_QUALITY_SYMBOL_CONCENTRATION_USE_OPEN_HISTORY = (
    os.getenv("V2_QUALITY_SYMBOL_CONCENTRATION_USE_OPEN_HISTORY", "true").lower() == "true"
)
V2_QUALITY_FRESH_SYMBOL_WINDOW_SECONDS = max(
    300,
    int(os.getenv("V2_QUALITY_FRESH_SYMBOL_WINDOW_SECONDS", "3600")),
)
V2_QUALITY_FRESH_SYMBOL_BOOST = max(
    0.0,
    float(os.getenv("V2_QUALITY_FRESH_SYMBOL_BOOST", "2.6")),
)
V2_QUALITY_SOURCE_BUDGET_ENABLED = os.getenv("V2_QUALITY_SOURCE_BUDGET_ENABLED", "true").lower() == "true"
V2_QUALITY_SOURCE_MIN_TRADES = max(
    2,
    int(os.getenv("V2_QUALITY_SOURCE_MIN_TRADES", "10")),
)
V2_QUALITY_SOURCE_WINDOW_SECONDS = max(
    600,
    int(os.getenv("V2_QUALITY_SOURCE_WINDOW_SECONDS", "14400")),
)
V2_QUALITY_SOURCE_GOOD_AVG_PNL_USD = float(os.getenv("V2_QUALITY_SOURCE_GOOD_AVG_PNL_USD", "0.0020"))
V2_QUALITY_SOURCE_BAD_AVG_PNL_USD = float(os.getenv("V2_QUALITY_SOURCE_BAD_AVG_PNL_USD", "-0.0015"))
V2_QUALITY_SOURCE_BAD_LOSS_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("V2_QUALITY_SOURCE_BAD_LOSS_SHARE", "0.64"))),
)
V2_QUALITY_SOURCE_SEVERE_AVG_PNL_USD = float(
    os.getenv("V2_QUALITY_SOURCE_SEVERE_AVG_PNL_USD", "-0.0030")
)
V2_QUALITY_SOURCE_SEVERE_LOSS_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("V2_QUALITY_SOURCE_SEVERE_LOSS_SHARE", "0.76"))),
)
V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY = max(
    0.0,
    min(1.0, float(os.getenv("V2_QUALITY_SOURCE_BAD_ENTRY_PROBABILITY", "0.72"))),
)
V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY = max(
    0.0,
    min(1.0, float(os.getenv("V2_QUALITY_SOURCE_SEVERE_ENTRY_PROBABILITY", "0.46"))),
)
V2_QUALITY_SOURCE_COOLDOWN_IN_RED_MULT = max(
    0.10,
    min(1.0, float(os.getenv("V2_QUALITY_SOURCE_COOLDOWN_IN_RED_MULT", "0.88"))),
)
V2_QUALITY_SOURCE_BOOST_MULT = max(
    0.50,
    float(os.getenv("V2_QUALITY_SOURCE_BOOST_MULT", "1.28")),
)
V2_QUALITY_SOURCE_CUT_MULT = max(
    0.10,
    min(1.0, float(os.getenv("V2_QUALITY_SOURCE_CUT_MULT", "0.62"))),
)
V2_QUALITY_SOURCE_MIN_SHARE = max(
    0.0,
    min(0.40, float(os.getenv("V2_QUALITY_SOURCE_MIN_SHARE", "0.08"))),
)
V2_QUALITY_SOURCE_MAX_SHARE = max(
    0.10,
    min(1.0, float(os.getenv("V2_QUALITY_SOURCE_MAX_SHARE", "0.58"))),
)
V2_QUALITY_SOURCE_CAP_SOFT_FILL_ENABLED = (
    os.getenv("V2_QUALITY_SOURCE_CAP_SOFT_FILL_ENABLED", "true").lower() == "true"
)
V2_QUALITY_PROVISIONAL_CORE_ENABLED = os.getenv("V2_QUALITY_PROVISIONAL_CORE_ENABLED", "true").lower() == "true"
V2_QUALITY_PROVISIONAL_CORE_MIN_SCORE = max(
    0,
    int(os.getenv("V2_QUALITY_PROVISIONAL_CORE_MIN_SCORE", "92")),
)
V2_QUALITY_PROVISIONAL_CORE_MIN_LIQUIDITY_USD = max(
    0.0,
    float(os.getenv("V2_QUALITY_PROVISIONAL_CORE_MIN_LIQUIDITY_USD", "120000")),
)
V2_QUALITY_PROVISIONAL_CORE_MIN_VOLUME_5M_USD = max(
    0.0,
    float(os.getenv("V2_QUALITY_PROVISIONAL_CORE_MIN_VOLUME_5M_USD", "3500")),
)
V2_QUALITY_PROVISIONAL_CORE_MIN_ABS_CHANGE_5M = max(
    0.0,
    float(os.getenv("V2_QUALITY_PROVISIONAL_CORE_MIN_ABS_CHANGE_5M", "2.8")),
)
V2_QUALITY_PROVISIONAL_CORE_MAX_SHARE = max(
    0.0,
    min(0.80, float(os.getenv("V2_QUALITY_PROVISIONAL_CORE_MAX_SHARE", "0.35"))),
)
V2_QUALITY_LOG_TOP_SYMBOLS = max(
    1,
    int(os.getenv("V2_QUALITY_LOG_TOP_SYMBOLS", "8")),
)
V2_QUALITY_CORE_MIN_ABS = max(
    0,
    int(os.getenv("V2_QUALITY_CORE_MIN_ABS", "1")),
)
V2_QUALITY_CORE_FALLBACK_FROM_EXPLORE_ENABLED = os.getenv(
    "V2_QUALITY_CORE_FALLBACK_FROM_EXPLORE_ENABLED",
    "true",
).lower() == "true"
V2_QUALITY_CORE_FALLBACK_SCORE_MIN = max(
    0,
    int(os.getenv("V2_QUALITY_CORE_FALLBACK_SCORE_MIN", "90")),
)
V2_QUALITY_CORE_FALLBACK_MAX_SHARE = max(
    0.0,
    min(0.60, float(os.getenv("V2_QUALITY_CORE_FALLBACK_MAX_SHARE", "0.25"))),
)

V2_SAFETY_BUDGET_ENABLED = os.getenv("V2_SAFETY_BUDGET_ENABLED", "false").lower() == "true"
V2_SAFETY_BUDGET_MAX_PER_CYCLE = max(
    1,
    int(os.getenv("V2_SAFETY_BUDGET_MAX_PER_CYCLE", "80")),
)
V2_SAFETY_BUDGET_PER_SOURCE = os.getenv(
    "V2_SAFETY_BUDGET_PER_SOURCE",
    "onchain:48,onchain+market:48,dexscreener:42,geckoterminal:42,watchlist:18,dex_boosts:18",
).strip()

V2_SOURCE_QOS_ENABLED = os.getenv("V2_SOURCE_QOS_ENABLED", "true").lower() == "true"
V2_SOURCE_QOS_FORCE_DUAL_ENTRY = os.getenv("V2_SOURCE_QOS_FORCE_DUAL_ENTRY", "true").lower() == "true"
V2_SOURCE_QOS_WINDOW_CYCLES = max(
    3,
    int(os.getenv("V2_SOURCE_QOS_WINDOW_CYCLES", "12")),
)
V2_SOURCE_QOS_MIN_EVAL_CYCLES = max(
    3,
    min(V2_SOURCE_QOS_WINDOW_CYCLES, int(os.getenv("V2_SOURCE_QOS_MIN_EVAL_CYCLES", "6"))),
)
V2_SOURCE_QOS_MIN_SEEN_PER_WINDOW = max(
    1,
    int(os.getenv("V2_SOURCE_QOS_MIN_SEEN_PER_WINDOW", "120")),
)
V2_SOURCE_QOS_MIN_PLAN_PER_WINDOW = max(
    1,
    int(os.getenv("V2_SOURCE_QOS_MIN_PLAN_PER_WINDOW", "8")),
)
V2_SOURCE_QOS_MIN_PASS_RATE = max(
    0.0,
    min(1.0, float(os.getenv("V2_SOURCE_QOS_MIN_PASS_RATE", "0.005"))),
)
V2_SOURCE_QOS_MIN_EV_POSITIVE_RATE = max(
    0.0,
    min(1.0, float(os.getenv("V2_SOURCE_QOS_MIN_EV_POSITIVE_RATE", "0.05"))),
)
V2_SOURCE_QOS_MIN_OPEN_RATE = max(
    0.0,
    min(1.0, float(os.getenv("V2_SOURCE_QOS_MIN_OPEN_RATE", "0.03"))),
)
V2_SOURCE_QOS_REQUIRED_FAIL_SIGNALS = max(
    1,
    int(os.getenv("V2_SOURCE_QOS_REQUIRED_FAIL_SIGNALS", "2")),
)
V2_SOURCE_QOS_COOLDOWN_SECONDS = max(
    60,
    int(os.getenv("V2_SOURCE_QOS_COOLDOWN_SECONDS", "900")),
)
V2_SOURCE_QOS_TOPK_ENABLED = os.getenv("V2_SOURCE_QOS_TOPK_ENABLED", "true").lower() == "true"
V2_SOURCE_QOS_TOPK_PER_CYCLE = max(
    0,
    int(os.getenv("V2_SOURCE_QOS_TOPK_PER_CYCLE", "220")),
)
V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE = max(
    0,
    int(os.getenv("V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE", "3")),
)
V2_SOURCE_QOS_SOURCE_CAPS = os.getenv(
    "V2_SOURCE_QOS_SOURCE_CAPS",
    "onchain:140,onchain+market:140,dexscreener:120,geckoterminal:120,watchlist:50,dex_boosts:40",
).strip()
V2_SOURCE_QOS_SOURCE_PRIORITY = os.getenv(
    "V2_SOURCE_QOS_SOURCE_PRIORITY",
    "onchain:1.10,onchain+market:1.12,dexscreener:1.00,geckoterminal:1.02,watchlist:0.95,dex_boosts:1.00",
).strip()
V2_SOURCE_QOS_CAP_MULT_FLOOR = max(
    0.10,
    min(1.0, float(os.getenv("V2_SOURCE_QOS_CAP_MULT_FLOOR", "0.40"))),
)
V2_SOURCE_QOS_CAP_MULT_CEIL = max(
    V2_SOURCE_QOS_CAP_MULT_FLOOR,
    float(os.getenv("V2_SOURCE_QOS_CAP_MULT_CEIL", "1.40")),
)
V2_SOURCE_QOS_BOOST_PASS_RATE = max(
    0.0,
    min(1.0, float(os.getenv("V2_SOURCE_QOS_BOOST_PASS_RATE", "0.02"))),
)
V2_SOURCE_QOS_BOOST_EV_POSITIVE_RATE = max(
    0.0,
    min(1.0, float(os.getenv("V2_SOURCE_QOS_BOOST_EV_POSITIVE_RATE", "0.20"))),
)
V2_SOURCE_QOS_BOOST_OPEN_RATE = max(
    0.0,
    min(1.0, float(os.getenv("V2_SOURCE_QOS_BOOST_OPEN_RATE", "0.08"))),
)

V2_CALIBRATION_ENABLED = os.getenv("V2_CALIBRATION_ENABLED", "false").lower() == "true"
V2_CALIBRATION_INTERVAL_SECONDS = max(
    120,
    int(os.getenv("V2_CALIBRATION_INTERVAL_SECONDS", "900")),
)
V2_CALIBRATION_MIN_CLOSED = max(
    20,
    int(os.getenv("V2_CALIBRATION_MIN_CLOSED", "120")),
)
V2_CALIBRATION_LOOKBACK_ROWS = max(
    50,
    int(os.getenv("V2_CALIBRATION_LOOKBACK_ROWS", "2000")),
)
V2_CALIBRATION_SMOOTH_ALPHA = max(
    0.05,
    min(1.0, float(os.getenv("V2_CALIBRATION_SMOOTH_ALPHA", "0.35"))),
)
V2_CALIBRATION_EDGE_USD_MIN = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_USD_MIN", "0.010")),
)
V2_CALIBRATION_EDGE_USD_MAX = max(
    V2_CALIBRATION_EDGE_USD_MIN,
    float(os.getenv("V2_CALIBRATION_EDGE_USD_MAX", "0.120")),
)
V2_CALIBRATION_EDGE_PERCENT_MIN = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_PERCENT_MIN", "0.35")),
)
V2_CALIBRATION_EDGE_PERCENT_MAX = max(
    V2_CALIBRATION_EDGE_PERCENT_MIN,
    float(os.getenv("V2_CALIBRATION_EDGE_PERCENT_MAX", "3.00")),
)
V2_CALIBRATION_EDGE_PERCENT_STEP_MAX = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_PERCENT_STEP_MAX", "0.06")),
)
V2_CALIBRATION_EDGE_USD_STEP_MAX = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_USD_STEP_MAX", "0.0030")),
)
V2_CALIBRATION_EDGE_PERCENT_DRIFT_UP_24H = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_PERCENT_DRIFT_UP_24H", "0.25")),
)
V2_CALIBRATION_EDGE_PERCENT_DRIFT_DOWN_24H = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_PERCENT_DRIFT_DOWN_24H", "0.20")),
)
V2_CALIBRATION_EDGE_USD_DRIFT_UP_24H = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_USD_DRIFT_UP_24H", "0.0060")),
)
V2_CALIBRATION_EDGE_USD_DRIFT_DOWN_24H = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_USD_DRIFT_DOWN_24H", "0.0060")),
)
V2_CALIBRATION_REANCHOR_INTERVAL_SECONDS = max(
    1800,
    int(os.getenv("V2_CALIBRATION_REANCHOR_INTERVAL_SECONDS", "14400")),
)
V2_CALIBRATION_REANCHOR_BLEND = max(
    0.0,
    min(1.0, float(os.getenv("V2_CALIBRATION_REANCHOR_BLEND", "0.22"))),
)
V2_CALIBRATION_VOLUME_MIN = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_VOLUME_MIN", "20")),
)
V2_CALIBRATION_VOLUME_MAX = max(
    V2_CALIBRATION_VOLUME_MIN,
    float(os.getenv("V2_CALIBRATION_VOLUME_MAX", "450")),
)
V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW = os.getenv(
    "V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW",
    "true",
).lower() == "true"
V2_CALIBRATION_NO_MOMENTUM_MAX_PNL_FLOOR = float(
    os.getenv("V2_CALIBRATION_NO_MOMENTUM_MAX_PNL_FLOOR", "0.02")
)
V2_CALIBRATION_EV_MIN_NET_MIN = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EV_MIN_NET_MIN", "0.0004")),
)
V2_CALIBRATION_EV_MIN_NET_MAX = max(
    V2_CALIBRATION_EV_MIN_NET_MIN,
    float(os.getenv("V2_CALIBRATION_EV_MIN_NET_MAX", "0.0038")),
)
V2_CALIBRATION_RUNNER_MIN_EV_MIN = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_RUNNER_MIN_EV_MIN", "0.0030")),
)
V2_CALIBRATION_RUNNER_MIN_EV_MAX = max(
    V2_CALIBRATION_RUNNER_MIN_EV_MIN,
    float(os.getenv("V2_CALIBRATION_RUNNER_MIN_EV_MAX", "0.0140")),
)
V2_CALIBRATION_SOURCE_BAD_ENTRY_PROB_MIN = max(
    0.0,
    min(1.0, float(os.getenv("V2_CALIBRATION_SOURCE_BAD_ENTRY_PROB_MIN", "0.45"))),
)
V2_CALIBRATION_SOURCE_BAD_ENTRY_PROB_MAX = max(
    V2_CALIBRATION_SOURCE_BAD_ENTRY_PROB_MIN,
    min(1.0, float(os.getenv("V2_CALIBRATION_SOURCE_BAD_ENTRY_PROB_MAX", "0.86"))),
)
V2_CALIBRATION_SOURCE_SEVERE_ENTRY_PROB_MIN = max(
    0.0,
    min(1.0, float(os.getenv("V2_CALIBRATION_SOURCE_SEVERE_ENTRY_PROB_MIN", "0.25"))),
)
V2_CALIBRATION_SOURCE_SEVERE_ENTRY_PROB_MAX = max(
    V2_CALIBRATION_SOURCE_SEVERE_ENTRY_PROB_MIN,
    min(1.0, float(os.getenv("V2_CALIBRATION_SOURCE_SEVERE_ENTRY_PROB_MAX", "0.68"))),
)
V2_CALIBRATION_CLAMP_ENABLED = os.getenv("V2_CALIBRATION_CLAMP_ENABLED", "true").lower() == "true"
V2_CALIBRATION_EDGE_PERCENT_MAX_FROM_BASE = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_PERCENT_MAX_FROM_BASE", "0.08")),
)
V2_CALIBRATION_EDGE_PERCENT_MIN_FROM_BASE = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_PERCENT_MIN_FROM_BASE", "0.10")),
)
V2_CALIBRATION_EDGE_USD_MAX_FROM_BASE = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_USD_MAX_FROM_BASE", "0.0020")),
)
V2_CALIBRATION_EDGE_USD_MIN_FROM_BASE = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_EDGE_USD_MIN_FROM_BASE", "0.0030")),
)
V2_CALIBRATION_SAFE_VOLUME_MAX_FROM_BASE = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_SAFE_VOLUME_MAX_FROM_BASE", "60.0")),
)
V2_CALIBRATION_NO_MOMENTUM_MIN = float(
    os.getenv("V2_CALIBRATION_NO_MOMENTUM_MIN", "0.08")
)
V2_CALIBRATION_WEAKNESS_MAX_NEG_DRIFT = max(
    0.0,
    float(os.getenv("V2_CALIBRATION_WEAKNESS_MAX_NEG_DRIFT", "0.35")),
)
V2_CALIBRATION_HOLD_MAX_ABS = max(
    90,
    int(os.getenv("V2_CALIBRATION_HOLD_MAX_ABS", "280")),
)
V2_CALIBRATION_DB_PATH = os.getenv(
    "V2_CALIBRATION_DB_PATH",
    os.path.join("data", "unified_dataset", "unified.db"),
)
V2_CALIBRATION_OUTPUT_JSON = os.getenv(
    "V2_CALIBRATION_OUTPUT_JSON",
    os.path.join("data", "analysis", "v2_calibration_latest.json"),
)

V2_REINVEST_ENABLED = os.getenv("V2_REINVEST_ENABLED", "false").lower() == "true"
V2_REINVEST_MIN_MULT = max(
    0.2,
    float(os.getenv("V2_REINVEST_MIN_MULT", "0.80")),
)
V2_REINVEST_MAX_MULT = max(
    V2_REINVEST_MIN_MULT,
    float(os.getenv("V2_REINVEST_MAX_MULT", "1.50")),
)
V2_REINVEST_GROWTH_STEP_USD = max(
    0.05,
    float(os.getenv("V2_REINVEST_GROWTH_STEP_USD", "0.60")),
)
V2_REINVEST_STEP_MULT = max(
    0.0,
    float(os.getenv("V2_REINVEST_STEP_MULT", "0.05")),
)
V2_REINVEST_DRAWDOWN_CUT_PERCENT = max(
    0.0,
    float(os.getenv("V2_REINVEST_DRAWDOWN_CUT_PERCENT", "2.5")),
)
V2_REINVEST_DRAWDOWN_MULT = max(
    0.2,
    float(os.getenv("V2_REINVEST_DRAWDOWN_MULT", "0.80")),
)
V2_REINVEST_LOSS_STREAK_STEP = max(
    0.0,
    float(os.getenv("V2_REINVEST_LOSS_STREAK_STEP", "0.08")),
)
V2_REINVEST_HIGH_EDGE_THRESHOLD_PERCENT = float(
    os.getenv("V2_REINVEST_HIGH_EDGE_THRESHOLD_PERCENT", "1.40")
)
V2_REINVEST_HIGH_EDGE_BONUS = max(
    0.0,
    float(os.getenv("V2_REINVEST_HIGH_EDGE_BONUS", "0.06")),
)

V2_CHAMPION_GUARD_ENABLED = os.getenv("V2_CHAMPION_GUARD_ENABLED", "false").lower() == "true"
V2_CHAMPION_GUARD_INTERVAL_SECONDS = max(
    30,
    int(os.getenv("V2_CHAMPION_GUARD_INTERVAL_SECONDS", "120")),
)
V2_CHAMPION_GUARD_MIN_RUNTIME_SECONDS = max(
    180,
    int(os.getenv("V2_CHAMPION_GUARD_MIN_RUNTIME_SECONDS", "3600")),
)
V2_CHAMPION_GUARD_MIN_CLOSED_TRADES = max(
    1,
    int(os.getenv("V2_CHAMPION_GUARD_MIN_CLOSED_TRADES", "20")),
)
V2_CHAMPION_GUARD_MAX_LAG_USD = max(
    0.0,
    float(os.getenv("V2_CHAMPION_GUARD_MAX_LAG_USD", "0.22")),
)
V2_CHAMPION_GUARD_FAIL_WINDOWS = max(
    1,
    int(os.getenv("V2_CHAMPION_GUARD_FAIL_WINDOWS", "3")),
)
V2_CHAMPION_GUARD_ACTIVE_MATRIX_PATH = os.getenv(
    "V2_CHAMPION_GUARD_ACTIVE_MATRIX_PATH",
    os.path.join("data", "matrix", "runs", "active_matrix.json"),
)

# V2 policy router: decouple data-quality policy from hard entry shutdown.
V2_POLICY_ROUTER_ENABLED = os.getenv("V2_POLICY_ROUTER_ENABLED", "false").lower() == "true"
V2_POLICY_FAIL_CLOSED_ACTION = os.getenv("V2_POLICY_FAIL_CLOSED_ACTION", "limited").strip().lower()  # limited|block
V2_POLICY_DEGRADED_ACTION = os.getenv("V2_POLICY_DEGRADED_ACTION", "limited").strip().lower()  # limited|block
V2_POLICY_LIMITED_ENTRY_RATIO = max(
    0.05,
    min(1.0, float(os.getenv("V2_POLICY_LIMITED_ENTRY_RATIO", "0.45"))),
)
V2_POLICY_LIMITED_MIN_PER_CYCLE = max(
    1,
    int(os.getenv("V2_POLICY_LIMITED_MIN_PER_CYCLE", "2")),
)
V2_POLICY_LIMITED_ONLY_STRICT = os.getenv("V2_POLICY_LIMITED_ONLY_STRICT", "true").lower() == "true"
V2_POLICY_LIMITED_ALLOW_EXPLORE_IN_RED = os.getenv("V2_POLICY_LIMITED_ALLOW_EXPLORE_IN_RED", "false").lower() == "true"

# Safety cache fallback for temporary safety API outages (does not bypass hard deny on truly unsafe tokens).
V2_SAFETY_CACHE_FALLBACK_ENABLED = os.getenv("V2_SAFETY_CACHE_FALLBACK_ENABLED", "true").lower() == "true"
V2_SAFETY_CACHE_TTL_SECONDS = max(
    120,
    int(os.getenv("V2_SAFETY_CACHE_TTL_SECONDS", "3600")),
)
V2_SAFETY_CACHE_ALLOWED_POLICY_MODES = [
    x.strip().upper()
    for x in os.getenv("V2_SAFETY_CACHE_ALLOWED_POLICY_MODES", "DEGRADED,FAIL_CLOSED").split(",")
    if x.strip()
]
V2_SAFETY_CACHE_ALLOWED_RISKS = [
    x.strip().upper()
    for x in os.getenv("V2_SAFETY_CACHE_ALLOWED_RISKS", "LOW,MEDIUM").split(",")
    if x.strip()
]
V2_SAFETY_CACHE_ALLOWED_SOURCES = [
    x.strip().lower()
    for x in os.getenv("V2_SAFETY_CACHE_ALLOWED_SOURCES", "watchlist,onchain+market").split(",")
    if x.strip()
]

# Dual entry channels (core/explore): core keeps quality, explore keeps adaptation flow with reduced risk.
V2_ENTRY_DUAL_CHANNEL_ENABLED = os.getenv("V2_ENTRY_DUAL_CHANNEL_ENABLED", "false").lower() == "true"
V2_ENTRY_EXPLORE_MAX_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("V2_ENTRY_EXPLORE_MAX_SHARE", "0.35"))),
)
V2_ENTRY_EXPLORE_MAX_PER_CYCLE = max(
    0,
    int(os.getenv("V2_ENTRY_EXPLORE_MAX_PER_CYCLE", "3")),
)
V2_ENTRY_EXPLORE_ALLOW_IN_RED = os.getenv("V2_ENTRY_EXPLORE_ALLOW_IN_RED", "false").lower() == "true"
V2_ENTRY_CORE_MIN_PER_CYCLE = max(
    0,
    int(os.getenv("V2_ENTRY_CORE_MIN_PER_CYCLE", "1")),
)
V2_ENTRY_EXPLORE_SIZE_MULT = max(
    0.10,
    min(1.0, float(os.getenv("V2_ENTRY_EXPLORE_SIZE_MULT", "0.45"))),
)
V2_ENTRY_EXPLORE_HOLD_MULT = max(
    0.20,
    min(1.2, float(os.getenv("V2_ENTRY_EXPLORE_HOLD_MULT", "0.75"))),
)

# Rolling edge governor (continuous adaptation in runtime, not only from offline db calibration).
V2_ROLLING_EDGE_ENABLED = os.getenv("V2_ROLLING_EDGE_ENABLED", "false").lower() == "true"
V2_ROLLING_EDGE_INTERVAL_SECONDS = max(
    60,
    int(os.getenv("V2_ROLLING_EDGE_INTERVAL_SECONDS", "240")),
)
V2_ROLLING_EDGE_MIN_CLOSED = max(
    10,
    int(os.getenv("V2_ROLLING_EDGE_MIN_CLOSED", "24")),
)
V2_ROLLING_EDGE_WINDOW_CLOSED = max(
    20,
    int(os.getenv("V2_ROLLING_EDGE_WINDOW_CLOSED", "120")),
)
V2_ROLLING_EDGE_RELAX_STEP_USD = max(
    0.0,
    float(os.getenv("V2_ROLLING_EDGE_RELAX_STEP_USD", "0.0020")),
)
V2_ROLLING_EDGE_TIGHTEN_STEP_USD = max(
    0.0,
    float(os.getenv("V2_ROLLING_EDGE_TIGHTEN_STEP_USD", "0.0025")),
)
V2_ROLLING_EDGE_RELAX_STEP_PERCENT = max(
    0.0,
    float(os.getenv("V2_ROLLING_EDGE_RELAX_STEP_PERCENT", "0.08")),
)
V2_ROLLING_EDGE_TIGHTEN_STEP_PERCENT = max(
    0.0,
    float(os.getenv("V2_ROLLING_EDGE_TIGHTEN_STEP_PERCENT", "0.10")),
)
V2_ROLLING_EDGE_MIN_USD = max(
    0.0,
    float(os.getenv("V2_ROLLING_EDGE_MIN_USD", "0.008")),
)
V2_ROLLING_EDGE_MAX_USD = max(
    V2_ROLLING_EDGE_MIN_USD,
    float(os.getenv("V2_ROLLING_EDGE_MAX_USD", "0.120")),
)
V2_ROLLING_EDGE_MIN_PERCENT = max(
    0.0,
    float(os.getenv("V2_ROLLING_EDGE_MIN_PERCENT", "0.35")),
)
V2_ROLLING_EDGE_MAX_PERCENT = max(
    V2_ROLLING_EDGE_MIN_PERCENT,
    float(os.getenv("V2_ROLLING_EDGE_MAX_PERCENT", "3.20")),
)
V2_ANTI_SELF_TIGHTEN_ENABLED = os.getenv("V2_ANTI_SELF_TIGHTEN_ENABLED", "true").lower() == "true"
V2_ANTI_SELF_TIGHTEN_OPEN_RATE_THRESHOLD = max(
    0.0,
    min(1.0, float(os.getenv("V2_ANTI_SELF_TIGHTEN_OPEN_RATE_THRESHOLD", "0.035"))),
)
V2_ANTI_SELF_TIGHTEN_EDGE_LOW_SHARE_THRESHOLD = max(
    0.0,
    min(1.0, float(os.getenv("V2_ANTI_SELF_TIGHTEN_EDGE_LOW_SHARE_THRESHOLD", "0.62"))),
)
V2_ANTI_SELF_TIGHTEN_COOLDOWN_SECONDS = max(
    60,
    int(os.getenv("V2_ANTI_SELF_TIGHTEN_COOLDOWN_SECONDS", "1800")),
)
V2_ROLLING_EDGE_MAX_TIGHTEN_PER_HOUR = max(
    0,
    int(os.getenv("V2_ROLLING_EDGE_MAX_TIGHTEN_PER_HOUR", "1")),
)
V2_ROLLING_EDGE_EDGE_LOW_SHARE_RELAX = max(
    0.0,
    min(1.0, float(os.getenv("V2_ROLLING_EDGE_EDGE_LOW_SHARE_RELAX", "0.65"))),
)
V2_ROLLING_EDGE_LOSS_SHARE_TIGHTEN = max(
    0.0,
    min(1.0, float(os.getenv("V2_ROLLING_EDGE_LOSS_SHARE_TIGHTEN", "0.58"))),
)
V2_EXPLORE_EDGE_USD_MULT = max(
    0.05,
    float(os.getenv("V2_EXPLORE_EDGE_USD_MULT", "0.75")),
)
V2_EXPLORE_EDGE_PERCENT_MULT = max(
    0.05,
    float(os.getenv("V2_EXPLORE_EDGE_PERCENT_MULT", "0.80")),
)

# Runtime KPI loop (controls throughput/diversity and prevents dead zones).
V2_KPI_LOOP_ENABLED = os.getenv("V2_KPI_LOOP_ENABLED", "false").lower() == "true"
V2_KPI_LOOP_INTERVAL_SECONDS = max(
    60,
    int(os.getenv("V2_KPI_LOOP_INTERVAL_SECONDS", "300")),
)
V2_KPI_LOOP_WINDOW_CYCLES = max(
    3,
    int(os.getenv("V2_KPI_LOOP_WINDOW_CYCLES", "20")),
)
V2_KPI_POLICY_MODE = str(os.getenv("V2_KPI_POLICY_MODE", "normal") or "normal").strip().lower()
if V2_KPI_POLICY_MODE not in {"normal", "lock_zero_open"}:
    V2_KPI_POLICY_MODE = "normal"
V2_KPI_EDGE_LOW_RELAX_TRIGGER = max(
    0.0,
    min(1.0, float(os.getenv("V2_KPI_EDGE_LOW_RELAX_TRIGGER", "0.70"))),
)
V2_KPI_OPEN_RATE_LOW_TRIGGER = max(
    0.0,
    float(os.getenv("V2_KPI_OPEN_RATE_LOW_TRIGGER", "0.03")),
)
V2_KPI_POLICY_BLOCK_TRIGGER = max(
    0.0,
    min(1.0, float(os.getenv("V2_KPI_POLICY_BLOCK_TRIGGER", "0.35"))),
)
V2_KPI_UNIQUE_SYMBOLS_MIN = max(
    1,
    int(os.getenv("V2_KPI_UNIQUE_SYMBOLS_MIN", "6")),
)
V2_KPI_MAX_BUYS_BOOST_STEP = max(
    0,
    int(os.getenv("V2_KPI_MAX_BUYS_BOOST_STEP", "4")),
)
V2_KPI_MAX_BUYS_CAP = max(
    1,
    int(os.getenv("V2_KPI_MAX_BUYS_CAP", "96")),
)
V2_KPI_TOPN_BOOST_STEP = max(
    0,
    int(os.getenv("V2_KPI_TOPN_BOOST_STEP", "1")),
)
V2_KPI_TOPN_CAP = max(
    1,
    int(os.getenv("V2_KPI_TOPN_CAP", "24")),
)
V2_KPI_EXPLORE_SHARE_STEP = max(
    0.0,
    min(0.30, float(os.getenv("V2_KPI_EXPLORE_SHARE_STEP", "0.03"))),
)
V2_KPI_EXPLORE_SHARE_MAX = max(
    0.05,
    min(1.0, float(os.getenv("V2_KPI_EXPLORE_SHARE_MAX", "0.55"))),
)
V2_KPI_NOVELTY_SHARE_STEP = max(
    0.0,
    min(0.30, float(os.getenv("V2_KPI_NOVELTY_SHARE_STEP", "0.03"))),
)
V2_KPI_NOVELTY_SHARE_MAX = max(
    0.05,
    min(1.0, float(os.getenv("V2_KPI_NOVELTY_SHARE_MAX", "0.60"))),
)
V2_KPI_QUALITY_REBALANCE_ENABLED = os.getenv("V2_KPI_QUALITY_REBALANCE_ENABLED", "true").lower() == "true"
V2_KPI_EDGE_LOW_HARD_TRIGGER = max(
    0.0,
    min(1.0, float(os.getenv("V2_KPI_EDGE_LOW_HARD_TRIGGER", "0.80"))),
)
V2_KPI_QUALITY_REBALANCE_EXPLORE_STEP = max(
    0.0,
    min(0.30, float(os.getenv("V2_KPI_QUALITY_REBALANCE_EXPLORE_STEP", "0.04"))),
)
V2_KPI_QUALITY_REBALANCE_NOVELTY_STEP = max(
    0.0,
    min(0.30, float(os.getenv("V2_KPI_QUALITY_REBALANCE_NOVELTY_STEP", "0.03"))),
)
V2_KPI_QUALITY_REBALANCE_TOPN_STEP = max(
    0,
    int(os.getenv("V2_KPI_QUALITY_REBALANCE_TOPN_STEP", "1")),
)
V2_KPI_QUALITY_REBALANCE_MAX_BUYS_STEP = max(
    0,
    int(os.getenv("V2_KPI_QUALITY_REBALANCE_MAX_BUYS_STEP", "4")),
)
# Explore lane fail-safe: shrink on bad explore EV window, recover on strong window.
V2_KPI_EXPLORE_FAILSAFE_ENABLED = os.getenv("V2_KPI_EXPLORE_FAILSAFE_ENABLED", "true").lower() == "true"
V2_KPI_EXPLORE_FAILSAFE_WINDOW_CLOSED = max(
    8,
    int(os.getenv("V2_KPI_EXPLORE_FAILSAFE_WINDOW_CLOSED", "40")),
)
V2_KPI_EXPLORE_FAILSAFE_MIN_CLOSED = max(
    6,
    int(os.getenv("V2_KPI_EXPLORE_FAILSAFE_MIN_CLOSED", "12")),
)
V2_KPI_EXPLORE_FAILSAFE_LOSS_SHARE_TRIGGER = max(
    0.0,
    min(1.0, float(os.getenv("V2_KPI_EXPLORE_FAILSAFE_LOSS_SHARE_TRIGGER", "0.62"))),
)
V2_KPI_EXPLORE_FAILSAFE_AVG_PNL_TRIGGER_USD = float(
    os.getenv("V2_KPI_EXPLORE_FAILSAFE_AVG_PNL_TRIGGER_USD", "-0.0015")
)
V2_KPI_EXPLORE_FAILSAFE_SHRINK_STEP = max(
    0.0,
    min(0.30, float(os.getenv("V2_KPI_EXPLORE_FAILSAFE_SHRINK_STEP", "0.06"))),
)
V2_KPI_EXPLORE_FAILSAFE_MIN_SHARE = max(
    0.01,
    min(0.60, float(os.getenv("V2_KPI_EXPLORE_FAILSAFE_MIN_SHARE", "0.08"))),
)
V2_KPI_EXPLORE_FAILSAFE_RECOVERY_WINRATE_MIN = max(
    0.0,
    min(1.0, float(os.getenv("V2_KPI_EXPLORE_FAILSAFE_RECOVERY_WINRATE_MIN", "0.54"))),
)
V2_KPI_EXPLORE_FAILSAFE_RECOVERY_AVG_PNL_MIN_USD = float(
    os.getenv("V2_KPI_EXPLORE_FAILSAFE_RECOVERY_AVG_PNL_MIN_USD", "0.0005")
)
V2_KPI_EXPLORE_FAILSAFE_RECOVERY_STEP = max(
    0.0,
    min(0.30, float(os.getenv("V2_KPI_EXPLORE_FAILSAFE_RECOVERY_STEP", "0.03"))),
)
V2_KPI_EXPLORE_FAILSAFE_COOLDOWN_SECONDS = max(
    0,
    int(os.getenv("V2_KPI_EXPLORE_FAILSAFE_COOLDOWN_SECONDS", "900")),
)
V2_KPI_FAST_ANTISTALL_ENABLED = os.getenv("V2_KPI_FAST_ANTISTALL_ENABLED", "true").lower() == "true"
V2_KPI_FAST_ANTISTALL_INTERVAL_SECONDS = max(
    60,
    int(os.getenv("V2_KPI_FAST_ANTISTALL_INTERVAL_SECONDS", "900")),
)
V2_KPI_FAST_ANTISTALL_STREAK_TRIGGER = max(
    1,
    int(os.getenv("V2_KPI_FAST_ANTISTALL_STREAK_TRIGGER", "2")),
)
V2_KPI_FAST_ANTISTALL_EDGE_LOW_SHARE_TRIGGER = max(
    0.0,
    min(1.0, float(os.getenv("V2_KPI_FAST_ANTISTALL_EDGE_LOW_SHARE_TRIGGER", "0.80"))),
)
V2_KPI_FAST_ANTISTALL_OPEN_RATE_TRIGGER = max(
    0.0,
    float(os.getenv("V2_KPI_FAST_ANTISTALL_OPEN_RATE_TRIGGER", "0.01")),
)
V2_KPI_FAST_ANTISTALL_EDGE_PERCENT_DROP = max(
    0.0,
    float(os.getenv("V2_KPI_FAST_ANTISTALL_EDGE_PERCENT_DROP", "0.12")),
)
V2_KPI_FAST_ANTISTALL_EDGE_USD_DROP = max(
    0.0,
    float(os.getenv("V2_KPI_FAST_ANTISTALL_EDGE_USD_DROP", "0.0015")),
)
V2_KPI_FAST_ANTISTALL_RECOVERY_WINDOW_SECONDS = max(
    120,
    int(os.getenv("V2_KPI_FAST_ANTISTALL_RECOVERY_WINDOW_SECONDS", "1200")),
)
V2_KPI_FAST_ANTISTALL_MAX_APPLY_PER_HOUR = max(
    1,
    int(os.getenv("V2_KPI_FAST_ANTISTALL_MAX_APPLY_PER_HOUR", "2")),
)

# Keep old behavior available for emergency rollback.
DATA_POLICY_HARD_BLOCK_ENABLED = os.getenv("DATA_POLICY_HARD_BLOCK_ENABLED", "false").lower() == "true"

_ORCH_BASE_MAX_OPEN = max(1, int(os.getenv("MAX_OPEN_TRADES", "3") or "3"))
_ORCH_BASE_TOP_N = max(1, int(os.getenv("AUTO_TRADE_TOP_N", "10") or "10"))
_ORCH_BASE_BUYS = max(1, int(os.getenv("MAX_BUYS_PER_HOUR", "24") or "24"))
_ORCH_BASE_SIZE_MAX = max(0.05, float(os.getenv("PAPER_TRADE_SIZE_MAX_USD", "1.0") or "1.0"))
_ORCH_BASE_HOLD_MAX = max(30, int(os.getenv("HOLD_MAX_SECONDS", os.getenv("PAPER_MAX_HOLD_SECONDS", "180")) or "180"))
_ORCH_BASE_NM_AGE = max(1.0, float(os.getenv("NO_MOMENTUM_EXIT_MIN_AGE_PERCENT", "12") or "12"))
_ORCH_BASE_NM_MAX_PNL = float(os.getenv("NO_MOMENTUM_EXIT_MAX_PNL_PERCENT", "0.3") or "0.3")
_ORCH_BASE_WEAK_AGE = max(1.0, float(os.getenv("WEAKNESS_EXIT_MIN_AGE_PERCENT", "14") or "14"))
_ORCH_BASE_WEAK_PNL = float(os.getenv("WEAKNESS_EXIT_PNL_PERCENT", "-2.6") or "-2.6")
_ORCH_BASE_PARTIAL_TRIGGER = max(0.1, float(os.getenv("PAPER_PARTIAL_TP_TRIGGER_PERCENT", "1.2") or "1.2"))
_ORCH_BASE_PARTIAL_FRAC = max(0.05, min(0.95, float(os.getenv("PAPER_PARTIAL_TP_SELL_FRACTION", "0.4") or "0.4")))

STRATEGY_ORCHESTRATOR_HARVEST_MAX_OPEN_TRADES = max(
    1,
    int(os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_MAX_OPEN_TRADES", str(_ORCH_BASE_MAX_OPEN))),
)
STRATEGY_ORCHESTRATOR_HARVEST_TOP_N = max(
    1,
    int(os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_TOP_N", str(_ORCH_BASE_TOP_N))),
)
STRATEGY_ORCHESTRATOR_HARVEST_MAX_BUYS_PER_HOUR = max(
    1,
    int(os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_MAX_BUYS_PER_HOUR", str(_ORCH_BASE_BUYS))),
)
STRATEGY_ORCHESTRATOR_HARVEST_TRADE_SIZE_MAX_USD = max(
    0.05,
    float(os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_TRADE_SIZE_MAX_USD", f"{_ORCH_BASE_SIZE_MAX:.4f}")),
)
STRATEGY_ORCHESTRATOR_HARVEST_HOLD_MAX_SECONDS = max(
    30,
    int(os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_HOLD_MAX_SECONDS", str(_ORCH_BASE_HOLD_MAX))),
)
STRATEGY_ORCHESTRATOR_HARVEST_NO_MOMENTUM_MIN_AGE_PERCENT = max(
    1.0,
    float(os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_NO_MOMENTUM_MIN_AGE_PERCENT", f"{_ORCH_BASE_NM_AGE:.2f}")),
)
STRATEGY_ORCHESTRATOR_HARVEST_NO_MOMENTUM_MAX_PNL_PERCENT = float(
    os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_NO_MOMENTUM_MAX_PNL_PERCENT", f"{_ORCH_BASE_NM_MAX_PNL:.2f}")
)
STRATEGY_ORCHESTRATOR_HARVEST_WEAKNESS_MIN_AGE_PERCENT = max(
    1.0,
    float(os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_WEAKNESS_MIN_AGE_PERCENT", f"{_ORCH_BASE_WEAK_AGE:.2f}")),
)
STRATEGY_ORCHESTRATOR_HARVEST_WEAKNESS_PNL_PERCENT = float(
    os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_WEAKNESS_PNL_PERCENT", f"{_ORCH_BASE_WEAK_PNL:.2f}")
)
STRATEGY_ORCHESTRATOR_HARVEST_PARTIAL_TP_TRIGGER_PERCENT = max(
    0.1,
    float(os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_PARTIAL_TP_TRIGGER_PERCENT", f"{_ORCH_BASE_PARTIAL_TRIGGER:.2f}")),
)
STRATEGY_ORCHESTRATOR_HARVEST_PARTIAL_TP_SELL_FRACTION = max(
    0.05,
    min(
        0.95,
        float(os.getenv("STRATEGY_ORCHESTRATOR_HARVEST_PARTIAL_TP_SELL_FRACTION", f"{_ORCH_BASE_PARTIAL_FRAC:.3f}")),
    ),
)

STRATEGY_ORCHESTRATOR_DEFENSE_MAX_OPEN_TRADES = max(
    1,
    int(os.getenv("STRATEGY_ORCHESTRATOR_DEFENSE_MAX_OPEN_TRADES", str(max(1, _ORCH_BASE_MAX_OPEN - 1)))),
)
STRATEGY_ORCHESTRATOR_DEFENSE_TOP_N = max(
    1,
    int(os.getenv("STRATEGY_ORCHESTRATOR_DEFENSE_TOP_N", str(max(1, _ORCH_BASE_TOP_N - 2)))),
)
STRATEGY_ORCHESTRATOR_DEFENSE_MAX_BUYS_PER_HOUR = max(
    1,
    int(os.getenv("STRATEGY_ORCHESTRATOR_DEFENSE_MAX_BUYS_PER_HOUR", str(max(1, int(_ORCH_BASE_BUYS * 0.7))))),
)
STRATEGY_ORCHESTRATOR_DEFENSE_TRADE_SIZE_MAX_USD = max(
    0.05,
    float(os.getenv("STRATEGY_ORCHESTRATOR_DEFENSE_TRADE_SIZE_MAX_USD", f"{max(0.05, _ORCH_BASE_SIZE_MAX * 0.85):.4f}")),
)
STRATEGY_ORCHESTRATOR_DEFENSE_HOLD_MAX_SECONDS = max(
    30,
    int(os.getenv("STRATEGY_ORCHESTRATOR_DEFENSE_HOLD_MAX_SECONDS", str(max(30, int(_ORCH_BASE_HOLD_MAX * 0.85))))),
)
STRATEGY_ORCHESTRATOR_DEFENSE_NO_MOMENTUM_MIN_AGE_PERCENT = max(
    1.0,
    float(
        os.getenv(
            "STRATEGY_ORCHESTRATOR_DEFENSE_NO_MOMENTUM_MIN_AGE_PERCENT",
            f"{max(1.0, _ORCH_BASE_NM_AGE * 0.85):.2f}",
        )
    ),
)
STRATEGY_ORCHESTRATOR_DEFENSE_NO_MOMENTUM_MAX_PNL_PERCENT = float(
    os.getenv(
        "STRATEGY_ORCHESTRATOR_DEFENSE_NO_MOMENTUM_MAX_PNL_PERCENT",
        f"{min(_ORCH_BASE_NM_MAX_PNL, 0.20):.2f}",
    )
)
STRATEGY_ORCHESTRATOR_DEFENSE_WEAKNESS_MIN_AGE_PERCENT = max(
    1.0,
    float(
        os.getenv(
            "STRATEGY_ORCHESTRATOR_DEFENSE_WEAKNESS_MIN_AGE_PERCENT",
            f"{max(1.0, _ORCH_BASE_WEAK_AGE * 0.9):.2f}",
        )
    ),
)
STRATEGY_ORCHESTRATOR_DEFENSE_WEAKNESS_PNL_PERCENT = float(
    os.getenv("STRATEGY_ORCHESTRATOR_DEFENSE_WEAKNESS_PNL_PERCENT", f"{max(-8.0, _ORCH_BASE_WEAK_PNL + 0.8):.2f}")
)
STRATEGY_ORCHESTRATOR_DEFENSE_PARTIAL_TP_TRIGGER_PERCENT = max(
    0.1,
    float(
        os.getenv(
            "STRATEGY_ORCHESTRATOR_DEFENSE_PARTIAL_TP_TRIGGER_PERCENT",
            f"{max(0.1, _ORCH_BASE_PARTIAL_TRIGGER * 0.9):.2f}",
        )
    ),
)
STRATEGY_ORCHESTRATOR_DEFENSE_PARTIAL_TP_SELL_FRACTION = max(
    0.05,
    min(
        0.95,
        float(
            os.getenv(
                "STRATEGY_ORCHESTRATOR_DEFENSE_PARTIAL_TP_SELL_FRACTION",
                f"{min(0.95, _ORCH_BASE_PARTIAL_FRAC + 0.06):.3f}",
            )
        ),
    ),
)

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
# Optional additional V2-compatible router addresses (comma-separated).
LIVE_ROUTER_ADDRESSES = os.getenv("LIVE_ROUTER_ADDRESSES", "").strip()
# Optional comma-separated intermediate tokens for V2 router paths in live mode.
# Example (Base): USDC token address.
LIVE_ROUTE_INTERMEDIATE_ADDRESSES = os.getenv(
    "LIVE_ROUTE_INTERMEDIATE_ADDRESSES",
    "0x833589fCD6eDb6E08f4c7C32D4f71b54bDa02913",
).strip()
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
# Also apply honeypot guard for paper/matrix mode to keep simulation close to live safety.
HONEYPOT_GUARD_IN_PAPER = os.getenv("HONEYPOT_GUARD_IN_PAPER", "true").lower() == "true"
# When false, transient honeypot API failures (status/timeouts) only trigger cooldown, not persistent blacklist.
HONEYPOT_TRANSIENT_TO_BLACKLIST = os.getenv("HONEYPOT_TRANSIENT_TO_BLACKLIST", "false").lower() == "true"
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
LIVE_ROUNDTRIP_MAX_LOSS_PERCENT = max(0.0, min(99.0, float(os.getenv("LIVE_ROUNDTRIP_MAX_LOSS_PERCENT", "8.0"))))
LIVE_ROUNDTRIP_REQUIRE_STABLE_ROUTER = os.getenv("LIVE_ROUNDTRIP_REQUIRE_STABLE_ROUTER", "true").lower() == "true"
LIVE_PRECHECK_USE_REAL_SIZE = os.getenv("LIVE_PRECHECK_USE_REAL_SIZE", "true").lower() == "true"
LIVE_PRECHECK_MIN_SPEND_USD = max(0.10, float(os.getenv("LIVE_PRECHECK_MIN_SPEND_USD", "2.00")))
LIVE_PRECHECK_MAX_SPEND_ETH = max(0.00001, float(os.getenv("LIVE_PRECHECK_MAX_SPEND_ETH", "0.0030")))
# Confirm bought token amount in wallet with short post-buy retries to avoid false zero-amount buys
# on RPC laggy nodes.
LIVE_BUY_BALANCE_RECHECK_ATTEMPTS = max(1, int(os.getenv("LIVE_BUY_BALANCE_RECHECK_ATTEMPTS", "8")))
LIVE_BUY_BALANCE_RECHECK_DELAY_SECONDS = max(
    0.05,
    float(os.getenv("LIVE_BUY_BALANCE_RECHECK_DELAY_SECONDS", "0.30")),
)
LIVE_BLACKLIST_UNSUPPORTED_ROUTE_TTL_SECONDS = max(
    300,
    int(os.getenv("LIVE_BLACKLIST_UNSUPPORTED_ROUTE_TTL_SECONDS", "7200")),
)
LIVE_BLACKLIST_ROUNDTRIP_FAIL_TTL_SECONDS = max(
    300,
    int(os.getenv("LIVE_BLACKLIST_ROUNDTRIP_FAIL_TTL_SECONDS", "21600")),
)
LIVE_BLACKLIST_ROUNDTRIP_RATIO_TTL_SECONDS = max(
    300,
    int(os.getenv("LIVE_BLACKLIST_ROUNDTRIP_RATIO_TTL_SECONDS", "3600")),
)
LIVE_BLACKLIST_ZERO_AMOUNT_TTL_SECONDS = max(
    300,
    int(os.getenv("LIVE_BLACKLIST_ZERO_AMOUNT_TTL_SECONDS", "1800")),
)
LIVE_BLACKLIST_SELL_FAIL_TTL_SECONDS = max(
    300,
    int(os.getenv("LIVE_BLACKLIST_SELL_FAIL_TTL_SECONDS", "21600")),
)

# New-token quarantine and risk caps for auto-discovery mode.
ENTRY_QUARANTINE_ENABLED = os.getenv("ENTRY_QUARANTINE_ENABLED", "true").lower() == "true"
ENTRY_QUARANTINE_REQUIRED_CYCLES = max(0, int(os.getenv("ENTRY_QUARANTINE_REQUIRED_CYCLES", "2")))
ENTRY_QUARANTINE_MAX_LIQUIDITY_DELTA_PCT = max(
    0.0,
    min(1.0, float(os.getenv("ENTRY_QUARANTINE_MAX_LIQUIDITY_DELTA_PCT", "0.35"))),
)
ENTRY_QUARANTINE_MAX_VOLUME_DELTA_PCT = max(
    0.0,
    min(1.5, float(os.getenv("ENTRY_QUARANTINE_MAX_VOLUME_DELTA_PCT", "0.80"))),
)
ENTRY_QUARANTINE_MAX_SPREAD_BPS = max(0.0, float(os.getenv("ENTRY_QUARANTINE_MAX_SPREAD_BPS", "220")))
ENTRY_QUARANTINE_MAX_SOURCE_DIVERGENCE_PCT = max(
    0.0,
    float(os.getenv("ENTRY_QUARANTINE_MAX_SOURCE_DIVERGENCE_PCT", "2.50")),
)
LIVE_UNVERIFIED_RISK_CAPS_ENABLED = os.getenv("LIVE_UNVERIFIED_RISK_CAPS_ENABLED", "true").lower() == "true"
LIVE_UNVERIFIED_MAX_OPEN_POSITIONS = max(1, int(os.getenv("LIVE_UNVERIFIED_MAX_OPEN_POSITIONS", "1")))
LIVE_UNVERIFIED_SIZE_MULT = max(
    0.05,
    min(1.0, float(os.getenv("LIVE_UNVERIFIED_SIZE_MULT", "0.60"))),
)
LIVE_UNVERIFIED_MAX_HOLD_SECONDS = max(15, int(os.getenv("LIVE_UNVERIFIED_MAX_HOLD_SECONDS", "120")))
LIVE_NEW_TOKEN_PAUSE_ON_SELL_FAIL_SECONDS = max(
    0,
    int(os.getenv("LIVE_NEW_TOKEN_PAUSE_ON_SELL_FAIL_SECONDS", "900")),
)

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
RECOVERY_DISCOVERY_INTERVAL_SECONDS = max(10, int(os.getenv("RECOVERY_DISCOVERY_INTERVAL_SECONDS", "45")))
RECOVERY_UNTRACKED_MIN_VALUE_USD = max(0.0, float(os.getenv("RECOVERY_UNTRACKED_MIN_VALUE_USD", "0.05")))
RECOVERY_ATTEMPT_INTERVAL_SECONDS = max(5, int(os.getenv("RECOVERY_ATTEMPT_INTERVAL_SECONDS", "30")))
RECOVERY_MAX_ATTEMPTS = max(1, int(os.getenv("RECOVERY_MAX_ATTEMPTS", "8")))

# Autotrade blacklist to avoid repeatedly touching tokens that already failed critical guards (route/honeypot/etc).
AUTOTRADE_BLACKLIST_ENABLED = os.getenv("AUTOTRADE_BLACKLIST_ENABLED", "true").lower() == "true"
AUTOTRADE_BLACKLIST_FILE = os.getenv(
    "AUTOTRADE_BLACKLIST_FILE",
    os.path.join("data", "autotrade_blacklist.json"),
)
AUTOTRADE_BLACKLIST_TTL_SECONDS = max(300, int(os.getenv("AUTOTRADE_BLACKLIST_TTL_SECONDS", "86400")))
AUTOTRADE_BLACKLIST_UNKNOWN_TTL_SECONDS = max(
    300,
    int(os.getenv("AUTOTRADE_BLACKLIST_UNKNOWN_TTL_SECONDS", "900")),
)
AUTOTRADE_BLACKLIST_TRANSIENT_SAFETY_TTL_SECONDS = max(
    300,
    int(os.getenv("AUTOTRADE_BLACKLIST_TRANSIENT_SAFETY_TTL_SECONDS", "900")),
)
AUTOTRADE_BLACKLIST_MAX_ENTRIES = max(100, int(os.getenv("AUTOTRADE_BLACKLIST_MAX_ENTRIES", "5000")))
AUTOTRADE_BLACKLIST_PAPER_HARD_ONLY = os.getenv("AUTOTRADE_BLACKLIST_PAPER_HARD_ONLY", "true").lower() == "true"
# AUTO_TRADE_ENTRY_MODE:
# - single: open only one best candidate per scan cycle
# - all: open every eligible candidate
# - top_n: open best N eligible candidates per scan cycle
AUTO_TRADE_ENTRY_MODE = os.getenv("AUTO_TRADE_ENTRY_MODE", "single").lower()
AUTO_TRADE_TOP_N = int(os.getenv("AUTO_TRADE_TOP_N", "10"))
# Plan batch diversity controls (applied in AutoTrader._rebalance_plan_batch_sources).
PLAN_MAX_SINGLE_SOURCE_SHARE = max(
    0.20,
    min(1.00, float(os.getenv("PLAN_MAX_SINGLE_SOURCE_SHARE", "0.50"))),
)
PLAN_MAX_WATCHLIST_SHARE = max(
    0.00,
    min(1.00, float(os.getenv("PLAN_MAX_WATCHLIST_SHARE", "0.30"))),
)
PLAN_MIN_NON_WATCHLIST_PER_BATCH = max(0, int(os.getenv("PLAN_MIN_NON_WATCHLIST_PER_BATCH", "1")))
# Optional hard guard at plan stage: enforce non-watchlist quota in rolling open window.
PLAN_NON_WATCHLIST_QUOTA_ENABLED = os.getenv("PLAN_NON_WATCHLIST_QUOTA_ENABLED", "true").lower() == "true"
PLAN_NON_WATCHLIST_QUOTA_WINDOW_SECONDS = max(
    300,
    int(os.getenv("PLAN_NON_WATCHLIST_QUOTA_WINDOW_SECONDS", "900")),
)
PLAN_NON_WATCHLIST_QUOTA_MIN_OPENS = max(
    2,
    int(os.getenv("PLAN_NON_WATCHLIST_QUOTA_MIN_OPENS", "4")),
)
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
AUTO_TRADE_HARD_BLOCKED_ADDRESSES = [
    x.strip().lower()
    for x in os.getenv(
        "AUTO_TRADE_HARD_BLOCKED_ADDRESSES",
        # Project hard-block list: known toxic token(s) that must never be traded.
        "0xf30bf00edd0c22db54c9274b90d2a4c21fc09b07",
    ).split(",")
    if x.strip()
]
AUTO_TRADE_EXCLUDED_SYMBOLS = [
    x.strip().upper()
    for x in os.getenv("AUTO_TRADE_EXCLUDED_SYMBOLS", "").split(",")
    if x.strip()
]
AUTO_TRADE_EXCLUDED_SYMBOL_KEYWORDS = [
    x.strip().upper()
    for x in os.getenv("AUTO_TRADE_EXCLUDED_SYMBOL_KEYWORDS", "").split(",")
    if x.strip()
]
# Profit engine control loop (target-driven throttling by realized PnL/hour + avg net/trade).
PROFIT_ENGINE_ENABLED = os.getenv("PROFIT_ENGINE_ENABLED", "true").lower() == "true"
PROFIT_ENGINE_TARGET_PNL_PER_HOUR_USD = float(os.getenv("PROFIT_ENGINE_TARGET_PNL_PER_HOUR_USD", "0.50"))
PROFIT_ENGINE_WINDOW_MINUTES = max(15, int(os.getenv("PROFIT_ENGINE_WINDOW_MINUTES", "60")))
PROFIT_ENGINE_MIN_CLOSED = max(4, int(os.getenv("PROFIT_ENGINE_MIN_CLOSED", "8")))
PROFIT_ENGINE_STRICT_AVG_NET_USD = float(os.getenv("PROFIT_ENGINE_STRICT_AVG_NET_USD", "0.010"))
PROFIT_ENGINE_GOOD_AVG_NET_USD = float(os.getenv("PROFIT_ENGINE_GOOD_AVG_NET_USD", "0.020"))
PROFIT_ENGINE_MAX_N_CAP = max(1, int(os.getenv("PROFIT_ENGINE_MAX_N_CAP", "4")))
PROFIT_ENGINE_STRICT_CORE_ONLY = os.getenv("PROFIT_ENGINE_STRICT_CORE_ONLY", "true").lower() == "true"

# Burst governor (prevents correlated multi-opens in one batch/window).
BURST_GOVERNOR_ENABLED = os.getenv("BURST_GOVERNOR_ENABLED", "true").lower() == "true"
BURST_GOVERNOR_MAX_OPENS_PER_BATCH = max(1, int(os.getenv("BURST_GOVERNOR_MAX_OPENS_PER_BATCH", "2")))
BURST_GOVERNOR_MAX_PER_SYMBOL_PER_BATCH = max(1, int(os.getenv("BURST_GOVERNOR_MAX_PER_SYMBOL_PER_BATCH", "1")))
BURST_GOVERNOR_MAX_PER_CLUSTER_PER_BATCH = max(1, int(os.getenv("BURST_GOVERNOR_MAX_PER_CLUSTER_PER_BATCH", "1")))
BURST_GOVERNOR_WINDOW_SECONDS = max(5, int(os.getenv("BURST_GOVERNOR_WINDOW_SECONDS", "45")))
BURST_GOVERNOR_MAX_OPENS_PER_WINDOW = max(1, int(os.getenv("BURST_GOVERNOR_MAX_OPENS_PER_WINDOW", "2")))

# Source router (dynamic route budget by realized EV of source windows).
SOURCE_ROUTER_ENABLED = os.getenv("SOURCE_ROUTER_ENABLED", "true").lower() == "true"
SOURCE_ROUTER_WINDOW_MINUTES = max(30, int(os.getenv("SOURCE_ROUTER_WINDOW_MINUTES", "120")))
SOURCE_ROUTER_MIN_TRADES = max(4, int(os.getenv("SOURCE_ROUTER_MIN_TRADES", "8")))
SOURCE_ROUTER_BAD_AVG_NET_USD = float(os.getenv("SOURCE_ROUTER_BAD_AVG_NET_USD", "-0.0005"))
SOURCE_ROUTER_BAD_LOSS_SHARE = float(os.getenv("SOURCE_ROUTER_BAD_LOSS_SHARE", "0.62"))
SOURCE_ROUTER_BAD_ENTRY_PROBABILITY = max(
    0.05,
    min(1.0, float(os.getenv("SOURCE_ROUTER_BAD_ENTRY_PROBABILITY", "0.55"))),
)
SOURCE_ROUTER_SEVERE_AVG_NET_USD = float(os.getenv("SOURCE_ROUTER_SEVERE_AVG_NET_USD", "-0.0012"))
SOURCE_ROUTER_SEVERE_LOSS_SHARE = float(os.getenv("SOURCE_ROUTER_SEVERE_LOSS_SHARE", "0.72"))
SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY = max(
    0.01,
    min(1.0, float(os.getenv("SOURCE_ROUTER_SEVERE_ENTRY_PROBABILITY", "0.35"))),
)

# Paper sellability parity proxy: reject entries with too-high notional-vs-liquidity impact.
PAPER_ENTRY_IMPACT_GUARD_ENABLED = os.getenv("PAPER_ENTRY_IMPACT_GUARD_ENABLED", "true").lower() == "true"
PAPER_MAX_ENTRY_IMPACT_PERCENT = max(0.05, float(os.getenv("PAPER_MAX_ENTRY_IMPACT_PERCENT", "1.20")))
PAPER_ALLOW_WATCHLIST_SOURCE = os.getenv("PAPER_ALLOW_WATCHLIST_SOURCE", "false").lower() == "true"
PAPER_WATCHLIST_STRICT_GUARD_ENABLED = os.getenv("PAPER_WATCHLIST_STRICT_GUARD_ENABLED", "true").lower() == "true"
PAPER_WATCHLIST_MIN_SCORE = int(os.getenv("PAPER_WATCHLIST_MIN_SCORE", "90"))
PAPER_WATCHLIST_MIN_LIQUIDITY_USD = max(
    0.0,
    float(os.getenv("PAPER_WATCHLIST_MIN_LIQUIDITY_USD", "150000")),
)
PAPER_WATCHLIST_MIN_VOLUME_5M_USD = max(
    0.0,
    float(os.getenv("PAPER_WATCHLIST_MIN_VOLUME_5M_USD", "500")),
)
PAPER_WATCHLIST_MIN_ABS_CHANGE_5M = max(
    0.0,
    float(os.getenv("PAPER_WATCHLIST_MIN_ABS_CHANGE_5M", "0.30")),
)

KILL_SWITCH_FILE = os.getenv("KILL_SWITCH_FILE", os.path.join("data", "kill.txt"))
GRACEFUL_STOP_FILE = os.getenv("GRACEFUL_STOP_FILE", os.path.join("data", "graceful_stop.signal"))
GRACEFUL_STOP_TIMEOUT_SECONDS = max(2, int(os.getenv("GRACEFUL_STOP_TIMEOUT_SECONDS", "12")))
WALLET_BALANCE_USD = float(os.getenv("WALLET_BALANCE_USD", "2.75"))
PAPER_TRADE_SIZE_USD = float(os.getenv("PAPER_TRADE_SIZE_USD", "1.0"))
PAPER_MAX_HOLD_SECONDS = int(os.getenv("PAPER_MAX_HOLD_SECONDS", "1800"))
PAPER_STATE_FILE = os.getenv("PAPER_STATE_FILE", os.path.join("trading", "paper_state.json"))
PAPER_STATE_FLUSH_INTERVAL_SECONDS = max(0.2, float(os.getenv("PAPER_STATE_FLUSH_INTERVAL_SECONDS", "0.2")))
STATE_FILE_LOCK_TIMEOUT_SECONDS = max(0.1, float(os.getenv("STATE_FILE_LOCK_TIMEOUT_SECONDS", "2.0")))
STATE_FILE_LOCK_RETRY_SECONDS = max(0.01, float(os.getenv("STATE_FILE_LOCK_RETRY_SECONDS", "0.05")))

# Realistic paper simulation
PAPER_REALISM_ENABLED = os.getenv("PAPER_REALISM_ENABLED", "true").lower() == "true"
PAPER_GAS_PER_TX_USD = float(os.getenv("PAPER_GAS_PER_TX_USD", "0.03"))
PAPER_APPROVE_TX_ALWAYS = os.getenv("PAPER_APPROVE_TX_ALWAYS", "false").lower() == "true"
PAPER_APPROVE_TX_PROBABILITY = max(
    0.0,
    min(1.0, float(os.getenv("PAPER_APPROVE_TX_PROBABILITY", "0.30"))),
)
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
NO_MOMENTUM_EXIT_ENABLED = os.getenv("NO_MOMENTUM_EXIT_ENABLED", "true").lower() == "true"
NO_MOMENTUM_EXIT_MIN_AGE_PERCENT = float(os.getenv("NO_MOMENTUM_EXIT_MIN_AGE_PERCENT", "45"))
NO_MOMENTUM_EXIT_MIN_HOLD_SECONDS = max(0, int(os.getenv("NO_MOMENTUM_EXIT_MIN_HOLD_SECONDS", "180")))
NO_MOMENTUM_EXIT_MAX_PEAK_PERCENT = float(os.getenv("NO_MOMENTUM_EXIT_MAX_PEAK_PERCENT", "1.0"))
NO_MOMENTUM_EXIT_MAX_PNL_PERCENT = float(os.getenv("NO_MOMENTUM_EXIT_MAX_PNL_PERCENT", "0.3"))
NO_MOMENTUM_STRONG_TRADE_PROTECT_ENABLED = os.getenv("NO_MOMENTUM_STRONG_TRADE_PROTECT_ENABLED", "true").lower() == "true"
NO_MOMENTUM_STRONG_EXPECTED_EDGE_PERCENT = float(os.getenv("NO_MOMENTUM_STRONG_EXPECTED_EDGE_PERCENT", "2.8"))
NO_MOMENTUM_STRONG_MIN_AGE_EXTRA_PERCENT = max(
    0.0,
    float(os.getenv("NO_MOMENTUM_STRONG_MIN_AGE_EXTRA_PERCENT", "18")),
)
NO_MOMENTUM_STRONG_MIN_HOLD_EXTRA_SECONDS = max(
    0,
    int(os.getenv("NO_MOMENTUM_STRONG_MIN_HOLD_EXTRA_SECONDS", "40")),
)
NO_MOMENTUM_STRONG_MAX_PNL_SHIFT = float(os.getenv("NO_MOMENTUM_STRONG_MAX_PNL_SHIFT", "-0.10"))
WEAK_TRADE_EARLY_EXIT_ENABLED = os.getenv("WEAK_TRADE_EARLY_EXIT_ENABLED", "true").lower() == "true"
WEAK_TRADE_EARLY_EDGE_MAX_PERCENT = float(os.getenv("WEAK_TRADE_EARLY_EDGE_MAX_PERCENT", "1.85"))
WEAK_TRADE_EARLY_MIN_AGE_PERCENT = max(0.0, float(os.getenv("WEAK_TRADE_EARLY_MIN_AGE_PERCENT", "22")))
WEAK_TRADE_EARLY_MIN_HOLD_SECONDS = max(0, int(os.getenv("WEAK_TRADE_EARLY_MIN_HOLD_SECONDS", "55")))
WEAK_TRADE_EARLY_MAX_PNL_PERCENT = float(os.getenv("WEAK_TRADE_EARLY_MAX_PNL_PERCENT", "0.18"))
WEAK_TRADE_EARLY_MAX_PEAK_PERCENT = float(os.getenv("WEAK_TRADE_EARLY_MAX_PEAK_PERCENT", "0.95"))
PAPER_PARTIAL_TP_ENABLED = os.getenv("PAPER_PARTIAL_TP_ENABLED", "false").lower() == "true"
PAPER_PARTIAL_TP_TRIGGER_PERCENT = float(os.getenv("PAPER_PARTIAL_TP_TRIGGER_PERCENT", "2.5"))
PAPER_PARTIAL_TP_SELL_FRACTION = float(os.getenv("PAPER_PARTIAL_TP_SELL_FRACTION", "0.5"))
PAPER_PARTIAL_TP_STAGE2_ENABLED = os.getenv("PAPER_PARTIAL_TP_STAGE2_ENABLED", "true").lower() == "true"
PAPER_PARTIAL_TP_STAGE2_TRIGGER_MULT = max(1.05, float(os.getenv("PAPER_PARTIAL_TP_STAGE2_TRIGGER_MULT", "1.9")))
PAPER_PARTIAL_TP_STAGE2_SELL_FRACTION = float(os.getenv("PAPER_PARTIAL_TP_STAGE2_SELL_FRACTION", "0.35"))
PAPER_PARTIAL_TP_MIN_REMAINING_USD = max(0.0, float(os.getenv("PAPER_PARTIAL_TP_MIN_REMAINING_USD", "0.10")))
PAPER_PARTIAL_TP_MOVE_SL_TO_BREAK_EVEN = (
    os.getenv("PAPER_PARTIAL_TP_MOVE_SL_TO_BREAK_EVEN", "true").lower() == "true"
)
PAPER_PARTIAL_TP_BREAK_EVEN_BUFFER_PERCENT = float(os.getenv("PAPER_PARTIAL_TP_BREAK_EVEN_BUFFER_PERCENT", "0.2"))
PAPER_PARTIAL_TP_BREAK_EVEN_STAGE2_BONUS_PERCENT = float(
    os.getenv("PAPER_PARTIAL_TP_BREAK_EVEN_STAGE2_BONUS_PERCENT", "0.10")
)
PAPER_PARTIAL_TP_BE_PROTECT_ENABLED = os.getenv("PAPER_PARTIAL_TP_BE_PROTECT_ENABLED", "true").lower() == "true"
PAPER_PARTIAL_TP_BE_PROTECT_FLOOR_PERCENT = float(os.getenv("PAPER_PARTIAL_TP_BE_PROTECT_FLOOR_PERCENT", "0.05"))
PAPER_PARTIAL_TP_BE_PROTECT_STAGE2_BONUS_PERCENT = float(
    os.getenv("PAPER_PARTIAL_TP_BE_PROTECT_STAGE2_BONUS_PERCENT", "0.10")
)
PAPER_TRAILING_EXIT_ENABLED = os.getenv("PAPER_TRAILING_EXIT_ENABLED", "true").lower() == "true"
PAPER_TRAILING_ACTIVATE_PEAK_PERCENT = float(os.getenv("PAPER_TRAILING_ACTIVATE_PEAK_PERCENT", "1.6"))
PAPER_TRAILING_GIVEBACK_PERCENT = max(0.0, float(os.getenv("PAPER_TRAILING_GIVEBACK_PERCENT", "1.1")))
PAPER_TRAILING_GIVEBACK_RATIO = max(0.05, min(0.95, float(os.getenv("PAPER_TRAILING_GIVEBACK_RATIO", "0.55"))))
PAPER_TRAILING_MIN_PNL_PERCENT = float(os.getenv("PAPER_TRAILING_MIN_PNL_PERCENT", "0.2"))
MOMENTUM_DECAY_EXIT_ENABLED = os.getenv("MOMENTUM_DECAY_EXIT_ENABLED", "true").lower() == "true"
MOMENTUM_DECAY_MIN_AGE_PERCENT = max(0.0, float(os.getenv("MOMENTUM_DECAY_MIN_AGE_PERCENT", "18")))
MOMENTUM_DECAY_MIN_HOLD_SECONDS = max(0, int(os.getenv("MOMENTUM_DECAY_MIN_HOLD_SECONDS", "60")))
MOMENTUM_DECAY_MIN_PEAK_PERCENT = float(os.getenv("MOMENTUM_DECAY_MIN_PEAK_PERCENT", "2.0"))
MOMENTUM_DECAY_RETAIN_RATIO = max(0.0, min(1.0, float(os.getenv("MOMENTUM_DECAY_RETAIN_RATIO", "0.35"))))
MOMENTUM_DECAY_MIN_DROP_PERCENT = max(0.0, float(os.getenv("MOMENTUM_DECAY_MIN_DROP_PERCENT", "1.0")))
MOMENTUM_DECAY_MAX_PNL_PERCENT = float(os.getenv("MOMENTUM_DECAY_MAX_PNL_PERCENT", "0.8"))
EXIT_TIMEOUT_EXTENSION_ENABLED = os.getenv("EXIT_TIMEOUT_EXTENSION_ENABLED", "true").lower() == "true"
EXIT_TIMEOUT_EXTENSION_MAX_SECONDS = max(0, int(os.getenv("EXIT_TIMEOUT_EXTENSION_MAX_SECONDS", "60")))
EXIT_TIMEOUT_EXTENSION_EDGE_SECONDS = max(0, int(os.getenv("EXIT_TIMEOUT_EXTENSION_EDGE_SECONDS", "15")))
EXIT_TIMEOUT_EXTENSION_MIN_EDGE_PERCENT = float(os.getenv("EXIT_TIMEOUT_EXTENSION_MIN_EDGE_PERCENT", "1.4"))
EXIT_TIMEOUT_EXTENSION_MOMENTUM_SECONDS = max(0, int(os.getenv("EXIT_TIMEOUT_EXTENSION_MOMENTUM_SECONDS", "20")))
EXIT_TIMEOUT_EXTENSION_MIN_PEAK_PERCENT = float(os.getenv("EXIT_TIMEOUT_EXTENSION_MIN_PEAK_PERCENT", "1.6"))
EXIT_TIMEOUT_EXTENSION_MIN_PNL_PERCENT = float(os.getenv("EXIT_TIMEOUT_EXTENSION_MIN_PNL_PERCENT", "0.2"))
EXIT_TIMEOUT_EXTENSION_POST_PARTIAL_SECONDS = max(0, int(os.getenv("EXIT_TIMEOUT_EXTENSION_POST_PARTIAL_SECONDS", "10")))
EXIT_TIMEOUT_STRONG_TRADE_EXTRA_SECONDS = max(0, int(os.getenv("EXIT_TIMEOUT_STRONG_TRADE_EXTRA_SECONDS", "18")))
EXIT_TIMEOUT_STRONG_TRADE_MIN_EXPECTED_EDGE_PERCENT = float(
    os.getenv("EXIT_TIMEOUT_STRONG_TRADE_MIN_EXPECTED_EDGE_PERCENT", "2.8")
)
EXIT_TIMEOUT_STRONG_TRADE_ALLOW_FOR_TIER_A = os.getenv("EXIT_TIMEOUT_STRONG_TRADE_ALLOW_FOR_TIER_A", "true").lower() == "true"
EXIT_TIMEOUT_STRONG_TRADE_ALLOW_FOR_CORE = os.getenv("EXIT_TIMEOUT_STRONG_TRADE_ALLOW_FOR_CORE", "true").lower() == "true"

# Position sizing and edge filtering
DYNAMIC_POSITION_SIZING_ENABLED = os.getenv("DYNAMIC_POSITION_SIZING_ENABLED", "true").lower() == "true"
EDGE_FILTER_ENABLED = os.getenv("EDGE_FILTER_ENABLED", "true").lower() == "true"
MIN_EXPECTED_EDGE_PERCENT = float(os.getenv("MIN_EXPECTED_EDGE_PERCENT", "2.0"))
EDGE_FILTER_MODE = os.getenv("EDGE_FILTER_MODE", "usd").strip().lower()
MIN_EXPECTED_EDGE_USD = float(os.getenv("MIN_EXPECTED_EDGE_USD", "0.10"))
ENTRY_EDGE_SOFTEN_GREEN_A_CORE_ENABLED = os.getenv(
    "ENTRY_EDGE_SOFTEN_GREEN_A_CORE_ENABLED",
    "true",
).lower() == "true"
ENTRY_EDGE_SOFTEN_GREEN_A_CORE_PCT_MULT = max(
    0.50,
    min(1.0, float(os.getenv("ENTRY_EDGE_SOFTEN_GREEN_A_CORE_PCT_MULT", "0.92"))),
)
ENTRY_EDGE_SOFTEN_GREEN_A_CORE_USD_MULT = max(
    0.50,
    min(1.0, float(os.getenv("ENTRY_EDGE_SOFTEN_GREEN_A_CORE_USD_MULT", "0.92"))),
)
ENTRY_A_CORE_MIN_TRADE_USD_ENABLED = os.getenv("ENTRY_A_CORE_MIN_TRADE_USD_ENABLED", "true").lower() == "true"
ENTRY_A_CORE_MIN_TRADE_USD = max(0.0, float(os.getenv("ENTRY_A_CORE_MIN_TRADE_USD", "0.45")))
ENTRY_A_CORE_MIN_TRADE_APPLY_IN_RED = os.getenv("ENTRY_A_CORE_MIN_TRADE_APPLY_IN_RED", "false").lower() == "true"
ENTRY_A_CORE_MIN_TRADE_APPLY_ANY_TIER_A = (
    os.getenv("ENTRY_A_CORE_MIN_TRADE_APPLY_ANY_TIER_A", "false").lower() == "true"
)
ENTRY_SMALL_SIZE_EV_RELAX_ENABLED = os.getenv("ENTRY_SMALL_SIZE_EV_RELAX_ENABLED", "true").lower() == "true"
ENTRY_SMALL_SIZE_EV_RELAX_NOMINAL_USD = max(0.1, float(os.getenv("ENTRY_SMALL_SIZE_EV_RELAX_NOMINAL_USD", "1.00")))
ENTRY_SMALL_SIZE_EV_RELAX_MIN_FACTOR = max(
    0.10,
    min(1.0, float(os.getenv("ENTRY_SMALL_SIZE_EV_RELAX_MIN_FACTOR", "0.50"))),
)
ENTRY_SMALL_SIZE_EDGE_RELAX_ENABLED = os.getenv("ENTRY_SMALL_SIZE_EDGE_RELAX_ENABLED", "true").lower() == "true"
ENTRY_SMALL_SIZE_EDGE_RELAX_NOMINAL_USD = max(
    0.1,
    float(os.getenv("ENTRY_SMALL_SIZE_EDGE_RELAX_NOMINAL_USD", "1.00")),
)
ENTRY_SMALL_SIZE_EDGE_RELAX_MIN_FACTOR = max(
    0.10,
    min(1.0, float(os.getenv("ENTRY_SMALL_SIZE_EDGE_RELAX_MIN_FACTOR", "0.50"))),
)
ENTRY_REQUIRE_POSITIVE_CHANGE_5M = os.getenv("ENTRY_REQUIRE_POSITIVE_CHANGE_5M", "false").lower() == "true"
ENTRY_MIN_PRICE_CHANGE_5M_PERCENT = float(os.getenv("ENTRY_MIN_PRICE_CHANGE_5M_PERCENT", "0.0"))
ENTRY_REQUIRE_VOLUME_BUFFER = os.getenv("ENTRY_REQUIRE_VOLUME_BUFFER", "false").lower() == "true"
ENTRY_MIN_VOLUME_5M_MULT = float(os.getenv("ENTRY_MIN_VOLUME_5M_MULT", "1.0"))
ENTRY_MIN_VOLUME_5M_USD = float(os.getenv("ENTRY_MIN_VOLUME_5M_USD", "0.0"))

# Position sizing quality multiplier (score/liquidity/volume/volatility).
POSITION_SIZE_QUALITY_ENABLED = os.getenv("POSITION_SIZE_QUALITY_ENABLED", "true").lower() == "true"
PAPER_TRADE_SIZE_MIN_USD = float(os.getenv("PAPER_TRADE_SIZE_MIN_USD", "0.25"))
PAPER_TRADE_SIZE_MAX_USD = float(os.getenv("PAPER_TRADE_SIZE_MAX_USD", str(PAPER_TRADE_SIZE_USD)))
CLOSED_TRADES_MAX_AGE_DAYS = max(0, int(os.getenv("CLOSED_TRADES_MAX_AGE_DAYS", "14")))
DYNAMIC_HOLD_ENABLED = os.getenv("DYNAMIC_HOLD_ENABLED", "true").lower() == "true"
HOLD_MIN_SECONDS = max(60, int(os.getenv("HOLD_MIN_SECONDS", "300")))
HOLD_MAX_SECONDS = max(HOLD_MIN_SECONDS, int(os.getenv("HOLD_MAX_SECONDS", str(PAPER_MAX_HOLD_SECONDS))))
PAPER_RESET_ON_START = os.getenv("PAPER_RESET_ON_START", "false").lower() == "true"
# In paper/matrix runs force a clean session even if stale open positions remained in state.
PAPER_RESET_FORCE_ON_START = os.getenv("PAPER_RESET_FORCE_ON_START", "true").lower() == "true"
PAPER_PRICE_GUARD_ENABLED = os.getenv("PAPER_PRICE_GUARD_ENABLED", "true").lower() == "true"
PAPER_PRICE_GUARD_MAX_JUMP_PERCENT = float(os.getenv("PAPER_PRICE_GUARD_MAX_JUMP_PERCENT", "250"))
PAPER_PRICE_GUARD_CONFIRMATIONS = max(1, int(os.getenv("PAPER_PRICE_GUARD_CONFIRMATIONS", "2")))
PAPER_PRICE_GUARD_WINDOW_SECONDS = max(5, int(os.getenv("PAPER_PRICE_GUARD_WINDOW_SECONDS", "90")))
PAPER_PRICE_GUARD_MAX_RATIO = max(1.5, float(os.getenv("PAPER_PRICE_GUARD_MAX_RATIO", "50")))
PAPER_PRICE_MAX_ABSOLUTE_USD = max(0.01, float(os.getenv("PAPER_PRICE_MAX_ABSOLUTE_USD", "1000000")))
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
AUTO_TRADE_TOKEN_COOLDOWN_DYNAMIC_ENABLED = (
    os.getenv("AUTO_TRADE_TOKEN_COOLDOWN_DYNAMIC_ENABLED", "true").lower() == "true"
)
AUTO_TRADE_TOKEN_COOLDOWN_STEP_SECONDS = max(
    0,
    int(os.getenv("AUTO_TRADE_TOKEN_COOLDOWN_STEP_SECONDS", "300")),
)
AUTO_TRADE_TOKEN_COOLDOWN_MAX_STRIKES = max(
    1,
    int(os.getenv("AUTO_TRADE_TOKEN_COOLDOWN_MAX_STRIKES", "4")),
)
AUTO_TRADE_TOKEN_COOLDOWN_RECOVERY_STEP = max(
    1,
    int(os.getenv("AUTO_TRADE_TOKEN_COOLDOWN_RECOVERY_STEP", "1")),
)
AUTO_TRADE_TOKEN_COOLDOWN_MAX_SECONDS = max(
    MAX_TOKEN_COOLDOWN_SECONDS,
    int(os.getenv("AUTO_TRADE_TOKEN_COOLDOWN_MAX_SECONDS", "3600")),
)
AUTO_TRADE_TOKEN_COOLDOWN_ESCALATE_REASONS = [
    x.strip().upper()
    for x in os.getenv(
        "AUTO_TRADE_TOKEN_COOLDOWN_ESCALATE_REASONS",
        "SL,NO_MOMENTUM,TIMEOUT,WEAKNESS,ABANDON",
    ).split(",")
    if x.strip()
]
EXTREME_SL_GUARD_ENABLED = os.getenv("EXTREME_SL_GUARD_ENABLED", "true").lower() == "true"
EXTREME_SL_GUARD_REASONS = [
    x.strip().upper()
    for x in os.getenv("EXTREME_SL_GUARD_REASONS", "SL").split(",")
    if x.strip()
]
EXTREME_SL_PNL_PERCENT_THRESHOLD = float(os.getenv("EXTREME_SL_PNL_PERCENT_THRESHOLD", "-20"))
EXTREME_SL_TOKEN_COOLDOWN_SECONDS = max(
    0,
    int(os.getenv("EXTREME_SL_TOKEN_COOLDOWN_SECONDS", "5400")),
)
EXTREME_SL_SYMBOL_COOLDOWN_SECONDS = max(
    0,
    int(os.getenv("EXTREME_SL_SYMBOL_COOLDOWN_SECONDS", "5400")),
)
CORE_BLOCK_L1_FOR_CORE_ENABLED = os.getenv("CORE_BLOCK_L1_FOR_CORE_ENABLED", "true").lower() == "true"
CORE_BLOCK_L1_FALLBACK_TO_EXPLORE = os.getenv("CORE_BLOCK_L1_FALLBACK_TO_EXPLORE", "true").lower() == "true"
CORE_L1_FALLBACK_SIZE_MULT = max(0.10, float(os.getenv("CORE_L1_FALLBACK_SIZE_MULT", "0.65")))
CORE_L1_FALLBACK_HOLD_MULT = max(0.20, float(os.getenv("CORE_L1_FALLBACK_HOLD_MULT", "0.75")))
CORE_L1_FALLBACK_EDGE_USD_MULT = max(1.00, float(os.getenv("CORE_L1_FALLBACK_EDGE_USD_MULT", "1.25")))
CORE_L1_FALLBACK_EDGE_PERCENT_MULT = max(1.00, float(os.getenv("CORE_L1_FALLBACK_EDGE_PERCENT_MULT", "1.20")))
SYMBOL_EV_GUARD_ENABLED = os.getenv("SYMBOL_EV_GUARD_ENABLED", "true").lower() == "true"
SYMBOL_EV_WINDOW_MINUTES = max(10, int(os.getenv("SYMBOL_EV_WINDOW_MINUTES", "120")))
SYMBOL_EV_MIN_TRADES = max(1, int(os.getenv("SYMBOL_EV_MIN_TRADES", "3")))
SYMBOL_EV_MIN_AVG_PNL_USD = float(os.getenv("SYMBOL_EV_MIN_AVG_PNL_USD", "0.0005"))
SYMBOL_EV_MAX_LOSS_SHARE = max(0.0, min(1.0, float(os.getenv("SYMBOL_EV_MAX_LOSS_SHARE", "0.60"))))
# `skip` hard-blocks bad symbol EV; `degrade` keeps symbol tradable with EV-memory penalties.
SYMBOL_EV_BAD_ACTION = str(os.getenv("SYMBOL_EV_BAD_ACTION", "degrade") or "degrade").strip().lower()
SYMBOL_EV_BAD_TO_STRICT_ONLY = os.getenv("SYMBOL_EV_BAD_TO_STRICT_ONLY", "true").lower() == "true"
SYMBOL_EV_BAD_COOLDOWN_SECONDS = max(0, int(os.getenv("SYMBOL_EV_BAD_COOLDOWN_SECONDS", "1200")))
SYMBOL_FATIGUE_MAX_TRADES_PER_WINDOW = max(0, int(os.getenv("SYMBOL_FATIGUE_MAX_TRADES_PER_WINDOW", "4")))
SYMBOL_FATIGUE_MAX_LOSS_STREAK = max(0, int(os.getenv("SYMBOL_FATIGUE_MAX_LOSS_STREAK", "3")))
SYMBOL_FATIGUE_COOLDOWN_SECONDS = max(0, int(os.getenv("SYMBOL_FATIGUE_COOLDOWN_SECONDS", "1800")))
SYMBOL_CONCENTRATION_GUARD_ENABLED = os.getenv("SYMBOL_CONCENTRATION_GUARD_ENABLED", "true").lower() == "true"
SYMBOL_CONCENTRATION_WINDOW_SECONDS = max(
    300,
    int(os.getenv("SYMBOL_CONCENTRATION_WINDOW_SECONDS", "3600")),
)
SYMBOL_CONCENTRATION_MIN_OPENS = max(
    4,
    int(os.getenv("SYMBOL_CONCENTRATION_MIN_OPENS", "12")),
)
SYMBOL_CONCENTRATION_MAX_SHARE = max(
    0.05,
    min(1.0, float(os.getenv("SYMBOL_CONCENTRATION_MAX_SHARE", "0.10"))),
)
SYMBOL_CONCENTRATION_TIER_A_SHARE_MULT = max(
    1.0,
    float(os.getenv("SYMBOL_CONCENTRATION_TIER_A_SHARE_MULT", "1.15")),
)
# Hard anti-repeat block for 15m rolling opens (separate from concentration guard and anti-choke bypasses).
TOP1_OPEN_SHARE_15M_GUARD_ENABLED = os.getenv("TOP1_OPEN_SHARE_15M_GUARD_ENABLED", "true").lower() == "true"
TOP1_OPEN_SHARE_15M_WINDOW_SECONDS = max(
    300,
    int(os.getenv("TOP1_OPEN_SHARE_15M_WINDOW_SECONDS", "900")),
)
TOP1_OPEN_SHARE_15M_MIN_OPENS = max(
    2,
    int(os.getenv("TOP1_OPEN_SHARE_15M_MIN_OPENS", "4")),
)
TOP1_OPEN_SHARE_15M_MAX_SHARE = max(
    0.20,
    min(1.0, float(os.getenv("TOP1_OPEN_SHARE_15M_MAX_SHARE", "0.72"))),
)

# EV-core (entry/sizing/exit) controls.
EV_FIRST_ENTRY_ENABLED = os.getenv("EV_FIRST_ENTRY_ENABLED", "true").lower() == "true"
EV_FIRST_ENTRY_MIN_SAMPLES = max(5, int(os.getenv("EV_FIRST_ENTRY_MIN_SAMPLES", "24")))
EV_FIRST_ENTRY_MIN_NET_USD = float(os.getenv("EV_FIRST_ENTRY_MIN_NET_USD", "0.0010"))
EV_FIRST_ENTRY_MIN_NET_USD_GREEN_MULT = max(0.2, float(os.getenv("EV_FIRST_ENTRY_MIN_NET_USD_GREEN_MULT", "1.00")))
EV_FIRST_ENTRY_MIN_NET_USD_GREEN_A_CORE_MULT = max(
    0.20,
    float(os.getenv("EV_FIRST_ENTRY_MIN_NET_USD_GREEN_A_CORE_MULT", "0.95")),
)
EV_FIRST_ENTRY_MIN_NET_USD_YELLOW_MULT = max(0.2, float(os.getenv("EV_FIRST_ENTRY_MIN_NET_USD_YELLOW_MULT", "1.08")))
EV_FIRST_ENTRY_MIN_NET_USD_RED_MULT = max(0.2, float(os.getenv("EV_FIRST_ENTRY_MIN_NET_USD_RED_MULT", "1.20")))
EV_FIRST_ENTRY_MIN_NET_USD_EXPLORE_MULT = max(0.2, float(os.getenv("EV_FIRST_ENTRY_MIN_NET_USD_EXPLORE_MULT", "1.20")))
EV_FIRST_ENTRY_FALLBACK_WEIGHT_MAX = max(0.0, min(1.0, float(os.getenv("EV_FIRST_ENTRY_FALLBACK_WEIGHT_MAX", "0.70"))))
EV_FIRST_ENTRY_COST_HAIRCUT_USD = max(0.0, float(os.getenv("EV_FIRST_ENTRY_COST_HAIRCUT_USD", "0.0005")))
EV_FIRST_ENTRY_LOCAL_WINDOW_MINUTES = max(30, int(os.getenv("EV_FIRST_ENTRY_LOCAL_WINDOW_MINUTES", "360")))
EV_FIRST_ENTRY_DB_LOOKBACK_ROWS = max(100, int(os.getenv("EV_FIRST_ENTRY_DB_LOOKBACK_ROWS", "2500")))
EV_FIRST_ENTRY_DB_BUCKET_SCORE_WIDTH = max(5, int(os.getenv("EV_FIRST_ENTRY_DB_BUCKET_SCORE_WIDTH", "10")))
EV_FIRST_ENTRY_DB_CACHE_TTL_SECONDS = max(10, int(os.getenv("EV_FIRST_ENTRY_DB_CACHE_TTL_SECONDS", "90")))
EV_FIRST_ENTRY_DB_PATH = str(
    os.getenv(
        "EV_FIRST_ENTRY_DB_PATH",
        os.getenv("V2_CALIBRATION_DB_PATH", os.path.join("data", "unified_dataset", "unified.db")),
    )
    or os.path.join("data", "unified_dataset", "unified.db")
)
EV_FIRST_ENTRY_CORE_PROBE_ENABLED = os.getenv("EV_FIRST_ENTRY_CORE_PROBE_ENABLED", "true").lower() == "true"
EV_FIRST_ENTRY_CORE_PROBE_RELAX_ONLY = os.getenv("EV_FIRST_ENTRY_CORE_PROBE_RELAX_ONLY", "true").lower() == "true"
EV_FIRST_ENTRY_CORE_PROBE_WINDOW_SECONDS = max(
    60,
    int(os.getenv("EV_FIRST_ENTRY_CORE_PROBE_WINDOW_SECONDS", "900")),
)
EV_FIRST_ENTRY_CORE_PROBE_MAX_OPENS = max(
    0,
    int(os.getenv("EV_FIRST_ENTRY_CORE_PROBE_MAX_OPENS", "1")),
)
EV_FIRST_ENTRY_CORE_PROBE_MIN_NET_USD = float(
    os.getenv("EV_FIRST_ENTRY_CORE_PROBE_MIN_NET_USD", "-0.0015")
)
EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD = max(
    0.0,
    float(os.getenv("EV_FIRST_ENTRY_CORE_PROBE_EV_TOLERANCE_USD", "0.0030")),
)
EV_FIRST_ENTRY_CORE_PROBE_EDGE_PERCENT_RELAX_RATIO = max(
    0.0,
    min(0.80, float(os.getenv("EV_FIRST_ENTRY_CORE_PROBE_EDGE_PERCENT_RELAX_RATIO", "0.20"))),
)
EV_FIRST_ENTRY_CORE_PROBE_EDGE_USD_RELAX_ABS = max(
    0.0,
    float(os.getenv("EV_FIRST_ENTRY_CORE_PROBE_EDGE_USD_RELAX_ABS", "0.0020")),
)
ENTRY_QUALITY_ADAPT_ENABLED = os.getenv("ENTRY_QUALITY_ADAPT_ENABLED", "true").lower() == "true"
ENTRY_QUALITY_SOURCE_EDGE_ADAPT_ENABLED = os.getenv(
    "ENTRY_QUALITY_SOURCE_EDGE_ADAPT_ENABLED", "true"
).lower() == "true"
ENTRY_QUALITY_SOURCE_EDGE_RELAX_MAX = max(
    0.0,
    min(0.60, float(os.getenv("ENTRY_QUALITY_SOURCE_EDGE_RELAX_MAX", "0.16"))),
)
ENTRY_QUALITY_SOURCE_EDGE_TIGHTEN_MAX = max(
    0.0,
    min(0.80, float(os.getenv("ENTRY_QUALITY_SOURCE_EDGE_TIGHTEN_MAX", "0.24"))),
)
ENTRY_QUALITY_EV_SIGNAL_EDGE_SCALE = max(
    0.0,
    float(os.getenv("ENTRY_QUALITY_EV_SIGNAL_EDGE_SCALE", "18.0")),
)
ENTRY_QUALITY_SOURCE_SIZE_ADAPT_ENABLED = os.getenv(
    "ENTRY_QUALITY_SOURCE_SIZE_ADAPT_ENABLED", "true"
).lower() == "true"
ENTRY_QUALITY_SOURCE_SIZE_RELAX_MAX = max(
    0.0,
    min(0.80, float(os.getenv("ENTRY_QUALITY_SOURCE_SIZE_RELAX_MAX", "0.18"))),
)
ENTRY_QUALITY_SOURCE_SIZE_CUT_MAX = max(
    0.0,
    min(0.80, float(os.getenv("ENTRY_QUALITY_SOURCE_SIZE_CUT_MAX", "0.24"))),
)
ENTRY_QUALITY_EV_SIGNAL_SIZE_SCALE = max(
    0.0,
    float(os.getenv("ENTRY_QUALITY_EV_SIGNAL_SIZE_SCALE", "22.0")),
)
ENTRY_QUALITY_CORE_BONUS_SIZE_MULT = max(
    0.1,
    float(os.getenv("ENTRY_QUALITY_CORE_BONUS_SIZE_MULT", "1.04")),
)
ENTRY_QUALITY_EXPLORE_CUT_SIZE_MULT = max(
    0.05,
    float(os.getenv("ENTRY_QUALITY_EXPLORE_CUT_SIZE_MULT", "0.92")),
)

KELLY_LITE_ENABLED = os.getenv("KELLY_LITE_ENABLED", "true").lower() == "true"
KELLY_LITE_MIN_MULT = max(0.10, float(os.getenv("KELLY_LITE_MIN_MULT", "0.55")))
KELLY_LITE_MAX_MULT = max(KELLY_LITE_MIN_MULT, float(os.getenv("KELLY_LITE_MAX_MULT", "1.55")))
KELLY_LITE_MAX_FRACTION = max(0.05, min(1.0, float(os.getenv("KELLY_LITE_MAX_FRACTION", "0.45"))))
KELLY_LITE_MIN_CONFIDENCE_SAMPLES = max(5, int(os.getenv("KELLY_LITE_MIN_CONFIDENCE_SAMPLES", "18")))

TOKEN_EV_MEMORY_ENABLED = os.getenv("TOKEN_EV_MEMORY_ENABLED", "true").lower() == "true"
TOKEN_EV_MEMORY_MIN_TRADES = max(2, int(os.getenv("TOKEN_EV_MEMORY_MIN_TRADES", "4")))
TOKEN_EV_MEMORY_BAD_AVG_PNL_USD = float(os.getenv("TOKEN_EV_MEMORY_BAD_AVG_PNL_USD", "0.0002"))
TOKEN_EV_MEMORY_BAD_LOSS_SHARE = max(0.0, min(1.0, float(os.getenv("TOKEN_EV_MEMORY_BAD_LOSS_SHARE", "0.62"))))
TOKEN_EV_MEMORY_SEVERE_AVG_PNL_USD = float(os.getenv("TOKEN_EV_MEMORY_SEVERE_AVG_PNL_USD", "-0.0018"))
TOKEN_EV_MEMORY_SEVERE_LOSS_SHARE = max(0.0, min(1.0, float(os.getenv("TOKEN_EV_MEMORY_SEVERE_LOSS_SHARE", "0.72"))))
TOKEN_EV_MEMORY_BAD_SIZE_MULT = max(0.05, float(os.getenv("TOKEN_EV_MEMORY_BAD_SIZE_MULT", "0.78")))
TOKEN_EV_MEMORY_SEVERE_SIZE_MULT = max(0.05, float(os.getenv("TOKEN_EV_MEMORY_SEVERE_SIZE_MULT", "0.58")))
TOKEN_EV_MEMORY_BAD_EDGE_MULT = max(0.20, float(os.getenv("TOKEN_EV_MEMORY_BAD_EDGE_MULT", "1.12")))
TOKEN_EV_MEMORY_SEVERE_EDGE_MULT = max(0.20, float(os.getenv("TOKEN_EV_MEMORY_SEVERE_EDGE_MULT", "1.28")))
TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY = max(0.0, min(1.0, float(os.getenv("TOKEN_EV_MEMORY_BAD_ENTRY_PROBABILITY", "0.72"))))
TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY = max(0.0, min(1.0, float(os.getenv("TOKEN_EV_MEMORY_SEVERE_ENTRY_PROBABILITY", "0.42"))))

EXECUTION_ARCH_ENABLED = os.getenv("EXECUTION_ARCH_ENABLED", "true").lower() == "true"
EXECUTION_DUAL_LANE_ENABLED = os.getenv("EXECUTION_DUAL_LANE_ENABLED", "true").lower() == "true"
EXECUTION_RUNNER_MIN_EDGE_PERCENT = float(os.getenv("EXECUTION_RUNNER_MIN_EDGE_PERCENT", "2.90"))
EXECUTION_RUNNER_MIN_EV_USD = float(os.getenv("EXECUTION_RUNNER_MIN_EV_USD", "0.0055"))
EXECUTION_RUNNER_MIN_SCORE = max(0, int(os.getenv("EXECUTION_RUNNER_MIN_SCORE", "92")))
EXECUTION_RUNNER_ALLOW_IN_RED = os.getenv("EXECUTION_RUNNER_ALLOW_IN_RED", "false").lower() == "true"
EXECUTION_RUNNER_ALLOW_EXPLORE = os.getenv("EXECUTION_RUNNER_ALLOW_EXPLORE", "false").lower() == "true"
EXECUTION_RUNNER_SIZE_MULT = max(0.10, float(os.getenv("EXECUTION_RUNNER_SIZE_MULT", "1.10")))
EXECUTION_RUNNER_HOLD_MULT = max(0.20, float(os.getenv("EXECUTION_RUNNER_HOLD_MULT", "1.32")))
EXECUTION_RUNNER_EDGE_GATE_MULT = max(0.20, float(os.getenv("EXECUTION_RUNNER_EDGE_GATE_MULT", "0.94")))
EXECUTION_SCALP_SIZE_MULT = max(0.10, float(os.getenv("EXECUTION_SCALP_SIZE_MULT", "0.93")))
EXECUTION_SCALP_HOLD_MULT = max(0.20, float(os.getenv("EXECUTION_SCALP_HOLD_MULT", "0.78")))
EXECUTION_SCALP_EDGE_GATE_MULT = max(0.20, float(os.getenv("EXECUTION_SCALP_EDGE_GATE_MULT", "1.06")))

EXECUTION_CLUSTER_ROUTING_ENABLED = os.getenv("EXECUTION_CLUSTER_ROUTING_ENABLED", "true").lower() == "true"
EXECUTION_CLUSTER_ROUTING_MIN_TRADES = max(2, int(os.getenv("EXECUTION_CLUSTER_ROUTING_MIN_TRADES", "10")))
EXECUTION_CLUSTER_ROUTING_BAD_AVG_NET_USD = float(os.getenv("EXECUTION_CLUSTER_ROUTING_BAD_AVG_NET_USD", "-0.0002"))
EXECUTION_CLUSTER_ROUTING_BAD_LOSS_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("EXECUTION_CLUSTER_ROUTING_BAD_LOSS_SHARE", "0.60"))),
)
EXECUTION_CLUSTER_ROUTING_SEVERE_AVG_NET_USD = float(
    os.getenv("EXECUTION_CLUSTER_ROUTING_SEVERE_AVG_NET_USD", "-0.0012")
)
EXECUTION_CLUSTER_ROUTING_SEVERE_LOSS_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("EXECUTION_CLUSTER_ROUTING_SEVERE_LOSS_SHARE", "0.72"))),
)
EXECUTION_CLUSTER_ROUTING_BAD_SIZE_MULT = max(0.05, float(os.getenv("EXECUTION_CLUSTER_ROUTING_BAD_SIZE_MULT", "0.84")))
EXECUTION_CLUSTER_ROUTING_BAD_HOLD_MULT = max(0.20, float(os.getenv("EXECUTION_CLUSTER_ROUTING_BAD_HOLD_MULT", "0.90")))
EXECUTION_CLUSTER_ROUTING_BAD_EDGE_MULT = max(0.20, float(os.getenv("EXECUTION_CLUSTER_ROUTING_BAD_EDGE_MULT", "1.10")))
EXECUTION_CLUSTER_ROUTING_BAD_ENTRY_PROBABILITY = max(
    0.0,
    min(1.0, float(os.getenv("EXECUTION_CLUSTER_ROUTING_BAD_ENTRY_PROBABILITY", "0.80"))),
)
EXECUTION_CLUSTER_ROUTING_SEVERE_SIZE_MULT = max(
    0.05,
    float(os.getenv("EXECUTION_CLUSTER_ROUTING_SEVERE_SIZE_MULT", "0.62")),
)
EXECUTION_CLUSTER_ROUTING_SEVERE_HOLD_MULT = max(
    0.20,
    float(os.getenv("EXECUTION_CLUSTER_ROUTING_SEVERE_HOLD_MULT", "0.72")),
)
EXECUTION_CLUSTER_ROUTING_SEVERE_EDGE_MULT = max(
    0.20,
    float(os.getenv("EXECUTION_CLUSTER_ROUTING_SEVERE_EDGE_MULT", "1.28")),
)
EXECUTION_CLUSTER_ROUTING_SEVERE_ENTRY_PROBABILITY = max(
    0.0,
    min(1.0, float(os.getenv("EXECUTION_CLUSTER_ROUTING_SEVERE_ENTRY_PROBABILITY", "0.48"))),
)
EXECUTION_CLUSTER_ROUTING_A_CORE_RELAX_EDGE_MULT = max(
    0.20,
    float(os.getenv("EXECUTION_CLUSTER_ROUTING_A_CORE_RELAX_EDGE_MULT", "0.94")),
)

STATEFUL_EXITS_ENABLED = os.getenv("STATEFUL_EXITS_ENABLED", "true").lower() == "true"
STATEFUL_EXITS_EARLY_PHASE_RATIO = max(
    0.05,
    min(0.95, float(os.getenv("STATEFUL_EXITS_EARLY_PHASE_RATIO", "0.33"))),
)
STATEFUL_EXITS_LATE_PHASE_RATIO = max(
    STATEFUL_EXITS_EARLY_PHASE_RATIO + 0.05,
    min(0.99, float(os.getenv("STATEFUL_EXITS_LATE_PHASE_RATIO", "0.78"))),
)
STATEFUL_TIMEOUT_RUNNER_EXTRA_SECONDS = max(
    0,
    int(os.getenv("STATEFUL_TIMEOUT_RUNNER_EXTRA_SECONDS", "26")),
)
STATEFUL_TIMEOUT_SCALP_REDUCE_SECONDS = max(
    0,
    int(os.getenv("STATEFUL_TIMEOUT_SCALP_REDUCE_SECONDS", "14")),
)
STATEFUL_NO_MOMENTUM_RUNNER_AGE_SHIFT_SECONDS = int(
    os.getenv("STATEFUL_NO_MOMENTUM_RUNNER_AGE_SHIFT_SECONDS", "24")
)
STATEFUL_NO_MOMENTUM_RUNNER_PNL_SHIFT = float(
    os.getenv("STATEFUL_NO_MOMENTUM_RUNNER_PNL_SHIFT", "-0.10")
)
STATEFUL_NO_MOMENTUM_SCALP_AGE_SHIFT_SECONDS = int(
    os.getenv("STATEFUL_NO_MOMENTUM_SCALP_AGE_SHIFT_SECONDS", "-10")
)
STATEFUL_NO_MOMENTUM_SCALP_PNL_SHIFT = float(
    os.getenv("STATEFUL_NO_MOMENTUM_SCALP_PNL_SHIFT", "0.06")
)
STATEFUL_WEAK_EARLY_RUNNER_EDGE_BONUS_PERCENT = float(
    os.getenv("STATEFUL_WEAK_EARLY_RUNNER_EDGE_BONUS_PERCENT", "0.35")
)
STATEFUL_WEAK_EARLY_SCALP_EDGE_PENALTY_PERCENT = float(
    os.getenv("STATEFUL_WEAK_EARLY_SCALP_EDGE_PENALTY_PERCENT", "0.25")
)
STATEFUL_PRE_PARTIAL_RUNNER_MIN_AGE_EXTRA_SECONDS = int(
    os.getenv("STATEFUL_PRE_PARTIAL_RUNNER_MIN_AGE_EXTRA_SECONDS", "14")
)
STATEFUL_PRE_PARTIAL_RUNNER_MAX_LOSS_SHIFT = float(
    os.getenv("STATEFUL_PRE_PARTIAL_RUNNER_MAX_LOSS_SHIFT", "-0.18")
)
STATEFUL_PRE_PARTIAL_SCALP_MIN_AGE_SHIFT_SECONDS = int(
    os.getenv("STATEFUL_PRE_PARTIAL_SCALP_MIN_AGE_SHIFT_SECONDS", "-8")
)
STATEFUL_PRE_PARTIAL_SCALP_MAX_LOSS_SHIFT = float(
    os.getenv("STATEFUL_PRE_PARTIAL_SCALP_MAX_LOSS_SHIFT", "0.15")
)
STATEFUL_POST_PARTIAL_RUNNER_GIVEBACK_MULT = max(
    0.10,
    float(os.getenv("STATEFUL_POST_PARTIAL_RUNNER_GIVEBACK_MULT", "0.82")),
)
STATEFUL_POST_PARTIAL_SCALP_GIVEBACK_MULT = max(
    0.10,
    float(os.getenv("STATEFUL_POST_PARTIAL_SCALP_GIVEBACK_MULT", "1.12")),
)

ASYMMETRIC_EXITS_ENABLED = os.getenv("ASYMMETRIC_EXITS_ENABLED", "true").lower() == "true"
ASYM_PRE_PARTIAL_RISK_ENABLED = os.getenv("ASYM_PRE_PARTIAL_RISK_ENABLED", "true").lower() == "true"
ASYM_PRE_PARTIAL_MIN_AGE_SECONDS = max(0, int(os.getenv("ASYM_PRE_PARTIAL_MIN_AGE_SECONDS", "24")))
ASYM_PRE_PARTIAL_MAX_PEAK_PERCENT = float(os.getenv("ASYM_PRE_PARTIAL_MAX_PEAK_PERCENT", "0.90"))
ASYM_PRE_PARTIAL_MAX_LOSS_PERCENT = float(os.getenv("ASYM_PRE_PARTIAL_MAX_LOSS_PERCENT", "-1.35"))
ASYM_POST_PARTIAL_PROTECT_ENABLED = os.getenv("ASYM_POST_PARTIAL_PROTECT_ENABLED", "true").lower() == "true"
ASYM_POST_PARTIAL_MIN_PEAK_PERCENT = float(os.getenv("ASYM_POST_PARTIAL_MIN_PEAK_PERCENT", "0.80"))
ASYM_POST_PARTIAL_PROTECT_MIN_FLOOR_PERCENT = float(os.getenv("ASYM_POST_PARTIAL_PROTECT_MIN_FLOOR_PERCENT", "0.12"))
ASYM_POST_PARTIAL_PROTECT_GIVEBACK_PERCENT = max(0.0, float(os.getenv("ASYM_POST_PARTIAL_PROTECT_GIVEBACK_PERCENT", "0.55")))
EXIT_PROFILE_ENABLED = os.getenv("EXIT_PROFILE_ENABLED", "true").lower() == "true"
EXIT_PROFILE_STRONG_MIN_EV_USD = float(os.getenv("EXIT_PROFILE_STRONG_MIN_EV_USD", "0.0065"))
EXIT_PROFILE_STRONG_MIN_CONFIDENCE = max(
    0.0,
    min(1.0, float(os.getenv("EXIT_PROFILE_STRONG_MIN_CONFIDENCE", "0.35"))),
)
EXIT_PROFILE_WEAK_MAX_EV_USD = float(os.getenv("EXIT_PROFILE_WEAK_MAX_EV_USD", "0.0012"))
EXIT_PROFILE_WEAK_MIN_CONFIDENCE = max(
    0.0,
    min(1.0, float(os.getenv("EXIT_PROFILE_WEAK_MIN_CONFIDENCE", "0.25"))),
)
EXIT_PROFILE_CLUSTER_MIN_SAMPLES = max(
    0.0,
    float(os.getenv("EXIT_PROFILE_CLUSTER_MIN_SAMPLES", "10")),
)
EXIT_PROFILE_CLUSTER_WEAK_LOSS_SHARE = max(
    0.0,
    min(1.0, float(os.getenv("EXIT_PROFILE_CLUSTER_WEAK_LOSS_SHARE", "0.68"))),
)
EXIT_PROFILE_STRONG_HOLD_MULT = max(
    0.20,
    float(os.getenv("EXIT_PROFILE_STRONG_HOLD_MULT", "1.18")),
)
EXIT_PROFILE_WEAK_HOLD_MULT = max(
    0.20,
    float(os.getenv("EXIT_PROFILE_WEAK_HOLD_MULT", "0.82")),
)
EXIT_PROFILE_STRONG_NO_MOMENTUM_AGE_SHIFT_SECONDS = int(
    os.getenv("EXIT_PROFILE_STRONG_NO_MOMENTUM_AGE_SHIFT_SECONDS", "28")
)
EXIT_PROFILE_WEAK_NO_MOMENTUM_AGE_SHIFT_SECONDS = int(
    os.getenv("EXIT_PROFILE_WEAK_NO_MOMENTUM_AGE_SHIFT_SECONDS", "-22")
)
EXIT_PROFILE_STRONG_NO_MOMENTUM_PNL_SHIFT = float(
    os.getenv("EXIT_PROFILE_STRONG_NO_MOMENTUM_PNL_SHIFT", "-0.06")
)
EXIT_PROFILE_WEAK_NO_MOMENTUM_PNL_SHIFT = float(
    os.getenv("EXIT_PROFILE_WEAK_NO_MOMENTUM_PNL_SHIFT", "0.14")
)
EXIT_PROFILE_STRONG_PRE_PARTIAL_MAX_LOSS_SHIFT = float(
    os.getenv("EXIT_PROFILE_STRONG_PRE_PARTIAL_MAX_LOSS_SHIFT", "-0.20")
)
EXIT_PROFILE_WEAK_PRE_PARTIAL_MAX_LOSS_SHIFT = float(
    os.getenv("EXIT_PROFILE_WEAK_PRE_PARTIAL_MAX_LOSS_SHIFT", "0.24")
)
EXIT_PROFILE_STRONG_POST_PARTIAL_GIVEBACK_MULT = max(
    0.10,
    float(os.getenv("EXIT_PROFILE_STRONG_POST_PARTIAL_GIVEBACK_MULT", "0.86")),
)
EXIT_PROFILE_WEAK_POST_PARTIAL_GIVEBACK_MULT = max(
    0.10,
    float(os.getenv("EXIT_PROFILE_WEAK_POST_PARTIAL_GIVEBACK_MULT", "1.18")),
)

V2_REGIME_REANCHOR_ENABLED = os.getenv("V2_REGIME_REANCHOR_ENABLED", "true").lower() == "true"
V2_REGIME_REANCHOR_EDGE_REFERENCE_PERCENT = max(
    0.05,
    float(os.getenv("V2_REGIME_REANCHOR_EDGE_REFERENCE_PERCENT", f"{MIN_EXPECTED_EDGE_PERCENT:.4f}")),
)
V2_REGIME_REANCHOR_EDGE_REFERENCE_USD = max(
    0.0005,
    float(os.getenv("V2_REGIME_REANCHOR_EDGE_REFERENCE_USD", f"{MIN_EXPECTED_EDGE_USD:.6f}")),
)
V2_REGIME_REANCHOR_SENSITIVITY = max(0.0, float(os.getenv("V2_REGIME_REANCHOR_SENSITIVITY", "0.85")))
V2_REGIME_REANCHOR_MAX_RELAX = max(0.0, min(0.8, float(os.getenv("V2_REGIME_REANCHOR_MAX_RELAX", "0.45"))))
# Cooldown after a stop-loss close to avoid immediate re-entries on the same token.
AUTO_TRADE_SL_REENTRY_COOLDOWN_SECONDS = max(
    0,
    int(os.getenv("AUTO_TRADE_SL_REENTRY_COOLDOWN_SECONDS", "900")),
)
MAX_LOSS_PER_TRADE_PERCENT_BALANCE = float(os.getenv("MAX_LOSS_PER_TRADE_PERCENT_BALANCE", "1.2"))
PNL_BREAKEVEN_EPSILON_USD = max(0.0, float(os.getenv("PNL_BREAKEVEN_EPSILON_USD", "0.0")))
DAILY_MAX_DRAWDOWN_PERCENT = float(os.getenv("DAILY_MAX_DRAWDOWN_PERCENT", "5.0"))
MAX_CONSECUTIVE_LOSSES = max(1, int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3")))
LOSS_STREAK_COOLDOWN_SECONDS = max(0, int(os.getenv("LOSS_STREAK_COOLDOWN_SECONDS", "1800")))
RISK_GOVERNOR_ENABLED = os.getenv("RISK_GOVERNOR_ENABLED", "true").lower() == "true"
RISK_GOVERNOR_MAX_LOSS_STREAK = max(
    1,
    int(os.getenv("RISK_GOVERNOR_MAX_LOSS_STREAK", str(MAX_CONSECUTIVE_LOSSES))),
)
RISK_GOVERNOR_DRAWDOWN_LIMIT_PERCENT = float(
    os.getenv("RISK_GOVERNOR_DRAWDOWN_LIMIT_PERCENT", str(DAILY_MAX_DRAWDOWN_PERCENT))
)
RISK_GOVERNOR_STREAK_PAUSE_SECONDS = max(
    0,
    int(os.getenv("RISK_GOVERNOR_STREAK_PAUSE_SECONDS", str(LOSS_STREAK_COOLDOWN_SECONDS))),
)
RISK_GOVERNOR_DRAWDOWN_PAUSE_SECONDS = max(
    0,
    int(os.getenv("RISK_GOVERNOR_DRAWDOWN_PAUSE_SECONDS", str(LOSS_STREAK_COOLDOWN_SECONDS))),
)
# If enabled, keep blocking new entries while loss streak >= limit even after pause elapsed.
# Default is false to avoid deadlocks where trading never resumes without manual intervention.
RISK_GOVERNOR_HARD_BLOCK_ON_STREAK = os.getenv("RISK_GOVERNOR_HARD_BLOCK_ON_STREAK", "false").lower() == "true"
RISK_GOVERNOR_LOG_INTERVAL_SECONDS = max(5, int(os.getenv("RISK_GOVERNOR_LOG_INTERVAL_SECONDS", "30")))
RISK_GOVERNOR_ADAPTIVE_PAUSE_ENABLED = os.getenv("RISK_GOVERNOR_ADAPTIVE_PAUSE_ENABLED", "true").lower() == "true"
RISK_GOVERNOR_PAUSE_GREEN_MULT = max(
    0.20,
    min(2.0, float(os.getenv("RISK_GOVERNOR_PAUSE_GREEN_MULT", "0.65"))),
)
RISK_GOVERNOR_PAUSE_YELLOW_MULT = max(
    0.20,
    min(2.0, float(os.getenv("RISK_GOVERNOR_PAUSE_YELLOW_MULT", "0.82"))),
)
RISK_GOVERNOR_PAUSE_RED_MULT = max(
    0.20,
    min(2.0, float(os.getenv("RISK_GOVERNOR_PAUSE_RED_MULT", "1.00"))),
)
RISK_GOVERNOR_PAUSE_MIN_SECONDS = max(0, int(os.getenv("RISK_GOVERNOR_PAUSE_MIN_SECONDS", "60")))
RISK_GOVERNOR_PAUSE_MAX_SECONDS = max(
    RISK_GOVERNOR_PAUSE_MIN_SECONDS,
    int(os.getenv("RISK_GOVERNOR_PAUSE_MAX_SECONDS", "1800")),
)
RISK_GOVERNOR_ANTI_CHOKE_MAX_PAUSE_SECONDS = max(
    RISK_GOVERNOR_PAUSE_MIN_SECONDS,
    int(os.getenv("RISK_GOVERNOR_ANTI_CHOKE_MAX_PAUSE_SECONDS", "300")),
)
RISK_GOVERNOR_DRAWDOWN_MIN_PAUSE_SECONDS = max(
    RISK_GOVERNOR_PAUSE_MIN_SECONDS,
    int(os.getenv("RISK_GOVERNOR_DRAWDOWN_MIN_PAUSE_SECONDS", "240")),
)
SESSION_PROFIT_LOCK_ENABLED = os.getenv("SESSION_PROFIT_LOCK_ENABLED", "false").lower() == "true"
SESSION_PROFIT_LOCK_ACTIVATE_USD = float(os.getenv("SESSION_PROFIT_LOCK_ACTIVATE_USD", "0.35"))
SESSION_PROFIT_LOCK_KEEP_RATIO = max(
    0.0,
    min(1.0, float(os.getenv("SESSION_PROFIT_LOCK_KEEP_RATIO", "0.55"))),
)
SESSION_PROFIT_LOCK_MIN_FLOOR_USD = float(os.getenv("SESSION_PROFIT_LOCK_MIN_FLOOR_USD", "0.10"))
SESSION_PROFIT_LOCK_PAUSE_SECONDS = max(0, int(os.getenv("SESSION_PROFIT_LOCK_PAUSE_SECONDS", "900")))
SESSION_PROFIT_LOCK_USE_EQUITY_GUARD = os.getenv("SESSION_PROFIT_LOCK_USE_EQUITY_GUARD", "true").lower() == "true"
SESSION_PROFIT_LOCK_USE_EQUITY_FOR_PEAK = os.getenv("SESSION_PROFIT_LOCK_USE_EQUITY_FOR_PEAK", "false").lower() == "true"
SESSION_PROFIT_LOCK_MIN_RETRIGGER_SECONDS = max(
    0,
    int(os.getenv("SESSION_PROFIT_LOCK_MIN_RETRIGGER_SECONDS", "300")),
)
SESSION_PROFIT_LOCK_REARM_BUFFER_USD = max(
    0.0,
    float(os.getenv("SESSION_PROFIT_LOCK_REARM_BUFFER_USD", "0.03")),
)
SESSION_PROFIT_LOCK_REARM_COOLDOWN_SECONDS = max(
    0,
    int(os.getenv("SESSION_PROFIT_LOCK_REARM_COOLDOWN_SECONDS", "300")),
)
SESSION_PROFIT_LOCK_RESET_ON_NEW_DAY = os.getenv("SESSION_PROFIT_LOCK_RESET_ON_NEW_DAY", "true").lower() == "true"
PERSIST_TRADING_PAUSE_ON_RESTART = os.getenv("PERSIST_TRADING_PAUSE_ON_RESTART", "true").lower() == "true"
ANTI_CHOKE_ENABLED = os.getenv("ANTI_CHOKE_ENABLED", "true").lower() == "true"
ANTI_CHOKE_IDLE_SECONDS = max(30, int(os.getenv("ANTI_CHOKE_IDLE_SECONDS", "1800")))
ANTI_CHOKE_ALLOW_SYMBOL_COOLDOWN_BYPASS = os.getenv("ANTI_CHOKE_ALLOW_SYMBOL_COOLDOWN_BYPASS", "true").lower() == "true"
ANTI_CHOKE_BYPASS_SYMBOL_COOLDOWN_MAX_SECONDS = max(
    0,
    int(os.getenv("ANTI_CHOKE_BYPASS_SYMBOL_COOLDOWN_MAX_SECONDS", "600")),
)
ANTI_CHOKE_ALLOW_TOKEN_COOLDOWN_BYPASS = os.getenv("ANTI_CHOKE_ALLOW_TOKEN_COOLDOWN_BYPASS", "true").lower() == "true"
ANTI_CHOKE_BYPASS_TOKEN_COOLDOWN_MAX_SECONDS = max(
    0,
    int(os.getenv("ANTI_CHOKE_BYPASS_TOKEN_COOLDOWN_MAX_SECONDS", "180")),
)
ANTI_CHOKE_ALLOW_SYMBOL_FATIGUE_BYPASS = os.getenv("ANTI_CHOKE_ALLOW_SYMBOL_FATIGUE_BYPASS", "true").lower() == "true"
ANTI_CHOKE_ALLOW_SYMBOL_STRICT_BYPASS = os.getenv("ANTI_CHOKE_ALLOW_SYMBOL_STRICT_BYPASS", "true").lower() == "true"

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
