# Base Alert Bot

Telegram-бот для сигналов по токенам сети Base, с локальным desktop-first запуском, GUI-панелью и live/paper авто-трейдингом.

## Быстрый Старт

```powershell
cd d:\earnforme\solana-alert-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Запуск

Локальный раннер:

```powershell
python main_local.py
```

GUI-панель:

```powershell
python launcher_gui.py
```

Важно: запуск/рестарт бота выполняется вручную из GUI (`Start/Restart`).

## Что Добавлено В Refix

- Атомарная инициализация `AutoTrader` (single-flight, состояния `initializing/ready/failed`).
- Graceful shutdown через сигнал с fallback на hard-kill по таймауту.
- Единый HTTP-клиент (`utils/http_client.py`):
  - `ClientTimeout(total=...)`
  - `TCPConnector(limit=...)`
  - retry/backoff/jitter
  - per-source concurrency limits
  - per-source sliding-window rate limits
  - per-source cooldown pause after `429`
  - runtime статистика (`ok/fail/429/retries/latency`).
- Gecko `new_pools` разделен на ingest + processing:
  - ingest-task (редкий, лимитированный) складывает токены в очередь
  - processing-cycle забирает из очереди без агрессивного повторного поллинга.
- Data policy режимы цикла:
  - `OK`
  - `DEGRADED`
  - `FAIL_CLOSED`
- Anti-flap гистерезис:
  - `DATA_POLICY_ENTER_STREAK`
  - `DATA_POLICY_EXIT_STREAK`
- Централизованная нормализация адресов (`utils/addressing.py`).
- Recovery/reconciliation логика в `AutoTrader` после рестартов.
- Fail-closed диагностика с деталями причин (`api_code_*`, `http_*`, `no_token_entry`, и т.д.).
- GUI-лента в компактном виде с цветовой категоризацией событий.
- Отсечение базовых токенов из BUY-кандидатов (`AUTO_TRADE_EXCLUDED_ADDRESSES`).

## Data Policy И Safety

Ключевые параметры:

```env
TOKEN_SAFETY_FAIL_CLOSED=true
DATA_POLICY_DEGRADED_ERROR_PERCENT=...
DATA_POLICY_FAIL_CLOSED_FAIL_CLOSED_RATIO=...
DATA_POLICY_FAIL_CLOSED_API_ERROR_PERCENT=...
DATA_POLICY_ENTER_STREAK=2
DATA_POLICY_EXIT_STREAK=2
```

Рекомендация для real-режима: оставлять `TOKEN_SAFETY_FAIL_CLOSED=true`.

## Showcase Mode ($2)

Базовый безопасный профиль под маленький баланс:

```env
SAFE_MIN_LIQUIDITY_USD=5000
SAFE_MIN_VOLUME_5M_USD=1200
SAFE_MIN_AGE_SECONDS=300
MAX_BUYS_PER_HOUR=1
MIN_TRADE_USD=0.25
MAX_TX_PER_DAY=8
LIVE_SWAP_DEADLINE_SECONDS=45
```

## API Stability Tuning

Параметры HTTP устойчивости:

```env
HTTP_CONNECTOR_LIMIT=...
HTTP_DEFAULT_CONCURRENCY=...
HTTP_RETRY_ATTEMPTS=...
HTTP_BACKOFF_BASE_SECONDS=...
HTTP_BACKOFF_MAX_SECONDS=...
HTTP_JITTER_SECONDS=...
HTTP_RATE_LIMIT_DELAY_SECONDS=...
HTTP_429_COOLDOWN_SECONDS=...
HTTP_SOURCE_RATE_LIMITS=geckoterminal:20/60,...
HTTP_SOURCE_429_COOLDOWNS=geckoterminal:120,...
GECKO_NEW_POOLS_INGEST_INTERVAL_SECONDS=75
GECKO_NEW_POOLS_QUEUE_MAX=400
GECKO_NEW_POOLS_DRAIN_MAX_PER_CYCLE=120
GECKO_INGEST_DEDUP_TTL_SECONDS=3600
HEAVY_CHECK_DEDUP_TTL_SECONDS=900
HEAVY_CHECK_OVERRIDE_LIQ_MULT=2.0
HEAVY_CHECK_OVERRIDE_VOL_MULT=3.0
HEAVY_CHECK_OVERRIDE_VOL_MIN_ABS_USD=500
```

Для снижения gecko-шумa:

```env
GECKO_NEW_POOLS_PAGES=1
WATCHLIST_GECKO_TRENDING_PAGES=1
WATCHLIST_GECKO_POOLS_PAGES=1
```

## Исключение Базовых Токенов

Чтобы не тратить циклы на WETH/стейблы:

```env
AUTO_TRADE_EXCLUDED_ADDRESSES=0x4200000000000000000000000000000000000006,0x833589fcd6edb6e08f4c7c32d4f71b54bda02913,0xfde4c96c8593536e31f229ea8f37b2ada2699bb2,0x50c5725949a6f0c72e6c4a641f24049a917db0cb
```

## On-Chain Signals (Base)

Основные параметры:

```env
SIGNAL_SOURCE=onchain
RPC_PRIMARY=https://...
RPC_SECONDARY=https://...
BASE_FACTORY_ADDRESS=0x...
WETH_ADDRESS=0x...
ONCHAIN_FINALITY_BLOCKS=2
ONCHAIN_SEEN_PAIR_TTL_SECONDS=7200
```

Self-test одного прохода:

```powershell
python -m monitor.onchain_factory --once
```

## Live Execution (Base)

```env
AUTO_TRADE_ENABLED=true
AUTO_TRADE_PAPER=false
LIVE_WALLET_ADDRESS=0x...
LIVE_PRIVATE_KEY=0x...
LIVE_CHAIN_ID=8453
LIVE_ROUTER_ADDRESS=0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24
LIVE_SLIPPAGE_BPS=200
LIVE_SWAP_DEADLINE_SECONDS=45
LIVE_TX_TIMEOUT_SECONDS=180
LIVE_MAX_GAS_GWEI=2.0
LIVE_PRIORITY_FEE_GWEI=0.02
```

Live исполнение в `trading/live_executor.py`.

## GUI

Вкладки:

- `Активность` — компактная лента ключевых событий + сырые runtime-логи.
- `Сигналы` — входящие локальные сигналы.
- `Кошелек` — баланс/режимы.
- `Сделки` — открытые/закрытые позиции.
- `Настройки` — редактирование `.env`, пресеты.
- `Аварийный` — kill-switch, критические события.

Кнопка `Перечитать .env` перечитывает настройки перед `Restart`.

## Диагностика

Если `cand > 0`, но `opened = 0`, смотрите `logs/app.log` и строки:

- `AutoTrade skip ... reason=negative_edge`
- `AutoTrade skip ... reason=blacklist`
- `AUTO_POLICY mode=DEGRADED/FAIL_CLOSED`

Это основной источник правды для настройки фильтров.

## Paper/Log Only

Для безопасного теста без on-chain сделок:

```env
AUTO_TRADE_ENABLED=true
AUTO_TRADE_PAPER=true
```

В этом режиме бот продолжает считать кандидатов/entry-exit в логах, но не отправляет live-транзакции.

## Adaptive Filters (Paper Calibration)

Адаптивный контур меняет пороги в рантайме (в памяти процесса), не переписывая `.env`.

```env
ADAPTIVE_FILTERS_ENABLED=true
ADAPTIVE_FILTERS_MODE=apply
ADAPTIVE_FILTERS_PAPER_ONLY=true
ADAPTIVE_FILTERS_INTERVAL_SECONDS=600
ADAPTIVE_FILTERS_MIN_WINDOW_CYCLES=5
ADAPTIVE_FILTERS_TARGET_CAND_MIN=2.0
ADAPTIVE_FILTERS_TARGET_CAND_MAX=12.0
ADAPTIVE_FILTERS_TARGET_OPEN_MIN=0.10
ADAPTIVE_FILTERS_NEG_REALIZED_TRIGGER_USD=0.60
ADAPTIVE_FILTERS_NEG_CLOSED_MIN=3
ADAPTIVE_SCORE_MIN=60
ADAPTIVE_SCORE_MAX=72
ADAPTIVE_SCORE_STEP=1
ADAPTIVE_SAFE_VOLUME_MIN=150
ADAPTIVE_SAFE_VOLUME_MAX=1200
ADAPTIVE_SAFE_VOLUME_STEP=50
ADAPTIVE_DEDUP_TTL_MIN=60
ADAPTIVE_DEDUP_TTL_MAX=900
ADAPTIVE_DEDUP_TTL_STEP=30
```

Проверка в логах:

- `ADAPTIVE_FILTERS mode=apply action=... score=... volume=... dedup_ttl=...`
- `PAPER_SUMMARY window=300s ...`

## Burst vs Strict Paper

- `Burst` (быстрый набор выборки):
  - ослабленные/адаптивные фильтры
  - можно снять лимиты частоты/tx
  - цель: быстро собрать статистику.
- `Strict paper` (ближе к live):
  - вернуть рабочие risk-лимиты
  - оставить safety/data-policy fail-closed
  - цель: проверить, как профиль ведет себя в условиях, близких к реальному запуску.

Рекомендация:

1. Ночью `Burst` для сбора выборки.
2. Перед live — `Strict paper` 30-90 минут.
3. Только после этого включать live.

## Важные Файлы

- `config.py`
- `main_local.py`
- `launcher_gui.py`
- `monitor/dexscreener.py`
- `monitor/onchain_factory.py`
- `monitor/watchlist.py`
- `monitor/token_checker.py`
- `trading/auto_trader.py`
- `trading/live_executor.py`
- `utils/http_client.py`
- `utils/addressing.py`
