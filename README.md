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
  - runtime статистика (`ok/fail/429/retries/latency`).
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
LIVE_SWAP_DEADLINE_SECONDS=120
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
