# Base Alert Bot

Telegram-бот для алертов по новым токенам в сети Base, с подписками и оплатой через CryptoBot.

## Быстрый старт

```powershell
cd d:\earnforme\solana-alert-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Основные команды

Пользователь:
- `/start`
- `/demo`
- `/subscribe`
- `/status`
- `/settings`
- `/testalert`
- `/mystats`
- `/setscore <0-100>`
- `/setamount <ETH>`
- `/togglefilter`

Админ:
- `/admin_stats`
- `/admin_grant <telegram_id> <days>`
- `/admin_pending_cards`
- `/admin_approve_card <request_id>`
- `/admin_reject_card <request_id> [reason]`

## Запуск

```powershell
python main_local.py
```

Этот проект сейчас ориентирован на desktop-first режим (без Telegram в рабочем контуре).

GUI-панель (Activity/Trades/Settings):

```powershell
python launcher_gui.py
```

В GUI есть вкладка `Сигналы` с входящими локальными алертами по новым токенам.
Также есть вкладка `Аварийный`:
- отдельная лента критических событий (`CRITICAL_AUTO_RESET`, `KILL_SWITCH`, `AUTO_SELL live_failed`, `[ERROR]`)
- кнопки `Включить/Выключить KILL SWITCH`
- `Критический сброс`, `Signal Check`, `Очистить логи`

## Paper trading

- `AUTO_TRADE_ENABLED=true`
- `AUTO_TRADE_PAPER=true`
- `AUTO_TRADE_ENTRY_MODE=single|all|top_n`
- `AUTO_TRADE_TOP_N=10`
- `MAX_OPEN_TRADES=0` (unlimited) or any limit
- `PAPER_TRADE_SIZE_USD=1.0`
- `PAPER_TRADE_SIZE_MIN_USD=0.25`
- `PAPER_TRADE_SIZE_MAX_USD=1.0`
- `PAPER_MAX_HOLD_SECONDS=1800`
- `DYNAMIC_HOLD_ENABLED=true`
- `HOLD_MIN_SECONDS=300`
- `HOLD_MAX_SECONDS=1800`
- `PAPER_REALISM_ENABLED=true`
- `PAPER_REALISM_CAP_ENABLED=true`
- `PAPER_REALISM_MAX_GAIN_PERCENT=600`
- `PAPER_REALISM_MAX_LOSS_PERCENT=95`
- `PAPER_GAS_PER_TX_USD=0.03`
- `PAPER_SWAP_FEE_BPS=30`
- `PAPER_BASE_SLIPPAGE_BPS=80`
- `DYNAMIC_POSITION_SIZING_ENABLED=true`
- `EDGE_FILTER_ENABLED=true`
- `MIN_EXPECTED_EDGE_PERCENT=2.0`
- `CLOSED_TRADES_MAX_AGE_DAYS=14`
- `PAPER_RESET_ON_START=false`
- `WETH_PRICE_FALLBACK_USD=3000`
- `STAIR_STEP_ENABLED=false`
- `STAIR_STEP_START_BALANCE_USD=2.75`
- `STAIR_STEP_SIZE_USD=5`
- `DEX_SEARCH_QUERIES=base,new`
- `GECKO_NEW_POOLS_PAGES=2`

В paper режиме бот открывает и закрывает сделки автоматически:
- BUY по сигналу (score/recommendation)
- SELL по `TP`, `SL` или `TIMEOUT`
- время удержания может подбираться автоматически по качеству/риску токена
- PnL учитывает комиссию, проскальзывание и gas (если `PAPER_REALISM_ENABLED=true`)
- если включен `PAPER_REALISM_CAP_ENABLED`, экстремальные бумажные пампы/дампы ограничиваются (`PAPER_REALISM_MAX_GAIN_PERCENT`/`PAPER_REALISM_MAX_LOSS_PERCENT`)
- размер позиции может меняться от ожидаемого edge (`PAPER_TRADE_SIZE_MIN_USD` .. `PAPER_TRADE_SIZE_MAX_USD`)
- если включен `STAIR_STEP_ENABLED`, часть баланса блокируется как floor и не тратится в новых входах; floor поднимается шагами `STAIR_STEP_SIZE_USD` по мере роста баланса
- состояния сделок сохраняются в `trading/paper_state.json` и восстанавливаются после перезапуска
- старые закрытые сделки автоматически чистятся по `CLOSED_TRADES_MAX_AGE_DAYS`
- статистика и PnL доступны в `/mystats`

В `launcher_gui.py` на вкладке `Кошелек` есть переключатель `Step protection` (`false/true`) с кнопкой `Apply step`, чтобы включать/выключать ступеньки без ручной правки `.env`.

## Аварийная защита

В авто-трейдер добавлены аварийные условия:
- при наличии `KILL_SWITCH_FILE` бот блокирует новые входы и пытается закрыть открытые позиции
- при падении доступного баланса ниже `AUTO_STOP_MIN_AVAILABLE_USD` (и отсутствии открытых позиций) срабатывает аварийный halt (`LOW_BALANCE_STOP`)
- при повторных сбоях live-продажи срабатывает аварийный halt (`LIVE_SELL_FAILED`)

Ключевые строки в логах:
- `CRITICAL_AUTO_RESET`
- `KILL_SWITCH`
- `AUTO_SELL live_failed`
- `AUTO_SELL forced_failed`

Путь kill-switch по умолчанию:

```env
KILL_SWITCH_FILE=data/kill.txt
```

## Важные файлы

- Конфиг: `config.py`
- Мониторинг Base: `monitor/dexscreener.py`
- On-chain мониторинг factory: `monitor/onchain_factory.py`
- Риск-чек: `monitor/token_checker.py`
- Скоринг: `monitor/token_scorer.py`
- Рассылка: `monitor/alerter.py`
- Автотрейд (подготовка): `trading/auto_trader.py`

## On-chain signals (Base)

Можно переключить источник сигналов на on-chain PairCreated:

```env
RPC_PRIMARY=https://...
RPC_SECONDARY=https://...
BASE_FACTORY_ADDRESS=0x...
PAIR_CREATED_TOPIC=0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9
ONCHAIN_ENABLE_UNISWAP_V3=true
UNISWAP_V3_FACTORY_ADDRESS=0x33128a8fC17869897dcE68Ed026d694621f6FDfD
UNISWAP_V3_POOL_CREATED_TOPIC=0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118
WETH_ADDRESS=0x...
SIGNAL_SOURCE=onchain
ONCHAIN_FINALITY_BLOCKS=2
ONCHAIN_SEEN_PAIR_TTL_SECONDS=7200
AUTO_TRADE_ENABLED=true
AUTO_TRADE_PAPER=true
```

Быстрый self-test одного прохода:

```powershell
python -m monitor.onchain_factory --once
```

Важно:
- На бесплатных/public RPC возможны rate-limit, пропуски логов и задержки.
- При 3 RPC-ошибках подряд `main_local.py` временно переключается на `dexscreener` на 60 секунд, затем пробует вернуть `onchain`.
- On-chain обработка идет с lag `latest-2` (`ONCHAIN_FINALITY_BLOCKS`) и дедупом пар (`ONCHAIN_SEEN_PAIRS_FILE` + TTL), чтобы снизить дубли/reorg-шум.
- Для аварийной остановки авто-входов можно создать файл `data/kill.txt` (`KILL_SWITCH_FILE`).

## Live execution (Base)

Для реальных on-chain сделок:

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

Режим `AUTO_TRADE_PAPER=false` использует `trading/live_executor.py`:
- BUY: `swapExactETHForTokensSupportingFeeOnTransferTokens`
- SELL: `swapExactTokensForETHSupportingFeeOnTransferTokens`
- Перед SELL делается `approve` при необходимости.

Для первого запуска используйте минимальные суммы и `MAX_OPEN_TRADES=1`.
