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
python main.py
```

GUI-панель (Activity/Trades/Settings):

```powershell
python launcher_gui.py
```

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
- `PAPER_GAS_PER_TX_USD=0.03`
- `PAPER_SWAP_FEE_BPS=30`
- `PAPER_BASE_SLIPPAGE_BPS=80`
- `DYNAMIC_POSITION_SIZING_ENABLED=true`
- `EDGE_FILTER_ENABLED=true`
- `MIN_EXPECTED_EDGE_PERCENT=2.0`
- `CLOSED_TRADES_MAX_AGE_DAYS=14`
- `DEX_SEARCH_QUERIES=base,new`
- `GECKO_NEW_POOLS_PAGES=2`

В paper режиме бот открывает и закрывает сделки автоматически:
- BUY по сигналу (score/recommendation)
- SELL по `TP`, `SL` или `TIMEOUT`
- время удержания может подбираться автоматически по качеству/риску токена
- PnL учитывает комиссию, проскальзывание и gas (если `PAPER_REALISM_ENABLED=true`)
- размер позиции может меняться от ожидаемого edge (`PAPER_TRADE_SIZE_MIN_USD` .. `PAPER_TRADE_SIZE_MAX_USD`)
- состояния сделок сохраняются в `trading/paper_state.json` и восстанавливаются после перезапуска
- старые закрытые сделки автоматически чистятся по `CLOSED_TRADES_MAX_AGE_DAYS`
- статистика и PnL доступны в `/mystats`

## Важные файлы

- Конфиг: `config.py`
- Мониторинг Base: `monitor/dexscreener.py`
- Риск-чек: `monitor/token_checker.py`
- Скоринг: `monitor/token_scorer.py`
- Рассылка: `monitor/alerter.py`
- Автотрейд (подготовка): `trading/auto_trader.py`
