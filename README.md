# Solana Alert Bot

Telegram-бот для алертов по новым Solana токенам, подпискам и оплате через CryptoBot.

## Быстрый старт

```powershell
cd d:\earn\solana-alert-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Заполни `.env`:

```env
TELEGRAM_BOT_TOKEN=...
CRYPTOBOT_TOKEN=...
GOPLUS_ACCESS_TOKEN=... # optional
DATABASE_URL=sqlite:///bot.db
TRIAL_HOURS=6
CRYPTOBOT_WEBHOOK_SECRET=... # обязательно
CARD_PAYMENT_DETAILS=CardNumber:0000 0000 0000 0000; Name:YOUR_NAME; Bank:YOUR_BANK
ADMIN_IDS=123456789
```

## Где взять webhook secret

Webhook secret не выдается сервисом, его создаешь ты сам.

Сгенерировать:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Скопируй результат в `.env` как `CRYPTOBOT_WEBHOOK_SECRET=...`.

## Тарифы

- 1 неделя: `$5`
- 2.5 недели: `$10`
- 1 месяц: `$15`
- Пробный триал: `6 часов`

## Запуск

```powershell
python main.py
```

## Запуск в фоне (PowerShell)

```powershell
.\run_bot_background.ps1
.\bot_status.ps1
.\stop_bot.ps1
```

Логи:
- `logs/app.log`
- `logs/bot.out.log`
- `logs/bot.err.log`

## Мини-окно управления

```powershell
.\.venv\Scripts\python.exe launcher_gui.py
```

## Команды в Telegram

Пользователь:
- `/start`
- `/demo`
- `/subscribe`
- `/status`
- `/settings`
- `/testalert`

Админ:
- `/admin_stats`
- `/admin_grant <telegram_id> <days>`
- `/admin_pending_cards`
- `/admin_approve_card <request_id>`
- `/admin_reject_card <request_id> [reason]`

## Webhook для CryptoBot

Endpoint:

```text
POST http://WEBHOOK_HOST:WEBHOOK_PORT/cryptobot/webhook
```

Параметры из `.env`:
- `WEBHOOK_HOST`
- `WEBHOOK_PORT`
- `CRYPTOBOT_WEBHOOK_PATH`
- `CRYPTOBOT_WEBHOOK_SECRET`

Сервер webhook теперь запускается только при заданном `CRYPTOBOT_WEBHOOK_SECRET`.

## Где что менять

- Конфиг: `config.py`
- Хендлеры Telegram: `bot/handlers.py`
- Мониторинг: `monitor/dexscreener.py`
- Риск-чек: `monitor/token_checker.py`
- Рассылка: `monitor/alerter.py`
- Оплаты: `payments/cryptobot.py`, `payments/webhook_server.py`
- База: `database/models.py`, `database/db.py`
