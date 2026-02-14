# Base Alert Bot

Production-style desktop control + local engine for Base meme/alt token monitoring, paper calibration and live execution.

Проект уже включает:
- on-chain + market ingestion,
- multi-layer safety,
- paper/live auto-trading,
- matrix A/B testing,
- adaptive runtime calibration,
- GUI control center.

---

## 1) Что это за проект

`solana-alert-bot` (историческое имя папки) — это локальный бот под сеть Base с фокусом на:

1. Защищенный вход в сделки (safety + route + policy).
2. Быструю итерацию пресетов через paper/matrix.
3. Контролируемый переход в live.
4. Ручное управление запуском из GUI (без автозапуска процесса).

Ключевая идея: сначала стабильность и риск-контур, потом агрессия.

---

## 2) Архитектура (верхнеуровнево)

- `main_local.py`
  - основной цикл сканирования/фильтрации/трейдинга,
  - data-policy (OK / DEGRADED / FAIL_CLOSED),
  - graceful stop,
  - adaptive filters.

- `monitor/*`
  - источники данных (Dex/Gecko/on-chain/watchlist),
  - подготовка сырья под фильтрацию и скоринг.

- `trading/auto_trader.py`
  - paper/live решение по входу/выходу,
  - risk caps, limits, cooldowns,
  - state persistence/recovery,
  - adaptive runtime knobs.

- `trading/live_executor.py`
  - live swap/tx pipeline,
  - preflight checks,
  - gas/fee guards,
  - route safety.

- `utils/http_client.py`
  - unified async HTTP layer,
  - per-source limits/cooldowns/retries/backoff/jitter,
  - runtime metrics.

- `launcher_gui.py`
  - управление single/matrix режимом,
  - просмотр позиций,
  - env controls,
  - runtime monitoring.

---

## 3) Safety и fail-closed контур

Перед BUY используются:
- route-check,
- honeypot-guard,
- gas reserve guard,
- gas cap,
- fail-closed policy при деградации safety API.

Data policy режимы цикла:
- `OK` — торгуем.
- `DEGRADED` — наблюдаем/блок BUY (по условиям).
- `FAIL_CLOSED` — BUY pause до восстановления источников.

SELL в paper/live не должен блокироваться тем же образом, чтобы не застревать в позиции.

---

## 4) Запуск

## 4.1 Подготовка

```powershell
cd d:\earnforme\solana-alert-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## 4.2 Локальный запуск

```powershell
python main_local.py
```

## 4.3 GUI

```powershell
python launcher_gui.py
```

Важно: запуск/рестарт торгового процесса выполняется вручную через GUI (`Start/Restart`).

---

## 5) Режимы торговли

- Paper:
  - `AUTO_TRADE_ENABLED=true`
  - `AUTO_TRADE_PAPER=true`

- Live:
  - `AUTO_TRADE_ENABLED=true`
  - `AUTO_TRADE_PAPER=false`
  - корректно заполненные `LIVE_*` ключи.

Рекомендуемый порядок:
1. Paper strict.
2. Matrix paper A/B.
3. Короткий strict validation.
4. Только затем live.

---

## 6) Matrix режим (1-4 инстанса)

Нужен для быстрого сравнения пресетов в одинаковом рыночном окне.

Каждый инстанс имеет собственные файлы:
- state,
- candidates log,
- blacklist,
- on-chain cursor,
- snapshots,
- run tag.

Также включен deterministic sharding кандидатов:
- `CANDIDATE_SHARD_MOD`
- `CANDIDATE_SHARD_SLOT`

Это снижает overlap между инстансами.

## 6.1 Команды

Запустить матрицу:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_launcher.ps1 -Count 1 -Run
```

Остановить:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_stop.ps1 -HardKill
```

Сводка:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_summary.ps1
```

## 6.2 Через GUI

- `Matrix Start`
- `Matrix Stop`
- `Matrix Сводка`
- вкладка `Сделки` -> `Источник` (`single`, `mx1_*`, `mx2_*`, ...)
- в `Сырые runtime-логи` теперь подтягиваются не только `app/session` логи, но и tail:
  - `logs/matrix/<instance>/candidates.jsonl`
  - `logs/matrix/<instance>/local_alerts.jsonl`

---

## 7) Как читать результаты

Минимальный набор метрик за сессию:
- closed trades,
- winrate,
- realized pnl (usd),
- median pnl per trade,
- close reasons distribution (`TP/SL/TIMEOUT/NO_MOMENTUM/WEAKNESS`),
- top skip reasons.

Если часто `NO_MOMENTUM + TIMEOUT`:
- ускоряйте exits,
- сдвигайте hold-window вниз,
- проверяйте edge threshold.

Если много `SL`:
- вход слишком шумный,
- поднимайте quality constraints (score/volume/edge).

---

## 8) Adaptive filters

Адаптер работает в рантайме (не переписывает `.env`) и двигает ограниченный набор порогов:
- `MIN_TOKEN_SCORE`
- `SAFE_MIN_VOLUME_5M_USD`
- `HEAVY_CHECK_DEDUP_TTL_SECONDS`
- (и связанные выходные параметры, если включено в логике)

Важные принципы:
- hysteresis,
- cooldown после изменения,
- узкие step sizes,
- floors/ceilings.

Это защищает от флаппинга и переоптимизации.

Дополнительно поддерживается dynamic dedup-режим:
- в окне адаптации собираются интервалы повторного появления токенов,
- считается перцентиль и строится динамический target TTL,
- итоговый `HEAVY_CHECK_DEDUP_TTL_SECONDS` зажимается в runtime-коридор вокруг target.

---

## 9) Ключевые env блоки

## 9.1 Risk / throughput

```env
MAX_OPEN_TRADES=...
MAX_BUYS_PER_HOUR=...
MAX_TX_PER_DAY=...
MIN_TRADE_USD=...
MIN_EXPECTED_EDGE_PERCENT=...
```

## 9.2 Exit behavior

```env
PAPER_MAX_HOLD_SECONDS=...
DYNAMIC_HOLD_ENABLED=true|false
HOLD_MIN_SECONDS=...
HOLD_MAX_SECONDS=...
PAPER_PARTIAL_TP_ENABLED=true|false
PAPER_PARTIAL_TP_TRIGGER_PERCENT=...
PAPER_PARTIAL_TP_SELL_FRACTION=...
PAPER_PARTIAL_TP_MOVE_SL_TO_BREAK_EVEN=true|false
PAPER_PARTIAL_TP_BREAK_EVEN_BUFFER_PERCENT=...
NO_MOMENTUM_EXIT_MIN_AGE_PERCENT=...
NO_MOMENTUM_EXIT_MAX_PNL_PERCENT=...
WEAKNESS_EXIT_MIN_AGE_PERCENT=...
WEAKNESS_EXIT_PNL_PERCENT=...
PROFIT_LOCK_TRIGGER_PERCENT=...
PROFIT_LOCK_FLOOR_PERCENT=...
```

## 9.3 Safety / policy

```env
TOKEN_SAFETY_FAIL_CLOSED=true
DATA_POLICY_ENTER_STREAK=...
DATA_POLICY_EXIT_STREAK=...
DATA_POLICY_FAIL_CLOSED_*...
```

## 9.4 HTTP stability

```env
HTTP_CONNECTOR_LIMIT=...
HTTP_RETRY_ATTEMPTS=...
HTTP_BACKOFF_BASE_SECONDS=...
HTTP_BACKOFF_MAX_SECONDS=...
HTTP_JITTER_SECONDS=...
HTTP_SOURCE_RATE_LIMITS=...
HTTP_SOURCE_429_COOLDOWNS=...
```

## 9.5 Adaptive dynamic dedup

```env
ADAPTIVE_DEDUP_DYNAMIC_ENABLED=true|false
ADAPTIVE_DEDUP_DYNAMIC_MIN=...
ADAPTIVE_DEDUP_DYNAMIC_MAX=...
ADAPTIVE_DEDUP_DYNAMIC_TARGET_PERCENTILE=...
ADAPTIVE_DEDUP_DYNAMIC_FACTOR=...
ADAPTIVE_DEDUP_DYNAMIC_MIN_SAMPLES=...
```

---

## 10) Типовые runbooks

## 10.1 Перед ночным прогоном (paper)

1. Остановить все инстансы.
2. Сбросить paper-state на одинаковый bankroll (например $7/$7).
3. Проверить, что нет open positions.
4. Запустить matrix на 2 профиля.
5. Через 60-120 минут снять срез.

## 10.2 Перед live

1. Берем профиль с лучшим risk-adjusted результатом (не только winrate).
2. Проверяем стабильность safety/policy за последний час.
3. Проверяем отсутствие регулярных FAIL_CLOSED флапов.
4. Подтверждаем лимиты риска под текущий баланс.
5. Только потом включаем live.

---

## 11) Логи и данные

Основные директории:
- `logs/`
- `data/matrix/reports/`
- `data/matrix/backups/`
- `data/matrix/presets/`
- `trading/paper_state*.json`

Рекомендация:
- каждый stop-срез сохранять в `reports + backups`,
- фиксировать, какой пресет и какой тайм-диапазон анализировался.

---

## 12) Ограничения и реализм

- Малый банк сильно ограничивает абсолютный PnL.
- Высокий throughput без edge быстро съедается комиссиями и шумом.
- Хороший score/safety не гарантирует рост (это фильтр качества, не alpha).
- Переоптимизация на коротком окне ломает переносимость в live.

Проект должен развиваться через системные сравнения, а не через хаотичные ручные дергания.

---

## 13) Файлы, которые чаще всего трогаются

- `README.md`
- `launcher_gui.py`
- `main_local.py`
- `config.py`
- `trading/auto_trader.py`
- `trading/live_executor.py`
- `tools/matrix_paper_launcher.ps1`
- `tools/matrix_paper_stop.ps1`
- `tools/matrix_paper_summary.ps1`

---

## 14) Примечание по управлению

- Бот не должен стартовать автоматически без явной команды пользователя.
- Любые live изменения должны сначала пройти paper/matrix валидацию.
- Все решения по пресетам фиксировать через срезы (report + backup), чтобы не терять контекст.
