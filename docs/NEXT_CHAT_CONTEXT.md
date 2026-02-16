# NEXT CHAT CONTEXT (Tech Audit)

Дата среза: 2026-02-15 17:41 (локальное время)
Проект: `d:\earnforme\solana-alert-bot`
Цель этого файла: быстро войти в контекст в новом чате и не повторять диагностику заново.

## 1) Краткий итог

Проблема не в одной настройке. Есть системный рассинхрон между:
- фактическим рантаймом процесса,
- метаданными matrix,
- тем, что показывает GUI,
- и тем, что реально лежит в on-chain кошельке.

Из-за этого визуально кажется, что "бот тупит/стоит/не торгует", хотя часть данных пишется, часть процессов уже умерла, а GUI смотрит не туда.

## 2) Что работает в коде (по архитектуре)

- `main_local.py`:
  - основной цикл сканирования и фильтрации,
  - policy-гейтинг (`OK/DEGRADED/FAIL_CLOSED`),
  - market regime,
  - adaptive filters,
  - запись candidate decisions в JSONL.
- `trading/auto_trader.py`:
  - вход/выход (paper/live),
  - risk governor,
  - route/honeypot/roundtrip pre-checks в live,
  - state persistence и recovery.
- `trading/live_executor.py`:
  - swap/approve/quote pipeline для Base.
- `launcher_gui.py`:
  - управление single/matrix,
  - отображение state и tail-логов.

Архитектура в целом рабочая, но есть несколько критичных точек интеграции.

## 3) Критичные проблемы (корень)

### P0-1. Matrix считает, что запущен, хотя процесса уже нет

Факты:
- `data/matrix/runs/active_matrix.json` содержит `running=true` и `pid=15308`.
- В системе `main_local.py` сейчас не запущен (виден только `launcher_gui.py`).

Причина:
- `active_matrix.json` не является источником истины, но GUI/скрипты на него сильно опираются.
- stop/start pipeline не нормализует метаданные после падения/остановки.

Эффект:
- ложный статус "matrix mode",
- ожидание входов при фактически мертвом трейдере.

---

### P0-2. BOM в `mx1_fragile_active.env` ломает `BOT_INSTANCE_ID`

Факт:
- `data/matrix/env/mx1_fragile_active.env` начинается с BOM (`EF BB BF`).
- При тестовом импорте: `config.BOT_INSTANCE_ID == ""`, хотя в файле есть `BOT_INSTANCE_ID=mx1_fragile_active`.

Почему важно:
- lock/mutex идут по `INSTANCE_ID` в `main_local.py`.
- При пустом `BOT_INSTANCE_ID` используется `main_local.lock` и "single-instance" mutex, а не `main_local.mx1_fragile_active.lock`.

Эффект:
- конфликт single/matrix lock-механики,
- ложные старты/псевдозапуски,
- тяжелая диагностика "почему то запускается, то нет".

---

### P0-3. Matrix stop/start pipeline хрупкий к stale PID

Факты:
- `tools/matrix_paper_launcher.ps1` пишет PID сразу после `Process::Start`, без проверки, что процесс реально вошел в рабочий цикл.
- `tools/matrix_paper_stop.ps1` не переписывает `active_matrix.json` в "остановлено".

Эффект:
- stale PID остаются как "живые" метаданные,
- GUI показывает режимы не по факту.

---

### P0-4. GUI в matrix режиме частично читает не те источники

Факты:
- `read_env_map()` в GUI читает только `.env`.
- В matrix рантайм реально идет из `BOT_ENV_FILE` (`data/matrix/env/*.env`).
- `Сигналы` таб читает `logs/local_alerts.jsonl` (single), а не matrix `logs/matrix/<id>/local_alerts.jsonl`.

Эффект:
- "лента пустая", хотя matrix пишет данные,
- часть параметров/режимов в UI визуально не совпадает с рантаймом.

## 4) Почему в кошельке есть токены, а в "Открытых позициях" пусто

Это не фантазия UI, а текущая логика recovery:

- `AutoTrader` ведет только `open_positions` из state.
- On-chain токены, найденные без соответствующей open-позиции, пишутся в `recovery_untracked`.
- Сейчас в `trading/paper_state.mx1_fragile_active.json`:
  - `open_positions = 0`
  - `recovery_untracked = 5` адресов.

Дополнительно:
- включен `LIVE_ABANDON_UNSELLABLE_POSITIONS=true`.
- При проблемном SELL позиция может быть закрыта локально (ABANDON), но токен остается в кошельке.

То есть "токен в кошельке, но не в сделках" сейчас возможно по дизайну recovery + abandon-логики.

## 5) Почему входы резко были и потом пропали

Не одна причина, а сумма:

1. Жесткий live pre-check pipeline:
  - `unsupported_buy_route`,
  - `roundtrip_quote_failed`,
  - `roundtrip_ratio:0.000`,
  - `live_buy_zero_amount`.
2. Эти токены попадают в `autotrade_blacklist` и перестают рассматриваться.
3. При узком текущем market universe это быстро "выжигает" доступные входы.
4. Если процесс вообще уже остановился, входов физически не будет.

В blacklist на срезе есть множество именно таких причин.

## 6) Риск-контур сейчас (по факту профиля `mx1_fragile_active`)

Профиль сильно расслаблен относительно anti-scam:
- `SAFE_REQUIRE_CONTRACT_SAFE=false`
- `SAFE_REQUIRE_RISK_LEVEL=HIGH` (фактически пропуск и LOW/MEDIUM/HIGH)
- `SAFE_MAX_WARNING_FLAGS=2`
- `SAFE_MIN_VOLUME_5M_USD=30`
- `SAFE_MIN_LIQUIDITY_USD=4000`

Это дает больше кандидатов, но повышает вероятность мусорных/проблемных токенов в live.

## 7) Текущее состояние запуска на момент среза

- Процесс:
  - `launcher_gui.py` запущен
  - `main_local.py` не запущен
- Matrix meta:
  - `active_matrix.json` считает, что `running=true` и есть PID
- State (`mx1_fragile_active`):
  - `open=0`, `closed=8`, `realized=0`
  - `recovery_untracked=5`

## 8) Что чинить в первую очередь (порядок)

### Фаза A (обязательная стабилизация, без тюнинга стратегии)

1. Починить env encoding pipeline:
   - гарантировать `UTF-8 without BOM` для `data/matrix/env/*.env`.
2. Нормализовать lifecycle matrix:
   - после stop/update корректно писать `active_matrix.json` (running/pid).
   - после start делать health-check процесса (жив + commandline содержит `main_local.py`).
3. В GUI проверять PID не только "процесс существует", но и что это именно `python ... main_local.py`.
4. В GUI matrix-сигналы читать из выбранного matrix instance, а не только single `logs/local_alerts.jsonl`.

### Фаза B (прозрачность состояния)

1. Явно показывать в GUI:
   - `open_positions`,
   - `recovery_untracked`,
   - `abandoned`,
   - `policy mode`,
   - `cannot_open_trade detail`.
2. Добавить явный health-бейдж:
   - `Trader process: alive/dead`,
   - `Data stream: active/stale`.

### Фаза C (только после A+B — повторная калибровка торговли)

1. Отдельно калибровать live pre-check strictness (roundtrip/sellability), а не дергать все фильтры сразу.
2. Развести профили:
   - strict-safe,
   - throughput-safe.
3. Мерить причину пропуска входа по top-N reason distribution, а не по ощущениям.

## 9) Быстрые команды для валидации в новом чате

Проверка процессов:
```powershell
Get-CimInstance Win32_Process | ? { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'main_local.py|launcher_gui.py' } | select ProcessId,CommandLine
```

Проверка BOM env:
```powershell
$b=[System.IO.File]::ReadAllBytes('data/matrix/env/mx1_fragile_active.env'); $b[0..2]
```

Проверка state:
```powershell
$j=Get-Content -Raw trading/paper_state.mx1_fragile_active.json | ConvertFrom-Json
@($j.open_positions).Count
@($j.recovery_untracked.PSObject.Properties).Count
```

Проверка последних candidate событий:
```powershell
Get-Content -Tail 120 logs/matrix/mx1_fragile_active/candidates.jsonl
```

## 10) Важное для нового чата

- Сначала фиксируем инфраструктурную правду состояния (A+B), потом снова трогаем торговые параметры.
- Без этого любые "подкрутки стратегии" будут давать ложную картину.
- Текущий главный баг-узел: env/lock/lifecycle/GUI-source рассинхрон, а не "рынок полностью мертв".
