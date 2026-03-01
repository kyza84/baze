# baze: локальный стек для торговли на Base

`baze` — локальный Python-проект для paper/matrix/live торговли с единым decision pipeline, runtime-тюнером и защитными контурами.

## Что делает проект
- Единая логика решений для `matrix` и `live`.
- Автоподстройка пропускной способности через runtime-тюнер без отключения hard safety.
- Форензика по этапам воронки: `candidate -> precheck -> plan -> execute -> exit`.
- Хранение и восстановление состояния с lock/atomic-паттерном.

## Точки входа
- `main_local.py` — основной торговый рантайм.
- `launcher_gui.py` — GUI для запуска/мониторинга.
- `tools/matrix_paper_launcher.ps1` — запуск matrix-профилей.
- `tools/matrix_paper_stop.ps1` — корректная остановка matrix.
- `tools/matrix_watchdog.py` — контроль живых PID и статусов matrix.

## Ключевые модули
- `trading/auto_trader.py` — управление входами/выходами и позициями.
- `trading/auto_trader_state.py` — загрузка/сохранение состояния торговли.
- `trading/live_executor.py` — live-исполнение.
- `trading/v2_runtime.py` — quality/source/funnel контуры.
- `utils/state_file.py` — файловые lock и атомарная запись state.
- `utils/log_contracts.py` — контракты структурированных логов.

## Runtime-тюнер
- `tools/matrix_runtime_tuner.py` — режимы `once/run/replay`, policy-фазы `expand/hold/tighten`.
- `tools/matrix_runtime_tuner.ps1` — оболочка запуска.
- `tools/matrix_runtime_tuner_open.ps1` — запуск в отдельном видимом окне.

Тюнер:
- читает метрики и воронку по окнам;
- изменяет только разрешённые mutable-ключи;
- использует rollback к стабильному набору при деградации;
- поддерживает `idle-relax` для controlled-расширения в простое;
- не имеет права отключать hard anti-scam и safety guard.

## Логи и данные
- Логи профилей: `logs/matrix/<profile_id>/...`
- Active-run matrix: `data/matrix/runs/active_matrix.json`
- Profile env: `data/matrix/env/<profile>.env`
- User presets: `data/matrix/user_presets/*.json`

## Быстрые команды

Запуск matrix-профиля:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_launcher.ps1 -Run -ProfileIds u_station_ab_night_autotune_v2
```

Остановка matrix:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_stop.ps1 -HardKill
```

Запуск runtime-тюнера (видимое окно):
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_runtime_tuner_open.ps1 -ProfileId u_station_ab_night_autotune_v2 -Mode conveyor
```

Dry-run тюнера:
```powershell
python tools\matrix_runtime_tuner.py once --profile-id u_station_ab_night_autotune_v2 --mode conveyor --dry-run
```

Тесты:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Приватные prompt/context файлы
Файлы локального контекста и экспорта переписок намеренно не должны уходить в Git.

Исключены через `.gitignore`:
- `FULL_CHAT_HISTORY_EXPORT.md`
- `NEXT_CHAT_CONTEXT.md`
- `docs/NEXT_CHAT_CONTEXT.md`
- `.private_ai/`
- `prompts/`
- `chat_exports/`

Они остаются на локальной машине для продолжения работы между сессиями, но не публикуются в репозиторий.

## Документация
- `docs/ARCHITECTURE.md`
- `docs/MATRIX_RUNTIME_TUNER.md`
- `docs/MATRIX_PRESET_MANUAL.md`
- `docs/SAFE_TUNING_AGENT_PROTOCOL.md`