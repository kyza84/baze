# LOGGING_REFERENCE

Единый справочник по логам и runtime-состоянию для matrix/tuner/watchdog.

## 1) Где лежат данные

- Профильные логи matrix:
  - `logs/matrix/<profile_id>/`
- Активный запуск matrix:
  - `data/matrix/runs/active_matrix.json`
- События watchdog:
  - `data/matrix/runs/watchdog_events.jsonl`
  - `data/matrix/runs/watchdog_state.json`
  - `data/matrix/runs/watchdog.lock.json`
- Профильный env snapshot:
  - `data/matrix/env/<profile_id>.env`

## 2) Файлы логов профиля

- `app.log`, `out.log`, `sessions/main_local_*.log`
  - Тип: текст.
  - Источник: `main_local.py`, `trading/*`.
  - Назначение: оперативная диагностика, traceback, старт/стоп цикла.

- `candidates.jsonl`
  - Тип: JSONL (`candidate_decision.v1`).
  - Источник: decision pipeline (`filter_fail`, `post_filters`, `quality_gate`, `plan_trade`).
  - Назначение: воронка кандидатов и причины отсева.

- `trade_decisions.jsonl`
  - Тип: JSONL (`trade_decision.v1`).
  - Источник: `trading/auto_trader.py`.
  - Назначение: план/открытие/частичное/закрытие, PnL/edge/cost forensics.

- `local_alerts.jsonl`
  - Тип: JSONL (`local_alert.v1`).
  - Источник: `monitor/local_alerter.py`.
  - Назначение: сигналы состояния/контуров (`V2_*`, market mode, safety).

- `runtime_tuner.jsonl`
  - Тип: JSONL (tick snapshot).
  - Источник: `tools/matrix_runtime_tuner.py`.
  - Назначение: действия тюнера, фаза policy, telemetry v2, blocked actions, rollback.

- `runtime_tuner_state.json`
  - Тип: JSON.
  - Источник: `tools/matrix_runtime_tuner.py`.
  - Назначение: персистентное состояние тюнера (stable snapshot, counters, locks).

- `runtime_tuner_runtime_overrides.json`
  - Тип: JSON.
  - Источник: `tools/matrix_runtime_tuner.py`.
  - Назначение: hot-apply patch для running `main_local.py`.

- `runtime_tuner.lock.json`
  - Тип: JSON.
  - Источник: `tools/matrix_runtime_tuner.py`.
  - Назначение: single-instance lock (защита от второго тюнера).

- `heartbeat.json`
  - Тип: JSON.
  - Источник: runtime.
  - Назначение: freshness/liveness для watchdog.

- `<profile>_orchestrator_decisions.jsonl`
  - Тип: JSONL.
  - Источник: orchestration layer.
  - Назначение: решения оркестратора/ограничителей профиля.

## 3) Общие поля JSONL (контракты)

Контракт stamping: `utils/log_contracts.py`.

Стандартные поля:
- `schema_version`
- `schema_name`
- `event_type`
- `ts` (unix, UTC)
- `timestamp` (ISO, UTC)
- `run_tag`
- `trace_id`
- `decision_id`
- `parent_decision_id`
- `position_id`
- `reason_code`
- `reason_severity`
- `reason_category`

Причины нормализуются в taxonomy (`REASON_CODE_TAXONOMY`) и stage-prefix mapping.

## 4) Ключевые поля по потокам

### 4.1 `candidates.jsonl`

Минимум:
- `decision_stage` (`filter_fail`, `post_filters`, `quality_gate`, `plan_trade`, ...)
- `decision`
- `reason`
- `reason_code`
- `source`
- `symbol`
- `score`
- `candidate_id`

Для срезов воронки:
- `decision_stage`
- `reason_code`
- `source`

### 4.2 `trade_decisions.jsonl`

Минимум:
- `decision_stage` (`plan_trade`, `trade_open`, `trade_partial`, `trade_close`, `plan_select`)
- `type` (`open`, `partial`, `close`)
- `reason`, `reason_code`
- `symbol`, `source`, `entry_channel`, `entry_tier`
- `position_size_usd`
- `expected_edge_percent`

Forensics (если есть в событии):
- `gross_percent`
- `cost_total_percent`
- `cost_buy_percent`
- `cost_sell_percent`
- `cost_gas_usd`
- `ev_expected_net_usd`

### 4.3 `runtime_tuner.jsonl`

Основные блоки tick:
- `mode`, `policy_phase`
- `metrics` (window counters)
- `actions` (применённые/планируемые ключи)
- `telemetry_v2`:
  - `funnel_15m`
  - `top_reasons_15m`
  - `exec_health_15m`
  - `source_profile_15m`
  - `plan_symbol_concentration_15m`
  - `supply_sanity_15m`
  - `silence_diagnostics_15m`
  - `ev_forensics_15m`
  - `exit_mix_60m`
- `decision_meta`
- `blocked_actions`
- `delta_capped_actions`
- `restart_guard`
- `runtime_patch_sync_ok`

## 5) Что смотреть первым при анализе

1. `data/matrix/runs/active_matrix.json`
   - `running`, `alive_count`, `items[].pid`, `items[].status`, `items[].id`
2. `logs/matrix/<profile>/runtime_tuner.jsonl` (последний tick)
   - `telemetry_v2.silence_diagnostics_15m`
   - `telemetry_v2.funnel_15m`
   - `telemetry_v2.top_reasons_15m`
   - `decision_meta`
3. `logs/matrix/<profile>/trade_decisions.jsonl` (30m/60m окно)
4. `logs/matrix/<profile>/candidates.jsonl` (30m/60m окно)

## 6) Минимальный набор для внешнего анализатора

Если даёте доступ второму чату/анализатору, достаточно:
- `data/matrix/runs/active_matrix.json`
- `logs/matrix/<profile>/candidates.jsonl`
- `logs/matrix/<profile>/trade_decisions.jsonl`
- `logs/matrix/<profile>/runtime_tuner.jsonl`
- `logs/matrix/<profile>/local_alerts.jsonl`
- (опционально) `sessions/main_local_*.log`, `app.log`, `out.log`

## 7) Важные замечания

- `trace_id/decision_id/position_id` используются для сквозной цепочки candidate -> decision -> open -> close.
- Текстовые логи и JSONL могут отличаться по детализации; источник истины для метрик — JSONL.
- `runtime_tuner.lock.json` и `watchdog.lock.json` должны иметь одного актуального владельца процесса; второй экземпляр считается конфликтом.

## 8) V5 диагностика (2026-03-12)

- `main_local` session text log:
  - `STARTUP_CONFIG_SYNC ...`
    - runtime/startup alignment markers for critical overrides.
  - `FLOW_LANE_METRICS ...`
    - lane/source funnel snapshot per cycle.
- `runtime_tuner.jsonl`:
  - `telemetry_v2.lane_profile_15m`
    - lane-level funnel and open/close profile for `stable`/`discovery`.
  - `blocked_actions[].blocked_by`
    - includes structural lock reasons (`structural_lock`, `dry_run_structural_lock`).
- `candidates.jsonl` / `trade_decisions.jsonl`:
  - `lane_tag` propagated through candidate and trade events.
  - use with `source` to debug conversion gaps (`non-watch raw -> post -> plan`).
