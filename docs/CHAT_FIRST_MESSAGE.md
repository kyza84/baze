# CHAT_FIRST_MESSAGE

Скопируй блок ниже целиком в первый запрос нового чата (при переходе между аккаунтами/сессиями).

```text
Ты работаешь в репозитории d:\\earnforme\\solana-alert-bot (remote: https://github.com/kyza84/baze, branch: main).

Режим работы:
- Только evidence-based инженерные действия.
- Ничего не ломать и не трогать торговую стратегию без фактов.
- Перед любыми правками: диагностика 30m/60m (funnel + причины + runtime tuner actions).
- Любое изменение оформлять как: проблема -> решение -> риск -> проверка.

Прочитай в таком порядке:
1) docs/PROJECT_STATE.md
2) docs/OPERATOR_PATTERNS.md
3) docs/ACCOUNT_TRANSFER_PROMPT.md
4) data/matrix/runs/active_matrix.json
5) последние логи профиля logs/matrix/u_station_ab_night_autotune_v2/

Текущий профиль:
- u_station_ab_night_autotune_v2

Первый ответ дай строго в формате:
1) Текущее состояние процессов (matrix/tuner/watchdog/sync)
2) Воронка 30m и 60m (raw->pre->plan->open->close)
3) Топ-5 reason_code по этапам
4) Узкое место (1 primary bottleneck)
5) План 1-2 безопасных шагов без изменения anti-scam hard-guard

Запрещено:
- destructive git
- отключать hard safety / anti-scam
- делать большие переписывания без локальных доказательств

Отдельно:
- Если пользователь пишет "только ответ" или "ничего не меняй" — соблюдать это строго.
- Команды из docs/OPERATOR_PATTERNS.md трактовать как контракт исполнения.
```

## Правило актуальности
- Файл `docs/CHAT_FIRST_MESSAGE.md` должен обновляться в каждом material-коммите вместе с `docs/PROJECT_STATE.md`.
- Проверяется pre-commit guard (`tools/context_commit_guard.py`, `.githooks/pre-commit`).

## Additional Handoff Checklist (Current)
- Confirm non-watch recovery keys are visible in active runtime/preset:
  - `SAFE_AGE_NON_WATCH_SOFT_RATIO`
  - `SAFE_AGE_NON_WATCH_MAX_PASSES_PER_CYCLE`
  - `SAFE_CHANGE_5M_NON_WATCH_SOFT_MULT`
  - `SAFE_CHANGE_5M_NON_WATCH_MAX_PASSES_PER_CYCLE`
- Verify telemetry fields in latest tuner tick:
  - `source_starvation_guard.non_watch_post_filter_starved`
  - `supply_sanity_15m.post_filters_pass_non_watch_15m`
- Keep hard anti-scam unchanged: no disabling of honeypot/hard blocklist paths.

## Commit Update 2026-03-03 (Latest)
- New runtime checks to verify in first diagnostic cut:
  - `decision_meta.prelimit_blocked_actions_count`
  - `decision_meta.non_watch_conversion.active`
  - `decision_meta.prefilter_plan_choke.active`
  - `decision_meta.symbol_loss_pressure_60m.active`
  - `decision_meta.edge_low_loop_15m.active`
- New plan prefilter telemetry in trade decisions:
  - `decision_stage=plan_select`
  - `prefilter_removed_total`
  - `prefilter_removed_by_reason`
  - `prefilter_removed_by_source`
- New auto-trader cost-dominant guard controls:
  - `EDGE_COST_DOMINANT_*`
  - `EDGE_COST_DOMINANT_NON_WATCH_EXPLORE_*`
  - `NON_WATCH_EXPLORE_MIN_TRADE_*`
- Watchdog resilience update:
  - lock file `data/matrix/runs/watchdog.lock.json`
  - stale detection now requires heartbeat + runtime activity age.
