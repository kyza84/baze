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
3) docs/LOGGING_REFERENCE.md
4) docs/ACCOUNT_TRANSFER_PROMPT.md
5) data/matrix/runs/active_matrix.json
6) последние логи профиля logs/matrix/u_station_ab_night_autotune_v2/

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

## Commit Update 2026-03-06 (Phase C)
- Runtime tuner phase/filter alignment:
  - `hold` phase now allows lane-recovery actions for `non_watch_conversion_guard` and `prefilter_plan_choke`.
- Pre-risk roundtrip gating:
  - new target policy key `pre_risk_roundtrip_min_closes_15m`.
  - roundtrip-only pre-risk no longer triggers early degrade rollback while non-watch conversion is improving.
- New policy telemetry flags to verify:
  - `policy_snapshot.pre_risk_roundtrip_only`
  - `policy_snapshot.pre_risk_roundtrip_guard_active`
  - `decision_meta.policy.pre_risk_roundtrip_only`
  - `decision_meta.policy.pre_risk_roundtrip_guard_active`

## Commit Update 2026-03-06 (Post-Entry Anti-Rug)
- New dynamic post-entry anti-rug controls:
  - `POST_ENTRY_RUG_GUARD_ENABLED`
  - `POST_ENTRY_RUG_MIN_ENTRY_LIQUIDITY_USD`
  - `POST_ENTRY_RUG_MIN_CURRENT_LIQUIDITY_USD`
  - `POST_ENTRY_RUG_MAX_LIQUIDITY_RATIO`
  - `POST_ENTRY_RUG_HITS_TO_TRIGGER`
- New close reason to monitor in `trade_decisions.jsonl`:
  - `reason=RUG_GUARD` / `reason_code=EXIT_RUG_GUARD`
- In paper hard-only blacklist mode, `rug_guard:*` is treated as hard block.

## Commit Update 2026-03-06 (Anti-Scam hardening + tuner lock)
- New pre-entry pump-history anti-scam block in runtime:
  - `safe_pump_history` (mapped to `PRE_PUMP_HISTORY_BLOCK`)
  - rolling window-based guard against cooled-down post-pump rug candidates.
- New config keys to verify in active profile/env:
  - `LOCAL_ANTISCAM_PUMP_HISTORY_ENABLED`
  - `LOCAL_ANTISCAM_PUMP_HISTORY_WINDOW_SECONDS`
  - `LOCAL_ANTISCAM_PUMP_HISTORY_BLOCK_MAX_ABS_CHANGE_5M`
  - `LOCAL_ANTISCAM_PUMP_HISTORY_BLOCK_MAX_AGE_SECONDS`
  - `LOCAL_ANTISCAM_PUMP_HISTORY_TO_BLACKLIST`
- Runtime tuner now hard-blocks mutation attempts for safety families:
  - `LOCAL_ANTISCAM_*`
  - `ENTRY_PRE_RUG_*`
  - `POST_ENTRY_RUG_*`
  - `TOKEN_SAFETY_*`
  - `HONEYPOT_*`
- Contract hardening:
  - `tools/matrix_safe_tuning_contract.json` now protects anti-scam/honeypot/rug/token-safety keys from tuning contract overrides.

## Commit Update 2026-03-07 (Runtime Hot-Apply lock/stale guard)
- Runtime apply safety in `main_local.py`:
  - hot-apply now requires active tuner lock with local control (`control_local_controllers=true`),
  - stale runtime patch payloads are ignored,
  - patch `profile_id` must match current `run_tag`.
- First diagnostic checks after restart:
  - verify `active_matrix` override values for `PAPER_TRADE_SIZE_MIN_USD`, `SOURCE_ROUTER_MIN_TRADES`, `TOKEN_EV_MEMORY_MIN_TRADES`,
  - verify latest session log has no unexpected `RUNTIME_TUNER_HOT_APPLY` while tuner runs in dry-run lock mode.

## Commit Update 2026-03-12 (V5 structural lanes + anti-scam hard)
- Lane split and lane metrics are first-class:
  - `lane_tag` (`stable|discovery`) now flows through candidate -> plan -> open/close telemetry.
  - Check `FLOW_LANE_METRICS` in `main_local` logs and `telemetry_v2.lane_profile_15m` in tuner ticks.
- Non-watch upstream bridge hardening:
  - onchain unresolved rows are retried/re-enriched (not dropped blindly),
  - low-flow fetch has timeout + cache-fill fallback path.
- Startup config consistency hardening:
  - runtime startup now validates/aligned critical override layers (`STARTUP_CONFIG_SYNC` markers).
  - diagnose unknown/unused/critical-diff from startup logs before touching knobs.
- Anti-scam hard contour extended:
  - non-watch hard pre-buy gate (`ENTRY_HARD_NON_WATCH_SCAM_*`),
  - post-entry pump guard (`POST_ENTRY_PUMP_*`) to prevent +300% distortion/rug-like outliers.
- Structural key ownership tightened:
  - tuner now applies structural locks in all modes (not only dry-run),
  - lane/source structural keys must not be tuned dynamically.
