# PROJECT_STATE

## Snapshot
- Updated: 2026-03-03
- Repo: d:\earnforme\solana-alert-bot
- Remote: https://github.com/kyza84/baze
- Branch: main
- Active profile: u_station_ab_night_autotune_v2
- Last validated runtime: matrix+tuner+watchdog running, no runtime exceptions, primary bottleneck remains plan-stage economics (edge/cost/cooldown).

## Documentation Update (2026-03-03, logging/transfer)
- Added `docs/LOGGING_REFERENCE.md`:
  - complete map of matrix profile logs,
  - core JSONL field contracts and telemetry blocks,
  - minimal file set for external analyzer chat handoff.
- Updated transfer chain to include logging reference:
  - `docs/CHAT_FIRST_MESSAGE.md`
  - `docs/ACCOUNT_TRANSFER_PROMPT.md`
- Updated documentation index links:
  - `README.md`
  - `docs/ARCHITECTURE.md`

## Latest Update (2026-03-03)
- `main_local.py`
  - Added early cycle-level skip for `address_or_duplicate/open_duplicate` when token is already in `auto_trader.open_positions`.
  - Effect: removes known non-openable rows before `post_filters`/`plan` bridge accounting.
- `trading/auto_trader.py`
  - Added `EDGE_COST_DOMINANT_*` guard path:
    - tracks repeated cost-dominant edge skips (`cost_total_percent > gross_percent` with thresholds),
    - applies temporary symbol cooldown after configurable hit threshold,
    - prunes guard window in daily refresh and reset path.
  - Added fast guard for non-watch explore micro-size attempts and non-watch explore size floor (`NON_WATCH_EXPLORE_*` controls).
  - Added richer skip forensics (`_bump_skip_reason_ext`) for `ev_net_low`/`edge_low`/`edge_usd_low`/`negative_edge`.
  - Added plan prefilter summary event with per-reason/source/stage counters.
- `config.py`
  - Added new env controls:
    - `EDGE_COST_DOMINANT_GUARD_ENABLED`
    - `EDGE_COST_DOMINANT_MIN_GROSS_PERCENT`
    - `EDGE_COST_DOMINANT_MIN_COST_PERCENT`
    - `EDGE_COST_DOMINANT_MIN_DELTA_PERCENT`
    - `EDGE_COST_DOMINANT_HIT_WINDOW_SECONDS`
    - `EDGE_COST_DOMINANT_HITS_TO_COOLDOWN`
    - `EDGE_COST_DOMINANT_SYMBOL_COOLDOWN_SECONDS`
    - `EDGE_COST_DOMINANT_NON_WATCH_EXPLORE_*`
    - `NON_WATCH_EXPLORE_MIN_TRADE_*`
- `tools/matrix_runtime_tuner.py`
  - Added protected-key prelimit filter in action planner with telemetry (`prelimit_blocked_actions`).
  - Added `non_watch_conversion_guard`, `prefilter_plan_choke`, `symbol_loss_pressure_60m`, `edge_low_loop_15m` control paths.
  - Reworked `prefilter_plan_choke` branch:
    - tightens `V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE`,
    - increases `V2_UNIVERSE_NOVELTY_MIN_SHARE`,
    - increases `AUTO_TRADE_TOP_N`,
    - does not lower `PLAN_MIN_NON_WATCHLIST_PER_BATCH`.
- `tools/matrix_watchdog.py`
  - Added single-follow lock (`data/matrix/runs/watchdog.lock.json`) to prevent duplicate watchdog followers.
  - Hardened PID liveness checks with command-line verification (`main_local.py` / `matrix_runtime_tuner.py`).
  - Added runtime activity-age gating (heartbeat + app/out log mtimes) before stale restart.
  - Made metadata/state writes resilient (`_try_write_json`) and always refreshes `running/alive_count`.
- `tools/matrix_paper_launcher.ps1`
  - Increased watchdog `--stale-seconds` from `180` to `360` to reduce false stale restarts under transient latency.
- `tools/matrix_safe_tuning_contract.json`
  - Extended safe contract for non-watch soft conversion keys (`MARKET_MODE_NON_WATCH_SOFT_*`, `SAFE_VOLUME_TWO_TIER_NON_WATCH_*`).
  - Relaxed bounds for `SAFE_AGE_NON_WATCH_SOFT_RATIO` and `SAFE_CHANGE_5M_NON_WATCH_SOFT_MULT` for controlled conversion tests.
- Tests:
  - `python -m unittest tests.test_config_env_loading tests.test_autotrader_safety_guards tests.test_matrix_runtime_tuner`
  - Result: `Ran 105 tests ... OK`.
- Runtime restart validation:
  - `main_local.py`, `matrix_runtime_tuner.py`, `matrix_watchdog.py` are alive after restart.

## Latest Commit Scope (Non-Watch Recovery)
- Added non-watch soft filter controls in runtime config:
  - `SAFE_AGE_NON_WATCH_SOFT_*`
  - `SAFE_CHANGE_5M_NON_WATCH_SOFT_*`
- Added main pipeline handling for non-watch soft passes with per-cycle caps and explicit logs:
  - `NON_WATCH_SAFE_AGE_SOFT_PASS`
  - `NON_WATCH_SAFE_CHANGE_SOFT_PASS`
- Extended tuner mutable/contract-aware key set for these controls.
- Extended tuner starvation diagnostics to include non-watch post-filter starvation.
- Added/updated tests:
  - `tests/test_config_env_loading.py`
  - `tests/test_matrix_runtime_tuner.py`

Safety boundary (unchanged):
- Hard anti-scam/honeypot/hard-blocklist behavior was not relaxed.

## Runtime Topology
- Trader runtime: `main_local.py`
- Matrix launcher/stop:
  - `tools/matrix_paper_launcher.ps1`
  - `tools/matrix_paper_stop.ps1`
- Runtime tuner:
  - `tools/matrix_runtime_tuner.py`
  - `tools/matrix_runtime_tuner.ps1`
  - `tools/matrix_runtime_tuner_open.ps1`
- Supervisors:
  - `tools/matrix_watchdog.py`
  - `tools/unified_dataset_sync.py`

## Core Trading Components
- Decision/execution core: `trading/auto_trader.py`
- State model/load-save: `trading/auto_trader_state.py`
- Live executor: `trading/live_executor.py`
- V2 runtime routing/quality/source: `trading/v2_runtime.py`
- Atomic state file IO + lock: `utils/state_file.py`
- Structured log contracts: `utils/log_contracts.py`

## Recent Engineering Changes (Critical)
- Added explicit `plan_trade/pass` event before each `trade_open` in `auto_trader`.
- Normalized `source` across `trade_open/trade_close/trade_partial/plan_trade` payloads.
- Tuner `action_count_15m` hardened: uses fallback `max(plan_pass, trade_open)`.
- Added new diagnostics fields:
  - `action_count_plan_15m`
  - `action_count_fallback_open_15m`
  - `decision_meta.flow_choke`
  - `decision_meta.blacklist_dominator`
- Added phase-routing support for:
  - `safe_volume_soft_flow_expand`
  - `blacklist_dominator_shaping`

## Recent Incident Reference (Network Outage)
- Local time outage window observed: ~19:40 to ~19:54 (2026-03-01).
- Symptoms:
  - DNS resolution failures (`NameResolutionError`, `getaddrinfo failed`)
  - cycles with `Scanned 0 tokens`
  - source fallback with 100% source errors
- Recovery:
  - first buy after outage: 19:54:36 (`SYND`)
  - stable onchain+market resumed around 19:54:46
  - first positive close after recovery: `SYND TIMEOUT +0.0189 USD`

## Bottleneck Pattern (Current)
- Primary choke often on `plan_trade` stage.
- Top recurring reasons:
  - `ev_net_low`
  - `cooldown`
  - `edge_low`
- Watchlist concentration decreased after non-watch conversion hooks, but still appears in stress windows.

## Data Locations
- Active matrix status: `data/matrix/runs/active_matrix.json`
- Profile env: `data/matrix/env/u_station_ab_night_autotune_v2.env`
- Profile logs: `logs/matrix/u_station_ab_night_autotune_v2/`
  - `candidates.jsonl`
  - `trade_decisions.jsonl`
  - `runtime_tuner.jsonl`
  - `sessions/main_local_*.log`

## Non-Negotiable Safety Constraints
- Hard anti-scam/safety keys are not to be relaxed by tuner patches.
- No destructive git commands.
- No strategy rewrites without measured regression/proof workflow.

## Commit-Time Context Policy
For every material commit (code/config/docs that affect behavior), update both files:
1) `docs/PROJECT_STATE.md`
2) `docs/CHAT_FIRST_MESSAGE.md`

Also ensure the handoff prompt remains valid:
- `docs/ACCOUNT_TRANSFER_PROMPT.md`
- `docs/OPERATOR_PATTERNS.md` (операционные команды пользователя)

## Quick Ops Commands
```powershell
# matrix run
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_launcher.ps1 -Run -ProfileIds u_station_ab_night_autotune_v2

# matrix stop
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_stop.ps1 -HardKill

# tuner visible window
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_runtime_tuner_open.ps1 -ProfileId u_station_ab_night_autotune_v2 -Mode conveyor
```

## Next-Session First Actions
1. Process liveness check (`main_local`, tuner, watchdog, sync).
2. 30m/60m funnel cut from candidates+decisions.
3. Correlate with tuner actions in same windows.
4. Patch only after single primary bottleneck is proven.

## Operator Interaction Contract
- Базовые пользовательские фразы и поведение агента зафиксированы в:
  - `docs/OPERATOR_PATTERNS.md`
- При переносе сессии сначала загружать этот файл, затем выполнять действия.
