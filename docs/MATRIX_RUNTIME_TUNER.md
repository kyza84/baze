# Matrix Runtime Tuner

`tools/matrix_runtime_tuner.py` is a sidecar controller for matrix paper runs.

It analyzes the latest session log window and applies small preset changes through the safe tuning contract.

## Scope

- Matrix only (paper calibration before live).
- User presets only (`data/matrix/user_presets/<profile>.json`).
- Hard safety remains untouched.
- Writes decision log to `logs/matrix/<profile>/runtime_tuner.jsonl`.

## Modes

- `conveyor`: stable flow focus.
- `fast`: aggressive throughput focus.
- `calm`: more conservative.
- `sniper`: strict selectivity.

## Safety Rules

- Only allowed keys are changed (validated with `matrix_preset_guard`).
- Runtime mutable whitelist is enforced in tuner code; out-of-scope keys are blocked and logged.
- Protected/safety keys are blocked before validation (attempts are recorded in `blocked_actions`).
- Hot-apply channel updates runtime config without restart for hot-safe keys.
- Restart is requested only for keys marked as restart-required by tuner config.
- Preset restarts are done only when `open=0` from latest `PAPER_SUMMARY`.
- Restart cooldown is enforced (`--restart-cooldown-seconds`).

## Policy Engine

The tuner now runs an explicit policy phase state machine:

- `expand`: prioritize throughput recovery when flow is starved.
- `hold`: neutral mode; avoid unnecessary drift.
- `tighten`: prioritize risk/blacklist containment.

By default `--policy-phase auto` chooses phase from metrics.
You can force a phase for diagnostics with `--policy-phase expand|hold|tighten`.

Target policy knobs (CLI or JSON file via `--target-policy-file`):

- `target_trades_per_hour`
- `target_pnl_per_hour_usd`
- `min_open_rate_15m`
- `min_selected_15m`
- `min_closed_for_risk_checks`
- `min_winrate_closed_15m`
- `max_blacklist_share_15m`
- `max_blacklist_added_15m`
- `pre_risk_min_plan_attempts_15m`
- `pre_risk_route_fail_rate_15m`
- `pre_risk_buy_fail_rate_15m`
- `pre_risk_sell_fail_rate_15m`
- `pre_risk_roundtrip_loss_median_pct_15m`
- `tail_loss_min_closes_60m`
- `tail_loss_ratio_max`
- `rollback_degrade_streak`
- `adaptive_target_enabled`
- `adaptive_target_floor_trades_per_hour`
- `adaptive_target_step_up_trades_per_hour`
- `adaptive_target_step_down_trades_per_hour`
- `adaptive_target_headroom_mult`
- `adaptive_target_headroom_add_trades_per_hour`
- `adaptive_target_stable_ticks_for_step_up`
- `adaptive_target_fail_ticks_for_step_down`

### Adaptive Target Control

`target_trades_per_hour` is now treated as a requested ceiling, not a hard immediate demand.

The tuner keeps an internal `effective_target_trades_per_hour` in runtime state and adjusts it:

- step-up: only after consecutive stable ticks,
- step-down: when flow/risk/blacklist failures persist,
- capped by observed throughput headroom (`observed_tph * mult + add`).

This prevents aggressive over-expansion when market conditions cannot satisfy a high requested target,
while still allowing gradual throughput growth when the system is healthy.

### Pre-Risk And Tail-Risk

Phase selection now includes two additional risk guards:

- `pre_risk_fail`: can trigger `tighten` before enough closes accumulate when route/buy/sell fail rates or median roundtrip loss deteriorate.
- `tail_loss_fail`: triggers `tighten` when 60m tail-loss ratio (`abs(largest_loss_usd) / median_win_usd`) exceeds policy limit.

## Rollback Guard

The tuner persists controller state in:

- `logs/matrix/<profile>/runtime_tuner_state.json`

If degradation (risk/blacklist) persists for `rollback_degrade_streak`, tuner can build
rollback actions toward the last stable mutable override snapshot.

## Quick Start

One dry run:

```powershell
python tools/matrix_runtime_tuner.py once --profile-id u_station_ab_diag_flow_peoff --mode conveyor --dry-run
```

Run for 60 minutes (pre-live calibration window):

```powershell
python tools/matrix_runtime_tuner.py run --profile-id u_station_ab_diag_flow_peoff --mode conveyor --duration-minutes 60 --interval-seconds 120
```

Replay recent tuner decisions (offline audit):

```powershell
python tools/matrix_runtime_tuner.py replay --profile-id u_station_ab_diag_flow_peoff --limit 240
```

PowerShell wrapper (same commands with project-root auto-wiring):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/matrix_runtime_tuner.ps1 -Command run -ProfileId u_station_ab_diag_flow_peoff -Mode conveyor -DurationMinutes 60 -IntervalSeconds 120
```

Open a dedicated visible tuner window (recommended for live monitoring):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/matrix_runtime_tuner_open.ps1 -ProfileId u_station_ab_diag_flow_peoff -Mode conveyor -DurationMinutes 600
```

This window mirrors stdout to a readable plain-text file:

- `logs/matrix/<profile>/runtime_tuner_console_<timestamp>.log`
- structured machine log remains in `logs/matrix/<profile>/runtime_tuner.jsonl`

Use more selective mode:

```powershell
python tools/matrix_runtime_tuner.py run --profile-id u_station_ab_diag_flow_peoff --mode sniper --duration-minutes 60
```

## Output

Console line per tick:

- scanned, trade candidates, selected, opened
- apply state (`noop`, `written_hot_applied`, `written_restarted`, `written_restart_deferred`, ...)
- concrete key deltas and reasons

Persistent JSONL log includes:

- window metrics
- top reason codes
- buy concentration metrics (`autobuy_total`, `unique_buy_symbols`, `top_buy_symbol_share`)
- action list
- validation issues (if any)
- restart result
- runtime patch sync result (hot-apply channel into running `main_local.py`)
- telemetry v2:
  - `funnel_15m` (`raw -> source -> pre -> thr -> quarantine -> exec -> buy`)
  - `top_reasons_15m` (filter/quality/plan/execute/exit)
  - `exec_health_15m` (open/close rates, fail reasons, roundtrip loss stats)
  - `exit_mix_60m` (close reason distribution)
  - `blacklist_forensics_15m` (share/additions/tokens/detail reasons + detail classes `honeypot/unknown/...`)
  - `config_hash_before/config_hash_after/config_diff`
  - `run_tag` and `commit_hash`
  - `tuner_effective`, `pending_runtime_diff_keys`, `pending_restart_diff_keys`, `pending_hot_apply_diff_keys`
- decision engine trace:
  - `decision_trace` (human-readable reason chain per tick)
  - `decision_meta.rule_hits` (which rules fired in this tick)
  - `decision_meta.policy` (phase and degrade counters)
  - `blocked_actions` and `delta_capped_actions`

## Anti-Concentration

The tuner detects repeated-symbol lock-in from recent `AUTO_BUY` events.

When concentration is high (for example one symbol dominates buys in the window), it can tune:

- `V2_UNIVERSE_NOVELTY_MIN_SHARE` (up),
- `SYMBOL_CONCENTRATION_MAX_SHARE` (down),
- `V2_SOURCE_QOS_MAX_PER_SYMBOL_PER_CYCLE` (down),
- `MAX_TOKEN_COOLDOWN_SECONDS` (up).

All changes are still validated by safe contract rules.

## Edge Deadlock Recovery

When flow exists but plan stage is dominated by edge-related skips (`edge_low` / `edge_usd_low` / `negative_edge` / `ev_net_low`), tuner now applies a dedicated recovery path:

- relaxes runtime edge floors (`V2_ROLLING_EDGE_MIN_USD`, `V2_ROLLING_EDGE_MIN_PERCENT`),
- relaxes calibration floors (`V2_CALIBRATION_EDGE_USD_MIN`, `V2_CALIBRATION_VOLUME_MIN`),
- enforces calibration anti-tighten window (`V2_CALIBRATION_NO_TIGHTEN_DURING_RELAX_WINDOW=true`),
- can temporarily disable calibration on severe deadlock (`V2_CALIBRATION_ENABLED=false`), and re-enable automatically after recovery.

When `source_qos_cap` dominates filter failures in low throughput, tuner also rebalances source caps and top-K (`V2_SOURCE_QOS_SOURCE_CAPS`, `V2_SOURCE_QOS_TOPK_PER_CYCLE`).

The tuner also has a recovery path: if flow exists but quality pressure is high
(`symbol_concentration`/`min_trade_size`/weak window outcome), it can tighten
score and edge thresholds back up.

## Churn Lock

The tuner now derives `symbol_churn_15m` from trade decisions and can trigger a
temporary churn lock when one symbol repeatedly opens/closes with near-flat exits.

When lock is active:

- dominant symbol is added to `AUTO_TRADE_EXCLUDED_SYMBOLS` for TTL (runtime state),
- diversity reserve is reinforced (`V2_UNIVERSE_NOVELTY_MIN_SHARE`,
  `SYMBOL_CONCENTRATION_MAX_SHARE`, `PLAN_MIN_NON_WATCHLIST_PER_BATCH`,
  `PLAN_MAX_SINGLE_SOURCE_SHARE`),
- re-entry and cooldown pressure are tightened (`V2_QUALITY_SYMBOL_REENTRY_MIN_SECONDS`,
  `MAX_TOKEN_COOLDOWN_SECONDS`),
- repeated-entry probability gates are tightened
  (`TOKEN_EV_MEMORY_*_ENTRY_PROBABILITY`, `SOURCE_ROUTER_*_ENTRY_PROBABILITY`).

Lock is released automatically when TTL expires or when diversity recovers in-window.
Churn lock state is persisted in `runtime_tuner_state.json`.
