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
- `rollback_degrade_streak`

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

Use more selective mode:

```powershell
python tools/matrix_runtime_tuner.py run --profile-id u_station_ab_diag_flow_peoff --mode sniper --duration-minutes 60
```

## Output

Console line per tick:

- scanned, trade candidates, selected, opened
- apply state (`noop`, `written`, `written_restarted`, `written_restart_deferred`, ...)
- concrete key deltas and reasons

Persistent JSONL log includes:

- window metrics
- top reason codes
- buy concentration metrics (`autobuy_total`, `unique_buy_symbols`, `top_buy_symbol_share`)
- action list
- validation issues (if any)
- restart result
- telemetry v2:
  - `funnel_15m` (`raw -> source -> pre -> thr -> quarantine -> exec -> buy`)
  - `top_reasons_15m` (filter/quality/plan/execute/exit)
  - `exec_health_15m` (open/close rates, fail reasons, roundtrip loss stats)
  - `exit_mix_60m` (close reason distribution)
  - `blacklist_forensics_15m` (share/additions/tokens/detail reasons)
  - `config_hash_before/config_hash_after/config_diff`
  - `run_tag` and `commit_hash`
  - `tuner_effective` and `pending_runtime_diff_keys`
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

The tuner also has a recovery path: if flow exists but quality pressure is high
(`symbol_concentration`/`min_trade_size`/weak window outcome), it can tighten
score and edge thresholds back up.
