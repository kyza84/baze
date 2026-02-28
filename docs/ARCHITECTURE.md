# Architecture Deep Dive

## 1. System Layers

### 1.1 Control Plane
- `launcher_gui.py`
  - Operator UI.
  - Starts/stops single and matrix modes.
  - Reads matrix status and selected profile logs.
- `tools/matrix_paper_launcher.ps1`
  - Profile materialization into env files.
  - Process bootstrap for `main_local.py` workers.
  - Writes matrix run metadata.
- `tools/matrix_paper_stop.ps1`
  - Creates graceful stop signals.
  - Optional hard kill fallback.
- `tools/matrix_watchdog.py`
  - Verifies worker liveness by PID and heartbeat.
  - Updates matrix run status.

### 1.2 Decision Engine
- `main_local.py`
  - Runtime loop and stage orchestration.
  - Candidate collection -> filter/precheck -> planning -> execution dispatch.
  - Emits stage-level diagnostics (`candidates.jsonl`).
- `trading/v2_runtime.py`
  - Quality routing controls and source shaping.
- `trading/runtime_policy.py`
  - Policy state machine and gating helpers.

### 1.3 Execution Layer
- `trading/auto_trader.py`
  - Position open/close lifecycle.
  - Trade governance, cooldowns, limits.
  - Interface between planner and paper/live executor.
- `trading/live_executor.py`
  - Route/quote/swap/receipt handling for live mode.

### 1.4 State + Persistence
- `trading/auto_trader_state.py`
  - In-memory/state object model, persistence glue.
- `utils/state_file.py`
  - Locking and atomic replace writes.
  - Shared defensive path against multi-writer corruption.

### 1.5 Observability
- `utils/log_contracts.py`
  - Schema-stamped log records.
- `monitor/local_alerter.py`
  - Alert event emission.
- Runtime files:
  - `logs/matrix/<profile>/candidates.jsonl`
  - `logs/matrix/<profile>/trade_decisions.jsonl`
  - `logs/matrix/<profile>/local_alerts.jsonl`
  - `logs/matrix/<profile>/runtime_tuner.jsonl`

### 1.6 Tuning Layer
- `tools/matrix_runtime_tuner.py`
  - Reads rolling funnel/execution telemetry.
  - Produces bounded parameter actions.
  - Supports dry-run and live apply.
- `tools/matrix_safe_tuning_contract.json`
  - Hard limits and allow-list.
- `tools/matrix_preset_guard.py`
  - Enforces contract compliance.

## 2. Data/State Flow
1. Control plane starts profile worker(s).
2. `main_local.py` loads env and state.
3. Candidate stream enters decision pipeline.
4. `auto_trader` executes paper/live decisions.
5. State saved via lock+atomic path.
6. Logs/alerts written for diagnostics.
7. Runtime tuner consumes logs and updates mutable knobs.

## 3. Safety Boundaries
- Anti-scam hard keys are contract-locked.
- Runtime tuner can only touch allow-listed mutable keys.
- State persistence uses lock + temp + replace semantics.
- Matrix watchdog isolates stale/dead workers.

## 4. Operational Files
- Matrix run descriptor: `data/matrix/runs/active_matrix.json`
- Profile env snapshots: `data/matrix/env/*.env`
- User presets: `data/matrix/user_presets/*.json`
- Per-profile logs: `logs/matrix/<profile>/`

## 5. 48h Diagnostic Snapshot Policy
Current repository state includes latest 48h logs for:
- `u_station_ab_night_autotune_v2`

Purpose:
- Reconstruct candidate funnel behavior.
- Validate tuner actions against realized decisions.
- Audit stage reasons (`filter_fail`, `ev_net_low`, `cooldown`, `blacklist`, etc.).

## 6. Test Surface
- `tests/test_matrix_runtime_tuner.py`
  - Tuner action selection, bounds, protections.
- Other `tests/test_*.py`
  - Policy behavior, state safety, runtime invariants.
