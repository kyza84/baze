# Base Bot Control Stack (baze)

Local-first trading stack for Base chain with paper calibration (Matrix), runtime autotuning, and guarded promotion to live.

## Repository Goal
- Keep one shared decision pipeline for paper/matrix/live.
- Tune throughput without breaking safety invariants.
- Preserve forensic traceability for every candidate/decision/trade.

## Project Architecture (File-Level)

### Entry Points
- `main_local.py`
  - Core runtime loop.
  - Candidate ingestion, filter/precheck pipeline, planner and execution calls.
  - Writes candidate/trade logs and heartbeat.
- `launcher_gui.py`
  - Desktop control center for single and matrix operations.
  - Reads active matrix status, live tails, trade tables.
- `tools/matrix_paper_launcher.ps1`
  - Matrix runner for one or many profiles.
  - Builds runtime env files and starts `main_local.py` workers.
- `tools/matrix_paper_stop.ps1`
  - Graceful/forced matrix shutdown.
- `tools/matrix_watchdog.py`
  - Watches matrix workers, stale heartbeat handling, status reconciliation.

### Trading Core
- `trading/auto_trader.py`
  - Entry/exit orchestration.
  - Risk caps, position lifecycle, blacklist/cooldown integration.
- `trading/live_executor.py`
  - Live transaction execution primitives and receipts path.
- `trading/auto_trader_state.py`
  - Position/account state serialization and recovery helpers.
- `trading/runtime_policy.py`
  - Policy state transitions and mode gating helpers.
- `trading/v2_runtime.py`
  - Quality/source controls and runtime knobs for candidate funnel shaping.

### Runtime Tuner
- `tools/matrix_runtime_tuner.py`
  - Matrix sidecar tuner.
  - Reads recent telemetry, computes actions, applies safe mutable overrides.
  - Supports `once`, `run`, `replay`, dry-run and active mode.
- `tools/matrix_runtime_tuner.ps1`
  - PowerShell wrapper for starting/stopping tuner.
- `tools/matrix_runtime_tuner_open.ps1`
  - Tuner console helper.

### Config + Guardrails
- `config.py`
  - Runtime env parsing and defaults.
- `tools/matrix_safe_tuning_contract.json`
  - Mutable allow-list and bounds for preset/runtime tuning.
- `tools/matrix_preset_guard.py`
  - Validation of user presets against contract.
- `tools/matrix_user_presets.py`
  - Create/clone/list/delete profile presets.

### State + IO Safety
- `utils/state_file.py`
  - Locked state read/write (file lock + atomic replace).
  - Shared by runtime state persistence and GUI maintenance actions.

### Observability
- `utils/log_contracts.py`
  - Schema fields for candidate/trade/alert records.
- `monitor/local_alerter.py`
  - Local alert emission.
- Main runtime logs location:
  - `logs/matrix/<profile_id>/...`

## Runtime Pipeline (Shared Logic)
1. Build candidate set from sources.
2. Apply safety/policy/quality filters.
3. Plan trade (`plan_trade` stage, EV/edge checks).
4. Execute (paper or live executor).
5. Postprocess state + trade lifecycle logs.

Paper/matrix/live differences are isolated at executor and environment level; decision stages are shared.

## Matrix + Autotuner Operating Model
- Matrix profile(s) run through launcher.
- Watchdog keeps status and stale-worker recovery.
- Runtime tuner reads rolling windows and proposes/applies bounded config deltas.
- Sensitive anti-scam/safety keys are contract-locked and excluded from tuner mutation.

## Directory Map
- `tools/` automation scripts (matrix launcher/stop/summary/tuner/preset guard).
- `trading/` trading engine modules.
- `monitor/` alerting and GUI support modules.
- `utils/` shared infra (state locks, log contracts, helpers).
- `tests/` unit/integration tests.
- `data/` runtime/generated data (ignored by default).
- `logs/` runtime logs (ignored by default; selected files tracked for diagnostics).
- `docs/` operator manuals and architecture notes.

## 48h Diagnostic Logs Included In This Update
Included for handoff/analysis window:
- Profile: `u_station_ab_night_autotune_v2`
- Path: `logs/matrix/u_station_ab_night_autotune_v2/`
- Contents committed:
  - rolling runtime logs (`app.log*`, `out.log*`)
  - decision streams (`candidates.jsonl`, `trade_decisions.jsonl`, `local_alerts.jsonl`)
  - tuner telemetry (`runtime_tuner.jsonl`, state/lock/overrides)
  - session logs from the last 48 hours (`sessions/main_local_*.log`)

## Key Run Files
- Active matrix status: `data/matrix/runs/active_matrix.json`
- Profile env files: `data/matrix/env/<profile>.env`
- User presets: `data/matrix/user_presets/*.json`

## Common Commands
Run matrix:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_launcher.ps1 -Run -Count 2
```

Run single profile:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_launcher.ps1 -Run -ProfileIds u_station_ab_night_autotune_v2
```

Stop matrix:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_stop.ps1 -HardKill
```

Run runtime tuner:
```powershell
python tools\matrix_runtime_tuner.py run --profile-id u_station_ab_night_autotune_v2 --mode conveyor --duration-minutes 60 --interval-seconds 120
```

Run tuner dry-run:
```powershell
python tools\matrix_runtime_tuner.py run --profile-id u_station_ab_night_autotune_v2 --mode conveyor --duration-minutes 60 --interval-seconds 120 --dry-run
```

Run tests:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Documentation Index
- `docs/ARCHITECTURE.md` - detailed architecture and ownership map.
- `docs/MATRIX_RUNTIME_TUNER.md` - tuner behavior and controls.
- `docs/MATRIX_PRESET_MANUAL.md` - preset operations.
- `docs/SAFE_TUNING_AGENT_PROTOCOL.md` - safe tuning process.
- `NEXT_CHAT_CONTEXT.md` - current working state for next session.

## Notes
- `.gitignore` excludes runtime `logs/` and `data/`; files already tracked historically can still be updated and committed.
- For fresh log snapshots under ignored trees, use `git add -f` explicitly.
