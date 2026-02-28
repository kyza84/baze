# NEXT_CHAT_CONTEXT

## Snapshot
- Date: 2026-02-28
- Repo: d:\earnforme\solana-alert-bot
- Main remote for this project: https://github.com/kyza84/baze
- Working profile: u_station_ab_night_autotune_v2

## Current State
- Matrix and runtime tuner have active integration and shared logs.
- Candidate diversity improved from single-token loops to multi-symbol activity windows.
- Main skip reasons in recent windows are still EV/cooldown dominated (`ev_net_low`, `cooldown`).
- Anti-scam keys are locked in safe tuning contract and excluded from tuner mutations.

## What Was Updated In This Wave
- Runtime tuning behavior and guardrails were iterated in:
  - `tools/matrix_runtime_tuner.py`
  - `tools/matrix_runtime_tuner.ps1`
  - `tools/matrix_runtime_tuner_open.ps1`
- Matrix stop/control adjustments in:
  - `tools/matrix_paper_stop.ps1`
- State write hardening in:
  - `utils/state_file.py`
- Runtime/GUI integration updates in:
  - `main_local.py`
  - `launcher_gui.py`
- Test coverage extended in:
  - `tests/test_matrix_runtime_tuner.py`

## 48h Logs Included
- `logs/matrix/u_station_ab_night_autotune_v2/`
  - candidates/trades/alerts JSONL
  - app/out rolling logs
  - runtime tuner logs + state
  - session logs (`sessions/main_local_*.log`) for last 48h window

## Known Constraints
- Market quality can still become bottleneck even with correct mechanics.
- Tuner cannot override contract-locked safety keys.
- Throughput target is bounded by actual candidate quality and execution viability.

## Quick Commands
Run matrix profile:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_launcher.ps1 -Run -ProfileIds u_station_ab_night_autotune_v2
```

Run tuner:
```powershell
python tools\matrix_runtime_tuner.py run --profile-id u_station_ab_night_autotune_v2 --mode conveyor --duration-minutes 60 --interval-seconds 120
```

Dry-run tuner:
```powershell
python tools\matrix_runtime_tuner.py run --profile-id u_station_ab_night_autotune_v2 --mode conveyor --duration-minutes 60 --interval-seconds 120 --dry-run
```

Stop matrix:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_stop.ps1 -HardKill
```

## Next Chat First Step
1. Validate process liveness vs active_matrix status.
2. Cut 30m and 60m funnel from `candidates.jsonl` and `trade_decisions.jsonl`.
3. Compare tuner actions in same window (`runtime_tuner.jsonl`) to trade outcomes.
4. Only then adjust mutable flow keys.
