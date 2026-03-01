# PROJECT_STATE

## Snapshot
- Updated: 2026-03-01
- Repo: d:\earnforme\solana-alert-bot
- Remote: https://github.com/kyza84/baze
- Branch: main
- Active profile: u_station_ab_night_autotune_v2

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
- Frequent source concentration on watchlist in planning windows.

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
