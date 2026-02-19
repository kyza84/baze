# Base Bot Control Stack

Local-first trading stack for Base with paper/matrix calibration and controlled live execution.

## What Is Included
- Market + on-chain ingestion pipeline.
- Multi-layer safety and policy gating.
- Paper and live auto-trader.
- Matrix A/B runs with profile isolation.
- Runtime adaptive controllers (policy router, dual entry, rolling edge, KPI loop).
- Desktop GUI control center.

## Core Modules
- `main_local.py`: runtime loop, market mode detection, candidate routing, orchestration.
- `trading/auto_trader.py`: entry/exit decisions, risk caps, cooldowns, persistence.
- `trading/runtime_policy.py`: extracted data-policy + market-mode helpers.
- `trading/auto_trader_state.py`: extracted state save/load logic.
- `monitor/local_alerter.py`: local alert writer (schema-stamped JSONL).
- `monitor/gui_engine_control.py`: extracted GUI engine process control.
- `launcher_gui.py`: UI shell, matrix control, logs, wallet/trades monitoring.

## Runtime Modes
- Paper:
  - `AUTO_TRADE_ENABLED=true`
  - `AUTO_TRADE_PAPER=true`
- Live:
  - `AUTO_TRADE_ENABLED=true`
  - `AUTO_TRADE_PAPER=false`
  - valid `LIVE_*` credentials and RPC keys.

Recommended flow:
1. Paper validation.
2. Matrix A/B run.
3. Promote winner with parity.
4. Live only after precheck passes.

## GUI (Current)
- Top bar is simplified to core controls:
  - `Start`, `Stop`, `Matrix Start/Stop/Summary`, `Save`, `Restart`, `Clear Logs`, `Refresh`.
- Settings tab is curated (critical runtime keys only), not legacy full `.env` dump.
- Activity feed focuses on:
  - market mode changes (`MARKET_MODE_CHANGE`),
  - opened trades (`trade_decisions.jsonl`, `decision_stage=trade_open`).
- Raw runtime panel still shows compact full logs.

## Matrix Commands
Run matrix:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_launcher.ps1 -Count 2 -Run
```

Stop matrix:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_stop.ps1 -HardKill
```

Summary:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_summary.ps1
```

Promote winner to live parity:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_promote_live.ps1 -ProfileId <profile_id>
```

## Live Precheck
No trading actions, only readiness checks:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\live_precheck.ps1
```

Python entry:
```powershell
python tools\preflight_live_check.py --env-file .env --json-out data\live_precheck_report.json
```

## Unified Dataset Sync
Incremental deduplicated dataset into SQLite:
```powershell
python tools\unified_dataset_sync.py --root .
```

Continuous sync loop:
```powershell
python tools\unified_dataset_sync.py --root . --follow --loop-seconds 45
```

Output:
- `data/unified_dataset/unified.db`
- `data/unified_dataset/sync_state.json`

## Log Contracts (Schema Versioned)
Writers now stamp shared contract fields:
- `schema_version`
- `schema_name`
- `event_type`
- `ts`, `timestamp`
- `run_tag` (when available)

Schemas:
- `candidate_decision.v1` (`logs/**/candidates.jsonl`)
- `trade_decision.v1` (`logs/**/trade_decisions.jsonl`)
- `local_alert.v1` (`logs/**/local_alerts.jsonl`)

Implementation: `utils/log_contracts.py`.

## Tests
Run mini integration tests:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Covered areas:
- policy router (`limited/block`, hard-block, legacy behavior),
- dual-entry quotas,
- rolling-edge bounds,
- fail-closed behavior in auto-trader,
- market-mode and policy hysteresis.

## Repo Hygiene
Runtime data and secrets are intentionally excluded from git:
- `.env`,
- `logs/`,
- `data/`,
- local states/snapshots.

This keeps the repository safe while preserving code-level reproducibility.
