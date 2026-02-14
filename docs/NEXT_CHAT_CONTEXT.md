# Fresh Review Context (Handoff)

## Goal
Do a fresh, unbiased technical review of the bot core, execution logic, adaptive tuning, and dataset quality. Focus on why long runs often converge to near-zero PnL or low trade frequency.

## Repo / Runtime
- Project root: `d:\earnforme\solana-alert-bot`
- GUI launcher: `launcher_gui.py`
- Main loop: `main_local.py`
- Trading core: `trading/auto_trader.py`
- Config: `config.py`
- Matrix launcher: `tools/matrix_paper_launcher.ps1`

## Important Constraints
- User starts/stops from GUI manually.
- Current calibration mode uses matrix (2 profiles):
  - `mx1_refine` (control, safer)
  - `mx2_explore_wide` (aggressive, adaptive apply)
- Paper calibration baseline balance target: `$7` per profile.

## What was added recently
1. Adaptive anti-stall reset in `AdaptiveFilterController`:
   - If windows keep producing candidates but open=0, controller performs `anti_stall_reset` and pulls thresholds back toward baseline.
   - New config keys:
     - `ADAPTIVE_ZERO_OPEN_RESET_ENABLED`
     - `ADAPTIVE_ZERO_OPEN_WINDOWS_BEFORE_RESET`
     - `ADAPTIVE_ZERO_OPEN_MIN_CANDIDATES`

2. Matrix profile guardrails:
   - `mx2_explore_wide` now uses bounded adaptive ranges and dedup relax to avoid self-choking.
   - `mx1_refine` loosened one step (score/volume/edge/hourly cap) to avoid too-few trades.

3. Dataset tooling:
   - `tools/analyze_paper_dataset.py` (aggregated paper-state analysis)
   - `tools/fetch_open_market_dataset.py` (collect external market JSONL)
   - `tools/analyze_open_market_dataset.py` (stats over external JSONL)

4. Existing risk additions:
   - `RISK_GOVERNOR_*` settings in `config.py`
   - enforcement in `trading/auto_trader.py`

## Current Key Files to inspect first
- `main_local.py`:
  - `AdaptiveFilterController`
  - per-cycle filter pass/fail and policy logs
  - candidate decision logging
- `trading/auto_trader.py`:
  - `can_open_trade()` and `_cannot_open_trade_detail()`
  - edge gating
  - risk governor blocks
- `tools/matrix_paper_launcher.ps1`:
  - per-profile env overrides
- `launcher_gui.py`:
  - matrix start/stop behavior and timeout fallback handling

## Data locations
- Matrix candidate logs:
  - `logs/matrix/mx1_refine/candidates.jsonl`
  - `logs/matrix/mx2_explore_wide/candidates.jsonl`
- Matrix states:
  - `trading/paper_state.mx1_refine.json`
  - `trading/paper_state.mx2_explore_wide.json`
- Analysis outputs:
  - `data/analysis/paper_dataset_report_*.json|md`
- External dataset:
  - `data/external/open_market_dataset.jsonl`

## Known pain pattern to verify
- High scan volume, but low opens due to combined gates:
  - `heavy_dedup_ttl`
  - `safe_volume`
  - `score_min`
  - occasional edge/risk gate
- Adaptive `apply` may over-tighten if not bounded.

## Review checklist (fresh look)
1. Confirm trade suppression root-cause ordering:
   - filter stage vs edge gate vs risk governor vs policy mode.
2. Validate adaptive controller behavior over long windows:
   - does anti-stall trigger when expected?
   - does it actually reduce thresholds and restore opens?
3. Validate matrix fairness:
   - control profile stays stable
   - aggressive profile explores but does not self-choke.
4. Evaluate exit reason economics:
   - contribution by `TIMEOUT/WEAKNESS/NO_MOMENTUM/SL/TP`.
5. Recommend one nightly live profile only after paper evidence.

## Useful commands
```powershell
python tools\analyze_paper_dataset.py
python tools\fetch_open_market_dataset.py --cycles 120 --interval 30
python tools\analyze_open_market_dataset.py --file data\external\open_market_dataset.jsonl
```

```powershell
rg -n "ADAPTIVE_FILTERS mode=|action=|anti_stall_reset|AUTO_POLICY|DATA_POLICY|AutoTrade skip" logs -S
```

## Expected output from fresh review
- Clear ranking of bottlenecks (top 3) with evidence.
- One conservative paper preset + one exploratory preset.
- A narrow, testable 2-4 hour experiment plan with pass/fail thresholds.
