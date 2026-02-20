# NEXT_CHAT_CONTEXT

## Project Snapshot (2026-02-20)
- Repo: `d:\earnforme\solana-alert-bot`
- Mode in focus: `MATRIX` only (paper test runs for preset selection).
- Objective: raise net profitability consistency, not just trade flow stability.
- Current active test profiles:
  - `mx20_quality_balanced`
  - `mx21_quality_aggressive`

## What Was Implemented (latest patch)
### 1) Universe Quality Gate (pre-entry)
- File: `trading/v2_runtime.py`
- Controller: `UniverseQualityGateController`
- Adds EV-first routing before dual-entry/policy router:
  - `core` pool: symbols/clusters with rolling EV+ and acceptable loss share.
  - `explore` pool: unknown/neutral candidates, quota-limited.
  - `cooldown` bucket: weak symbols/clusters; not permanent ban.
- Cooldown logic:
  - rare probe entries (probabilistic re-check),
  - reduced size/hold and stricter edge multipliers for probe entries.

### 2) Dynamic Source Budget
- File: `trading/v2_runtime.py`
- Per-source allocation based on recent rolling source EV (windowed).
- Strong sources receive larger effective slot share.
- Weak sources are cut but not hard-removed.

### 3) Symbol Concentration Protection
- File: `trading/v2_runtime.py`
- Per-window limit on same-symbol entry share.
- Prevents one false-good or false-bad symbol from dominating run results.

### 4) Online Pool Rotation
- File: `trading/v2_runtime.py`
- Rotation refresh every configured interval.
- Logs `V2_QUALITY_ROTATE` with top core/cooldown symbols.

### 5) Main Pipeline Integration + Visibility
- File: `main_local.py`
- Quality gate inserted before dual-entry/router.
- Added quality-stage drop logging (`decision_stage="quality_gate"`).
- Cycle summary now includes quality metrics:
  - `Quality out/core/explore/probe`.

### 6) DualEntry Compatibility Fix
- File: `trading/v2_runtime.py`
- DualEntry now multiplies existing `_entry_channel_*` multipliers instead of overwriting.
- Required so quality-gate penalties survive downstream tagging.

### 7) Config Surface Added
- File: `config.py`
- Added `V2_QUALITY_*` controls:
  - gate on/off, windows, min trades,
  - EV/loss thresholds,
  - explore quota,
  - cooldown probe and risk multipliers,
  - source budget tuning,
  - symbol concentration limits,
  - top-symbol logging count.

### 8) Matrix Profiles Added
- File: `tools/matrix_paper_launcher.ps1`
- Added:
  - `mx20_quality_balanced` (base `mx18_v2_balanced`)
  - `mx21_quality_aggressive` (base `mx19_v2_aggressive`)

## Critical Bug Fixed
- File: `trading/v2_runtime.py`
- In quality-gate config reads, `0` values were previously overridden by defaults due to `or default` pattern.
- Fixed via explicit `None` handling so values like `0.0` apply correctly.

## Validation Status
- Compile/tests state after patch:
  - `python -m unittest discover -s tests -p "test_*.py" -v`
  - Result: `26 tests, OK`.
- Quality-gate tests present in:
  - `tests/test_v2_runtime_controls.py`

## Non-negotiable Log-Cut Protocol (must use every time)
During each session cut, evaluate ALL layers, not only PnL:
1. Flow: candidates in/out, open rate, idle gaps.
2. Pool: core/explore/cooldown shares, rotation behavior, max symbol share.
3. Source: dynamic source caps, who gets boosted/cut.
4. Entries: tier A/B distribution, edge, size, probe frequency.
5. Exits: TP/timeout/SL split, avg win/loss, hold-time profile.
6. Regime: GREEN/YELLOW/RED transitions and reasons.
7. Policy: OK/DEGRADED/FAIL_CLOSED impact on routing.
8. Outcome: realized PnL, EV/trade, EV/hour, equity smoothness.

## Interpretation Rule
- First 20-40 minutes: adaptation warming (signals valid but estimates still stabilizing).
- Stronger effect expected after enough rolling window fill (roughly 40-90 minutes).

## Current Working Principle
- MATRIX is the only place for testing/ranking presets.
- LIVE is only for selected winner preset after evidence.
- Do not claim success from one short burst; evaluate by full checklist and windowed EV.

## Commands (quick ops)
- Stop matrix:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File tools/matrix_paper_stop.ps1 -HardKill`
- Launch two quality profiles:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File tools/matrix_paper_launcher.ps1 -Run -Count 2 -ProfileIds mx20_quality_balanced,mx21_quality_aggressive`
- Matrix summary:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File tools/matrix_paper_summary.ps1`

## What To Do First In New Chat
1. Confirm only expected matrix processes are active.
2. Take a time-bounded log cut (at least 60 minutes preferred).
3. Produce checklist-based diagnosis (8 sections above).
4. Decide next patch by root cause layer (pool/source/entry/exit), not by raw PnL alone.

## Ready-to-use Prompt For Next Chat
"Continue from NEXT_CHAT_CONTEXT.md. Start with a full matrix log-cut diagnosis using the 8-layer checklist (flow/pool/source/entries/exits/regime/policy/outcome), then propose exact parameter/code changes for the weakest layer and run only two matrix profiles."