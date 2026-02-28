# NEXT_CHAT_CONTEXT (docs copy)

Use this file as a fallback context handoff mirror.
Primary context file is repository root `NEXT_CHAT_CONTEXT.md`.

## Snapshot
- Date: 2026-02-28
- Profile focus: u_station_ab_night_autotune_v2
- Repository: d:\earnforme\solana-alert-bot

## Mirrors
- Root context: `NEXT_CHAT_CONTEXT.md`
- Architecture: `docs/ARCHITECTURE.md`
- Tuner manual: `docs/MATRIX_RUNTIME_TUNER.md`

## Included Diagnostic Scope
- Last 48h logs for matrix+tuner under:
  - `logs/matrix/u_station_ab_night_autotune_v2/`

## Immediate Operator Routine
1. Check matrix/tuner process liveness.
2. Verify `active_matrix.json` status matches real PID state.
3. Build funnel cut (30m/60m) from candidates/trade decisions.
4. Correlate with tuner actions before any further knob change.
