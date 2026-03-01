# ACCOUNT_TRANSFER_PROMPT

Use this prompt when switching to a second account/VSCode session and continuing work from current project state without losing context.

Before opening a new chat, send `docs/CHAT_FIRST_MESSAGE.md` first (copy-paste as the first message).

## Prompt Template

```text
You are Codex working in repository d:\\earnforme\\solana-alert-bot.

Goal:
Continue from the latest matrix+tuner state without re-planning from scratch.
Do not change strategy. Do only evidence-based engineering/debug work.

Read first:
1) NEXT_CHAT_CONTEXT.md
2) docs/NEXT_CHAT_CONTEXT.md
3) data/matrix/runs/active_matrix.json
4) latest logs in logs/matrix/u_station_ab_night_autotune_v2/

Hard rules:
- No destructive git commands.
- No safety-key downgrades.
- Before edits: check current process/liveness and last 30-60m funnel.
- Every change must include: problem -> fix -> risk -> verification.
- If uncertain: add diagnostics/metrics first, then refactor.

Working profile:
- u_station_ab_night_autotune_v2

Expected first actions:
1) Validate matrix/tuner/watchdog/unified_sync processes.
2) Build 30m and 60m cuts from candidates.jsonl + trade_decisions.jsonl + runtime_tuner.jsonl.
3) Report top bottleneck stage and top reasons.
4) Only then propose minimal safe patch.

Output format each iteration:
- Current state (facts only)
- Bottleneck (stage + reason codes)
- Change made (if any)
- Verification evidence
- Next safe step
```

## Quick Start Commands

```powershell
# Open repo
cd d:\earnforme\solana-alert-bot

# Check active matrix
Get-Content data\matrix\runs\active_matrix.json -Raw

# Process liveness
Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(\\.exe)?$' -and $_.CommandLine -match 'main_local\\.py|matrix_runtime_tuner\\.py|matrix_watchdog\\.py|unified_dataset_sync\\.py' } | Select-Object ProcessId,CommandLine

# Latest session log tail
$session = Get-ChildItem logs\matrix\u_station_ab_night_autotune_v2\sessions\main_local_*.log -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content $session.FullName -Tail 120
```
