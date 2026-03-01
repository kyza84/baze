# CONTEXT_UPDATE_RULES

## Goal
Keep transfer context always up to date across accounts/sessions.

## Mandatory files per material commit
- `docs/PROJECT_STATE.md`
- `docs/CHAT_FIRST_MESSAGE.md`

## Optional supporting file
- `docs/ACCOUNT_TRANSFER_PROMPT.md`

## Enforcement
- Pre-commit guard: `tools/context_commit_guard.py`
- Hook: `.githooks/pre-commit`

If a commit includes material changes and misses context updates, commit is blocked.

## What is a material change
- Any staged file not under:
  - `logs/`
  - `data/`
  - `snapshots/`
  - `collected_info/`
  - `__pycache__/`
and not one of the context files themselves.

## One-time setup
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\install_git_hooks.ps1
```

## Manual check
```powershell
python tools\context_commit_guard.py
```
