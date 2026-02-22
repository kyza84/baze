# Safe Tuning Agent Protocol

This protocol is designed for external agents so they cannot break critical runtime controls.

## Mandatory Rules
1. Change presets only via:
   - GUI tab `Matrix Presets`, or
   - `tools\matrix_user_presets.ps1` commands.
2. Never edit `tools\matrix_paper_launcher.ps1` directly for tuning.
3. Only keys from the safe contract are allowed.
4. Any key outside allow-list is blocked automatically.
5. Protected keys (safety/live/router/identity) are blocked automatically.

## Safety Contract
- Contract file: `tools/matrix_safe_tuning_contract.json`
- Validator: `tools/matrix_preset_guard.py`

Show allowed keys:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_user_presets.ps1 allowed
```

Validate a preset file manually:

```powershell
python tools\matrix_preset_guard.py validate-preset-file --root . --path data\matrix\user_presets\u_test.json
```

## Execution Guard
- `matrix_user_presets.py` validates on `create` and `clone`.
- `matrix_paper_launcher.ps1` validates user preset files before loading.
- Invalid/unsafe user presets are skipped and do not start.

## Recommended Agent Workflow
1. Pick trusted base (`mx30_guarded_balanced` or `mx31_guarded_aggressive`).
2. Apply only small changes (3-8 keys).
3. Keep one profile balanced and one aggressive.
4. Run matrix for a fixed window.
5. Compare net realized PnL and stability.
6. Iterate in small deltas, not large jumps.

## Prompt Template For External Agent
Use this exact instruction:

`Use only user presets. Do not edit launcher or core code. Modify only keys allowed by tools/matrix_safe_tuning_contract.json. If validation fails, stop and report exact keys/errors.`
