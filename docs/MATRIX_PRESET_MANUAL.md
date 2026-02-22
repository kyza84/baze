# Matrix Preset Manual

This guide adds an operator layer for matrix presets without editing `tools/matrix_paper_launcher.ps1` manually.

## GUI Workflow (Recommended)
In `launcher_gui.py` use tab `Matrix Presets`:
- `Refresh Catalog`: reload built-in/user presets + recent winners.
- Select profiles and click `Use Selected For Matrix`.
- Create/update user preset from form:
  - `Preset name`
  - `Base profile`
  - `Overrides editor` (key/value table with Add/Update, Remove, Clear)
- `Matrix Start` in header will start selected profiles.
- `Delete Selected User Preset` removes only `kind=user` presets.
- `Load Selected User Preset` fills the form from an existing user preset (also works by double-click on user row).

## 1) View All Profiles
List built-in presets, user presets, and recent report winners:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_profile_catalog.ps1
```

JSON output:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_profile_catalog.ps1 --json
```

## 2) Manage User Presets
List user presets:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_user_presets.ps1 list
```

Create a user preset from a built-in base (example from `mx30_guarded_balanced`):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_user_presets.ps1 create --name u_mx30_safe --base mx30_guarded_balanced --set MAX_OPEN_TRADES=5 --set AUTO_TRADE_TOP_N=24 --note "safe variant from mx30"
```

Clone preset quickly (base can be built-in or another user preset):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_user_presets.ps1 clone --source mx31_guarded_aggressive --name u_mx31_probe --set MAX_BUYS_PER_HOUR=110 --note "probe from mx31"
```

Show one preset:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_user_presets.ps1 show --name u_mx30_safe
```

Delete one preset:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_user_presets.ps1 delete --name u_mx31_probe
```

## 3) Launch Matrix With User Presets
`matrix_paper_launcher.ps1` now auto-loads JSON files from:

- `data\matrix\user_presets\*.json`

So you can launch user presets exactly like built-ins:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\matrix_paper_launcher.ps1 -Run -Count 2 -ProfileIds u_mx30_safe,u_mx31_probe
```

## 4) Safety Rules
- User preset name cannot collide with built-in profile names.
- Preset file must contain: `name`, `base`, `overrides` (object).
- Base chain is resolved by launcher (same logic as built-ins).
- Invalid JSON preset is skipped with warning (launcher continues).
- Unsafe keys/values are blocked by contract validation (`tools/matrix_safe_tuning_contract.json`).
- Launcher also re-validates user presets before run and skips invalid ones.

## 5) Recommended Workflow
1. Use catalog to see what exists and what recently won.
2. Clone winner (`mx30`/`mx31`) into `u_*`.
3. Apply only small override set per variant.
4. Run 2 profiles in matrix.
5. Compare by net realized PnL and stability window, then keep winner.
