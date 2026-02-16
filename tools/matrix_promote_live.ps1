param(
  [string]$ProfileId = "",
  [double]$LookbackHours = 4,
  [ValidateSet('off','last','median')]
  [string]$WindowControls = 'median',
  [int]$MinClosed = 10,
  [string]$Confirm = "",
  [switch]$Apply
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  throw "Python not found: $python"
}

$args = @((Join-Path $PSScriptRoot 'matrix_promote_live.py'), '--root', $root)
if ($ProfileId -and $ProfileId.Trim().Length -gt 0) {
  $args += @('--profile-id', $ProfileId.Trim())
}
$args += @('--lookback-hours', [string]$LookbackHours, '--min-closed', [string]$MinClosed, '--window-controls', $WindowControls)
if ($Confirm -and $Confirm.Trim().Length -gt 0) {
  $args += @('--confirm', $Confirm.Trim())
}
if ($Apply) {
  if (-not $ProfileId -or $ProfileId.Trim().Length -eq 0) {
    throw "Apply requires -ProfileId (manual winner selection)."
  }
  if ($Confirm.Trim() -ne 'CONFIRM_LIVE_SWITCH') {
    throw "Apply blocked. Pass explicit confirm phrase: -Confirm CONFIRM_LIVE_SWITCH"
  }
  $args += '--apply'
}

& $python @args
