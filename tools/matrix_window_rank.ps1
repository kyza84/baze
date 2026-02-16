param(
  [double]$LookbackHours = 4,
  [int]$MinClosed = 10
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  throw "Python not found: $python"
}

$scriptPath = Join-Path $PSScriptRoot 'matrix_window_rank.py'
& $python $scriptPath --root $root --lookback-hours $LookbackHours --min-closed $MinClosed

