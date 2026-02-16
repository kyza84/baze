param(
  [double]$LookbackHours = 6,
  [int]$MinClosed = 8
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  throw "Python not found: $python"
}

& $python (Join-Path $PSScriptRoot 'matrix_rank_profiles.py') --root $root --lookback-hours $LookbackHours --min-closed $MinClosed
