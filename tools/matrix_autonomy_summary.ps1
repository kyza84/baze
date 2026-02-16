param(
  [int]$Tail = 3
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  throw "Python not found: $python"
}

& $python (Join-Path $PSScriptRoot 'matrix_autonomy_summary.py') --root $root --tail $Tail

