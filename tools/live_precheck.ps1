$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $PythonExe)) {
  $PythonExe = 'python'
}

$ReportFile = Join-Path $ProjectRoot 'data\live_precheck_report.json'

& $PythonExe tools\preflight_live_check.py --env-file .env --json-out $ReportFile
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
  Write-Host "Precheck: PASS"
} else {
  Write-Host "Precheck: FAIL (exit=$exitCode)"
}

Write-Host "Report: $ReportFile"
exit $exitCode

