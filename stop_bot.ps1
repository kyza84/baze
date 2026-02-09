$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $ProjectRoot 'bot.pid'

if (-not (Test-Path $PidFile)) {
    Write-Host 'bot.pid not found. Bot may already be stopped.'
    exit 0
}

$pidValue = Get-Content $PidFile -ErrorAction SilentlyContinue
if (-not $pidValue) {
    Remove-Item $PidFile -ErrorAction SilentlyContinue
    Write-Host 'Empty PID file removed.'
    exit 0
}

$proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
if ($proc) {
    Stop-Process -Id $pidValue -Force
    Write-Host "Bot stopped. PID: $pidValue"
} else {
    Write-Host "Process $pidValue not found."
}

Remove-Item $PidFile -ErrorAction SilentlyContinue
