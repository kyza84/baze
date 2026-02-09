$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $ProjectRoot 'bot.pid'
$OutLog = Join-Path $ProjectRoot 'logs\bot.out.log'
$ErrLog = Join-Path $ProjectRoot 'logs\bot.err.log'

if (-not (Test-Path $PidFile)) {
    Write-Host 'Status: STOPPED (no pid file)'
    exit 0
}

$pidValue = Get-Content $PidFile -ErrorAction SilentlyContinue
$proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue

if ($proc) {
    Write-Host "Status: RUNNING (PID: $pidValue)"
} else {
    Write-Host "Status: NOT RUNNING (stale PID: $pidValue)"
}

if (Test-Path $OutLog) {
    Write-Host "\nLast OUT lines:"
    Get-Content $OutLog -Tail 10
}

if (Test-Path $ErrLog) {
    Write-Host "\nLast ERR lines:"
    Get-Content $ErrLog -Tail 10
}
