param(
    [switch]$Force
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$PidFile = Join-Path $ProjectRoot 'bot.pid'
$LogsDir = Join-Path $ProjectRoot 'logs'
$OutLog = Join-Path $LogsDir 'bot.out.log'
$ErrLog = Join-Path $LogsDir 'bot.err.log'

if (-not (Test-Path $PythonPath)) {
    Write-Error "Python not found: $PythonPath"
    exit 1
}

if (Test-Path $PidFile) {
    $existingPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($existingPid) {
        $proc = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($proc -and -not $Force) {
            Write-Host "Bot already running with PID $existingPid"
            exit 0
        }
    }
    Remove-Item $PidFile -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

$proc = Start-Process -FilePath $PythonPath `
    -ArgumentList 'main.py' `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Set-Content -Path $PidFile -Encoding ascii
Write-Host "Bot started in background. PID: $($proc.Id)"
Write-Host "OUT log: $OutLog"
Write-Host "ERR log: $ErrLog"
