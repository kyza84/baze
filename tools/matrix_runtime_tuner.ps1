param(
  [ValidateSet('once','run','replay')]
  [string]$Command = 'once',
  [string]$ProfileId = '',
  [ValidateSet('conveyor','fast','calm','sniper')]
  [string]$Mode = 'conveyor',
  [ValidateSet('auto','expand','hold','tighten')]
  [string]$PolicyPhase = 'auto',
  [int]$WindowMinutes = 12,
  [int]$RestartCooldownSeconds = 180,
  [int]$RestartMaxPerHour = 6,
  [int]$DurationMinutes = 60,
  [int]$IntervalSeconds = 120,
  [int]$Limit = 240,
  [string]$TargetPolicyFile = '',
  [double]$TargetTradesPerHour = 12.0,
  [double]$TargetPnlPerHourUsd = 0.05,
  [double]$MinOpenRate15m = 0.04,
  [int]$MinSelected15m = 16,
  [int]$MinClosedForRiskChecks = 6,
  [double]$MinWinrateClosed15m = 0.35,
  [double]$MaxBlacklistShare15m = 0.45,
  [int]$MaxBlacklistAdded15m = 80,
  [int]$PreRiskMinPlanAttempts15m = 8,
  [double]$PreRiskRouteFailRate15m = 0.35,
  [double]$PreRiskBuyFailRate15m = 0.35,
  [double]$PreRiskSellFailRate15m = 0.30,
  [double]$PreRiskRoundtripLossMedianPct15m = -1.2,
  [int]$TailLossMinCloses60m = 6,
  [double]$TailLossRatioMax = 8.0,
  [int]$RollbackDegradeStreak = 3,
  [double]$HoldHysteresisOpenRate = 0.07,
  [double]$HoldHysteresisTradesPerHour = 6.0,
  [switch]$AllowZeroCooldown,
  [switch]$DryRun,
  [switch]$Json
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

# Keep console output readable for Cyrillic/non-ASCII symbols.
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

if (-not $ProfileId -or $ProfileId.Trim().Length -eq 0) {
  throw "ProfileId is required."
}
$profile = $ProfileId.Trim()

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  $python = 'python'
}

$script = Join-Path $PSScriptRoot 'matrix_runtime_tuner.py'
$args = @('-u', $script, $Command, '--profile-id', $profile, '--root', $root)

$logDir = Join-Path $root ("logs\matrix\{0}" -f $profile)
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$consoleLog = Join-Path $logDir ("runtime_tuner_console_{0}.log" -f $stamp)

if ($Command -eq 'replay') {
  $args += @('--limit', [string]$Limit)
  if ($Json) {
    $args += '--json'
  }
} else {
  $args += @('--mode', $Mode)
  $args += @('--policy-phase', $PolicyPhase)
  $args += @('--window-minutes', [string]$WindowMinutes)
  $args += @('--restart-cooldown-seconds', [string]$RestartCooldownSeconds)
  $args += @('--restart-max-per-hour', [string]$RestartMaxPerHour)
  if ($TargetPolicyFile -and $TargetPolicyFile.Trim().Length -gt 0) {
    $args += @('--target-policy-file', $TargetPolicyFile.Trim())
  }
  $args += @('--target-trades-per-hour', [string]$TargetTradesPerHour)
  $args += @('--target-pnl-per-hour-usd', [string]$TargetPnlPerHourUsd)
  $args += @('--min-open-rate-15m', [string]$MinOpenRate15m)
  $args += @('--min-selected-15m', [string]$MinSelected15m)
  $args += @('--min-closed-for-risk-checks', [string]$MinClosedForRiskChecks)
  $args += @('--min-winrate-closed-15m', [string]$MinWinrateClosed15m)
  $args += @('--max-blacklist-share-15m', [string]$MaxBlacklistShare15m)
  $args += @('--max-blacklist-added-15m', [string]$MaxBlacklistAdded15m)
  $args += @('--pre-risk-min-plan-attempts-15m', [string]$PreRiskMinPlanAttempts15m)
  $args += @('--pre-risk-route-fail-rate-15m', [string]$PreRiskRouteFailRate15m)
  $args += @('--pre-risk-buy-fail-rate-15m', [string]$PreRiskBuyFailRate15m)
  $args += @('--pre-risk-sell-fail-rate-15m', [string]$PreRiskSellFailRate15m)
  $args += @('--pre-risk-roundtrip-loss-median-pct-15m', [string]$PreRiskRoundtripLossMedianPct15m)
  $args += @('--tail-loss-min-closes-60m', [string]$TailLossMinCloses60m)
  $args += @('--tail-loss-ratio-max', [string]$TailLossRatioMax)
  $args += @('--rollback-degrade-streak', [string]$RollbackDegradeStreak)
  $args += @('--hold-hysteresis-open-rate', [string]$HoldHysteresisOpenRate)
  $args += @('--hold-hysteresis-trades-per-hour', [string]$HoldHysteresisTradesPerHour)
  if ($AllowZeroCooldown) {
    $args += '--allow-zero-cooldown'
  }
  if ($Command -eq 'run') {
    $args += @('--duration-minutes', [string]$DurationMinutes)
    $args += @('--interval-seconds', [string]$IntervalSeconds)
  }
  if ($DryRun) {
    $args += '--dry-run'
  }
}

Write-Host ("[runtime_tuner] python {0}" -f (($args -join ' '))) -ForegroundColor Cyan
Write-Host ("[runtime_tuner] console_log {0}" -f $consoleLog) -ForegroundColor DarkCyan

# Mirror live tuner output to a plain-text console log for easy review.
& $python @args 2>&1 | Tee-Object -FilePath $consoleLog -Append
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
  throw ("[runtime_tuner] python exited with code {0}" -f $exitCode)
}
