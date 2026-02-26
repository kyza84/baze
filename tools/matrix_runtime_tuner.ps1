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
  [int]$RollbackDegradeStreak = 3,
  [double]$HoldHysteresisOpenRate = 0.07,
  [double]$HoldHysteresisTradesPerHour = 6.0,
  [switch]$DryRun,
  [switch]$Json
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

if (-not $ProfileId -or $ProfileId.Trim().Length -eq 0) {
  throw "ProfileId is required."
}
$profile = $ProfileId.Trim()

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  $python = 'python'
}

$script = Join-Path $PSScriptRoot 'matrix_runtime_tuner.py'
$args = @($script, $Command, '--profile-id', $profile, '--root', $root)

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
  $args += @('--rollback-degrade-streak', [string]$RollbackDegradeStreak)
  $args += @('--hold-hysteresis-open-rate', [string]$HoldHysteresisOpenRate)
  $args += @('--hold-hysteresis-trades-per-hour', [string]$HoldHysteresisTradesPerHour)
  if ($Command -eq 'run') {
    $args += @('--duration-minutes', [string]$DurationMinutes)
    $args += @('--interval-seconds', [string]$IntervalSeconds)
  }
  if ($DryRun) {
    $args += '--dry-run'
  }
}

& $python @args
