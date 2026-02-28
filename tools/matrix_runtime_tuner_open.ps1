param(
  [string]$ProfileId = '',
  [ValidateSet('conveyor','fast','calm','sniper')]
  [string]$Mode = 'conveyor',
  [ValidateSet('auto','expand','hold','tighten')]
  [string]$PolicyPhase = 'auto',
  [int]$WindowMinutes = 12,
  [int]$RestartCooldownSeconds = 180,
  [int]$DurationMinutes = 600,
  [int]$IntervalSeconds = 120,
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
  [string]$TargetPolicyFile = '',
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

if (-not $ProfileId -or $ProfileId.Trim().Length -eq 0) {
  throw "ProfileId is required."
}

$runner = Join-Path $PSScriptRoot 'matrix_runtime_tuner.ps1'
$cmdArgs = @(
  '-NoProfile',
  '-NoExit',
  '-ExecutionPolicy', 'Bypass',
  '-File', $runner,
  '-Command', 'run',
  '-ProfileId', $ProfileId.Trim(),
  '-Mode', $Mode,
  '-PolicyPhase', $PolicyPhase,
  '-WindowMinutes', [string]$WindowMinutes,
  '-RestartCooldownSeconds', [string]$RestartCooldownSeconds,
  '-DurationMinutes', [string]$DurationMinutes,
  '-IntervalSeconds', [string]$IntervalSeconds,
  '-TargetTradesPerHour', [string]$TargetTradesPerHour,
  '-TargetPnlPerHourUsd', [string]$TargetPnlPerHourUsd,
  '-MinOpenRate15m', [string]$MinOpenRate15m,
  '-MinSelected15m', [string]$MinSelected15m,
  '-MinClosedForRiskChecks', [string]$MinClosedForRiskChecks,
  '-MinWinrateClosed15m', [string]$MinWinrateClosed15m,
  '-MaxBlacklistShare15m', [string]$MaxBlacklistShare15m,
  '-MaxBlacklistAdded15m', [string]$MaxBlacklistAdded15m,
  '-PreRiskMinPlanAttempts15m', [string]$PreRiskMinPlanAttempts15m,
  '-PreRiskRouteFailRate15m', [string]$PreRiskRouteFailRate15m,
  '-PreRiskBuyFailRate15m', [string]$PreRiskBuyFailRate15m,
  '-PreRiskSellFailRate15m', [string]$PreRiskSellFailRate15m,
  '-PreRiskRoundtripLossMedianPct15m', [string]$PreRiskRoundtripLossMedianPct15m,
  '-TailLossMinCloses60m', [string]$TailLossMinCloses60m,
  '-TailLossRatioMax', [string]$TailLossRatioMax,
  '-RollbackDegradeStreak', [string]$RollbackDegradeStreak,
  '-HoldHysteresisOpenRate', [string]$HoldHysteresisOpenRate,
  '-HoldHysteresisTradesPerHour', [string]$HoldHysteresisTradesPerHour
)

if ($TargetPolicyFile -and $TargetPolicyFile.Trim().Length -gt 0) {
  $cmdArgs += @('-TargetPolicyFile', $TargetPolicyFile.Trim())
}
if ($DryRun) {
  $cmdArgs += '-DryRun'
}

Write-Host "[runtime_tuner] opening visible window..." -ForegroundColor Cyan
Start-Process -FilePath "powershell" -ArgumentList $cmdArgs -WorkingDirectory $root | Out-Null
Write-Host "[runtime_tuner] window started for profile '$ProfileId'" -ForegroundColor Green
