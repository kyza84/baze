$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$sessionsDir = Join-Path $root 'logs\matrix'
if (-not (Test-Path $sessionsDir)) {
  Write-Host "No matrix logs yet: $sessionsDir"
  exit 0
}

Get-ChildItem -Path $sessionsDir -Directory | ForEach-Object {
  $id = $_.Name
  $tradeDecisionFile = Join-Path $_.FullName 'trade_decisions.jsonl'
  if (Test-Path $tradeDecisionFile) {
    $buy = 0
    $sell = 0
    $pnl = 0.0
    Get-Content -Path $tradeDecisionFile -Encoding UTF8 -ErrorAction SilentlyContinue | ForEach-Object {
      $line = [string]$_
      if ([string]::IsNullOrWhiteSpace($line)) { return }
      try {
        $row = $line | ConvertFrom-Json -ErrorAction Stop
      } catch {
        return
      }
      $decision = [string]$row.decision
      if ($decision -eq 'open') {
        $buy += 1
      } elseif ($decision -eq 'close') {
        $sell += 1
        try { $pnl += [double]$row.pnl_usd } catch {}
      }
    }
    "{0}`tbuys={1}`tsells={2}`trealized={3:N2}`tfile={4}" -f $id,$buy,$sell,$pnl,'trade_decisions.jsonl'
    return
  }

  $latest = Get-ChildItem -Path $_.FullName -Recurse -Filter 'main_local_*.log' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (-not $latest) { return }
  $buy = (Select-String -Path $latest.FullName -Pattern 'AUTO_BUY Paper BUY').Count
  $sellLines = Select-String -Path $latest.FullName -Pattern 'AUTO_SELL Paper SELL'
  $sell = $sellLines.Count
  $pnl = 0.0
  foreach ($line in $sellLines) {
    if ($line.Line -match 'pnl=[-+0-9.]+% \(\$([-+0-9.]+)\)') {
      $pnl += [double]$matches[1]
    }
  }
  "{0}`tbuys={1}`tsells={2}`trealized={3:N2}`tfile={4}" -f $id,$buy,$sell,$pnl,$latest.Name
}
