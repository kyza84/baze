param(
  [switch]$HardKill
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

$metaPath = Join-Path $root 'data\matrix\runs\active_matrix.json'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Test-MainLocalProcess {
  param(
    [int]$ProcessId
  )
  if ($ProcessId -le 0) { return $false }
  try {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
    if (-not $proc) { return $false }
    $cmd = [string]$proc.CommandLine
    if ($cmd -notmatch 'main_local\.py') { return $false }
    return $true
  } catch {
    return $false
  }
}

$items = @()
if (Test-Path $metaPath) {
  try {
    $raw = Get-Content $metaPath -Raw | ConvertFrom-Json
    $items = @($raw.items)
  } catch {
    $items = @()
  }
}

foreach ($item in $items) {
  $graceful = Join-Path $root ($item.graceful_stop_file)
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $graceful) | Out-Null
  $payload = [ordered]@{
    ts = [double]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())
    timestamp = (Get-Date).ToUniversalTime().ToString('o')
    source = 'matrix_paper_stop.ps1'
    reason = $(if ($HardKill) { 'operator_stop_hard' } else { 'operator_stop' })
    actor = [string]$env:USERNAME
    host = [string]$env:COMPUTERNAME
    profile_id = [string]($item.id)
    hard_kill = [bool]$HardKill
  }
  $json = $payload | ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText($graceful, $json, $utf8NoBom)
}

Start-Sleep -Seconds 8

foreach ($item in $items) {
  $procId = 0
  try { $procId = [int]($item.pid) } catch { $procId = 0 }
  if ($procId -le 0) { continue }
  if (-not (Test-MainLocalProcess -ProcessId $procId)) { continue }
  if ($HardKill) {
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    Write-Host "Hard killed pid=$procId id=$($item.id)"
  } else {
    Write-Host "Still running pid=$procId id=$($item.id) (graceful signal sent)"
  }
}

# Fallback: stop orphan matrix processes discovered by lock files.
$lockFiles = Get-ChildItem -Path $root -Filter 'main_local.mx*.lock' -File -ErrorAction SilentlyContinue
foreach ($lf in $lockFiles) {
  $id = [System.IO.Path]::GetFileNameWithoutExtension($lf.Name).Replace('main_local.', '')
  $pidRaw = Get-Content $lf.FullName -Raw -ErrorAction SilentlyContinue
  $pidText = ""
  if ($null -ne $pidRaw) {
    $pidText = ([string]$pidRaw).Trim()
  }
  $procId = 0
  [void][int]::TryParse($pidText, [ref]$procId)
  if ($procId -le 0) {
    Remove-Item -Force -ErrorAction SilentlyContinue $lf.FullName
    continue
  }
  if (-not (Test-MainLocalProcess -ProcessId $procId)) {
    Remove-Item -Force -ErrorAction SilentlyContinue $lf.FullName
    continue
  }
  if ($HardKill) {
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    Write-Host "Hard killed orphan pid=$procId id=$id"
  } else {
    Write-Host "Orphan instance still running pid=$procId id=$id"
  }
}

# Rebuild matrix metadata from facts after stop.
$aliveCount = 0
$updatedItems = @()
foreach ($item in $items) {
  $procId = 0
  try { $procId = [int]($item.pid) } catch { $procId = 0 }
  $alive = $false
  if ($procId -gt 0) {
    $alive = Test-MainLocalProcess -ProcessId $procId
  }
  if ($alive) { $aliveCount += 1 }

  $updatedItems += [pscustomobject]([ordered]@{
      id = $item.id
      env_file = $item.env_file
      log_dir = $item.log_dir
      paper_state_file = $item.paper_state_file
      graceful_stop_file = $item.graceful_stop_file
      overrides = $item.overrides
      pid = $(if ($alive) { $procId } else { $null })
      status = $(if ($alive) { 'running' } else { 'stopped' })
    })
}

$meta = [ordered]@{
  updated_at = (Get-Date).ToString('s')
  stopped_at = (Get-Date).ToString('s')
  count = @($updatedItems).Count
  requested_run = $false
  running = [bool]($aliveCount -gt 0)
  alive_count = $aliveCount
  items = $updatedItems
}
$metaJson = $meta | ConvertTo-Json -Depth 8
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $metaPath) | Out-Null
[System.IO.File]::WriteAllText($metaPath, $metaJson, $utf8NoBom)
