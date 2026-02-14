param(
  [switch]$HardKill
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

$metaPath = Join-Path $root 'data\matrix\runs\active_matrix.json'
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
  New-Item -ItemType File -Force -Path $graceful | Out-Null
}

Start-Sleep -Seconds 8

foreach ($item in $items) {
  $procId = [int]($item.pid)
  if ($procId -le 0) { continue }
  $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
  if (-not $proc) { continue }
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
  $pidText = (Get-Content $lf.FullName -Raw -ErrorAction SilentlyContinue).Trim()
  $procId = 0
  [void][int]::TryParse($pidText, [ref]$procId)
  if ($procId -le 0) {
    Remove-Item -Force -ErrorAction SilentlyContinue $lf.FullName
    continue
  }
  $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
  if (-not $proc) {
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
