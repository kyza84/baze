param(
  [ValidateSet('2','3','4')]
  [string]$Count = '2',
  [switch]$Run
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  throw "Python not found: $python"
}

$matrixDir = Join-Path $root 'data\matrix'
$envDir = Join-Path $matrixDir 'env'
$runDir = Join-Path $matrixDir 'runs'
New-Item -ItemType Directory -Force -Path $matrixDir, $envDir, $runDir | Out-Null

if ($Run) {
  $stopper = Join-Path $PSScriptRoot 'matrix_paper_stop.ps1'
  if (Test-Path $stopper) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $stopper -HardKill | Out-Null
    Start-Sleep -Seconds 1
  }
}

# Variant matrix:
# - mx1_refine: close to current working profile with small exit corrections.
# - mx2_explore_wide: wider parameter spread for faster discovery.
# - mx3_explore_timeout / mx4_explore_momentum: optional extra probes for Count=3/4.
$variants = @(
  @{ name='mx1_refine'; overrides=@{
      PROFIT_LOCK_TRIGGER_PERCENT='10';
      PROFIT_LOCK_FLOOR_PERCENT='1.7';
      NO_MOMENTUM_EXIT_MIN_AGE_PERCENT='22';
      NO_MOMENTUM_EXIT_MAX_PNL_PERCENT='0.6';
      WEAKNESS_EXIT_MIN_AGE_PERCENT='22';
      WEAKNESS_EXIT_PNL_PERCENT='-3.2';
      PAPER_MAX_HOLD_SECONDS='300';
      MIN_EXPECTED_EDGE_PERCENT='1.1';
      MIN_TOKEN_SCORE='62';
      SAFE_MIN_VOLUME_5M_USD='180';
      DYNAMIC_HOLD_ENABLED='true';
      HOLD_MIN_SECONDS='120';
      HOLD_MAX_SECONDS='420';
    }
  },
  @{ name='mx2_explore_wide'; overrides=@{
      MIN_TOKEN_SCORE='60';
      SAFE_MIN_VOLUME_5M_USD='140';
      SAFE_MIN_LIQUIDITY_USD='4000';
      MIN_EXPECTED_EDGE_PERCENT='0.9';
      PROFIT_LOCK_TRIGGER_PERCENT='9';
      PROFIT_LOCK_FLOOR_PERCENT='2.0';
      PAPER_MAX_HOLD_SECONDS='360';
      NO_MOMENTUM_EXIT_MIN_AGE_PERCENT='18';
      NO_MOMENTUM_EXIT_MAX_PNL_PERCENT='0.8';
      WEAKNESS_EXIT_MIN_AGE_PERCENT='20';
      WEAKNESS_EXIT_PNL_PERCENT='-3.0';
      MAX_BUYS_PER_HOUR='10';
      DYNAMIC_HOLD_ENABLED='true';
      HOLD_MIN_SECONDS='90';
      HOLD_MAX_SECONDS='330';
    }
  },
  @{ name='mx3_explore_timeout'; overrides=@{
      PAPER_MAX_HOLD_SECONDS='900';
      NO_MOMENTUM_EXIT_MIN_AGE_PERCENT='35';
      WEAKNESS_EXIT_MIN_AGE_PERCENT='35';
      WEAKNESS_EXIT_PNL_PERCENT='-6';
    }
  },
  @{ name='mx4_explore_momentum'; overrides=@{
      PROFIT_LOCK_TRIGGER_PERCENT='11';
      PROFIT_LOCK_FLOOR_PERCENT='2.0';
      PAPER_MAX_HOLD_SECONDS='900';
      NO_MOMENTUM_EXIT_MIN_AGE_PERCENT='35';
      NO_MOMENTUM_EXIT_MAX_PNL_PERCENT='0.3';
      ENTRY_REQUIRE_POSITIVE_CHANGE_5M='true';
      ENTRY_MIN_PRICE_CHANGE_5M_PERCENT='0.0';
    }
  }
)

$take = [int]$Count
$selected = $variants | Select-Object -First $take
$records = @()

for ($i = 0; $i -lt $selected.Count; $i++) {
  $v = $selected[$i]
  $id = $v.name
  $logDir = "logs/matrix/$id"
  $paperState = "trading/paper_state.$id.json"
  $blFile = "data/autotrade_blacklist.$id.json"
  $lastBlock = "data/last_block_base.$id.txt"
  $seenPairs = "data/seen_pairs_base.$id.json"
  $graceful = "data/graceful_stop.$id.signal"
  $candLog = "logs/matrix/$id/candidates.jsonl"
  $snapshotDir = "snapshots/$id"

  $envPath = Join-Path $envDir "$id.env"

  $lines = @(
    "BOT_INSTANCE_ID=$id",
    "RUN_TAG=$id",
    "LOG_DIR=$logDir",
    "PAPER_STATE_FILE=$paperState",
    "CANDIDATE_DECISIONS_LOG_FILE=$candLog",
    "SNAPSHOT_DIR=$snapshotDir",
    "CANDIDATE_SHARD_MOD=$take",
    "CANDIDATE_SHARD_SLOT=$i",
    "AUTOTRADE_BLACKLIST_FILE=$blFile",
    "ONCHAIN_LAST_BLOCK_FILE=$lastBlock",
    "ONCHAIN_SEEN_PAIRS_FILE=$seenPairs",
    "GRACEFUL_STOP_FILE=$graceful"
  )
  foreach ($k in $v.overrides.Keys) {
    $lines += "$k=$($v.overrides[$k])"
  }
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllLines($envPath, $lines, $utf8NoBom)
  $gracefulAbs = Join-Path $root $graceful
  if (Test-Path $gracefulAbs) {
    Remove-Item -Force -ErrorAction SilentlyContinue $gracefulAbs
  }

  $rec = [ordered]@{
    id = $id
    env_file = $envPath
    log_dir = $logDir
    paper_state_file = $paperState
    graceful_stop_file = $graceful
    overrides = $v.overrides
    pid = $null
  }

  if ($Run) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $python
    $psi.Arguments = 'main_local.py'
    $psi.WorkingDirectory = $root
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $false
    $psi.RedirectStandardError = $false
    $psi.EnvironmentVariables['BOT_ENV_FILE'] = $envPath
    $psi.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'
    $proc = [System.Diagnostics.Process]::Start($psi)
    if ($null -ne $proc) {
      $rec.pid = $proc.Id
    }
  }

  $records += [pscustomobject]$rec
}

$meta = [ordered]@{
  created_at = (Get-Date).ToString('s')
  count = $take
  running = [bool]$Run
  items = $records
}

$metaPath = Join-Path $runDir 'active_matrix.json'
$metaJson = $meta | ConvertTo-Json -Depth 8
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($metaPath, $metaJson, $utf8NoBom)

Write-Host "Matrix prepared: $metaPath"
if ($Run) {
  Write-Host "Started instances:"
  $records | ForEach-Object { Write-Host ("  {0} pid={1}" -f $_.id, $_.pid) }
} else {
  Write-Host "Use -Run to start instances."
}
