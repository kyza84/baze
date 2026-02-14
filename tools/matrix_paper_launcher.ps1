param(
  [ValidateSet('1','2','3','4')]
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
# - mx1_refine: single-wallet fast-eval profile (priority: enough entries + tighter loss exits).
# - mx2_explore_wide: cautious-explore profile (priority: controlled flow without self-choke).
# - mx3_explore_timeout / mx4_explore_momentum: optional extra probes for Count=3/4.
$variants = @(
  @{ name='mx1_refine'; overrides=@{
      ADAPTIVE_FILTERS_ENABLED='true';
      ADAPTIVE_FILTERS_MODE='apply';
      ADAPTIVE_FILTERS_INTERVAL_SECONDS='600';
      ADAPTIVE_FILTERS_COOLDOWN_WINDOWS='1';
      ADAPTIVE_FILTERS_TARGET_CAND_MIN='2.0';
      ADAPTIVE_FILTERS_TARGET_CAND_MAX='10.0';
      ADAPTIVE_FILTERS_TARGET_OPEN_MIN='0.08';
      ADAPTIVE_ZERO_OPEN_RESET_ENABLED='true';
      ADAPTIVE_ZERO_OPEN_WINDOWS_BEFORE_RESET='1';
      ADAPTIVE_ZERO_OPEN_MIN_CANDIDATES='1.0';
      ADAPTIVE_SCORE_MIN='56';
      ADAPTIVE_SCORE_MAX='64';
      ADAPTIVE_SAFE_VOLUME_MIN='90';
      ADAPTIVE_SAFE_VOLUME_MAX='160';
      ADAPTIVE_EDGE_MIN='1.0';
      ADAPTIVE_EDGE_MAX='1.4';
      ADAPTIVE_DEDUP_TTL_MIN='60';
      ADAPTIVE_DEDUP_TTL_MAX='180';
      ADAPTIVE_DEDUP_RELAX_ENABLED='true';
      ADAPTIVE_DEDUP_DYNAMIC_ENABLED='true';
      ADAPTIVE_DEDUP_DYNAMIC_MIN='8';
      ADAPTIVE_DEDUP_DYNAMIC_MAX='30';
      ADAPTIVE_DEDUP_DYNAMIC_TARGET_PERCENTILE='90';
      ADAPTIVE_DEDUP_DYNAMIC_FACTOR='1.2';
      ADAPTIVE_DEDUP_DYNAMIC_MIN_SAMPLES='150';
      AUTO_TRADE_ENTRY_MODE='top_n';
      AUTO_TRADE_TOP_N='6';
      PROFIT_LOCK_TRIGGER_PERCENT='5';
      PROFIT_LOCK_FLOOR_PERCENT='1.2';
      NO_MOMENTUM_EXIT_MIN_AGE_PERCENT='12';
      NO_MOMENTUM_EXIT_MAX_PNL_PERCENT='0.3';
      PAPER_PARTIAL_TP_ENABLED='true';
      PAPER_PARTIAL_TP_TRIGGER_PERCENT='2.0';
      PAPER_PARTIAL_TP_SELL_FRACTION='0.30';
      PAPER_PARTIAL_TP_MOVE_SL_TO_BREAK_EVEN='true';
      PAPER_PARTIAL_TP_BREAK_EVEN_BUFFER_PERCENT='0.10';
      WEAKNESS_EXIT_MIN_AGE_PERCENT='12';
      WEAKNESS_EXIT_PNL_PERCENT='-2.0';
      PAPER_MAX_HOLD_SECONDS='240';
      MIN_EXPECTED_EDGE_PERCENT='1.10';
      MIN_TOKEN_SCORE='61';
      SAFE_MIN_VOLUME_5M_USD='100';
      SAFE_MIN_LIQUIDITY_USD='5000';
      HEAVY_CHECK_DEDUP_TTL_SECONDS='120';
      STOP_LOSS_PERCENT='4';
      MAX_BUYS_PER_HOUR='16';
      MAX_OPEN_TRADES='2';
      PAPER_TRADE_SIZE_MIN_USD='0.55';
      PAPER_TRADE_SIZE_MAX_USD='1.00';
      ENTRY_REQUIRE_POSITIVE_CHANGE_5M='true';
      ENTRY_MIN_PRICE_CHANGE_5M_PERCENT='-0.10';
      WALLET_BALANCE_USD='7.00';
      DYNAMIC_HOLD_ENABLED='true';
      HOLD_MIN_SECONDS='60';
      HOLD_MAX_SECONDS='210';
    }
  },
  @{ name='mx2_explore_wide'; overrides=@{
      ADAPTIVE_FILTERS_ENABLED='true';
      ADAPTIVE_FILTERS_MODE='apply';
      ADAPTIVE_FILTERS_INTERVAL_SECONDS='600';
      ADAPTIVE_FILTERS_COOLDOWN_WINDOWS='1';
      ADAPTIVE_FILTERS_TARGET_CAND_MIN='1.2';
      ADAPTIVE_FILTERS_TARGET_CAND_MAX='8.0';
      ADAPTIVE_FILTERS_TARGET_OPEN_MIN='0.04';
      ADAPTIVE_ZERO_OPEN_RESET_ENABLED='true';
      ADAPTIVE_ZERO_OPEN_WINDOWS_BEFORE_RESET='1';
      ADAPTIVE_ZERO_OPEN_MIN_CANDIDATES='1.0';
      ADAPTIVE_SCORE_MIN='57';
      ADAPTIVE_SCORE_MAX='62';
      ADAPTIVE_SAFE_VOLUME_MIN='120';
      ADAPTIVE_SAFE_VOLUME_MAX='170';
      ADAPTIVE_EDGE_MIN='1.0';
      ADAPTIVE_EDGE_MAX='1.5';
      ADAPTIVE_DEDUP_TTL_MIN='60';
      ADAPTIVE_DEDUP_TTL_MAX='240';
      ADAPTIVE_DEDUP_RELAX_ENABLED='true';
      MIN_TOKEN_SCORE='60';
      SAFE_MIN_VOLUME_5M_USD='120';
      SAFE_MIN_LIQUIDITY_USD='4000';
      MIN_EXPECTED_EDGE_PERCENT='1.15';
      PROFIT_LOCK_TRIGGER_PERCENT='6';
      PROFIT_LOCK_FLOOR_PERCENT='0.9';
      PAPER_MAX_HOLD_SECONDS='270';
      NO_MOMENTUM_EXIT_MIN_AGE_PERCENT='16';
      NO_MOMENTUM_EXIT_MAX_PNL_PERCENT='0.4';
      WEAKNESS_EXIT_MIN_AGE_PERCENT='16';
      WEAKNESS_EXIT_PNL_PERCENT='-2.8';
      MAX_BUYS_PER_HOUR='12';
      MAX_OPEN_TRADES='2';
      STOP_LOSS_PERCENT='5';
      PAPER_TRADE_SIZE_MIN_USD='0.70';
      PAPER_TRADE_SIZE_MAX_USD='1.25';
      ENTRY_REQUIRE_POSITIVE_CHANGE_5M='true';
      ENTRY_MIN_PRICE_CHANGE_5M_PERCENT='0.05';
      WALLET_BALANCE_USD='7.00';
      DYNAMIC_HOLD_ENABLED='true';
      HOLD_MIN_SECONDS='60';
      HOLD_MAX_SECONDS='220';
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
$selected = @($variants | Select-Object -First $take)
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

