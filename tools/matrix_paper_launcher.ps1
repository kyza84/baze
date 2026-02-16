param(
  [ValidateSet('1','2','3','4')]
  [string]$Count = '2',
  [string[]]$ProfileIds = @(),
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
  # Ensure we do not mix matrix with stale single/main_local workers.
  $existing = Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { [string]$_.CommandLine -match 'main_local\.py' } |
    Select-Object -ExpandProperty ProcessId
  foreach ($procId in @($existing)) {
    try { Stop-Process -Id ([int]$procId) -Force -ErrorAction SilentlyContinue } catch {}
  }
  Start-Sleep -Milliseconds 800
}

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

# Variant matrix:
# - mx1_flow_balanced: stable high-flow profile (more quality guards, steady exits).
# - mx2_flow_aggressive: higher-throughput profile (more entries, faster recycle).
# - mx3_flow_compound / mx4_flow_guarded: winner-derived forks for Count=3/4.
$variants = @(
  @{ name='mx1_flow_balanced'; overrides=@{
      ADAPTIVE_FILTERS_ENABLED='true';
      ADAPTIVE_FILTERS_MODE='apply';
      ADAPTIVE_FILTERS_INTERVAL_SECONDS='240';
      ADAPTIVE_FILTERS_MIN_WINDOW_CYCLES='2';
      ADAPTIVE_FILTERS_COOLDOWN_WINDOWS='0';
      ADAPTIVE_FILTERS_TARGET_CAND_MIN='1.8';
      ADAPTIVE_FILTERS_TARGET_CAND_MAX='10.0';
      ADAPTIVE_FILTERS_TARGET_OPEN_MIN='0.15';
      ADAPTIVE_ZERO_OPEN_RESET_ENABLED='true';
      ADAPTIVE_ZERO_OPEN_WINDOWS_BEFORE_RESET='1';
      ADAPTIVE_ZERO_OPEN_MIN_CANDIDATES='0.6';
      ADAPTIVE_SCORE_MIN='50';
      ADAPTIVE_SCORE_MAX='58';
      ADAPTIVE_SAFE_VOLUME_MIN='45';
      ADAPTIVE_SAFE_VOLUME_MAX='160';
      ADAPTIVE_EDGE_MIN='0.75';
      ADAPTIVE_EDGE_MAX='1.30';
      ADAPTIVE_DEDUP_TTL_MIN='20';
      ADAPTIVE_DEDUP_TTL_MAX='90';
      ADAPTIVE_DEDUP_RELAX_ENABLED='true';
      ADAPTIVE_DEDUP_DYNAMIC_ENABLED='true';
      ADAPTIVE_DEDUP_DYNAMIC_MIN='8';
      ADAPTIVE_DEDUP_DYNAMIC_MAX='24';
      ADAPTIVE_DEDUP_DYNAMIC_TARGET_PERCENTILE='90';
      ADAPTIVE_DEDUP_DYNAMIC_FACTOR='1.15';
      ADAPTIVE_DEDUP_DYNAMIC_MIN_SAMPLES='150';
      WALLET_MODE='paper';
      AUTO_TRADE_ENABLED='true';
      AUTO_TRADE_PAPER='true';
      AUTO_TRADE_EXCLUDED_SYMBOLS='ZORA';
      ONCHAIN_ENABLE_UNISWAP_V3='true';
      DEX_SEARCH_QUERIES='base,eth,usdc,cbbtc,aero,virtual,bridged,ai,defi,memes';
      DEX_BOOSTS_MAX_TOKENS='35';
      AUTO_TRADE_ENTRY_MODE='top_n';
      AUTO_TRADE_TOP_N='10';
      AUTONOMOUS_CONTROL_ENABLED='true';
      AUTONOMOUS_CONTROL_MODE='apply';
      AUTONOMOUS_CONTROL_INTERVAL_SECONDS='300';
      AUTONOMOUS_CONTROL_MIN_WINDOW_CYCLES='2';
      AUTONOMOUS_CONTROL_TARGET_CANDIDATES_MIN='1.5';
      AUTONOMOUS_CONTROL_TARGET_CANDIDATES_HIGH='6.0';
      AUTONOMOUS_CONTROL_TARGET_OPENED_MIN='0.15';
      AUTONOMOUS_CONTROL_NEG_REALIZED_TRIGGER_USD='0.05';
      AUTONOMOUS_CONTROL_POS_REALIZED_TRIGGER_USD='0.05';
      AUTONOMOUS_CONTROL_MAX_LOSS_STREAK_TRIGGER='3';
      AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MIN='2';
      AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MAX='4';
      AUTONOMOUS_CONTROL_TOP_N_MIN='8';
      AUTONOMOUS_CONTROL_TOP_N_MAX='14';
      AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN='20';
      AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MAX='48';
      AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MIN='0.65';
      AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MAX='1.05';
      AUTONOMOUS_CONTROL_RISK_OFF_OPEN_TRADES_CAP='2';
      AUTONOMOUS_CONTROL_RISK_OFF_TOP_N_CAP='8';
      AUTONOMOUS_CONTROL_RISK_OFF_MAX_BUYS_PER_HOUR_CAP='24';
      AUTONOMOUS_CONTROL_RISK_OFF_TRADE_SIZE_MAX_CAP='0.70';
      PROFIT_LOCK_TRIGGER_PERCENT='3.5';
      PROFIT_LOCK_FLOOR_PERCENT='1.0';
      NO_MOMENTUM_EXIT_MIN_AGE_PERCENT='14';
      NO_MOMENTUM_EXIT_MAX_PNL_PERCENT='0.35';
      PAPER_PARTIAL_TP_ENABLED='true';
      PAPER_PARTIAL_TP_TRIGGER_PERCENT='1.8';
      PAPER_PARTIAL_TP_SELL_FRACTION='0.30';
      PAPER_PARTIAL_TP_MOVE_SL_TO_BREAK_EVEN='true';
      PAPER_PARTIAL_TP_BREAK_EVEN_BUFFER_PERCENT='0.10';
      WEAKNESS_EXIT_MIN_AGE_PERCENT='18';
      WEAKNESS_EXIT_PNL_PERCENT='-2.8';
      PAPER_MAX_HOLD_SECONDS='240';
      MIN_EXPECTED_EDGE_PERCENT='0.60';
      MIN_EXPECTED_EDGE_USD='0.015';
      EDGE_FILTER_MODE='both';
      MIN_TOKEN_SCORE='52';
      SAFE_TEST_MODE='true';
      SAFE_MIN_VOLUME_5M_USD='60';
      SAFE_MIN_LIQUIDITY_USD='6000';
      SAFE_MIN_AGE_SECONDS='240';
      SAFE_REQUIRE_RISK_LEVEL='MEDIUM';
      SAFE_REQUIRE_CONTRACT_SAFE='true';
      SAFE_MAX_WARNING_FLAGS='1';
      TOKEN_SAFETY_FAIL_CLOSED='true';
      HONEYPOT_API_ENABLED='true';
      HONEYPOT_API_FAIL_CLOSED='true';
      HEAVY_CHECK_DEDUP_TTL_SECONDS='25';
      STOP_LOSS_PERCENT='4';
      MAX_BUYS_PER_HOUR='36';
      MAX_TOKEN_COOLDOWN_SECONDS='90';
      RISK_GOVERNOR_MAX_LOSS_STREAK='4';
      RISK_GOVERNOR_STREAK_PAUSE_SECONDS='300';
      RISK_GOVERNOR_DRAWDOWN_LIMIT_PERCENT='5';
      RISK_GOVERNOR_DRAWDOWN_PAUSE_SECONDS='1200';
      PNL_BREAKEVEN_EPSILON_USD='0.001';
      MAX_OPEN_TRADES='3';
      PAPER_TRADE_SIZE_MIN_USD='0.50';
      PAPER_TRADE_SIZE_MAX_USD='0.95';
      AUTOTRADE_BLACKLIST_TTL_SECONDS='10800';
      PAPER_RESET_ON_START='true';
      ENTRY_REQUIRE_POSITIVE_CHANGE_5M='false';
      ENTRY_MIN_PRICE_CHANGE_5M_PERCENT='-0.25';
      ENTRY_REQUIRE_VOLUME_BUFFER='true';
      ENTRY_MIN_VOLUME_5M_MULT='1.10';
      WALLET_BALANCE_USD='7.00';
      DYNAMIC_HOLD_ENABLED='true';
      HOLD_MIN_SECONDS='45';
      HOLD_MAX_SECONDS='210';
    }
  },
  @{ name='mx2_flow_aggressive'; overrides=@{
      ADAPTIVE_FILTERS_ENABLED='true';
      ADAPTIVE_FILTERS_MODE='apply';
      ADAPTIVE_FILTERS_INTERVAL_SECONDS='180';
      ADAPTIVE_FILTERS_COOLDOWN_WINDOWS='0';
      ADAPTIVE_FILTERS_TARGET_CAND_MIN='2.5';
      ADAPTIVE_FILTERS_TARGET_CAND_MAX='16.0';
      ADAPTIVE_FILTERS_TARGET_OPEN_MIN='0.22';
      ADAPTIVE_ZERO_OPEN_RESET_ENABLED='true';
      ADAPTIVE_ZERO_OPEN_WINDOWS_BEFORE_RESET='1';
      ADAPTIVE_ZERO_OPEN_MIN_CANDIDATES='0.4';
      ADAPTIVE_SCORE_MIN='50';
      ADAPTIVE_SCORE_MAX='56';
      ADAPTIVE_SAFE_VOLUME_MIN='30';
      ADAPTIVE_SAFE_VOLUME_MAX='130';
      ADAPTIVE_EDGE_MIN='0.60';
      ADAPTIVE_EDGE_MAX='1.05';
      ADAPTIVE_DEDUP_TTL_MIN='12';
      ADAPTIVE_DEDUP_TTL_MAX='45';
      ADAPTIVE_DEDUP_RELAX_ENABLED='true';
      ADAPTIVE_DEDUP_DYNAMIC_ENABLED='true';
      ADAPTIVE_DEDUP_DYNAMIC_MIN='6';
      ADAPTIVE_DEDUP_DYNAMIC_MAX='16';
      ADAPTIVE_DEDUP_DYNAMIC_TARGET_PERCENTILE='90';
      ADAPTIVE_DEDUP_DYNAMIC_FACTOR='1.20';
      ADAPTIVE_DEDUP_DYNAMIC_MIN_SAMPLES='120';
      WALLET_MODE='paper';
      AUTO_TRADE_ENABLED='true';
      AUTO_TRADE_PAPER='true';
      AUTO_TRADE_EXCLUDED_SYMBOLS='ZORA';
      ONCHAIN_ENABLE_UNISWAP_V3='true';
      DEX_SEARCH_QUERIES='base,eth,usdc,cbbtc,aero,virtual';
      AUTO_TRADE_ENTRY_MODE='top_n';
      AUTO_TRADE_TOP_N='12';
      AUTONOMOUS_CONTROL_ENABLED='true';
      AUTONOMOUS_CONTROL_MODE='apply';
      AUTONOMOUS_CONTROL_INTERVAL_SECONDS='240';
      AUTONOMOUS_CONTROL_MIN_WINDOW_CYCLES='2';
      AUTONOMOUS_CONTROL_TARGET_CANDIDATES_MIN='2.0';
      AUTONOMOUS_CONTROL_TARGET_CANDIDATES_HIGH='8.5';
      AUTONOMOUS_CONTROL_TARGET_OPENED_MIN='0.20';
      AUTONOMOUS_CONTROL_NEG_REALIZED_TRIGGER_USD='0.06';
      AUTONOMOUS_CONTROL_POS_REALIZED_TRIGGER_USD='0.06';
      AUTONOMOUS_CONTROL_MAX_LOSS_STREAK_TRIGGER='3';
      AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MIN='2';
      AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MAX='5';
      AUTONOMOUS_CONTROL_TOP_N_MIN='8';
      AUTONOMOUS_CONTROL_TOP_N_MAX='16';
      AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN='24';
      AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MAX='72';
      AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MIN='0.60';
      AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MAX='0.95';
      AUTONOMOUS_CONTROL_RISK_OFF_OPEN_TRADES_CAP='2';
      AUTONOMOUS_CONTROL_RISK_OFF_TOP_N_CAP='8';
      AUTONOMOUS_CONTROL_RISK_OFF_MAX_BUYS_PER_HOUR_CAP='24';
      AUTONOMOUS_CONTROL_RISK_OFF_TRADE_SIZE_MAX_CAP='0.70';
      PROFIT_LOCK_TRIGGER_PERCENT='2.4';
      PROFIT_LOCK_FLOOR_PERCENT='0.8';
      NO_MOMENTUM_EXIT_MIN_AGE_PERCENT='10';
      NO_MOMENTUM_EXIT_MAX_PNL_PERCENT='0.30';
      PAPER_PARTIAL_TP_ENABLED='true';
      PAPER_PARTIAL_TP_TRIGGER_PERCENT='1.1';
      PAPER_PARTIAL_TP_SELL_FRACTION='0.40';
      PAPER_PARTIAL_TP_MOVE_SL_TO_BREAK_EVEN='true';
      PAPER_PARTIAL_TP_BREAK_EVEN_BUFFER_PERCENT='0.10';
      WEAKNESS_EXIT_MIN_AGE_PERCENT='14';
      WEAKNESS_EXIT_PNL_PERCENT='-2.9';
      PAPER_MAX_HOLD_SECONDS='160';
      MIN_EXPECTED_EDGE_PERCENT='0.60';
      MIN_EXPECTED_EDGE_USD='0.015';
      EDGE_FILTER_MODE='both';
      MIN_TOKEN_SCORE='50';
      SAFE_TEST_MODE='true';
      SAFE_MIN_VOLUME_5M_USD='55';
      SAFE_MIN_LIQUIDITY_USD='6000';
      SAFE_MIN_AGE_SECONDS='210';
      SAFE_REQUIRE_RISK_LEVEL='MEDIUM';
      SAFE_REQUIRE_CONTRACT_SAFE='true';
      SAFE_MAX_WARNING_FLAGS='1';
      TOKEN_SAFETY_FAIL_CLOSED='true';
      HONEYPOT_API_ENABLED='true';
      HONEYPOT_API_FAIL_CLOSED='true';
      HEAVY_CHECK_DEDUP_TTL_SECONDS='12';
      STOP_LOSS_PERCENT='4';
      MAX_BUYS_PER_HOUR='56';
      MAX_TOKEN_COOLDOWN_SECONDS='60';
      RISK_GOVERNOR_MAX_LOSS_STREAK='4';
      RISK_GOVERNOR_STREAK_PAUSE_SECONDS='300';
      RISK_GOVERNOR_DRAWDOWN_LIMIT_PERCENT='5';
      RISK_GOVERNOR_DRAWDOWN_PAUSE_SECONDS='900';
      PNL_BREAKEVEN_EPSILON_USD='0.001';
      MAX_OPEN_TRADES='4';
      PAPER_TRADE_SIZE_MIN_USD='0.45';
      PAPER_TRADE_SIZE_MAX_USD='0.85';
      AUTOTRADE_BLACKLIST_TTL_SECONDS='10800';
      PAPER_RESET_ON_START='true';
      ENTRY_REQUIRE_POSITIVE_CHANGE_5M='false';
      ENTRY_MIN_PRICE_CHANGE_5M_PERCENT='-0.25';
      ENTRY_REQUIRE_VOLUME_BUFFER='true';
      ENTRY_MIN_VOLUME_5M_MULT='1.00';
      WALLET_BALANCE_USD='7.00';
      DYNAMIC_HOLD_ENABLED='true';
      HOLD_MIN_SECONDS='30';
      HOLD_MAX_SECONDS='150';
    }
  },
  @{ name='mx3_flow_compound'; base='mx1_flow_balanced'; overrides=@{
      ADAPTIVE_FILTERS_INTERVAL_SECONDS='210';
      ADAPTIVE_FILTERS_TARGET_CAND_MIN='1.6';
      ADAPTIVE_FILTERS_TARGET_CAND_MAX='12.5';
      ADAPTIVE_SAFE_VOLUME_MIN='32';
      ADAPTIVE_EDGE_MIN='0.60';
      ADAPTIVE_DEDUP_TTL_MIN='8';
      ADAPTIVE_DEDUP_TTL_MAX='40';
      ADAPTIVE_DEDUP_DYNAMIC_MIN='6';
      ADAPTIVE_DEDUP_DYNAMIC_MAX='20';
      AUTO_TRADE_TOP_N='15';
      AUTONOMOUS_CONTROL_INTERVAL_SECONDS='240';
      AUTONOMOUS_CONTROL_TARGET_CANDIDATES_MIN='1.1';
      AUTONOMOUS_CONTROL_TARGET_CANDIDATES_HIGH='7.5';
      AUTONOMOUS_CONTROL_TARGET_OPENED_MIN='0.10';
      AUTONOMOUS_CONTROL_NEG_REALIZED_TRIGGER_USD='0.07';
      AUTONOMOUS_CONTROL_POS_REALIZED_TRIGGER_USD='0.06';
      AUTONOMOUS_CONTROL_MAX_LOSS_STREAK_TRIGGER='4';
      AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN='32';
      AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MAX='64';
      AUTONOMOUS_CONTROL_TOP_N_MAX='18';
      AUTONOMOUS_CONTROL_TOP_N_MIN='10';
      AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MAX='4';
      AUTONOMOUS_CONTROL_ANTI_STALL_ENABLED='true';
      AUTONOMOUS_CONTROL_ANTI_STALL_MIN_CANDIDATES='0.9';
      AUTONOMOUS_CONTROL_ANTI_STALL_MIN_UTILIZATION='0.80';
      AUTONOMOUS_CONTROL_ANTI_STALL_LIMIT_SKIP_MIN='1';
      AUTONOMOUS_CONTROL_ANTI_STALL_EXPAND_MULT='1.5';
      AUTONOMOUS_CONTROL_RECOVERY_ENABLED='true';
      AUTONOMOUS_CONTROL_RECOVERY_MIN_CANDIDATES='0.9';
      AUTONOMOUS_CONTROL_RECOVERY_EXPAND_MULT='1.4';
      AUTONOMOUS_CONTROL_FRAGILE_TIGHTEN_ENABLED='true';
      PROFIT_LOCK_TRIGGER_PERCENT='2.9';
      PROFIT_LOCK_FLOOR_PERCENT='0.95';
      NO_MOMENTUM_EXIT_MIN_AGE_PERCENT='12';
      PAPER_PARTIAL_TP_TRIGGER_PERCENT='1.4';
      PAPER_PARTIAL_TP_SELL_FRACTION='0.34';
      WEAKNESS_EXIT_MIN_AGE_PERCENT='16';
      WEAKNESS_EXIT_PNL_PERCENT='-3.0';
      PAPER_MAX_HOLD_SECONDS='195';
      MIN_TOKEN_SCORE='50';
      SAFE_MIN_VOLUME_5M_USD='44';
      MAX_BUYS_PER_HOUR='58';
      MAX_OPEN_TRADES='4';
      PAPER_TRADE_SIZE_MIN_USD='0.50';
      PAPER_TRADE_SIZE_MAX_USD='0.90';
      ENTRY_MIN_VOLUME_5M_MULT='1.05';
      HOLD_MIN_SECONDS='35';
      HOLD_MAX_SECONDS='175';
      HEAVY_CHECK_DEDUP_TTL_SECONDS='10';
      MAX_TOKEN_COOLDOWN_SECONDS='180';
      DATA_POLICY_ENTER_STREAK='2';
      DATA_POLICY_EXIT_STREAK='1';
      MARKET_REGIME_FAIL_CLOSED_RATIO='32';
      MARKET_REGIME_SOURCE_ERROR_PERCENT='28';
      AUTONOMOUS_CONTROL_RISK_OFF_OPEN_TRADES_CAP='3';
      AUTONOMOUS_CONTROL_RISK_OFF_MAX_BUYS_PER_HOUR_CAP='34';
      RISK_GOVERNOR_MAX_LOSS_STREAK='5';
      RISK_GOVERNOR_STREAK_PAUSE_SECONDS='120';
      RISK_GOVERNOR_HARD_BLOCK_ON_STREAK='false';
      AUTO_TRADE_SL_REENTRY_COOLDOWN_SECONDS='1200';
      PAPER_RESET_ON_START='false';
    }
  },
  @{ name='mx4_flow_guarded'; base='mx3_flow_compound'; overrides=@{
      ADAPTIVE_FILTERS_INTERVAL_SECONDS='180';
      ADAPTIVE_FILTERS_TARGET_CAND_MIN='1.8';
      ADAPTIVE_FILTERS_TARGET_CAND_MAX='14.0';
      ADAPTIVE_SCORE_MIN='49';
      ADAPTIVE_SAFE_VOLUME_MIN='28';
      ADAPTIVE_EDGE_MIN='0.55';
      ADAPTIVE_DEDUP_TTL_MIN='6';
      ADAPTIVE_DEDUP_TTL_MAX='30';
      ADAPTIVE_DEDUP_DYNAMIC_MIN='5';
      ADAPTIVE_DEDUP_DYNAMIC_MAX='16';
      AUTO_TRADE_TOP_N='18';
      AUTONOMOUS_CONTROL_INTERVAL_SECONDS='210';
      AUTONOMOUS_CONTROL_TARGET_CANDIDATES_MIN='1.6';
      AUTONOMOUS_CONTROL_TARGET_CANDIDATES_HIGH='9.0';
      AUTONOMOUS_CONTROL_TARGET_OPENED_MIN='0.14';
      AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MIN='36';
      AUTONOMOUS_CONTROL_MAX_BUYS_PER_HOUR_MAX='76';
      AUTONOMOUS_CONTROL_TOP_N_MIN='12';
      AUTONOMOUS_CONTROL_TOP_N_MAX='20';
      AUTONOMOUS_CONTROL_MAX_OPEN_TRADES_MAX='5';
      AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MIN='0.55';
      AUTONOMOUS_CONTROL_TRADE_SIZE_MAX_MAX='0.95';
      AUTONOMOUS_CONTROL_ANTI_STALL_ENABLED='true';
      AUTONOMOUS_CONTROL_ANTI_STALL_MIN_CANDIDATES='1.0';
      AUTONOMOUS_CONTROL_ANTI_STALL_MIN_UTILIZATION='0.80';
      AUTONOMOUS_CONTROL_ANTI_STALL_LIMIT_SKIP_MIN='1';
      AUTONOMOUS_CONTROL_ANTI_STALL_EXPAND_MULT='1.6';
      AUTONOMOUS_CONTROL_RECOVERY_ENABLED='true';
      AUTONOMOUS_CONTROL_RECOVERY_MIN_CANDIDATES='1.0';
      AUTONOMOUS_CONTROL_RECOVERY_EXPAND_MULT='1.5';
      AUTONOMOUS_CONTROL_FRAGILE_TIGHTEN_ENABLED='true';
      PROFIT_LOCK_TRIGGER_PERCENT='2.6';
      PROFIT_LOCK_FLOOR_PERCENT='0.85';
      NO_MOMENTUM_EXIT_MIN_AGE_PERCENT='10';
      NO_MOMENTUM_EXIT_MAX_PNL_PERCENT='0.28';
      PAPER_PARTIAL_TP_TRIGGER_PERCENT='1.2';
      PAPER_PARTIAL_TP_SELL_FRACTION='0.38';
      WEAKNESS_EXIT_MIN_AGE_PERCENT='15';
      WEAKNESS_EXIT_PNL_PERCENT='-2.8';
      PAPER_MAX_HOLD_SECONDS='180';
      MIN_TOKEN_SCORE='49';
      SAFE_MIN_VOLUME_5M_USD='38';
      MAX_BUYS_PER_HOUR='66';
      MAX_OPEN_TRADES='5';
      PAPER_TRADE_SIZE_MIN_USD='0.45';
      PAPER_TRADE_SIZE_MAX_USD='0.82';
      ENTRY_REQUIRE_POSITIVE_CHANGE_5M='true';
      ENTRY_MIN_PRICE_CHANGE_5M_PERCENT='0.10';
      ENTRY_MIN_VOLUME_5M_MULT='1.00';
      HOLD_MIN_SECONDS='30';
      HOLD_MAX_SECONDS='165';
      HEAVY_CHECK_DEDUP_TTL_SECONDS='8';
      MAX_TOKEN_COOLDOWN_SECONDS='45';
      AUTONOMOUS_CONTROL_RISK_OFF_OPEN_TRADES_CAP='3';
      AUTONOMOUS_CONTROL_RISK_OFF_MAX_BUYS_PER_HOUR_CAP='36';
      RISK_GOVERNOR_MAX_LOSS_STREAK='5';
      RISK_GOVERNOR_STREAK_PAUSE_SECONDS='90';
      RISK_GOVERNOR_HARD_BLOCK_ON_STREAK='false';
      PAPER_RESET_ON_START='false';
    }
  }
)

$variantLookup = @{}
foreach ($variant in $variants) {
  $variantLookup[[string]$variant.name] = $variant
}

function Resolve-VariantOverrides {
  param(
    [string]$VariantName,
    [hashtable]$Stack
  )
  if ([string]::IsNullOrWhiteSpace($VariantName)) {
    throw "Empty variant name."
  }
  if ($Stack.ContainsKey($VariantName)) {
    throw ("Cyclic variant base chain detected: {0}" -f $VariantName)
  }
  if (-not $variantLookup.ContainsKey($VariantName)) {
    throw ("Unknown variant in base chain: {0}" -f $VariantName)
  }

  $Stack[$VariantName] = $true
  $variant = $variantLookup[$VariantName]
  $resolved = @{}

  $baseName = [string]($variant.base)
  if (-not [string]::IsNullOrWhiteSpace($baseName)) {
    $baseResolved = Resolve-VariantOverrides -VariantName $baseName -Stack $Stack
    foreach ($k in $baseResolved.Keys) {
      $resolved[[string]$k] = [string]$baseResolved[$k]
    }
  }

  $overrides = $variant.overrides
  if ($overrides) {
    foreach ($k in $overrides.Keys) {
      $resolved[[string]$k] = [string]$overrides[$k]
    }
  }
  $null = $Stack.Remove($VariantName)
  return $resolved
}

$selected = @()
if ($ProfileIds -and @($ProfileIds).Count -gt 0) {
  $wanted = @{}
  foreach ($rawId in @($ProfileIds)) {
    foreach ($piece in ([string]$rawId -split ',')) {
      $id = [string]$piece
      if ([string]::IsNullOrWhiteSpace($id)) { continue }
      $wanted[$id.Trim()] = $true
    }
  }
  if ($wanted.Count -eq 0) {
    throw "ProfileIds was provided but no valid ids were found."
  }
  foreach ($v in $variants) {
    if ($wanted.ContainsKey([string]$v.name)) {
      $selected += $v
    }
  }
  $selectedNames = @($selected | ForEach-Object { [string]$_.name })
  $missing = @($wanted.Keys | Where-Object { $_ -notin $selectedNames })
  if (@($missing).Count -gt 0) {
    throw ("Unknown profile ids: {0}" -f (@($missing) -join ', '))
  }
} else {
  $take = [int]$Count
  $defaultByCount = @{
    1 = @('mx3_flow_compound')
    2 = @('mx3_flow_compound', 'mx4_flow_guarded')
    3 = @('mx1_flow_balanced', 'mx3_flow_compound', 'mx4_flow_guarded')
    4 = @('mx1_flow_balanced', 'mx2_flow_aggressive', 'mx3_flow_compound', 'mx4_flow_guarded')
  }
  if ($defaultByCount.ContainsKey($take)) {
    $selected = @()
    foreach ($id in @($defaultByCount[$take])) {
      if (-not $variantLookup.ContainsKey([string]$id)) {
        throw ("Default matrix profile missing: {0}" -f $id)
      }
      $selected += $variantLookup[[string]$id]
    }
  } else {
    $selected = @($variants | Select-Object -First $take)
  }
}

$take = @($selected).Count
if ($take -le 0) {
  throw "No matrix profiles selected."
}

$selectedResolved = @()
foreach ($v in $selected) {
  $id = [string]$v.name
  $resolved = Resolve-VariantOverrides -VariantName $id -Stack @{}
  $selectedResolved += [pscustomobject]@{
    name = $id
    overrides = $resolved
  }
}
$selected = $selectedResolved

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
    "AUTONOMOUS_CONTROL_DECISIONS_LOG_FILE=logs/matrix/$id/autonomy_decisions.jsonl",
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
  if ($Run -and [string]($v.overrides['PAPER_RESET_ON_START']) -eq 'true') {
    $paperStateAbs = Join-Path $root $paperState
    if (Test-Path $paperStateAbs) {
      Remove-Item -Force -ErrorAction SilentlyContinue $paperStateAbs
    }
  }

  $rec = [ordered]@{
    id = $id
    env_file = $envPath
    log_dir = $logDir
    paper_state_file = $paperState
    graceful_stop_file = $graceful
    overrides = $v.overrides
    pid = $null
    status = 'prepared'
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
      $startedPid = [int]$proc.Id
      $alive = $false
      for ($n = 0; $n -lt 24; $n++) {
        Start-Sleep -Milliseconds 500
        if (Test-MainLocalProcess -ProcessId $startedPid) {
          $alive = $true
          break
        }
      }
      if ($alive) {
        $rec.pid = $startedPid
        $rec.status = 'running'
      } else {
        $rec.status = 'start_failed_or_exited'
      }
    } else {
      $rec.status = 'start_failed'
    }
  }

  $records += [pscustomobject]$rec
}

$aliveCount = 0
foreach ($r in $records) {
  $procIdInMeta = 0
  try { $procIdInMeta = [int]($r.pid) } catch { $procIdInMeta = 0 }
  if ($procIdInMeta -gt 0 -and (Test-MainLocalProcess -ProcessId $procIdInMeta)) {
    $aliveCount += 1
  }
}

$meta = [ordered]@{
  created_at = (Get-Date).ToString('s')
  count = $take
  requested_run = [bool]$Run
  running = [bool]($aliveCount -gt 0)
  alive_count = $aliveCount
  items = $records
}

$metaPath = Join-Path $runDir 'active_matrix.json'
$metaJson = $meta | ConvertTo-Json -Depth 8
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($metaPath, $metaJson, $utf8NoBom)

Write-Host "Matrix prepared: $metaPath"
if ($Run) {
  Write-Host "Started instances:"
  $records | ForEach-Object { Write-Host ("  {0} pid={1} status={2}" -f $_.id, $_.pid, $_.status) }
  Write-Host ("Alive instances: {0}/{1}" -f $aliveCount, $take)
} else {
  Write-Host "Use -Run to start instances."
}

