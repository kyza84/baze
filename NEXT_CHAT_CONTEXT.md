# NEXT_CHAT_CONTEXT

## Project
- Repo: `d:\earnforme\solana-alert-bot`
- Goal: stabilize `LIVE` mode so behavior/visibility is close to `PAPER` (without unsafe relax of scam filters).
- GUI is the main control surface (`launcher_gui.py`).

## What happened before
- In `LIVE`, user often saw:
  - opened positions not visible in GUI,
  - `UNTRACKED` rows hanging after sell,
  - mismatch between wallet and `open_positions`,
  - lower entry flow than in `PAPER` due stricter live gates.
- User explicitly requires manual confirmation before any live switch/start.

## Important constraints from user
- Do not start/stop live trading without direct explicit user command.
- Preserve quality of entries (anti-scam guardrails must stay strong).
- GUI must show real runtime state clearly (open/closed/live reasons/health).

## Latest code changes (current session)
### 1) Address-key consistency fix in trader state
- File: `trading/auto_trader.py`
- Added helpers:
  - `_position_key(token_address)`
  - `_set_open_position(position)`
  - `_pop_open_position(token_address)`
- Purpose:
  - enforce normalized address keys in `open_positions`,
  - remove stale legacy raw keys on close,
  - reduce `UNTRACKED`/stuck-open mismatches caused by mixed key formats.
- Updated open/close paths to use helpers instead of direct dict access.
- Updated `live_buy_zero_amount` recovery map write to normalized key.

### 2) In-progress GUI consistency pass
- Target file: `launcher_gui.py`
- Next intended adjustment: KPI/summary should count visible untracked wallet positions (not show misleading `Open: 0` when `UNTRACKED` rows exist).
- This part was planned and should be completed/validated in the next chat if not yet applied.

## Runtime/data state notes
- Repo contains many runtime artifacts (`trading/paper_state*.json`, `logs/*`, `data/matrix/backups/*`, `collected_info/*`).
- There are deep nested backup paths from old `full_project_*` recursion artifacts.
- Keep code commits clean: avoid committing runtime junk unless explicitly requested.

## What to verify first in next chat
1. Run static sanity:
   - `python -m py_compile launcher_gui.py main_local.py trading/auto_trader.py trading/live_executor.py`
2. Verify key consistency behavior:
   - open live position -> ensure appears in GUI open table,
   - close/sell -> ensure row leaves open table and appears in closed,
   - ensure no stale `UNTRACKED` remains when wallet balance is zero.
3. Check GUI source selector:
   - in `Trades` tab, verify correct source (`single` vs matrix profile) before judging “no trades”.
4. Validate `LIVE` guard reasons in GUI:
   - limits line, idle reasons, block-after-pass, health line.

## Operational checklist before next live start
- Confirm `.env` mode coherence:
  - `AUTO_TRADE_ENABLED=true`
  - `AUTO_TRADE_PAPER=false` for live
  - correct wallet/rpc/router fields
- Ensure bot process state is clean (no stale old PID lock confusion).
- Clear only GUI-visible history if needed (without deleting forensic logs).
- Start only after user says explicit command.

## Suggested next-step implementation
- Finish GUI KPI fix for untracked/open count consistency.
- Add a small invariant check in refresh loop:
  - if wallet shows token raw>0 and neither `open_positions` nor `recovery_untracked` has it, emit warning event and attempt one-shot resync.
- Add an integration smoke script for state lifecycle (buy/open -> close -> persisted state).

## Quick starter prompt for new chat
Use this:
"Continue from NEXT_CHAT_CONTEXT.md. First verify current code state and compile checks, then finish GUI open/untracked consistency, then run a safe dry validation plan for live visibility without starting live until I confirm."
