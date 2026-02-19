"""Incrementally sync runtime logs/states into one deduplicated SQLite dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _normalize_json_line(line: str) -> str:
    # Keep raw compact form for speed on very large JSONL logs.
    return str(line or "").strip()


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _profile_from_state_path(path: str) -> str:
    base = os.path.basename(path).lower()
    if base.startswith("paper_state.") and base.endswith(".json"):
        return base.replace("paper_state.", "").replace(".json", "")
    return "single_or_other"


class UnifiedDatasetSync:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.dataset_dir = os.path.join(self.root, "data", "unified_dataset")
        self.db_path = os.path.join(self.dataset_dir, "unified.db")
        self.state_path = os.path.join(self.dataset_dir, "sync_state.json")
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.state: dict[str, Any] = self._load_state()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    def _load_state(self) -> dict[str, Any]:
        if not os.path.exists(self.state_path):
            return {"offsets": {}}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return {"offsets": {}}
            payload.setdefault("offsets", {})
            return payload
        except Exception:
            return {"offsets": {}}

    def _save_state(self) -> None:
        tmp = f"{self.state_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, self.state_path)

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                event_id TEXT PRIMARY KEY,
                ts REAL,
                run_tag TEXT,
                candidate_id TEXT,
                token_address TEXT,
                symbol TEXT,
                decision_stage TEXT,
                decision TEXT,
                reason TEXT,
                market_regime TEXT,
                entry_tier TEXT,
                score INTEGER,
                recommendation TEXT,
                source_mode TEXT,
                source_file TEXT NOT NULL,
                raw_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                ts REAL,
                run_tag TEXT,
                symbol TEXT,
                recommendation TEXT,
                risk_level TEXT,
                event_type TEXT,
                source_file TEXT NOT NULL,
                raw_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS closed_trades (
                trade_id TEXT PRIMARY KEY,
                profile TEXT,
                symbol TEXT,
                token_address TEXT,
                candidate_id TEXT,
                opened_at TEXT,
                closed_at TEXT,
                close_reason TEXT,
                market_mode TEXT,
                entry_tier TEXT,
                score INTEGER,
                position_size_usd REAL,
                original_position_size_usd REAL,
                pnl_usd REAL,
                pnl_percent REAL,
                partial_tp_done INTEGER,
                partial_realized_pnl_usd REAL,
                max_hold_seconds INTEGER,
                expected_edge_percent REAL,
                buy_cost_percent REAL,
                sell_cost_percent REAL,
                gas_cost_usd REAL,
                source_mode TEXT,
                run_tag TEXT,
                raw_hash TEXT,
                source_file TEXT NOT NULL,
                raw_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_decisions (
                decision_id TEXT PRIMARY KEY,
                ts REAL,
                run_tag TEXT,
                candidate_id TEXT,
                token_address TEXT,
                symbol TEXT,
                decision_stage TEXT,
                decision TEXT,
                reason TEXT,
                market_mode TEXT,
                entry_tier TEXT,
                score INTEGER,
                position_size_usd REAL,
                expected_edge_percent REAL,
                pnl_usd REAL,
                pnl_percent REAL,
                max_hold_seconds INTEGER,
                source_file TEXT NOT NULL,
                raw_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_candidates_stage_reason ON candidates(decision_stage, reason);
            CREATE INDEX IF NOT EXISTS idx_closed_trades_profile_closed ON closed_trades(profile, closed_at);
            CREATE INDEX IF NOT EXISTS idx_trade_decisions_stage_reason ON trade_decisions(decision_stage, reason);
            """
        )

        self._ensure_column("candidates", "candidate_id TEXT")
        self._ensure_column("candidates", "token_address TEXT")
        self._ensure_column("candidates", "entry_tier TEXT")
        self._ensure_column("candidates", "source_mode TEXT")

        self._ensure_column("closed_trades", "candidate_id TEXT")
        self._ensure_column("closed_trades", "market_mode TEXT")
        self._ensure_column("closed_trades", "entry_tier TEXT")
        self._ensure_column("closed_trades", "original_position_size_usd REAL")
        self._ensure_column("closed_trades", "partial_tp_done INTEGER")
        self._ensure_column("closed_trades", "partial_realized_pnl_usd REAL")
        self._ensure_column("closed_trades", "max_hold_seconds INTEGER")
        self._ensure_column("closed_trades", "expected_edge_percent REAL")
        self._ensure_column("closed_trades", "buy_cost_percent REAL")
        self._ensure_column("closed_trades", "sell_cost_percent REAL")
        self._ensure_column("closed_trades", "gas_cost_usd REAL")
        self._ensure_column("closed_trades", "source_mode TEXT")
        self._ensure_column("closed_trades", "run_tag TEXT")
        self._ensure_column("closed_trades", "raw_hash TEXT")
        self.conn.commit()

    def _table_columns(self, table_name: str) -> set[str]:
        rows = self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        out: set[str] = set()
        for row in rows:
            try:
                out.add(str(row[1]))
            except Exception:
                continue
        return out

    def _ensure_column(self, table_name: str, column_def: str) -> None:
        column = str(column_def).split(" ", 1)[0].strip()
        if not column:
            return
        existing = self._table_columns(table_name)
        if column in existing:
            return
        self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_def}")

    def _candidate_files(self) -> list[str]:
        out: list[str] = []
        logs_dir = os.path.join(self.root, "logs")
        if os.path.isdir(logs_dir):
            for dp, _, files in os.walk(logs_dir):
                for name in files:
                    if name.lower() == "candidates.jsonl":
                        out.append(os.path.join(dp, name))
        return sorted(set(out))

    def _alert_files(self) -> list[str]:
        out: list[str] = []
        logs_dir = os.path.join(self.root, "logs")
        if os.path.isdir(logs_dir):
            for dp, _, files in os.walk(logs_dir):
                for name in files:
                    if name.lower() == "local_alerts.jsonl":
                        out.append(os.path.join(dp, name))
        return sorted(set(out))

    def _trade_decision_files(self) -> list[str]:
        out: list[str] = []
        logs_dir = os.path.join(self.root, "logs")
        if os.path.isdir(logs_dir):
            for dp, _, files in os.walk(logs_dir):
                for name in files:
                    if name.lower() == "trade_decisions.jsonl":
                        out.append(os.path.join(dp, name))
        return sorted(set(out))

    def _state_files(self) -> list[str]:
        out: list[str] = []
        for rel in ("trading", os.path.join("data", "matrix", "backups")):
            base = os.path.join(self.root, rel)
            if not os.path.isdir(base):
                continue
            for dp, _, files in os.walk(base):
                for name in files:
                    low = name.lower()
                    if "paper_state" in low and low.endswith(".json"):
                        out.append(os.path.join(dp, name))
        return sorted(set(out))

    def _ingest_jsonl_incremental(self, path: str, kind: str) -> tuple[int, int]:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return 0, 0
        offsets = self.state.setdefault("offsets", {})
        prev = offsets.get(path, {"offset": 0, "size": 0})
        try:
            prev_offset = int(prev.get("offset", 0) or 0)
            prev_size = int(prev.get("size", 0) or 0)
        except Exception:
            prev_offset, prev_size = 0, 0

        size_now = os.path.getsize(path)
        if size_now < prev_size:
            prev_offset = 0

        inserted = 0
        read_lines = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            if prev_offset > 0:
                f.seek(prev_offset)
            for line in f:
                read_lines += 1
                norm = _normalize_json_line(line)
                if not norm:
                    continue
                event_id = _sha1_text(norm)
                row: dict[str, Any]
                try:
                    row = json.loads(norm) if norm.startswith("{") else {}
                except Exception:
                    row = {}

                if kind == "candidate":
                    cur = self.conn.execute(
                        """
                        INSERT OR IGNORE INTO candidates(
                            event_id, ts, run_tag, candidate_id, token_address, symbol, decision_stage,
                            decision, reason, market_regime, entry_tier, score, recommendation,
                            source_mode, source_file, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event_id,
                            _safe_float(row.get("ts"), 0.0),
                            str(row.get("run_tag", "")),
                            str(row.get("candidate_id", "")),
                            str(row.get("address", row.get("token_address", ""))).lower(),
                            str(row.get("symbol", "")),
                            str(row.get("decision_stage", "")),
                            str(row.get("decision", "")),
                            str(row.get("reason", "")),
                            str(row.get("market_regime", "")),
                            str(row.get("entry_tier", "")),
                            _safe_int(row.get("score"), 0),
                            str(row.get("recommendation", "")),
                            str(row.get("source_mode", "")),
                            path,
                            norm,
                        ),
                    )
                elif kind == "trade_decision":
                    cur = self.conn.execute(
                        """
                        INSERT OR IGNORE INTO trade_decisions(
                            decision_id, ts, run_tag, candidate_id, token_address, symbol,
                            decision_stage, decision, reason, market_mode, entry_tier,
                            score, position_size_usd, expected_edge_percent, pnl_usd, pnl_percent,
                            max_hold_seconds, source_file, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event_id,
                            _safe_float(row.get("ts"), 0.0),
                            str(row.get("run_tag", "")),
                            str(row.get("candidate_id", "")),
                            str(row.get("token_address", row.get("address", ""))).lower(),
                            str(row.get("symbol", "")),
                            str(row.get("decision_stage", "")),
                            str(row.get("decision", "")),
                            str(row.get("reason", "")),
                            str(row.get("market_mode", "")),
                            str(row.get("entry_tier", "")),
                            _safe_int(row.get("score"), 0),
                            _safe_float(row.get("position_size_usd"), 0.0),
                            _safe_float(row.get("expected_edge_percent"), 0.0),
                            _safe_float(row.get("pnl_usd"), 0.0),
                            _safe_float(row.get("pnl_percent"), 0.0),
                            _safe_int(row.get("max_hold_seconds"), 0),
                            path,
                            norm,
                        ),
                    )
                else:
                    cur = self.conn.execute(
                        """
                        INSERT OR IGNORE INTO alerts(
                            alert_id, ts, run_tag, symbol, recommendation, risk_level, event_type,
                            source_file, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event_id,
                            _safe_float(row.get("ts"), _safe_float(row.get("timestamp"), 0.0)),
                            str(row.get("run_tag", "")),
                            str(row.get("symbol", "")),
                            str(row.get("recommendation", "")),
                            str(row.get("risk_level", "")),
                            str(row.get("event_type", "")),
                            path,
                            norm,
                        ),
                    )
                if int(getattr(cur, "rowcount", 0) or 0) > 0:
                    inserted += 1
            new_offset = f.tell()

        offsets[path] = {"offset": new_offset, "size": size_now}
        return read_lines, inserted

    def _ingest_closed_trades(self, path: str) -> int:
        path = os.path.abspath(path)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                payload = json.load(f)
        except Exception:
            return 0
        closed = payload.get("closed_positions") or []
        if not isinstance(closed, list):
            return 0
        profile = _profile_from_state_path(path)
        inserted = 0
        for row in closed:
            if not isinstance(row, dict):
                continue
            stable = json.dumps(
                {
                    "profile": profile,
                    "symbol": str(row.get("symbol", "")),
                    "token_address": str(row.get("token_address", "")).lower(),
                    "opened_at": str(row.get("opened_at", "")),
                    "closed_at": str(row.get("closed_at", "")),
                    "close_reason": str(row.get("close_reason", "")),
                    "position_size_usd": round(_safe_float(row.get("position_size_usd"), 0.0), 6),
                    "pnl_usd": round(_safe_float(row.get("pnl_usd"), 0.0), 6),
                    "pnl_percent": round(_safe_float(row.get("pnl_percent"), 0.0), 6),
                    "score": _safe_int(row.get("score"), 0),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            trade_id = _sha1_text(stable)
            raw_json = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            run_tag = str(payload.get("run_tag", ""))
            cur = self.conn.execute(
                """
                INSERT OR IGNORE INTO closed_trades(
                    trade_id, profile, symbol, token_address, candidate_id, opened_at, closed_at, close_reason,
                    market_mode, entry_tier, score, position_size_usd, original_position_size_usd,
                    pnl_usd, pnl_percent, partial_tp_done, partial_realized_pnl_usd, max_hold_seconds,
                    expected_edge_percent, buy_cost_percent, sell_cost_percent, gas_cost_usd,
                    source_mode, run_tag, raw_hash, source_file, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    profile,
                    str(row.get("symbol", "")),
                    str(row.get("token_address", "")),
                    str(row.get("candidate_id", "")),
                    str(row.get("opened_at", "")),
                    str(row.get("closed_at", "")),
                    str(row.get("close_reason", "")),
                    str(row.get("market_mode", "")),
                    str(row.get("entry_tier", "")),
                    _safe_int(row.get("score"), 0),
                    _safe_float(row.get("position_size_usd"), 0.0),
                    _safe_float(row.get("original_position_size_usd"), _safe_float(row.get("position_size_usd"), 0.0)),
                    _safe_float(row.get("pnl_usd"), 0.0),
                    _safe_float(row.get("pnl_percent"), 0.0),
                    int(bool(row.get("partial_tp_done", False))),
                    _safe_float(row.get("partial_realized_pnl_usd"), 0.0),
                    _safe_int(row.get("max_hold_seconds"), 0),
                    _safe_float(row.get("expected_edge_percent"), 0.0),
                    _safe_float(row.get("buy_cost_percent"), 0.0),
                    _safe_float(row.get("sell_cost_percent"), 0.0),
                    _safe_float(row.get("gas_cost_usd"), 0.0),
                    str(row.get("source", "")),
                    run_tag,
                    _sha1_text(raw_json),
                    path,
                    raw_json,
                ),
            )
            if int(getattr(cur, "rowcount", 0) or 0) > 0:
                inserted += 1
        return inserted

    def sync_once(self, *, verbose: bool = True) -> dict[str, Any]:
        cand_read = 0
        cand_new = 0
        alert_read = 0
        alert_new = 0
        td_read = 0
        td_new = 0
        trades_new = 0

        for path in self._candidate_files():
            r, i = self._ingest_jsonl_incremental(path, "candidate")
            cand_read += r
            cand_new += i

        for path in self._alert_files():
            r, i = self._ingest_jsonl_incremental(path, "alert")
            alert_read += r
            alert_new += i

        for path in self._trade_decision_files():
            r, i = self._ingest_jsonl_incremental(path, "trade_decision")
            td_read += r
            td_new += i

        for path in self._state_files():
            trades_new += self._ingest_closed_trades(path)

        self.conn.commit()
        self._save_state()

        counts = {
            "candidates_total": self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
            "alerts_total": self.conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0],
            "trade_decisions_total": self.conn.execute("SELECT COUNT(*) FROM trade_decisions").fetchone()[0],
            "closed_trades_total": self.conn.execute("SELECT COUNT(*) FROM closed_trades").fetchone()[0],
        }
        result = {
            "ts": int(time.time()),
            "db_path": self.db_path,
            "cand_read_lines": cand_read,
            "cand_new_rows": cand_new,
            "alert_read_lines": alert_read,
            "alert_new_rows": alert_new,
            "trade_decision_read_lines": td_read,
            "trade_decision_new_rows": td_new,
            "closed_trades_new_rows": trades_new,
            **counts,
        }
        if verbose:
            print(json.dumps(result, ensure_ascii=False))
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync unified deduplicated dataset into SQLite.")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--follow", action="store_true", help="Run continuously")
    parser.add_argument("--loop-seconds", type=int, default=45, help="Loop interval when --follow")
    args = parser.parse_args()

    syncer = UnifiedDatasetSync(args.root)
    try:
        if not args.follow:
            syncer.sync_once(verbose=True)
            return 0
        while True:
            try:
                syncer.sync_once(verbose=True)
            except Exception as exc:
                print(json.dumps({"ts": int(time.time()), "error": str(exc)}, ensure_ascii=False))
            time.sleep(max(10, int(args.loop_seconds)))
    finally:
        syncer.close()


if __name__ == "__main__":
    raise SystemExit(main())
