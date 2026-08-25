"""
logs/history_db.py
------------------
SQLite-backed history store for every document analysis.
Used by the dashboard to show a searchable log of previous uploads.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "history.db"


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    """Create the analyses table if it doesn't exist."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                filename    TEXT,
                text_snippet TEXT,
                rule_score  REAL,
                anomaly_score REAL,
                classifier_score REAL,
                consistency_score REAL,
                risk_score  REAL,
                decision    TEXT,
                matched_rules TEXT,
                processing_time_ms REAL,
                extra_json  TEXT
            )
        """)


def log_analysis(
    filename: str,
    text_snippet: str,
    rule_score: float,
    anomaly_score: float,
    classifier_score: float,
    consistency_score: float,
    risk_score: float,
    decision: str,
    matched_rules: list[str],
    processing_time_ms: float,
    extra: dict[str, Any] | None = None,
) -> int:
    """Insert one analysis record. Returns the new row id."""
    init_db()
    with _conn() as con:
        cur = con.execute("""
            INSERT INTO analyses (
                timestamp, filename, text_snippet,
                rule_score, anomaly_score, classifier_score, consistency_score,
                risk_score, decision, matched_rules, processing_time_ms, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            filename,
            text_snippet[:300],
            rule_score,
            anomaly_score,
            classifier_score,
            consistency_score,
            risk_score,
            decision,
            json.dumps(matched_rules),
            processing_time_ms,
            json.dumps(extra or {}),
        ))
        return cur.lastrowid


def get_history(limit: int = 200, search: str = "") -> list[dict]:
    """Return recent analyses, optionally filtered by filename or text."""
    init_db()
    with _conn() as con:
        if search:
            rows = con.execute("""
                SELECT * FROM analyses
                WHERE filename LIKE ? OR text_snippet LIKE ? OR decision LIKE ?
                ORDER BY id DESC LIMIT ?
            """, (f"%{search}%", f"%{search}%", f"%{search}%", limit)).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    """Return aggregate stats for the dashboard home page."""
    init_db()
    with _conn() as con:
        row = con.execute("""
            SELECT
                COUNT(*)                         AS total,
                SUM(CASE WHEN decision='pass'       THEN 1 ELSE 0 END) AS n_clean,
                SUM(CASE WHEN decision IN ('flag','quarantine') THEN 1 ELSE 0 END) AS n_malicious,
                AVG(processing_time_ms)          AS avg_time_ms,
                AVG(risk_score)                  AS avg_risk
            FROM analyses
        """).fetchone()
    return dict(row) if row else {}
