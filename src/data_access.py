"""Read-only access to `hotel.db`.

Every column in the source database is stored as TEXT, so nothing is coerced
here — loading and cleaning are kept separate so the EDA can show the raw
state before any repair, which is a finding in its own right.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from .config import resolve_db_path

BOOKINGS_TABLE = "bookings"
REVIEWS_TABLE = "reviews"


@contextmanager
def connect(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a read-only connection, closed on exit."""
    path = resolve_db_path(db_path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        yield conn
    finally:
        conn.close()


def load_table(name: str, db_path: str | Path | None = None) -> pd.DataFrame:
    """Load one table verbatim, every column as TEXT."""
    with connect(db_path) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{name}"', conn)


def load_bookings(db_path: str | Path | None = None) -> pd.DataFrame:
    return load_table(BOOKINGS_TABLE, db_path)


def load_reviews(db_path: str | Path | None = None) -> pd.DataFrame:
    return load_table(REVIEWS_TABLE, db_path)


def table_summary(db_path: str | Path | None = None) -> pd.DataFrame:
    """Row counts and declared column types, for the provenance section."""
    with connect(db_path) as conn:
        rows = []
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        for name in names:
            n = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            cols = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
            declared = {c[2] or "(none)" for c in cols}
            rows.append(
                {
                    "table": name,
                    "rows": n,
                    "columns": len(cols),
                    "declared_types": ", ".join(sorted(declared)),
                }
            )
    return pd.DataFrame(rows)
