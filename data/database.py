import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path("data/trading.db")


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't exist."""
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv_bars (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL,
                open        REAL    NOT NULL,
                high        REAL    NOT NULL,
                low         REAL    NOT NULL,
                close       REAL    NOT NULL,
                volume      REAL    NOT NULL,
                timeframe   TEXT    NOT NULL DEFAULT '1Day',
                UNIQUE(symbol, timestamp, timeframe)
            );
            CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts
                ON ohlcv_bars(symbol, timestamp);
        """)
    log.info("Database ready at %s", db_path)


@contextmanager
def _connect(db_path: Path = DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_bars(rows: list[dict], db_path: Path = DB_PATH) -> int:
    """Insert OHLCV rows, silently skipping duplicates. Returns rows inserted."""
    if not rows:
        return 0
    sql = """
        INSERT OR IGNORE INTO ohlcv_bars
            (symbol, timestamp, open, high, low, close, volume, timeframe)
        VALUES
            (:symbol, :timestamp, :open, :high, :low, :close, :volume, :timeframe)
    """
    with _connect(db_path) as conn:
        cur = conn.executemany(sql, rows)
        return cur.rowcount


def fetch_bars(symbol: str, limit: int = 10, db_path: Path = DB_PATH) -> list[dict]:
    """Return the most recent `limit` bars for a symbol, newest first."""
    sql = """
        SELECT symbol, timestamp, open, high, low, close, volume, timeframe
        FROM   ohlcv_bars
        WHERE  symbol = ?
        ORDER  BY timestamp DESC
        LIMIT  ?
    """
    with _connect(db_path) as conn:
        rows = conn.execute(sql, (symbol, limit)).fetchall()
    return [dict(r) for r in rows]


def get_bar_count(symbol: str, db_path: Path = DB_PATH) -> int:
    """Return the total number of stored bars for a symbol."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM ohlcv_bars WHERE symbol = ?", (symbol,)
        ).fetchone()
    return row[0] if row else 0
