from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


class MarketStore:
    """SQLite-backed storage for normalized market events."""

    def __init__(self, db_path: str = "data/market_data.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )

        self._configure()
        self._create_tables()

    def _configure(self) -> None:
        """Configure SQLite for local streaming writes."""
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA synchronous=NORMAL;")

    def _create_tables(self) -> None:
        """Create the market events table if it does not exist."""
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume_24h REAL,
                quote_volume_24h REAL,
                event_time_ms INTEGER NOT NULL,
                received_at_utc TEXT NOT NULL
            )
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_market_events_symbol_time
            ON market_events(symbol, event_time_ms)
            """
        )

        self.connection.commit()

    def insert_event(
        self,
        symbol: str,
        price: float,
        volume_24h: Optional[float],
        quote_volume_24h: Optional[float],
        event_time_ms: int,
        received_at_utc: str,
    ) -> None:
        """Insert one normalized market event."""
        self.connection.execute(
            """
            INSERT INTO market_events (
                symbol,
                price,
                volume_24h,
                quote_volume_24h,
                event_time_ms,
                received_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                price,
                volume_24h,
                quote_volume_24h,
                event_time_ms,
                received_at_utc,
            ),
        )

        self.connection.commit()

    def count_events(self) -> int:
        """Return the total number of stored market events."""
        cursor = self.connection.execute(
            "SELECT COUNT(*) FROM market_events"
        )
        return int(cursor.fetchone()[0])

    def close(self) -> None:
        """Close the SQLite connection."""
        self.connection.close()