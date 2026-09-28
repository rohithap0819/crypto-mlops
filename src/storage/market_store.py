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
        self._create_candle_table()

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


    def _create_candle_table(self) -> None:
        """Create the 1-minute candle table if it does not exist."""
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time_ms INTEGER NOT NULL,
                close_time_ms INTEGER NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL NOT NULL,
                trade_count INTEGER NOT NULL,
                taker_buy_volume REAL,
                taker_buy_quote_volume REAL,
                is_closed INTEGER NOT NULL,
                received_at_utc TEXT NOT NULL,
                UNIQUE(symbol, interval, open_time_ms)
            )
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_market_candles_symbol_time
            ON market_candles(symbol, interval, open_time_ms)
            """
        )

        self.connection.commit()


    def insert_candle(
        self,
        symbol: str,
        interval: str,
        open_time_ms: int,
        close_time_ms: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        quote_volume: float,
        trade_count: int,
        taker_buy_volume: float | None,
        taker_buy_quote_volume: float | None,
        is_closed: bool,
        received_at_utc: str,
    ) -> None:
        """Insert or update one kline candle."""

        self.connection.execute(
            """
            INSERT INTO market_candles (
                symbol,
                interval,
                open_time_ms,
                close_time_ms,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                quote_volume,
                trade_count,
                taker_buy_volume,
                taker_buy_quote_volume,
                is_closed,
                received_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, interval, open_time_ms)
            DO UPDATE SET
                close_time_ms = excluded.close_time_ms,
                open_price = excluded.open_price,
                high_price = excluded.high_price,
                low_price = excluded.low_price,
                close_price = excluded.close_price,
                volume = excluded.volume,
                quote_volume = excluded.quote_volume,
                trade_count = excluded.trade_count,
                taker_buy_volume = excluded.taker_buy_volume,
                taker_buy_quote_volume = excluded.taker_buy_quote_volume,
                is_closed = excluded.is_closed,
                received_at_utc = excluded.received_at_utc
            """,
            (
                symbol,
                interval,
                open_time_ms,
                close_time_ms,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                quote_volume,
                trade_count,
                taker_buy_volume,
                taker_buy_quote_volume,
                int(is_closed),
                received_at_utc,
            ),
        )

        self.connection.commit()


    def insert_event(
        self,
        symbol: str,
        price: float,
        volume_24h: float | None,
        quote_volume_24h: float | None,
        event_time_ms: int,
        received_at_utc: str,
    ) -> None:
        """Insert one ticker market event."""

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