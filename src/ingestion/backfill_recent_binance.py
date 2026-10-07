from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import requests


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "data" / "crypto_live.db"


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

INTERVAL = "1m"

# Enough for:
# - V2 indicator warm-up
# - 60-minute GRU sequence
# - temporary ingestion gaps
LIMIT = 500

# Public Binance market-data endpoint.
BASE_URL = "https://api.binance.com/api/v3/klines"

REQUEST_TIMEOUT = 15


# ============================================================
# DATABASE
# ============================================================

INSERT_SQL = """
INSERT INTO market_klines_1m (
    symbol,
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
    event_time_ms,
    received_at_ms
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol, open_time_ms)
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
    event_time_ms = excluded.event_time_ms,
    received_at_ms = excluded.received_at_ms;
"""


def get_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DB_PATH}"
        )

    connection = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )

    connection.execute(
        "PRAGMA journal_mode=WAL;"
    )

    connection.execute(
        "PRAGMA synchronous=NORMAL;"
    )

    connection.execute(
        "PRAGMA busy_timeout=30000;"
    )

    return connection


# ============================================================
# BINANCE
# ============================================================

def fetch_recent_klines(
    symbol: str,
) -> list[list]:

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": LIMIT,
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise RuntimeError(
            f"Unexpected Binance response for {symbol}: "
            f"{data}"
        )

    return data


def insert_klines(
    connection: sqlite3.Connection,
    symbol: str,
    klines: list[list],
) -> int:

    now_ms = int(time.time() * 1000)

    rows = []

    for kline in klines:

        open_time_ms = int(kline[0])
        close_time_ms = int(kline[6])

        rows.append(
            (
                symbol,
                open_time_ms,
                close_time_ms,

                float(kline[1]),
                float(kline[2]),
                float(kline[3]),
                float(kline[4]),

                float(kline[5]),
                float(kline[7]),

                int(kline[8]),

                float(kline[9]),
                float(kline[10]),

                now_ms,
                now_ms,
            )
        )

    connection.executemany(
        INSERT_SQL,
        rows,
    )

    return len(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CRYPTO MLOPS - RECENT BINANCE HISTORY BACKFILL")
    print("=" * 70)

    print(f"Database: {DB_PATH}")
    print(f"Symbols : {', '.join(SYMBOLS)}")
    print(f"Interval: {INTERVAL}")
    print(f"Limit   : {LIMIT}")

    with get_connection() as connection:

        for symbol in SYMBOLS:

            print()
            print(
                f"Fetching {symbol}..."
            )

            klines = fetch_recent_klines(
                symbol
            )

            print(
                f"Received {len(klines)} candles"
            )

            inserted = insert_klines(
                connection,
                symbol,
                klines,
            )

            connection.commit()

            print(
                f"Stored {inserted} candles"
            )

    print()
    print("=" * 70)
    print("BINANCE BACKFILL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()