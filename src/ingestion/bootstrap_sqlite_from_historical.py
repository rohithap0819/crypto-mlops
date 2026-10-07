from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_DIR = ROOT / "data" / "historical"
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

# Keep enough history for:
# - indicator warm-up
# - 60-minute GRU sequence
# - temporary live-data gaps
HISTORY_CANDLES_PER_SYMBOL = 500


# ============================================================
# DATABASE
# ============================================================

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(
        DB_PATH,
        timeout=30,
    )


# ============================================================
# TIMESTAMP NORMALIZATION
# ============================================================

def normalize_open_time(series: pd.Series) -> pd.Series:
    """Convert historical timestamps to Unix milliseconds."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    # If the source is numeric timestamps, determine the
    # unit from the magnitude.
    if numeric.notna().mean() > 0.95:

        median = float(
            numeric.dropna().median()
        )

        if median > 1e17:
            # nanoseconds
            return (
                numeric // 1_000_000
            ).astype("int64")

        if median > 1e14:
            # microseconds
            return (
                numeric // 1_000
            ).astype("int64")

        if median > 1e11:
            # milliseconds
            return numeric.astype("int64")

        if median > 1e8:
            # seconds
            return (
                numeric * 1_000
            ).astype("int64")

        raise ValueError(
            f"Unable to determine timestamp unit. "
            f"Median timestamp value: {median}"
        )

    # Otherwise parse normal datetime/string values.
    timestamps = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    if timestamps.isna().any():
        raise ValueError(
            "Historical data contains invalid timestamps."
        )

    return (
        timestamps.astype("int64")
        // 1_000_000
    ).astype("int64")


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_recent_history(
    symbol: str,
) -> pd.DataFrame:

    file_path = (
        HISTORICAL_DIR
        / f"{symbol}_1m.parquet"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Historical file not found: {file_path}"
        )

    df = pd.read_parquet(file_path)

    required_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol} is missing columns: {missing}"
        )

    df = df[required_columns].copy()

    df["open_time_ms"] = normalize_open_time(
        df["open_time"]
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "open_time_ms",
            *numeric_columns,
        ]
    )

    df = (
        df.sort_values("open_time_ms")
        .drop_duplicates(
            subset=["open_time_ms"]
        )
        .tail(HISTORY_CANDLES_PER_SYMBOL)
        .reset_index(drop=True)
    )

    return df


# ============================================================
# INSERT
# ============================================================

INSERT_SQL = """
INSERT OR IGNORE INTO market_klines_1m (
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
"""


def insert_symbol_history(
    connection: sqlite3.Connection,
    symbol: str,
    df: pd.DataFrame,
) -> int:

    rows = []

    for row in df.itertuples(index=False):

        open_time_ms = int(
            row.open_time_ms
        )

        close_time_ms = (
            open_time_ms + 59_999
        )

        rows.append(
            (
                symbol,
                open_time_ms,
                close_time_ms,
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),

                # The historical parquet used for V2
                # requires only OHLCV. These auxiliary
                # trade fields are not used by the live
                # V2 feature builder.
                0.0,
                0,
                0.0,
                0.0,

                # Historical/backfilled rows do not have
                # an actual WebSocket reception timestamp.
                close_time_ms,
                close_time_ms,
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
    print("CRYPTO MLOPS - SQLITE HISTORICAL BOOTSTRAP")
    print("=" * 70)

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DB_PATH}. "
            "Run the live Binance ingestor first."
        )

    total_rows_attempted = 0

    with get_connection() as connection:

        for symbol in SYMBOLS:

            print()
            print(f"Loading {symbol}...")

            df = load_recent_history(symbol)

            print(
                f"Historical rows selected: "
                f"{len(df):,}"
            )

            inserted = insert_symbol_history(
                connection,
                symbol,
                df,
            )

            total_rows_attempted += inserted

            print(
                f"Rows submitted to SQLite: "
                f"{inserted:,}"
            )

        connection.commit()

    print()
    print("=" * 70)
    print("HISTORICAL SQLITE BOOTSTRAP COMPLETE")
    print("=" * 70)
    print(
        f"Rows submitted: "
        f"{total_rows_attempted:,}"
    )
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    main()