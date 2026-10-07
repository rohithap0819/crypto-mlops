from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# IMPORT THE EXISTING V2 FEATURE DEFINITIONS
# ============================================================

FEATURE_DIR = Path(__file__).resolve().parent

if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

from build_features_v2 import (  # noqa: E402
    SYMBOLS,
    calculate_features,
    create_cross_asset_features,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "data" / "crypto_live.db"

FEATURE_COLUMNS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "feature_columns_v2.txt"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "live_features_latest.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

# 60 minutes are needed by the GRU sequence.
# Extra history is loaded for indicator warm-up.
LOOKBACK_CANDLES = 120

EXPECTED_FEATURE_COUNT = 61


# ============================================================
# FEATURE LIST
# ============================================================

def load_feature_columns() -> list[str]:
    """Load the frozen V2 feature list."""

    if not FEATURE_COLUMNS_FILE.exists():
        raise FileNotFoundError(
            f"Feature list not found: {FEATURE_COLUMNS_FILE}"
        )

    features = [
        line.strip()
        for line in FEATURE_COLUMNS_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Unexpected V2 feature count: "
            f"expected {EXPECTED_FEATURE_COUNT}, "
            f"found {len(features)}"
        )

    return features


# ============================================================
# DATABASE
# ============================================================

def get_connection() -> sqlite3.Connection:
    """Open the live SQLite database."""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Live database not found: {DB_PATH}"
        )

    connection = sqlite3.connect(DB_PATH)

    return connection


def load_recent_symbol_data(
    symbol: str,
    limit: int = LOOKBACK_CANDLES,
) -> pd.DataFrame:
    """
    Load recent OHLCV candles for one symbol.

    SQLite stores timestamps as Unix milliseconds.
    The feature pipeline converts them to pandas datetimes.
    """

    query = """
    SELECT
        open_time_ms,
        open_price AS open,
        high_price AS high,
        low_price AS low,
        close_price AS close,
        volume
    FROM market_klines_1m
    WHERE symbol = ?
    ORDER BY open_time_ms DESC
    LIMIT ?
    """

    with get_connection() as connection:
        df = pd.read_sql_query(
            query,
            connection,
            params=(symbol, limit),
        )

    if df.empty:
        raise ValueError(
            f"No live candles found for {symbol}"
        )

    df = df.sort_values("open_time_ms").reset_index(
        drop=True
    )

    df["open_time"] = pd.to_datetime(
        df["open_time_ms"],
        unit="ms",
        utc=True,
    )

    df = df.drop(columns=["open_time_ms"])

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

    df = (
        df.sort_values("open_time")
        .drop_duplicates(subset=["open_time"])
        .reset_index(drop=True)
    )

    return df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


# ============================================================
# DATA AVAILABILITY
# ============================================================

def find_latest_common_timestamp(
    dataframes: dict[str, pd.DataFrame],
) -> pd.Timestamp:
    """
    Find the latest candle timestamp available
    for every configured symbol.
    """

    common_times: set[pd.Timestamp] | None = None

    for symbol, df in dataframes.items():

        symbol_times = set(
            df["open_time"].dropna().tolist()
        )

        if common_times is None:
            common_times = symbol_times
        else:
            common_times &= symbol_times

    if not common_times:
        raise ValueError(
            "No common candle timestamp exists across "
            "all configured symbols."
        )

    return max(common_times)


def validate_recent_history(
    dataframes: dict[str, pd.DataFrame],
    latest_common_time: pd.Timestamp,
) -> None:
    """
    Validate that enough consecutive 1-minute candles exist
    before attempting live feature generation.
    """

    required = 61

    for symbol, df in dataframes.items():

        usable = (
            df[df["open_time"] <= latest_common_time]
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        if len(usable) < required:
            raise ValueError(
                f"{symbol} has only {len(usable)} candles "
                f"up to {latest_common_time}. "
                f"At least {required} candles are required "
                "for V2 features and the 60-minute GRU sequence. "
                "Bootstrap the SQLite database with historical "
                "data before running live inference."
            )

        recent_times = (
            usable["open_time"]
            .tail(required)
            .reset_index(drop=True)
        )

        diffs = (
            recent_times
            .diff()
            .dropna()
        )

        invalid_gaps = (
            diffs != pd.Timedelta(minutes=1)
        )

        if invalid_gaps.any():

            bad_gap = diffs[
                invalid_gaps
            ].iloc[0]

            raise ValueError(
                f"{symbol} contains a gap in the most recent "
                f"{required} candles before "
                f"{latest_common_time}. "
                f"Detected interval: {bad_gap}. "
                "Live inference requires a continuous "
                "1-minute sequence."
            )
    """
    Validate that enough consecutive 1-minute candles exist
    before attempting live feature generation.
    """

    required = 61

    for symbol, df in dataframes.items():

        usable = df[
            df["open_time"] <= latest_common_time
        ].sort_values("open_time")

        if len(usable) < required:
            raise ValueError(
                f"{symbol} has only {len(usable)} candles "
                f"up to {latest_common_time}. "
                f"At least {required} candles are required "
                "for V2 features and the 60-minute GRU sequence. "
                "Bootstrap the SQLite database with historical "
                "data before running live inference."
            )

        recent_times = usable["open_time"].tail(61)

        gaps = (
            recent_times.diff()
            .dropna()
            != pd.Timedelta(minutes=1)
        )

        if gaps.any():
            bad_gap = recent_times.diff()[
                gaps
            ].iloc[0]

            raise ValueError(
                f"{symbol} contains a gap in the most recent "
                f"61 candles before {latest_common_time}. "
                f"Detected gap: {bad_gap}. "
                "Live inference requires a continuous sequence."
            )


# ============================================================
# BUILD LIVE FEATURES
# ============================================================

def build_latest_live_features() -> pd.DataFrame:
    """
    Build the latest common V2 feature row for each symbol.

    Returns:
        DataFrame containing:
        - symbol
        - open_time
        - close
        - exact 61 V2 model features
    """

    feature_columns = load_feature_columns()

    # --------------------------------------------------------
    # Load recent live OHLCV
    # --------------------------------------------------------

    symbol_data: dict[str, pd.DataFrame] = {}

    for symbol in SYMBOLS:

        df = load_recent_symbol_data(
            symbol=symbol,
            limit=LOOKBACK_CANDLES,
        )

        symbol_data[symbol] = df

    # --------------------------------------------------------
    # Latest timestamp shared by all 5 coins
    # --------------------------------------------------------

    latest_common_time = find_latest_common_timestamp(
        symbol_data
    )

    # --------------------------------------------------------
    # Verify enough continuous history exists
    # --------------------------------------------------------

    validate_recent_history(
        symbol_data,
        latest_common_time,
    )

    # --------------------------------------------------------
    # Apply EXACT existing V2 feature calculations
    # --------------------------------------------------------

    feature_data: dict[str, pd.DataFrame] = {}

    for symbol in SYMBOLS:

        df = symbol_data[symbol].copy()

        df = calculate_features(df)

        feature_data[symbol] = df

    # --------------------------------------------------------
    # Apply EXACT existing cross-asset calculations
    # --------------------------------------------------------

    cross_features = create_cross_asset_features(
        feature_data
    )

    # --------------------------------------------------------
    # Merge cross-asset features
    # --------------------------------------------------------

    latest_frames = []

    for symbol in SYMBOLS:

        df = feature_data[symbol].copy()

        df = df.merge(
            cross_features,
            on="open_time",
            how="left",
        )

        df["symbol"] = symbol

        current = df[
            df["open_time"] == latest_common_time
        ].copy()

        if current.empty:
            raise ValueError(
                f"No feature row found for {symbol} "
                f"at {latest_common_time}"
            )

        latest_frames.append(current)

    final_df = pd.concat(
        latest_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Clean invalid numeric values
    # --------------------------------------------------------

    final_df = final_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # Validate exact 61-feature contract
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in final_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing V2 features:\n"
            + "\n".join(missing_features)
        )

    null_features = (
        final_df[feature_columns]
        .isna()
        .sum()
    )

    bad_null_features = (
        null_features[null_features > 0]
    )

    if not bad_null_features.empty:
        raise ValueError(
            "Live feature generation produced NaN values:\n"
            + bad_null_features.to_string()
        )

    # --------------------------------------------------------
    # Keep only inference metadata + exact 61 features
    # --------------------------------------------------------

    output_columns = [
        "symbol",
        "open_time",
        "close",
    ] + feature_columns

    final_df = final_df[
        output_columns
    ].copy()

    final_df = final_df.sort_values(
        "symbol"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Final contract checks
    # --------------------------------------------------------

    if len(final_df) != len(SYMBOLS):
        raise ValueError(
            f"Expected {len(SYMBOLS)} symbols, "
            f"found {len(final_df)} rows"
        )

    if final_df[feature_columns].shape[1] != (
        EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "Final live feature matrix does not contain "
            f"{EXPECTED_FEATURE_COUNT} features."
        )

    return final_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CRYPTO MLOPS - LIVE V2 FEATURE BUILDER")
    print("=" * 70)

    print(f"Database: {DB_PATH}")
    print(f"Feature list: {FEATURE_COLUMNS_FILE}")

    result = build_latest_live_features()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    feature_columns = load_feature_columns()

    print()
    print("=" * 70)
    print("LIVE FEATURE GENERATION COMPLETE")
    print("=" * 70)

    print(f"Rows: {len(result)}")
    print(f"Model features: {len(feature_columns)}")
    print(
        f"Timestamp: "
        f"{result['open_time'].iloc[0]}"
    )

    print()
    print(
        result[
            ["symbol", "open_time", "close"]
        ].to_string(index=False)
    )

    print()
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()