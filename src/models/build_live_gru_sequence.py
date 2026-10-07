from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "src" / "features"

SEQUENCE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "sequence_v1"
)

FEATURE_COLUMNS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "feature_columns_v2.txt"
)

SCALER_FILE = (
    SEQUENCE_DIR
    / "feature_scaler.joblib"
)

LIVE_SEQUENCE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "live_sequence_v1"
)

LIVE_SEQUENCE_FILE = (
    LIVE_SEQUENCE_DIR
    / "latest_gru_sequence.npz"
)


# ============================================================
# IMPORT EXISTING FEATURE DEFINITIONS
# ============================================================

if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

from build_features_v2 import (  # noqa: E402
    SYMBOLS,
    calculate_features,
    create_cross_asset_features,
)

from live_feature_builder import (  # noqa: E402
    get_connection,
    load_recent_symbol_data,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 60
FEATURE_COUNT = 61

# More than 60 rows are loaded so indicators and
# rolling features have enough warm-up history.
LOOKBACK_CANDLES = 180

# ============================================================
# FEATURE LIST
# ============================================================

def load_feature_columns() -> list[str]:
    """Load the exact frozen V2 feature order."""

    features = [
        line.strip()
        for line in FEATURE_COLUMNS_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(features) != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features, "
            f"found {len(features)}"
        )

    return features


# ============================================================
# BUILD FEATURE HISTORY
# ============================================================

def build_feature_history() -> dict[str, pd.DataFrame]:
    """
    Build the V2 feature history for all five symbols.

    This uses the exact existing V2 feature functions.
    """

    raw_data = {}

    for symbol in SYMBOLS:
        raw_data[symbol] = load_recent_symbol_data(
            symbol=symbol,
            limit=LOOKBACK_CANDLES,
        )

    feature_data = {}

    for symbol in SYMBOLS:

        df = raw_data[symbol].copy()

        df = calculate_features(df)

        feature_data[symbol] = df

    cross_features = create_cross_asset_features(
        feature_data
    )

    for symbol in SYMBOLS:

        df = feature_data[symbol].copy()

        df = df.merge(
            cross_features,
            on="open_time",
            how="left",
        )

        feature_data[symbol] = df

    return feature_data


# ============================================================
# CONTINUITY CHECK
# ============================================================

def validate_continuity(
    timestamps: pd.Series,
) -> None:
    """Require exactly one row every minute."""

    timestamps = (
        pd.to_datetime(timestamps)
        .sort_values()
        .reset_index(drop=True)
    )

    if len(timestamps) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Only {len(timestamps)} timestamps available; "
            f"{SEQUENCE_LENGTH} required."
        )

    recent = timestamps.tail(SEQUENCE_LENGTH)

    diffs = recent.diff().dropna()

    bad = diffs[
        diffs != pd.Timedelta(minutes=1)
    ]

    if not bad.empty:

        raise ValueError(
            "The latest GRU sequence is not continuous. "
            f"Invalid interval: {bad.iloc[0]}"
        )


# ============================================================
# BUILD ONE SYMBOL SEQUENCE
# ============================================================

def build_symbol_sequence(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler,
    symbol: str,
) -> tuple[np.ndarray, pd.Timestamp, float]:
    """
    Build one scaled 60 x 61 GRU sequence.
    """

    df = (
        df.sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate feature columns
    # --------------------------------------------------------

    missing = [
        feature
        for feature in feature_columns
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol} missing features: {missing}"
        )

    # --------------------------------------------------------
    # Remove invalid feature rows
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=feature_columns
    ).reset_index(drop=True)

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"{symbol} has only {len(df)} valid feature rows; "
            f"{SEQUENCE_LENGTH} required."
        )

    # --------------------------------------------------------
    # Check the latest 60 timestamps
    # --------------------------------------------------------

    validate_continuity(
        df["open_time"]
    )

    recent = df.tail(SEQUENCE_LENGTH).copy()

    # --------------------------------------------------------
    # Raw feature matrix
    # --------------------------------------------------------

    X_raw = recent[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    if X_raw.shape != (
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    ):
        raise ValueError(
            f"{symbol} produced shape {X_raw.shape}; "
            f"expected ({SEQUENCE_LENGTH}, {FEATURE_COUNT})"
        )

    # --------------------------------------------------------
    # EXACT TRAINING SCALER
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X_raw
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    if not np.isfinite(X_scaled).all():
        raise ValueError(
            f"{symbol} sequence contains non-finite values."
        )

    latest_timestamp = recent[
        "open_time"
    ].iloc[-1]

    latest_close = float(
        recent["close"].iloc[-1]
    )

    return (
        X_scaled,
        latest_timestamp,
        latest_close,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CRYPTO MLOPS - LIVE GRU SEQUENCE BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate preprocessing artifacts
    # --------------------------------------------------------

    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Training scaler not found: {SCALER_FILE}"
        )

    feature_columns = (
        load_feature_columns()
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    if not hasattr(scaler, "transform"):
        raise TypeError(
            "Loaded scaler does not provide transform()."
        )

    print(
        f"Features : {len(feature_columns)}"
    )

    print(
        f"Sequence : {SEQUENCE_LENGTH} x "
        f"{FEATURE_COUNT}"
    )

    print(
        f"Scaler   : {SCALER_FILE}"
    )

    # --------------------------------------------------------
    # Build feature history
    # --------------------------------------------------------

    feature_data = (
        build_feature_history()
    )

    # --------------------------------------------------------
    # Build sequences for all symbols
    # --------------------------------------------------------

    sequences = []
    symbol_ids = []
    symbols = []

    metadata = []

    for symbol_id, symbol in enumerate(
        SYMBOLS
    ):

        print()
        print(
            f"Processing {symbol} "
            f"(symbol_id={symbol_id})..."
        )

        (
            sequence,
            latest_timestamp,
            latest_close,
        ) = build_symbol_sequence(
            df=feature_data[symbol],
            feature_columns=feature_columns,
            scaler=scaler,
            symbol=symbol,
        )

        sequences.append(
            sequence
        )

        symbol_ids.append(
            symbol_id
        )

        symbols.append(
            symbol
        )

        metadata.append(
            {
                "symbol": symbol,
                "symbol_id": symbol_id,
                "timestamp": str(
                    latest_timestamp
                ),
                "close": latest_close,
            }
        )

        print(
            f"Shape     : {sequence.shape}"
        )

        print(
            f"Timestamp : {latest_timestamp}"
        )

        print(
            f"Close     : {latest_close}"
        )

    # --------------------------------------------------------
    # Convert to model batch
    # --------------------------------------------------------

    X_batch = np.stack(
        sequences
    ).astype(
        np.float32
    )

    symbol_id_array = np.asarray(
        symbol_ids,
        dtype=np.int64,
    )

    # Expected:
    # (5, 60, 61)

    expected_shape = (
        len(SYMBOLS),
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    )

    if X_batch.shape != expected_shape:
        raise ValueError(
            f"Unexpected batch shape: "
            f"{X_batch.shape}; "
            f"expected {expected_shape}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    LIVE_SEQUENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        LIVE_SEQUENCE_FILE,
        X=X_batch,
        symbol_ids=symbol_id_array,
    )

    print()
    print("=" * 70)
    print("LIVE GRU SEQUENCE BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Batch shape: {X_batch.shape}"
    )

    print(
        f"Symbol IDs : {symbol_id_array.tolist()}"
    )

    print(
        f"Output     : {LIVE_SEQUENCE_FILE}"
    )


if __name__ == "__main__":
    main()