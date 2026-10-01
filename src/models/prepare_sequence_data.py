from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

SPLITS_DIR = Path(
    "data/processed/splits_v2"
)

OUTPUT_DIR = Path(
    "data/processed/sequence_v1"
)

SCALER_FILE = (
    OUTPUT_DIR / "feature_scaler.joblib"
)

FEATURE_COLUMNS_FILE = Path(
    "data/processed/feature_columns_v2.txt"
)

METADATA_FILE = (
    OUTPUT_DIR / "metadata.json"
)


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

SEQUENCE_LENGTH = 60
FORECAST_HORIZON = 5

SPLITS = [
    "train",
    "validation",
    "test",
]

CLASS_MAPPING = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}


# ============================================================
# FEATURE COLUMNS
# ============================================================

def load_feature_columns():
    """Load the V2 model feature list."""

    if not FEATURE_COLUMNS_FILE.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: "
            f"{FEATURE_COLUMNS_FILE}"
        )

    with open(
        FEATURE_COLUMNS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        columns = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not columns:
        raise ValueError(
            "Feature column file is empty."
        )

    return columns


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(split_name):
    """Load only the columns needed for sequence modeling."""

    input_file = (
        SPLITS_DIR
        / f"{split_name}.parquet"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"Split file not found: "
            f"{input_file}"
        )

    feature_columns = (
        load_feature_columns()
    )

    columns = [
        "open_time",
        "symbol",
        "close",
        "future_return_5m",
        "target_direction_5m",
    ] + feature_columns

    df = pd.read_parquet(
        input_file,
        columns=columns,
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"]
    )

    df = (
        df.sort_values(
            [
                "symbol",
                "open_time",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# VALID SEQUENCE INDICES
# ============================================================

def build_valid_end_indices(
    timestamps,
    sequence_length=SEQUENCE_LENGTH,
    forecast_horizon=FORECAST_HORIZON,
):
    """
    Return sequence-end indices where:

    - The previous sequence_length observations are
      consecutive 1-minute observations.
    - The next forecast_horizon observations are also
      consecutive 1-minute observations.
    """

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(timestamps)
    )

    total_rows = len(timestamps)

    required_rows = (
        sequence_length
        + forecast_horizon
    )

    if total_rows < required_rows:
        return np.array(
            [],
            dtype=np.int32,
        )

    one_minute = pd.Timedelta(
        minutes=1
    )

    valid_indices = []

    first_end_index = (
        sequence_length - 1
    )

    last_end_index = (
        total_rows
        - forecast_horizon
        - 1
    )

    for end_index in range(
        first_end_index,
        last_end_index + 1,
    ):

        # ----------------------------------------------------
        # Historical sequence
        # ----------------------------------------------------

        history = timestamps[
            end_index
            - sequence_length
            + 1 :
            end_index
            + 1
        ]

        history_diffs = (
            history[1:]
            - history[:-1]
        )

        history_is_continuous = (
            len(history_diffs)
            == sequence_length - 1
            and np.all(
                history_diffs == one_minute
            )
        )

        if not history_is_continuous:
            continue

        # ----------------------------------------------------
        # Future horizon
        # ----------------------------------------------------

        future = timestamps[
            end_index :
            end_index
            + forecast_horizon
            + 1
        ]

        future_diffs = (
            future[1:]
            - future[:-1]
        )

        future_is_continuous = (
            len(future_diffs)
            == forecast_horizon
            and np.all(
                future_diffs == one_minute
            )
        )

        if not future_is_continuous:
            continue

        valid_indices.append(
            end_index
        )

    return np.array(
        valid_indices,
        dtype=np.int32,
    )

# ============================================================
# FIT SCALER
# ============================================================

def fit_training_scaler(
    train_df,
    feature_columns,
):
    """
    Fit StandardScaler using TRAIN data only.

    Validation and test data never participate in fitting
    the scaler.
    """

    scaler = StandardScaler()

    for symbol in SYMBOLS:

        symbol_df = (
            train_df[
                train_df["symbol"] == symbol
            ]
            .sort_values("open_time")
        )

        X = (
            symbol_df[
                feature_columns
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        if len(X) == 0:
            raise ValueError(
                f"No training data for {symbol}"
            )

        scaler.partial_fit(X)

    return scaler


# ============================================================
# SAVE ONE SYMBOL
# ============================================================

def save_symbol_data(
    df,
    symbol,
    split_name,
    scaler,
    feature_columns,
):
    """Save standardized sequence inputs for one symbol."""

    symbol_df = (
        df[
            df["symbol"] == symbol
        ]
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    if symbol_df.empty:
        raise ValueError(
            f"No {split_name} data for {symbol}"
        )

    # --------------------------------------------------------
    # Raw features
    # --------------------------------------------------------

    X_raw = (
        symbol_df[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # Standardize using TRAIN scaler
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X_raw
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    regression_target = (
        symbol_df[
            "future_return_5m"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    direction_strings = (
        symbol_df[
            "target_direction_5m"
        ]
        .astype(str)
        .to_numpy()
    )

    classification_target = np.array(
        [
            CLASS_MAPPING[value]
            for value in direction_strings
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Metadata needed later
    # --------------------------------------------------------

    close_values = (
        symbol_df["close"]
        .to_numpy(
            dtype=np.float32
        )
    )

    time_values = (
        symbol_df["open_time"]
        .astype("int64")
        .to_numpy(
            dtype=np.int64
        )
    )

    # --------------------------------------------------------
    # Valid sequence endpoints
    # --------------------------------------------------------

    valid_indices = (
        build_valid_end_indices(
            symbol_df["open_time"]
        )
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    symbol_dir = (
        OUTPUT_DIR
        / split_name
    )

    symbol_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------

    np.save(
        symbol_dir
        / f"{symbol}_X.npy",
        X_scaled,
    )

    np.save(
        symbol_dir
        / f"{symbol}_return_5m.npy",
        regression_target,
    )

    np.save(
        symbol_dir
        / f"{symbol}_direction.npy",
        classification_target,
    )

    np.save(
        symbol_dir
        / f"{symbol}_close.npy",
        close_values,
    )

    np.save(
        symbol_dir
        / f"{symbol}_time.npy",
        time_values,
    )

    np.save(
        symbol_dir
        / f"{symbol}_valid_indices.npy",
        valid_indices,
    )

    return {
        "rows": len(symbol_df),
        "valid_sequences": len(
            valid_indices
        ),
        "feature_count": len(
            feature_columns
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRYPTO MLOPS - SEQUENCE DATA PREPARATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_columns = (
        load_feature_columns()
    )

    print()
    print(
        f"Feature columns: "
        f"{len(feature_columns)}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH} minutes"
    )

    print(
        f"Prediction horizon: "
        f"{FORECAST_HORIZON} minutes"
    )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print()
    print("Loading training data...")

    train_df = load_split(
        "train"
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    # --------------------------------------------------------
    # Fit scaler
    # --------------------------------------------------------

    print()
    print(
        "Fitting scaler using training "
        "data only..."
    )

    scaler = fit_training_scaler(
        train_df,
        feature_columns,
    )

    joblib.dump(
        scaler,
        SCALER_FILE,
    )

    print(
        f"Scaler saved to: "
        f"{SCALER_FILE}"
    )

    # --------------------------------------------------------
    # Prepare all splits
    # --------------------------------------------------------

    metadata = {
        "sequence_version": "v1",
        "sequence_length": (
            SEQUENCE_LENGTH
        ),
        "forecast_horizon": (
            FORECAST_HORIZON
        ),
        "feature_count": len(
            feature_columns
        ),
        "symbols": SYMBOLS,
        "class_mapping": CLASS_MAPPING,
        "splits": {},
    }

    for split_name in SPLITS:

        print()
        print("=" * 70)
        print(
            f"Preparing {split_name.upper()} "
            "sequence data..."
        )
        print("=" * 70)

        if split_name == "train":

            split_df = train_df

        else:

            split_df = load_split(
                split_name
            )

        metadata[
            "splits"
        ][split_name] = {}

        for symbol in SYMBOLS:

            print()
            print(
                f"Processing "
                f"{split_name} / "
                f"{symbol}..."
            )

            summary = save_symbol_data(
                df=split_df,
                symbol=symbol,
                split_name=split_name,
                scaler=scaler,
                feature_columns=feature_columns,
            )

            metadata[
                "splits"
            ][split_name][symbol] = summary

            print(
                f"Rows: "
                f"{summary['rows']:,}"
            )

            print(
                f"Valid sequences: "
                f"{summary['valid_sequences']:,}"
            )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print()
    print("=" * 70)
    print("SEQUENCE DATA PREPARATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Metadata: "
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()