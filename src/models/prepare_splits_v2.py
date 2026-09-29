from pathlib import Path

import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/processed/crypto_ml_dataset_v2.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/splits_v2"
)

TRAIN_FILE = OUTPUT_DIR / "train.parquet"
VALIDATION_FILE = OUTPUT_DIR / "validation.parquet"
TEST_FILE = OUTPUT_DIR / "test.parquet"
MANIFEST_FILE = OUTPUT_DIR / "split_manifest.json"


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Maximum prediction horizon is 5 minutes.
# Keep a 5-minute temporal gap between the partitions.
GAP_MINUTES = 5


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    """Load and chronologically sort the V2 feature dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    required_columns = [
        "open_time",
        "symbol",
        "next_return_1m",
        "future_return_5m",
        "target_direction_5m",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"]
    )

    df = (
        df.sort_values(
            ["open_time", "symbol"]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# SPLIT CALCULATION
# ============================================================

def calculate_split_times(df):
    """
    Calculate chronological train, validation and test
    boundaries using unique timestamps.
    """

    timestamps = (
        df["open_time"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    total_timestamps = len(timestamps)

    train_index = int(
        total_timestamps * TRAIN_RATIO
    )

    validation_index = int(
        total_timestamps
        * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_end = timestamps.iloc[
        train_index - 1
    ]

    validation_start = timestamps.iloc[
        train_index
    ]

    validation_end = timestamps.iloc[
        validation_index - 1
    ]

    test_start = timestamps.iloc[
        validation_index
    ]

    test_end = timestamps.iloc[
        -1
    ]

    return {
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "test_end": test_end,
        "total_timestamps": total_timestamps,
    }


# ============================================================
# TEMPORAL GAP
# ============================================================

def apply_temporal_gap(
    train_end,
    validation_start,
    validation_end,
    test_start,
):
    """
    Add a 5-minute gap around validation/test boundaries
    so future target windows cannot cross partitions.
    """

    train_cutoff = (
        validation_start
        - pd.Timedelta(
            minutes=GAP_MINUTES
        )
    )

    validation_start_cutoff = (
        validation_start
        + pd.Timedelta(
            minutes=GAP_MINUTES
        )
    )

    validation_end_cutoff = (
        test_start
        - pd.Timedelta(
            minutes=GAP_MINUTES
        )
    )

    test_start_cutoff = (
        test_start
        + pd.Timedelta(
            minutes=GAP_MINUTES
        )
    )

    return {
        "train_cutoff": train_cutoff,
        "validation_start_cutoff": (
            validation_start_cutoff
        ),
        "validation_end_cutoff": (
            validation_end_cutoff
        ),
        "test_start_cutoff": test_start_cutoff,
    }


# ============================================================
# CREATE SPLITS
# ============================================================

def create_splits(df):
    """Create leakage-safe chronological partitions."""

    split_times = calculate_split_times(df)

    gaps = apply_temporal_gap(
        train_end=split_times["train_end"],
        validation_start=split_times[
            "validation_start"
        ],
        validation_end=split_times[
            "validation_end"
        ],
        test_start=split_times[
            "test_start"
        ],
    )

    train = df[
        df["open_time"]
        <= gaps["train_cutoff"]
    ].copy()

    validation = df[
        (
            df["open_time"]
            >= gaps["validation_start_cutoff"]
        )
        & (
            df["open_time"]
            <= gaps["validation_end_cutoff"]
        )
    ].copy()

    test = df[
        df["open_time"]
        >= gaps["test_start_cutoff"]
    ].copy()

    return train, validation, test, split_times, gaps


# ============================================================
# VALIDATION
# ============================================================

def validate_splits(
    train,
    validation,
    test,
):
    """Verify chronological ordering and no overlap."""

    if train.empty:
        raise ValueError(
            "Training split is empty."
        )

    if validation.empty:
        raise ValueError(
            "Validation split is empty."
        )

    if test.empty:
        raise ValueError(
            "Test split is empty."
        )

    train_max = train["open_time"].max()
    validation_min = validation["open_time"].min()
    validation_max = validation["open_time"].max()
    test_min = test["open_time"].min()

    if not (
        train_max < validation_min
        and validation_max < test_min
    ):
        raise ValueError(
            "Chronological split ordering failed."
        )

    train_timestamps = set(
        train["open_time"].unique()
    )

    validation_timestamps = set(
        validation["open_time"].unique()
    )

    test_timestamps = set(
        test["open_time"].unique()
    )

    if train_timestamps & validation_timestamps:
        raise ValueError(
            "Train and validation timestamps overlap."
        )

    if train_timestamps & test_timestamps:
        raise ValueError(
            "Train and test timestamps overlap."
        )

    if validation_timestamps & test_timestamps:
        raise ValueError(
            "Validation and test timestamps overlap."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRYPTO MLOPS - V2 TIME SERIES SPLIT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    print()
    print(
        f"Total rows: {len(df):,}"
    )

    print(
        f"Unique timestamps: "
        f"{df['open_time'].nunique():,}"
    )

    train, validation, test, split_times, gaps = (
        create_splits(df)
    )

    validate_splits(
        train,
        validation,
        test,
    )

    # --------------------------------------------------------
    # Save partitions
    # --------------------------------------------------------

    train.to_parquet(
        TRAIN_FILE,
        index=False,
    )

    validation.to_parquet(
        VALIDATION_FILE,
        index=False,
    )

    test.to_parquet(
        TEST_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest = {
        "dataset": str(INPUT_FILE),
        "feature_version": "v2",
        "total_rows": int(len(df)),
        "total_timestamps": int(
            split_times["total_timestamps"]
        ),
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "gap_minutes": GAP_MINUTES,
        "train": {
            "rows": int(len(train)),
            "start": train["open_time"].min().isoformat(),
            "end": train["open_time"].max().isoformat(),
        },
        "validation": {
            "rows": int(len(validation)),
            "start": validation["open_time"].min().isoformat(),
            "end": validation["open_time"].max().isoformat(),
        },
        "test": {
            "rows": int(len(test)),
            "start": test["open_time"].min().isoformat(),
            "end": test["open_time"].max().isoformat(),
        },
        "original_boundaries": {
            "train_end": split_times[
                "train_end"
            ].isoformat(),
            "validation_start": split_times[
                "validation_start"
            ].isoformat(),
            "validation_end": split_times[
                "validation_end"
            ].isoformat(),
            "test_start": split_times[
                "test_start"
            ].isoformat(),
            "test_end": split_times[
                "test_end"
            ].isoformat(),
        },
        "applied_boundaries": {
            "train_cutoff": gaps[
                "train_cutoff"
            ].isoformat(),
            "validation_start_cutoff": gaps[
                "validation_start_cutoff"
            ].isoformat(),
            "validation_end_cutoff": gaps[
                "validation_end_cutoff"
            ].isoformat(),
            "test_start_cutoff": gaps[
                "test_start_cutoff"
            ].isoformat(),
        },
    }

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V2 SPLIT COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Train rows:      {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Test rows:       {len(test):,}"
    )

    print()
    print(
        f"Train period: "
        f"{train['open_time'].min()} "
        f"→ "
        f"{train['open_time'].max()}"
    )

    print(
        f"Validation period: "
        f"{validation['open_time'].min()} "
        f"→ "
        f"{validation['open_time'].max()}"
    )

    print(
        f"Test period: "
        f"{test['open_time'].min()} "
        f"→ "
        f"{test['open_time'].max()}"
    )

    print()
    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()