from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/crypto_ml_dataset.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/splits"
)

MANIFEST_FILE = OUTPUT_DIR / "split_manifest.json"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Maximum prediction horizon = 5 minutes.
GAP_MINUTES = 5


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load and chronologically sort the ML dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df = df.sort_values(
        ["open_time", "symbol"]
    ).reset_index(drop=True)

    return df


# ============================================================
# SPLIT DATA
# ============================================================

def prepare_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Create chronological train/validation/test splits.

    70% train
    15% validation
    15% test

    A 5-minute temporal gap is inserted between
    each split because the largest target horizon is
    5 minutes.
    """

    if not 0 < TRAIN_RATIO < 1:
        raise ValueError("TRAIN_RATIO must be between 0 and 1.")

    if not 0 < VALIDATION_RATIO < 1:
        raise ValueError(
            "VALIDATION_RATIO must be between 0 and 1."
        )

    if abs(
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
        - 1.0
    ) > 1e-9:
        raise ValueError(
            "Train/validation/test ratios must sum to 1."
        )

    timestamps = (
        df["open_time"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    n_timestamps = len(timestamps)

    if n_timestamps < 100:
        raise ValueError(
            "Not enough timestamps for time-series splitting."
        )

    train_index = int(
        n_timestamps * TRAIN_RATIO
    )

    validation_index = int(
        n_timestamps
        * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_boundary = timestamps.iloc[
        train_index
    ]

    validation_boundary = timestamps.iloc[
        validation_index
    ]

    gap = pd.Timedelta(
        minutes=GAP_MINUTES
    )

    train_end = train_boundary - gap
    validation_start = train_boundary + gap

    validation_end = validation_boundary - gap
    test_start = validation_boundary + gap

    train_df = df[
        df["open_time"] <= train_end
    ].copy()

    validation_df = df[
        (df["open_time"] >= validation_start)
        & (df["open_time"] <= validation_end)
    ].copy()

    test_df = df[
        df["open_time"] >= test_start
    ].copy()

    # --------------------------------------------------------
    # Validation checks
    # --------------------------------------------------------

    if train_df.empty:
        raise ValueError("Training split is empty.")

    if validation_df.empty:
        raise ValueError("Validation split is empty.")

    if test_df.empty:
        raise ValueError("Test split is empty.")

    if (
        train_df["open_time"].max()
        >= validation_df["open_time"].min()
    ):
        raise ValueError(
            "Training and validation periods overlap."
        )

    if (
        validation_df["open_time"].max()
        >= test_df["open_time"].min()
    ):
        raise ValueError(
            "Validation and test periods overlap."
        )

    manifest = {
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "gap_minutes": GAP_MINUTES,
        "train": {
            "rows": len(train_df),
            "start": str(
                train_df["open_time"].min()
            ),
            "end": str(
                train_df["open_time"].max()
            ),
        },
        "validation": {
            "rows": len(validation_df),
            "start": str(
                validation_df["open_time"].min()
            ),
            "end": str(
                validation_df["open_time"].max()
            ),
        },
        "test": {
            "rows": len(test_df),
            "start": str(
                test_df["open_time"].min()
            ),
            "end": str(
                test_df["open_time"].max()
            ),
        },
    }

    return (
        train_df,
        validation_df,
        test_df,
        manifest,
    )


# ============================================================
# SAVE SPLITS
# ============================================================

def save_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    manifest: dict,
) -> None:
    """Save the three datasets and split metadata."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_parquet(
        OUTPUT_DIR / "train.parquet",
        index=False,
    )

    validation_df.to_parquet(
        OUTPUT_DIR / "validation.parquet",
        index=False,
    )

    test_df.to_parquet(
        OUTPUT_DIR / "test.parquet",
        index=False,
    )

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("CRYPTO MLOPS — TIME SERIES DATA SPLIT")
    print("=" * 70)

    df = load_dataset()

    print(
        f"Total rows: {len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['open_time'].min()} "
        f"→ "
        f"{df['open_time'].max()}"
    )

    train_df, validation_df, test_df, manifest = (
        prepare_splits(df)
    )

    save_splits(
        train_df,
        validation_df,
        test_df,
        manifest,
    )

    print()
    print("=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    print(
        f"Train rows:      {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    print(
        f"Test rows:       {len(test_df):,}"
    )

    print()
    print("TRAIN")
    print(
        train_df["open_time"].min(),
        "→",
        train_df["open_time"].max(),
    )

    print()
    print("VALIDATION")
    print(
        validation_df["open_time"].min(),
        "→",
        validation_df["open_time"].max(),
    )

    print()
    print("TEST")
    print(
        test_df["open_time"].min(),
        "→",
        test_df["open_time"].max(),
    )

    print()
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()