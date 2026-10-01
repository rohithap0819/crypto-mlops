"""
Diagnose simple baselines for pure 5-minute direction prediction.

Target:
    future_return_5m > 0 -> UP
    future_return_5m < 0 -> DOWN

Exact-zero future returns are excluded.

Baselines:
    1. Majority class
    2. Previous 1m return sign
    3. Previous 5m return sign
    4. Market 1m return sign
    5. Market 5m return sign
    6. Opposite previous 1m sign
    7. Opposite previous 5m sign

Purpose:
    Determine how much value the LightGBM model is adding
    beyond simple directional rules.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits_v2"
    / "validation.parquet"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = (
    "future_return_5m"
)

REQUIRED_COLUMNS = [
    TARGET_COLUMN,
    "return_1m",
    "return_5m",
    "market_return_1m",
    "market_return_5m",
    "symbol",
]


# ============================================================
# HELPERS
# ============================================================

def evaluate(
    name,
    actual,
    predicted,
):
    """
    Evaluate one directional baseline.
    """

    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        actual,
        predicted,
    )

    return {
        "baseline": name,
        "macro_f1": float(
            macro_f1
        ),
        "accuracy": float(
            accuracy
        ),
    }


def sign_to_direction(
    values,
):
    """
    Convert numeric returns to:

        -1 -> DOWN = 0
        +1 -> UP   = 1

    Zero is handled separately before this function.
    """

    return (
        values > 0
    ).astype(
        np.int64
    )


# ============================================================
# LOAD DATA
# ============================================================

if not VALIDATION_FILE.exists():

    raise FileNotFoundError(
        f"Validation file not found:\n"
        f"{VALIDATION_FILE}"
    )

print(
    "Loading validation data..."
)

df = pd.read_parquet(
    VALIDATION_FILE
)

print(
    f"Validation rows: "
    f"{len(df):,}"
)


# ============================================================
# CHECK COLUMNS
# ============================================================

missing = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns:\n"
        f"{missing}"
    )


# ============================================================
# CLEAN TARGET
# ============================================================

df = df.replace(
    [
        np.inf,
        -np.inf,
    ],
    np.nan,
)

df = df.dropna(
    subset=REQUIRED_COLUMNS
)

# Pure direction is undefined for exact-zero
# future returns, so remove those rows.
df = df[
    df[TARGET_COLUMN] != 0
].copy()

print(
    f"Rows after zero-return removal: "
    f"{len(df):,}"
)


# ============================================================
# TARGET
# ============================================================

actual = (
    df[TARGET_COLUMN] > 0
).astype(
    np.int64
)


# ============================================================
# BASELINE PREDICTIONS
# ============================================================

results = []


# ------------------------------------------------------------
# 1. MAJORITY CLASS
# ------------------------------------------------------------

majority_class = int(
    actual.value_counts().idxmax()
)

majority_prediction = np.full(
    len(actual),
    majority_class,
    dtype=np.int64,
)

results.append(
    evaluate(
        "majority",
        actual,
        majority_prediction,
    )
)


# ------------------------------------------------------------
# 2. PREVIOUS 1M SIGN
# ------------------------------------------------------------

previous_1m = (
    df["return_1m"]
    .to_numpy(
        dtype=np.float64
    )
)

prediction_1m = (
    previous_1m > 0
).astype(
    np.int64
)

results.append(
    evaluate(
        "previous_1m_sign",
        actual,
        prediction_1m,
    )
)


# ------------------------------------------------------------
# 3. PREVIOUS 5M SIGN
# ------------------------------------------------------------

previous_5m = (
    df["return_5m"]
    .to_numpy(
        dtype=np.float64
    )
)

prediction_5m = (
    previous_5m > 0
).astype(
    np.int64
)

results.append(
    evaluate(
        "previous_5m_sign",
        actual,
        prediction_5m,
    )
)


# ------------------------------------------------------------
# 4. MARKET 1M SIGN
# ------------------------------------------------------------

market_1m = (
    df["market_return_1m"]
    .to_numpy(
        dtype=np.float64
    )
)

prediction_market_1m = (
    market_1m > 0
).astype(
    np.int64
)

results.append(
    evaluate(
        "market_1m_sign",
        actual,
        prediction_market_1m,
    )
)


# ------------------------------------------------------------
# 5. MARKET 5M SIGN
# ------------------------------------------------------------

market_5m = (
    df["market_return_5m"]
    .to_numpy(
        dtype=np.float64
    )
)

prediction_market_5m = (
    market_5m > 0
).astype(
    np.int64
)

results.append(
    evaluate(
        "market_5m_sign",
        actual,
        prediction_market_5m,
    )
)


# ------------------------------------------------------------
# 6. OPPOSITE PREVIOUS 1M SIGN
# ------------------------------------------------------------

prediction_inverse_1m = (
    previous_1m < 0
).astype(
    np.int64
)

results.append(
    evaluate(
        "inverse_previous_1m",
        actual,
        prediction_inverse_1m,
    )
)


# ------------------------------------------------------------
# 7. OPPOSITE PREVIOUS 5M SIGN
# ------------------------------------------------------------

prediction_inverse_5m = (
    previous_5m < 0
).astype(
    np.int64
)

results.append(
    evaluate(
        "inverse_previous_5m",
        actual,
        prediction_inverse_5m,
    )
)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        "accuracy",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print(
    "SIMPLE DIRECTIONAL BASELINES"
)
print("=" * 70)

print()

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print(
    "Target distribution:"
)

down_count = int(
    np.sum(actual == 0)
)

up_count = int(
    np.sum(actual == 1)
)

print(
    f"  DOWN: {down_count:,} "
    f"({down_count / len(actual) * 100:.2f}%)"
)

print(
    f"  UP:   {up_count:,} "
    f"({up_count / len(actual) * 100:.2f}%)"
)


# ============================================================
# REPORT FOR BEST SIMPLE BASELINE
# ============================================================

best_baseline = (
    results_df.iloc[0]["baseline"]
)

print()
print(
    f"Highest-accuracy simple baseline: "
    f"{best_baseline}"
)


# ============================================================
# SAVE
# ============================================================

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = (
    REPORTS_DIR
    / "binary_direction_baselines_v1.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print()
print(
    "Report saved:"
)

print(
    output_path
)


# ============================================================
# LIGHTGBM REFERENCE
# ============================================================

print()
print("=" * 70)
print(
    "REFERENCE: LIGHTGBM"
)
print("=" * 70)

print(
    "Macro-F1: 0.522568"
)

print(
    "Accuracy: 0.522625"
)

print()
print(
    "Compare these values with the simple "
    "baselines above."
)