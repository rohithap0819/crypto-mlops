"""
Diagnose the predictive signal available to the sequence models.

Uses the validation split only.

The sequence dataset is stored as:
    X -> rows x 61 features

The 60-minute sequence is constructed dynamically using valid_indices:
    sequence = X[end_index - 59 : end_index + 1]

Therefore this diagnostic uses the exact same sequence endpoints
as the training Dataset.

Diagnostics:
- Target distribution
- Zero-return regression baseline
- Previous 1-minute return baseline
- Previous 5-minute return baseline
- Majority-class classification baseline
- Momentum direction baseline
- Target autocorrelation
- Direction persistence
- Per-coin baseline metrics
- Feature/target correlations
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_v1"
)

VALIDATION_DIR = (
    SEQUENCE_DIR
    / "validation"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

FEATURE_COLUMNS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_columns_v2.txt"
)

SCALER_PATH = (
    SEQUENCE_DIR
    / "feature_scaler.joblib"
)

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

CLASS_NAMES = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}

SEQUENCE_LENGTH = 60

VOLATILITY_MULTIPLIER = 1.11803398875


# ============================================================
# FEATURE INDICES
# ============================================================

# Exact order comes from:
# data/processed/feature_columns_v2.txt

RETURN_1M_INDEX = 0
RETURN_5M_INDEX = 1
VOLATILITY_15M_INDEX = 25


# ============================================================
# HELPERS
# ============================================================

def inverse_scale_feature(
    values,
    feature_index,
    scaler,
):
    """
    Convert one standardized feature back to original scale.
    """

    return (
        values
        * scaler.scale_[feature_index]
        + scaler.mean_[feature_index]
    )


def regression_metrics(
    actual,
    predicted,
):
    """
    Calculate regression metrics.
    """

    return {
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "r2": float(
            r2_score(
                actual,
                predicted,
            )
        ),
    }


def classification_metrics(
    actual,
    predicted,
):
    """
    Calculate classification metrics.
    """

    return {
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                actual,
                predicted,
            )
        ),
    }


def autocorrelation(
    values,
    lag,
):
    """
    Calculate Pearson autocorrelation at a lag.
    """

    if len(values) <= lag:
        return float("nan")

    first = values[:-lag]
    second = values[lag:]

    if (
        np.std(first) == 0
        or np.std(second) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            first,
            second,
        )[0, 1]
    )


def direction_persistence(
    directions,
    lag,
):
    """
    Percentage of observations where the
    direction remains identical after lag.
    """

    if len(directions) <= lag:
        return float("nan")

    return float(
        np.mean(
            directions[:-lag]
            == directions[lag:]
        )
    )


def safe_correlation(
    first,
    second,
):
    """
    Calculate correlation while handling
    constant arrays safely.
    """

    if (
        len(first) == 0
        or len(second) == 0
    ):
        return float("nan")

    if (
        np.std(first) == 0
        or np.std(second) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            first,
            second,
        )[0, 1]
    )


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

feature_names = [
    line.strip()
    for line in FEATURE_COLUMNS_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

if len(feature_names) != 61:

    raise ValueError(
        "Expected exactly 61 feature names, "
        f"found {len(feature_names)}."
    )


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(
    SCALER_PATH
)


# ============================================================
# VERIFY SCALER
# ============================================================

if not hasattr(scaler, "mean_"):
    raise ValueError(
        "Loaded scaler does not contain mean_."
    )

if not hasattr(scaler, "scale_"):
    raise ValueError(
        "Loaded scaler does not contain scale_."
    )

if len(scaler.mean_) != 61:
    raise ValueError(
        "Scaler does not contain 61 features."
    )


# ============================================================
# RESULT STORAGE
# ============================================================

baseline_rows = []

feature_correlation_rows = []

all_targets = []

all_directions = []


# ============================================================
# PER-SYMBOL DIAGNOSTICS
# ============================================================

for symbol in SYMBOLS:

    print()
    print("=" * 70)
    print(f"SYMBOL: {symbol}")
    print("=" * 70)

    # --------------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------------

    x_path = (
        VALIDATION_DIR
        / f"{symbol}_X.npy"
    )

    return_path = (
        VALIDATION_DIR
        / f"{symbol}_return_5m.npy"
    )

    direction_path = (
        VALIDATION_DIR
        / f"{symbol}_direction.npy"
    )

    valid_path = (
        VALIDATION_DIR
        / f"{symbol}_valid_indices.npy"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    required_files = [
        x_path,
        return_path,
        direction_path,
        valid_path,
    ]

    for required_file in required_files:

        if not required_file.exists():

            raise FileNotFoundError(
                f"Missing file: "
                f"{required_file}"
            )

    # --------------------------------------------------------
    # LOAD FEATURE ARRAY
    # --------------------------------------------------------

    X = np.load(
        x_path,
        mmap_mode="r",
    )

    # Expected format:
    #
    # rows x 61
    #
    # NOT:
    # samples x 60 x 61
    #
    # The training Dataset constructs
    # the 60-step sequence dynamically.

    if X.ndim != 2:

        raise ValueError(
            f"{symbol}: expected X to be 2-dimensional, "
            f"found ndim={X.ndim}."
        )

    if X.shape[1] != 61:

        raise ValueError(
            f"{symbol}: expected 61 features, "
            f"found {X.shape[1]}."
        )

    # --------------------------------------------------------
    # LOAD TARGETS
    # --------------------------------------------------------

    returns = np.load(
        return_path,
        mmap_mode="r",
    )

    directions = np.load(
        direction_path,
        mmap_mode="r",
    )

    valid_indices = np.load(
        valid_path
    )

    # --------------------------------------------------------
    # VALIDATE ARRAY LENGTHS
    # --------------------------------------------------------

    if len(returns) != X.shape[0]:

        raise ValueError(
            f"{symbol}: return array length "
            f"{len(returns)} does not match "
            f"X rows {X.shape[0]}."
        )

    if len(directions) != X.shape[0]:

        raise ValueError(
            f"{symbol}: direction array length "
            f"{len(directions)} does not match "
            f"X rows {X.shape[0]}."
        )

    # --------------------------------------------------------
    # VALID INDICES
    # --------------------------------------------------------

    valid_indices = np.asarray(
        valid_indices,
        dtype=np.int64,
    )

    if len(valid_indices) == 0:

        raise ValueError(
            f"{symbol}: valid_indices is empty."
        )

    if np.min(valid_indices) < (
        SEQUENCE_LENGTH - 1
    ):

        raise ValueError(
            f"{symbol}: valid_indices contains "
            "an index too early for a 60-minute sequence."
        )

    if np.max(valid_indices) >= X.shape[0]:

        raise ValueError(
            f"{symbol}: valid_indices contains "
            "an out-of-range index."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Use exactly the same endpoint rows that
    # CryptoSequenceDataset uses during training.
    #
    # Training does:
    #
    # start_index = end_index - 59
    # sequence = X[start_index:end_index + 1]
    #
    # Therefore:
    #
    # X[valid_indices]
    #
    # is the final timestep available to the
    # model immediately before prediction.
    # --------------------------------------------------------

    last_step = np.asarray(
        X[valid_indices, :],
        dtype=np.float64,
    )

    target = np.asarray(
        returns[valid_indices],
        dtype=np.float64,
    )

    direction = np.asarray(
        directions[valid_indices],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # FINAL LENGTH CHECK
    # --------------------------------------------------------

    if len(target) != len(
        valid_indices
    ):

        raise ValueError(
            f"{symbol}: target length does not "
            "match valid_indices."
        )

    if len(direction) != len(
        valid_indices
    ):

        raise ValueError(
            f"{symbol}: direction length does not "
            "match valid_indices."
        )

    # --------------------------------------------------------
    # EXTRACT CURRENT FEATURES
    # --------------------------------------------------------

    previous_return_1m = (
        inverse_scale_feature(
            last_step[
                :,
                RETURN_1M_INDEX,
            ],
            RETURN_1M_INDEX,
            scaler,
        )
    )

    previous_return_5m = (
        inverse_scale_feature(
            last_step[
                :,
                RETURN_5M_INDEX,
            ],
            RETURN_5M_INDEX,
            scaler,
        )
    )

    volatility_15m = (
        inverse_scale_feature(
            last_step[
                :,
                VOLATILITY_15M_INDEX,
            ],
            VOLATILITY_15M_INDEX,
            scaler,
        )
    )

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    class_counts = np.bincount(
        direction,
        minlength=3,
    )

    class_percentages = (
        class_counts
        / len(direction)
        * 100.0
    )

    # --------------------------------------------------------
    # REGRESSION BASELINE 1
    #
    # Predict zero future return.
    # --------------------------------------------------------

    zero_prediction = np.zeros_like(
        target
    )

    zero_metrics = regression_metrics(
        target,
        zero_prediction,
    )

    # --------------------------------------------------------
    # REGRESSION BASELINE 2
    #
    # Predict next 5m return using current 1m return.
    # --------------------------------------------------------

    previous_1m_prediction = (
        previous_return_1m
    )

    previous_1m_metrics = (
        regression_metrics(
            target,
            previous_1m_prediction,
        )
    )

    # --------------------------------------------------------
    # REGRESSION BASELINE 3
    #
    # Predict future 5m return using current 5m return.
    # --------------------------------------------------------

    previous_5m_prediction = (
        previous_return_5m
    )

    previous_5m_metrics = (
        regression_metrics(
            target,
            previous_5m_prediction,
        )
    )

    # --------------------------------------------------------
    # CLASSIFICATION BASELINE 1
    #
    # Always predict the majority class.
    # --------------------------------------------------------

    majority_class = int(
        np.argmax(
            class_counts
        )
    )

    majority_prediction = np.full(
        len(direction),
        majority_class,
        dtype=np.int64,
    )

    majority_metrics = (
        classification_metrics(
            direction,
            majority_prediction,
        )
    )

    # --------------------------------------------------------
    # CLASSIFICATION BASELINE 2
    #
    # Use previous 5m return and current
    # 15m volatility to determine direction.
    #
    # Same class mapping:
    # DOWN    = 0
    # NEUTRAL = 1
    # UP      = 2
    # --------------------------------------------------------

    threshold = (
        volatility_15m
        * VOLATILITY_MULTIPLIER
    )

    momentum_prediction = np.full(
        len(direction),
        1,
        dtype=np.int64,
    )

    momentum_prediction[
        previous_return_5m > threshold
    ] = 2

    momentum_prediction[
        previous_return_5m < -threshold
    ] = 0

    momentum_metrics = (
        classification_metrics(
            direction,
            momentum_prediction,
        )
    )

    # --------------------------------------------------------
    # TARGET AUTOCORRELATION
    # --------------------------------------------------------

    target_acf_1 = autocorrelation(
        target,
        lag=1,
    )

    target_acf_5 = autocorrelation(
        target,
        lag=5,
    )

    target_acf_15 = autocorrelation(
        target,
        lag=15,
    )

    target_acf_60 = autocorrelation(
        target,
        lag=60,
    )

    # --------------------------------------------------------
    # DIRECTION PERSISTENCE
    # --------------------------------------------------------

    direction_persistence_1 = (
        direction_persistence(
            direction,
            lag=1,
        )
    )

    direction_persistence_5 = (
        direction_persistence(
            direction,
            lag=5,
        )
    )

    # --------------------------------------------------------
    # SIMPLE SIGNAL CORRELATIONS
    # --------------------------------------------------------

    previous_1m_target_corr = (
        safe_correlation(
            previous_return_1m,
            target,
        )
    )

    previous_5m_target_corr = (
        safe_correlation(
            previous_return_5m,
            target,
        )
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    baseline_rows.append(
        {
            "symbol": symbol,
            "samples": len(target),
            "target_mean": float(
                np.mean(target)
            ),
            "target_std": float(
                np.std(target)
            ),
            "down_pct": float(
                class_percentages[0]
            ),
            "neutral_pct": float(
                class_percentages[1]
            ),
            "up_pct": float(
                class_percentages[2]
            ),
            "majority_class": CLASS_NAMES[
                majority_class
            ],
            "zero_rmse": zero_metrics[
                "rmse"
            ],
            "zero_mae": zero_metrics[
                "mae"
            ],
            "zero_r2": zero_metrics[
                "r2"
            ],
            "previous_1m_rmse": (
                previous_1m_metrics[
                    "rmse"
                ]
            ),
            "previous_1m_mae": (
                previous_1m_metrics[
                    "mae"
                ]
            ),
            "previous_1m_r2": (
                previous_1m_metrics[
                    "r2"
                ]
            ),
            "previous_5m_rmse": (
                previous_5m_metrics[
                    "rmse"
                ]
            ),
            "previous_5m_mae": (
                previous_5m_metrics[
                    "mae"
                ]
            ),
            "previous_5m_r2": (
                previous_5m_metrics[
                    "r2"
                ]
            ),
            "majority_macro_f1": (
                majority_metrics[
                    "macro_f1"
                ]
            ),
            "majority_accuracy": (
                majority_metrics[
                    "accuracy"
                ]
            ),
            "momentum_macro_f1": (
                momentum_metrics[
                    "macro_f1"
                ]
            ),
            "momentum_accuracy": (
                momentum_metrics[
                    "accuracy"
                ]
            ),
            "target_acf_lag_1": (
                target_acf_1
            ),
            "target_acf_lag_5": (
                target_acf_5
            ),
            "target_acf_lag_15": (
                target_acf_15
            ),
            "target_acf_lag_60": (
                target_acf_60
            ),
            "direction_persistence_lag_1": (
                direction_persistence_1
            ),
            "direction_persistence_lag_5": (
                direction_persistence_5
            ),
            "previous_1m_target_corr": (
                previous_1m_target_corr
            ),
            "previous_5m_target_corr": (
                previous_5m_target_corr
            ),
        }
    )

    all_targets.append(
        target
    )

    all_directions.append(
        direction
    )

    # ========================================================
    # FEATURE/TARGET CORRELATIONS
    # ========================================================

    for feature_index, feature_name in (
        enumerate(feature_names)
    ):

        feature_values = (
            last_step[
                :,
                feature_index,
            ]
        )

        correlation = safe_correlation(
            feature_values,
            target,
        )

        feature_correlation_rows.append(
            {
                "symbol": symbol,
                "feature": feature_name,
                "feature_index": feature_index,
                "target_correlation": correlation,
                "absolute_correlation": (
                    abs(correlation)
                    if not np.isnan(
                        correlation
                    )
                    else np.nan
                ),
            }
        )

    # ========================================================
    # PRINT SYMBOL RESULTS
    # ========================================================

    print(
        f"Samples: "
        f"{len(target):,}"
    )

    print()
    print(
        f"Target mean: "
        f"{np.mean(target):.10f}"
    )

    print(
        f"Target std: "
        f"{np.std(target):.10f}"
    )

    print()
    print(
        "Class distribution:"
    )

    for class_id in range(3):

        print(
            f"  "
            f"{CLASS_NAMES[class_id]:8s}: "
            f"{class_counts[class_id]:,} "
            f"("
            f"{class_percentages[class_id]:.2f}%"
            f")"
        )

    print()
    print(
        "Regression baselines:"
    )

    print(
        f"  Zero return       "
        f"RMSE={zero_metrics['rmse']:.8f} "
        f"MAE={zero_metrics['mae']:.8f} "
        f"R2={zero_metrics['r2']:.6f}"
    )

    print(
        f"  Previous 1m       "
        f"RMSE={previous_1m_metrics['rmse']:.8f} "
        f"MAE={previous_1m_metrics['mae']:.8f} "
        f"R2={previous_1m_metrics['r2']:.6f}"
    )

    print(
        f"  Previous 5m       "
        f"RMSE={previous_5m_metrics['rmse']:.8f} "
        f"MAE={previous_5m_metrics['mae']:.8f} "
        f"R2={previous_5m_metrics['r2']:.6f}"
    )

    print()
    print(
        "Classification baselines:"
    )

    print(
        f"  Majority class    "
        f"Macro-F1="
        f"{majority_metrics['macro_f1']:.6f} "
        f"Accuracy="
        f"{majority_metrics['accuracy']:.6f}"
    )

    print(
        f"  Momentum          "
        f"Macro-F1="
        f"{momentum_metrics['macro_f1']:.6f} "
        f"Accuracy="
        f"{momentum_metrics['accuracy']:.6f}"
    )

    print()
    print(
        "Target autocorrelation:"
    )

    print(
        f"  Lag 1:            "
        f"{target_acf_1:.6f}"
    )

    print(
        f"  Lag 5:            "
        f"{target_acf_5:.6f}"
    )

    print(
        f"  Lag 15:           "
        f"{target_acf_15:.6f}"
    )

    print(
        f"  Lag 60:           "
        f"{target_acf_60:.6f}"
    )

    print()
    print(
        "Direction persistence:"
    )

    print(
        f"  Lag 1:            "
        f"{direction_persistence_1:.6f}"
    )

    print(
        f"  Lag 5:            "
        f"{direction_persistence_5:.6f}"
    )

    print()
    print(
        "Simple signal correlations:"
    )

    print(
        f"  Previous 1m -> future 5m: "
        f"{previous_1m_target_corr:.6f}"
    )

    print(
        f"  Previous 5m -> future 5m: "
        f"{previous_5m_target_corr:.6f}"
    )


# ============================================================
# POOLED VALIDATION DATA
# ============================================================

pooled_targets = np.concatenate(
    all_targets
)

pooled_directions = np.concatenate(
    all_directions
)


# ============================================================
# POOLED REGRESSION BASELINES
# ============================================================

pooled_zero_prediction = np.zeros_like(
    pooled_targets
)

pooled_regression_zero = (
    regression_metrics(
        pooled_targets,
        pooled_zero_prediction,
    )
)


# ============================================================
# POOLED CLASS DISTRIBUTION
# ============================================================

pooled_class_counts = np.bincount(
    pooled_directions,
    minlength=3,
)

pooled_class_percentages = (
    pooled_class_counts
    / len(pooled_directions)
    * 100.0
)

pooled_majority_class = int(
    np.argmax(
        pooled_class_counts
    )
)

pooled_majority_prediction = np.full(
    len(pooled_directions),
    pooled_majority_class,
    dtype=np.int64,
)

pooled_majority_metrics = (
    classification_metrics(
        pooled_directions,
        pooled_majority_prediction,
    )
)


# ============================================================
# SAVE BASELINE REPORT
# ============================================================

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

baseline_df = pd.DataFrame(
    baseline_rows
)

baseline_path = (
    REPORTS_DIR
    / "sequence_signal_diagnostics_v1.csv"
)

baseline_df.to_csv(
    baseline_path,
    index=False,
)


# ============================================================
# SAVE FEATURE CORRELATIONS
# ============================================================

correlation_df = pd.DataFrame(
    feature_correlation_rows
)

correlation_path = (
    REPORTS_DIR
    / "sequence_feature_correlations_v1.csv"
)

correlation_df.to_csv(
    correlation_path,
    index=False,
)


# ============================================================
# AGGREGATE FEATURE CORRELATIONS
# ============================================================

pooled_feature_corr = (
    correlation_df
    .groupby(
        [
            "feature_index",
            "feature",
        ],
        as_index=False,
    )[
        "target_correlation"
    ]
    .mean()
)

pooled_feature_corr[
    "absolute_correlation"
] = (
    pooled_feature_corr[
        "target_correlation"
    ].abs()
)

pooled_feature_corr = (
    pooled_feature_corr
    .sort_values(
        "absolute_correlation",
        ascending=False,
    )
)


# ============================================================
# FINAL POOLED SUMMARY
# ============================================================

print()
print("=" * 70)
print(
    "POOLED VALIDATION SUMMARY"
)
print("=" * 70)

print()
print(
    f"Validation samples: "
    f"{len(pooled_targets):,}"
)

print()
print(
    "Pooled target distribution:"
)

for class_id in range(3):

    print(
        f"  "
        f"{CLASS_NAMES[class_id]:8s}: "
        f"{pooled_class_counts[class_id]:,} "
        f"("
        f"{pooled_class_percentages[class_id]:.2f}%"
        f")"
    )

print()
print(
    "Pooled zero-return baseline:"
)

print(
    f"  RMSE: "
    f"{pooled_regression_zero['rmse']:.8f}"
)

print(
    f"  MAE:  "
    f"{pooled_regression_zero['mae']:.8f}"
)

print(
    f"  R2:   "
    f"{pooled_regression_zero['r2']:.6f}"
)

print()
print(
    "Pooled majority-class baseline:"
)

print(
    f"  Class: "
    f"{CLASS_NAMES[pooled_majority_class]}"
)

print(
    f"  Macro-F1: "
    f"{pooled_majority_metrics['macro_f1']:.6f}"
)

print(
    f"  Accuracy: "
    f"{pooled_majority_metrics['accuracy']:.6f}"
)


# ============================================================
# TOP FEATURE CORRELATIONS
# ============================================================

print()
print(
    "Top 15 feature/target correlations:"
)

top_features = (
    pooled_feature_corr[
        [
            "feature",
            "target_correlation",
            "absolute_correlation",
        ]
    ]
    .head(15)
)

print(
    top_features.to_string(
        index=False
    )
)


# ============================================================
# REPORT PATHS
# ============================================================

print()
print("=" * 70)
print(
    "REPORTS SAVED"
)
print("=" * 70)

print(
    f"Baseline diagnostics:"
)

print(
    baseline_path
)

print()
print(
    f"Feature correlations:"
)

print(
    correlation_path
)