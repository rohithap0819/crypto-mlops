"""
Compare 5-minute magnitude-aware targets.

Target definition for threshold T:

    future_return_5m > +T  -> UP
    future_return_5m < -T  -> DOWN
    otherwise               -> HOLD

Validation only.
The test set is NOT used.

Purpose:
    Determine whether ignoring tiny 5-minute moves produces a more
    useful trading-oriented target than the current pure direction target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits_v2"
    / "train.parquet"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits_v2"
    / "validation.parquet"
)

FEATURE_COLUMNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_columns_v2.txt"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
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

TARGET_COLUMN = "future_return_5m"

# 0.05%, 0.10%, 0.15%, 0.20%, 0.30%
THRESHOLDS = [
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0030,
]

RANDOM_STATE = 42


# ============================================================
# FEATURES
# ============================================================

def load_feature_columns() -> list[str]:

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

    if len(columns) != 61:
        raise ValueError(
            f"Expected 61 features, found {len(columns)}"
        )

    return columns


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns:\n{missing}"
        )

    X = df[
        feature_columns
    ].copy()

    symbol_dummies = pd.get_dummies(
        df["symbol"],
        prefix="symbol",
        dtype=np.float32,
    )

    expected_symbols = [
        f"symbol_{symbol}"
        for symbol in SYMBOLS
    ]

    for column in expected_symbols:
        if column not in symbol_dummies.columns:
            symbol_dummies[column] = 0.0

    symbol_dummies = symbol_dummies[
        expected_symbols
    ]

    return pd.concat(
        [
            X.reset_index(drop=True),
            symbol_dummies.reset_index(drop=True),
        ],
        axis=1,
    )


# ============================================================
# TARGET
# ============================================================

def create_magnitude_target(
    future_returns: pd.Series,
    threshold: float,
) -> np.ndarray:

    """
    Class mapping:

        0 = DOWN
        1 = HOLD
        2 = UP
    """

    values = (
        future_returns
        .to_numpy(
            dtype=float
        )
    )

    target = np.full(
        len(values),
        1,
        dtype=np.int64,
    )

    target[
        values > threshold
    ] = 2

    target[
        values < -threshold
    ] = 0

    return target


# ============================================================
# DATASET
# ============================================================

def prepare_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
):

    clean = df[
        df[TARGET_COLUMN].notna()
    ].copy()

    clean = clean.reset_index(
        drop=True
    )

    X = prepare_features(
        clean,
        feature_columns,
    )

    y = create_magnitude_target(
        clean[
            TARGET_COLUMN
        ],
        threshold,
    )

    return X, y, clean


# ============================================================
# MODEL
# ============================================================

def build_model():

    return CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    # Actionable predictions = UP or DOWN.
    actionable_mask = (
        y_pred != 1
    )

    actionable_count = int(
        actionable_mask.sum()
    )

    coverage = (
        actionable_count
        /
        len(y_pred)
    )

    if actionable_count > 0:

        actionable_accuracy = (
            accuracy_score(
                y_true[
                    actionable_mask
                ]
                != 1,
                y_pred[
                    actionable_mask
                ]
                != 1,
            )
        )

    else:

        actionable_accuracy = np.nan

    return (
        accuracy,
        macro_f1,
        actionable_count,
        coverage,
        actionable_accuracy,
    )


# ============================================================
# GROSS RETURN DIAGNOSTIC
# ============================================================

def calculate_directional_return(
    future_returns: np.ndarray,
    predictions: np.ndarray,
):

    actionable_mask = (
        predictions != 1
    )

    if actionable_mask.sum() == 0:

        return {
            "mean_action_return":
                np.nan,
            "median_action_return":
                np.nan,
            "positive_action_rate":
                np.nan,
        }

    actionable_returns = np.where(
        predictions[
            actionable_mask
        ] == 2,
        future_returns[
            actionable_mask
        ],
        -
        future_returns[
            actionable_mask
        ],
    )

    return {
        "mean_action_return":
            float(
                actionable_returns.mean()
            ),
        "median_action_return":
            float(
                np.median(
                    actionable_returns
                )
            ),
        "positive_action_rate":
            float(
                np.mean(
                    actionable_returns > 0
                )
            ),
    }


# ============================================================
# SIMPLE PREVIOUS-RETURN BASELINE
# ============================================================

def inverse_momentum_baseline(
    df: pd.DataFrame,
    threshold: float,
):

    """
    Simple mean-reversion baseline:

        previous return > +T -> DOWN
        previous return < -T -> UP
        otherwise             -> HOLD

    Uses only information available at the current timestamp.
    """

    previous_return = (
        df[
            "return_5m"
        ]
        .to_numpy(
            dtype=float
        )
    )

    future_return = (
        df[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    valid = (
        np.isfinite(
            previous_return
        )
        &
        np.isfinite(
            future_return
        )
    )

    previous_return = (
        previous_return[
            valid
        ]
    )

    future_return = (
        future_return[
            valid
        ]
    )

    target = create_magnitude_target(
        pd.Series(
            future_return
        ),
        threshold,
    )

    prediction = np.full(
        len(previous_return),
        1,
        dtype=np.int64,
    )

    prediction[
        previous_return < -threshold
    ] = 2

    prediction[
        previous_return > threshold
    ] = 0

    accuracy = accuracy_score(
        target,
        prediction,
    )

    macro_f1 = f1_score(
        target,
        prediction,
        average="macro",
    )

    actionable_mask = (
        prediction != 1
    )

    coverage = (
        actionable_mask.mean()
    )

    if actionable_mask.sum() > 0:

        actionable_accuracy = (
            accuracy_score(
                target[
                    actionable_mask
                ] != 1,
                prediction[
                    actionable_mask
                ] != 1,
            )
        )

        action_returns = np.where(
            prediction[
                actionable_mask
            ] == 2,
            future_return[
                actionable_mask
            ],
            -
            future_return[
                actionable_mask
            ],
        )

        mean_action_return = (
            action_returns.mean()
        )

    else:

        actionable_accuracy = np.nan
        mean_action_return = np.nan

    return {
        "accuracy":
            accuracy,
        "macro_f1":
            macro_f1,
        "coverage":
            coverage,
        "actionable_accuracy":
            actionable_accuracy,
        "mean_action_return":
            mean_action_return,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "5-MINUTE MAGNITUDE / HOLD TARGET COMPARISON"
    )
    print("=" * 70)

    print(
        "\nTest set will NOT be used."
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "\nLoading train and validation data..."
    )

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    feature_columns = (
        load_feature_columns()
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    results = []

    # --------------------------------------------------------
    # Threshold loop
    # --------------------------------------------------------

    for threshold in THRESHOLDS:

        threshold_pct = (
            threshold * 100
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"THRESHOLD: "
            f"{threshold_pct:.2f}%"
        )

        print(
            "=" * 70
        )

        (
            X_train,
            y_train,
            train_clean,
        ) = prepare_dataset(
            train_df,
            feature_columns,
            threshold,
        )

        (
            X_validation,
            y_validation,
            validation_clean,
        ) = prepare_dataset(
            validation_df,
            feature_columns,
            threshold,
        )

        print(
            f"Train samples: "
            f"{len(X_train):,}"
        )

        print(
            f"Validation samples: "
            f"{len(X_validation):,}"
        )

        # ----------------------------------------------------
        # Actual target distribution
        # ----------------------------------------------------

        train_counts = (
            pd.Series(
                y_train
            )
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        )

        validation_counts = (
            pd.Series(
                y_validation
            )
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        )

        print(
            "\nValidation target distribution:"
        )

        print(
            f"DOWN  : "
            f"{validation_counts.get(0, 0):.2f}%"
        )

        print(
            f"HOLD  : "
            f"{validation_counts.get(1, 0):.2f}%"
        )

        print(
            f"UP    : "
            f"{validation_counts.get(2, 0):.2f}%"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = build_model()

        print(
            "\nTraining CatBoost..."
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_validation
            )
        )

        predictions = (
            np.argmax(
                probabilities,
                axis=1,
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        (
            accuracy,
            macro_f1,
            actionable_count,
            coverage,
            actionable_accuracy,
        ) = calculate_metrics(
            y_validation,
            predictions,
        )

        future_returns = (
            validation_clean[
                TARGET_COLUMN
            ]
            .to_numpy(
                dtype=float
            )
        )

        return_metrics = (
            calculate_directional_return(
                future_returns,
                predictions,
            )
        )

        baseline = (
            inverse_momentum_baseline(
                validation_clean,
                threshold,
            )
        )

        print(
            "\nCatBoost:"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.6f}"
        )

        print(
            f"Macro-F1: "
            f"{macro_f1:.6f}"
        )

        print(
            f"Actionable coverage: "
            f"{coverage * 100:.2f}%"
        )

        print(
            f"Actionable directional accuracy: "
            f"{actionable_accuracy:.6f}"
        )

        print(
            f"Mean return per action: "
            f"{return_metrics['mean_action_return']:.8f}"
        )

        print(
            "\nInverse-momentum baseline:"
        )

        print(
            f"Accuracy: "
            f"{baseline['accuracy']:.6f}"
        )

        print(
            f"Macro-F1: "
            f"{baseline['macro_f1']:.6f}"
        )

        print(
            f"Coverage: "
            f"{baseline['coverage'] * 100:.2f}%"
        )

        print(
            f"Actionable directional accuracy: "
            f"{baseline['actionable_accuracy']:.6f}"
        )

        print(
            f"Mean return per action: "
            f"{baseline['mean_action_return']:.8f}"
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        print(
            "\nCatBoost confusion matrix:"
        )

        print(
            confusion_matrix(
                y_validation,
                predictions,
                labels=[
                    0,
                    1,
                    2,
                ],
            )
        )

        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        print(
            "\nCatBoost classification report:"
        )

        print(
            classification_report(
                y_validation,
                predictions,
                labels=[
                    0,
                    1,
                    2,
                ],
                target_names=[
                    "DOWN",
                    "HOLD",
                    "UP",
                ],
                digits=6,
                zero_division=0,
            )
        )

        # ----------------------------------------------------
        # Save result row
        # ----------------------------------------------------

        results.append(
            {
                "threshold":
                    threshold,
                "threshold_pct":
                    threshold_pct,
                "train_samples":
                    len(X_train),
                "validation_samples":
                    len(X_validation),
                "validation_down_pct":
                    validation_counts.get(
                        0,
                        0,
                    ),
                "validation_hold_pct":
                    validation_counts.get(
                        1,
                        0,
                    ),
                "validation_up_pct":
                    validation_counts.get(
                        2,
                        0,
                    ),
                "accuracy":
                    accuracy,
                "macro_f1":
                    macro_f1,
                "actionable_samples":
                    actionable_count,
                "actionable_coverage":
                    coverage,
                "actionable_directional_accuracy":
                    actionable_accuracy,
                "mean_action_return":
                    return_metrics[
                        "mean_action_return"
                    ],
                "median_action_return":
                    return_metrics[
                        "median_action_return"
                    ],
                "positive_action_rate":
                    return_metrics[
                        "positive_action_rate"
                    ],
                "baseline_accuracy":
                    baseline[
                        "accuracy"
                    ],
                "baseline_macro_f1":
                    baseline[
                        "macro_f1"
                    ],
                "baseline_coverage":
                    baseline[
                        "coverage"
                    ],
                "baseline_actionable_accuracy":
                    baseline[
                        "actionable_accuracy"
                    ],
                "baseline_mean_action_return":
                    baseline[
                        "mean_action_return"
                    ],
            }
        )

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL MAGNITUDE / HOLD COMPARISON"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "threshold_pct",
        "validation_down_pct",
        "validation_hold_pct",
        "validation_up_pct",
        "accuracy",
        "macro_f1",
        "actionable_coverage",
        "actionable_directional_accuracy",
        "mean_action_return",
        "baseline_actionable_accuracy",
        "baseline_mean_action_return",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Candidate selection
    #
    # We don't select purely on accuracy.
    #
    # Primary filter:
    #   >= 10% actionable coverage
    #
    # Then rank by:
    #   mean action return
    #
    # This is still validation-only.
    # --------------------------------------------------------

    viable = results_df[
        results_df[
            "actionable_coverage"
        ]
        >= 0.10
    ].copy()

    if len(viable) > 0:

        best_return_row = viable.loc[
            viable[
                "mean_action_return"
            ].idxmax()
        ]

        best_f1_row = viable.loc[
            viable[
                "macro_f1"
            ].idxmax()
        ]

        print(
            "\nBest viable threshold "
            "by mean action return:"
        )

        print(
            f"{best_return_row['threshold_pct']:.2f}%"
        )

        print(
            f"Mean action return: "
            f"{best_return_row['mean_action_return']:.8f}"
        )

        print(
            f"Coverage: "
            f"{best_return_row['actionable_coverage'] * 100:.2f}%"
        )

        print(
            "\nBest viable threshold "
            "by Macro-F1:"
        )

        print(
            f"{best_f1_row['threshold_pct']:.2f}%"
        )

        print(
            f"Macro-F1: "
            f"{best_f1_row['macro_f1']:.6f}"
        )

        print(
            f"Coverage: "
            f"{best_f1_row['actionable_coverage'] * 100:.2f}%"
        )

    else:

        print(
            "\nNo threshold retained at least "
            "10% actionable coverage."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORTS_DIR
        / "magnitude_hold_target_comparison_v1.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nReport saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main() 