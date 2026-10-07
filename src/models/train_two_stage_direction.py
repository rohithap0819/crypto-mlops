"""
Two-stage 5-minute crypto direction model.

Stage A:
    Detect whether the next 5-minute move is meaningful.

    abs(future_return_5m) > 0.15% -> EVENT
    otherwise                       -> NO_EVENT

Stage B:
    Given that a meaningful move is happening,
    predict its direction.

    future_return_5m > 0 -> UP
    future_return_5m < 0 -> DOWN

Validation only.
The test set is NOT used.

Purpose:
    Separate event detection from direction prediction instead of
    forcing one 3-class model to learn DOWN / HOLD / UP together.
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
    precision_score,
    recall_score,
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

# Candidate meaningful-move threshold:
# 0.15% = 0.0015
EVENT_THRESHOLD = 0.0015

# Probability threshold used to decide whether Stage A
# considers the next move tradeable.
EVENT_PROBABILITY_THRESHOLD = 0.50

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
# DATASET PREPARATION
# ============================================================

def prepare_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
):

    clean = df[
        df[TARGET_COLUMN].notna()
    ].copy()

    clean = clean.reset_index(
        drop=True
    )

    future_returns = clean[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Stage A target
    #
    # 1 = meaningful move
    # 0 = no meaningful move
    # --------------------------------------------------------

    event_target = (
        np.abs(
            future_returns
        )
        > EVENT_THRESHOLD
    ).astype(
        np.int64
    )

    # --------------------------------------------------------
    # Stage B target
    #
    # Only meaningful-move rows are used for this model.
    #
    # 0 = DOWN
    # 1 = UP
    # --------------------------------------------------------

    event_mask = (
        event_target == 1
    )

    direction_target = (
        future_returns[event_mask] > 0
    ).astype(
        np.int64
    )

    X_all = prepare_features(
        clean,
        feature_columns,
    )

    X_event = X_all.loc[
        event_mask
    ].reset_index(
        drop=True
    )

    return (
        X_all,
        event_target,
        X_event,
        direction_target,
        clean,
    )


# ============================================================
# MODELS
# ============================================================

def build_event_model():

    return CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


def build_direction_model():

    return CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


# ============================================================
# METRIC HELPERS
# ============================================================

def print_event_metrics(
    y_true,
    prediction,
):

    print(
        "\nStage A — Event Detection"
    )

    print(
        f"Accuracy : "
        f"{accuracy_score(y_true, prediction):.6f}"
    )

    print(
        f"Macro-F1 : "
        f"{f1_score(y_true, prediction, average='macro'):.6f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_true, prediction, zero_division=0):.6f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_true, prediction, zero_division=0):.6f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        confusion_matrix(
            y_true,
            prediction,
            labels=[
                0,
                1,
            ],
        )
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_true,
            prediction,
            labels=[
                0,
                1,
            ],
            target_names=[
                "NO_EVENT",
                "EVENT",
            ],
            digits=6,
            zero_division=0,
        )
    )


def calculate_action_metrics(
    y_true_direction,
    predicted_direction,
    future_returns,
):

    actionable_count = len(
        predicted_direction
    )

    if actionable_count == 0:

        return {
            "actionable_samples":
                0,
            "actionable_directional_accuracy":
                np.nan,
            "mean_action_return":
                np.nan,
            "median_action_return":
                np.nan,
            "positive_action_rate":
                np.nan,
        }

    directional_accuracy = accuracy_score(
        y_true_direction,
        predicted_direction,
    )

    # Convert predicted direction into a trading return.
    #
    # 1 = UP / LONG
    # 0 = DOWN / SHORT
    action_returns = np.where(
        predicted_direction == 1,
        future_returns,
        -future_returns,
    )

    return {
        "actionable_samples":
            actionable_count,
        "actionable_directional_accuracy":
            float(
                directional_accuracy
            ),
        "mean_action_return":
            float(
                action_returns.mean()
            ),
        "median_action_return":
            float(
                np.median(
                    action_returns
                )
            ),
        "positive_action_rate":
            float(
                np.mean(
                    action_returns > 0
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TWO-STAGE EVENT + DIRECTION MODEL"
    )
    print("=" * 70)

    print(
        f"Event threshold: "
        f"{EVENT_THRESHOLD * 100:.2f}%"
    )

    print(
        f"Stage A probability threshold: "
        f"{EVENT_PROBABILITY_THRESHOLD:.2f}"
    )

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

    # --------------------------------------------------------
    # Prepare Stage A + Stage B datasets
    # --------------------------------------------------------

    (
        X_train_all,
        y_train_event,
        X_train_event,
        y_train_direction,
        train_clean,
    ) = prepare_dataset(
        train_df,
        feature_columns,
    )

    (
        X_validation_all,
        y_validation_event,
        X_validation_event,
        y_validation_direction,
        validation_clean,
    ) = prepare_dataset(
        validation_df,
        feature_columns,
    )

    # --------------------------------------------------------
    # Target distributions
    # --------------------------------------------------------

    train_event_rate = (
        y_train_event.mean()
    )

    validation_event_rate = (
        y_validation_event.mean()
    )

    print(
        "\nStage A target distribution:"
    )

    print(
        f"Train NO_EVENT: "
        f"{(1 - train_event_rate) * 100:.2f}%"
    )

    print(
        f"Train EVENT: "
        f"{train_event_rate * 100:.2f}%"
    )

    print(
        f"Validation NO_EVENT: "
        f"{(1 - validation_event_rate) * 100:.2f}%"
    )

    print(
        f"Validation EVENT: "
        f"{validation_event_rate * 100:.2f}%"
    )

    print(
        "\nStage B event samples:"
    )

    print(
        f"Train: "
        f"{len(X_train_event):,}"
    )

    print(
        f"Validation true events: "
        f"{len(X_validation_event):,}"
    )

    train_up_rate = (
        y_train_direction.mean()
    )

    validation_up_rate = (
        y_validation_direction.mean()
    )

    print(
        f"Stage B train UP: "
        f"{train_up_rate * 100:.2f}%"
    )

    print(
        f"Stage B validation UP: "
        f"{validation_up_rate * 100:.2f}%"
    )

    # ========================================================
    # STAGE A
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STAGE A — MEANINGFUL MOVE DETECTION"
    )

    print(
        "=" * 70
    )

    event_model = build_event_model()

    event_model.fit(
        X_train_all,
        y_train_event,
    )

    event_probability = (
        event_model.predict_proba(
            X_validation_all
        )[:, 1]
    )

    event_prediction = (
        event_probability
        >= EVENT_PROBABILITY_THRESHOLD
    ).astype(
        np.int64
    )

    print_event_metrics(
        y_validation_event,
        event_prediction,
    )

    # ========================================================
    # STAGE B
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STAGE B — DIRECTION GIVEN EVENT"
    )

    print(
        "=" * 70
    )

    direction_model = (
        build_direction_model()
    )

    direction_model.fit(
        X_train_event,
        y_train_direction,
    )

    # Evaluate Stage B on TRUE validation events.
    direction_probability = (
        direction_model.predict_proba(
            X_validation_event
        )[:, 1]
    )

    direction_prediction = (
        direction_probability
        >= 0.50
    ).astype(
        np.int64
    )

    print(
        f"\nDirection accuracy: "
        f"{accuracy_score(y_validation_direction, direction_prediction):.6f}"
    )

    print(
        f"Direction Macro-F1: "
        f"{f1_score(y_validation_direction, direction_prediction, average='macro'):.6f}"
    )

    print(
        "\nDirection classification report:"
    )

    print(
        classification_report(
            y_validation_direction,
            direction_prediction,
            labels=[
                0,
                1,
            ],
            target_names=[
                "DOWN",
                "UP",
            ],
            digits=6,
            zero_division=0,
        )
    )

    print(
        "\nDirection confusion matrix:"
    )

    print(
        confusion_matrix(
            y_validation_direction,
            direction_prediction,
            labels=[
                0,
                1,
            ],
        )
    )

    # ========================================================
    # END-TO-END GATED PREDICTION
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "END-TO-END TWO-STAGE SIGNAL"
    )

    print(
        "=" * 70
    )

    # Stage A decides which validation rows are actionable.
    actionable_mask = (
        event_prediction == 1
    )

    actionable_indices = np.where(
        actionable_mask
    )[0]

    print(
        f"\nPredicted actionable rows: "
        f"{len(actionable_indices):,}"
    )

    print(
        f"Action coverage: "
        f"{len(actionable_indices) / len(validation_clean) * 100:.2f}%"
    )

    if len(actionable_indices) == 0:

        raise ValueError(
            "Stage A produced zero actionable predictions."
        )

    # Get Stage B features only for Stage A predicted events.
    X_predicted_events = (
        X_validation_all.iloc[
            actionable_indices
        ]
        .reset_index(
            drop=True
        )
    )

    stage_b_probability = (
        direction_model.predict_proba(
            X_predicted_events
        )[:, 1]
    )

    stage_b_prediction = (
        stage_b_probability
        >= 0.50
    ).astype(
        np.int64
    )

    # True future returns corresponding to Stage A predictions.
    actionable_returns = (
        validation_clean[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )[
            actionable_indices
        ]
    )

    # True direction for these rows.
    actionable_true_direction = (
        actionable_returns > 0
    ).astype(
        np.int64
    )

    action_metrics = (
        calculate_action_metrics(
            actionable_true_direction,
            stage_b_prediction,
            actionable_returns,
        )
    )

    print(
        "\nEnd-to-end actionable metrics:"
    )

    print(
        f"Actionable samples: "
        f"{action_metrics['actionable_samples']:,}"
    )

    print(
        f"Actionable directional accuracy: "
        f"{action_metrics['actionable_directional_accuracy']:.6f}"
    )

    print(
        f"Mean gross return/action: "
        f"{action_metrics['mean_action_return']:.8f}"
    )

    print(
        f"Median gross return/action: "
        f"{action_metrics['median_action_return']:.8f}"
    )

    print(
        f"Positive action rate: "
        f"{action_metrics['positive_action_rate']:.6f}"
    )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    # Current pure binary target baseline:
    # inverse previous 5m return.
    #
    # We only evaluate it on the same predicted-action rows
    # so the economic comparison is meaningful.

    previous_return = (
        validation_clean[
            "return_5m"
        ]
        .to_numpy(
            dtype=float
        )[
            actionable_indices
        ]
    )

    baseline_prediction = (
        previous_return < 0
    ).astype(
        np.int64
    )

    baseline_metrics = (
        calculate_action_metrics(
            actionable_true_direction,
            baseline_prediction,
            actionable_returns,
        )
    )

    print(
        "\nInverse-momentum baseline on same action set:"
    )

    print(
        f"Directional accuracy: "
        f"{baseline_metrics['actionable_directional_accuracy']:.6f}"
    )

    print(
        f"Mean gross return/action: "
        f"{baseline_metrics['mean_action_return']:.8f}"
    )

    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    summary = {
        "event_threshold":
            EVENT_THRESHOLD,
        "event_probability_threshold":
            EVENT_PROBABILITY_THRESHOLD,
        "stage_a_validation_accuracy":
            float(
                accuracy_score(
                    y_validation_event,
                    event_prediction,
                )
            ),
        "stage_a_validation_macro_f1":
            float(
                f1_score(
                    y_validation_event,
                    event_prediction,
                    average="macro",
                )
            ),
        "stage_a_validation_precision":
            float(
                precision_score(
                    y_validation_event,
                    event_prediction,
                    zero_division=0,
                )
            ),
        "stage_a_validation_recall":
            float(
                recall_score(
                    y_validation_event,
                    event_prediction,
                    zero_division=0,
                )
            ),
        "stage_b_true_event_accuracy":
            float(
                accuracy_score(
                    y_validation_direction,
                    direction_prediction,
                )
            ),
        "stage_b_true_event_macro_f1":
            float(
                f1_score(
                    y_validation_direction,
                    direction_prediction,
                    average="macro",
                )
            ),
        "end_to_end_action_coverage":
            float(
                len(actionable_indices)
                /
                len(validation_clean)
            ),
        "end_to_end_directional_accuracy":
            action_metrics[
                "actionable_directional_accuracy"
            ],
        "end_to_end_mean_action_return":
            action_metrics[
                "mean_action_return"
            ],
        "end_to_end_median_action_return":
            action_metrics[
                "median_action_return"
            ],
        "end_to_end_positive_action_rate":
            action_metrics[
                "positive_action_rate"
            ],
        "baseline_directional_accuracy_same_actions":
            baseline_metrics[
                "actionable_directional_accuracy"
            ],
        "baseline_mean_action_return_same_actions":
            baseline_metrics[
                "mean_action_return"
            ],
    }

    # ========================================================
    # SAVE
    # ========================================================

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORTS_DIR
        / "two_stage_direction_validation_v1.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        import json

        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Stage A Macro-F1: "
        f"{summary['stage_a_validation_macro_f1']:.6f}"
    )

    print(
        f"Stage B Macro-F1: "
        f"{summary['stage_b_true_event_macro_f1']:.6f}"
    )

    print(
        f"Action coverage: "
        f"{summary['end_to_end_action_coverage'] * 100:.2f}%"
    )

    print(
        f"Action directional accuracy: "
        f"{summary['end_to_end_directional_accuracy']:.6f}"
    )

    print(
        f"Mean gross return/action: "
        f"{summary['end_to_end_mean_action_return']:.8f}"
    )

    print(
        f"Baseline directional accuracy: "
        f"{summary['baseline_directional_accuracy_same_actions']:.6f}"
    )

    print(
        f"Baseline mean return/action: "
        f"{summary['baseline_mean_action_return_same_actions']:.8f}"
    )

    print(
        f"\nReport saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()