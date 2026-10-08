"""
Compare binary direction prediction across multiple forecast horizons.

Horizons:
    5m
    10m
    15m
    30m

Important methodology:
    - Chronological train/validation split is preserved.
    - Test set is NOT used in this screening experiment.
    - Targets are created from future close prices.
    - Exact timestamp spacing is verified.
    - Exact-zero future returns are removed.
    - Same 61 V2 features + 5 symbol one-hot features.
    - Same CatBoost configuration for every horizon.

Purpose:
    Determine whether a longer forecast horizon provides a stronger
    and potentially more economically useful directional signal
    before rebuilding the sequence/ensemble pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
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

HORIZONS = [
    5,
    10,
    15,
    30,
]

RANDOM_STATE = 42


# ============================================================
# FEATURE PREPARATION
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

    X = pd.concat(
        [
            X.reset_index(drop=True),
            symbol_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    return X


# ============================================================
# HORIZON TARGET CREATION
# ============================================================

def add_horizon_target(
    df: pd.DataFrame,
    horizon_minutes: int,
) -> pd.DataFrame:

    """
    Create:

        future_return_H

    only when the actual future timestamp exists exactly H minutes
    later for the same symbol.

    Also creates:

        previous_return_H

    for the inverse-momentum baseline.
    """

    frames = []

    for symbol in SYMBOLS:

        symbol_df = (
            df[
                df["symbol"] == symbol
            ]
            .copy()
            .sort_values(
                "open_time"
            )
            .reset_index(
                drop=True
            )
        )

        symbol_df[
            "open_time"
        ] = pd.to_datetime(
            symbol_df[
                "open_time"
            ],
            utc=True,
        )

        # ----------------------------------------------------
        # Future target
        # ----------------------------------------------------

        future_close = (
            symbol_df[
                "close"
            ]
            .shift(
                -horizon_minutes
            )
        )

        future_time = (
            symbol_df[
                "open_time"
            ]
            .shift(
                -horizon_minutes
            )
        )

        expected_future_time = (
            symbol_df[
                "open_time"
            ]
            +
            pd.Timedelta(
                minutes=horizon_minutes
            )
        )

        exact_future_match = (
            future_time
            == expected_future_time
        )

        symbol_df[
            "future_return_h"
        ] = (
            future_close
            /
            symbol_df[
                "close"
            ]
            - 1.0
        )

        symbol_df.loc[
            ~exact_future_match,
            "future_return_h",
        ] = np.nan

        # ----------------------------------------------------
        # Previous-return baseline
        # ----------------------------------------------------

        previous_close = (
            symbol_df[
                "close"
            ]
            .shift(
                horizon_minutes
            )
        )

        previous_time = (
            symbol_df[
                "open_time"
            ]
            .shift(
                horizon_minutes
            )
        )

        expected_previous_time = (
            symbol_df[
                "open_time"
            ]
            -
            pd.Timedelta(
                minutes=horizon_minutes
            )
        )

        exact_previous_match = (
            previous_time
            == expected_previous_time
        )

        symbol_df[
            "previous_return_h"
        ] = (
            symbol_df[
                "close"
            ]
            /
            previous_close
            - 1.0
        )

        symbol_df.loc[
            ~exact_previous_match,
            "previous_return_h",
        ] = np.nan

        frames.append(
            symbol_df
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_binary_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    horizon_minutes: int,
):

    df = add_horizon_target(
        df,
        horizon_minutes,
    )

    # Only targets known to exist and non-zero are used
    # for classification training/evaluation.
    target_mask = (
        df[
            "future_return_h"
        ].notna()
        &
        (
            df[
                "future_return_h"
            ] != 0
        )
    )

    clean = (
        df.loc[
            target_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    X = prepare_features(
        clean,
        feature_columns,
    )

    y = (
        clean[
            "future_return_h"
        ]
        > 0
    ).astype(
        np.int64
    )

    return (
        X,
        y,
        clean,
    )


# ============================================================
# MODEL
# ============================================================

def build_catboost() -> CatBoostClassifier:

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
# METRICS
# ============================================================

def evaluate_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
):

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.int64
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
    )

    return (
        accuracy,
        macro_f1,
        predictions,
    )


def evaluate_inverse_baseline(
    df: pd.DataFrame,
):

    valid = df[
        "previous_return_h"
    ].notna()

    valid = (
        valid
        &
        df[
            "future_return_h"
        ].notna()
        &
        (
            df[
                "future_return_h"
            ] != 0
        )
    )

    baseline_df = df.loc[
        valid
    ]

    y = (
        baseline_df[
            "future_return_h"
        ] > 0
    ).astype(
        np.int64
    ).to_numpy()

    # Mean-reversion baseline:
    # previous return < 0 -> predict UP
    # previous return > 0 -> predict DOWN
    prediction = (
        baseline_df[
            "previous_return_h"
        ].to_numpy()
        < 0
    ).astype(
        np.int64
    )

    return (
        accuracy_score(
            y,
            prediction,
        ),
        f1_score(
            y,
            prediction,
            average="macro",
        ),
        len(
            baseline_df
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "FORECAST HORIZON COMPARISON"
    )
    print("=" * 70)

    print(
        "\nHorizons:"
    )

    print(
        HORIZONS
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Test set will NOT be used."
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

    train_df[
        "open_time"
    ] = pd.to_datetime(
        train_df[
            "open_time"
        ],
        utc=True,
    )

    validation_df[
        "open_time"
    ] = pd.to_datetime(
        validation_df[
            "open_time"
        ],
        utc=True,
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    feature_columns = (
        load_feature_columns()
    )

    results = []

    # --------------------------------------------------------
    # Horizon loop
    # --------------------------------------------------------

    for horizon in HORIZONS:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"HORIZON: {horizon} MINUTES"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Prepare datasets
        # ----------------------------------------------------

        (
            X_train,
            y_train,
            train_clean,
        ) = prepare_binary_dataset(
            train_df,
            feature_columns,
            horizon,
        )

        (
            X_validation,
            y_validation,
            validation_clean,
        ) = prepare_binary_dataset(
            validation_df,
            feature_columns,
            horizon,
        )

        print(
            f"Train samples: "
            f"{len(X_train):,}"
        )

        print(
            f"Validation samples: "
            f"{len(X_validation):,}"
        )

        train_up_pct = (
            y_train.mean()
            * 100
        )

        validation_up_pct = (
            y_validation.mean()
            * 100
        )

        print(
            f"Train UP: "
            f"{train_up_pct:.2f}%"
        )

        print(
            f"Validation UP: "
            f"{validation_up_pct:.2f}%"
        )

        # ----------------------------------------------------
        # Target magnitude diagnostics
        # ----------------------------------------------------

        validation_returns = (
            validation_clean[
                "future_return_h"
            ].to_numpy()
        )

        median_abs_return = (
            np.median(
                np.abs(
                    validation_returns
                )
            )
        )

        mean_abs_return = (
            np.mean(
                np.abs(
                    validation_returns
                )
            )
        )

        print(
            f"Validation median |return|: "
            f"{median_abs_return:.6f}"
        )

        print(
            f"Validation mean |return|: "
            f"{mean_abs_return:.6f}"
        )

        # ----------------------------------------------------
        # CatBoost
        # ----------------------------------------------------

        model = build_catboost()

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
            )[:, 1]
        )

        (
            accuracy,
            macro_f1,
            predictions,
        ) = evaluate_predictions(
            y_validation,
            probabilities,
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        (
            baseline_accuracy,
            baseline_macro_f1,
            baseline_samples,
        ) = evaluate_inverse_baseline(
            validation_clean
        )

        print(
            "\nCatBoost:"
        )

        print(
            f"Accuracy : "
            f"{accuracy:.6f}"
        )

        print(
            f"Macro-F1 : "
            f"{macro_f1:.6f}"
        )

        print(
            "\nInverse previous-h baseline:"
        )

        print(
            f"Accuracy : "
            f"{baseline_accuracy:.6f}"
        )

        print(
            f"Macro-F1 : "
            f"{baseline_macro_f1:.6f}"
        )

        # ----------------------------------------------------
        # Return-by-prediction diagnostic
        # ----------------------------------------------------

        validation_for_return = (
            validation_clean.copy()
        )

        validation_for_return[
            "prediction"
        ] = predictions

        validation_for_return[
            "directional_gross_return"
        ] = np.where(
            validation_for_return[
                "prediction"
            ] == 1,
            validation_for_return[
                "future_return_h"
            ],
            -
            validation_for_return[
                "future_return_h"
            ],
        )

        mean_directional_return = (
            validation_for_return[
                "directional_gross_return"
            ].mean()
        )

        median_directional_return = (
            validation_for_return[
                "directional_gross_return"
            ].median()
        )

        print(
            "\nDirectional gross-return diagnostic:"
        )

        print(
            f"Mean return per prediction: "
            f"{mean_directional_return:.8f}"
        )

        print(
            f"Median return per prediction: "
            f"{median_directional_return:.8f}"
        )

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        results.append(
            {
                "horizon_minutes":
                    horizon,
                "train_samples":
                    len(X_train),
                "validation_samples":
                    len(X_validation),
                "train_up_pct":
                    train_up_pct,
                "validation_up_pct":
                    validation_up_pct,
                "median_abs_future_return":
                    median_abs_return,
                "mean_abs_future_return":
                    mean_abs_return,
                "catboost_accuracy":
                    accuracy,
                "catboost_macro_f1":
                    macro_f1,
                "inverse_baseline_accuracy":
                    baseline_accuracy,
                "inverse_baseline_macro_f1":
                    baseline_macro_f1,
                "catboost_accuracy_gain":
                    accuracy
                    -
                    baseline_accuracy,
                "catboost_macro_f1_gain":
                    macro_f1
                    -
                    baseline_macro_f1,
                "mean_directional_gross_return":
                    mean_directional_return,
                "median_directional_gross_return":
                    median_directional_return,
            }
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "horizon_minutes"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL VALIDATION HORIZON COMPARISON"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "horizon_minutes",
        "validation_samples",
        "validation_up_pct",
        "median_abs_future_return",
        "catboost_accuracy",
        "catboost_macro_f1",
        "inverse_baseline_accuracy",
        "inverse_baseline_macro_f1",
        "catboost_accuracy_gain",
        "catboost_macro_f1_gain",
        "mean_directional_gross_return",
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
    # Recommended horizon
    #
    # Primary selection:
    #   Macro-F1
    #
    # Secondary:
    #   directional gross return
    # --------------------------------------------------------

    best_horizon_row = (
        results_df.loc[
            results_df[
                "catboost_macro_f1"
            ].idxmax()
        ]
    )

    print(
        "\nBest validation horizon "
        "by CatBoost Macro-F1:"
    )

    print(
        f"{int(best_horizon_row['horizon_minutes'])} minutes"
    )

    print(
        f"Macro-F1: "
        f"{best_horizon_row['catboost_macro_f1']:.6f}"
    )

    print(
        f"Accuracy: "
        f"{best_horizon_row['catboost_accuracy']:.6f}"
    )

    print(
        f"Mean directional return: "
        f"{best_horizon_row['mean_directional_gross_return']:.8f}"
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
        / "forecast_horizon_comparison_v1.csv"
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