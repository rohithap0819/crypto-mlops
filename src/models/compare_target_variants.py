"""
Compare target definitions for 5-minute crypto direction prediction.

Variants:

A. Existing 3-class target:
   DOWN / NEUTRAL / UP
   using the existing volatility threshold.

B. Pure 2-class direction:
   future_return_5m < 0 -> DOWN
   future_return_5m > 0 -> UP

The purpose is to determine whether the current
volatility-dependent target definition is limiting
directional prediction performance.

Both variants use:
- Same V2 train/validation splits
- Same 61 V2 features
- Same 5 symbol one-hot features
- Same LightGBM configuration
- Balanced class weighting
- Random state 42
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

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

RANDOM_STATE = 42

FUTURE_RETURN_COLUMN = (
    "future_return_5m"
)

CURRENT_TARGET_COLUMN = (
    "target_direction_5m"
)


# ============================================================
# LIGHTGBM CONFIGURATION
# ============================================================

LIGHTGBM_PARAMS_3CLASS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multiclass",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "verbosity": -1,
}

LIGHTGBM_PARAMS_2CLASS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "verbosity": -1,
}


# ============================================================
# FEATURE LOADING
# ============================================================

def load_feature_columns():

    if not FEATURE_COLUMNS_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n"
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

    if len(columns) != 61:

        raise ValueError(
            f"Expected 61 features, "
            f"found {len(columns)}."
        )

    return columns


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training file not found:\n"
            f"{TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{VALIDATION_FILE}"
        )

    print(
        "Loading training data..."
    )

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        "Loading validation data..."
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    train_df["open_time"] = (
        pd.to_datetime(
            train_df["open_time"]
        )
    )

    validation_df["open_time"] = (
        pd.to_datetime(
            validation_df["open_time"]
        )
    )

    train_df = (
        train_df
        .sort_values(
            [
                "open_time",
                "symbol",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    validation_df = (
        validation_df
        .sort_values(
            [
                "open_time",
                "symbol",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        train_df,
        validation_df,
    )


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    df,
    feature_columns,
):

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing feature columns:\n"
            f"{missing}"
        )

    X = df[
        feature_columns
    ].copy()

    symbol_dummies = pd.get_dummies(
        df["symbol"],
        prefix="symbol",
        dtype=float,
    )

    expected_symbols = [
        f"symbol_{symbol}"
        for symbol in SYMBOLS
    ]

    for column in expected_symbols:

        if column not in symbol_dummies.columns:

            symbol_dummies[column] = 0.0

    symbol_dummies = (
        symbol_dummies[
            expected_symbols
        ]
    )

    X = pd.concat(
        [
            X.reset_index(
                drop=True
            ),
            symbol_dummies.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return X


# ============================================================
# CLEAN DATASET
# ============================================================

def prepare_dataset(
    df,
    feature_columns,
):

    X = prepare_features(
        df,
        feature_columns,
    )

    y_current = (
        df[
            CURRENT_TARGET_COLUMN
        ]
        .reset_index(
            drop=True
        )
    )

    future_return = (
        df[
            FUTURE_RETURN_COLUMN
        ]
        .reset_index(
            drop=True
        )
    )

    combined = pd.concat(
        [
            X,
            y_current.rename(
                CURRENT_TARGET_COLUMN
            ),
            future_return.rename(
                FUTURE_RETURN_COLUMN
            ),
        ],
        axis=1,
    )

    combined = combined.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    combined = combined.dropna()

    X = combined[
        X.columns
    ]

    current_target = combined[
        CURRENT_TARGET_COLUMN
    ]

    future_return = combined[
        FUTURE_RETURN_COLUMN
    ]

    return (
        X,
        current_target,
        future_return,
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_distribution(
    name,
    y,
    class_names,
):

    print()
    print(
        f"{name} class distribution:"
    )

    counts = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    total = len(y)

    for class_id, class_name in (
        class_names.items()
    ):

        count = int(
            counts.get(
                class_id,
                0,
            )
        )

        percentage = (
            count
            / total
            * 100.0
        )

        print(
            f"  {class_name:8s}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
    class_names,
):

    predictions = model.predict(
        X_validation
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    print()
    print(
        f"Macro-F1: "
        f"{macro_f1:.6f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.6f}"
    )

    print()
    print(
        classification_report(
            y_validation,
            predictions,
            labels=list(
                class_names.keys()
            ),
            target_names=list(
                class_names.values()
            ),
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=list(
            class_names.keys()
        ),
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{name}"
            for name in class_names.values()
        ],
        columns=[
            f"predicted_{name}"
            for name in class_names.values()
        ],
    )

    return (
        predictions,
        macro_f1,
        accuracy,
        confusion_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(
        "5-MINUTE TARGET VARIANT ANALYSIS"
    )
    print("=" * 70)

    feature_columns = (
        load_feature_columns()
    )

    (
        train_df,
        validation_df,
    ) = load_data()

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    print()
    print(
        "Preparing training data..."
    )

    (
        X_train,
        y_train_current,
        return_train,
    ) = prepare_dataset(
        train_df,
        feature_columns,
    )

    print(
        "Preparing validation data..."
    )

    (
        X_validation,
        y_validation_current,
        return_validation,
    ) = prepare_dataset(
        validation_df,
        feature_columns,
    )

    print()
    print(
        f"Training feature shape: "
        f"{X_train.shape}"
    )

    print(
        f"Validation feature shape: "
        f"{X_validation.shape}"
    )

    # ========================================================
    # TARGET A
    # EXISTING 3-CLASS TARGET
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TARGET A: EXISTING 3-CLASS"
    )
    print("=" * 70)

    # Encode the existing labels exactly
    # using deterministic sorted ordering.
    current_classes = sorted(
        y_train_current.unique()
    )

    current_mapping = {
        label: index
        for index, label in enumerate(
            current_classes
        )
    }

    y_train_3 = (
        y_train_current
        .map(
            current_mapping
        )
        .astype(np.int64)
    )

    y_validation_3 = (
        y_validation_current
        .map(
            current_mapping
        )
        .astype(np.int64)
    )

    current_class_names = {
        index: str(label)
        for label, index
        in current_mapping.items()
    }

    print(
        f"Class mapping: "
        f"{current_mapping}"
    )

    print_distribution(
        "Training",
        y_train_3,
        current_class_names,
    )

    print_distribution(
        "Validation",
        y_validation_3,
        current_class_names,
    )

    model_3class = (
        LGBMClassifier(
            **LIGHTGBM_PARAMS_3CLASS
        )
    )

    print()
    print(
        "Training 3-class LightGBM..."
    )

    model_3class.fit(
        X_train,
        y_train_3,
    )

    (
        predictions_3,
        macro_f1_3,
        accuracy_3,
        confusion_3,
    ) = evaluate_model(
        model_3class,
        X_validation,
        y_validation_3,
        current_class_names,
    )

    confusion_3_path = (
        REPORTS_DIR
        / "target_variant_3class_confusion_v1.csv"
    )

    confusion_3.to_csv(
        confusion_3_path
    )

    # ========================================================
    # TARGET B
    # PURE 2-CLASS DIRECTION
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TARGET B: PURE 2-CLASS DIRECTION"
    )
    print("=" * 70)

    # Drop exact zero returns.
    #
    # The probability of exactly zero is expected to be
    # extremely small, but we explicitly handle it rather
    # than arbitrarily assigning it to UP or DOWN.
    train_direction_mask = (
        return_train != 0
    )

    validation_direction_mask = (
        return_validation != 0
    )

    X_train_binary = X_train.loc[
        train_direction_mask
    ].copy()

    X_validation_binary = (
        X_validation.loc[
            validation_direction_mask
        ].copy()
    )

    y_train_binary = (
        return_train.loc[
            train_direction_mask
        ]
        .gt(0)
        .astype(np.int64)
    )

    y_validation_binary = (
        return_validation.loc[
            validation_direction_mask
        ]
        .gt(0)
        .astype(np.int64)
    )

    binary_class_names = {
        0: "DOWN",
        1: "UP",
    }

    print(
        f"Training rows after zero-return removal: "
        f"{len(y_train_binary):,}"
    )

    print(
        f"Validation rows after zero-return removal: "
        f"{len(y_validation_binary):,}"
    )

    print_distribution(
        "Training",
        y_train_binary,
        binary_class_names,
    )

    print_distribution(
        "Validation",
        y_validation_binary,
        binary_class_names,
    )

    model_binary = (
        LGBMClassifier(
            **LIGHTGBM_PARAMS_2CLASS
        )
    )

    print()
    print(
        "Training binary LightGBM..."
    )

    model_binary.fit(
        X_train_binary,
        y_train_binary,
    )

    (
        predictions_binary,
        macro_f1_binary,
        accuracy_binary,
        confusion_binary,
    ) = evaluate_model(
        model_binary,
        X_validation_binary,
        y_validation_binary,
        binary_class_names,
    )

    confusion_binary_path = (
        REPORTS_DIR
        / "target_variant_2class_confusion_v1.csv"
    )

    confusion_binary.to_csv(
        confusion_binary_path
    )

    # ========================================================
    # BASELINE FOR BINARY TARGET
    # ========================================================

    majority_binary = int(
        y_train_binary
        .value_counts()
        .idxmax()
    )

    baseline_prediction = np.full(
        len(y_validation_binary),
        majority_binary,
        dtype=np.int64,
    )

    baseline_macro_f1 = (
        f1_score(
            y_validation_binary,
            baseline_prediction,
            average="macro",
            zero_division=0,
        )
    )

    baseline_accuracy = (
        accuracy_score(
            y_validation_binary,
            baseline_prediction,
        )
    )

    print()
    print(
        "Binary majority baseline:"
    )

    print(
        f"  Macro-F1: "
        f"{baseline_macro_f1:.6f}"
    )

    print(
        f"  Accuracy: "
        f"{baseline_accuracy:.6f}"
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    comparison = pd.DataFrame(
        [
            {
                "target_variant": (
                    "3_class_volatility_threshold"
                ),
                "classes": 3,
                "training_rows": len(
                    y_train_3
                ),
                "validation_rows": len(
                    y_validation_3
                ),
                "macro_f1": (
                    macro_f1_3
                ),
                "accuracy": (
                    accuracy_3
                ),
                "baseline_macro_f1": np.nan,
                "baseline_accuracy": np.nan,
            },
            {
                "target_variant": (
                    "2_class_pure_direction"
                ),
                "classes": 2,
                "training_rows": len(
                    y_train_binary
                ),
                "validation_rows": len(
                    y_validation_binary
                ),
                "macro_f1": (
                    macro_f1_binary
                ),
                "accuracy": (
                    accuracy_binary
                ),
                "baseline_macro_f1": (
                    baseline_macro_f1
                ),
                "baseline_accuracy": (
                    baseline_accuracy
                ),
            },
        ]
    )

    comparison_path = (
        REPORTS_DIR
        / "target_variant_comparison_v1.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TARGET VARIANT SUMMARY"
    )
    print("=" * 70)

    print()

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        "Reports saved:"
    )

    print(
        comparison_path
    )

    print(
        confusion_3_path
    )

    print(
        confusion_binary_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()