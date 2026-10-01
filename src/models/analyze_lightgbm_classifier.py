"""
Analyze the V3 LightGBM classification model.

This script reproduces the original V3 LightGBM classifier
configuration and analyzes:

1. Validation Macro-F1 and Accuracy
2. Confusion matrix
3. Per-class precision / recall / F1
4. Native LightGBM feature importance by split count
5. Native LightGBM feature importance by gain
6. Permutation importance using Macro-F1
7. Prediction probability distribution
8. Prediction confidence
9. Actual vs predicted class distribution

The original model is retrained exactly according to
src/models/train_boosting_v3.py.

No changes are made to the original training script.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

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

CLASSIFICATION_TARGET = (
    "target_direction_5m"
)

RANDOM_STATE = 42

# Use a deterministic validation subset for
# permutation importance to keep runtime reasonable.
PERMUTATION_SAMPLE_SIZE = 50_000

PERMUTATION_REPEATS = 3


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
            f"Expected 61 V2 feature columns, "
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

    # Match the original V3 training script exactly.
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

            symbol_dummies[
                column
            ] = 0.0

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


def prepare_target_dataset(
    df,
    target_column,
    feature_columns,
):

    X = prepare_features(
        df,
        feature_columns,
    )

    y = (
        df[target_column]
        .reset_index(
            drop=True
        )
    )

    combined = pd.concat(
        [
            X,
            y.rename(
                target_column
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

    X = combined.drop(
        columns=[
            target_column
        ]
    )

    y = combined[
        target_column
    ]

    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD FEATURE NAMES
    # --------------------------------------------------------

    feature_columns = (
        load_feature_columns()
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
    ) = load_data()

    # --------------------------------------------------------
    # PREPARE TRAINING DATA
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "PREPARING TRAINING DATA"
    )
    print(
        "=" * 70
    )

    X_train, y_train = (
        prepare_target_dataset(
            train_df,
            CLASSIFICATION_TARGET,
            feature_columns,
        )
    )

    print(
        f"X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"y_train shape: "
        f"{y_train.shape}"
    )

    # --------------------------------------------------------
    # PREPARE VALIDATION DATA
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "PREPARING VALIDATION DATA"
    )
    print(
        "=" * 70
    )

    X_validation, y_validation = (
        prepare_target_dataset(
            validation_df,
            CLASSIFICATION_TARGET,
            feature_columns,
        )
    )

    print(
        f"X_validation shape: "
        f"{X_validation.shape}"
    )

    print(
        f"y_validation shape: "
        f"{y_validation.shape}"
    )

    # --------------------------------------------------------
    # CLASS ENCODING
    # --------------------------------------------------------

    classes = sorted(
        y_train.unique()
    )

    class_to_id = {
        label: index
        for index, label in enumerate(
            classes
        )
    }

    id_to_class = {
        index: label
        for label, index in (
            class_to_id.items()
        )
    }

    y_train_encoded = (
        y_train.map(
            class_to_id
        )
        .astype(np.int64)
    )

    y_validation_encoded = (
        y_validation.map(
            class_to_id
        )
        .astype(np.int64)
    )

    print()
    print(
        f"Classes: {classes}"
    )

    print(
        f"Class mapping: "
        f"{class_to_id}"
    )

    # --------------------------------------------------------
    # BUILD EXACT V3 LIGHTGBM MODEL
    # --------------------------------------------------------

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multiclass",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        verbosity=-1,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "TRAINING LIGHTGBM"
    )
    print(
        "=" * 70
    )

    print(
        "This reproduces the V3 LightGBM "
        "configuration exactly."
    )

    model.fit(
        X_train,
        y_train_encoded,
    )

    print()
    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "VALIDATION EVALUATION"
    )
    print(
        "=" * 70
    )

    predictions = (
        model.predict(
            X_validation
        )
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )
    )

    macro_f1 = f1_score(
        y_validation_encoded,
        predictions,
        average="macro",
    )

    accuracy = accuracy_score(
        y_validation_encoded,
        predictions,
    )

    print(
        f"Macro-F1: "
        f"{macro_f1:.6f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.6f}"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_validation_encoded,
        predictions,
        labels=list(
            range(
                len(classes)
            )
        ),
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{id_to_class[i]}"
            for i in range(
                len(classes)
            )
        ],
        columns=[
            f"predicted_{id_to_class[i]}"
            for i in range(
                len(classes)
            )
        ],
    )

    confusion_path = (
        REPORTS_DIR
        / "lightgbm_confusion_matrix_v3.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_validation_encoded,
        predictions,
        labels=list(
            range(
                len(classes)
            )
        ),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    report_rows = []

    for label in classes:

        row = report[label]

        report_rows.append(
            {
                "class": label,
                "precision": row[
                    "precision"
                ],
                "recall": row[
                    "recall"
                ],
                "f1": row[
                    "f1-score"
                ],
                "support": row[
                    "support"
                ],
            }
        )

    classification_report_df = (
        pd.DataFrame(
            report_rows
        )
    )

    classification_report_path = (
        REPORTS_DIR
        / "lightgbm_classification_report_v3.csv"
    )

    classification_report_df.to_csv(
        classification_report_path,
        index=False,
    )

    # --------------------------------------------------------
    # ACTUAL/PREDICTED DISTRIBUTION
    # --------------------------------------------------------

    actual_counts = np.bincount(
        y_validation_encoded,
        minlength=len(classes),
    )

    predicted_counts = np.bincount(
        predictions,
        minlength=len(classes),
    )

    distribution_rows = []

    for class_id, class_name in (
        id_to_class.items()
    ):

        distribution_rows.append(
            {
                "class": class_name,
                "actual_count": int(
                    actual_counts[class_id]
                ),
                "predicted_count": int(
                    predicted_counts[class_id]
                ),
                "actual_pct": (
                    actual_counts[class_id]
                    / len(
                        y_validation_encoded
                    )
                    * 100.0
                ),
                "predicted_pct": (
                    predicted_counts[class_id]
                    / len(
                        y_validation_encoded
                    )
                    * 100.0
                ),
            }
        )

    distribution_df = pd.DataFrame(
        distribution_rows
    )

    distribution_path = (
        REPORTS_DIR
        / "lightgbm_prediction_distribution_v3.csv"
    )

    distribution_df.to_csv(
        distribution_path,
        index=False,
    )

    # --------------------------------------------------------
    # PROBABILITY ANALYSIS
    # --------------------------------------------------------

    probability_rows = []

    max_probability = (
        np.max(
            probabilities,
            axis=1,
        )
    )

    predicted_class_probability = (
        probabilities[
            np.arange(
                len(predictions)
            ),
            predictions,
        ]
    )

    for class_id, class_name in (
        id_to_class.items()
    ):

        class_probability = (
            probabilities[
                :,
                class_id,
            ]
        )

        probability_rows.append(
            {
                "class": class_name,
                "mean_probability": float(
                    np.mean(
                        class_probability
                    )
                ),
                "median_probability": float(
                    np.median(
                        class_probability
                    )
                ),
                "std_probability": float(
                    np.std(
                        class_probability
                    )
                ),
            }
        )

    probability_summary_df = (
        pd.DataFrame(
            probability_rows
        )
    )

    probability_summary_path = (
        REPORTS_DIR
        / "lightgbm_probability_summary_v3.csv"
    )

    probability_summary_df.to_csv(
        probability_summary_path,
        index=False,
    )

    confidence_summary = pd.DataFrame(
        [
            {
                "mean_max_probability": float(
                    np.mean(
                        max_probability
                    )
                ),
                "median_max_probability": float(
                    np.median(
                        max_probability
                    )
                ),
                "std_max_probability": float(
                    np.std(
                        max_probability
                    )
                ),
                "mean_predicted_class_probability": float(
                    np.mean(
                        predicted_class_probability
                    )
                ),
            }
        ]
    )

    confidence_path = (
        REPORTS_DIR
        / "lightgbm_confidence_v3.csv"
    )

    confidence_summary.to_csv(
        confidence_path,
        index=False,
    )

    # --------------------------------------------------------
    # NATIVE LIGHTGBM FEATURE IMPORTANCE
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "NATIVE FEATURE IMPORTANCE"
    )
    print(
        "=" * 70
    )

    feature_names = list(
        X_train.columns
    )

    gain_importance = (
        model.booster_.feature_importance(
            importance_type="gain"
        )
    )

    split_importance = (
        model.booster_.feature_importance(
            importance_type="split"
        )
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "gain_importance": (
                gain_importance
            ),
            "split_importance": (
                split_importance
            ),
        }
    )

    importance_df[
        "gain_importance_pct"
    ] = (
        importance_df[
            "gain_importance"
        ]
        / importance_df[
            "gain_importance"
        ].sum()
        * 100.0
    )

    importance_df[
        "split_importance_pct"
    ] = (
        importance_df[
            "split_importance"
        ]
        / importance_df[
            "split_importance"
        ].sum()
        * 100.0
    )

    importance_df = (
        importance_df
        .sort_values(
            "gain_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    native_importance_path = (
        REPORTS_DIR
        / "lightgbm_feature_importance_v3.csv"
    )

    importance_df.to_csv(
        native_importance_path,
        index=False,
    )

    print()
    print(
        "Top 20 features by gain:"
    )

    print(
        importance_df[
            [
                "feature",
                "gain_importance_pct",
                "split_importance_pct",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # PERMUTATION IMPORTANCE
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "PERMUTATION IMPORTANCE"
    )
    print(
        "=" * 70
    )

    print(
        f"Using up to "
        f"{PERMUTATION_SAMPLE_SIZE:,} "
        f"stratified validation rows."
    )

    # Stratified subset so all three classes
    # remain represented proportionally.
    if len(
        X_validation
    ) > PERMUTATION_SAMPLE_SIZE:

        (
            X_permutation,
            _,
            y_permutation,
            _,
        ) = train_test_split(
            X_validation,
            y_validation_encoded,
            train_size=(
                PERMUTATION_SAMPLE_SIZE
            ),
            stratify=(
                y_validation_encoded
            ),
            random_state=(
                RANDOM_STATE
            ),
        )

    else:

        X_permutation = (
            X_validation.copy()
        )

        y_permutation = (
            y_validation_encoded.copy()
        )

    permutation = (
        permutation_importance(
            model,
            X_permutation,
            y_permutation,
            scoring="f1_macro",
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    permutation_df = pd.DataFrame(
        {
            "feature": (
                X_permutation.columns
            ),
            "mean_importance": (
                permutation.importances_mean
            ),
            "std_importance": (
                permutation.importances_std
            ),
        }
    )

    permutation_df = (
        permutation_df
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    permutation_path = (
        REPORTS_DIR
        / "lightgbm_permutation_importance_v3.csv"
    )

    permutation_df.to_csv(
        permutation_path,
        index=False,
    )

    print()
    print(
        "Top 20 features by permutation importance:"
    )

    print(
        permutation_df
        .head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "LIGHTGBM ANALYSIS COMPLETE"
    )
    print(
        "=" * 70
    )

    print()
    print(
        f"Validation Macro-F1: "
        f"{macro_f1:.6f}"
    )

    print(
        f"Validation Accuracy: "
        f"{accuracy:.6f}"
    )

    print()
    print(
        "Reports:"
    )

    print(
        confusion_path
    )

    print(
        classification_report_path
    )

    print(
        distribution_path
    )

    print(
        probability_summary_path
    )

    print(
        confidence_path
    )

    print(
        native_importance_path
    )

    print(
        permutation_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()