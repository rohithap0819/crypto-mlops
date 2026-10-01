"""
Binary 5-minute direction benchmark.

Target:
    future_return_5m > 0 -> UP
    future_return_5m < 0 -> DOWN

Exact-zero future returns are removed.

Models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost

Reference:
    inverse previous 5-minute return baseline

All models use:
    - Same V2 train/validation split
    - Same 61 V2 features
    - Same 5 symbol one-hot features
    - Same target definition
    - Same random state

Reports:
    reports/binary_model_comparison_v1.csv
    reports/binary_model_per_class_v1.csv
    reports/binary_model_confusion_v1.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
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

TARGET_COLUMN = (
    "future_return_5m"
)

RANDOM_STATE = 42


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODELS = {

    "LogisticRegression":
        LogisticRegression(
            C=1.0,
            max_iter=200,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbosity=-1,
        ),

    "CatBoost":
        CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            thread_count=-1,
            verbose=False,
        ),
}


# ============================================================
# LOAD FEATURE COLUMNS
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
# LOAD DATA
# ============================================================

def load_data():

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

    return (
        train_df,
        validation_df,
    )


# ============================================================
# PREPARE FEATURES
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
            f"Missing feature columns:\n"
            f"{missing}"
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
# PREPARE BINARY DATASET
# ============================================================

def prepare_dataset(
    df,
    feature_columns,
):

    X = prepare_features(
        df,
        feature_columns,
    )

    future_return = (
        df[
            TARGET_COLUMN
        ]
        .reset_index(
            drop=True
        )
    )

    combined = pd.concat(
        [
            X,
            future_return.rename(
                TARGET_COLUMN
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

    # Remove exact-zero future returns.
    combined = combined[
        combined[
            TARGET_COLUMN
        ] != 0
    ]

    X = combined[
        X.columns
    ].astype(
        np.float32
    )

    y = (
        combined[
            TARGET_COLUMN
        ]
        .gt(0)
        .astype(
            np.int8
        )
    )

    return X, y


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name,
    model,
    X_validation,
    y_validation,
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

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_validation,
            predictions,
            labels=[
                0,
                1,
            ],
            zero_division=0,
        )
    )

    results = {
        "model": name,
        "macro_f1": float(
            macro_f1
        ),
        "accuracy": float(
            accuracy
        ),
    }

    per_class = []

    class_names = {
        0: "DOWN",
        1: "UP",
    }

    for class_id in [
        0,
        1,
    ]:

        per_class.append(
            {
                "model": name,
                "class": class_names[
                    class_id
                ],
                "precision": float(
                    precision[class_id]
                ),
                "recall": float(
                    recall[class_id]
                ),
                "f1": float(
                    f1[class_id]
                ),
                "support": int(
                    support[class_id]
                ),
            }
        )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    confusion = {
        "model": name,
        "actual_DOWN_pred_DOWN": int(
            matrix[0, 0]
        ),
        "actual_DOWN_pred_UP": int(
            matrix[0, 1]
        ),
        "actual_UP_pred_DOWN": int(
            matrix[1, 0]
        ),
        "actual_UP_pred_UP": int(
            matrix[1, 1]
        ),
    }

    return (
        results,
        per_class,
        confusion,
    )


# ============================================================
# BASELINE
# ============================================================

def evaluate_inverse_momentum(
    df,
):

    clean = df[
        [
            TARGET_COLUMN,
            "return_5m",
        ]
    ].copy()

    clean = clean.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    clean = clean.dropna()

    clean = clean[
        clean[
            TARGET_COLUMN
        ] != 0
    ]

    actual = (
        clean[
            TARGET_COLUMN
        ]
        .gt(0)
        .astype(
            np.int8
        )
        .to_numpy()
    )

    previous = (
        clean[
            "return_5m"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    prediction = (
        previous < 0
    ).astype(
        np.int8
    )

    return {
        "model": (
            "InversePrevious5mBaseline"
        ),
        "macro_f1": float(
            f1_score(
                actual,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                actual,
                prediction,
            )
        ),
    }


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
        "BINARY 5-MINUTE MODEL BENCHMARK"
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
        "Preparing training dataset..."
    )

    (
        X_train,
        y_train,
    ) = prepare_dataset(
        train_df,
        feature_columns,
    )

    print(
        f"X_train: "
        f"{X_train.shape}"
    )

    print(
        f"y_train: "
        f"{y_train.shape}"
    )

    print()
    print(
        "Preparing validation dataset..."
    )

    (
        X_validation,
        y_validation,
    ) = prepare_dataset(
        validation_df,
        feature_columns,
    )

    print(
        f"X_validation: "
        f"{X_validation.shape}"
    )

    print(
        f"y_validation: "
        f"{y_validation.shape}"
    )

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    print()
    print(
        "Training distribution:"
    )

    print(
        y_train.value_counts(
            normalize=False
        )
        .sort_index()
    )

    print()
    print(
        "Validation distribution:"
    )

    print(
        y_validation.value_counts(
            normalize=False
        )
        .sort_index()
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_result = (
        evaluate_inverse_momentum(
            validation_df
        )
    )

    comparison_rows = [
        baseline_result
    ]

    per_class_rows = []

    confusion_rows = []

    # --------------------------------------------------------
    # MODEL BENCHMARK
    # --------------------------------------------------------

    for name, model in (
        MODELS.items()
    ):

        print()
        print("=" * 70)
        print(
            f"MODEL: {name}"
        )
        print("=" * 70)

        print(
            "Training..."
        )

        model.fit(
            X_train,
            y_train,
        )

        print(
            "Training complete."
        )

        (
            result,
            per_class,
            confusion,
        ) = evaluate_model(
            name,
            model,
            X_validation,
            y_validation,
        )

        comparison_rows.append(
            result
        )

        per_class_rows.extend(
            per_class
        )

        confusion_rows.append(
            confusion
        )

        print()
        print(
            f"Macro-F1: "
            f"{result['macro_f1']:.6f}"
        )

        print(
            f"Accuracy: "
            f"{result['accuracy']:.6f}"
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df = (
        comparison_df
        .sort_values(
            "accuracy",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    confusion_df = pd.DataFrame(
        confusion_rows
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    comparison_path = (
        REPORTS_DIR
        / "binary_model_comparison_v1.csv"
    )

    per_class_path = (
        REPORTS_DIR
        / "binary_model_per_class_v1.csv"
    )

    confusion_path = (
        REPORTS_DIR
        / "binary_model_confusion_v1.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    per_class_df.to_csv(
        per_class_path,
        index=False,
    )

    confusion_df.to_csv(
        confusion_path,
        index=False,
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "BINARY MODEL BENCHMARK SUMMARY"
    )
    print("=" * 70)

    print()

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Per-class results:"
    )

    print(
        per_class_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Confusion matrices:"
    )

    print(
        confusion_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "REPORTS SAVED"
    )
    print("=" * 70)

    print(
        comparison_path
    )

    print(
        per_class_path
    )

    print(
        confusion_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()