"""
Final untouched-test evaluation for binary 5-minute direction.

Target:
    future_return_5m > 0 -> UP
    future_return_5m < 0 -> DOWN

Training:
    train.parquet + validation.parquet

Final evaluation:
    test.parquet ONLY

Final candidate:
    CatBoost

Reference:
    Inverse previous-5-minute return baseline

IMPORTANT:
    The test set is not used for model selection or tuning.
"""

from pathlib import Path
import json

import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
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

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits_v2"
    / "test.parquet"
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

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "final"
)

MLFLOW_DB = (
    PROJECT_ROOT
    / "mlflow.db"
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
# FINAL CATBOOST CONFIGURATION
# ============================================================

CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 8,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_STATE,
    "thread_count": -1,
    "verbose": False,
}


# ============================================================
# FEATURE COLUMNS
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
            f"Expected 61 V2 features, "
            f"found {len(columns)}."
        )

    return columns


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    for path in [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
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

    print(
        "Loading TEST data..."
    )

    test_df = pd.read_parquet(
        TEST_FILE
    )

    print(
        f"TEST rows: "
        f"{len(test_df):,}"
    )

    return (
        train_df,
        validation_df,
        test_df,
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
            X.reset_index(drop=True),
            symbol_dummies.reset_index(drop=True),
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
        .reset_index(drop=True)
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

    # Pure direction is undefined for exact zero returns.
    combined = combined[
        combined[
            TARGET_COLUMN
        ] != 0
    ].copy()

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

    return (
        X,
        y,
    )


# ============================================================
# INVERSE MOMENTUM BASELINE
# ============================================================

def evaluate_inverse_momentum(
    test_df,
):

    baseline_df = test_df[
        [
            TARGET_COLUMN,
            "return_5m",
        ]
    ].copy()

    baseline_df = baseline_df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    baseline_df = baseline_df.dropna()

    baseline_df = baseline_df[
        baseline_df[
            TARGET_COLUMN
        ] != 0
    ]

    actual = (
        baseline_df[
            TARGET_COLUMN
        ]
        .gt(0)
        .astype(
            np.int8
        )
        .to_numpy()
    )

    previous_return = (
        baseline_df[
            "return_5m"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # Inverse momentum:
    # previous positive return -> predict DOWN
    # previous negative return -> predict UP
    prediction = (
        previous_return < 0
    ).astype(
        np.int8
    )

    return {
        "accuracy": float(
            accuracy_score(
                actual,
                prediction,
            )
        ),
        "macro_f1": float(
            f1_score(
                actual,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "actual": actual,
        "prediction": prediction,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL BINARY TEST EVALUATION"
    )
    print("=" * 70)

    feature_columns = (
        load_feature_columns()
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = load_data()

    # --------------------------------------------------------
    # COMBINE TRAIN + VALIDATION
    # --------------------------------------------------------

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    print()
    print(
        f"Combined train+validation rows: "
        f"{len(development_df):,}"
    )

    # --------------------------------------------------------
    # PREPARE DEVELOPMENT DATA
    # --------------------------------------------------------

    print()
    print(
        "Preparing train+validation..."
    )

    (
        X_development,
        y_development,
    ) = prepare_dataset(
        development_df,
        feature_columns,
    )

    print(
        f"X_development: "
        f"{X_development.shape}"
    )

    print(
        f"y_development: "
        f"{y_development.shape}"
    )

    # --------------------------------------------------------
    # PREPARE TEST DATA
    # --------------------------------------------------------

    print()
    print(
        "Preparing TEST data..."
    )

    (
        X_test,
        y_test,
    ) = prepare_dataset(
        test_df,
        feature_columns,
    )

    print(
        f"X_test: "
        f"{X_test.shape}"
    )

    print(
        f"y_test: "
        f"{y_test.shape}"
    )

    # --------------------------------------------------------
    # DISTRIBUTIONS
    # --------------------------------------------------------

    print()
    print(
        "Development target distribution:"
    )

    print(
        y_development
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "TEST target distribution:"
    )

    print(
        y_test
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # TRAIN FINAL CATBOOST
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TRAINING FINAL CATBOOST"
    )
    print("=" * 70)

    model = CatBoostClassifier(
        **CATBOOST_PARAMS
    )

    model.fit(
        X_development,
        y_development,
    )

    print(
        "Final CatBoost training complete."
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        MODELS_DIR
        / "binary_direction_catboost_v1.cbm"
    )

    model.save_model(
        str(
            model_path
        )
    )

    print()
    print(
        f"Model saved to:\n"
        f"{model_path}"
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "UNTOUCHED TEST EVALUATION"
    )
    print("=" * 70)

    predictions = model.predict(
        X_test
    )

    predictions = (
        np.asarray(
            predictions
        )
        .reshape(-1)
        .astype(np.int8)
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print(
        f"CatBoost TEST Macro-F1: "
        f"{macro_f1:.6f}"
    )

    print(
        f"CatBoost TEST Accuracy: "
        f"{accuracy:.6f}"
    )

    # --------------------------------------------------------
    # TEST CLASSIFICATION REPORT
    # --------------------------------------------------------

    print()
    print(
        classification_report(
            y_test,
            predictions,
            labels=[
                0,
                1,
            ],
            target_names=[
                "DOWN",
                "UP",
            ],
            zero_division=0,
        )
    )

    report_dict = classification_report(
        y_test,
        predictions,
        labels=[
            0,
            1,
        ],
        target_names=[
            "DOWN",
            "UP",
        ],
        output_dict=True,
        zero_division=0,
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            "actual_DOWN",
            "actual_UP",
        ],
        columns=[
            "predicted_DOWN",
            "predicted_UP",
        ],
    )

    confusion_path = (
        REPORTS_DIR
        / "binary_catboost_test_confusion_v1.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # --------------------------------------------------------
    # PER CLASS REPORT
    # --------------------------------------------------------

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            labels=[
                0,
                1,
            ],
            zero_division=0,
        )
    )

    per_class_df = pd.DataFrame(
        [
            {
                "class": "DOWN",
                "precision": float(
                    precision[0]
                ),
                "recall": float(
                    recall[0]
                ),
                "f1": float(
                    f1[0]
                ),
                "support": int(
                    support[0]
                ),
            },
            {
                "class": "UP",
                "precision": float(
                    precision[1]
                ),
                "recall": float(
                    recall[1]
                ),
                "f1": float(
                    f1[1]
                ),
                "support": int(
                    support[1]
                ),
            },
        ]
    )

    per_class_path = (
        REPORTS_DIR
        / "binary_catboost_test_per_class_v1.csv"
    )

    per_class_df.to_csv(
        per_class_path,
        index=False,
    )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED DISTRIBUTION
    # --------------------------------------------------------

    actual_counts = np.bincount(
        y_test,
        minlength=2,
    )

    predicted_counts = np.bincount(
        predictions,
        minlength=2,
    )

    distribution_df = pd.DataFrame(
        [
            {
                "class": "DOWN",
                "actual_count": int(
                    actual_counts[0]
                ),
                "predicted_count": int(
                    predicted_counts[0]
                ),
            },
            {
                "class": "UP",
                "actual_count": int(
                    actual_counts[1]
                ),
                "predicted_count": int(
                    predicted_counts[1]
                ),
            },
        ]
    )

    distribution_path = (
        REPORTS_DIR
        / "binary_catboost_test_distribution_v1.csv"
    )

    distribution_df.to_csv(
        distribution_path,
        index=False,
    )

    # --------------------------------------------------------
    # PREDICTION CONFIDENCE
    # --------------------------------------------------------

    max_probability = np.max(
        probabilities,
        axis=1,
    )

    confidence_df = pd.DataFrame(
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
                "min_max_probability": float(
                    np.min(
                        max_probability
                    )
                ),
                "max_max_probability": float(
                    np.max(
                        max_probability
                    )
                ),
            }
        ]
    )

    confidence_path = (
        REPORTS_DIR
        / "binary_catboost_test_confidence_v1.csv"
    )

    confidence_df.to_csv(
        confidence_path,
        index=False,
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline = (
        evaluate_inverse_momentum(
            test_df
        )
    )

    baseline_accuracy = (
        baseline["accuracy"]
    )

    baseline_macro_f1 = (
        baseline["macro_f1"]
    )

    print()
    print(
        "=" * 70
    )
    print(
        "TEST BASELINE"
    )
    print(
        "=" * 70
    )

    print(
        f"Inverse previous-5m "
        f"Accuracy: "
        f"{baseline_accuracy:.6f}"
    )

    print(
        f"Inverse previous-5m "
        f"Macro-F1: "
        f"{baseline_macro_f1:.6f}"
    )

    # --------------------------------------------------------
    # MODEL VS BASELINE
    # --------------------------------------------------------

    improvement_accuracy = (
        accuracy
        - baseline_accuracy
    )

    improvement_macro_f1 = (
        macro_f1
        - baseline_macro_f1
    )

    comparison_df = pd.DataFrame(
        [
            {
                "model": "Final CatBoost",
                "accuracy": float(
                    accuracy
                ),
                "macro_f1": float(
                    macro_f1
                ),
            },
            {
                "model": (
                    "InversePrevious5mBaseline"
                ),
                "accuracy": float(
                    baseline_accuracy
                ),
                "macro_f1": float(
                    baseline_macro_f1
                ),
            },
        ]
    )

    comparison_path = (
        REPORTS_DIR
        / "binary_catboost_test_comparison_v1.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {
        "task": (
            "binary_5m_direction"
        ),
        "target": TARGET_COLUMN,
        "target_definition": {
            "UP": "future_return_5m > 0",
            "DOWN": "future_return_5m < 0",
            "zero_returns_removed": True,
        },
        "train_rows_used": int(
            len(y_development)
        ),
        "test_rows_used": int(
            len(y_test)
        ),
        "feature_count": int(
            X_development.shape[1]
        ),
        "v2_feature_count": 61,
        "symbol_feature_count": 5,
        "model": "CatBoostClassifier",
        "model_parameters": CATBOOST_PARAMS,
        "test_accuracy": float(
            accuracy
        ),
        "test_macro_f1": float(
            macro_f1
        ),
        "baseline_accuracy": float(
            baseline_accuracy
        ),
        "baseline_macro_f1": float(
            baseline_macro_f1
        ),
        "accuracy_improvement": float(
            improvement_accuracy
        ),
        "macro_f1_improvement": float(
            improvement_macro_f1
        ),
        "model_path": str(
            model_path.relative_to(
                PROJECT_ROOT
            )
        ),
    }

    metadata_path = (
        REPORTS_DIR
        / "binary_catboost_test_metadata_v1.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # MLFLOW
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB}"
    )

    mlflow.set_experiment(
        "CryptoML_Final_Binary"
    )

    with mlflow.start_run(
        run_name=(
            "Final_CatBoost_Binary_"
            "5m_Test"
        )
    ) as run:

        mlflow.log_param(
            "feature_version",
            "v2",
        )

        mlflow.log_param(
            "task",
            "binary_5m_direction",
        )

        mlflow.log_param(
            "train_rows",
            len(y_development),
        )

        mlflow.log_param(
            "test_rows",
            len(y_test),
        )

        mlflow.log_param(
            "feature_count",
            X_development.shape[1],
        )

        for key, value in (
            CATBOOST_PARAMS.items()
        ):

            mlflow.log_param(
                key,
                value,
            )

        mlflow.log_metric(
            "test_accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "test_macro_f1",
            macro_f1,
        )

        mlflow.log_metric(
            "baseline_accuracy",
            baseline_accuracy,
        )

        mlflow.log_metric(
            "baseline_macro_f1",
            baseline_macro_f1,
        )

        mlflow.log_metric(
            "accuracy_improvement",
            improvement_accuracy,
        )

        mlflow.log_metric(
            "macro_f1_improvement",
            improvement_macro_f1,
        )

        mlflow.catboost.log_model(
            cb_model=model,
            name="model",
        )

        mlflow.log_artifact(
            str(
                metadata_path
            )
        )

        print()
        print(
            f"MLflow run ID: "
            f"{run.info.run_id}"
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL TEST SUMMARY"
    )
    print("=" * 70)

    print()

    print(
        f"CatBoost Accuracy: "
        f"{accuracy:.6f}"
    )

    print(
        f"CatBoost Macro-F1: "
        f"{macro_f1:.6f}"
    )

    print()

    print(
        f"Baseline Accuracy: "
        f"{baseline_accuracy:.6f}"
    )

    print(
        f"Baseline Macro-F1: "
        f"{baseline_macro_f1:.6f}"
    )

    print()

    print(
        f"Accuracy improvement: "
        f"{improvement_accuracy:.6f}"
    )

    print(
        f"Macro-F1 improvement: "
        f"{improvement_macro_f1:.6f}"
    )

    print()
    print(
        "Reports saved:"
    )

    print(
        confusion_path
    )

    print(
        per_class_path
    )

    print(
        distribution_path
    )

    print(
        confidence_path
    )

    print(
        comparison_path
    )

    print(
        metadata_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()