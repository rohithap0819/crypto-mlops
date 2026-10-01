from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    "data/processed/splits_v2/train.parquet"
)

VALIDATION_FILE = Path(
    "data/processed/splits_v2/validation.parquet"
)

FEATURE_COLUMNS_FILE = Path(
    "data/processed/feature_columns_v2.txt"
)

REPORTS_DIR = Path("reports")

RESULTS_FILE = (
    REPORTS_DIR / "tuned_candidate_results.csv"
)

MLFLOW_DB = Path("mlflow.db")


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

REGRESSION_1M_TARGET = (
    "next_return_1m"
)

REGRESSION_5M_TARGET = (
    "future_return_5m"
)

CLASSIFICATION_TARGET = (
    "target_direction_5m"
)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_feature_columns():
    """Load the V2 feature list."""

    if not FEATURE_COLUMNS_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: "
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

    if not columns:
        raise ValueError(
            "Feature column list is empty."
        )

    return columns


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: "
            f"{TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: "
            f"{VALIDATION_FILE}"
        )

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    train_df["open_time"] = pd.to_datetime(
        train_df["open_time"]
    )

    validation_df["open_time"] = pd.to_datetime(
        validation_df["open_time"]
    )

    train_df = (
        train_df
        .sort_values(
            ["open_time", "symbol"]
        )
        .reset_index(drop=True)
    )

    validation_df = (
        validation_df
        .sort_values(
            ["open_time", "symbol"]
        )
        .reset_index(drop=True)
    )

    return train_df, validation_df


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    df,
    feature_columns,
):
    """Prepare numeric features and symbol indicators."""

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
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

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return X


def prepare_target_dataset(
    df,
    target,
    feature_columns,
):
    """Prepare a clean X/y dataset."""

    X = prepare_features(
        df,
        feature_columns,
    )

    y = (
        df[target]
        .reset_index(drop=True)
    )

    combined = pd.concat(
        [
            X,
            y.rename(target),
        ],
        axis=1,
    )

    combined = combined.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    combined = combined.dropna()

    X = combined.drop(
        columns=[target]
    )

    y = combined[target]

    return X, y


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_sample_weights(y):
    """Create balanced sample weights."""

    class_counts = y.value_counts()

    total = len(y)
    class_count = len(class_counts)

    class_weights = {
        label:
            total
            / (
                class_count
                * count
            )
        for label, count
        in class_counts.items()
    }

    return (
        y.map(class_weights)
        .to_numpy()
    )


# ============================================================
# TARGET ENCODING
# ============================================================

def encode_target(
    y_train,
    y_validation,
):
    """Encode DOWN/NEUTRAL/UP deterministically."""

    classes = sorted(
        y_train.unique()
    )

    mapping = {
        label: index
        for index, label
        in enumerate(classes)
    }

    train_encoded = y_train.map(
        mapping
    )

    validation_encoded = (
        y_validation.map(mapping)
    )

    return (
        train_encoded,
        validation_encoded,
        classes,
    )


# ============================================================
# 1-MINUTE CATBOOST
# ============================================================

def train_1m_catboost(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """
    Train the CatBoost configuration that was the
    best V3 baseline for the 1-minute return target.
    """

    print()
    print("=" * 70)
    print("TUNED CANDIDATE: CATBOOST 1-MINUTE REGRESSION")
    print("=" * 70)

    model = CatBoostRegressor(
        iterations=300,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions,
        )
    )

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    r2 = r2_score(
        y_validation,
        predictions,
    )

    print(
        f"RMSE={rmse:.8f} | "
        f"MAE={mae:.8f} | "
        f"R2={r2:.6f}"
    )

    return model, {
        "task": "regression",
        "target": REGRESSION_1M_TARGET,
        "model": "CatBoost",
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "macro_f1": np.nan,
        "accuracy": np.nan,
    }


# ============================================================
# 5-MINUTE CATBOOST
# ============================================================

def train_5m_catboost(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """
    Train the best CatBoost configuration discovered
    by Optuna on the full training set.
    """

    print()
    print("=" * 70)
    print("TUNED CANDIDATE: CATBOOST 5-MINUTE REGRESSION")
    print("=" * 70)

    model = CatBoostRegressor(
        iterations=425,
        depth=5,
        learning_rate=0.026959743909391057,
        l2_leaf_reg=4.109669467732618,
        random_strength=1.9752914677327187,
        bagging_temperature=2.906925624530661,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions,
        )
    )

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    r2 = r2_score(
        y_validation,
        predictions,
    )

    print(
        f"RMSE={rmse:.8f} | "
        f"MAE={mae:.8f} | "
        f"R2={r2:.6f}"
    )

    return model, {
        "task": "regression",
        "target": REGRESSION_5M_TARGET,
        "model": "CatBoost_Optuna",
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "macro_f1": np.nan,
        "accuracy": np.nan,
    }


# ============================================================
# LIGHTGBM CLASSIFIER
# ============================================================

def train_lightgbm_classifier(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """
    Train the best LightGBM configuration discovered
    by Optuna on the full training set.
    """

    print()
    print("=" * 70)
    print("TUNED CANDIDATE: LIGHTGBM 5-MINUTE CLASSIFICATION")
    print("=" * 70)

    sample_weights = (
        calculate_sample_weights(
            pd.Series(y_train)
        )
    )

    model = LGBMClassifier(
        n_estimators=393,
        learning_rate=0.010336245845523069,
        num_leaves=32,
        max_depth=11,
        min_child_samples=11,
        subsample=0.7956949411279022,
        colsample_bytree=0.6719810794263651,
        reg_alpha=3.145482744647803,
        reg_lambda=0.011465168833811051,
        objective="multiclass",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

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

    print(
        f"Macro-F1={macro_f1:.6f} | "
        f"Accuracy={accuracy:.6f}"
    )

    return model, {
        "task": "classification",
        "target": CLASSIFICATION_TARGET,
        "model": "LightGBM_Optuna",
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
        "rmse": np.nan,
        "mae": np.nan,
        "r2": np.nan,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRYPTO MLOPS - FULL-DATA TUNED CANDIDATES")
    print("=" * 70)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        "sqlite:///"
        f"{MLFLOW_DB.resolve().as_posix()}"
    )

    mlflow.set_experiment(
        "CryptoML_Final_Candidates"
    )

    feature_columns = (
        load_feature_columns()
    )

    train_df, validation_df = (
        load_data()
    )

    print()
    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Feature columns: "
        f"{len(feature_columns)}"
    )

    results = []

    # ========================================================
    # 1-MINUTE REGRESSION
    # ========================================================

    (
        X_train_1m,
        y_train_1m,
    ) = prepare_target_dataset(
        train_df,
        REGRESSION_1M_TARGET,
        feature_columns,
    )

    (
        X_validation_1m,
        y_validation_1m,
    ) = prepare_target_dataset(
        validation_df,
        REGRESSION_1M_TARGET,
        feature_columns,
    )

    with mlflow.start_run(
        run_name="FinalCandidate_CatBoost_1M"
    ):

        model_1m, result_1m = (
            train_1m_catboost(
                X_train=X_train_1m,
                y_train=y_train_1m,
                X_validation=X_validation_1m,
                y_validation=y_validation_1m,
            )
        )

        mlflow.log_param(
            "model_family",
            "CatBoost",
        )

        mlflow.log_param(
            "target",
            REGRESSION_1M_TARGET,
        )

        mlflow.log_param(
            "training_rows",
            len(X_train_1m),
        )

        mlflow.log_param(
            "feature_count",
            X_train_1m.shape[1],
        )

        mlflow.log_param(
            "selection_source",
            "V3_baseline",
        )

        mlflow.log_metric(
            "rmse",
            result_1m["rmse"],
        )

        mlflow.log_metric(
            "mae",
            result_1m["mae"],
        )

        mlflow.log_metric(
            "r2",
            result_1m["r2"],
        )

        mlflow.sklearn.log_model(
            sk_model=model_1m,
            name="model",
            serialization_format="cloudpickle",
        )

    results.append(result_1m)

    # ========================================================
    # 5-MINUTE REGRESSION
    # ========================================================

    (
        X_train_5m,
        y_train_5m,
    ) = prepare_target_dataset(
        train_df,
        REGRESSION_5M_TARGET,
        feature_columns,
    )

    (
        X_validation_5m,
        y_validation_5m,
    ) = prepare_target_dataset(
        validation_df,
        REGRESSION_5M_TARGET,
        feature_columns,
    )

    with mlflow.start_run(
        run_name="FinalCandidate_CatBoost_5M_Optuna"
    ):

        model_5m, result_5m = (
            train_5m_catboost(
                X_train=X_train_5m,
                y_train=y_train_5m,
                X_validation=X_validation_5m,
                y_validation=y_validation_5m,
            )
        )

        mlflow.log_param(
            "model_family",
            "CatBoost",
        )

        mlflow.log_param(
            "target",
            REGRESSION_5M_TARGET,
        )

        mlflow.log_param(
            "training_rows",
            len(X_train_5m),
        )

        mlflow.log_param(
            "feature_count",
            X_train_5m.shape[1],
        )

        mlflow.log_param(
            "selection_source",
            "Optuna",
        )

        mlflow.log_metric(
            "rmse",
            result_5m["rmse"],
        )

        mlflow.log_metric(
            "mae",
            result_5m["mae"],
        )

        mlflow.log_metric(
            "r2",
            result_5m["r2"],
        )

        mlflow.sklearn.log_model(
            sk_model=model_5m,
            name="model",
            serialization_format="cloudpickle",
        )

    results.append(result_5m)

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    (
        X_train_cls,
        y_train_cls_raw,
    ) = prepare_target_dataset(
        train_df,
        CLASSIFICATION_TARGET,
        feature_columns,
    )

    (
        X_validation_cls,
        y_validation_cls_raw,
    ) = prepare_target_dataset(
        validation_df,
        CLASSIFICATION_TARGET,
        feature_columns,
    )

    (
        y_train_cls,
        y_validation_cls,
        classes,
    ) = encode_target(
        y_train_cls_raw,
        y_validation_cls_raw,
    )

    with mlflow.start_run(
        run_name="FinalCandidate_LightGBM_5M_Optuna"
    ):

        model_cls, result_cls = (
            train_lightgbm_classifier(
                X_train=X_train_cls,
                y_train=y_train_cls,
                X_validation=X_validation_cls,
                y_validation=y_validation_cls,
            )
        )

        mlflow.log_param(
            "model_family",
            "LightGBM",
        )

        mlflow.log_param(
            "target",
            CLASSIFICATION_TARGET,
        )

        mlflow.log_param(
            "classes",
            ",".join(classes),
        )

        mlflow.log_param(
            "training_rows",
            len(X_train_cls),
        )

        mlflow.log_param(
            "feature_count",
            X_train_cls.shape[1],
        )

        mlflow.log_param(
            "selection_source",
            "Optuna",
        )

        mlflow.log_metric(
            "macro_f1",
            result_cls["macro_f1"],
        )

        mlflow.log_metric(
            "accuracy",
            result_cls["accuracy"],
        )

        mlflow.sklearn.log_model(
            sk_model=model_cls,
            name="model",
            serialization_format="cloudpickle",
        )

    results.append(result_cls)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("FULL-DATA TUNED CANDIDATES COMPLETE")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()