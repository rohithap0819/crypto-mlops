from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


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
    REPORTS_DIR / "baseline_results_v2.csv"
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

REGRESSION_TARGETS = [
    "next_return_1m",
    "future_return_5m",
]

CLASSIFICATION_TARGET = (
    "target_direction_5m"
)

MAX_TRAIN_ROWS = 500_000

RANDOM_STATE = 42


# ============================================================
# FEATURE COLUMNS
# ============================================================

def load_feature_columns():
    """Load V2 feature columns generated during feature engineering."""

    if not FEATURE_COLUMNS_FILE.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: "
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
            "No feature columns found."
        )

    return columns


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load V2 train and validation datasets."""

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
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
# TRAINING SIZE
# ============================================================

def limit_training_rows(train_df):
    """
    Keep the most recent chronological rows for the baseline
    experiment.
    """

    if len(train_df) <= MAX_TRAIN_ROWS:
        return train_df.copy()

    return train_df.tail(
        MAX_TRAIN_ROWS
    ).copy()


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    df,
    feature_columns,
):
    """Prepare V2 model features and one-hot encode symbols."""

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

    # --------------------------------------------------------
    # One-hot encode cryptocurrency symbol
    # --------------------------------------------------------

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
    target_column,
    feature_columns,
):
    """Prepare a clean X/y dataset."""

    X = prepare_features(
        df,
        feature_columns,
    )

    y = (
        df[target_column]
        .reset_index(drop=True)
    )

    clean = pd.concat(
        [
            X,
            y.rename(target_column),
        ],
        axis=1,
    )

    clean = clean.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    clean = clean.dropna()

    X = clean.drop(
        columns=[target_column]
    )

    y = clean[target_column]

    return X, y


# ============================================================
# REGRESSION MODELS
# ============================================================

def get_regression_models():

    return {
        "LinearRegression": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LinearRegression(),
                ),
            ]
        ),

        "RandomForestRegressor":
            RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),

        "XGBRegressor":
            XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
    }


# ============================================================
# CLASSIFICATION MODELS
# ============================================================

def get_classification_models():

    return {
        "LogisticRegression": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=300,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),

        "RandomForestClassifier":
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),

        "XGBClassifier":
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
    }


# ============================================================
# XGBOOST CLASS WEIGHTS
# ============================================================

def calculate_sample_weights(y):

    class_counts = y.value_counts()

    total = len(y)

    number_of_classes = (
        len(class_counts)
    )

    class_weights = {
        class_name:
            total
            / (
                number_of_classes
                * count
            )
        for class_name, count
        in class_counts.items()
    }

    return (
        y.map(class_weights)
        .to_numpy()
    )


# ============================================================
# REGRESSION TRAINING
# ============================================================

def train_regression_baselines(
    train_df,
    validation_df,
    target_column,
    feature_columns,
    results,
):

    train_df = limit_training_rows(
        train_df
    )

    X_train, y_train = (
        prepare_target_dataset(
            train_df,
            target_column,
            feature_columns,
        )
    )

    X_validation, y_validation = (
        prepare_target_dataset(
            validation_df,
            target_column,
            feature_columns,
        )
    )

    models = get_regression_models()

    print()
    print("=" * 70)
    print(
        f"V2 REGRESSION TARGET: "
        f"{target_column}"
    )
    print("=" * 70)

    for model_name, model in models.items():

        print()
        print(
            f"Training V2 {model_name}..."
        )

        mlflow.set_experiment(
            "CryptoML_V2_Regression"
        )

        with mlflow.start_run(
            run_name=(
                f"V2_{model_name}_"
                f"{target_column}"
            )
        ) as run:

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

            mlflow.log_param(
                "feature_version",
                "v2",
            )

            mlflow.log_param(
                "model_family",
                model_name,
            )

            mlflow.log_param(
                "target",
                target_column,
            )

            mlflow.log_param(
                "train_rows",
                len(X_train),
            )

            mlflow.log_param(
                "validation_rows",
                len(X_validation),
            )

            mlflow.log_param(
                "feature_count",
                X_train.shape[1],
            )

            mlflow.log_metric(
                "rmse",
                rmse,
            )

            mlflow.log_metric(
                "mae",
                mae,
            )

            mlflow.log_metric(
                "r2",
                r2,
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format=(
                    "cloudpickle"
                ),
            )

            results.append(
                {
                    "feature_version": "v2",
                    "task": "regression",
                    "target": target_column,
                    "model": model_name,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "macro_f1": np.nan,
                    "accuracy": np.nan,
                    "run_id": run.info.run_id,
                }
            )

            print(
                f"RMSE={rmse:.8f} | "
                f"MAE={mae:.8f} | "
                f"R2={r2:.6f}"
            )


# ============================================================
# CLASSIFICATION TRAINING
# ============================================================

def train_classification_baselines(
    train_df,
    validation_df,
    feature_columns,
    results,
):

    train_df = limit_training_rows(
        train_df
    )

    X_train, y_train = (
        prepare_target_dataset(
            train_df,
            CLASSIFICATION_TARGET,
            feature_columns,
        )
    )

    X_validation, y_validation = (
        prepare_target_dataset(
            validation_df,
            CLASSIFICATION_TARGET,
            feature_columns,
        )
    )

    label_encoder = LabelEncoder()

    y_train_encoded = (
        label_encoder.fit_transform(
            y_train
        )
    )

    y_validation_encoded = (
        label_encoder.transform(
            y_validation
        )
    )

    models = (
        get_classification_models()
    )

    print()
    print("=" * 70)
    print(
        "V2 CLASSIFICATION TARGET: "
        f"{CLASSIFICATION_TARGET}"
    )
    print("=" * 70)

    print(
        "Classes: "
        f"{list(label_encoder.classes_)}"
    )

    for model_name, model in models.items():

        print()
        print(
            f"Training V2 {model_name}..."
        )

        mlflow.set_experiment(
            "CryptoML_V2_Classification"
        )

        with mlflow.start_run(
            run_name=(
                f"V2_{model_name}_"
                f"{CLASSIFICATION_TARGET}"
            )
        ) as run:

            if (
                model_name
                == "XGBClassifier"
            ):
                sample_weights = (
                    calculate_sample_weights(
                        pd.Series(
                            y_train_encoded
                        )
                    )
                )

                model.fit(
                    X_train,
                    y_train_encoded,
                    sample_weight=(
                        sample_weights
                    ),
                )

            else:
                model.fit(
                    X_train,
                    y_train_encoded,
                )

            predictions = (
                model.predict(
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

            mlflow.log_param(
                "feature_version",
                "v2",
            )

            mlflow.log_param(
                "model_family",
                model_name,
            )

            mlflow.log_param(
                "target",
                CLASSIFICATION_TARGET,
            )

            mlflow.log_param(
                "classes",
                ",".join(
                    label_encoder.classes_
                ),
            )

            mlflow.log_param(
                "train_rows",
                len(X_train),
            )

            mlflow.log_param(
                "validation_rows",
                len(X_validation),
            )

            mlflow.log_param(
                "feature_count",
                X_train.shape[1],
            )

            mlflow.log_metric(
                "macro_f1",
                macro_f1,
            )

            mlflow.log_metric(
                "accuracy",
                accuracy,
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format=(
                    "cloudpickle"
                ),
            )

            results.append(
                {
                    "feature_version": "v2",
                    "task": "classification",
                    "target": (
                        CLASSIFICATION_TARGET
                    ),
                    "model": model_name,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "r2": np.nan,
                    "macro_f1": macro_f1,
                    "accuracy": accuracy,
                    "run_id": run.info.run_id,
                }
            )

            print(
                f"Macro-F1={macro_f1:.6f} | "
                f"Accuracy={accuracy:.6f}"
            )


# ============================================================
# BEST MODEL SELECTION
# ============================================================

def print_best_models(
    results_df
):

    print()
    print("=" * 70)
    print("BEST V2 BASELINE MODELS")
    print("=" * 70)

    for target in REGRESSION_TARGETS:

        subset = results_df[
            (
                results_df["task"]
                == "regression"
            )
            & (
                results_df["target"]
                == target
            )
        ].sort_values(
            "rmse"
        )

        best = subset.iloc[0]

        print()
        print(
            f"Regression target: "
            f"{target}"
        )

        print(
            f"Best model: "
            f"{best['model']}"
        )

        print(
            f"RMSE: "
            f"{best['rmse']:.8f}"
        )

        print(
            f"MAE: "
            f"{best['mae']:.8f}"
        )

        print(
            f"R2: "
            f"{best['r2']:.6f}"
        )

        print(
            f"MLflow run: "
            f"{best['run_id']}"
        )

    classification = results_df[
        results_df["task"]
        == "classification"
    ].sort_values(
        "macro_f1",
        ascending=False,
    )

    best = classification.iloc[0]

    print()
    print(
        "Classification target: "
        f"{CLASSIFICATION_TARGET}"
    )

    print(
        f"Best model: "
        f"{best['model']}"
    )

    print(
        f"Macro-F1: "
        f"{best['macro_f1']:.6f}"
    )

    print(
        f"Accuracy: "
        f"{best['accuracy']:.6f}"
    )

    print(
        f"MLflow run: "
        f"{best['run_id']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CRYPTO MLOPS - V2 BASELINE "
        "MODEL TRAINING"
    )
    print("=" * 70)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        "sqlite:///"
        f"{MLFLOW_DB.resolve().as_posix()}"
    )

    feature_columns = (
        load_feature_columns()
    )

    train_df, validation_df = (
        load_data()
    )

    print()
    print(
        f"V2 feature columns: "
        f"{len(feature_columns)}"
    )

    print(
        f"Full train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Baseline train limit: "
        f"{min(len(train_df), MAX_TRAIN_ROWS):,}"
    )

    results = []

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    for target_column in (
        REGRESSION_TARGETS
    ):

        train_regression_baselines(
            train_df=train_df,
            validation_df=validation_df,
            target_column=target_column,
            feature_columns=feature_columns,
            results=results,
        )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    train_classification_baselines(
        train_df=train_df,
        validation_df=validation_df,
        feature_columns=feature_columns,
        results=results,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print(
        "V2 baseline results saved to: "
        f"{RESULTS_FILE}"
    )

    print_best_models(
        results_df
    )

    print()
    print("=" * 70)
    print(
        "V2 BASELINE TRAINING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()