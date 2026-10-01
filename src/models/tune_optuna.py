from pathlib import Path

import json

import mlflow
import optuna
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
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

OPTUNA_RESULTS_FILE = (
    REPORTS_DIR / "optuna_results.csv"
)

BEST_PARAMS_FILE = (
    REPORTS_DIR / "optuna_best_params.json"
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

REGRESSION_TARGET = "future_return_5m"

CLASSIFICATION_TARGET = (
    "target_direction_5m"
)

MAX_TRAIN_ROWS = 500_000

N_TRIALS = 20

RANDOM_STATE = 42


# ============================================================
# DATA LOADING
# ============================================================

def load_feature_columns():
    """Load the V2 feature list."""

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

    return columns


def load_data():
    """Load V2 training and validation data."""

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

    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.tail(
            MAX_TRAIN_ROWS
        ).copy()

    return train_df, validation_df


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    df,
    feature_columns,
):
    """Prepare numeric model features."""

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
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
    """Create clean X/y data."""

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
    """Calculate balanced sample weights."""

    class_counts = y.value_counts()

    total = len(y)
    number_of_classes = len(
        class_counts
    )

    class_weights = {
        label:
            total
            / (
                number_of_classes
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
# CATEGORICAL ENCODING
# ============================================================

def encode_target(y_train, y_validation):

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
# MLFLOW
# ============================================================

def setup_mlflow():
    """Configure local MLflow."""

    mlflow.set_tracking_uri(
        "sqlite:///"
        f"{MLFLOW_DB.resolve().as_posix()}"
    )

    mlflow.set_experiment(
        "CryptoML_Optuna"
    )


# ============================================================
# CATBOOST REGRESSION
# ============================================================

def tune_catboost_regression(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Tune CatBoost for 5-minute return prediction."""

    print()
    print("=" * 70)
    print("OPTUNA: CATBOOST 5-MINUTE REGRESSION")
    print("=" * 70)

    def objective(trial):

        params = {
            "iterations": trial.suggest_int(
                "iterations",
                200,
                800,
            ),
            "depth": trial.suggest_int(
                "depth",
                5,
                10,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.15,
                log=True,
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                1.0,
                20.0,
                log=True,
            ),
            "random_strength": trial.suggest_float(
                "random_strength",
                0.0,
                2.0,
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature",
                0.0,
                3.0,
            ),
        }

        model = CatBoostRegressor(
            **params,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=RANDOM_STATE,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )

        with mlflow.start_run(
            run_name=(
                f"Optuna_CatBoost_"
                f"Trial_{trial.number}"
            ),
            nested=True,
        ) as run:

            model.fit(
                X_train,
                y_train,
                eval_set=(
                    X_validation,
                    y_validation,
                ),
                early_stopping_rounds=50,
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

            r2 = (
                1
                - (
                    np.sum(
                        (
                            y_validation
                            - predictions
                        ) ** 2
                    )
                    / np.sum(
                        (
                            y_validation
                            - y_validation.mean()
                        ) ** 2
                    )
                )
            )

            mlflow.log_params(
                params
            )

            mlflow.log_param(
                "model_family",
                "CatBoost",
            )

            mlflow.log_param(
                "task",
                "regression",
            )

            mlflow.log_param(
                "target",
                REGRESSION_TARGET,
            )

            mlflow.log_param(
                "train_rows",
                len(X_train),
            )

            mlflow.log_param(
                "validation_rows",
                len(X_validation),
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

            trial.set_user_attr(
                "mlflow_run_id",
                run.info.run_id,
            )

        print(
            f"Trial {trial.number:02d} | "
            f"RMSE={rmse:.8f} | "
            f"MAE={mae:.8f} | "
            f"R2={r2:.6f}"
        )

        return rmse

    study = optuna.create_study(
        study_name=(
            "CatBoost_5M_Return"
        ),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
    )

    return study


# ============================================================
# LIGHTGBM CLASSIFICATION
# ============================================================

def tune_lightgbm_classification(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Tune LightGBM for 5-minute direction."""

    print()
    print("=" * 70)
    print("OPTUNA: LIGHTGBM 5-MINUTE CLASSIFICATION")
    print("=" * 70)

    sample_weights = (
        calculate_sample_weights(
            pd.Series(y_train)
        )
    )

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                800,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.15,
                log=True,
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                15,
                127,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12,
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                10,
                200,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-4,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-4,
                10.0,
                log=True,
            ),
        }

        model = LGBMClassifier(
            **params,
            objective="multiclass",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )

        with mlflow.start_run(
            run_name=(
                f"Optuna_LightGBM_"
                f"Trial_{trial.number}"
            ),
            nested=True,
        ) as run:

            model.fit(
                X_train,
                y_train,
                sample_weight=(
                    sample_weights
                ),
                eval_set=[
                    (
                        X_validation,
                        y_validation,
                    )
                ],
                callbacks=[],
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

            mlflow.log_params(
                params
            )

            mlflow.log_param(
                "model_family",
                "LightGBM",
            )

            mlflow.log_param(
                "task",
                "classification",
            )

            mlflow.log_param(
                "target",
                CLASSIFICATION_TARGET,
            )

            mlflow.log_param(
                "train_rows",
                len(X_train),
            )

            mlflow.log_param(
                "validation_rows",
                len(X_validation),
            )

            mlflow.log_metric(
                "macro_f1",
                macro_f1,
            )

            mlflow.log_metric(
                "accuracy",
                accuracy,
            )

            trial.set_user_attr(
                "mlflow_run_id",
                run.info.run_id,
            )

        print(
            f"Trial {trial.number:02d} | "
            f"Macro-F1={macro_f1:.6f} | "
            f"Accuracy={accuracy:.6f}"
        )

        return (
            1.0 - macro_f1
        )

    study = optuna.create_study(
        study_name=(
            "LightGBM_5M_Direction"
        ),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
    )

    return study


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    regression_study,
    classification_study,
):
    """Save Optuna results and best parameters."""

    rows = []

    for trial in (
        regression_study.trials
    ):

        rows.append(
            {
                "model": "CatBoost",
                "task": "regression",
                "target": REGRESSION_TARGET,
                "trial": trial.number,
                "state": trial.state.name,
                "objective": trial.value,
                "rmse": trial.value,
                "mlflow_run_id": trial.user_attrs.get(
                    "mlflow_run_id"
                ),
            }
        )

    for trial in (
        classification_study.trials
    ):

        rows.append(
            {
                "model": "LightGBM",
                "task": "classification",
                "target": CLASSIFICATION_TARGET,
                "trial": trial.number,
                "state": trial.state.name,
                "objective": trial.value,
                "rmse": np.nan,
                "mlflow_run_id": trial.user_attrs.get(
                    "mlflow_run_id"
                ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        OPTUNA_RESULTS_FILE,
        index=False,
    )

    best_params = {
        "CatBoost_regression": {
            "target": REGRESSION_TARGET,
            "best_value": (
                regression_study.best_value
            ),
            "best_params": (
                regression_study.best_params
            ),
        },
        "LightGBM_classification": {
            "target": CLASSIFICATION_TARGET,
            "best_macro_f1": (
                1.0
                - classification_study.best_value
            ),
            "best_params": (
                classification_study.best_params
            ),
        },
    }

    with open(
        BEST_PARAMS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_params,
            file,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRYPTO MLOPS - OPTUNA HYPERPARAMETER TUNING")
    print("=" * 70)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    setup_mlflow()

    feature_columns = (
        load_feature_columns()
    )

    train_df, validation_df = (
        load_data()
    )

    print()
    print(
        f"Tuning train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Optuna trials: "
        f"{N_TRIALS}"
    )

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    X_train_reg, y_train_reg = (
        prepare_target_dataset(
            train_df,
            REGRESSION_TARGET,
            feature_columns,
        )
    )

    X_validation_reg, y_validation_reg = (
        prepare_target_dataset(
            validation_df,
            REGRESSION_TARGET,
            feature_columns,
        )
    )

    regression_study = (
        tune_catboost_regression(
            X_train=X_train_reg,
            y_train=y_train_reg,
            X_validation=X_validation_reg,
            y_validation=y_validation_reg,
        )
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    X_train_cls, y_train_cls = (
        prepare_target_dataset(
            train_df,
            CLASSIFICATION_TARGET,
            feature_columns,
        )
    )

    X_validation_cls, y_validation_cls = (
        prepare_target_dataset(
            validation_df,
            CLASSIFICATION_TARGET,
            feature_columns,
        )
    )

    (
        y_train_cls,
        y_validation_cls,
        classes,
    ) = encode_target(
        y_train_cls,
        y_validation_cls,
    )

    classification_study = (
        tune_lightgbm_classification(
            X_train=X_train_cls,
            y_train=y_train_cls,
            X_validation=X_validation_cls,
            y_validation=y_validation_cls,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        regression_study,
        classification_study,
    )

    # --------------------------------------------------------
    # Print best results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BEST OPTUNA RESULTS")
    print("=" * 70)

    print()
    print("CatBoost 5-minute regression:")
    print(
        f"Best RMSE: "
        f"{regression_study.best_value:.8f}"
    )

    print(
        f"Best parameters: "
        f"{regression_study.best_params}"
    )

    print()
    print("LightGBM 5-minute classification:")

    best_macro_f1 = (
        1.0
        - classification_study.best_value
    )

    print(
        f"Best Macro-F1: "
        f"{best_macro_f1:.6f}"
    )

    print(
        f"Best parameters: "
        f"{classification_study.best_params}"
    )

    print()
    print(
        f"Trial results: "
        f"{OPTUNA_RESULTS_FILE}"
    )

    print(
        f"Best parameters: "
        f"{BEST_PARAMS_FILE}"
    )

    print()
    print("=" * 70)
    print("OPTUNA TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()