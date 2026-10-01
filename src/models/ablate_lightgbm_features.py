"""
LightGBM feature-ablation experiment.

Goal:
    Determine how much of the V3 LightGBM classification signal
    comes from volatility/regime features and time features.

Experiments:
    A. All features
    B. Remove volatility/regime features
    C. Remove time features
    D. Remove volatility + time features

All experiments use:
    - Same V2 train/validation splits
    - Same target
    - Same class encoding
    - Same LightGBM configuration
    - Same random state
    - Same class_weight="balanced"

Outputs:
    reports/lightgbm_feature_ablation_v1.csv
    reports/lightgbm_feature_ablation_per_class_v1.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
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

TARGET = "target_direction_5m"

RANDOM_STATE = 42


# ============================================================
# ORIGINAL V3 LIGHTGBM CONFIGURATION
# ============================================================

LIGHTGBM_PARAMS = {
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


# ============================================================
# FEATURE GROUPS
# ============================================================

VOLATILITY_FEATURES = {
    "rolling_return_std_5",
    "rolling_return_std_15",
    "rolling_return_std_30",
    "rolling_return_std_60",
    "volatility_15m",
    "volatility_60m",
    "atr_14_pct",
    "market_volatility_15m",
}

TIME_FEATURES = {
    "time_sin",
    "time_cos",
    "day_sin",
    "day_cos",
}


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

def load_feature_columns():
    """Load the exact V2 feature list."""

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
            X.reset_index(drop=True),
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


def prepare_dataset(
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
        .reset_index(drop=True)
    )

    combined = pd.concat(
        [
            X,
            y.rename(target_column),
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
# CLASS ENCODING
# ============================================================

def encode_targets(
    y_train,
    y_validation,
):

    classes = sorted(
        y_train.unique()
    )

    class_to_id = {
        label: index
        for index, label in enumerate(
            classes
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

    return (
        classes,
        y_train_encoded,
        y_validation_encoded,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
    classes,
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
            labels=list(
                range(
                    len(classes)
                )
            ),
            zero_division=0,
        )
    )

    per_class = []

    for index, class_name in enumerate(
        classes
    ):

        per_class.append(
            {
                "class": class_name,
                "precision": float(
                    precision[index]
                ),
                "recall": float(
                    recall[index]
                ),
                "f1": float(
                    f1[index]
                ),
                "support": int(
                    support[index]
                ),
            }
        )

    return (
        macro_f1,
        accuracy,
        per_class,
    )


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

def build_experiments(
    feature_columns,
):

    return {
        "A_all_features": {
            "removed_features": [],
            "features": list(
                feature_columns
            ),
        },

        "B_no_volatility": {
            "removed_features": sorted(
                VOLATILITY_FEATURES
            ),
            "features": [
                feature
                for feature in feature_columns
                if feature
                not in VOLATILITY_FEATURES
            ],
        },

        "C_no_time": {
            "removed_features": sorted(
                TIME_FEATURES
            ),
            "features": [
                feature
                for feature in feature_columns
                if feature
                not in TIME_FEATURES
            ],
        },

        "D_no_volatility_no_time": {
            "removed_features": sorted(
                VOLATILITY_FEATURES
                | TIME_FEATURES
            ),
            "features": [
                feature
                for feature in feature_columns
                if feature
                not in (
                    VOLATILITY_FEATURES
                    | TIME_FEATURES
                )
            ],
        },
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
        "LIGHTGBM FEATURE ABLATION"
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
    # ORIGINAL FEATURE DATA
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "PREPARING BASE DATA"
    )
    print(
        "=" * 70
    )

    X_train_full, y_train = (
        prepare_dataset(
            train_df,
            TARGET,
            feature_columns,
        )
    )

    X_validation_full, y_validation = (
        prepare_dataset(
            validation_df,
            TARGET,
            feature_columns,
        )
    )

    (
        classes,
        y_train_encoded,
        y_validation_encoded,
    ) = encode_targets(
        y_train,
        y_validation,
    )

    print(
        f"Original model features: "
        f"{X_train_full.shape[1]}"
    )

    print(
        f"Classes: {classes}"
    )

    experiments = build_experiments(
        feature_columns
    )

    results = []

    per_class_results = []

    # --------------------------------------------------------
    # RUN EACH EXPERIMENT
    # --------------------------------------------------------

    for experiment_name, config in (
        experiments.items()
    ):

        selected_feature_names = (
            config["features"]
        )

        removed_features = (
            config["removed_features"]
        )

        final_feature_names = (
            selected_feature_names
            + [
                f"symbol_{symbol}"
                for symbol in SYMBOLS
            ]
        )

        X_train = X_train_full[
            final_feature_names
        ]

        X_validation = (
            X_validation_full[
                final_feature_names
            ]
        )

        print()
        print("=" * 70)
        print(
            f"EXPERIMENT: "
            f"{experiment_name}"
        )
        print("=" * 70)

        print(
            f"V2 features used: "
            f"{len(selected_feature_names)}"
        )

        print(
            f"Total features including symbols: "
            f"{X_train.shape[1]}"
        )

        print(
            "Removed:"
        )

        if removed_features:
            for feature in removed_features:
                print(
                    f"  - {feature}"
                )
        else:
            print(
                "  None"
            )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = LGBMClassifier(
            **LIGHTGBM_PARAMS
        )

        print()
        print(
            "Training LightGBM..."
        )

        model.fit(
            X_train,
            y_train_encoded,
        )

        print(
            "Training complete."
        )

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        (
            macro_f1,
            accuracy,
            per_class,
        ) = evaluate_model(
            model,
            X_validation,
            y_validation_encoded,
            classes,
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

        results.append(
            {
                "experiment": (
                    experiment_name
                ),
                "v2_feature_count": (
                    len(
                        selected_feature_names
                    )
                ),
                "total_feature_count": (
                    X_train.shape[1]
                ),
                "removed_feature_count": (
                    len(
                        removed_features
                    )
                ),
                "macro_f1": (
                    macro_f1
                ),
                "accuracy": (
                    accuracy
                ),
            }
        )

        # ----------------------------------------------------
        # PER-CLASS RESULTS
        # ----------------------------------------------------

        for row in per_class:

            per_class_results.append(
                {
                    "experiment": (
                        experiment_name
                    ),
                    **row,
                }
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    per_class_df = pd.DataFrame(
        per_class_results
    )

    results_path = (
        REPORTS_DIR
        / "lightgbm_feature_ablation_v1.csv"
    )

    per_class_path = (
        REPORTS_DIR
        / "lightgbm_feature_ablation_per_class_v1.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    per_class_df.to_csv(
        per_class_path,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "ABLATION SUMMARY"
    )
    print("=" * 70)

    print()

    print(
        results_df.to_string(
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
    print("=" * 70)
    print(
        "REPORTS SAVED"
    )
    print("=" * 70)

    print(
        results_path
    )

    print(
        per_class_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()