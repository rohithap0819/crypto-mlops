"""
Final comparative evaluation of the fixed CatBoost + GRU ensemble.

Validation-selected weights:
    CatBoost = 0.55
    GRU      = 0.45

Important:
    - No test-set tuning.
    - CatBoost is trained on train + validation.
    - GRU uses the already-trained binary GRU checkpoint.
    - Test set is used only for comparative evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.models.binary_sequence_models import BinaryGRU


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

SEQUENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_v1"
)

SEQUENCE_TEST_DIR = (
    SEQUENCE_ROOT / "test"
)

GRU_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "binary_sequence_v1"
    / "gru_binary_best.pt"
)

REPORTS_DIR = (
    PROJECT_ROOT / "reports"
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

SEQUENCE_LENGTH = 60
FEATURE_COUNT = 61

CATBOOST_WEIGHT = 0.55
GRU_WEIGHT = 0.45

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HELPERS
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


def normalize_time(values) -> pd.Series:

    series = pd.Series(values)

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(
            series,
            utc=True,
        )

    if pd.api.types.is_numeric_dtype(series):

        numeric_values = series.astype(float)

        median_value = np.nanmedian(
            numeric_values
        )

        # Sequence time.npy uses microseconds.
        if median_value > 1e14:
            return pd.to_datetime(
                series,
                unit="us",
                utc=True,
            )

        if median_value > 1e11:
            return pd.to_datetime(
                series,
                unit="ms",
                utc=True,
            )

        if median_value > 1e9:
            return pd.to_datetime(
                series,
                unit="s",
                utc=True,
            )

    return pd.to_datetime(
        series,
        utc=True,
    )


def make_binary_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
):

    df = df.copy()

    mask = (
        df[TARGET_COLUMN].notna()
        & (df[TARGET_COLUMN] != 0)
    )

    df = df.loc[mask].reset_index(drop=True)

    X = prepare_features(
        df,
        feature_columns,
    )

    y = (
        df[TARGET_COLUMN] > 0
    ).astype(np.int64)

    return X, y, df


# ============================================================
# CATBOOST
# ============================================================

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):

    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=42,
        thread_count=-1,
        verbose=False,
    )

    print("\nTraining final CatBoost on train + validation...")

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# GRU
# ============================================================

def load_gru():

    model = BinaryGRU(
        input_size=FEATURE_COUNT,
        hidden_size=64,
        dropout=0.2,
        num_symbols=len(SYMBOLS),
        symbol_embedding_dim=4,
    ).to(DEVICE)

    checkpoint = torch.load(
        GRU_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def build_gru_test_predictions():

    model = load_gru()

    records = []

    print("\nGenerating GRU test predictions...")

    for symbol_id, symbol in enumerate(SYMBOLS):

        X_path = (
            SEQUENCE_TEST_DIR
            / f"{symbol}_X.npy"
        )

        valid_indices_path = (
            SEQUENCE_TEST_DIR
            / f"{symbol}_valid_indices.npy"
        )

        return_path = (
            SEQUENCE_TEST_DIR
            / f"{symbol}_return_5m.npy"
        )

        time_path = (
            SEQUENCE_TEST_DIR
            / f"{symbol}_time.npy"
        )

        X = np.load(X_path)
        valid_indices = np.load(
            valid_indices_path
        ).astype(np.int64)

        future_returns = np.load(
            return_path
        )

        times = np.load(
            time_path
        )

        valid_mask = (
            valid_indices
            >= SEQUENCE_LENGTH - 1
        )

        valid_indices = valid_indices[
            valid_mask
        ]

        target_returns = (
            future_returns[
                valid_indices
            ]
        )

        endpoint_times = (
            times[
                valid_indices
            ]
        )

        nonzero_mask = (
            target_returns != 0
        )

        valid_indices = (
            valid_indices[
                nonzero_mask
            ]
        )

        target_returns = (
            target_returns[
                nonzero_mask
            ]
        )

        endpoint_times = (
            endpoint_times[
                nonzero_mask
            ]
        )

        sequences = []

        for end_index in valid_indices:

            start_index = (
                end_index
                - SEQUENCE_LENGTH
                + 1
            )

            sequences.append(
                X[
                    start_index:
                    end_index + 1
                ]
            )

        sequences = np.asarray(
            sequences,
            dtype=np.float32,
        )

        symbol_ids = np.full(
            len(sequences),
            symbol_id,
            dtype=np.int64,
        )

        probabilities = []

        batch_size = 256

        with torch.no_grad():

            for start in range(
                0,
                len(sequences),
                batch_size,
            ):

                end = min(
                    start + batch_size,
                    len(sequences),
                )

                X_batch = torch.tensor(
                    sequences[start:end],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                symbol_batch = torch.tensor(
                    symbol_ids[start:end],
                    dtype=torch.long,
                    device=DEVICE,
                )

                logits = model(
                    X_batch,
                    symbol_batch,
                )

                probs = torch.sigmoid(
                    logits
                )

                probabilities.append(
                    probs.cpu().numpy()
                )

        probabilities = np.concatenate(
            probabilities
        )

        records.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "test_time": normalize_time(
                        endpoint_times
                    ),
                    "gru_probability_up":
                        probabilities,
                    "gru_target":
                        (
                            target_returns > 0
                        ).astype(np.int64),
                }
            )
        )

        print(
            f"{symbol}: "
            f"{len(probabilities):,} sequences"
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FIXED 55/45 CATBOOST + GRU TEST EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    feature_columns = (
        load_feature_columns()
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading train/validation/test data...")

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    test_df = pd.read_parquet(
        TEST_FILE
    )

    print(
        f"Train rows      : {len(train_df):,}"
    )

    print(
        f"Validation rows : {len(validation_df):,}"
    )

    print(
        f"Test rows       : {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Build full CatBoost training set
    # --------------------------------------------------------

    X_train, y_train, _ = make_binary_dataset(
        train_df,
        feature_columns,
    )

    X_validation, y_validation, _ = (
        make_binary_dataset(
            validation_df,
            feature_columns,
        )
    )

    X_test, y_test, test_clean = (
        make_binary_dataset(
            test_df,
            feature_columns,
        )
    )

    X_full_train = pd.concat(
        [
            X_train,
            X_validation,
        ],
        ignore_index=True,
    )

    y_full_train = pd.concat(
        [
            y_train,
            y_validation,
        ],
        ignore_index=True,
    )

    print(
        f"\nFull CatBoost training rows: "
        f"{len(X_full_train):,}"
    )

    print(
        f"Binary test rows: "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # Train CatBoost
    # --------------------------------------------------------

    catboost_model = train_catboost(
        X_full_train,
        y_full_train,
    )

    catboost_probability = (
        catboost_model.predict_proba(
            X_test
        )[:, 1]
    )

    catboost_prediction = (
        catboost_probability >= 0.5
    ).astype(np.int64)

    # --------------------------------------------------------
    # GRU
    # --------------------------------------------------------

    gru_predictions = (
        build_gru_test_predictions()
    )

    # --------------------------------------------------------
    # Prepare CatBoost timestamps
    # --------------------------------------------------------

    test_clean = test_clean.copy()

    test_clean[
        "test_time"
    ] = normalize_time(
        test_clean["open_time"]
    )

    test_clean[
        "catboost_probability_up"
    ] = catboost_probability

    test_clean[
        "catboost_prediction"
    ] = catboost_prediction

    test_clean[
        "catboost_target"
    ] = y_test.to_numpy()

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = test_clean[
        [
            "symbol",
            "test_time",
            "catboost_probability_up",
            "catboost_prediction",
            "catboost_target",
        ]
    ].merge(
        gru_predictions,
        on=[
            "symbol",
            "test_time",
        ],
        how="inner",
    )

    print(
        f"\nMatched CatBoost/GRU test rows: "
        f"{len(merged):,}"
    )

    if len(merged) < 100_000:
        raise ValueError(
            "Too few rows matched between CatBoost and GRU."
        )

    target_match = np.mean(
        merged["catboost_target"].to_numpy()
        ==
        merged["gru_target"].to_numpy()
    )

    print(
        f"Target agreement: {target_match:.6f}"
    )

    if target_match < 0.999:
        raise ValueError(
            "CatBoost and GRU targets do not agree."
        )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_true = (
        merged["catboost_target"]
        .to_numpy()
        .astype(np.int64)
    )

    cb_prob = (
        merged["catboost_probability_up"]
        .to_numpy()
    )

    gru_prob = (
        merged["gru_probability_up"]
        .to_numpy()
    )

    cb_pred = (
        cb_prob >= 0.5
    ).astype(np.int64)

    gru_pred = (
        gru_prob >= 0.5
    ).astype(np.int64)

    ensemble_probability = (
        CATBOOST_WEIGHT * cb_prob
        +
        GRU_WEIGHT * gru_prob
    )

    ensemble_prediction = (
        ensemble_probability >= 0.5
    ).astype(np.int64)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    models = {
        "CatBoost": cb_pred,
        "GRU": gru_pred,
        "CatBoost_GRU_55_45": ensemble_prediction,
    }

    results = []

    for model_name, prediction in models.items():

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(
                    y_true,
                    prediction,
                ),
                "macro_f1": f1_score(
                    y_true,
                    prediction,
                    average="macro",
                ),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Inverse 5m baseline
    #
    # Prediction = opposite sign of previous 5m return.
    # return_5m is the current 5-minute return feature.
    # --------------------------------------------------------

    inverse_previous_5m = (
        test_clean.loc[
            merged.index,
            "return_5m"
        ]
        if "return_5m" in test_clean.columns
        else None
    )

    # Calculate baseline directly from merged timestamps
    # using the corresponding test rows.
    #
    # The merged dataframe contains the rows in the same
    # order as the CatBoost test predictions after matching.
    merged_with_return = (
        merged.copy()
    )

    # Reconstruct the previous 5m feature from test_clean.
    baseline_source = test_clean[
        [
            "symbol",
            "test_time",
            "return_5m",
        ]
    ]

    merged_with_return = merged.merge(
        baseline_source,
        on=[
            "symbol",
            "test_time",
        ],
        how="left",
    )

    baseline_prediction = (
        merged_with_return[
            "return_5m"
        ].to_numpy() < 0
    ).astype(np.int64)

    baseline_metrics = {
        "model": "Inverse_Previous_5m",
        "accuracy": accuracy_score(
            y_true,
            baseline_prediction,
        ),
        "macro_f1": f1_score(
            y_true,
            baseline_prediction,
            average="macro",
        ),
    }

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [baseline_metrics]
            ),
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL COMPARATIVE TEST RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nClassification reports:")

    for model_name, prediction in models.items():

        print(
            f"\n--- {model_name} ---"
        )

        print(
            classification_report(
                y_true,
                prediction,
                target_names=[
                    "DOWN",
                    "UP",
                ],
                digits=6,
            )
        )

    print(
        "\n--- Inverse Previous 5m ---"
    )

    print(
        classification_report(
            y_true,
            baseline_prediction,
            target_names=[
                "DOWN",
                "UP",
            ],
            digits=6,
        )
    )

    print("\nEnsemble confusion matrix:")

    print(
        confusion_matrix(
            y_true,
            ensemble_prediction,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        REPORTS_DIR
        / "binary_ensemble_test_comparison_v1.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    ensemble_cm_path = (
        REPORTS_DIR
        / "binary_ensemble_test_confusion_v1.csv"
    )

    np.savetxt(
        ensemble_cm_path,
        confusion_matrix(
            y_true,
            ensemble_prediction,
        ),
        delimiter=",",
        fmt="%d",
    )

    metadata = {
        "catboost_weight": CATBOOST_WEIGHT,
        "gru_weight": GRU_WEIGHT,
        "matched_test_rows": int(
            len(merged)
        ),
        "target_agreement": float(
            target_match
        ),
        "note": (
            "Weights were selected on validation "
            "and frozen before this test evaluation."
        ),
        "test_usage_note": (
            "The test set has previously been used "
            "for individual CatBoost and GRU evaluation, "
            "so this is a final comparative evaluation "
            "rather than a pristine untouched holdout."
        ),
    }

    metadata_path = (
        REPORTS_DIR
        / "binary_ensemble_test_metadata_v1.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nReports saved:")
    print(results_path)
    print(ensemble_cm_path)
    print(metadata_path)


if __name__ == "__main__":
    main()