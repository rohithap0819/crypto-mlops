"""
Per-coin evaluation of the selected 55/45 CatBoost + GRU ensemble.

This is a diagnostic evaluation only.

Models:
    CatBoost       = 55%
    GRU            = 45%
    Ensemble       = 55% CatBoost + 45% GRU
    Baseline       = inverse previous 5-minute return

The 55/45 ensemble weight is already frozen from validation.
No test-set tuning is performed here.
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

SEQUENCE_TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_v1"
    / "test"
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

        # Microseconds
        if median_value > 1e14:
            return pd.to_datetime(
                series,
                unit="us",
                utc=True,
            )

        # Milliseconds
        if median_value > 1e11:
            return pd.to_datetime(
                series,
                unit="ms",
                utc=True,
            )

        # Seconds
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


def calculate_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> dict:

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                prediction,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                average="macro",
            )
        ),
    }


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

    print(
        "\nTraining CatBoost on train + validation..."
    )

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

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


def generate_gru_predictions():

    model = load_gru()

    records = []

    print(
        "\nGenerating GRU test predictions..."
    )

    for symbol_id, symbol in enumerate(SYMBOLS):

        X = np.load(
            SEQUENCE_TEST_DIR
            / f"{symbol}_X.npy"
        )

        valid_indices = np.load(
            SEQUENCE_TEST_DIR
            / f"{symbol}_valid_indices.npy"
        ).astype(np.int64)

        return_5m = np.load(
            SEQUENCE_TEST_DIR
            / f"{symbol}_return_5m.npy"
        )

        times = np.load(
            SEQUENCE_TEST_DIR
            / f"{symbol}_time.npy"
        )

        # Only sequence endpoints with at least
        # 60 observations.
        valid_mask = (
            valid_indices
            >= SEQUENCE_LENGTH - 1
        )

        valid_indices = (
            valid_indices[
                valid_mask
            ]
        )

        target_returns = (
            return_5m[
                valid_indices
            ]
        )

        endpoint_times = (
            times[
                valid_indices
            ]
        )

        # Match binary benchmark:
        # remove exact-zero future returns.
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

        symbol_df = pd.DataFrame(
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

        records.append(
            symbol_df
        )

        print(
            f"{symbol}: "
            f"{len(symbol_df):,} sequences"
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
    print("PER-COIN CATBOOST + GRU ENSEMBLE EVALUATION")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    feature_columns = (
        load_feature_columns()
    )

    print(
        "\nLoading train/validation/test data..."
    )

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    test_df = pd.read_parquet(
        TEST_FILE
    )

    # --------------------------------------------------------
    # Prepare CatBoost data
    # --------------------------------------------------------

    X_train, y_train, _ = (
        make_binary_dataset(
            train_df,
            feature_columns,
        )
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
        f"Full CatBoost training rows: "
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

    gru_df = (
        generate_gru_predictions()
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
            "return_5m",
            "catboost_probability_up",
            "catboost_prediction",
            "catboost_target",
        ]
    ].merge(
        gru_df,
        on=[
            "symbol",
            "test_time",
        ],
        how="inner",
    )

    print(
        f"\nMatched rows: "
        f"{len(merged):,}"
    )

    if len(merged) < 100_000:
        raise ValueError(
            "Too few rows matched between "
            "CatBoost and GRU."
        )

    # --------------------------------------------------------
    # Verify targets
    # --------------------------------------------------------

    target_agreement = np.mean(
        merged[
            "catboost_target"
        ].to_numpy()
        ==
        merged[
            "gru_target"
        ].to_numpy()
    )

    print(
        f"Target agreement: "
        f"{target_agreement:.6f}"
    )

    if target_agreement < 0.999:
        raise ValueError(
            "CatBoost and GRU targets do not agree."
        )

    # --------------------------------------------------------
    # Calculate predictions
    # --------------------------------------------------------

    y_true = (
        merged[
            "catboost_target"
        ].to_numpy()
        .astype(np.int64)
    )

    cb_probability = (
        merged[
            "catboost_probability_up"
        ].to_numpy()
    )

    gru_probability = (
        merged[
            "gru_probability_up"
        ].to_numpy()
    )

    cb_prediction = (
        cb_probability >= 0.5
    ).astype(np.int64)

    gru_prediction = (
        gru_probability >= 0.5
    ).astype(np.int64)

    ensemble_probability = (
        CATBOOST_WEIGHT
        * cb_probability
        +
        GRU_WEIGHT
        * gru_probability
    )

    ensemble_prediction = (
        ensemble_probability >= 0.5
    ).astype(np.int64)

    # Inverse previous-5m baseline.
    baseline_prediction = (
        merged[
            "return_5m"
        ].to_numpy()
        < 0
    ).astype(np.int64)

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall_models = {
        "CatBoost": cb_prediction,
        "GRU": gru_prediction,
        "Ensemble_55_45": ensemble_prediction,
        "Inverse_Previous_5m": baseline_prediction,
    }

    overall_results = []

    for model_name, prediction in (
        overall_models.items()
    ):

        metrics = calculate_metrics(
            y_true,
            prediction,
        )

        overall_results.append(
            {
                "scope": "overall",
                "symbol": "ALL",
                "model": model_name,
                "samples": len(y_true),
                "accuracy":
                    metrics["accuracy"],
                "macro_f1":
                    metrics["macro_f1"],
            }
        )

    # --------------------------------------------------------
    # Per-coin metrics
    # --------------------------------------------------------

    per_coin_results = []

    for symbol in SYMBOLS:

        coin_mask = (
            merged["symbol"]
            == symbol
        ).to_numpy()

        coin_y = y_true[
            coin_mask
        ]

        coin_predictions = {
            "CatBoost":
                cb_prediction[
                    coin_mask
                ],
            "GRU":
                gru_prediction[
                    coin_mask
                ],
            "Ensemble_55_45":
                ensemble_prediction[
                    coin_mask
                ],
            "Inverse_Previous_5m":
                baseline_prediction[
                    coin_mask
                ],
        }

        for model_name, prediction in (
            coin_predictions.items()
        ):

            metrics = calculate_metrics(
                coin_y,
                prediction,
            )

            per_coin_results.append(
                {
                    "scope": "per_coin",
                    "symbol": symbol,
                    "model": model_name,
                    "samples": len(coin_y),
                    "accuracy":
                        metrics["accuracy"],
                    "macro_f1":
                        metrics["macro_f1"],
                }
            )

    results_df = pd.DataFrame(
        overall_results
        +
        per_coin_results
    )

    print("\n" + "=" * 70)
    print("PER-COIN RESULTS")
    print("=" * 70)

    display_df = results_df[
        results_df["scope"]
        == "per_coin"
    ].copy()

    display_df[
        "accuracy"
    ] = display_df[
        "accuracy"
    ].map(
        lambda x: f"{x:.6f}"
    )

    display_df[
        "macro_f1"
    ] = display_df[
        "macro_f1"
    ].map(
        lambda x: f"{x:.6f}"
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    print("\nOverall:")
    print(
        results_df[
            results_df["scope"]
            == "overall"
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Best model per coin
    # --------------------------------------------------------

    best_by_coin = []

    for symbol in SYMBOLS:

        coin_df = results_df[
            (results_df["scope"] == "per_coin")
            &
            (results_df["symbol"] == symbol)
        ]

        best_row = coin_df.loc[
            coin_df["macro_f1"].idxmax()
        ]

        best_by_coin.append(
            {
                "symbol": symbol,
                "best_model":
                    best_row["model"],
                "macro_f1":
                    float(
                        best_row["macro_f1"]
                    ),
                "accuracy":
                    float(
                        best_row["accuracy"]
                    ),
            }
        )

    best_df = pd.DataFrame(
        best_by_coin
    )

    print("\nBest model by coin:")
    print(
        best_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        REPORTS_DIR
        / "binary_ensemble_per_coin_v1.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    best_path = (
        REPORTS_DIR
        / "binary_ensemble_best_model_per_coin_v1.csv"
    )

    best_df.to_csv(
        best_path,
        index=False,
    )

    metadata = {
        "catboost_weight":
            CATBOOST_WEIGHT,
        "gru_weight":
            GRU_WEIGHT,
        "matched_rows":
            int(len(merged)),
        "target_agreement":
            float(target_agreement),
        "note":
            "55/45 ensemble weight was selected on validation and frozen before test evaluation.",
        "test_usage_note":
            "Per-coin analysis is diagnostic and uses the same previously evaluated test period.",
    }

    metadata_path = (
        REPORTS_DIR
        / "binary_ensemble_per_coin_metadata_v1.json"
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
    print(best_path)
    print(metadata_path)


if __name__ == "__main__":
    main()