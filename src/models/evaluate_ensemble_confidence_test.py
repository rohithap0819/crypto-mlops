"""
Final comparative evaluation of the frozen confidence strategy.

Frozen from validation:
    CatBoost weight = 0.55
    GRU weight      = 0.45
    Confidence threshold = 0.55

No test-set tuning.
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

CONFIDENCE_THRESHOLD = 0.55

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

        median_value = np.nanmedian(
            series.astype(float)
        )

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


# ============================================================
# CATBOOST
# ============================================================

def train_catboost(
    X_train,
    y_train,
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
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def generate_gru_test_predictions():

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

        with torch.no_grad():

            for start in range(
                0,
                len(sequences),
                256,
            ):

                end = min(
                    start + 256,
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

                probabilities.append(
                    torch.sigmoid(
                        logits
                    ).cpu().numpy()
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
    print("FROZEN 0.55 CONFIDENCE ENSEMBLE TEST EVALUATION")
    print("=" * 70)

    feature_columns = (
        load_feature_columns()
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

    catboost = train_catboost(
        X_full_train,
        y_full_train,
    )

    catboost_probability = (
        catboost.predict_proba(
            X_test
        )[:, 1]
    )

    gru_df = (
        generate_gru_test_predictions()
    )

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
        "target"
    ] = y_test.to_numpy()

    merged = test_clean[
        [
            "symbol",
            "test_time",
            "return_5m",
            "catboost_probability_up",
            "target",
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
        f"\nMatched rows: {len(merged):,}"
    )

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    merged[
        "ensemble_probability"
    ] = (
        CATBOOST_WEIGHT
        * merged[
            "catboost_probability_up"
        ]
        +
        GRU_WEIGHT
        * merged[
            "gru_probability_up"
        ]
    )

    merged[
        "ensemble_prediction"
    ] = (
        merged[
            "ensemble_probability"
        ]
        >= 0.50
    ).astype(np.int64)

    merged[
        "confidence"
    ] = np.maximum(
        merged[
            "ensemble_probability"
        ],
        1.0
        -
        merged[
            "ensemble_probability"
        ],
    )

    # --------------------------------------------------------
    # Full ensemble
    # --------------------------------------------------------

    y_true = (
        merged["target"]
        .to_numpy()
        .astype(np.int64)
    )

    full_prediction = (
        merged[
            "ensemble_prediction"
        ]
        .to_numpy()
    )

    full_accuracy = accuracy_score(
        y_true,
        full_prediction,
    )

    full_f1 = f1_score(
        y_true,
        full_prediction,
        average="macro",
    )

    # --------------------------------------------------------
    # Frozen confidence rule
    # --------------------------------------------------------

    selected_mask = (
        merged["confidence"]
        >= CONFIDENCE_THRESHOLD
    )

    selected = merged.loc[
        selected_mask
    ]

    selected_y = (
        selected["target"]
        .to_numpy()
        .astype(np.int64)
    )

    selected_prediction = (
        selected[
            "ensemble_prediction"
        ]
        .to_numpy()
        .astype(np.int64)
    )

    filtered_accuracy = accuracy_score(
        selected_y,
        selected_prediction,
    )

    filtered_f1 = f1_score(
        selected_y,
        selected_prediction,
        average="macro",
    )

    coverage = (
        len(selected)
        / len(merged)
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_prediction = (
        merged["return_5m"]
        .to_numpy()
        < 0
    ).astype(np.int64)

    baseline_accuracy = accuracy_score(
        y_true,
        baseline_prediction,
    )

    baseline_f1 = f1_score(
        y_true,
        baseline_prediction,
        average="macro",
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Full ensemble accuracy : "
        f"{full_accuracy:.6f}"
    )

    print(
        f"Full ensemble Macro-F1 : "
        f"{full_f1:.6f}"
    )

    print(
        f"\nConfidence threshold   : "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"Filtered samples       : "
        f"{len(selected):,}"
    )

    print(
        f"Coverage               : "
        f"{coverage * 100:.2f}%"
    )

    print(
        f"Filtered accuracy      : "
        f"{filtered_accuracy:.6f}"
    )

    print(
        f"Filtered Macro-F1      : "
        f"{filtered_f1:.6f}"
    )

    print(
        f"Accuracy change        : "
        f"{(filtered_accuracy - full_accuracy) * 100:+.2f} pp"
    )

    print(
        f"\nInverse-5m accuracy    : "
        f"{baseline_accuracy:.6f}"
    )

    print(
        f"Inverse-5m Macro-F1    : "
        f"{baseline_f1:.6f}"
    )

    print("\nFiltered classification report:")

    print(
        classification_report(
            selected_y,
            selected_prediction,
            target_names=[
                "DOWN",
                "UP",
            ],
            digits=6,
        )
    )

    print(
        "Filtered confusion matrix:"
    )

    print(
        confusion_matrix(
            selected_y,
            selected_prediction,
        )
    )

    # --------------------------------------------------------
    # Per-coin confidence performance
    # --------------------------------------------------------

    per_coin = []

    for symbol in SYMBOLS:

        coin = selected[
            selected["symbol"]
            == symbol
        ]

        if len(coin) == 0:
            continue

        coin_y = (
            coin["target"]
            .to_numpy()
        )

        coin_prediction = (
            coin[
                "ensemble_prediction"
            ]
            .to_numpy()
        )

        per_coin.append(
            {
                "symbol": symbol,
                "samples": len(coin),
                "coverage_relative_to_matched_test":
                    len(coin)
                    /
                    len(
                        merged[
                            merged["symbol"]
                            == symbol
                        ]
                    ),
                "accuracy":
                    accuracy_score(
                        coin_y,
                        coin_prediction,
                    ),
                "macro_f1":
                    f1_score(
                        coin_y,
                        coin_prediction,
                        average="macro",
                    ),
            }
        )

    per_coin_df = pd.DataFrame(
        per_coin
    )

    print(
        "\nPer-coin filtered results:"
    )

    print(
        per_coin_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "catboost_weight":
            CATBOOST_WEIGHT,
        "gru_weight":
            GRU_WEIGHT,
        "confidence_threshold":
            CONFIDENCE_THRESHOLD,
        "matched_rows":
            int(len(merged)),
        "filtered_rows":
            int(len(selected)),
        "coverage":
            float(coverage),
        "full_ensemble_accuracy":
            float(full_accuracy),
        "full_ensemble_macro_f1":
            float(full_f1),
        "filtered_accuracy":
            float(filtered_accuracy),
        "filtered_macro_f1":
            float(filtered_f1),
        "accuracy_gain_vs_full":
            float(
                filtered_accuracy
                - full_accuracy
            ),
        "baseline_accuracy":
            float(baseline_accuracy),
        "baseline_macro_f1":
            float(baseline_f1),
        "note":
            "The 0.55 confidence threshold was selected on validation and frozen before this test evaluation.",
    }

    results_path = (
        REPORTS_DIR
        / "binary_ensemble_confidence_test_v1.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    per_coin_path = (
        REPORTS_DIR
        / "binary_ensemble_confidence_test_per_coin_v1.csv"
    )

    per_coin_df.to_csv(
        per_coin_path,
        index=False,
    )

    print("\nReports saved:")
    print(results_path)
    print(per_coin_path)


if __name__ == "__main__":
    main()