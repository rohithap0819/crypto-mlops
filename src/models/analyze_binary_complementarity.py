"""
Binary CatBoost + GRU complementarity analysis.

Purpose:
    Determine whether the GRU makes sufficiently different predictions
    from CatBoost to justify an ensemble.

IMPORTANT:
    - Validation set only.
    - Test set is NOT used.
    - CatBoost configuration matches train_binary_baselines.py.
    - GRU uses the existing best binary GRU checkpoint.
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
    confusion_matrix,
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

SEQUENCE_VALIDATION_DIR = (
    SEQUENCE_ROOT / "validation"
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FEATURE LOADING
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

    X = pd.concat(
        [
            X.reset_index(drop=True),
            symbol_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    return X


# ============================================================
# BINARY TARGET
# ============================================================

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

    print("\nTraining CatBoost...")

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# TIME COLUMN DETECTION
# ============================================================

def detect_time_column(
    df: pd.DataFrame,
) -> str:

    candidates = [
        "timestamp",
        "time",
        "open_time",
        "datetime",
        "date",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for column in candidates:

        if column in lower_map:
            return lower_map[column]

    time_like = [
        column
        for column in df.columns
        if "time" in str(column).lower()
        or "timestamp" in str(column).lower()
    ]

    if len(time_like) == 1:
        return time_like[0]

    raise ValueError(
        "Could not uniquely identify a time column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def normalize_time(values) -> pd.Series:
    """
    Normalize timestamps to timezone-aware UTC timestamps.

    Handles:
    - pandas timezone-aware datetime
    - pandas timezone-naive datetime
    - Unix timestamps in microseconds
    - Unix timestamps in milliseconds
    - Unix timestamps in seconds
    - string timestamps
    """
    series = pd.Series(values)

    # Pandas datetime, including timezone-aware datetime64[*, UTC].
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(
            series,
            utc=True,
        )

    # Numeric Unix timestamps.
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

    # Strings / object timestamps.
    return pd.to_datetime(
        series,
        utc=True,
    )


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


def build_gru_validation_predictions():

    model = load_gru()

    records = []

    print("\nGenerating GRU validation predictions...")

    for symbol_id, symbol in enumerate(SYMBOLS):

        X_path = (
            SEQUENCE_VALIDATION_DIR
            / f"{symbol}_X.npy"
        )

        valid_indices_path = (
            SEQUENCE_VALIDATION_DIR
            / f"{symbol}_valid_indices.npy"
        )

        return_path = (
            SEQUENCE_VALIDATION_DIR
            / f"{symbol}_return_5m.npy"
        )

        time_path = (
            SEQUENCE_VALIDATION_DIR
            / f"{symbol}_time.npy"
        )

        for path in [
            X_path,
            valid_indices_path,
            return_path,
            time_path,
        ]:

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing file:\n{path}"
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

        # Build sequences.
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
                "sequence_time": normalize_time(
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

        records.append(symbol_df)

        print(
            f"{symbol}: "
            f"{len(symbol_df):,} validation rows"
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


# ============================================================
# ENSEMBLE ANALYSIS
# ============================================================

def evaluate_predictions(
    y_true: np.ndarray,
    predictions: np.ndarray,
):

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
            )
        ),
    }


def main():

    print("=" * 70)
    print("CATBOOST + GRU COMPLEMENTARITY ANALYSIS")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    feature_columns = (
        load_feature_columns()
    )

    # --------------------------------------------------------
    # Load tabular train / validation data
    # --------------------------------------------------------

    print("\nLoading CatBoost train/validation data...")

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    # --------------------------------------------------------
    # Prepare binary datasets
    # --------------------------------------------------------

    X_train, y_train, train_clean = (
        make_binary_dataset(
            train_df,
            feature_columns,
        )
    )

    X_validation, y_validation, validation_clean = (
        make_binary_dataset(
            validation_df,
            feature_columns,
        )
    )

    print(
        f"\nBinary training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Binary validation rows: "
        f"{len(X_validation):,}"
    )

    # --------------------------------------------------------
    # Train CatBoost
    # --------------------------------------------------------

    catboost_model = train_catboost(
        X_train,
        y_train,
    )

    catboost_probability = (
        catboost_model.predict_proba(
            X_validation
        )[:, 1]
    )

    catboost_prediction = (
        catboost_probability >= 0.5
    ).astype(np.int64)

    # --------------------------------------------------------
    # GRU predictions
    # --------------------------------------------------------

    gru_predictions = (
        build_gru_validation_predictions()
    )

    # --------------------------------------------------------
    # Identify CatBoost time column
    # --------------------------------------------------------

    time_column = detect_time_column(
        validation_clean
    )

    validation_clean = validation_clean.copy()

    validation_clean[
        "validation_time"
    ] = normalize_time(
        validation_clean[
            time_column
        ]
    )

    # Add CatBoost outputs.
    validation_clean[
        "catboost_probability_up"
    ] = catboost_probability

    validation_clean[
        "catboost_prediction"
    ] = catboost_prediction

    validation_clean[
        "catboost_target"
    ] = y_validation.to_numpy()

    # --------------------------------------------------------
    # Merge CatBoost and GRU
    # --------------------------------------------------------

    merged = validation_clean[
        [
            "symbol",
            "validation_time",
            "catboost_probability_up",
            "catboost_prediction",
            "catboost_target",
        ]
    ].merge(
        gru_predictions,
        left_on=[
            "symbol",
            "validation_time",
        ],
        right_on=[
            "symbol",
            "sequence_time",
        ],
        how="inner",
    )

    print(
        "\nMatched validation rows: "
        f"{len(merged):,}"
    )

    if len(merged) < 100_000:
        raise ValueError(
            "Too few CatBoost/GRU rows matched.\n"
            f"Matched rows: {len(merged):,}\n"
            "Check the validation time-column alignment."
        )

    # --------------------------------------------------------
    # Validate targets agree
    # --------------------------------------------------------

    target_match_rate = np.mean(
        merged["catboost_target"].to_numpy()
        ==
        merged["gru_target"].to_numpy()
    )

    print(
        "Target agreement: "
        f"{target_match_rate:.6f}"
    )

    if target_match_rate < 0.999:
        raise ValueError(
            "CatBoost and GRU targets do not align."
        )

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

    # --------------------------------------------------------
    # Individual performance
    # --------------------------------------------------------

    cb_metrics = evaluate_predictions(
        y_true,
        cb_pred,
    )

    gru_metrics = evaluate_predictions(
        y_true,
        gru_pred,
    )

    print("\nIndividual validation performance:")

    print(
        f"CatBoost Accuracy : "
        f"{cb_metrics['accuracy']:.6f}"
    )

    print(
        f"CatBoost Macro-F1 : "
        f"{cb_metrics['macro_f1']:.6f}"
    )

    print(
        f"GRU Accuracy      : "
        f"{gru_metrics['accuracy']:.6f}"
    )

    print(
        f"GRU Macro-F1      : "
        f"{gru_metrics['macro_f1']:.6f}"
    )

    # --------------------------------------------------------
    # Error complementarity
    # --------------------------------------------------------

    cb_correct = (
        cb_pred == y_true
    )

    gru_correct = (
        gru_pred == y_true
    )

    both_correct = (
        cb_correct & gru_correct
    )

    cb_only = (
        cb_correct & ~gru_correct
    )

    gru_only = (
        ~cb_correct & gru_correct
    )

    both_wrong = (
        ~cb_correct & ~gru_correct
    )

    complementarity = {
        "both_correct": int(
            both_correct.sum()
        ),
        "catboost_only_correct": int(
            cb_only.sum()
        ),
        "gru_only_correct": int(
            gru_only.sum()
        ),
        "both_wrong": int(
            both_wrong.sum()
        ),
        "disagreement_count": int(
            (cb_pred != gru_pred).sum()
        ),
        "disagreement_rate": float(
            np.mean(
                cb_pred != gru_pred
            )
        ),
    }

    print("\nError complementarity:")

    print(
        f"Both correct       : "
        f"{complementarity['both_correct']:,}"
    )

    print(
        f"CatBoost only      : "
        f"{complementarity['catboost_only_correct']:,}"
    )

    print(
        f"GRU only           : "
        f"{complementarity['gru_only_correct']:,}"
    )

    print(
        f"Both wrong         : "
        f"{complementarity['both_wrong']:,}"
    )

    print(
        f"Disagreement rate  : "
        f"{complementarity['disagreement_rate']:.6f}"
    )

    # --------------------------------------------------------
    # Probability correlation
    # --------------------------------------------------------

    probability_correlation = float(
        np.corrcoef(
            cb_prob,
            gru_prob,
        )[0, 1]
    )

    print(
        "\nProbability correlation: "
        f"{probability_correlation:.6f}"
    )

    # --------------------------------------------------------
    # Blend sweep
    # --------------------------------------------------------

    blend_results = []

    for catboost_weight in np.arange(
        0.0,
        1.01,
        0.05,
    ):

        gru_weight = (
            1.0 - catboost_weight
        )

        blended_probability = (
            catboost_weight * cb_prob
            +
            gru_weight * gru_prob
        )

        blended_prediction = (
            blended_probability >= 0.5
        ).astype(np.int64)

        metrics = evaluate_predictions(
            y_true,
            blended_prediction,
        )

        blend_results.append(
            {
                "catboost_weight":
                    round(
                        float(
                            catboost_weight
                        ),
                        2,
                    ),
                "gru_weight":
                    round(
                        float(
                            gru_weight
                        ),
                        2,
                    ),
                "accuracy":
                    metrics["accuracy"],
                "macro_f1":
                    metrics["macro_f1"],
            }
        )

    blend_df = pd.DataFrame(
        blend_results
    )

    best_blend = blend_df.iloc[
        blend_df["macro_f1"].idxmax()
    ]

    print("\nBest blend:")
    print(
        best_blend.to_string()
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_output = (
        REPORTS_DIR
        / "binary_catboost_gru_validation_predictions_v1.parquet"
    )

    merged.to_parquet(
        merged_output,
        index=False,
    )

    blend_output = (
        REPORTS_DIR
        / "binary_catboost_gru_blend_v1.csv"
    )

    blend_df.to_csv(
        blend_output,
        index=False,
    )

    confusion_output = (
        REPORTS_DIR
        / "binary_catboost_gru_complementarity_v1.json"
    )

    summary = {
        "validation_rows":
            int(len(merged)),
        "target_agreement":
            float(target_match_rate),
        "catboost":
            cb_metrics,
        "gru":
            gru_metrics,
        "complementarity":
            complementarity,
        "probability_correlation":
            probability_correlation,
        "best_blend":
            {
                key: (
                    float(value)
                    if isinstance(
                        value,
                        (np.floating, float),
                    )
                    else int(value)
                    if isinstance(
                        value,
                        (np.integer, int),
                    )
                    else value
                )
                for key, value
                in best_blend.to_dict().items()
            },
        "time_column":
            time_column,
    }

    with open(
        confusion_output,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\nReports saved:"
    )

    print(
        merged_output
    )

    print(
        blend_output
    )

    print(
        confusion_output
    )


if __name__ == "__main__":
    main()