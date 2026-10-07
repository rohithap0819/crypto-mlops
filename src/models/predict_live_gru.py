from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    ROOT
    / "models"
    / "binary_sequence_v1"
    / "gru_binary_best.pt"
)

SEQUENCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "live_sequence_v1"
    / "latest_gru_sequence.npz"
)

LIVE_FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "live_features_latest.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "live_gru_predictions_v1.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "live_gru_predictions_v1.json"
)


# ============================================================
# IMPORT FROZEN MODEL ARCHITECTURE
# ============================================================

MODEL_DIR = (
    ROOT
    / "src"
    / "models"
)

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from binary_sequence_models import BinaryGRU  # noqa: E402


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

FEATURE_COUNT = 61
SEQUENCE_LENGTH = 60
NUM_SYMBOLS = 5
HIDDEN_SIZE = 64
DROPOUT = 0.2
SYMBOL_EMBEDDING_DIM = 4


# ============================================================
# MODEL LOADING
# ============================================================

def load_model() -> BinaryGRU:
    """Load the frozen BinaryGRU checkpoint."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"GRU checkpoint not found: {MODEL_FILE}"
        )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected checkpoint to be a dictionary."
        )

    required_keys = [
        "model_state_dict",
        "feature_count",
        "sequence_length",
        "hidden_size",
    ]

    missing = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if missing:
        raise ValueError(
            f"Checkpoint missing keys: {missing}"
        )

    if checkpoint["feature_count"] != FEATURE_COUNT:
        raise ValueError(
            "Checkpoint feature count mismatch: "
            f"{checkpoint['feature_count']} "
            f"!= {FEATURE_COUNT}"
        )

    if checkpoint["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError(
            "Checkpoint sequence length mismatch: "
            f"{checkpoint['sequence_length']} "
            f"!= {SEQUENCE_LENGTH}"
        )

    if checkpoint["hidden_size"] != HIDDEN_SIZE:
        raise ValueError(
            "Checkpoint hidden size mismatch: "
            f"{checkpoint['hidden_size']} "
            f"!= {HIDDEN_SIZE}"
        )

    model = BinaryGRU(
        input_size=FEATURE_COUNT,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        dropout=DROPOUT,
        num_symbols=NUM_SYMBOLS,
        symbol_embedding_dim=SYMBOL_EMBEDDING_DIM,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# LOAD LIVE INPUT
# ============================================================

def load_live_sequence() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Load the latest 60x61 sequence batch."""

    if not SEQUENCE_FILE.exists():
        raise FileNotFoundError(
            f"Live GRU sequence not found: "
            f"{SEQUENCE_FILE}"
        )

    data = np.load(
        SEQUENCE_FILE
    )

    if "X" not in data:
        raise ValueError(
            "Live sequence file does not contain X."
        )

    if "symbol_ids" not in data:
        raise ValueError(
            "Live sequence file does not contain symbol_ids."
        )

    X = data["X"]
    symbol_ids = data["symbol_ids"]

    expected_shape = (
        NUM_SYMBOLS,
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"Unexpected GRU input shape: "
            f"{X.shape}; "
            f"expected {expected_shape}"
        )

    if symbol_ids.shape != (NUM_SYMBOLS,):
        raise ValueError(
            f"Unexpected symbol_ids shape: "
            f"{symbol_ids.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "GRU input contains non-finite values."
        )

    return (
        X.astype(np.float32),
        symbol_ids.astype(np.int64),
    )


# ============================================================
# INFERENCE
# ============================================================

def predict(
    model: BinaryGRU,
    X: np.ndarray,
    symbol_ids: np.ndarray,
) -> pd.DataFrame:
    """Generate binary direction probabilities."""

    X_tensor = torch.from_numpy(
        X
    )

    symbol_tensor = torch.from_numpy(
        symbol_ids
    )

    with torch.no_grad():

        logits = model(
            X_tensor,
            symbol_tensor,
        )

        probabilities_up = torch.sigmoid(
            logits
        )

    probabilities_up = (
        probabilities_up
        .cpu()
        .numpy()
    )

    probabilities_down = (
        1.0
        - probabilities_up
    )

    predictions = np.where(
        probabilities_up >= 0.5,
        "UP",
        "DOWN",
    )

    confidence = np.maximum(
        probabilities_up,
        probabilities_down,
    )

    rows = []

    for index, symbol_id in enumerate(
        symbol_ids
    ):

        symbol = SYMBOLS[
            int(symbol_id)
        ]

        rows.append(
            {
                "symbol": symbol,
                "symbol_id": int(symbol_id),
                "prediction": predictions[index],
                "probability_down": float(
                    probabilities_down[index]
                ),
                "probability_up": float(
                    probabilities_up[index]
                ),
                "confidence": float(
                    confidence[index]
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CRYPTO MLOPS - LIVE GRU INFERENCE")
    print("=" * 70)

    print(
        f"Model: {MODEL_FILE}"
    )

    print(
        f"Input: {SEQUENCE_FILE}"
    )

    model = load_model()

    X, symbol_ids = (
        load_live_sequence()
    )

    predictions = predict(
        model=model,
        X=X,
        symbol_ids=symbol_ids,
    )

    if not LIVE_FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Live feature file not found: {LIVE_FEATURE_FILE}"
        )

    timestamp_df = pd.read_parquet(
        LIVE_FEATURE_FILE,
        columns=["symbol", "open_time"],
    )

    timestamp_df["open_time"] = (
        timestamp_df["open_time"].astype(str)
    )

    predictions = predictions.merge(
        timestamp_df,
        on="symbol",
        how="left",
        validate="one_to_one",
    )

    if predictions["open_time"].isna().any():
        raise ValueError(
            "Missing timestamp for one or more GRU predictions."
        )

    predictions = predictions[
        [
            "symbol",
            "symbol_id",
            "open_time",
            "prediction",
            "probability_down",
            "probability_up",
            "confidence",
        ]
    ].copy()

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    predictions.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    metadata = {
        "model": "BinaryGRU",
        "model_file": str(
            MODEL_FILE.relative_to(ROOT)
        ),
        "sequence_file": str(
            SEQUENCE_FILE.relative_to(ROOT)
        ),
        "feature_count": FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "forecast_horizon_minutes": 5,
        "rows": predictions.to_dict(
            orient="records"
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print(
        predictions.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("GRU INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"CSV : {OUTPUT_CSV}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()