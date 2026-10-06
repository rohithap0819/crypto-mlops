"""
Evaluate the best binary sequence GRU checkpoint on the held-out test set.

This evaluates the checkpoint trained during the binary sequence benchmark.
It does NOT retrain the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.models.binary_sequence_models import BinaryGRU


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_v1"
)

SEQUENCE_DIR = SEQUENCE_ROOT / "test"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "binary_sequence_v1"
    / "gru_binary_best.pt"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

SEQUENCE_LENGTH = 60
HIDDEN_SIZE = 64
DROPOUT = 0.2
EMBEDDING_DIM = 4

SYMBOLS = [
    "BNBUSDT",
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HELPERS
# ============================================================

def load_array(symbol: str, suffix: str) -> np.ndarray:
    """
    Load a test sequence-data array using the actual
    sequence_v1/test filename convention.
    """
    path = SEQUENCE_DIR / f"{symbol}_{suffix}.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find:\n{path}\n\n"
            f"Check the files inside:\n{SEQUENCE_DIR}"
        )

    return np.load(path)


def make_sequences(
    X: np.ndarray,
    valid_indices: np.ndarray,
) -> np.ndarray:
    """
    Build 60-step sequences ending at each valid endpoint.
    """
    sequences = []

    for end_index in valid_indices:
        start_index = end_index - SEQUENCE_LENGTH + 1

        if start_index < 0:
            continue

        sequences.append(X[start_index:end_index + 1])

    return np.asarray(sequences, dtype=np.float32)


def predict(
    model: torch.nn.Module,
    X_sequences: np.ndarray,
    symbol_ids: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()

    all_probabilities = []
    all_predictions = []

    with torch.no_grad():

        for start in range(0, len(X_sequences), batch_size):

            end = min(start + batch_size, len(X_sequences))

            X_batch = torch.tensor(
                X_sequences[start:end],
                dtype=torch.float32,
                device=DEVICE,
            )

            symbol_batch = torch.tensor(
                symbol_ids[start:end],
                dtype=torch.long,
                device=DEVICE,
            )

            logits = model(X_batch, symbol_batch)

            probabilities = torch.sigmoid(logits).squeeze(-1)

            predictions = (probabilities >= 0.5).long()

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

    return (
        np.concatenate(all_probabilities),
        np.concatenate(all_predictions),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BINARY SEQUENCE GRU — TEST EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"GRU checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata_path = SEQUENCE_ROOT / "metadata.json"

    if metadata_path.exists():

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("\nSequence metadata:")
        print(json.dumps(metadata, indent=2))

        feature_count = int(
            metadata.get("feature_count", 61)
        )

    else:
        feature_count = 61

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = BinaryGRU(
        input_size=feature_count,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        num_symbols=len(SYMBOLS),
        symbol_embedding_dim=EMBEDDING_DIM,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print("\nGRU checkpoint loaded successfully.")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # --------------------------------------------------------
    # Build test dataset
    # --------------------------------------------------------

    X_all = []
    y_all = []
    symbol_ids_all = []

    symbol_counts = {}

    print("\nLoading test sequences...")

    for symbol_id, symbol in enumerate(SYMBOLS):

        X = load_array(symbol, "X")
        valid_indices = load_array(symbol, "valid_indices")
        future_returns = load_array(symbol, "return_5m")

        valid_indices = valid_indices.astype(np.int64)

        # Keep only valid sequence endpoints.
        valid_mask = (
            valid_indices >= SEQUENCE_LENGTH - 1
        )

        valid_indices = valid_indices[valid_mask]

        target_returns = future_returns[valid_indices]

        # Remove exact zero targets to match binary benchmark.
        nonzero_mask = target_returns != 0

        valid_indices = valid_indices[nonzero_mask]
        target_returns = target_returns[nonzero_mask]

        sequences = make_sequences(
            X,
            valid_indices,
        )

        # Binary target:
        # DOWN = 0
        # UP   = 1
        y = (target_returns > 0).astype(np.int64)

        # Make sure lengths match.
        n = min(
            len(sequences),
            len(y),
        )

        sequences = sequences[:n]
        y = y[:n]

        symbol_ids = np.full(
            n,
            symbol_id,
            dtype=np.int64,
        )

        X_all.append(sequences)
        y_all.append(y)
        symbol_ids_all.append(symbol_ids)

        symbol_counts[symbol] = n

        print(
            f"{symbol}: {n:,} test sequences"
        )

    X_test = np.concatenate(X_all, axis=0)
    y_test = np.concatenate(y_all, axis=0)
    symbol_ids_test = np.concatenate(symbol_ids_all, axis=0)

    print("\nTotal test sequences:")
    print(f"{len(X_test):,}")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities, predictions = predict(
        model,
        X_test,
        symbol_ids_test,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy : {accuracy:.6f}")
    print(f"Macro-F1 : {macro_f1:.6f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["DOWN", "UP"],
            digits=6,
        )
    )

    print("Confusion matrix:")

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print(cm)

    # --------------------------------------------------------
    # Inverse previous-5m baseline
    #
    # This requires the previous return feature.
    # It is evaluated separately when the required array exists.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = {
        "model": "BinaryGRU",
        "test_sequences": int(len(X_test)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "checkpoint": str(CHECKPOINT_PATH),
        "training_note": (
            "Checkpoint was trained on the 500k-sample binary "
            "sequence benchmark, not the full train+validation set."
        ),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": feature_count,
        "symbols": SYMBOLS,
        "symbol_counts": symbol_counts,
    }

    output_path = (
        REPORTS_DIR
        / "binary_sequence_gru_test_v1.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            comparison,
            f,
            indent=2,
        )

    np.savetxt(
        REPORTS_DIR / "binary_sequence_gru_test_confusion_v1.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    main()