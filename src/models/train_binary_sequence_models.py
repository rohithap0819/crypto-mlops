"""
Train binary GRU/LSTM/CNN-LSTM models for 5-minute crypto direction.

Target:
    future_return_5m > 0 -> UP
    future_return_5m < 0 -> DOWN

Exact-zero future returns are excluded.

Experiment:
    500,000 training sequences
    100,000 validation sequences
    15 maximum epochs
    early stopping patience 3

The sequence X arrays are stored as:
    rows x 61 features

Sequences are constructed exactly like the existing
sequence training pipeline using valid_indices.
"""

from pathlib import Path
import copy
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from src.models.binary_sequence_models import (
    BinaryGRU,
    BinaryLSTM,
    BinaryCNNLSTM,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_v1"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "binary_sequence_v1"
)

MLFLOW_DB = (
    PROJECT_ROOT
    / "mlflow.db"
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

SYMBOL_TO_ID = {
    symbol: index
    for index, symbol in enumerate(
        SYMBOLS
    )
}

SEQUENCE_LENGTH = 60
FEATURE_COUNT = 61

TRAIN_SEQUENCES_PER_SYMBOL = 100_000
VALIDATION_SEQUENCES_PER_SYMBOL = 20_000

BATCH_SIZE = 128
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

HIDDEN_SIZE = 64
DROPOUT = 0.2

RANDOM_STATE = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed,
):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# EVEN SAMPLE SELECTION
# ============================================================

def select_evenly(
    indices,
    limit,
):
    """
    Select evenly spaced indices.

    This preserves coverage across the complete
    available split instead of selecting only from
    one local region.
    """

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    if len(indices) <= limit:
        return indices

    positions = np.linspace(
        0,
        len(indices) - 1,
        num=limit,
        dtype=np.int64,
    )

    return indices[
        positions
    ]


# ============================================================
# DATASET
# ============================================================

class BinaryCryptoSequenceDataset(
    Dataset
):
    """
    Sequence dataset for binary direction.

    Only non-zero future returns are eligible.
    """

    def __init__(
        self,
        split,
        sequences_per_symbol,
    ):
        self.split = split

        self.samples = []

        self.feature_arrays = {}
        self.return_arrays = {}

        for symbol in SYMBOLS:

            symbol_dir = (
                SEQUENCE_DIR
                / split
            )

            X_file = (
                symbol_dir
                / f"{symbol}_X.npy"
            )

            return_file = (
                symbol_dir
                / f"{symbol}_return_5m.npy"
            )

            valid_file = (
                symbol_dir
                / f"{symbol}_valid_indices.npy"
            )

            if not X_file.exists():
                raise FileNotFoundError(
                    f"Missing file:\n"
                    f"{X_file}"
                )

            X = np.load(
                X_file,
                mmap_mode="r",
            )

            returns = np.load(
                return_file,
                mmap_mode="r",
            )

            valid_indices = np.load(
                valid_file
            )

            # Remove exact-zero targets.
            valid_indices = (
                valid_indices[
                    returns[
                        valid_indices
                    ] != 0
                ]
            )

            selected_indices = (
                select_evenly(
                    valid_indices,
                    sequences_per_symbol,
                )
            )

            symbol_id = (
                SYMBOL_TO_ID[
                    symbol
                ]
            )

            for index in selected_indices:

                self.samples.append(
                    (
                        symbol_id,
                        int(index),
                    )
                )

            self.feature_arrays[
                symbol_id
            ] = X

            self.return_arrays[
                symbol_id
            ] = returns

        # Deterministically shuffle training samples.
        if split == "train":

            rng = np.random.default_rng(
                RANDOM_STATE
            )

            rng.shuffle(
                self.samples
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        symbol_id, end_index = (
            self.samples[
                index
            ]
        )

        X = self.feature_arrays[
            symbol_id
        ]

        returns = self.return_arrays[
            symbol_id
        ]

        start_index = (
            end_index
            - SEQUENCE_LENGTH
            + 1
        )

        sequence = X[
            start_index:
            end_index + 1
        ]

        if (
            sequence.shape
            != (
                SEQUENCE_LENGTH,
                FEATURE_COUNT,
            )
        ):

            raise ValueError(
                "Unexpected sequence shape: "
                f"{sequence.shape}"
            )

        # Copy because mmap-backed arrays can be non-writable.
        sequence_tensor = torch.from_numpy(
            np.array(
                sequence,
                dtype=np.float32,
                copy=True,
            )
        )

        symbol_tensor = torch.tensor(
            symbol_id,
            dtype=torch.long,
        )

        target = float(
            returns[
                end_index
            ]
        )

        target_tensor = torch.tensor(
            1.0 if target > 0 else 0.0,
            dtype=torch.float32,
        )

        return (
            sequence_tensor,
            symbol_tensor,
            target_tensor,
        )


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    model_name,
):
    if model_name == "GRU":

        return BinaryGRU(
            input_size=FEATURE_COUNT,
            hidden_size=HIDDEN_SIZE,
            dropout=DROPOUT,
            num_symbols=len(
                SYMBOLS
            ),
        )

    if model_name == "LSTM":

        return BinaryLSTM(
            input_size=FEATURE_COUNT,
            hidden_size=HIDDEN_SIZE,
            dropout=DROPOUT,
            num_symbols=len(
                SYMBOLS
            ),
        )

    if model_name == "CNNLSTM":

        return BinaryCNNLSTM(
            input_size=FEATURE_COUNT,
            cnn_channels=64,
            hidden_size=HIDDEN_SIZE,
            dropout=DROPOUT,
            num_symbols=len(
                SYMBOLS
            ),
        )

    raise ValueError(
        f"Unknown model: "
        f"{model_name}"
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for (
        sequence,
        symbol_id,
        target,
    ) in loader:

        sequence = (
            sequence.to(device)
        )

        symbol_id = (
            symbol_id.to(device)
        )

        target = (
            target.to(device)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            sequence,
            symbol_id,
        )

        loss = criterion(
            logits,
            target,
        )

        loss.backward()

        optimizer.step()

        batch_size = (
            target.shape[0]
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    return (
        total_loss
        / total_samples
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_predictions = []

    for (
        sequence,
        symbol_id,
        target,
    ) in loader:

        sequence = (
            sequence.to(device)
        )

        symbol_id = (
            symbol_id.to(device)
        )

        target = (
            target.to(device)
        )

        logits = model(
            sequence,
            symbol_id,
        )

        loss = criterion(
            logits,
            target,
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).long()

        batch_size = (
            target.shape[0]
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        all_targets.append(
            target.cpu().numpy()
        )

        all_predictions.append(
            predictions.cpu().numpy()
        )

    validation_loss = (
        total_loss
        / total_samples
    )

    targets = np.concatenate(
        all_targets
    ).astype(
        np.int64
    )

    predictions = np.concatenate(
        all_predictions
    ).astype(
        np.int64
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=[
                0,
                1,
            ],
            zero_division=0,
        )
    )

    return {
        "loss": float(
            validation_loss
        ),
        "macro_f1": float(
            macro_f1
        ),
        "accuracy": float(
            accuracy
        ),
        "down_precision": float(
            precision[0]
        ),
        "down_recall": float(
            recall[0]
        ),
        "down_f1": float(
            f1[0]
        ),
        "up_precision": float(
            precision[1]
        ),
        "up_recall": float(
            recall[1]
        ),
        "up_f1": float(
            f1[1]
        ),
        "down_support": int(
            support[0]
        ),
        "up_support": int(
            support[1]
        ),
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def train_model(
    model_name,
):

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(
        RANDOM_STATE
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print(
        f"BINARY SEQUENCE MODEL: "
        f"{model_name}"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print()
    print(
        "Loading training sequences..."
    )

    train_dataset = (
        BinaryCryptoSequenceDataset(
            split="train",
            sequences_per_symbol=(
                TRAIN_SEQUENCES_PER_SYMBOL
            ),
        )
    )

    print(
        f"Training sequences: "
        f"{len(train_dataset):,}"
    )

    print()
    print(
        "Loading validation sequences..."
    )

    validation_dataset = (
        BinaryCryptoSequenceDataset(
            split="validation",
            sequences_per_symbol=(
                VALIDATION_SEQUENCES_PER_SYMBOL
            ),
        )
    )

    print(
        f"Validation sequences: "
        f"{len(validation_dataset):,}"
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    # Read sampled target values from the dataset
    # for balanced BCE weighting.

    train_targets = np.array(
        [
            1 if (
                train_dataset
                .return_arrays[
                    symbol_id
                ][end_index]
                > 0
            )
            else 0
            for symbol_id, end_index
            in train_dataset.samples
        ],
        dtype=np.int64,
    )

    negative_count = np.sum(
        train_targets == 0
    )

    positive_count = np.sum(
        train_targets == 1
    )

    pos_weight_value = (
        negative_count
        / positive_count
    )

    print()
    print(
        f"Training DOWN: "
        f"{negative_count:,}"
    )

    print(
        f"Training UP: "
        f"{positive_count:,}"
    )

    print(
        f"Positive class weight: "
        f"{pos_weight_value:.6f}"
    )

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model(
        model_name
    ).to(device)

    print()
    print(
        f"Trainable parameters: "
        f"{sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ):,}"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight_value,
                dtype=torch.float32,
                device=device,
            )
        )
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # MLFLOW
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB}"
    )

    mlflow.set_experiment(
        "CryptoML_Binary_Sequence"
    )

    history = []

    best_validation_loss = float(
        "inf"
    )

    best_state = None
    best_epoch = 0

    epochs_without_improvement = 0

    with mlflow.start_run(
        run_name=(
            f"Binary_{model_name}_5m"
        )
    ) as run:

        mlflow.log_param(
            "model",
            model_name,
        )

        mlflow.log_param(
            "task",
            "binary_5m_direction",
        )

        mlflow.log_param(
            "sequence_length",
            SEQUENCE_LENGTH,
        )

        mlflow.log_param(
            "feature_count",
            FEATURE_COUNT,
        )

        mlflow.log_param(
            "train_sequences",
            len(train_dataset),
        )

        mlflow.log_param(
            "validation_sequences",
            len(validation_dataset),
        )

        mlflow.log_param(
            "batch_size",
            BATCH_SIZE,
        )

        mlflow.log_param(
            "epochs",
            EPOCHS,
        )

        mlflow.log_param(
            "early_stopping_patience",
            EARLY_STOPPING_PATIENCE,
        )

        for epoch in range(
            1,
            EPOCHS + 1,
        ):

            train_loss = (
                train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                )
            )

            metrics = evaluate(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **metrics,
                }
            )

            print()
            print(
                f"Epoch "
                f"{epoch}/{EPOCHS}"
            )

            print(
                f"Train loss: "
                f"{train_loss:.6f}"
            )

            print(
                f"Val loss: "
                f"{metrics['loss']:.6f}"
            )

            print(
                f"Val Macro-F1: "
                f"{metrics['macro_f1']:.6f}"
            )

            print(
                f"Val Accuracy: "
                f"{metrics['accuracy']:.6f}"
            )

            print(
                f"DOWN F1: "
                f"{metrics['down_f1']:.6f}"
            )

            print(
                f"UP F1: "
                f"{metrics['up_f1']:.6f}"
            )

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "validation_loss",
                metrics["loss"],
                step=epoch,
            )

            mlflow.log_metric(
                "validation_macro_f1",
                metrics["macro_f1"],
                step=epoch,
            )

            mlflow.log_metric(
                "validation_accuracy",
                metrics["accuracy"],
                step=epoch,
            )

            mlflow.log_metric(
                "validation_down_f1",
                metrics["down_f1"],
                step=epoch,
            )

            mlflow.log_metric(
                "validation_up_f1",
                metrics["up_f1"],
                step=epoch,
            )

            # ------------------------------------------------
            # BEST CHECKPOINT
            # ------------------------------------------------

            if (
                metrics["loss"]
                < best_validation_loss
            ):

                best_validation_loss = (
                    metrics["loss"]
                )

                best_state = (
                    copy.deepcopy(
                        model.state_dict()
                    )
                )

                best_epoch = (
                    epoch
                )

                epochs_without_improvement = (
                    0
                )

            else:

                epochs_without_improvement += (
                    1
                )

                if (
                    epochs_without_improvement
                    >= EARLY_STOPPING_PATIENCE
                ):

                    print()
                    print(
                        "Early stopping triggered."
                    )

                    break

        # ----------------------------------------------------
        # RESTORE BEST MODEL
        # ----------------------------------------------------

        if best_state is None:
            raise RuntimeError(
                "No best model state was saved."
            )

        model.load_state_dict(
            best_state
        )

        best_metrics = min(
            history,
            key=lambda row: row[
                "loss"
            ],
        )

        checkpoint_path = (
            MODELS_DIR
            / f"{model_name.lower()}_binary_best.pt"
        )

        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": (
                    model.state_dict()
                ),
                "feature_count": (
                    FEATURE_COUNT
                ),
                "sequence_length": (
                    SEQUENCE_LENGTH
                ),
                "hidden_size": (
                    HIDDEN_SIZE
                ),
                "dropout": (
                    DROPOUT
                ),
                "best_epoch": (
                    best_epoch
                ),
                "best_validation_loss": (
                    best_validation_loss
                ),
                "best_validation_macro_f1": (
                    best_metrics[
                        "macro_f1"
                    ]
                ),
                "best_validation_accuracy": (
                    best_metrics[
                        "accuracy"
                    ]
                ),
            },
            checkpoint_path,
        )

        history_path = (
            REPORTS_DIR
            / f"binary_sequence_{model_name.lower()}_history_v1.csv"
        )

        import pandas as pd

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        mlflow.log_artifact(
            str(checkpoint_path)
        )

        mlflow.log_artifact(
            str(history_path)
        )

        mlflow.log_metric(
            "best_validation_loss",
            best_validation_loss,
        )

        mlflow.log_metric(
            "best_validation_macro_f1",
            best_metrics[
                "macro_f1"
            ],
        )

        mlflow.log_metric(
            "best_validation_accuracy",
            best_metrics[
                "accuracy"
            ],
        )

        print()
        print(
            "BEST CHECKPOINT"
        )

        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Best validation loss: "
            f"{best_validation_loss:.6f}"
        )

        print(
            f"Best validation Macro-F1: "
            f"{best_metrics['macro_f1']:.6f}"
        )

        print(
            f"Best validation Accuracy: "
            f"{best_metrics['accuracy']:.6f}"
        )

        print()
        print(
            f"Checkpoint saved to: "
            f"{checkpoint_path}"
        )

        print(
            f"History saved to: "
            f"{history_path}"
        )

        print()
        print(
            f"MLflow run ID: "
            f"{run.info.run_id}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "GRU",
            "LSTM",
            "CNNLSTM",
        ],
        required=True,
    )

    args = parser.parse_args()

    train_model(
        args.model
    )