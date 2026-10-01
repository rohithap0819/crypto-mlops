import argparse
import copy
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.sequence_models import (
    MultiTaskCNNLSTM,
    MultiTaskGRU,
    MultiTaskLSTM,
)


# ============================================================
# PATHS
# ============================================================

SEQUENCE_DIR = Path(
    "data/processed/sequence_v1"
)

REPORTS_DIR = Path("reports")

RESULTS_FILE = (
    REPORTS_DIR / "sequence_results_v1.csv"
)

MLFLOW_DB = Path("mlflow.db")

MODEL_DIR = Path(
    "models/sequence_v1"
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
    for index, symbol in enumerate(SYMBOLS)
}

SEQUENCE_LENGTH = 60
FEATURE_COUNT = 61
NUM_CLASSES = 3

TRAIN_SEQUENCES_PER_SYMBOL = 100_000
VALIDATION_SEQUENCES_PER_SYMBOL = 20_000

BATCH_SIZE = 128
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

HIDDEN_SIZE = 64
DROPOUT = 0.2

CLASSIFICATION_LOSS_WEIGHT = 1.0

RANDOM_STATE = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_STATE):
    """Set deterministic random seeds."""

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """Return CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# EVEN SAMPLE SELECTION
# ============================================================

def select_evenly(indices, limit):
    """
    Select approximately evenly distributed indices.

    This preserves coverage across the complete training
    period instead of selecting only the most recent rows.
    """

    indices = np.asarray(
        indices,
        dtype=np.int32,
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
# SEQUENCE DATASET
# ============================================================

class CryptoSequenceDataset(Dataset):
    """Lazy-loading dataset for one or more cryptocurrency symbols."""

    def __init__(
        self,
        split,
        sequences_per_symbol,
    ):
        self.split = split

        self.samples = []

        self.feature_arrays = {}
        self.return_arrays = {}
        self.direction_arrays = {}

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

            direction_file = (
                symbol_dir
                / f"{symbol}_direction.npy"
            )

            valid_file = (
                symbol_dir
                / f"{symbol}_valid_indices.npy"
            )

            if not X_file.exists():
                raise FileNotFoundError(
                    f"Missing file: {X_file}"
                )

            X = np.load(
                X_file,
                mmap_mode="r",
            )

            returns = np.load(
                return_file,
                mmap_mode="r",
            )

            directions = np.load(
                direction_file,
                mmap_mode="r",
            )

            valid_indices = np.load(
                valid_file
            )

            selected_indices = select_evenly(
                valid_indices,
                sequences_per_symbol,
            )

            symbol_id = SYMBOL_TO_ID[
                symbol
            ]

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

            self.direction_arrays[
                symbol_id
            ] = directions

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

    def __getitem__(self, index):

        symbol_id, end_index = (
            self.samples[index]
        )

        X = self.feature_arrays[
            symbol_id
        ]

        returns = self.return_arrays[
            symbol_id
        ]

        directions = (
            self.direction_arrays[
                symbol_id
            ]
        )

        start_index = (
            end_index
            - SEQUENCE_LENGTH
            + 1
        )

        sequence = X[
            start_index:
            end_index + 1
        ]

        regression_target = (
            returns[end_index]
        )

        classification_target = (
            directions[end_index]
        )

        sequence_tensor = torch.from_numpy(
            np.array(
                sequence,
                dtype=np.float32,
                copy=True,
            )
        )

        regression_tensor = torch.tensor(
            regression_target,
            dtype=torch.float32,
        )

        classification_tensor = torch.tensor(
            classification_target,
            dtype=torch.long,
        )

        symbol_tensor = torch.tensor(
            symbol_id,
            dtype=torch.long,
        )

        return (
            sequence_tensor,
            symbol_tensor,
            regression_tensor,
            classification_tensor,
        )


# ============================================================
# TARGET STATISTICS
# ============================================================

def calculate_target_statistics(
    dataset,
):
    """Calculate regression mean/std and classification weights."""

    regression_values = []

    class_counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    for symbol_id, end_index in (
        dataset.samples
    ):

        regression_values.append(
            dataset.return_arrays[
                symbol_id
            ][end_index]
        )

        direction = int(
            dataset.direction_arrays[
                symbol_id
            ][end_index]
        )

        class_counts[
            direction
        ] += 1

    regression_values = np.asarray(
        regression_values,
        dtype=np.float32,
    )

    mean = float(
        regression_values.mean()
    )

    std = float(
        regression_values.std()
    )

    if std < 1e-12:
        std = 1.0

    total = class_counts.sum()

    class_weights = (
        total
        / (
            NUM_CLASSES
            * np.maximum(
                class_counts,
                1,
            )
        )
    )

    return (
        mean,
        std,
        torch.tensor(
            class_weights,
            dtype=torch.float32,
        ),
    )


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    model_name,
):
    """Create the requested sequence architecture."""

    if model_name == "GRU":

        return MultiTaskGRU(
            input_size=FEATURE_COUNT,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            dropout=DROPOUT,
            num_symbols=len(SYMBOLS),
            num_classes=NUM_CLASSES,
        )

    if model_name == "LSTM":

        return MultiTaskLSTM(
            input_size=FEATURE_COUNT,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            dropout=DROPOUT,
            num_symbols=len(SYMBOLS),
            num_classes=NUM_CLASSES,
        )

    if model_name == "CNNLSTM":

        return MultiTaskCNNLSTM(
            input_size=FEATURE_COUNT,
            cnn_channels=64,
            hidden_size=HIDDEN_SIZE,
            dropout=DROPOUT,
            num_symbols=len(SYMBOLS),
            num_classes=NUM_CLASSES,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    regression_mean,
    regression_std,
    classification_loss,
    device,
):
    """Run one training epoch."""

    model.train()

    total_loss = 0.0

    total_samples = 0

    for (
        sequences,
        symbol_ids,
        regression_targets,
        classification_targets,
    ) in loader:

        sequences = sequences.to(
            device
        )

        symbol_ids = symbol_ids.to(
            device
        )

        regression_targets = (
            regression_targets.to(
                device
            )
        )

        classification_targets = (
            classification_targets.to(
                device
            )
        )

        normalized_targets = (
            (
                regression_targets
                - regression_mean
            )
            / regression_std
        )

        optimizer.zero_grad()

        regression_output, classification_output = (
            model(
                sequences,
                symbol_ids,
            )
        )

        regression_loss = (
            nn.functional.smooth_l1_loss(
                regression_output,
                normalized_targets,
            )
        )

        classification_loss_value = (
            classification_loss(
                classification_output,
                classification_targets,
            )
        )

        loss = (
            regression_loss
            + (
                CLASSIFICATION_LOSS_WEIGHT
                * classification_loss_value
            )
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = (
            sequences.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    regression_mean,
    regression_std,
    classification_loss,
    device,
):
    """Evaluate both prediction heads."""

    model.eval()

    total_loss = 0.0

    total_samples = 0

    actual_returns = []
    predicted_returns = []

    actual_classes = []
    predicted_classes = []

    with torch.no_grad():

        for (
            sequences,
            symbol_ids,
            regression_targets,
            classification_targets,
        ) in loader:

            sequences = sequences.to(
                device
            )

            symbol_ids = symbol_ids.to(
                device
            )

            regression_targets = (
                regression_targets.to(
                    device
                )
            )

            classification_targets = (
                classification_targets.to(
                    device
                )
            )

            normalized_targets = (
                (
                    regression_targets
                    - regression_mean
                )
                / regression_std
            )

            regression_output, classification_output = (
                model(
                    sequences,
                    symbol_ids,
                )
            )

            regression_loss = (
                nn.functional.smooth_l1_loss(
                    regression_output,
                    normalized_targets,
                )
            )

            classification_loss_value = (
                classification_loss(
                    classification_output,
                    classification_targets,
                )
            )

            loss = (
                regression_loss
                + (
                    CLASSIFICATION_LOSS_WEIGHT
                    * classification_loss_value
                )
            )

            batch_size = (
                sequences.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            predictions = (
                (
                    regression_output
                    * regression_std
                )
                + regression_mean
            )

            classes = (
                torch.argmax(
                    classification_output,
                    dim=1,
                )
            )

            actual_returns.extend(
                regression_targets.cpu().numpy()
            )

            predicted_returns.extend(
                predictions.cpu().numpy()
            )

            actual_classes.extend(
                classification_targets.cpu().numpy()
            )

            predicted_classes.extend(
                classes.cpu().numpy()
            )

    actual_returns = np.asarray(
        actual_returns
    )

    predicted_returns = np.asarray(
        predicted_returns
    )

    actual_classes = np.asarray(
        actual_classes
    )

    predicted_classes = np.asarray(
        predicted_classes
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_returns,
            predicted_returns,
        )
    )

    mae = mean_absolute_error(
        actual_returns,
        predicted_returns,
    )

    r2 = r2_score(
        actual_returns,
        predicted_returns,
    )

    macro_f1 = f1_score(
        actual_classes,
        predicted_classes,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        actual_classes,
        predicted_classes,
    )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def train_model(
    model_name,
):
    """Train one sequence architecture."""

    set_seed()

    device = get_device()

    print()
    print("=" * 70)
    print(
        f"SEQUENCE MODEL: {model_name}"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print()
    print(
        "Loading training sequences..."
    )

    train_dataset = (
        CryptoSequenceDataset(
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

    print(
        "Loading validation sequences..."
    )

    validation_dataset = (
        CryptoSequenceDataset(
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
    # Target statistics
    # --------------------------------------------------------

    (
        regression_mean,
        regression_std,
        class_weights,
    ) = calculate_target_statistics(
        train_dataset
    )

    print()
    print(
        f"Regression mean: "
        f"{regression_mean:.10f}"
    )

    print(
        f"Regression std: "
        f"{regression_std:.10f}"
    )

    print(
        f"Class weights: "
        f"{class_weights.tolist()}"
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        model_name
    ).to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{total_parameters:,}"
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    classification_loss = (
        nn.CrossEntropyLoss(
            weight=class_weights.to(
                device
            )
        )
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        "sqlite:///"
        f"{MLFLOW_DB.resolve().as_posix()}"
    )

    mlflow.set_experiment(
        "CryptoML_Sequence"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_state = None

    history = []

    with mlflow.start_run(
        run_name=(
            f"{model_name}_Sequence_v1"
        )
    ) as run:

        mlflow.log_param(
            "model",
            model_name,
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
            "learning_rate",
            LEARNING_RATE,
        )

        mlflow.log_param(
            "hidden_size",
            HIDDEN_SIZE,
        )

        mlflow.log_param(
            "dropout",
            DROPOUT,
        )

        mlflow.log_param(
            "regression_mean",
            regression_mean,
        )

        mlflow.log_param(
            "regression_std",
            regression_std,
        )

        epochs_without_improvement = 0
        best_epoch = 0

        for epoch in range(
            1,
            EPOCHS + 1,
        ):

            train_loss = (
                train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    regression_mean=(
                        regression_mean
                    ),
                    regression_std=(
                        regression_std
                    ),
                    classification_loss=(
                        classification_loss
                    ),
                    device=device,
                )
            )

            metrics = evaluate(
                model=model,
                loader=validation_loader,
                regression_mean=(
                    regression_mean
                ),
                regression_std=(
                    regression_std
                ),
                classification_loss=(
                    classification_loss
                ),
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
                f"Epoch {epoch}/{EPOCHS}"
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
                f"Val RMSE: "
                f"{metrics['rmse']:.8f}"
            )

            print(
                f"Val MAE: "
                f"{metrics['mae']:.8f}"
            )

            print(
                f"Val R2: "
                f"{metrics['r2']:.6f}"
            )

            print(
                f"Val Macro-F1: "
                f"{metrics['macro_f1']:.6f}"
            )

            print(
                f"Val Accuracy: "
                f"{metrics['accuracy']:.6f}"
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
                "validation_rmse",
                metrics["rmse"],
                step=epoch,
            )

            mlflow.log_metric(
                "validation_mae",
                metrics["mae"],
                step=epoch,
            )

            mlflow.log_metric(
                "validation_r2",
                metrics["r2"],
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

            if (
                metrics["loss"]
                < best_validation_loss
            ):

                best_validation_loss = (
                    metrics["loss"]
                )

                best_state = copy.deepcopy(
                    model.state_dict()
                )
                best_epoch = epoch
                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1

                if (
                    epochs_without_improvement
                    >= EARLY_STOPPING_PATIENCE
                ):

                    print()
                    print(
                        "Early stopping triggered."
                    )

                    break

            print()
            print(
                f"Best epoch: {best_epoch}"
            )

            print(
                f"Best validation loss: "
                f"{best_validation_loss:.6f}"
            )

        # ----------------------------------------------------
        # Restore best state
        # ----------------------------------------------------

        if best_state is not None:

            model.load_state_dict(
                best_state
            )

        final_metrics = evaluate(
            model=model,
            loader=validation_loader,
            regression_mean=(
                regression_mean
            ),
            regression_std=(
                regression_std
            ),
            classification_loss=(
                classification_loss
            ),
            device=device,
        )

        mlflow.log_metric(
            "best_validation_loss",
            final_metrics["loss"],
        )

        mlflow.log_metric(
            "best_validation_rmse",
            final_metrics["rmse"],
        )

        mlflow.log_metric(
            "best_validation_mae",
            final_metrics["mae"],
        )

        mlflow.log_metric(
            "best_validation_r2",
            final_metrics["r2"],
        )

        mlflow.log_metric(
            "best_validation_macro_f1",
            final_metrics["macro_f1"],
        )

        mlflow.log_metric(
            "best_validation_accuracy",
            final_metrics["accuracy"],
        )


        # ----------------------------------------------------
        # Save local checkpoint
        # ----------------------------------------------------

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = (
            MODEL_DIR
            / f"{model_name.lower()}_best.pt"
        )

        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": (
                    model.state_dict()
                ),
                "input_size": FEATURE_COUNT,
                "sequence_length": (
                    SEQUENCE_LENGTH
                ),
                "hidden_size": HIDDEN_SIZE,
                "dropout": DROPOUT,
                "regression_mean": (
                    regression_mean
                ),
                "regression_std": (
                    regression_std
                ),
                "symbol_to_id": (
                    SYMBOL_TO_ID
                ),
                "metrics": final_metrics,
            },
            checkpoint_path,
        )

        mlflow.log_artifact(
            str(checkpoint_path)
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history_file = (
            REPORTS_DIR
            / f"sequence_{model_name.lower()}_history.csv"
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_file,
            index=False,
        )

        mlflow.log_artifact(
            str(history_file)
        )

    return final_metrics


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "GRU",
            "LSTM",
            "CNNLSTM",
        ],
        default="GRU",
    )

    args = parser.parse_args()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = train_model(
        args.model
    )

    # --------------------------------------------------------
    # Save/append summary
    # --------------------------------------------------------

    result = pd.DataFrame(
        [
            {
                "model": args.model,
                "sequence_length": (
                    SEQUENCE_LENGTH
                ),
                "train_sequences": (
                    TRAIN_SEQUENCES_PER_SYMBOL
                    * len(SYMBOLS)
                ),
                "validation_sequences": (
                    VALIDATION_SEQUENCES_PER_SYMBOL
                    * len(SYMBOLS)
                ),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
        ]
    )

    if RESULTS_FILE.exists():

        existing = pd.read_csv(
            RESULTS_FILE
        )

        existing = existing[
            existing["model"]
            != args.model
        ]

        result = pd.concat(
            [
                existing,
                result,
            ],
            ignore_index=True,
        )

    result.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print(
        f"{args.model} TRAINING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Validation RMSE: "
        f"{metrics['rmse']:.8f}"
    )

    print(
        f"Validation MAE: "
        f"{metrics['mae']:.8f}"
    )

    print(
        f"Validation R2: "
        f"{metrics['r2']:.6f}"
    )

    print(
        f"Validation Macro-F1: "
        f"{metrics['macro_f1']:.6f}"
    )

    print(
        f"Validation Accuracy: "
        f"{metrics['accuracy']:.6f}"
    )

    print()
    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()