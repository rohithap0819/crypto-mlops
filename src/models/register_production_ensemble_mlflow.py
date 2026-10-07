from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CATBOOST_MODEL = (
    ROOT / "models" / "final" / "binary_direction_catboost_v1.cbm"
)

GRU_MODEL = (
    ROOT / "models" / "binary_sequence_v1" / "gru_binary_best.pt"
)

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FROZEN PRODUCTION CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "CryptoML_Production_Ensemble"
RUN_NAME = "Production_Ensemble_CatBoost55_GRU45_5m"

CATBOOST_WEIGHT = 0.55
GRU_WEIGHT = 0.45

CONFIDENCE_THRESHOLD = 0.55
FORECAST_HORIZON_MINUTES = 5
SEQUENCE_LENGTH = 60

FEATURE_VERSION = "v2"
FEATURE_COUNT = 61
TABULAR_MODEL_FEATURE_COUNT = 66
NUM_SYMBOLS = 5

CATBOOST_RUN_ID = "c9bb6545fc3448859fc620d9474e2d25"

GRU_RUN_ID = "4f1ca6ff7e7e43bba2b5f806d12df0ed"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def sha256_file(path: Path) -> str:
    """Return SHA-256 checksum for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit() -> str:
    """Return current Git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def find_feature_file() -> Path:
    """
    Locate feature_columns_v2.txt.

    Prefer the canonical src/features location.
    """
    preferred = ROOT / "src" / "features" / "feature_columns_v2.txt"

    if preferred.exists():
        return preferred

    candidates = list(ROOT.rglob("feature_columns_v2.txt"))

    candidates = [
        path
        for path in candidates
        if ".venv" not in path.parts
        and "site-packages" not in path.parts
    ]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "Could not find feature_columns_v2.txt"
        )

    raise RuntimeError(
        "Multiple feature_columns_v2.txt files found:\n"
        + "\n".join(str(path) for path in candidates)
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs() -> Path:
    """Validate frozen artifacts and feature definition."""

    if not CATBOOST_MODEL.exists():
        raise FileNotFoundError(
            f"Missing CatBoost model: {CATBOOST_MODEL}"
        )

    if not GRU_MODEL.exists():
        raise FileNotFoundError(
            f"Missing GRU model: {GRU_MODEL}"
        )

    feature_file = find_feature_file()

    features = [
        line.strip()
        for line in feature_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(features) != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features, "
            f"found {len(features)} in {feature_file}"
        )

    return feature_file


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    feature_file = validate_inputs()

    git_commit = get_git_commit()
    git_dirty = git_is_dirty()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME) as run:

        run_id = run.info.run_id

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "model_type": "CatBoost_GRU_Ensemble",
                "task": "binary_5m_direction",
                "feature_version": FEATURE_VERSION,
                "feature_count": FEATURE_COUNT,
                "tabular_model_feature_count": TABULAR_MODEL_FEATURE_COUNT,
                "sequence_length": SEQUENCE_LENGTH,
                "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
                "catboost_weight": CATBOOST_WEIGHT,
                "gru_weight": GRU_WEIGHT,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "num_symbols": NUM_SYMBOLS,
                "catboost_run_id": CATBOOST_RUN_ID,
                "gru_run_id": GRU_RUN_ID,
                "production_artifact_status": "frozen_existing_artifacts",
            }
        )

        # ----------------------------------------------------
        # Frozen evaluation metrics
        #
        # These are research metrics already produced earlier.
        # No test-set tuning is performed here.
        # ----------------------------------------------------

        mlflow.log_metrics(
            {
                "validation_ensemble_accuracy": 0.525361,
                "validation_ensemble_macro_f1": 0.525356,
                "test_ensemble_accuracy": 0.516428,
                "test_ensemble_macro_f1": 0.516428,
                "test_confidence_055_accuracy": 0.539314,
                "test_confidence_055_coverage": 0.110997,
            }
        )

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        mlflow.set_tags(
            {
                "model_stage": "production_candidate",
                "deployment_status": "not_deployed",
                "task_type": "binary_direction",
                "forecast_horizon": "5m",
                "ensemble": "CatBoost_55_GRU_45",
                "confidence_threshold": "0.55",
                "feature_version": "v2",
                "source_git_commit": git_commit,
                "source_git_dirty": str(git_dirty),
                "catboost_source_run": CATBOOST_RUN_ID,
                "gru_source_run": GRU_RUN_ID,
                "test_set_pristine": "false",
                "test_set_usage_note": (
                    "Test set was used for multiple frozen evaluations; "
                    "do not use it for future tuning."
                ),
                "economic_edge_status": (
                    "insufficient_after_realistic_transaction_costs"
                ),
            }
        )

        # ----------------------------------------------------
        # Log frozen model artifacts
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(CATBOOST_MODEL),
            artifact_path="models/catboost",
        )

        mlflow.log_artifact(
            str(GRU_MODEL),
            artifact_path="models/gru",
        )

        # ----------------------------------------------------
        # Log feature definition
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(feature_file),
            artifact_path="features",
        )

        # ----------------------------------------------------
        # File checksums
        # ----------------------------------------------------

        catboost_sha256 = sha256_file(CATBOOST_MODEL)
        gru_sha256 = sha256_file(GRU_MODEL)

        # ----------------------------------------------------
        # Production manifest
        # ----------------------------------------------------

        manifest = {
            "model_name": RUN_NAME,
            "mlflow_run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),

            "task": "binary_5m_direction",
            "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,

            "ensemble": {
                "catboost_weight": CATBOOST_WEIGHT,
                "gru_weight": GRU_WEIGHT,
            },

            "confidence": {
                "threshold": CONFIDENCE_THRESHOLD,
                "interpretation": (
                    "Selective prediction research setting; "
                    "not a guarantee of correctness."
                ),
            },

            "features": {
                "version": FEATURE_VERSION,
                "feature_count": FEATURE_COUNT,
                "tabular_model_feature_count": (
                    TABULAR_MODEL_FEATURE_COUNT
                ),
                "sequence_length": SEQUENCE_LENGTH,
                "num_symbols": NUM_SYMBOLS,
                "feature_file": str(
                    feature_file.relative_to(ROOT)
                ),
            },

            "catboost": {
                "artifact": str(
                    CATBOOST_MODEL.relative_to(ROOT)
                ),
                "sha256": catboost_sha256,
                "source_mlflow_run": CATBOOST_RUN_ID,
                "iterations": 300,
                "depth": 8,
                "learning_rate": 0.05,
                "loss_function": "Logloss",
                "auto_class_weights": "Balanced",
                "random_seed": 42,
            },

            "gru": {
                "artifact": str(
                    GRU_MODEL.relative_to(ROOT)
                ),
                "sha256": gru_sha256,
                "source_mlflow_run": GRU_RUN_ID,
                "architecture": "BinaryGRU",
                "input_features": FEATURE_COUNT,
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.2,
                "symbol_embedding_dim": 4,
            },

            "evaluation": {
                "validation_accuracy": 0.525361,
                "validation_macro_f1": 0.525356,
                "test_accuracy": 0.516428,
                "test_macro_f1": 0.516428,
                "confidence_055_test_accuracy": 0.539314,
                "confidence_055_test_coverage": 0.110997,
            },

            "governance": {
                "model_stage": "production_candidate",
                "deployment_status": "not_deployed",
                "test_set_pristine": False,
                "future_tuning_on_test_allowed": False,
                "economic_edge_status": (
                    "insufficient_after_realistic_transaction_costs"
                ),
            },

            "source_control": {
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "repository": (
                    "https://github.com/rohithap0819/crypto-mlops"
                ),
            },
        }

        manifest_path = (
            REPORTS_DIR
            / "production_ensemble_manifest_v1.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(manifest_path),
            artifact_path="metadata",
        )

        print()
        print("=" * 70)
        print("PRODUCTION ENSEMBLE REGISTERED IN MLFLOW")
        print("=" * 70)
        print(f"Experiment : {EXPERIMENT_NAME}")
        print(f"Run name   : {RUN_NAME}")
        print(f"Run ID     : {run_id}")
        print()
        print("CatBoost   : 55%")
        print("GRU        : 45%")
        print("Horizon    : 5 minutes")
        print("Threshold  : 0.55")
        print()
        print("Validation Accuracy : 0.525361")
        print("Validation Macro-F1 : 0.525356")
        print("Test Accuracy       : 0.516428")
        print("Test Macro-F1       : 0.516428")
        print()
        print("Manifest:")
        print(manifest_path)
        print("=" * 70)


if __name__ == "__main__":
    main()