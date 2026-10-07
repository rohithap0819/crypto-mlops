from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from catboost import CatBoostClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_CATBOOST = (
    ROOT
    / "models"
    / "final"
    / "binary_direction_catboost_v1.cbm"
)

MODEL_GRU = (
    ROOT
    / "models"
    / "binary_sequence_v1"
    / "gru_binary_best.pt"
)

SCALER_FILE = (
    ROOT
    / "data"
    / "processed"
    / "sequence_v1"
    / "feature_scaler.joblib"
)

FEATURE_COLUMNS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "feature_columns_v2.txt"
)

FEATURE_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "live_features_latest.parquet"
)

SEQUENCE_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "live_sequence_v1"
)

SEQUENCE_OUTPUT = (
    SEQUENCE_OUTPUT_DIR
    / "latest_gru_sequence.npz"
)

REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# IMPORT EXISTING PROJECT COMPONENTS
# ============================================================

FEATURE_DIR = (
    ROOT
    / "src"
    / "features"
)

MODEL_DIR = (
    ROOT
    / "src"
    / "models"
)

INFERENCE_DIR = (
    ROOT
    / "src"
    / "inference"
)

MONITORING_DIR = (
    ROOT
    / "src"
    / "monitoring"
)

for directory in [
    FEATURE_DIR,
    MODEL_DIR,
    INFERENCE_DIR,
    MONITORING_DIR,
]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


from build_features_v2 import (  # noqa: E402
    SYMBOLS,
    calculate_features,
    create_cross_asset_features,
)

from live_feature_builder import (  # noqa: E402
    get_connection,
    load_recent_symbol_data,
    validate_recent_history,
    find_latest_common_timestamp,
)

from binary_sequence_models import BinaryGRU  # noqa: E402

from build_live_gru_sequence import (  # noqa: E402
    build_symbol_sequence,
)

from predict_live_catboost import (  # noqa: E402
    build_model_input,
)

from predict_live_gru import (  # noqa: E402
    predict as predict_gru,
)

from predict_live_catboost import (  # noqa: E402
    predict as predict_catboost,
)

from predict_live_ensemble import (  # noqa: E402
    build_ensemble,
)

from live_inference_runner import (  # noqa: E402
    get_latest_common_timestamp,
    has_processed_timestamp,
    initialize_prediction_table,
    store_predictions,
    display_predictions,
)

from monitor_live import (  # noqa: E402
    main as run_monitoring,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 60
FEATURE_COUNT = 61
LOOKBACK_CANDLES = 180

MODEL_VERSION = (
    "catboost_v1_gru_v1_ensemble_v1"
)

POLL_SECONDS = 5

CATBOOST_WEIGHT = 0.55
GRU_WEIGHT = 0.45
CONFIDENCE_THRESHOLD = 0.55

SYMBOL_IDS = {
    symbol: index
    for index, symbol in enumerate(
        SYMBOLS
    )
}


# ============================================================
# RESIDENT SERVICE
# ============================================================

class ResidentInferenceService:

    def __init__(self) -> None:

        print("=" * 70)
        print("CRYPTO MLOPS - RESIDENT INFERENCE SERVICE")
        print("=" * 70)

        print()
        print("Loading production components...")

        self.feature_columns = (
            self.load_feature_columns()
        )

        self.scaler = joblib.load(
            SCALER_FILE
        )

        self.catboost = (
            self.load_catboost()
        )

        self.gru = (
            self.load_gru()
        )

        self.warm_up_models()

        initialize_prediction_table()

        print()
        print("Production components loaded:")
        print(
            f"  V2 features : "
            f"{len(self.feature_columns)}"
        )
        print(
            f"  GRU sequence: "
            f"{SEQUENCE_LENGTH} x "
            f"{FEATURE_COUNT}"
        )
        print(
            f"  CatBoost    : loaded"
        )
        print(
            f"  GRU         : loaded"
        )
        print(
            f"  Scaler      : loaded"
        )
        print(
            f"  Ensemble    : 55% CatBoost / 45% GRU"
        )
        print(
            f"  Threshold   : {CONFIDENCE_THRESHOLD}"
        )

    # ========================================================
    # LOADERS
    # ========================================================

    def load_feature_columns(
        self,
    ) -> list[str]:

        if not FEATURE_COLUMNS_FILE.exists():
            raise FileNotFoundError(
                f"Feature file not found: "
                f"{FEATURE_COLUMNS_FILE}"
            )

        features = [
            line.strip()
            for line in FEATURE_COLUMNS_FILE.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if len(features) != FEATURE_COUNT:
            raise ValueError(
                f"Expected {FEATURE_COUNT} features, "
                f"found {len(features)}"
            )

        return features

    def load_catboost(
        self,
    ) -> CatBoostClassifier:

        model = CatBoostClassifier()

        model.load_model(
            str(MODEL_CATBOOST)
        )

        if len(
            model.feature_names_
        ) != 66:
            raise ValueError(
                "CatBoost model must contain 66 features."
            )

        return model

    def load_gru(
        self,
    ) -> BinaryGRU:

        checkpoint = torch.load(
            MODEL_GRU,
            map_location="cpu",
            weights_only=False,
        )

        model = BinaryGRU(
            input_size=int(
                checkpoint["feature_count"]
            ),
            hidden_size=int(
                checkpoint["hidden_size"]
            ),
            num_layers=1,
            dropout=float(
                checkpoint["dropout"]
            ),
            num_symbols=5,
            symbol_embedding_dim=4,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model.eval()

        return model

    # ========================================================
    # MODEL WARMUP
    # ========================================================

    def warm_up_models(self) -> None:

        print()
        print("Warming up models...")

        dummy_X = np.zeros(
            (
                1,
                SEQUENCE_LENGTH,
                FEATURE_COUNT,
            ),
            dtype=np.float32,
        )

        dummy_symbol = np.array(
            [0],
            dtype=np.int64,
        )

        X_tensor = torch.from_numpy(
            dummy_X
        )

        symbol_tensor = torch.from_numpy(
            dummy_symbol
        )

        with torch.inference_mode():

            self.gru(
                X_tensor,
                symbol_tensor,
            )

        dummy_catboost = pd.DataFrame(
            np.zeros(
                (
                    1,
                    66,
                ),
                dtype=np.float32,
            ),
            columns=[
                name
                for name in self.catboost.feature_names_
            ],
        )

        self.catboost.predict_proba(
            dummy_catboost
        )

        print(
            "Model warm-up complete."
        )

    # ========================================================
    # BUILD FEATURES
    # ========================================================

    def build_feature_state(
        self,
    ) -> tuple[
        pd.DataFrame,
        dict[str, pd.DataFrame],
        pd.Timestamp,
    ]:

        raw_data = {}

        for symbol in SYMBOLS:

            raw_data[symbol] = (
                load_recent_symbol_data(
                    symbol=symbol,
                    limit=LOOKBACK_CANDLES,
                )
            )

        latest_common_time = (
            find_latest_common_timestamp(
                raw_data
            )
        )

        validate_recent_history(
            raw_data,
            latest_common_time,
        )

        feature_data = {}

        for symbol in SYMBOLS:

            df = raw_data[symbol].copy()

            df = calculate_features(
                df
            )

            feature_data[symbol] = df

        cross_features = (
            create_cross_asset_features(
                feature_data
            )
        )

        latest_frames = []

        merged_feature_data = {}

        for symbol in SYMBOLS:

            df = feature_data[symbol].merge(
                cross_features,
                on="open_time",
                how="left",
            )

            df["symbol"] = symbol

            merged_feature_data[
                symbol
            ] = df

            current = df[
                df["open_time"]
                == latest_common_time
            ].copy()

            latest_frames.append(
                current
            )

        live_features = pd.concat(
            latest_frames,
            ignore_index=True,
        )

        live_features = (
            live_features
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        missing = [
            feature
            for feature in self.feature_columns
            if feature not in live_features.columns
        ]

        if missing:
            raise ValueError(
                f"Missing features: {missing}"
            )

        if (
            live_features[
                self.feature_columns
            ].isna().any().any()
        ):
            raise ValueError(
                "Live features contain NaN values."
            )

        live_features = live_features[
            [
                "symbol",
                "open_time",
                "close",
            ]
            + self.feature_columns
        ].copy()

        live_features = (
            live_features
            .sort_values("symbol")
            .reset_index(drop=True)
        )

        return (
            live_features,
            merged_feature_data,
            latest_common_time,
        )

    # ========================================================
    # BUILD GRU INPUT
    # ========================================================

    def build_gru_batch(
        self,
        feature_data: dict[str, pd.DataFrame],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        sequences = []
        symbol_ids = []

        for symbol in SYMBOLS:

            sequence, _, _ = (
                build_symbol_sequence(
                    df=feature_data[
                        symbol
                    ],
                    feature_columns=(
                        self.feature_columns
                    ),
                    scaler=self.scaler,
                    symbol=symbol,
                )
            )

            sequences.append(
                sequence
            )

            symbol_ids.append(
                SYMBOL_IDS[symbol]
            )

        X = np.stack(
            sequences
        ).astype(
            np.float32
        )

        ids = np.asarray(
            symbol_ids,
            dtype=np.int64,
        )

        expected = (
            len(SYMBOLS),
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        )

        if X.shape != expected:
            raise ValueError(
                f"Unexpected GRU batch shape: "
                f"{X.shape}; "
                f"expected {expected}"
            )

        return (
            X,
            ids,
        )

    # ========================================================
    # SAVE CURRENT INPUT SNAPSHOT
    # ========================================================

    def save_input_snapshot(
        self,
        live_features: pd.DataFrame,
        X: np.ndarray,
        symbol_ids: np.ndarray,
    ) -> None:

        FEATURE_OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        SEQUENCE_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        live_features.to_parquet(
            FEATURE_OUTPUT,
            index=False,
        )

        np.savez(
            SEQUENCE_OUTPUT,
            X=X,
            symbol_ids=symbol_ids,
        )

    # ========================================================
    # RUN ONE INFERENCE CYCLE
    # ========================================================

    def run_cycle(
        self,
        candle_open_time_ms: int,
    ) -> float:

        started = (
            time.perf_counter()
        )

        print()
        print("=" * 70)
        print(
            "RESIDENT INFERENCE CYCLE"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------

        (
            live_features,
            feature_data,
            latest_common_time,
        ) = self.build_feature_state()

        print(
            f"Feature timestamp: "
            f"{latest_common_time}"
        )

        # ----------------------------------------------------
        # GRU sequence
        # ----------------------------------------------------

        X, symbol_ids = (
            self.build_gru_batch(
                feature_data
            )
        )

        self.save_input_snapshot(
            live_features,
            X,
            symbol_ids,
        )

        print(
            f"GRU batch shape: "
            f"{X.shape}"
        )

        # ----------------------------------------------------
        # GRU prediction
        # ----------------------------------------------------

        gru_start = (
            time.perf_counter()
        )

        gru_predictions = (
            predict_gru(
                model=self.gru,
                X=X,
                symbol_ids=symbol_ids,
            )
        )

        gru_latency_ms = (
            time.perf_counter()
            - gru_start
        ) * 1000.0

        # Add exact candle timestamp.
        timestamp_map = (
            live_features[
                [
                    "symbol",
                    "open_time",
                ]
            ].copy()
        )

        gru_predictions = (
            gru_predictions
            .drop(columns=["open_time"], errors="ignore")
            .merge(
                timestamp_map,
                on="symbol",
                how="left",
                validate="one_to_one",
            )
        )

        # ----------------------------------------------------
        # CatBoost prediction
        # ----------------------------------------------------

        cat_start = (
            time.perf_counter()
        )

        metadata_df, model_input = (
            build_model_input(
                model=self.catboost,
                df=live_features,
            )
        )

        catboost_predictions = (
            predict_catboost(
                model=self.catboost,
                metadata_df=metadata_df,
                model_input=model_input,
            )
        )

        cat_latency_ms = (
            time.perf_counter()
            - cat_start
        ) * 1000.0

        # ----------------------------------------------------
        # Ensemble
        # ----------------------------------------------------

        ensemble = build_ensemble(
            catboost=catboost_predictions,
            gru=gru_predictions,
        )

        # ----------------------------------------------------
        # Save latest report snapshots
        # ----------------------------------------------------

        gru_predictions.to_csv(
            REPORTS_DIR
            / "live_gru_predictions_v1.csv",
            index=False,
        )

        catboost_predictions.to_csv(
            REPORTS_DIR
            / "live_catboost_predictions_v1.csv",
            index=False,
        )

        ensemble.to_csv(
            REPORTS_DIR
            / "live_ensemble_predictions_v1.csv",
            index=False,
        )

        # ----------------------------------------------------
        # JSON snapshot
        # ----------------------------------------------------

        json_payload = {
            "model_version": MODEL_VERSION,
            "candle_open_time": str(
                latest_common_time
            ),
            "catboost_weight": (
                CATBOOST_WEIGHT
            ),
            "gru_weight": GRU_WEIGHT,
            "confidence_threshold": (
                CONFIDENCE_THRESHOLD
            ),
            "rows": ensemble.to_dict(
                orient="records"
            ),
        }

        (
            REPORTS_DIR
            / "live_ensemble_predictions_v1.json"
        ).write_text(
            json.dumps(
                json_payload,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Total cycle latency
        # ----------------------------------------------------

        total_latency_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        store_predictions(
            ensemble=ensemble,
            catboost=catboost_predictions,
            gru=gru_predictions,
            prices=live_features[
                [
                    "symbol",
                    "open_time",
                    "close",
                ]
            ],
            latency_ms=total_latency_ms,
        )

        display_predictions(
            ensemble=ensemble,
            latency_ms=total_latency_ms,
        )

        print(
            f"GRU inference: "
            f"{gru_latency_ms:.3f} ms"
        )

        print(
            f"CatBoost inference: "
            f"{cat_latency_ms:.3f} ms"
        )

        print(
            f"Resident total: "
            f"{total_latency_ms:.3f} ms"
        )

        # ----------------------------------------------------
        # Monitoring
        # ----------------------------------------------------

        try:

            run_monitoring()

        except Exception as exc:

            print(
                f"Monitoring failed: {exc}",
                file=sys.stderr,
            )

        return total_latency_ms

    # ========================================================
    # SERVICE LOOP
    # ========================================================

    def run(self) -> None:

        processed_timestamp = None

        print()
        print(
            "Waiting for new completed candles..."
        )

        try:

            while True:

                try:

                    latest_timestamp = (
                        get_latest_common_timestamp()
                    )

                    if latest_timestamp is None:

                        time.sleep(
                            POLL_SECONDS
                        )

                        continue

                    if (
                        latest_timestamp
                        != processed_timestamp
                        and not has_processed_timestamp(
                            latest_timestamp
                        )
                    ):

                        readable = (
                            pd.to_datetime(
                                latest_timestamp,
                                unit="ms",
                                utc=True,
                            )
                        )

                        print()
                        print(
                            f"New candle: "
                            f"{readable}"
                        )

                        self.run_cycle(
                            latest_timestamp
                        )

                        processed_timestamp = (
                            latest_timestamp
                        )

                except Exception as exc:

                    print()
                    print(
                        "Inference cycle failed:",
                        file=sys.stderr,
                    )

                    print(
                        str(exc),
                        file=sys.stderr,
                    )

                    import traceback

                    traceback.print_exc()

                time.sleep(
                    POLL_SECONDS
                )

        except KeyboardInterrupt:

            print()
            print(
                "Resident inference service stopped."
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    service = (
        ResidentInferenceService()
    )

    service.run()