from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CATBOOST_FILE = (
    ROOT
    / "reports"
    / "live_catboost_predictions_v1.csv"
)

GRU_FILE = (
    ROOT
    / "reports"
    / "live_gru_predictions_v1.csv"
)

OUTPUT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "live_ensemble_predictions_v1.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "live_ensemble_predictions_v1.json"
)


# ============================================================
# FROZEN PRODUCTION CONFIGURATION
# ============================================================

CATBOOST_WEIGHT = 0.55
GRU_WEIGHT = 0.45

CONFIDENCE_THRESHOLD = 0.55
DIRECTION_THRESHOLD = 0.50

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]


# ============================================================
# LOAD MODEL OUTPUTS
# ============================================================

def load_predictions(
    path: Path,
    model_name: str,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} prediction file not found: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "symbol",
        "open_time",
        "probability_down",
        "probability_up",
        "confidence",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{model_name} output is missing columns: "
            f"{missing}"
        )

    df = df[
        required
    ].copy()

    df["open_time"] = (
        pd.to_datetime(
            df["open_time"],
            utc=True,
        )
        .astype(str)
    )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs(
    catboost: pd.DataFrame,
    gru: pd.DataFrame,
) -> None:

    expected_symbols = set(SYMBOLS)

    catboost_symbols = set(
        catboost["symbol"]
    )

    gru_symbols = set(
        gru["symbol"]
    )

    if catboost_symbols != expected_symbols:
        raise ValueError(
            "CatBoost symbols do not match expected symbols: "
            f"{sorted(catboost_symbols)}"
        )

    if gru_symbols != expected_symbols:
        raise ValueError(
            "GRU symbols do not match expected symbols: "
            f"{sorted(gru_symbols)}"
        )

    if len(catboost) != len(SYMBOLS):
        raise ValueError(
            f"Expected {len(SYMBOLS)} CatBoost rows, "
            f"found {len(catboost)}"
        )

    if len(gru) != len(SYMBOLS):
        raise ValueError(
            f"Expected {len(SYMBOLS)} GRU rows, "
            f"found {len(gru)}"
        )

    catboost_keys = set(
        zip(
            catboost["symbol"],
            catboost["open_time"],
        )
    )

    gru_keys = set(
        zip(
            gru["symbol"],
            gru["open_time"],
        )
    )

    if catboost_keys != gru_keys:
        raise ValueError(
            "CatBoost and GRU timestamps/symbols do not align."
        )


# ============================================================
# BUILD ENSEMBLE
# ============================================================

def build_ensemble(
    catboost: pd.DataFrame,
    gru: pd.DataFrame,
) -> pd.DataFrame:

    merged = catboost.merge(
        gru,
        on=[
            "symbol",
            "open_time",
        ],
        how="inner",
        suffixes=(
            "_catboost",
            "_gru",
        ),
        validate="one_to_one",
    )

    if len(merged) != len(SYMBOLS):
        raise ValueError(
            "Unexpected ensemble row count: "
            f"{len(merged)}"
        )

    # --------------------------------------------------------
    # Ensemble probabilities
    # --------------------------------------------------------

    merged["ensemble_probability_up"] = (
        CATBOOST_WEIGHT
        * merged["probability_up_catboost"]
        +
        GRU_WEIGHT
        * merged["probability_up_gru"]
    )

    merged["ensemble_probability_down"] = (
        1.0
        - merged["ensemble_probability_up"]
    )

    # --------------------------------------------------------
    # Raw direction
    # --------------------------------------------------------

    merged["raw_prediction"] = np.where(
        merged["ensemble_probability_up"]
        >= DIRECTION_THRESHOLD,
        "UP",
        "DOWN",
    )

    merged["ensemble_confidence"] = (
        np.maximum(
            merged["ensemble_probability_up"],
            merged["ensemble_probability_down"],
        )
    )

    # --------------------------------------------------------
    # Selective production/research signal
    #
    # Below 0.55 confidence we deliberately do not
    # produce an actionable signal.
    # --------------------------------------------------------

    merged["actionable"] = (
        merged["ensemble_confidence"]
        >= CONFIDENCE_THRESHOLD
    )

    merged["signal"] = np.where(
        merged["actionable"],
        merged["raw_prediction"],
        "HOLD",
    )

    # --------------------------------------------------------
    # Return clean output
    # --------------------------------------------------------

    output_columns = [
        "symbol",
        "open_time",
        "probability_up_catboost",
        "probability_down_catboost",
        "probability_up_gru",
        "probability_down_gru",
        "ensemble_probability_up",
        "ensemble_probability_down",
        "ensemble_confidence",
        "raw_prediction",
        "actionable",
        "signal",
    ]

    return (
        merged[
            output_columns
        ]
        .sort_values("symbol")
        .reset_index(drop=True)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CRYPTO MLOPS - LIVE ENSEMBLE INFERENCE")
    print("=" * 70)

    print()
    print("Configuration:")
    print(
        f"CatBoost weight      : {CATBOOST_WEIGHT}"
    )
    print(
        f"GRU weight           : {GRU_WEIGHT}"
    )
    print(
        f"Direction threshold  : {DIRECTION_THRESHOLD}"
    )
    print(
        f"Confidence threshold : {CONFIDENCE_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    catboost = load_predictions(
        CATBOOST_FILE,
        "CatBoost",
    )

    gru = load_predictions(
        GRU_FILE,
        "GRU",
    )

    # --------------------------------------------------------
    # Validate alignment
    # --------------------------------------------------------

    validate_inputs(
        catboost,
        gru,
    )

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    result = build_ensemble(
        catboost,
        gru,
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    metadata = {
        "model": "CatBoost_GRU_Ensemble",
        "task": "binary_5m_direction",
        "forecast_horizon_minutes": 5,
        "catboost_weight": CATBOOST_WEIGHT,
        "gru_weight": GRU_WEIGHT,
        "direction_threshold": DIRECTION_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "rows": result.to_dict(
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
        result.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("LIVE ENSEMBLE INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Actionable signals: "
        f"{int(result['actionable'].sum())}"
        f"/{len(result)}"
    )

    print(
        f"CSV : {OUTPUT_CSV}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()