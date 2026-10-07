from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    ROOT
    / "models"
    / "final"
    / "binary_direction_catboost_v1.cbm"
)

LIVE_FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "live_features_latest.parquet"
)

REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    REPORTS_DIR
    / "live_catboost_predictions_v1.csv"
)

OUTPUT_JSON = (
    REPORTS_DIR
    / "live_catboost_predictions_v1.json"
)


# ============================================================
# FROZEN SYMBOL CONTRACT
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

SYMBOL_FEATURES = [
    "symbol_BTCUSDT",
    "symbol_ETHUSDT",
    "symbol_SOLUSDT",
    "symbol_BNBUSDT",
    "symbol_XRPUSDT",
]

EXPECTED_V2_FEATURE_COUNT = 61
EXPECTED_MODEL_FEATURE_COUNT = 66


# ============================================================
# LOAD MODEL
# ============================================================

def load_model() -> CatBoostClassifier:
    """Load the frozen CatBoost binary classifier."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"CatBoost model not found: {MODEL_FILE}"
        )

    model = CatBoostClassifier()

    model.load_model(
        str(MODEL_FILE)
    )

    feature_names = model.feature_names_

    if len(feature_names) != (
        EXPECTED_MODEL_FEATURE_COUNT
    ):
        raise ValueError(
            "Unexpected CatBoost feature count: "
            f"{len(feature_names)}; "
            f"expected {EXPECTED_MODEL_FEATURE_COUNT}"
        )

    return model


# ============================================================
# LOAD LIVE FEATURES
# ============================================================

def load_live_features() -> pd.DataFrame:
    """Load the latest V2 feature rows."""

    if not LIVE_FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Live feature file not found: "
            f"{LIVE_FEATURE_FILE}"
        )

    df = pd.read_parquet(
        LIVE_FEATURE_FILE
    )

    required_columns = [
        "symbol",
        "open_time",
        "close",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Live feature data is missing: {missing}"
        )

    return df


# ============================================================
# BUILD CATBOOST INPUT
# ============================================================

def build_model_input(
    model: CatBoostClassifier,
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Construct the exact 66-column CatBoost input.

    Returns:
        metadata_df
        model_input
    """

    # --------------------------------------------------------
    # Identify the exact 61 V2 features
    # --------------------------------------------------------

    excluded = {
        "symbol",
        "open_time",
        "close",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    if len(feature_columns) != (
        EXPECTED_V2_FEATURE_COUNT
    ):
        raise ValueError(
            "Unexpected live V2 feature count: "
            f"{len(feature_columns)}; "
            f"expected {EXPECTED_V2_FEATURE_COUNT}"
        )

    # --------------------------------------------------------
    # Check for missing / invalid values
    # --------------------------------------------------------

    if df[feature_columns].isna().any().any():
        raise ValueError(
            "Live V2 features contain NaN values."
        )

    if (
        ~df[feature_columns]
        .map(pd.api.types.is_number)
        .all()
        .all()
    ):
        raise ValueError(
            "Live V2 features contain non-numeric values."
        )

    # --------------------------------------------------------
    # Add explicit symbol one-hot columns
    # --------------------------------------------------------

    working = df.copy()

    for symbol in SYMBOLS:

        column = (
            f"symbol_{symbol}"
        )

        working[column] = (
            working["symbol"]
            == symbol
        ).astype(int)

    # --------------------------------------------------------
    # Validate symbol values
    # --------------------------------------------------------

    unknown_symbols = sorted(
        set(working["symbol"])
        - set(SYMBOLS)
    )

    if unknown_symbols:
        raise ValueError(
            f"Unknown symbols found: "
            f"{unknown_symbols}"
        )

    # --------------------------------------------------------
    # Use the model's exact feature order
    # --------------------------------------------------------

    model_feature_names = (
        model.feature_names_
    )

    missing_model_features = [
        feature
        for feature in model_feature_names
        if feature not in working.columns
    ]

    if missing_model_features:
        raise ValueError(
            "Missing CatBoost features:\n"
            + "\n".join(
                missing_model_features
            )
        )

    model_input = working[
        model_feature_names
    ].copy()

    # --------------------------------------------------------
    # Ensure everything is numeric
    # --------------------------------------------------------

    for column in model_feature_names:
        model_input[column] = pd.to_numeric(
            model_input[column],
            errors="coerce",
        )

    if model_input.isna().any().any():
        raise ValueError(
            "CatBoost input contains NaN values "
            "after numeric conversion."
        )

    # --------------------------------------------------------
    # Preserve metadata separately
    # --------------------------------------------------------

    metadata_df = working[
        [
            "symbol",
            "open_time",
            "close",
        ]
    ].copy()

    return (
        metadata_df,
        model_input,
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model: CatBoostClassifier,
    metadata_df: pd.DataFrame,
    model_input: pd.DataFrame,
) -> pd.DataFrame:
    """Generate CatBoost binary direction probabilities."""

    probabilities = model.predict_proba(
        model_input
    )

    # CatBoost binary classification returns:
    # column 0 = DOWN / class 0
    # column 1 = UP / class 1
    probability_down = probabilities[:, 0]
    probability_up = probabilities[:, 1]

    prediction = [
        "UP"
        if value >= 0.5
        else "DOWN"
        for value in probability_up
    ]

    confidence = (
        pd.Series(
            probability_up
        )
        .where(
            probability_up >= 0.5,
            1.0 - probability_up,
        )
        .to_numpy()
    )

    result = metadata_df.copy()

    result["prediction"] = prediction

    result["probability_down"] = (
        probability_down
    )

    result["probability_up"] = (
        probability_up
    )

    result["confidence"] = confidence

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CRYPTO MLOPS - LIVE CATBOOST INFERENCE")
    print("=" * 70)

    print(
        f"Model   : {MODEL_FILE}"
    )

    print(
        f"Features: {LIVE_FEATURE_FILE}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model = load_model()

    live_features = (
        load_live_features()
    )

    # --------------------------------------------------------
    # Build exact 66-column input
    # --------------------------------------------------------

    (
        metadata_df,
        model_input,
    ) = build_model_input(
        model=model,
        df=live_features,
    )

    print()
    print(
        f"Model input shape: "
        f"{model_input.shape}"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = predict(
        model=model,
        metadata_df=metadata_df,
        model_input=model_input,
    )

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
        "model": "CatBoostClassifier",
        "model_file": str(
            MODEL_FILE.relative_to(ROOT)
        ),
        "feature_version": "v2",
        "feature_count": EXPECTED_V2_FEATURE_COUNT,
        "model_feature_count": (
            EXPECTED_MODEL_FEATURE_COUNT
        ),
        "forecast_horizon_minutes": 5,
        "rows": predictions.assign(
            open_time=predictions["open_time"].astype(str)
        ).to_dict(
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
    print("CATBOOST INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"CSV : {OUTPUT_CSV}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()