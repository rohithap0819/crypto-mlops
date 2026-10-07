from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PYTHON = sys.executable

DB_PATH = (
    ROOT
    / "data"
    / "crypto_live.db"
)

ENSEMBLE_FILE = (
    ROOT
    / "reports"
    / "live_ensemble_predictions_v1.csv"
)

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "live_features_latest.parquet"
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

POLL_SECONDS = 5

MODEL_VERSION = (
    "catboost_v1_gru_v1_ensemble_v1"
)

CATBOOST_WEIGHT = 0.55
GRU_WEIGHT = 0.45
CONFIDENCE_THRESHOLD = 0.55


# ============================================================
# PIPELINE STEPS
# ============================================================

PIPELINE_COMMANDS = [
    [
        str(PYTHON),
        str(
            ROOT
            / "src"
            / "features"
            / "live_feature_builder.py"
        ),
    ],
    [
        str(PYTHON),
        str(
            ROOT
            / "src"
            / "models"
            / "build_live_gru_sequence.py"
        ),
    ],
    [
        str(PYTHON),
        str(
            ROOT
            / "src"
            / "models"
            / "predict_live_gru.py"
        ),
    ],
    [
        str(PYTHON),
        str(
            ROOT
            / "src"
            / "models"
            / "predict_live_catboost.py"
        ),
    ],
    [
        str(PYTHON),
        str(
            ROOT
            / "src"
            / "models"
            / "predict_live_ensemble.py"
        ),
    ],
]


# ============================================================
# DATABASE
# ============================================================

CREATE_PREDICTION_TABLE = """
CREATE TABLE IF NOT EXISTS live_predictions (
    symbol TEXT NOT NULL,
    candle_open_time_ms INTEGER NOT NULL,

    prediction_generated_at_ms INTEGER NOT NULL,

    close_price REAL NOT NULL,

    catboost_probability_up REAL NOT NULL,
    gru_probability_up REAL NOT NULL,

    ensemble_probability_up REAL NOT NULL,
    ensemble_probability_down REAL NOT NULL,
    ensemble_confidence REAL NOT NULL,

    raw_prediction TEXT NOT NULL,
    signal TEXT NOT NULL,
    actionable INTEGER NOT NULL,

    inference_latency_ms REAL NOT NULL,

    model_version TEXT NOT NULL,
    catboost_weight REAL NOT NULL,
    gru_weight REAL NOT NULL,
    confidence_threshold REAL NOT NULL,

    PRIMARY KEY (
        symbol,
        candle_open_time_ms,
        model_version
    )
);
"""


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(
        DB_PATH,
        timeout=30,
    )


def initialize_prediction_table() -> None:

    with get_connection() as connection:

        connection.execute(
            CREATE_PREDICTION_TABLE
        )

        connection.commit()


# ============================================================
# LATEST COMMON CANDLE
# ============================================================

def get_latest_common_timestamp() -> int | None:

    query = """
    SELECT
        symbol,
        MAX(open_time_ms) AS latest_time
    FROM market_klines_1m
    GROUP BY symbol
    """

    with get_connection() as connection:

        rows = connection.execute(
            query
        ).fetchall()

    if len(rows) != len(SYMBOLS):
        return None

    timestamps = [
        int(row[1])
        for row in rows
        if row[1] is not None
    ]

    if len(timestamps) != len(SYMBOLS):
        return None

    # Only run when every symbol has reached
    # the same completed candle.
    if len(set(timestamps)) != 1:
        return None

    return timestamps[0]


# ============================================================
# ALREADY PROCESSED
# ============================================================

def has_processed_timestamp(
    candle_open_time_ms: int,
) -> bool:

    query = """
    SELECT COUNT(*)
    FROM live_predictions
    WHERE candle_open_time_ms = ?
      AND model_version = ?
    """

    with get_connection() as connection:

        count = connection.execute(
            query,
            (
                candle_open_time_ms,
                MODEL_VERSION,
            ),
        ).fetchone()[0]

    return count > 0


# ============================================================
# RUN PIPELINE
# ============================================================

def run_command(
    command: list[str],
) -> None:

    print()
    print(
        "RUN:",
        " ".join(command),
    )

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:

        if result.stderr:
            print(
                result.stderr,
                file=sys.stderr,
            )

        raise RuntimeError(
            f"Pipeline step failed with exit code "
            f"{result.returncode}: {command}"
        )


def run_prediction_pipeline() -> float:

    started = time.perf_counter()

    for command in PIPELINE_COMMANDS:
        run_command(command)

    elapsed_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    return elapsed_ms


# ============================================================
# READ ENSEMBLE OUTPUT
# ============================================================

def load_ensemble_output() -> pd.DataFrame:

    if not ENSEMBLE_FILE.exists():
        raise FileNotFoundError(
            f"Ensemble output not found: "
            f"{ENSEMBLE_FILE}"
        )

    df = pd.read_csv(
        ENSEMBLE_FILE
    )

    required = [
        "symbol",
        "open_time",
        "ensemble_probability_up",
        "ensemble_probability_down",
        "ensemble_confidence",
        "raw_prediction",
        "actionable",
        "signal",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Ensemble output missing columns: "
            f"{missing}"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    return df


# ============================================================
# READ CATBOOST + GRU OUTPUT
# ============================================================

def load_model_outputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    catboost_file = (
        ROOT
        / "reports"
        / "live_catboost_predictions_v1.csv"
    )

    gru_file = (
        ROOT
        / "reports"
        / "live_gru_predictions_v1.csv"
    )

    catboost = pd.read_csv(
        catboost_file
    )

    gru = pd.read_csv(
        gru_file
    )

    catboost["open_time"] = pd.to_datetime(
        catboost["open_time"],
        utc=True,
    )

    gru["open_time"] = pd.to_datetime(
        gru["open_time"],
        utc=True,
    )

    return (
        catboost,
        gru,
    )


# ============================================================
# LOAD CLOSE PRICES
# ============================================================

def load_latest_prices() -> pd.DataFrame:

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Live feature file not found: "
            f"{FEATURE_FILE}"
        )

    df = pd.read_parquet(
        FEATURE_FILE,
        columns=[
            "symbol",
            "open_time",
            "close",
        ],
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    return df


# ============================================================
# STORE PREDICTIONS
# ============================================================

def store_predictions(
    ensemble: pd.DataFrame,
    catboost: pd.DataFrame,
    gru: pd.DataFrame,
    prices: pd.DataFrame,
    latency_ms: float,
) -> None:

    merged = (
        ensemble
        .merge(
            catboost[
                [
                    "symbol",
                    "open_time",
                    "probability_up",
                ]
            ],
            on=[
                "symbol",
                "open_time",
            ],
            how="inner",
            validate="one_to_one",
        )
        .rename(
            columns={
                "probability_up":
                    "catboost_probability_up"
            }
        )
        .merge(
            gru[
                [
                    "symbol",
                    "open_time",
                    "probability_up",
                ]
            ],
            on=[
                "symbol",
                "open_time",
            ],
            how="inner",
            validate="one_to_one",
        )
        .rename(
            columns={
                "probability_up":
                    "gru_probability_up"
            }
        )
        .merge(
            prices,
            on=[
                "symbol",
                "open_time",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    if len(merged) != len(SYMBOLS):
        raise ValueError(
            "Expected predictions for all five symbols. "
            f"Received {len(merged)} rows."
        )

    generated_at_ms = int(
        time.time() * 1000
    )

    rows = []

    for row in merged.itertuples(
        index=False
    ):

        timestamp = pd.Timestamp(
            row.open_time
        )

        candle_open_time_ms = int(
            timestamp.timestamp() * 1000
        )

        rows.append(
            (
                row.symbol,
                candle_open_time_ms,
                generated_at_ms,

                float(row.close),

                float(
                    row.catboost_probability_up
                ),

                float(
                    row.gru_probability_up
                ),

                float(
                    row.ensemble_probability_up
                ),

                float(
                    row.ensemble_probability_down
                ),

                float(
                    row.ensemble_confidence
                ),

                str(
                    row.raw_prediction
                ),

                str(
                    row.signal
                ),

                int(
                    bool(row.actionable)
                ),

                float(latency_ms),

                MODEL_VERSION,
                CATBOOST_WEIGHT,
                GRU_WEIGHT,
                CONFIDENCE_THRESHOLD,
            )
        )

    insert_sql = """
    INSERT OR REPLACE INTO live_predictions (
        symbol,
        candle_open_time_ms,
        prediction_generated_at_ms,
        close_price,
        catboost_probability_up,
        gru_probability_up,
        ensemble_probability_up,
        ensemble_probability_down,
        ensemble_confidence,
        raw_prediction,
        signal,
        actionable,
        inference_latency_ms,
        model_version,
        catboost_weight,
        gru_weight,
        confidence_threshold
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    with get_connection() as connection:

        connection.executemany(
            insert_sql,
            rows,
        )

        connection.commit()


# ============================================================
# DISPLAY
# ============================================================

def display_predictions(
    ensemble: pd.DataFrame,
    latency_ms: float,
) -> None:

    print()
    print("=" * 70)
    print("LIVE ENSEMBLE RESULT")
    print("=" * 70)

    print(
        ensemble[
            [
                "symbol",
                "ensemble_probability_up",
                "ensemble_confidence",
                "raw_prediction",
                "signal",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Inference latency: "
        f"{latency_ms:.2f} ms"
    )

    print()


# ============================================================
# MAIN LOOP
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CRYPTO MLOPS - LIVE INFERENCE RUNNER")
    print("=" * 70)

    print()
    print(
        f"Database: {DB_PATH}"
    )

    print(
        f"Model version: {MODEL_VERSION}"
    )

    initialize_prediction_table()

    processed_timestamp = None

    print()
    print(
        "Waiting for a new common 1-minute candle..."
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

                    readable_time = datetime.fromtimestamp(
                        latest_timestamp / 1000,
                        tz=timezone.utc,
                    )

                    print()
                    print(
                        f"New candle detected: "
                        f"{readable_time.isoformat()}"
                    )

                    latency_ms = (
                        run_prediction_pipeline()
                    )

                    ensemble = (
                        load_ensemble_output()
                    )

                    catboost, gru = (
                        load_model_outputs()
                    )

                    prices = (
                        load_latest_prices()
                    )

                    store_predictions(
                        ensemble=ensemble,
                        catboost=catboost,
                        gru=gru,
                        prices=prices,
                        latency_ms=latency_ms,
                    )

                    display_predictions(
                        ensemble,
                        latency_ms,
                    )

                    processed_timestamp = (
                        latest_timestamp
                    )

            except Exception:

                print()
                print(
                    "Inference cycle failed:",
                    file=sys.stderr,
                )

                import traceback

                traceback.print_exc()

                # Do not terminate the long-running
                # inference process because one cycle failed.

                time.sleep(
                    POLL_SECONDS
                )

            time.sleep(
                POLL_SECONDS
            )

    except KeyboardInterrupt:

        print()
        print(
            "Live inference runner stopped."
        )


if __name__ == "__main__":
    main()