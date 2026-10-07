from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = (
    ROOT
    / "data"
    / "crypto_live.db"
)

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "live_features_latest.parquet"
)

FEATURE_COLUMNS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "feature_columns_v2.txt"
)

SCALER_FILE = (
    ROOT
    / "data"
    / "processed"
    / "sequence_v1"
    / "feature_scaler.joblib"
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

MODEL_VERSION = (
    "catboost_v1_gru_v1_ensemble_v1"
)

EXPECTED_SYMBOL_COUNT = 5
EXPECTED_FEATURE_COUNT = 61
EXPECTED_PREDICTIONS_PER_CANDLE = 5

# Operational monitoring thresholds.
MAX_DATA_AGE_SECONDS = 120.0
MAX_ALLOWED_BAD_INTERVALS = 0


# ============================================================
# DATABASE
# ============================================================

CREATE_MONITORING_TABLE = """
CREATE TABLE IF NOT EXISTS monitoring_metrics (
    metric_time_ms INTEGER NOT NULL,
    candle_open_time_ms INTEGER NOT NULL,

    data_age_seconds REAL NOT NULL,
    symbol_count INTEGER NOT NULL,
    prediction_count INTEGER NOT NULL,

    mean_inference_latency_ms REAL NOT NULL,
    max_inference_latency_ms REAL NOT NULL,

    mean_confidence REAL NOT NULL,
    max_confidence REAL NOT NULL,

    hold_count INTEGER NOT NULL,
    up_count INTEGER NOT NULL,
    down_count INTEGER NOT NULL,

    feature_abs_z_mean REAL NOT NULL,
    feature_abs_z_max REAL NOT NULL,
    feature_abs_z_gt3_pct REAL NOT NULL,

    data_freshness_status TEXT NOT NULL,
    sequence_continuity_status TEXT NOT NULL,
    prediction_completeness_status TEXT NOT NULL,

    model_version TEXT NOT NULL,

    PRIMARY KEY (
        candle_open_time_ms,
        model_version
    )
);
"""


def get_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    return sqlite3.connect(
        DB_PATH,
        timeout=30,
    )


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            CREATE_MONITORING_TABLE
        )
        connection.commit()


# ============================================================
# FEATURE CONTRACT
# ============================================================

def load_feature_columns() -> list[str]:

    if not FEATURE_COLUMNS_FILE.exists():
        raise FileNotFoundError(
            f"Feature list not found: "
            f"{FEATURE_COLUMNS_FILE}"
        )

    features = [
        line.strip()
        for line in FEATURE_COLUMNS_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"found {len(features)}"
        )

    return features


# ============================================================
# LATEST CANDLE
# ============================================================

def get_latest_common_candle() -> int:

    query = """
    SELECT
        symbol,
        MAX(open_time_ms) AS latest_open_time
    FROM market_klines_1m
    GROUP BY symbol
    """

    with get_connection() as connection:
        rows = connection.execute(
            query
        ).fetchall()

    if len(rows) != EXPECTED_SYMBOL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SYMBOL_COUNT} symbols, "
            f"found {len(rows)}"
        )

    timestamps = [
        int(row[1])
        for row in rows
        if row[1] is not None
    ]

    if len(timestamps) != EXPECTED_SYMBOL_COUNT:
        raise ValueError(
            "At least one symbol has no candle data."
        )

    if len(set(timestamps)) != 1:
        raise ValueError(
            "Symbols do not share the same latest candle."
        )

    return timestamps[0]



def get_latest_prediction_candle() -> int | None:
    """Return the latest candle for which predictions were stored."""

    query = """
    SELECT MAX(candle_open_time_ms)
    FROM live_predictions
    WHERE model_version = ?
    """

    with get_connection() as connection:
        result = connection.execute(
            query,
            (MODEL_VERSION,),
        ).fetchone()

    if result is None or result[0] is None:
        return None

    return int(result[0])

# ============================================================
# CONTINUITY
# ============================================================

def check_sequence_continuity(
    candle_open_time_ms: int,
) -> tuple[str, int]:

    query = """
    SELECT
        symbol,
        open_time_ms
    FROM market_klines_1m
    WHERE open_time_ms <= ?
    ORDER BY symbol, open_time_ms DESC
    """

    with get_connection() as connection:
        rows = connection.execute(
            query,
            (candle_open_time_ms,),
        ).fetchall()

    by_symbol: dict[str, list[int]] = {
        symbol: []
        for symbol in SYMBOLS
    }

    for symbol, timestamp in rows:
        if (
            symbol in by_symbol
            and len(by_symbol[symbol]) < 61
        ):
            by_symbol[symbol].append(
                int(timestamp)
            )

    bad_intervals = 0

    for symbol in SYMBOLS:

        timestamps = list(
            reversed(
                by_symbol[symbol]
            )
        )

        if len(timestamps) < 61:
            bad_intervals += 1
            continue

        diffs = np.diff(
            np.asarray(
                timestamps,
                dtype=np.int64,
            )
        )

        bad = int(
            np.sum(
                diffs != 60_000
            )
        )

        bad_intervals += bad

    status = (
        "PASS"
        if bad_intervals
        <= MAX_ALLOWED_BAD_INTERVALS
        else "FAIL"
    )

    return status, bad_intervals


# ============================================================
# PREDICTION METRICS
# ============================================================

def load_latest_predictions(
    candle_open_time_ms: int,
) -> pd.DataFrame:

    query = """
    SELECT
        symbol,
        ensemble_confidence,
        signal,
        inference_latency_ms
    FROM live_predictions
    WHERE candle_open_time_ms = ?
      AND model_version = ?
    ORDER BY symbol
    """

    with get_connection() as connection:
        df = pd.read_sql_query(
            query,
            connection,
            params=(
                candle_open_time_ms,
                MODEL_VERSION,
            ),
        )

    return df


# ============================================================
# FEATURE ANOMALY MONITOR
# ============================================================

def calculate_feature_zscores(
    feature_columns: list[str],
) -> tuple[
    float,
    float,
    float,
]:

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Live feature file not found: "
            f"{FEATURE_FILE}"
        )

    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Training scaler not found: "
            f"{SCALER_FILE}"
        )

    df = pd.read_parquet(
        FEATURE_FILE
    )

    missing = [
        feature
        for feature in feature_columns
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Live feature file missing: {missing}"
        )

    X = df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X).all():
        raise ValueError(
            "Live features contain non-finite values."
        )

    scaler = joblib.load(
        SCALER_FILE
    )

    # Use the exact training-fitted scaler.
    Z = scaler.transform(X)

    absolute_z = np.abs(Z)

    mean_abs_z = float(
        absolute_z.mean()
    )

    max_abs_z = float(
        absolute_z.max()
    )

    gt3_pct = float(
        np.mean(
            absolute_z > 3.0
        )
        * 100.0
    )

    return (
        mean_abs_z,
        max_abs_z,
        gt3_pct,
    )


# ============================================================
# STORE MONITORING SNAPSHOT
# ============================================================

def store_metrics(
    candle_open_time_ms: int,
    data_age_seconds: float,
    symbol_count: int,
    prediction_count: int,
    mean_latency: float,
    max_latency: float,
    mean_confidence: float,
    max_confidence: float,
    hold_count: int,
    up_count: int,
    down_count: int,
    mean_abs_z: float,
    max_abs_z: float,
    gt3_pct: float,
    freshness_status: str,
    continuity_status: str,
    completeness_status: str,
) -> None:

    sql = """
    INSERT OR REPLACE INTO monitoring_metrics (
        metric_time_ms,
        candle_open_time_ms,
        data_age_seconds,
        symbol_count,
        prediction_count,
        mean_inference_latency_ms,
        max_inference_latency_ms,
        mean_confidence,
        max_confidence,
        hold_count,
        up_count,
        down_count,
        feature_abs_z_mean,
        feature_abs_z_max,
        feature_abs_z_gt3_pct,
        data_freshness_status,
        sequence_continuity_status,
        prediction_completeness_status,
        model_version
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    now_ms = int(
        time.time() * 1000
    )

    with get_connection() as connection:
        connection.execute(
            sql,
            (
                now_ms,
                candle_open_time_ms,
                data_age_seconds,
                symbol_count,
                prediction_count,
                mean_latency,
                max_latency,
                mean_confidence,
                max_confidence,
                hold_count,
                up_count,
                down_count,
                mean_abs_z,
                max_abs_z,
                gt3_pct,
                freshness_status,
                continuity_status,
                completeness_status,
                MODEL_VERSION,
            ),
        )

        connection.commit()


# ============================================================
# MAIN MONITOR
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CRYPTO MLOPS - LIVE MONITORING")
    print("=" * 70)

    initialize_database()

    feature_columns = (
        load_feature_columns()
    )

    latest_market_candle_ms = (
        get_latest_common_candle()
    )

    latest_prediction_candle_ms = (
        get_latest_prediction_candle()
    )

    if latest_prediction_candle_ms is None:
        raise ValueError(
            "No prediction records exist yet. "
            "Run the live inference runner first."
        )

    # Monitor the latest candle that actually has
    # a complete prediction set.
    candle_open_time_ms = (
        latest_prediction_candle_ms
    )

    now_ms = int(
        time.time() * 1000
    )

    # Market-data freshness is based on the newest
    # completed market candle, not the prediction candle.
    candle_close_estimate_ms = (
        latest_market_candle_ms
        + 60_000
    )

    data_age_seconds = max(
        0.0,
        (
            now_ms
            - candle_close_estimate_ms
        ) / 1000.0,
    )

    prediction_lag_seconds = max(
        0.0,
        (
            latest_market_candle_ms
            - candle_open_time_ms
        ) / 1000.0,
    )

    predictions = (
        load_latest_predictions(
            candle_open_time_ms
        )
    )

    symbol_count = (
        predictions["symbol"]
        .nunique()
        if not predictions.empty
        else 0
    )

    prediction_count = len(
        predictions
    )

    completeness_status = (
        "PASS"
        if prediction_count
        == EXPECTED_PREDICTIONS_PER_CANDLE
        and symbol_count
        == EXPECTED_SYMBOL_COUNT
        else "FAIL"
    )

    if prediction_count:

        mean_latency = float(
            predictions[
                "inference_latency_ms"
            ].mean()
        )

        max_latency = float(
            predictions[
                "inference_latency_ms"
            ].max()
        )

        mean_confidence = float(
            predictions[
                "ensemble_confidence"
            ].mean()
        )

        max_confidence = float(
            predictions[
                "ensemble_confidence"
            ].max()
        )

        signal_counts = (
            predictions["signal"]
            .value_counts()
        )

        hold_count = int(
            signal_counts.get(
                "HOLD",
                0,
            )
        )

        up_count = int(
            signal_counts.get(
                "UP",
                0,
            )
        )

        down_count = int(
            signal_counts.get(
                "DOWN",
                0,
            )
        )

    else:

        mean_latency = 0.0
        max_latency = 0.0
        mean_confidence = 0.0
        max_confidence = 0.0
        hold_count = 0
        up_count = 0
        down_count = 0

    continuity_status, bad_intervals = (
        check_sequence_continuity(
            candle_open_time_ms
        )
    )

    (
        mean_abs_z,
        max_abs_z,
        gt3_pct,
    ) = calculate_feature_zscores(
        feature_columns
    )

    freshness_status = (
        "PASS"
        if data_age_seconds
        <= MAX_DATA_AGE_SECONDS
        else "WARN"
    )

    store_metrics(
        candle_open_time_ms=(
            candle_open_time_ms
        ),
        data_age_seconds=data_age_seconds,
        symbol_count=symbol_count,
        prediction_count=prediction_count,
        mean_latency=mean_latency,
        max_latency=max_latency,
        mean_confidence=mean_confidence,
        max_confidence=max_confidence,
        hold_count=hold_count,
        up_count=up_count,
        down_count=down_count,
        mean_abs_z=mean_abs_z,
        max_abs_z=max_abs_z,
        gt3_pct=gt3_pct,
        freshness_status=freshness_status,
        continuity_status=continuity_status,
        completeness_status=completeness_status,
    )

    timestamp = pd.to_datetime(
        candle_open_time_ms,
        unit="ms",
        utc=True,
    )

    print()
    print(
        f"Candle: {timestamp}"
    )

    print(
        f"Market -> prediction lag: "
        f"{prediction_lag_seconds:.0f} sec"
    )

    print()
    print("DATA:")
    print(
        f"  Age: {data_age_seconds:.2f} sec"
    )
    print(
        f"  Symbols: {symbol_count}/5"
    )
    print(
        f"  Sequence continuity: "
        f"{continuity_status}"
    )
    print(
        f"  Bad intervals: "
        f"{bad_intervals}"
    )

    print()
    print("PREDICTIONS:")
    print(
        f"  Rows: {prediction_count}/5"
    )
    print(
        f"  Mean latency: "
        f"{mean_latency:.2f} ms"
    )
    print(
        f"  Max latency: "
        f"{max_latency:.2f} ms"
    )
    print(
        f"  Mean confidence: "
        f"{mean_confidence:.4f}"
    )
    print(
        f"  Max confidence: "
        f"{max_confidence:.4f}"
    )

    print()
    print("SIGNALS:")
    print(
        f"  HOLD: {hold_count}"
    )
    print(
        f"  UP:   {up_count}"
    )
    print(
        f"  DOWN: {down_count}"
    )

    print()
    print("FEATURE Z-SCORE MONITOR:")
    print(
        f"  Mean |z|: "
        f"{mean_abs_z:.4f}"
    )
    print(
        f"  Max |z|: "
        f"{max_abs_z:.4f}"
    )
    print(
        f"  |z| > 3: "
        f"{gt3_pct:.2f}%"
    )

    print()
    print("STATUS:")
    print(
        f"  Freshness: "
        f"{freshness_status}"
    )
    print(
        f"  Prediction completeness: "
        f"{completeness_status}"
    )
    print(
        f"  Sequence continuity: "
        f"{continuity_status}"
    )

    print()
    print(
        "Monitoring snapshot stored in SQLite."
    )


if __name__ == "__main__":
    main()