from __future__ import annotations

import os

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Gauge,
    generate_latest,
)
from fastapi.responses import Response


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_PATH = Path(
    os.getenv(
        "CRYPTO_DB_PATH",
        str(
            PROJECT_ROOT
            / "data"
            / "crypto_live.db"
        ),
    )
)


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_SYMBOLS = {
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
}

SUPPORTED_INTERVALS = {
    "1m": "1min",
    "5m": "5min",
    "1h": "1h",
    "4h": "4h",
}

MODEL_VERSION = (
    "catboost_v1_gru_v1_ensemble_v1"
)



# ============================================================
# PROMETHEUS METRICS
# ============================================================

API_UP = Gauge(
    "crypto_api_up",
    "FastAPI application availability",
)

MARKET_ROWS = Gauge(
    "crypto_market_rows",
    "Number of market rows stored",
)

PREDICTION_ROWS = Gauge(
    "crypto_prediction_rows",
    "Number of prediction rows stored",
)

MONITORING_ROWS = Gauge(
    "crypto_monitoring_rows",
    "Number of monitoring metric rows stored",
)

MEAN_INFERENCE_LATENCY = Gauge(
    "crypto_mean_inference_latency_ms",
    "Latest mean inference latency in milliseconds",
)

MAX_INFERENCE_LATENCY = Gauge(
    "crypto_max_inference_latency_ms",
    "Latest maximum inference latency in milliseconds",
)

MEAN_CONFIDENCE = Gauge(
    "crypto_mean_confidence",
    "Latest mean prediction confidence",
)

FEATURE_DRIFT_GT3_PCT = Gauge(
    "crypto_feature_abs_z_gt3_pct",
    "Percentage of monitored features with absolute z-score greater than 3",
)

DATA_FRESHNESS = Gauge(
    "crypto_data_freshness_status",
    "Data freshness status: 1=PASS, 0=FAIL",
)

SEQUENCE_CONTINUITY = Gauge(
    "crypto_sequence_continuity_status",
    "Sequence continuity status: 1=PASS, 0=FAIL",
)

PREDICTION_COMPLETENESS = Gauge(
    "crypto_prediction_completeness_status",
    "Prediction completeness status: 1=PASS, 0=FAIL",
)

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Crypto MLOps API",
    description=(
        "Read-only API for live market data, "
        "production predictions and monitoring."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

def get_connection() -> sqlite3.Connection:

    if not DATABASE_PATH.exists():

        raise RuntimeError(
            f"SQLite database not found: "
            f"{DATABASE_PATH}"
        )

    connection = sqlite3.connect(
        str(DATABASE_PATH)
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATAFRAME JSON CLEANUP
# ============================================================

def dataframe_to_records(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:

    if df.empty:
        return []

    result = df.copy()

    for column in result.columns:

        if pd.api.types.is_datetime64_any_dtype(
            result[column]
        ):

            result[column] = (
                result[column]
                .dt.strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                )
            )

    result = result.where(
        pd.notna(result),
        None,
    )

    return result.to_dict(
        orient="records"
    )


# ============================================================
# MARKET DATA
# ============================================================

def load_latest_market() -> pd.DataFrame:

    connection = get_connection()

    try:

        query = """
            WITH latest AS (
                SELECT
                    symbol,
                    MAX(open_time_ms)
                    AS max_open_time_ms
                FROM market_klines_1m
                GROUP BY symbol
            )

            SELECT
                k.symbol,
                k.open_time_ms,
                k.open_price AS open,
                k.high_price AS high,
                k.low_price AS low,
                k.close_price AS close,
                k.volume
            FROM market_klines_1m k

            INNER JOIN latest l
                ON k.symbol = l.symbol
                AND k.open_time_ms =
                    l.max_open_time_ms

            ORDER BY k.symbol
        """

        df = pd.read_sql_query(
            query,
            connection,
        )

    finally:

        connection.close()

    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(
        df["open_time_ms"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    return df[
        [
            "symbol",
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


def load_market_history(
    symbol: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:

    if interval not in SUPPORTED_INTERVALS:

        raise ValueError(
            "Unsupported interval"
        )

    multiplier = {
        "1m": 1,
        "5m": 5,
        "1h": 60,
        "4h": 240,
    }[interval]

    # Extra rows provide enough warm-up for
    # resampling and technical indicators.
    raw_limit = (
        limit * multiplier
        + multiplier * 10
    )

    connection = get_connection()

    try:

        query = """
            SELECT
                symbol,
                open_time_ms,
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            FROM market_klines_1m
            WHERE symbol = ?
            ORDER BY open_time_ms DESC
            LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            connection,
            params=[
                symbol,
                raw_limit,
            ],
        )

    finally:

        connection.close()

    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(
        df["open_time_ms"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["open_time"]
    )

    for column in [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
        ]
    )

    df = df.sort_values(
        "open_time"
    )

    # --------------------------------------------------------
    # 1-minute
    # --------------------------------------------------------

    if interval == "1m":

        result = df.tail(
            limit
        ).copy()

        result["open"] = result[
            "open_price"
        ]

        result["high"] = result[
            "high_price"
        ]

        result["low"] = result[
            "low_price"
        ]

        result["close"] = result[
            "close_price"
        ]

        return result[
            [
                "symbol",
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]

    # --------------------------------------------------------
    # Higher timeframes
    # --------------------------------------------------------

    result = (
        df
        .set_index("open_time")
        .resample(
            SUPPORTED_INTERVALS[
                interval
            ],
            label="left",
            closed="left",
        )
        .agg(
            {
                "open_price": "first",
                "high_price": "max",
                "low_price": "min",
                "close_price": "last",
                "volume": "sum",
            }
        )
        .dropna(
            subset=[
                "open_price",
                "high_price",
                "low_price",
                "close_price",
            ]
        )
        .tail(limit)
        .reset_index()
    )

    result["symbol"] = symbol

    result = result.rename(
        columns={
            "open_price": "open",
            "high_price": "high",
            "low_price": "low",
            "close_price": "close",
        }
    )

    return result[
        [
            "symbol",
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


# ============================================================
# PREDICTIONS
# ============================================================

def load_latest_predictions() -> pd.DataFrame:

    connection = get_connection()

    try:

        query = """
            SELECT
                p.*
            FROM live_predictions p

            INNER JOIN (
                SELECT
                    MAX(candle_open_time_ms)
                    AS latest_candle
                FROM live_predictions
            ) latest

            ON p.candle_open_time_ms =
               latest.latest_candle

            ORDER BY p.symbol
        """

        df = pd.read_sql_query(
            query,
            connection,
        )

    finally:

        connection.close()

    if df.empty:
        return df

    df["candle_time"] = pd.to_datetime(
        df["candle_open_time_ms"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df["prediction_generated_at"] = (
        pd.to_datetime(
            df[
                "prediction_generated_at_ms"
            ],
            unit="ms",
            utc=True,
            errors="coerce",
        )
    )

    return df


def load_prediction_history(
    symbol: str | None,
    limit: int,
) -> pd.DataFrame:

    connection = get_connection()

    try:

        if symbol:

            query = """
                SELECT *
                FROM live_predictions
                WHERE symbol = ?
                ORDER BY candle_open_time_ms DESC
                LIMIT ?
            """

            params = [
                symbol,
                limit,
            ]

        else:

            query = """
                SELECT *
                FROM live_predictions
                ORDER BY candle_open_time_ms DESC
                LIMIT ?
            """

            params = [
                limit,
            ]

        df = pd.read_sql_query(
            query,
            connection,
            params=params,
        )

    finally:

        connection.close()

    if df.empty:
        return df

    df["candle_time"] = pd.to_datetime(
        df["candle_open_time_ms"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df["prediction_generated_at"] = (
        pd.to_datetime(
            df[
                "prediction_generated_at_ms"
            ],
            unit="ms",
            utc=True,
            errors="coerce",
        )
    )

    return df.sort_values(
        "candle_time"
    )


# ============================================================
# MONITORING
# ============================================================

def load_latest_monitoring() -> dict:

    connection = get_connection()

    try:

        query = """
            SELECT
                *
            FROM monitoring_metrics
            ORDER BY metric_time_ms DESC
            LIMIT 1
        """

        df = pd.read_sql_query(
            query,
            connection,
        )

    finally:

        connection.close()

    if df.empty:

        return {
            "data_freshness_status": "UNKNOWN",
            "sequence_continuity_status": "UNKNOWN",
            "prediction_completeness_status": (
                "UNKNOWN"
            ),
            "model_version": MODEL_VERSION,
        }

    row = df.iloc[0].to_dict()

    row["metric_time"] = pd.to_datetime(
        row["metric_time_ms"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    row["candle_time"] = pd.to_datetime(
        row["candle_open_time_ms"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    result: dict[str, Any] = {}

    for key, value in row.items():

        if isinstance(
            value,
            pd.Timestamp,
        ):

            result[key] = (
                value.isoformat()
            )

        elif hasattr(
            value,
            "item",
        ):

            try:
                result[key] = value.item()
            except Exception:
                result[key] = value

        else:

            result[key] = value

    result.setdefault(
        "model_version",
        MODEL_VERSION,
    )

    # The dashboard uses this name.
    if "feature_abs_z_max" not in result:

        result[
            "feature_abs_z_max"
        ] = result.get(
            "feature_abs_z_max",
            result.get(
                "feature_abs_z_max",
                0,
            ),
        )

    return result


# ============================================================
# PROMETHEUS
# ============================================================

@app.get("/metrics")
def metrics():
    connection = get_connection()

    try:
        market_count = connection.execute(
            "SELECT COUNT(*) FROM market_klines_1m"
        ).fetchone()[0]

        prediction_count = connection.execute(
            "SELECT COUNT(*) FROM live_predictions"
        ).fetchone()[0]

        monitoring_count = connection.execute(
            "SELECT COUNT(*) FROM monitoring_metrics"
        ).fetchone()[0]

        latest_monitoring = connection.execute(
            """
            SELECT
                mean_inference_latency_ms,
                max_inference_latency_ms,
                mean_confidence,
                feature_abs_z_gt3_pct,
                data_freshness_status,
                sequence_continuity_status,
                prediction_completeness_status
            FROM monitoring_metrics
            ORDER BY metric_time_ms DESC
            LIMIT 1
            """
        ).fetchone()

    finally:
        connection.close()

    API_UP.set(1)

    MARKET_ROWS.set(market_count)
    PREDICTION_ROWS.set(prediction_count)
    MONITORING_ROWS.set(monitoring_count)

    if latest_monitoring:

        MEAN_INFERENCE_LATENCY.set(
            latest_monitoring["mean_inference_latency_ms"]
            or 0
        )

        MAX_INFERENCE_LATENCY.set(
            latest_monitoring["max_inference_latency_ms"]
            or 0
        )

        MEAN_CONFIDENCE.set(
            latest_monitoring["mean_confidence"]
            or 0
        )

        FEATURE_DRIFT_GT3_PCT.set(
            latest_monitoring["feature_abs_z_gt3_pct"]
            or 0
        )

        DATA_FRESHNESS.set(
            1
            if latest_monitoring["data_freshness_status"]
            == "PASS"
            else 0
        )

        SEQUENCE_CONTINUITY.set(
            1
            if latest_monitoring["sequence_continuity_status"]
            == "PASS"
            else 0
        )

        PREDICTION_COMPLETENESS.set(
            1
            if latest_monitoring[
                "prediction_completeness_status"
            ]
            == "PASS"
            else 0
        )

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Crypto MLOps API",
        "status": "running",
        "model_version": MODEL_VERSION,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    database_exists = (
        DATABASE_PATH.exists()
    )

    return {
        "status": (
            "healthy"
            if database_exists
            else "degraded"
        ),
        "database": database_exists,
        "model_version": MODEL_VERSION,
    }


# ============================================================
# MARKET LATEST
# ============================================================

@app.get("/market/latest")
def market_latest():

    try:

        df = load_latest_market()

        return dataframe_to_records(
            df
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# MARKET HISTORY
# ============================================================

@app.get("/market/history")
def market_history(
    symbol: str = Query(
        ...
    ),
    interval: str = Query(
        "1m"
    ),
    limit: int = Query(
        500,
        ge=1,
        le=5000,
    ),
):

    symbol = symbol.upper()
    interval = interval.lower()

    if symbol not in SUPPORTED_SYMBOLS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported symbol: "
                f"{symbol}"
            ),
        )

    if interval not in SUPPORTED_INTERVALS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported interval. "
                "Use 1m, 5m, 1h or 4h."
            ),
        )

    try:

        df = load_market_history(
            symbol,
            interval,
            limit,
        )

        return dataframe_to_records(
            df
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# PREDICTIONS LATEST
# ============================================================

@app.get("/predictions/latest")
def predictions_latest():

    try:

        df = load_latest_predictions()

        return dataframe_to_records(
            df
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# PREDICTIONS HISTORY
# ============================================================

@app.get("/predictions/history")
def predictions_history(
    symbol: str | None = Query(
        None
    ),
    limit: int = Query(
        200,
        ge=1,
        le=5000,
    ),
):

    if symbol:

        symbol = symbol.upper()

        if symbol not in SUPPORTED_SYMBOLS:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported symbol: "
                    f"{symbol}"
                ),
            )

    try:

        df = load_prediction_history(
            symbol,
            limit,
        )

        return dataframe_to_records(
            df
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# MONITORING LATEST
# ============================================================

@app.get("/monitoring/latest")
def monitoring_latest():

    try:

        return load_latest_monitoring()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# MONITORING HISTORY
# ============================================================

@app.get("/monitoring/history")
def monitoring_history(
    limit: int = Query(
        100,
        ge=1,
        le=5000,
    ),
):

    connection = get_connection()

    try:

        query = """
            SELECT *
            FROM monitoring_metrics
            ORDER BY metric_time_ms DESC
            LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            connection,
            params=[limit],
        )

    finally:

        connection.close()

    if df.empty:
        return []

    df["metric_time"] = pd.to_datetime(
        df["metric_time_ms"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df["candle_time"] = pd.to_datetime(
        df["candle_open_time_ms"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    return dataframe_to_records(
        df
    )