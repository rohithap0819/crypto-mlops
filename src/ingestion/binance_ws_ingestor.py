from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import websocket


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "crypto_live.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)


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

INTERVAL = "1m"

# Binance Spot combined-stream endpoint.
STREAM_NAMES = [
    f"{symbol.lower()}@kline_{INTERVAL}"
    for symbol in SYMBOLS
]

WS_URL = (
    "wss://stream.binance.com:9443/stream?streams="
    + "/".join(STREAM_NAMES)
)

RECONNECT_DELAY_SECONDS = 5


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("binance_ingestor")


# ============================================================
# DATABASE
# ============================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_klines_1m (
    symbol TEXT NOT NULL,
    open_time_ms INTEGER NOT NULL,
    close_time_ms INTEGER NOT NULL,

    open_price REAL NOT NULL,
    high_price REAL NOT NULL,
    low_price REAL NOT NULL,
    close_price REAL NOT NULL,

    volume REAL NOT NULL,
    quote_volume REAL NOT NULL,

    trade_count INTEGER NOT NULL,
    taker_buy_volume REAL NOT NULL,
    taker_buy_quote_volume REAL NOT NULL,

    event_time_ms INTEGER NOT NULL,
    received_at_ms INTEGER NOT NULL,

    PRIMARY KEY (symbol, open_time_ms)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_market_klines_symbol_time
ON market_klines_1m(symbol, open_time_ms);
"""


def get_connection() -> sqlite3.Connection:
    """Create and configure the SQLite connection."""

    connection = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )

    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA busy_timeout=30000;")

    return connection


def initialize_database() -> None:
    """Create SQLite tables and indexes."""

    with get_connection() as connection:
        connection.execute(CREATE_TABLE_SQL)
        connection.execute(CREATE_INDEX_SQL)
        connection.commit()

    logger.info("SQLite database ready: %s", DB_PATH)


def save_closed_kline(payload: dict[str, Any]) -> None:
    """
    Persist one closed 1-minute kline.

    Binance sends the kline repeatedly while the candle is forming.
    We only persist the final closed candle where k.x == True.
    """

    kline = payload["k"]

    if not kline["x"]:
        return

    symbol = kline["s"].upper()

    row = (
        symbol,
        int(kline["t"]),
        int(kline["T"]),
        float(kline["o"]),
        float(kline["h"]),
        float(kline["l"]),
        float(kline["c"]),
        float(kline["v"]),
        float(kline["q"]),
        int(kline["n"]),
        float(kline["V"]),
        float(kline["Q"]),
        int(payload["E"]),
        int(time.time() * 1000),
    )

    sql = """
    INSERT INTO market_klines_1m (
        symbol,
        open_time_ms,
        close_time_ms,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
        quote_volume,
        trade_count,
        taker_buy_volume,
        taker_buy_quote_volume,
        event_time_ms,
        received_at_ms
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(symbol, open_time_ms)
    DO UPDATE SET
        close_time_ms = excluded.close_time_ms,
        open_price = excluded.open_price,
        high_price = excluded.high_price,
        low_price = excluded.low_price,
        close_price = excluded.close_price,
        volume = excluded.volume,
        quote_volume = excluded.quote_volume,
        trade_count = excluded.trade_count,
        taker_buy_volume = excluded.taker_buy_volume,
        taker_buy_quote_volume = excluded.taker_buy_quote_volume,
        event_time_ms = excluded.event_time_ms,
        received_at_ms = excluded.received_at_ms;
    """

    with get_connection() as connection:
        connection.execute(sql, row)
        connection.commit()

    logger.info(
        "CLOSED | %s | %s | O=%.8f H=%.8f L=%.8f C=%.8f V=%.8f",
        symbol,
        kline["t"],
        float(kline["o"]),
        float(kline["h"]),
        float(kline["l"]),
        float(kline["c"]),
        float(kline["v"]),
    )


# ============================================================
# WEBSOCKET CALLBACKS
# ============================================================

def on_open(ws: websocket.WebSocketApp) -> None:
    logger.info("Connected to Binance WebSocket")
    logger.info("Streams:")
    for stream in STREAM_NAMES:
        logger.info("  %s", stream)


def on_message(
    ws: websocket.WebSocketApp,
    message: str,
) -> None:

    try:
        event = json.loads(message)

        # Combined-stream response:
        # {
        #   "stream": "...",
        #   "data": {...}
        # }

        payload = event.get("data")

        if not payload:
            return

        if payload.get("e") != "kline":
            return

        save_closed_kline(payload)

    except json.JSONDecodeError:
        logger.exception("Failed to decode WebSocket message")

    except Exception:
        logger.exception("Error processing WebSocket message")


def on_error(
    ws: websocket.WebSocketApp,
    error: Any,
) -> None:
    logger.error("WebSocket error: %s", error)


def on_close(
    ws: websocket.WebSocketApp,
    close_status_code: Any,
    close_msg: Any,
) -> None:
    logger.warning(
        "WebSocket closed | code=%s | message=%s",
        close_status_code,
        close_msg,
    )


# ============================================================
# RUNNER
# ============================================================

def run() -> None:

    initialize_database()

    logger.info("Starting Binance live ingestion")
    logger.info("Database: %s", DB_PATH)

    while True:

        ws_app = websocket.WebSocketApp(
            WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        try:
            ws_app.run_forever(
                ping_interval=30,
                ping_timeout=10,
                ping_payload="crypto-mlops",
            )

        except KeyboardInterrupt:
            logger.info("Stopping ingestion...")
            break

        except Exception:
            logger.exception("Unexpected WebSocket failure")

        logger.info(
            "Reconnecting in %s seconds...",
            RECONNECT_DELAY_SECONDS,
        )

        time.sleep(RECONNECT_DELAY_SECONDS)


if __name__ == "__main__":
    run()