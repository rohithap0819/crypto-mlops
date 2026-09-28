import json
import time
from datetime import datetime, timezone

import websocket

from src.storage.market_store import MarketStore


SYMBOLS = [
    "btcusdt",
    "ethusdt",
    "solusdt",
    "bnbusdt",
    "xrpusdt",
]

INTERVAL = "1m"

STREAMS = "/".join(
    f"{symbol}@kline_{INTERVAL}"
    for symbol in SYMBOLS
)

WS_URL = (
    "wss://stream.binance.com:9443/stream"
    f"?streams={STREAMS}"
)

STORE = MarketStore()


def on_message(ws, message):
    try:
        payload = json.loads(message)
        data = payload.get("data", {})

        if data.get("e") != "kline":
            return

        candle = data.get("k", {})

        symbol = candle.get("s")
        interval = candle.get("i")
        open_time_ms = candle.get("t")
        close_time_ms = candle.get("T")

        if not all(
            value is not None
            for value in [
                symbol,
                interval,
                open_time_ms,
                close_time_ms,
            ]
        ):
            return

        received_at = datetime.now(timezone.utc).isoformat()

        STORE.insert_candle(
            symbol=symbol,
            interval=interval,
            open_time_ms=int(open_time_ms),
            close_time_ms=int(close_time_ms),
            open_price=float(candle["o"]),
            high_price=float(candle["h"]),
            low_price=float(candle["l"]),
            close_price=float(candle["c"]),
            volume=float(candle["v"]),
            quote_volume=float(candle["q"]),
            trade_count=int(candle["n"]),
            taker_buy_volume=float(candle["V"]),
            taker_buy_quote_volume=float(candle["Q"]),
            is_closed=bool(candle["x"]),
            received_at_utc=received_at,
        )

        state = "CLOSED" if candle["x"] else "OPEN"

        print(
            f"{received_at} | "
            f"{symbol} | "
            f"{interval} | "
            f"O={candle['o']} | "
            f"H={candle['h']} | "
            f"L={candle['l']} | "
            f"C={candle['c']} | "
            f"V={candle['v']} | "
            f"{state}"
        )

    except json.JSONDecodeError:
        print("Received invalid JSON")

    except (KeyError, TypeError, ValueError) as exc:
        print(f"Invalid kline event: {exc}")

    except Exception as exc:
        print(f"Kline processing error: {exc}")


def on_error(ws, error):
    print(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    print(
        f"WebSocket closed | "
        f"code={close_status_code} | "
        f"message={close_msg}"
    )


def on_open(ws):
    print("=" * 70)
    print("Connected to Binance 1-minute kline streams")
    print("Persistent storage: data/market_data.db")
    print("Symbols:")

    for symbol in SYMBOLS:
        print(f"  - {symbol.upper()}")

    print("=" * 70)


def run():
    while True:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )

            ws.run_forever(
                ping_interval=20,
                ping_timeout=10,
            )

        except KeyboardInterrupt:
            print("\nStopping kline ingestion...")
            STORE.close()
            break

        except Exception as exc:
            print(f"Connection error: {exc}")

        print("Reconnecting in 5 seconds...")
        time.sleep(5)


if __name__ == "__main__":
    run()