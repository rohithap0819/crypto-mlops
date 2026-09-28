import json
import time
from datetime import datetime, timezone

import websocket


SYMBOLS = [
    "btcusdt",
    "ethusdt",
    "solusdt",
    "bnbusdt",
    "xrpusdt",
]

STREAMS = "/".join(f"{symbol}@ticker" for symbol in SYMBOLS)

WS_URL = (
    "wss://stream.binance.com:9443/stream"
    f"?streams={STREAMS}"
)


def on_message(ws, message):
    try:
        payload = json.loads(message)

        stream = payload.get("stream")
        data = payload.get("data", {})

        symbol = data.get("s")
        price = data.get("c")
        volume = data.get("v")
        event_time = data.get("E")

        timestamp = (
            datetime.fromtimestamp(
                event_time / 1000,
                tz=timezone.utc,
            )
            if event_time
            else datetime.now(timezone.utc)
        )

        print(
            f"{timestamp.isoformat()} | "
            f"{symbol} | "
            f"price={price} | "
            f"volume={volume} | "
            f"stream={stream}"
        )

    except json.JSONDecodeError:
        print("Received non-JSON message")

    except Exception as exc:
        print(f"Message processing error: {exc}")


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
    print("Connected to Binance market data stream")
    print("Subscribed symbols:")
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
            print("\nStopping market data ingestion...")
            break

        except Exception as exc:
            print(f"Connection error: {exc}")

        print("Reconnecting in 5 seconds...")
        time.sleep(5)


if __name__ == "__main__":
    run()