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

STREAMS = "/".join(f"{symbol}@ticker" for symbol in SYMBOLS)

WS_URL = (
    "wss://stream.binance.com:9443/stream"
    f"?streams={STREAMS}"
)

STORE = MarketStore()


def on_message(ws, message):
    try:
        payload = json.loads(message)
        data = payload.get("data", {})

        symbol = data.get("s")
        price = data.get("c")
        volume = data.get("v")
        quote_volume = data.get("q")
        event_time = data.get("E")

        if not symbol or price is None or event_time is None:
            return

        received_at = datetime.now(timezone.utc).isoformat()

        STORE.insert_event(
            symbol=symbol,
            price=float(price),
            volume_24h=float(volume) if volume is not None else None,
            quote_volume_24h=(
                float(quote_volume)
                if quote_volume is not None
                else None
            ),
            event_time_ms=int(event_time),
            received_at_utc=received_at,
        )

        print(
            f"{received_at} | "
            f"{symbol} | "
            f"price={price} | "
            f"stored={STORE.count_events()}"
        )

    except json.JSONDecodeError:
        print("Received invalid JSON")

    except (TypeError, ValueError) as exc:
        print(f"Invalid market event: {exc}")

    except Exception as exc:
        print(f"Market event processing error: {exc}")


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
    print("Persistent storage: data/market_data.db")
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
            STORE.close()
            break

        except Exception as exc:
            print(f"Connection error: {exc}")

        print("Reconnecting in 5 seconds...")
        time.sleep(5)


if __name__ == "__main__":
    run()