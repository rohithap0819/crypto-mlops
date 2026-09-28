from src.storage.market_store import MarketStore


def test_market_store_insert(tmp_path):
    db_path = tmp_path / "test_market_data.db"

    store = MarketStore(str(db_path))

    store.insert_event(
        symbol="BTCUSDT",
        price=100000.0,
        volume_24h=12345.0,
        quote_volume_24h=123456789.0,
        event_time_ms=1234567890000,
        received_at_utc="2026-09-28T00:00:00+00:00",
    )

    assert store.count_events() == 1

    store.close()