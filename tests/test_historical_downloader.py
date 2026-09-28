from datetime import date

from scripts.download_historical_data import (
    INTERVAL,
    SYMBOLS,
    build_daily_url,
    build_monthly_url,
)


def test_historical_symbols():
    assert len(SYMBOLS) == 5
    assert "BTCUSDT" in SYMBOLS
    assert "ETHUSDT" in SYMBOLS
    assert "SOLUSDT" in SYMBOLS
    assert "BNBUSDT" in SYMBOLS
    assert "XRPUSDT" in SYMBOLS


def test_historical_interval():
    assert INTERVAL == "1m"


def test_monthly_url():
    url = build_monthly_url(
        "BTCUSDT",
        2026,
        8,
    )

    assert (
        url
        == "https://data.binance.vision/data/spot/"
        "monthly/klines/BTCUSDT/1m/"
        "BTCUSDT-1m-2026-08.zip"
    )


def test_daily_url():
    url = build_daily_url(
        "BTCUSDT",
        date(2026, 9, 27),
    )

    assert (
        url
        == "https://data.binance.vision/data/spot/"
        "daily/klines/BTCUSDT/1m/"
        "BTCUSDT-1m-2026-09-27.zip"
    )