from src.ingestion.kline_client import INTERVAL, STREAMS, SYMBOLS


def test_kline_symbols():
    assert len(SYMBOLS) == 5
    assert "btcusdt" in SYMBOLS
    assert "ethusdt" in SYMBOLS
    assert "solusdt" in SYMBOLS
    assert "bnbusdt" in SYMBOLS
    assert "xrpusdt" in SYMBOLS


def test_kline_interval():
    assert INTERVAL == "1m"


def test_kline_streams():
    assert "btcusdt@kline_1m" in STREAMS
    assert "ethusdt@kline_1m" in STREAMS
    assert "solusdt@kline_1m" in STREAMS
    assert "bnbusdt@kline_1m" in STREAMS
    assert "xrpusdt@kline_1m" in STREAMS