from src.ingestion.websocket_client import SYMBOLS, STREAMS


def test_symbols_configured():
    assert len(SYMBOLS) == 5
    assert "btcusdt" in SYMBOLS
    assert "ethusdt" in SYMBOLS
    assert "solusdt" in SYMBOLS
    assert "bnbusdt" in SYMBOLS
    assert "xrpusdt" in SYMBOLS


def test_stream_configuration():
    assert "btcusdt@ticker" in STREAMS
    assert "ethusdt@ticker" in STREAMS
    assert "solusdt@ticker" in STREAMS