from scripts.validate_historical_data import EXPECTED_SYMBOLS


def test_expected_symbols():
    assert EXPECTED_SYMBOLS == {
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
    }