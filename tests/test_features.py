import pandas as pd

from src.features.build_features import (
    calculate_features,
    calculate_rsi,
    create_targets,
)


def create_test_data(rows: int = 100):
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="min",
        tz="UTC",
    )

    prices = pd.Series(
        range(
            100,
            100 + rows,
        ),
        dtype=float,
    )

    return pd.DataFrame(
        {
            "open_time": timestamps,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 1000.0,
            "symbol": "BTCUSDT",
            "interval": "1m",
        }
    )


def test_rsi_created():
    df = create_test_data()

    result = calculate_rsi(
        df["close"]
    )

    assert len(result) == len(df)


def test_features_created():
    df = create_test_data()

    result = calculate_features(df)

    required_features = {
        "return_1m",
        "return_5m",
        "return_15m",
        "return_30m",
        "volatility_15m",
        "volatility_60m",
        "volume_change_5m",
        "volume_ratio",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "candle_range",
        "body_size",
    }

    assert required_features.issubset(
        result.columns
    )


def test_targets_created():
    df = create_test_data()

    df = calculate_features(df)
    df = create_targets(df)

    assert "next_return_1m" in df.columns
    assert "future_return_5m" in df.columns
    assert "target_direction_5m" in df.columns