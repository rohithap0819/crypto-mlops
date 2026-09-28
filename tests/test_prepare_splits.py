import pandas as pd

from src.models.prepare_splits import prepare_splits


def create_test_dataset():
    timestamps = pd.date_range(
        "2026-01-01",
        periods=1000,
        freq="min",
        tz="UTC",
    )

    rows = []

    for timestamp in timestamps:
        rows.append(
            {
                "open_time": timestamp,
                "symbol": "BTCUSDT",
                "close": 100.0,
                "next_return_1m": 0.001,
                "future_return_5m": 0.002,
                "target_direction_5m": "UP",
            }
        )

        rows.append(
            {
                "open_time": timestamp,
                "symbol": "ETHUSDT",
                "close": 100.0,
                "next_return_1m": 0.001,
                "future_return_5m": 0.002,
                "target_direction_5m": "UP",
            }
        )

    return pd.DataFrame(rows)


def test_time_series_splits_are_chronological():
    df = create_test_dataset()

    train, validation, test, manifest = (
        prepare_splits(df)
    )

    assert (
        train["open_time"].max()
        < validation["open_time"].min()
    )

    assert (
        validation["open_time"].max()
        < test["open_time"].min()
    )


def test_all_splits_have_data():
    df = create_test_dataset()

    train, validation, test, _ = prepare_splits(df)

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0


def test_split_ratios_are_recorded():
    df = create_test_dataset()

    _, _, _, manifest = prepare_splits(df)

    assert manifest["train_ratio"] == 0.70
    assert manifest["validation_ratio"] == 0.15
    assert manifest["test_ratio"] == 0.15
    assert manifest["gap_minutes"] == 5