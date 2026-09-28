from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/historical")

EXPECTED_SYMBOLS = {
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
}


def validate_file(path: Path) -> None:
    print()
    print("=" * 70)
    print(f"VALIDATING: {path.name}")
    print("=" * 70)

    df = pd.read_parquet(path)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "interval",
    }

    missing_columns = required_columns - set(df.columns)

    assert not missing_columns, (
        f"Missing columns: {missing_columns}"
    )

    # --------------------------------------------------------
    # Symbol validation
    # --------------------------------------------------------

    symbols = set(df["symbol"].unique())

    assert len(symbols) == 1, (
        f"Expected one symbol per file, found: {symbols}"
    )

    symbol = next(iter(symbols))

    assert symbol in EXPECTED_SYMBOLS, (
        f"Unexpected symbol: {symbol}"
    )

    print(f"Symbol: {symbol}")

    # --------------------------------------------------------
    # Interval validation
    # --------------------------------------------------------

    intervals = set(df["interval"].unique())

    assert intervals == {"1m"}, (
        f"Unexpected intervals: {intervals}"
    )

    print("Interval: 1m")

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df = df.sort_values("open_time")

    print(
        f"First candle: "
        f"{df['open_time'].min()}"
    )

    print(
        f"Last candle: "
        f"{df['open_time'].max()}"
    )

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=["symbol", "interval", "open_time"]
    ).sum()

    print(f"Duplicate candles: {duplicates}")

    assert duplicates == 0, (
        f"Found {duplicates} duplicate candles."
    )

    # --------------------------------------------------------
    # Null validation
    # --------------------------------------------------------

    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    null_counts = df[price_columns].isna().sum()

    print("Null values:")
    print(null_counts)

    assert null_counts.sum() == 0, (
        "Found null values in core market columns."
    )

    # --------------------------------------------------------
    # OHLC sanity checks
    # --------------------------------------------------------

    invalid_high = (
        df["high"]
        < df[["open", "close", "low"]].max(axis=1)
    ).sum()

    invalid_low = (
        df["low"]
        > df[["open", "close", "high"]].min(axis=1)
    ).sum()

    print(f"Invalid high candles: {invalid_high}")
    print(f"Invalid low candles: {invalid_low}")

    assert invalid_high == 0
    assert invalid_low == 0

    # --------------------------------------------------------
    # Negative / zero checks
    # --------------------------------------------------------

    assert (df["close"] > 0).all()
    assert (df["open"] > 0).all()
    assert (df["high"] > 0).all()
    assert (df["low"] > 0).all()
    assert (df["volume"] >= 0).all()

    # --------------------------------------------------------
    # Time-gap analysis
    # --------------------------------------------------------

    time_diff = df["open_time"].diff()

    expected_interval = pd.Timedelta(minutes=1)

    gaps = (
        time_diff
        .dropna()
        .loc[lambda x: x != expected_interval]
    )

    print(f"Unexpected time gaps: {len(gaps)}")

    if len(gaps) > 0:
        print("Largest gaps:")
        print(
            gaps
            .sort_values(ascending=False)
            .head(10)
        )

    print("Validation completed successfully.")


def main() -> None:
    files = sorted(
        DATA_DIR.glob("*_1m.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No historical parquet files found in {DATA_DIR}"
        )

    print("=" * 70)
    print("CRYPTO MLOPS — HISTORICAL DATA VALIDATION")
    print("=" * 70)

    print(f"Files found: {len(files)}")

    for path in files:
        validate_file(path)

    print()
    print("=" * 70)
    print("ALL HISTORICAL DATA VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()