from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

HISTORICAL_DIR = Path("data/historical")
OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = (
    OUTPUT_DIR / "crypto_ml_dataset_v2.parquet"
)

FEATURE_COLUMNS_FILE = (
    OUTPUT_DIR / "feature_columns_v2.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

VOLATILITY_MULTIPLIER = 0.5


# ============================================================
# BASIC INDICATORS
# ============================================================

def calculate_rsi(series, period=14):
    """Calculate RSI using average gains and losses."""

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        period,
        min_periods=period,
    ).mean()

    avg_loss = loss.rolling(
        period,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_ema(series, span):
    """Calculate exponential moving average."""

    return series.ewm(
        span=span,
        adjust=False,
    ).mean()


def calculate_features(df):
    """Create technical, lag, volatility and temporal features."""

    df = df.copy()

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    df["return_1m"] = (
        df["close"].pct_change(1)
    )

    df["return_5m"] = (
        df["close"].pct_change(5)
    )

    df["return_15m"] = (
        df["close"].pct_change(15)
    )

    df["return_30m"] = (
        df["close"].pct_change(30)
    )

    # --------------------------------------------------------
    # Lagged returns
    # --------------------------------------------------------

    for lag in [1, 2, 3, 5, 10, 15, 30, 60]:
        df[f"return_1m_lag_{lag}"] = (
            df["return_1m"].shift(lag)
        )

    for lag in [1, 2, 3, 6, 12]:
        df[f"return_5m_lag_{lag}"] = (
            df["return_5m"].shift(lag)
        )

    # --------------------------------------------------------
    # Rolling return statistics
    # --------------------------------------------------------

    for window in [5, 15, 30, 60]:

        df[f"rolling_return_mean_{window}"] = (
            df["return_1m"]
            .rolling(window)
            .mean()
        )

        df[f"rolling_return_std_{window}"] = (
            df["return_1m"]
            .rolling(window)
            .std()
        )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["volatility_15m"] = (
        df["return_1m"]
        .rolling(15)
        .std()
    )

    df["volatility_60m"] = (
        df["return_1m"]
        .rolling(60)
        .std()
    )

    # --------------------------------------------------------
    # Volume features
    # --------------------------------------------------------

    df["volume_change_5m"] = (
        df["volume"]
        .pct_change(5)
    )

    df["volume_ratio"] = (
        df["volume"]
        / df["volume"]
        .rolling(20)
        .mean()
    )

    volume_mean_60 = (
        df["volume"]
        .rolling(60)
        .mean()
    )

    volume_std_60 = (
        df["volume"]
        .rolling(60)
        .std()
    )

    df["volume_zscore_60"] = (
        (df["volume"] - volume_mean_60)
        / volume_std_60.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi_14"] = calculate_rsi(
        df["close"],
        14,
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema_12 = calculate_ema(
        df["close"],
        12,
    )

    ema_26 = calculate_ema(
        df["close"],
        26,
    )

    df["macd"] = (
        ema_12 - ema_26
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    df["macd_histogram"] = (
        df["macd"]
        - df["macd_signal"]
    )

    # --------------------------------------------------------
    # Price structure
    # --------------------------------------------------------

    df["candle_range"] = (
        df["high"] - df["low"]
    )

    df["body_size"] = (
        df["close"] - df["open"]
    ).abs()

    df["upper_shadow"] = (
        df["high"]
        - df[["open", "close"]].max(axis=1)
    )

    df["lower_shadow"] = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    )

    df["close_position_in_range"] = (
        (
            df["close"] - df["low"]
        )
        / df["candle_range"].replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - previous_close
            ).abs(),
            (
                df["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_14 = (
        true_range
        .rolling(14)
        .mean()
    )

    df["atr_14_pct"] = (
        atr_14
        / df["close"]
    )

    # --------------------------------------------------------
    # Moving-average relationships
    # --------------------------------------------------------

    sma_20 = (
        df["close"]
        .rolling(20)
        .mean()
    )

    sma_50 = (
        df["close"]
        .rolling(50)
        .mean()
    )

    ema_20 = calculate_ema(
        df["close"],
        20,
    )

    df["price_vs_sma_20"] = (
        df["close"] / sma_20 - 1
    )

    df["price_vs_sma_50"] = (
        df["close"] / sma_50 - 1
    )

    df["price_vs_ema_20"] = (
        df["close"] / ema_20 - 1
    )

    # --------------------------------------------------------
    # Bollinger Band position
    # --------------------------------------------------------

    rolling_mean_20 = (
        df["close"]
        .rolling(20)
        .mean()
    )

    rolling_std_20 = (
        df["close"]
        .rolling(20)
        .std()
    )

    upper_band = (
        rolling_mean_20
        + 2 * rolling_std_20
    )

    lower_band = (
        rolling_mean_20
        - 2 * rolling_std_20
    )

    df["bollinger_position"] = (
        (df["close"] - lower_band)
        / (
            upper_band - lower_band
        ).replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    dt = pd.to_datetime(
        df["open_time"]
    )

    minute_of_day = (
        dt.dt.hour * 60
        + dt.dt.minute
    )

    day_of_week = (
        dt.dt.dayofweek
    )

    df["time_sin"] = np.sin(
        2
        * np.pi
        * minute_of_day
        / 1440
    )

    df["time_cos"] = np.cos(
        2
        * np.pi
        * minute_of_day
        / 1440
    )

    df["day_sin"] = np.sin(
        2
        * np.pi
        * day_of_week
        / 7
    )

    df["day_cos"] = np.cos(
        2
        * np.pi
        * day_of_week
        / 7
    )

    return df


# ============================================================
# TARGETS
# ============================================================

def create_targets(df):
    """Create future return and 5-minute direction targets."""

    df = df.copy()

    df["next_close_1m"] = (
        df["close"].shift(-1)
    )

    df["future_close_5m"] = (
        df["close"].shift(-5)
    )

    df["next_return_1m"] = (
        df["next_close_1m"]
        / df["close"]
        - 1
    )

    df["future_return_5m"] = (
        df["future_close_5m"]
        / df["close"]
        - 1
    )

    threshold = (
        df["volatility_15m"]
        * VOLATILITY_MULTIPLIER
    )

    df["target_direction_5m"] = np.select(
        [
            df["future_return_5m"] > threshold,
            df["future_return_5m"] < -threshold,
        ],
        [
            "UP",
            "DOWN",
        ],
        default="NEUTRAL",
    )

    df = df.drop(
        columns=[
            "next_close_1m",
            "future_close_5m",
        ]
    )

    return df


# ============================================================
# CROSS-ASSET FEATURES
# ============================================================

def create_cross_asset_features(dataframes):
    """Create cross-coin return and market features."""

    cross_frames = []

    for symbol, df in dataframes.items():

        cross_frames.append(
            df[
                [
                    "open_time",
                    "return_1m",
                    "return_5m",
                ]
            ].assign(
                symbol=symbol
            )
        )

    combined = pd.concat(
        cross_frames,
        ignore_index=True,
    )

    return_1m = combined.pivot(
        index="open_time",
        columns="symbol",
        values="return_1m",
    )

    return_5m = combined.pivot(
        index="open_time",
        columns="symbol",
        values="return_5m",
    )

    return_1m.columns = [
        f"{symbol}_return_1m"
        for symbol in return_1m.columns
    ]

    return_5m.columns = [
        f"{symbol}_return_5m"
        for symbol in return_5m.columns
    ]

    cross_features = pd.concat(
        [
            return_1m,
            return_5m,
        ],
        axis=1,
    )

    cross_features[
        "market_return_1m"
    ] = return_1m.mean(axis=1)

    cross_features[
        "market_return_5m"
    ] = return_5m.mean(axis=1)

    cross_features[
        "market_volatility_15m"
    ] = (
        cross_features["market_return_1m"]
        .rolling(15)
        .std()
    )

    cross_features = (
        cross_features
        .reset_index()
    )

    return cross_features


# ============================================================
# LOAD ONE SYMBOL
# ============================================================

def load_symbol_data(symbol):
    """Load historical data for one symbol."""

    file_path = (
        HISTORICAL_DIR
        / f"{symbol}_1m.parquet"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Historical file not found: "
            f"{file_path}"
        )

    df = pd.read_parquet(
        file_path
    )

    required_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol} is missing columns: "
            f"{missing}"
        )

    df = df[
        required_columns
    ].copy()

    df["open_time"] = pd.to_datetime(
        df["open_time"]
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRYPTO MLOPS - FEATURE ENGINEERING V2")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    symbol_data = {}

    # --------------------------------------------------------
    # Load and engineer each coin
    # --------------------------------------------------------

    for symbol in SYMBOLS:

        print()
        print(
            f"Processing {symbol}..."
        )

        df = load_symbol_data(
            symbol
        )

        df = calculate_features(
            df
        )

        df = create_targets(
            df
        )

        symbol_data[symbol] = df

        print(
            f"{symbol}: "
            f"{len(df):,} rows"
        )

    # --------------------------------------------------------
    # Cross-asset features
    # --------------------------------------------------------

    print()
    print(
        "Creating cross-asset features..."
    )

    cross_features = (
        create_cross_asset_features(
            symbol_data
        )
    )

    # --------------------------------------------------------
    # Combine all symbols
    # --------------------------------------------------------

    final_frames = []

    for symbol in SYMBOLS:

        df = symbol_data[symbol].copy()

        df = df.merge(
            cross_features,
            on="open_time",
            how="left",
        )

        df["symbol"] = symbol

        final_frames.append(
            df
        )

    final_df = pd.concat(
        final_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Clean invalid values
    # --------------------------------------------------------

    final_df = final_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    final_df = (
        final_df
        .sort_values(
            ["open_time", "symbol"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save feature dataset
    # --------------------------------------------------------

    final_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Feature list
    # --------------------------------------------------------

    excluded_columns = {
        "open_time",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "next_return_1m",
        "future_return_5m",
        "target_direction_5m",
    }

    feature_columns = [
        column
        for column in final_df.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(
            final_df[column]
        )
    ]

    with open(
        FEATURE_COLUMNS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        for column in feature_columns:
            file.write(
                f"{column}\n"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE ENGINEERING V2 COMPLETE")
    print("=" * 70)

    print(
        f"Rows: "
        f"{len(final_df):,}"
    )

    print(
        f"Columns: "
        f"{len(final_df.columns)}"
    )

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Output: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Feature list: "
        f"{FEATURE_COLUMNS_FILE}"
    )


if __name__ == "__main__":
    main()