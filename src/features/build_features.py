from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

HISTORICAL_DIR = Path("data/historical")
OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = OUTPUT_DIR / "crypto_ml_dataset.parquet"

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

# Forecast horizons
NEXT_1M = 1
NEXT_5M = 5

# Used only to create the UP/DOWN/NEUTRAL classification target.
VOLATILITY_MULTIPLIER = 0.5


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_symbol_data(symbol: str) -> pd.DataFrame:
    """Load one symbol's historical 1-minute candles."""

    path = HISTORICAL_DIR / f"{symbol}_1m.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Historical file not found: {path}"
        )

    df = pd.read_parquet(path)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df = df.sort_values("open_time").copy()

    return df


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate RSI."""

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create features using information available at time t."""

    df = df.copy()

    # --------------------------------------------------------
    # Price returns
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
    # Rolling volatility
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
    # Volume
    # --------------------------------------------------------

    df["volume_change_5m"] = (
        df["volume"].pct_change(5)
    )

    volume_ma_15m = (
        df["volume"].rolling(15).mean()
    )

    df["volume_ratio"] = (
        df["volume"]
        / volume_ma_15m
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi_14"] = calculate_rsi(
        df["close"]
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema_12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema_26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    df["macd"] = ema_12 - ema_26

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
    # Candle structure
    # --------------------------------------------------------

    df["candle_range"] = (
        (df["high"] - df["low"])
        / df["close"]
    )

    df["body_size"] = (
        (df["close"] - df["open"])
        / df["open"]
    )

    return df


# ============================================================
# CROSS-ASSET FEATURES
# ============================================================

def build_market_features(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create market-wide features from all five coins."""

    return_features = []

    for symbol, df in frames.items():

        temp = df[
            [
                "open_time",
                "return_1m",
            ]
        ].copy()

        temp = temp.rename(
            columns={
                "return_1m": f"{symbol}_return_1m"
            }
        )

        return_features.append(temp)

    market_features = return_features[0]

    for temp in return_features[1:]:
        market_features = market_features.merge(
            temp,
            on="open_time",
            how="inner",
        )

    return_columns = [
        column
        for column in market_features.columns
        if column.endswith("_return_1m")
    ]

    market_features["market_return_1m"] = (
        market_features[return_columns]
        .mean(axis=1)
    )

    market_features["market_volatility_15m"] = (
        market_features["market_return_1m"]
        .rolling(15)
        .std()
    )

    return market_features


# ============================================================
# TARGET CREATION
# ============================================================

def create_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create future prediction targets.

    Regression targets:
        next_return_1m
        future_return_5m

    Classification target:
        target_direction_5m

    Future price columns are temporary and are removed later.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Future prices
    # --------------------------------------------------------

    df["next_close_1m"] = (
        df["close"].shift(-NEXT_1M)
    )

    df["future_close_5m"] = (
        df["close"].shift(-NEXT_5M)
    )

    # --------------------------------------------------------
    # Future returns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Direction threshold
    # --------------------------------------------------------

    volatility_threshold = (
        df["volatility_15m"]
        * VOLATILITY_MULTIPLIER
        * np.sqrt(NEXT_5M)
    )

    df["target_direction_5m"] = np.select(
        [
            df["future_return_5m"]
            > volatility_threshold,

            df["future_return_5m"]
            < -volatility_threshold,
        ],
        [
            "UP",
            "DOWN",
        ],
        default="NEUTRAL",
    )

    return df


# ============================================================
# BUILD ONE COIN
# ============================================================

def build_symbol_dataset(
    symbol: str,
) -> pd.DataFrame:
    """Build features and targets for one coin."""

    df = load_symbol_data(symbol)

    df = calculate_features(df)

    df = create_targets(df)

    df["symbol"] = symbol

    return df


# ============================================================
# BUILD COMPLETE DATASET
# ============================================================

def build_dataset() -> pd.DataFrame:
    """Build the complete multi-asset ML dataset."""

    frames: dict[str, pd.DataFrame] = {}

    for symbol in SYMBOLS:
        print(
            f"Building features for {symbol}..."
        )

        frames[symbol] = build_symbol_dataset(
            symbol
        )

    # --------------------------------------------------------
    # Cross-asset market features
    # --------------------------------------------------------

    market_features = build_market_features(
        frames
    )

    final_frames = []

    for symbol, df in frames.items():

        df = df.merge(
            market_features,
            on="open_time",
            how="left",
        )

        final_frames.append(df)

    # --------------------------------------------------------
    # Combine all coins
    # --------------------------------------------------------

    combined = pd.concat(
        final_frames,
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["open_time", "symbol"]
    )

    # --------------------------------------------------------
    # Remove rows without enough history/future
    # --------------------------------------------------------

    required_columns = [
        "return_30m",
        "volatility_15m",
        "volatility_60m",
        "rsi_14",
        "future_return_5m",
        "next_return_1m",
    ]

    combined = combined.dropna(
        subset=required_columns
    )

    # --------------------------------------------------------
    # Require all cross-asset returns
    # --------------------------------------------------------

    market_return_columns = [
        f"{symbol}_return_1m"
        for symbol in SYMBOLS
    ]

    combined = combined.dropna(
        subset=market_return_columns
    )

    # --------------------------------------------------------
    # Remove future information
    # --------------------------------------------------------

    combined = combined.drop(
        columns=[
            "next_close_1m",
            "future_close_5m",
        ]
    )

    return combined


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CRYPTO MLOPS — FEATURE DATASET BUILDER")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = build_dataset()

    dataset.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("FEATURE DATASET CREATED")
    print("=" * 70)

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(dataset):,}"
    )

    print(
        f"Columns: {len(dataset.columns)}"
    )

    print()
    print("Symbols:")
    print(
        dataset["symbol"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Direction target:")
    print(
        dataset["target_direction_5m"]
        .value_counts()
    )

    print()
    print("Regression targets:")
    print(
        dataset[
            [
                "next_return_1m",
                "future_return_5m",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()