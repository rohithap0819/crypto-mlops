from __future__ import annotations

import io
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

INTERVAL = "1m"

# Binance public market-data archive
BASE_URL = "https://data.binance.vision/data/spot"

OUTPUT_DIR = Path("data/historical")

# We download one year of COMPLETED days.
# Example:
# today = 2026-09-28
# end   = 2026-09-27
# start = 2025-09-28
TODAY = date.today()
END_DATE = TODAY - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=364)


# Binance kline columns
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def build_monthly_url(symbol: str, year: int, month: int) -> str:
    """Build the official Binance monthly kline URL."""

    return (
        f"{BASE_URL}/monthly/klines/"
        f"{symbol}/{INTERVAL}/"
        f"{symbol}-{INTERVAL}-{year}-{month:02d}.zip"
    )


def build_daily_url(symbol: str, current_date: date) -> str:
    """Build the official Binance daily kline URL."""

    return (
        f"{BASE_URL}/daily/klines/"
        f"{symbol}/{INTERVAL}/"
        f"{symbol}-{INTERVAL}-"
        f"{current_date.year}-"
        f"{current_date.month:02d}-"
        f"{current_date.day:02d}.zip"
    )


def download_zip(url: str) -> bytes | None:
    """Download one ZIP archive with retries."""

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                timeout=60,
            )

            if response.status_code == 200:
                return response.content

            if response.status_code == 404:
                print(f"NOT FOUND: {url}")
                return None

            print(
                f"HTTP {response.status_code} "
                f"(attempt {attempt + 1}/3): {url}"
            )

        except requests.RequestException as exc:
            print(
                f"Download error "
                f"(attempt {attempt + 1}/3): {exc}"
            )

        time.sleep(2)

    return None


def read_kline_zip(content: bytes) -> pd.DataFrame:
    """Read a Binance kline ZIP archive into a DataFrame."""

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError("No CSV file found inside ZIP archive.")

        with archive.open(csv_files[0]) as csv_file:
            df = pd.read_csv(
                csv_file,
                header=None,
                names=KLINE_COLUMNS,
            )

    return df


# ============================================================
# DATA CLEANING
# ============================================================

def normalize_timestamp(series: pd.Series) -> pd.Series:
    """
    Binance Spot timestamps changed to microseconds from 2025.

    Detect timestamp scale by magnitude and convert to UTC datetime.
    """

    numeric = pd.to_numeric(series, errors="coerce")

    sample = numeric.dropna()

    if sample.empty:
        raise ValueError("Timestamp column contains no valid values.")

    median_value = sample.median()

    if median_value > 1e14:
        unit = "us"
    else:
        unit = "ms"

    return pd.to_datetime(
        numeric,
        unit=unit,
        utc=True,
    )


def clean_kline_data(
    df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Clean and standardize Binance kline data."""

    df = df.copy()

    # Remove Binance's unused final column.
    df = df.drop(columns=["ignore"])

    # Numeric columns.
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Normalize timestamps.
    df["open_time"] = normalize_timestamp(df["open_time"])
    df["close_time"] = normalize_timestamp(df["close_time"])

    # Add symbol and interval.
    df["symbol"] = symbol
    df["interval"] = INTERVAL

    # Remove invalid rows.
    df = df.dropna(
        subset=[
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    # Remove duplicate candles.
    df = df.drop_duplicates(
        subset=["symbol", "interval", "open_time"]
    )

    # Sort chronologically.
    df = df.sort_values("open_time")

    return df


# ============================================================
# DATE RANGE
# ============================================================

def month_range(
    start_date: date,
    end_date: date,
):
    """Yield first day of every month in the range."""

    current = date(
        start_date.year,
        start_date.month,
        1,
    )

    while current <= end_date:
        yield current

        if current.month == 12:
            current = date(
                current.year + 1,
                1,
                1,
            )
        else:
            current = date(
                current.year,
                current.month + 1,
                1,
            )


# ============================================================
# DOWNLOAD ONE SYMBOL
# ============================================================

def download_symbol(symbol: str) -> None:
    """Download one year of 1-minute data for one symbol."""

    print()
    print("=" * 70)
    print(f"Downloading {symbol}")
    print(f"Date range: {START_DATE} → {END_DATE}")
    print("=" * 70)

    frames: list[pd.DataFrame] = []

    # --------------------------------------------------------
    # Part 1: Monthly files
    # --------------------------------------------------------

    print()
    print("Downloading monthly archives...")

    for month_start in month_range(
        START_DATE,
        END_DATE,
    ):
        # Current month will be handled with daily files below.
        if (
            month_start.year == END_DATE.year
            and month_start.month == END_DATE.month
        ):
            continue

        url = build_monthly_url(
            symbol,
            month_start.year,
            month_start.month,
        )

        print(
            f"  Monthly: "
            f"{month_start.year}-{month_start.month:02d}"
        )

        content = download_zip(url)

        if content is None:
            continue

        month_df = read_kline_zip(content)
        month_df = clean_kline_data(
            month_df,
            symbol,
        )

        frames.append(month_df)

    # --------------------------------------------------------
    # Part 2: Daily files for the current month
    # --------------------------------------------------------

    print()
    print("Downloading current-month daily archives...")

    current_day = date(
        END_DATE.year,
        END_DATE.month,
        1,
    )

    while current_day <= END_DATE:

        if current_day >= START_DATE:

            url = build_daily_url(
                symbol,
                current_day,
            )

            print(
                f"  Daily: {current_day}"
            )

            content = download_zip(url)

            if content is not None:
                day_df = read_kline_zip(content)
                day_df = clean_kline_data(
                    day_df,
                    symbol,
                )

                frames.append(day_df)

        current_day += timedelta(days=1)

    if not frames:
        raise RuntimeError(
            f"No data downloaded for {symbol}."
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    # Filter EXACT requested range.
    start_timestamp = pd.Timestamp(
        START_DATE,
        tz="UTC",
    )

    end_timestamp = (
        pd.Timestamp(
            END_DATE,
            tz="UTC",
        )
        + pd.Timedelta(days=1)
    )

    df = df[
        (df["open_time"] >= start_timestamp)
        & (df["open_time"] < end_timestamp)
    ]

    # Remove duplicates again after combining.
    df = df.drop_duplicates(
        subset=["symbol", "interval", "open_time"]
    )

    df = df.sort_values("open_time")

    # --------------------------------------------------------
    # Save Parquet
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{symbol}_{INTERVAL}.parquet"
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Validation summary
    # --------------------------------------------------------

    print()
    print(f"Saved: {output_path}")
    print(f"Rows: {len(df):,}")

    if not df.empty:
        print(
            f"First candle: "
            f"{df['open_time'].min()}"
        )

        print(
            f"Last candle: "
            f"{df['open_time'].max()}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("CRYPTO MLOPS — HISTORICAL DATA DOWNLOADER")
    print("=" * 70)

    print(f"Start date : {START_DATE}")
    print(f"End date   : {END_DATE}")
    print(f"Interval   : {INTERVAL}")
    print(f"Symbols    : {', '.join(SYMBOLS)}")
    print(f"Output     : {OUTPUT_DIR}")

    for symbol in SYMBOLS:
        download_symbol(symbol)

    print()
    print("=" * 70)
    print("HISTORICAL DOWNLOAD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()