"""
Execution-aware portfolio backtest V3.

Key corrections from V1/V2:

1. Does NOT filter test rows using future_return_5m.
   Predictions are generated on all valid test timestamps.

2. Signal is generated at candle t close.

3. Entry occurs at candle t+1 open.

4. Exit occurs five minutes later at candle t+6 open.

5. Each coin has an independent capital account.

6. No overlapping position for the same coin.

7. Portfolio equity is forward-filled correctly across time.

8. Calculates:
       - total return
       - win rate
       - profit factor
       - maximum drawdown
       - daily Sharpe ratio
       - number of trades

Frozen model:
    CatBoost = 55%
    GRU      = 45%

Frozen confidence threshold:
    0.55

This is still a research backtest, not a production execution
simulator. It does not model order-book slippage, latency,
funding, liquidation mechanics, or exchange-specific fees.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from catboost import CatBoostClassifier

from src.models.binary_sequence_models import BinaryGRU


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits_v2"
    / "train.parquet"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits_v2"
    / "validation.parquet"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits_v2"
    / "test.parquet"
)

FEATURE_COLUMNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_columns_v2.txt"
)

SEQUENCE_TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_v1"
    / "test"
)

GRU_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "binary_sequence_v1"
    / "gru_binary_best.pt"
)

REPORTS_DIR = (
    PROJECT_ROOT / "reports"
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

TARGET_COLUMN = "future_return_5m"

SEQUENCE_LENGTH = 60

CATBOOST_WEIGHT = 0.55
GRU_WEIGHT = 0.45

CONFIDENCE_THRESHOLD = 0.55

INITIAL_CAPITAL = 100_000.0

CAPITAL_PER_SYMBOL = (
    INITIAL_CAPITAL / len(SYMBOLS)
)

HOLDING_MINUTES = 5

# Round-trip transaction-cost scenarios.
ROUND_TRIP_COSTS = [
    0.0000,  # 0 bps
    0.0005,  # 5 bps
    0.0010,  # 10 bps
    0.0020,  # 20 bps
]


# ============================================================
# FEATURES
# ============================================================

def load_feature_columns() -> list[str]:

    with open(
        FEATURE_COLUMNS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        columns = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(columns) != 61:

        raise ValueError(
            f"Expected 61 features, found {len(columns)}"
        )

    return columns


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing feature columns:\n{missing}"
        )

    X = df[
        feature_columns
    ].copy()

    symbol_dummies = pd.get_dummies(
        df["symbol"],
        prefix="symbol",
        dtype=np.float32,
    )

    expected_symbols = [
        f"symbol_{symbol}"
        for symbol in SYMBOLS
    ]

    for column in expected_symbols:

        if column not in symbol_dummies.columns:
            symbol_dummies[column] = 0.0

    symbol_dummies = symbol_dummies[
        expected_symbols
    ]

    return pd.concat(
        [
            X.reset_index(drop=True),
            symbol_dummies.reset_index(drop=True),
        ],
        axis=1,
    )


def prepare_training_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
):

    # The model was trained using only non-zero returns.
    # This applies to TRAINING only.
    mask = (
        df[TARGET_COLUMN].notna()
        & (df[TARGET_COLUMN] != 0)
    )

    clean = (
        df.loc[mask]
        .reset_index(drop=True)
    )

    X = prepare_features(
        clean,
        feature_columns,
    )

    y = (
        clean[TARGET_COLUMN] > 0
    ).astype(np.int64)

    return X, y


def prepare_prediction_features(
    df: pd.DataFrame,
    feature_columns: list[str],
):

    # IMPORTANT:
    # Do NOT remove zero future-return rows here.
    # The backtest must generate predictions without
    # looking at future outcomes.
    clean = df.copy()

    X = prepare_features(
        clean,
        feature_columns,
    )

    return X


# ============================================================
# CATBOOST
# ============================================================

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):

    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=42,
        thread_count=-1,
        verbose=False,
    )

    print(
        "\nTraining CatBoost..."
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# GRU
# ============================================================

def load_gru():

    model = BinaryGRU(
        input_size=61,
        hidden_size=64,
        dropout=0.2,
        num_symbols=len(SYMBOLS),
        symbol_embedding_dim=4,
    )

    checkpoint = torch.load(
        GRU_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


def generate_gru_test_predictions():

    model = load_gru()

    records = []

    print(
        "\nGenerating GRU predictions "
        "on all valid test timestamps..."
    )

    for symbol_id, symbol in enumerate(
        SYMBOLS
    ):

        X = np.load(
            SEQUENCE_TEST_DIR
            / f"{symbol}_X.npy"
        )

        valid_indices = np.load(
            SEQUENCE_TEST_DIR
            / f"{symbol}_valid_indices.npy"
        ).astype(np.int64)

        times = np.load(
            SEQUENCE_TEST_DIR
            / f"{symbol}_time.npy"
        )

        valid_mask = (
            valid_indices
            >= SEQUENCE_LENGTH - 1
        )

        valid_indices = (
            valid_indices[
                valid_mask
            ]
        )

        endpoint_times = (
            times[
                valid_indices
            ]
        )

        sequences = np.asarray(
            [
                X[
                    index
                    - SEQUENCE_LENGTH
                    + 1:
                    index + 1
                ]
                for index in valid_indices
            ],
            dtype=np.float32,
        )

        symbol_ids = np.full(
            len(sequences),
            symbol_id,
            dtype=np.int64,
        )

        probabilities = []

        with torch.no_grad():

            for start in range(
                0,
                len(sequences),
                256,
            ):

                end = min(
                    start + 256,
                    len(sequences),
                )

                X_batch = torch.tensor(
                    sequences[start:end],
                    dtype=torch.float32,
                )

                symbol_batch = torch.tensor(
                    symbol_ids[start:end],
                    dtype=torch.long,
                )

                logits = model(
                    X_batch,
                    symbol_batch,
                )

                probabilities.append(
                    torch.sigmoid(
                        logits
                    ).numpy()
                )

        probabilities = np.concatenate(
            probabilities
        )

        records.append(
            pd.DataFrame(
                {
                    "symbol":
                        symbol,
                    "open_time":
                        pd.to_datetime(
                            endpoint_times,
                            unit="us",
                            utc=True,
                        ),
                    "gru_probability":
                        probabilities,
                }
            )
        )

        print(
            f"{symbol}: "
            f"{len(probabilities):,} predictions"
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


# ============================================================
# SIGNALS
# ============================================================

def build_signals(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df[
        "ensemble_probability"
    ] = (
        CATBOOST_WEIGHT
        * df[
            "catboost_probability"
        ]
        +
        GRU_WEIGHT
        * df[
            "gru_probability"
        ]
    )

    df[
        "confidence"
    ] = np.maximum(
        df[
            "ensemble_probability"
        ],
        1.0
        -
        df[
            "ensemble_probability"
        ],
    )

    # Always trade:
    # probability >= 0.50 -> LONG
    # probability <  0.50 -> SHORT
    df[
        "always_signal"
    ] = np.where(
        df[
            "ensemble_probability"
        ] >= 0.50,
        1,
        -1,
    )

    # Confidence-filtered:
    # >= 0.55 -> LONG
    # <= 0.45 -> SHORT
    # otherwise -> NO TRADE
    df[
        "confidence_signal"
    ] = np.where(
        df[
            "ensemble_probability"
        ] >= CONFIDENCE_THRESHOLD,
        1,
        np.where(
            df[
                "ensemble_probability"
            ]
            <=
            1.0
            - CONFIDENCE_THRESHOLD,
            -1,
            0,
        ),
    )

    # Inverse previous-5m momentum baseline.
    #
    # Negative previous 5m return -> LONG
    # Positive previous 5m return -> SHORT
    df[
        "inverse_signal"
    ] = np.where(
        df[
            "return_5m"
        ] < 0,
        1,
        np.where(
            df[
                "return_5m"
            ] > 0,
            -1,
            0,
        ),
    )

    return df


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_symbol(
    symbol_df: pd.DataFrame,
    signal_column: str,
    initial_capital: float,
    round_trip_cost: float,
    strategy_name: str,
):

    symbol_df = (
        symbol_df
        .sort_values(
            "open_time"
        )
        .reset_index(
            drop=True
        )
    )

    # Timestamp -> row index.
    timestamp_to_index = {
        timestamp: index
        for index, timestamp
        in enumerate(
            symbol_df[
                "open_time"
            ]
        )
    }

    capital = (
        initial_capital
    )

    trades = []

    equity_events = []

    for timestamp in (
        symbol_df[
            "open_time"
        ]
    ):

        equity_events.append(
            {
                "open_time":
                    timestamp,
                "symbol":
                    symbol_df.iloc[
                        0
                    ]["symbol"],
                "equity":
                    capital,
            }
        )

    # --------------------------------------------------------
    # Iterate through signal timestamps.
    # --------------------------------------------------------

    i = 0

    while i < len(symbol_df):

        row = symbol_df.iloc[i]

        signal = int(
            row[
                signal_column
            ]
        )

        if signal == 0:

            i += 1
            continue

        signal_time = row[
            "open_time"
        ]

        # ----------------------------------------------------
        # Execution timing:
        #
        # Signal at candle t close.
        # Entry at t + 1 minute open.
        # Exit five minutes later = t + 6 minutes open.
        # ----------------------------------------------------

        entry_time = (
            signal_time
            + pd.Timedelta(
                minutes=1
            )
        )

        exit_time = (
            signal_time
            + pd.Timedelta(
                minutes=HOLDING_MINUTES + 1
            )
        )

        if (
            entry_time
            not in timestamp_to_index
        ):

            i += 1
            continue

        if (
            exit_time
            not in timestamp_to_index
        ):

            i += 1
            continue

        entry_index = (
            timestamp_to_index[
                entry_time
            ]
        )

        exit_index = (
            timestamp_to_index[
                exit_time
            ]
        )

        entry_price = float(
            symbol_df.iloc[
                entry_index
            ]["open"]
        )

        exit_price = float(
            symbol_df.iloc[
                exit_index
            ]["open"]
        )

        market_return = (
            exit_price
            /
            entry_price
            - 1.0
        )

        gross_return = (
            signal
            * market_return
        )

        net_return = (
            gross_return
            -
            round_trip_cost
        )

        capital_before = (
            capital
        )

        capital_after = (
            capital
            * (
                1.0
                + net_return
            )
        )

        trades.append(
            {
                "strategy":
                    strategy_name,
                "symbol":
                    row["symbol"],
                "signal_time":
                    signal_time,
                "entry_time":
                    entry_time,
                "exit_time":
                    exit_time,
                "direction":
                    (
                        "LONG"
                        if signal == 1
                        else "SHORT"
                    ),
                "entry_price":
                    entry_price,
                "exit_price":
                    exit_price,
                "market_return":
                    market_return,
                "gross_return":
                    gross_return,
                "transaction_cost":
                    round_trip_cost,
                "net_return":
                    net_return,
                "capital_before":
                    capital_before,
                "capital_after":
                    capital_after,
            }
        )

        capital = (
            capital_after
        )

        # Do not allow an overlapping trade for this symbol.
        #
        # The next usable signal must occur after exit_time.
        i = exit_index + 1

    # --------------------------------------------------------
    # Build correct event-based equity.
    # --------------------------------------------------------

    equity = pd.DataFrame(
        equity_events
    )

    # The event list above represents the opening balance.
    # Replace it with actual post-trade balances at exit times.
    #
    # Start with initial capital.
    equity[
        "equity"
    ] = initial_capital

    for trade in trades:

        equity.loc[
            equity[
                "open_time"
            ]
            >= trade[
                "exit_time"
            ],
            "equity",
        ] = trade[
            "capital_after"
        ]

    # Deduplicate timestamp rows.
    equity = (
        equity
        .sort_values(
            "open_time"
        )
        .drop_duplicates(
            subset=[
                "open_time"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    trades_df = pd.DataFrame(
        trades
    )

    return (
        trades_df,
        equity,
    )


# ============================================================
# PORTFOLIO METRICS
# ============================================================

def calculate_portfolio_metrics(
    trades: pd.DataFrame,
    portfolio_equity: pd.DataFrame,
    initial_capital: float,
):

    if len(
        portfolio_equity
    ) == 0:

        return {
            "trades": 0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": np.nan,
            "ending_capital":
                initial_capital,
        }

    final_equity = float(
        portfolio_equity[
            "portfolio_equity"
        ].iloc[-1]
    )

    total_return = (
        final_equity
        /
        initial_capital
        - 1.0
    )

    peak = (
        portfolio_equity[
            "portfolio_equity"
        ]
        .cummax()
    )

    drawdown = (
        portfolio_equity[
            "portfolio_equity"
        ]
        /
        peak
        - 1.0
    )

    max_drawdown = float(
        drawdown.min()
    )

    if len(trades) > 0:

        trade_returns = (
            trades[
                "net_return"
            ].to_numpy(
                dtype=float
            )
        )

        win_rate = float(
            np.mean(
                trade_returns > 0
            )
        )

        profits = (
            trade_returns[
                trade_returns > 0
            ]
        )

        losses = (
            trade_returns[
                trade_returns < 0
            ]
        )

        if len(losses) == 0:

            profit_factor = np.inf

        else:

            profit_factor = float(
                profits.sum()
                /
                abs(
                    losses.sum()
                )
            )

    else:

        win_rate = np.nan
        profit_factor = np.nan

    # --------------------------------------------------------
    # Daily Sharpe
    # --------------------------------------------------------

    equity_series = (
        portfolio_equity[
            [
                "open_time",
                "portfolio_equity",
            ]
        ]
        .drop_duplicates(
            subset=[
                "open_time"
            ]
        )
        .set_index(
            "open_time"
        )[
            "portfolio_equity"
        ]
    )

    daily_equity = (
        equity_series
        .resample("1D")
        .last()
        .ffill()
    )

    daily_returns = (
        daily_equity
        .pct_change()
        .dropna()
    )

    if (
        len(daily_returns) >= 2
        and daily_returns.std() > 0
    ):

        sharpe = float(
            (
                daily_returns.mean()
                /
                daily_returns.std()
            )
            *
            np.sqrt(365)
        )

    else:

        sharpe = np.nan

    return {
        "trades":
            int(len(trades)),
        "win_rate":
            win_rate,
        "profit_factor":
            profit_factor,
        "total_return":
            float(total_return),
        "max_drawdown":
            max_drawdown,
        "sharpe":
            sharpe,
        "ending_capital":
            final_equity,
    }


# ============================================================
# BUY AND HOLD
# ============================================================

def calculate_buy_hold(
    merged: pd.DataFrame,
):

    returns = []

    for symbol in SYMBOLS:

        symbol_df = (
            merged[
                merged["symbol"]
                == symbol
            ]
            .sort_values(
                "open_time"
            )
        )

        if len(symbol_df) < 2:
            continue

        first_entry_time = (
            symbol_df.iloc[
                0
            ][
                "open_time"
            ]
            +
            pd.Timedelta(
                minutes=1
            )
        )

        first_entry_candidates = (
            symbol_df[
                symbol_df[
                    "open_time"
                ]
                == first_entry_time
            ]
        )

        if len(
            first_entry_candidates
        ) == 0:

            continue

        first_price = float(
            first_entry_candidates.iloc[
                0
            ]["open"]
        )

        last_price = float(
            symbol_df.iloc[
                -1
            ]["close"]
        )

        returns.append(
            last_price
            /
            first_price
            - 1.0
        )

    if not returns:

        return np.nan

    return float(
        np.mean(
            returns
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "EXECUTION-AWARE ENSEMBLE BACKTEST V3"
    )
    print("=" * 70)

    print(
        f"Initial capital: "
        f"{INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Capital per symbol: "
        f"{CAPITAL_PER_SYMBOL:,.2f}"
    )

    print(
        f"Entry: next 1-minute candle OPEN"
    )

    print(
        f"Exit: {HOLDING_MINUTES} minutes after entry"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    feature_columns = (
        load_feature_columns()
    )

    print(
        "\nLoading data..."
    )

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    test_df = pd.read_parquet(
        TEST_FILE
    )

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    X_train, y_train = (
        prepare_training_dataset(
            train_df,
            feature_columns,
        )
    )

    X_validation, y_validation = (
        prepare_training_dataset(
            validation_df,
            feature_columns,
        )
    )

    X_full_train = pd.concat(
        [
            X_train,
            X_validation,
        ],
        ignore_index=True,
    )

    y_full_train = pd.concat(
        [
            y_train,
            y_validation,
        ],
        ignore_index=True,
    )

    print(
        f"Full training rows: "
        f"{len(X_full_train):,}"
    )

    # --------------------------------------------------------
    # Test prediction features
    #
    # IMPORTANT:
    # all test rows are retained.
    # --------------------------------------------------------

    X_test = (
        prepare_prediction_features(
            test_df,
            feature_columns,
        )
    )

    print(
        f"Full test prediction rows: "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # Train CatBoost
    # --------------------------------------------------------

    catboost = train_catboost(
        X_full_train,
        y_full_train,
    )

    catboost_probability = (
        catboost.predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = test_df.copy()

    test_predictions[
        "open_time"
    ] = pd.to_datetime(
        test_predictions[
            "open_time"
        ],
        utc=True,
    )

    test_predictions[
        "catboost_probability"
    ] = catboost_probability

    # --------------------------------------------------------
    # GRU
    # --------------------------------------------------------

    gru_predictions = (
        generate_gru_test_predictions()
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = test_predictions.merge(
        gru_predictions,
        on=[
            "symbol",
            "open_time",
        ],
        how="inner",
    )

    print(
        f"\nMatched prediction rows: "
        f"{len(merged):,}"
    )

    if len(merged) < 100_000:

        raise ValueError(
            "Too few CatBoost/GRU rows matched."
        )

    merged = build_signals(
        merged
    )

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    strategies = {
        "Always_Trade_Ensemble":
            "always_signal",
        "Confidence_Filtered_Ensemble":
            "confidence_signal",
        "Inverse_5m":
            "inverse_signal",
    }

    summary_rows = []

    all_trade_rows = []

    all_equity_rows = []

    for cost in ROUND_TRIP_COSTS:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"ROUND-TRIP COST: "
            f"{cost * 10000:.0f} bps"
        )

        print(
            "=" * 70
        )

        for (
            strategy_name,
            signal_column,
        ) in strategies.items():

            symbol_trades = []
            symbol_equities = []

            for symbol in SYMBOLS:

                symbol_df = (
                    merged[
                        merged[
                            "symbol"
                        ]
                        == symbol
                    ]
                    .copy()
                )

                trades, equity = (
                    simulate_symbol(
                        symbol_df,
                        signal_column,
                        CAPITAL_PER_SYMBOL,
                        cost,
                        strategy_name,
                    )
                )

                if len(trades) > 0:

                    trades[
                        "round_trip_cost"
                    ] = cost

                    symbol_trades.append(
                        trades
                    )

                symbol_equities.append(
                    equity
                )

            # ------------------------------------------------
            # Combine trades
            # ------------------------------------------------

            if symbol_trades:

                trades = pd.concat(
                    symbol_trades,
                    ignore_index=True,
                )

            else:

                trades = pd.DataFrame()

            # ------------------------------------------------
            # Correctly combine equity curves.
            #
            # Each coin is forward-filled through time.
            # ------------------------------------------------

            equity_frames = []

            for equity in (
                symbol_equities
            ):

                symbol = (
                    equity[
                        "symbol"
                    ].iloc[0]
                )

                temp = (
                    equity[
                        [
                            "open_time",
                            "equity",
                        ]
                    ]
                    .drop_duplicates(
                        subset=[
                            "open_time"
                        ]
                    )
                    .set_index(
                        "open_time"
                    )
                    .rename(
                        columns={
                            "equity":
                                symbol
                        }
                    )
                )

                equity_frames.append(
                    temp
                )

            portfolio_equity = (
                pd.concat(
                    equity_frames,
                    axis=1,
                )
                .sort_index()
                .ffill()
                .fillna(
                    CAPITAL_PER_SYMBOL
                )
            )

            portfolio_equity[
                "portfolio_equity"
            ] = portfolio_equity[
                list(
                    SYMBOLS
                )
            ].sum(
                axis=1
            )

            portfolio_equity = (
                portfolio_equity[
                    [
                        "portfolio_equity"
                    ]
                ]
                .reset_index()
            )

            metrics = (
                calculate_portfolio_metrics(
                    trades,
                    portfolio_equity,
                    INITIAL_CAPITAL,
                )
            )

            summary_rows.append(
                {
                    "strategy":
                        strategy_name,
                    "round_trip_cost":
                        cost,
                    "cost_bps":
                        cost * 10000,
                    **metrics,
                }
            )

            print(
                f"{strategy_name:32s} "
                f"Trades={metrics['trades']:6d} "
                f"Win={metrics['win_rate'] * 100:6.2f}% "
                f"Return={metrics['total_return'] * 100:8.2f}% "
                f"MaxDD={metrics['max_drawdown'] * 100:8.2f}% "
                f"Sharpe={metrics['sharpe']:7.3f} "
                f"PF={metrics['profit_factor']:7.3f}"
            )

            if len(trades) > 0:

                trades[
                    "round_trip_cost"
                ] = cost

                all_trade_rows.append(
                    trades
                )

            portfolio_equity[
                "strategy"
            ] = strategy_name

            portfolio_equity[
                "round_trip_cost"
            ] = cost

            all_equity_rows.append(
                portfolio_equity
            )

    # --------------------------------------------------------
    # Buy and hold
    # --------------------------------------------------------

    buy_hold = (
        calculate_buy_hold(
            merged
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Equal-weight Buy & Hold: "
        f"{buy_hold * 100:.2f}%"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        REPORTS_DIR
        / "binary_strategy_backtest_v3.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    if all_trade_rows:

        trades_df = pd.concat(
            all_trade_rows,
            ignore_index=True,
        )

        trades_path = (
            REPORTS_DIR
            / "binary_strategy_backtest_trades_v3.csv"
        )

        trades_df.to_csv(
            trades_path,
            index=False,
        )

    else:

        trades_path = None

    if all_equity_rows:

        equity_df = pd.concat(
            all_equity_rows,
            ignore_index=True,
        )

        equity_path = (
            REPORTS_DIR
            / "binary_strategy_backtest_equity_v3.csv"
        )

        equity_df.to_csv(
            equity_path,
            index=False,
        )

    else:

        equity_path = None

    metadata = {
        "version":
            "v3",
        "initial_capital":
            INITIAL_CAPITAL,
        "capital_per_symbol":
            CAPITAL_PER_SYMBOL,
        "symbols":
            SYMBOLS,
        "catboost_weight":
            CATBOOST_WEIGHT,
        "gru_weight":
            GRU_WEIGHT,
        "confidence_threshold":
            CONFIDENCE_THRESHOLD,
        "holding_minutes":
            HOLDING_MINUTES,
        "entry_rule":
            "next 1-minute candle open after signal",
        "exit_rule":
            "5 minutes after entry at candle open",
        "round_trip_costs":
            ROUND_TRIP_COSTS,
        "future_target_filtering":
            False,
        "buy_hold_equal_weight_return":
            buy_hold,
        "research_note":
            (
                "Predictions are generated for all valid test "
                "timestamps. Future returns are not used to "
                "filter signal opportunities."
            ),
        "execution_note":
            (
                "This remains a simplified execution model. "
                "Order-book slippage, latency, funding, and "
                "exchange-specific fee schedules are not modeled."
            ),
    }

    metadata_path = (
        REPORTS_DIR
        / "binary_strategy_backtest_metadata_v3.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "\nReports saved:"
    )

    print(
        summary_path
    )

    if trades_path:
        print(
            trades_path
        )

    if equity_path:
        print(
            equity_path
        )

    print(
        metadata_path
    )


if __name__ == "__main__":
    main()