"""
Corrected portfolio backtest for the frozen CatBoost + GRU strategy.

V2 corrections:
    - Independent equity account per coin.
    - Equal initial capital allocation across coins.
    - No overlapping position for the same coin.
    - Explicit entry/exit trade accounting.
    - Explicit transaction costs.
    - Portfolio equity reconstructed from per-coin equity.

Frozen model configuration:
    CatBoost = 0.55
    GRU      = 0.45
    Confidence threshold = 0.55

Important:
    The model's 5-minute target is used directly for this research
    backtest. This means the result evaluates the predictive signal
    under the same target definition used during model training.

A later deployment-grade backtest should additionally model:
    signal generation at candle close
    -> execution at next available price
    -> execution latency
    -> spread
    -> slippage.
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
    PROJECT_ROOT
    / "reports"
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

# Test several round-trip costs.
ROUND_TRIP_COSTS = [
    0.0000,  # 0 bps
    0.0010,  # 10 bps
    0.0020,  # 20 bps
    0.0030,  # 30 bps
]


# ============================================================
# FEATURE HELPERS
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


def make_binary_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
):

    df = df.copy()

    mask = (
        df[TARGET_COLUMN].notna()
        & (df[TARGET_COLUMN] != 0)
    )

    df = df.loc[
        mask
    ].reset_index(drop=True)

    X = prepare_features(
        df,
        feature_columns,
    )

    y = (
        df[TARGET_COLUMN] > 0
    ).astype(np.int64)

    return X, y, df


# ============================================================
# MODEL
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
        "Training CatBoost..."
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


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


# ============================================================
# GRU TEST PREDICTIONS
# ============================================================

def generate_gru_predictions():

    model = load_gru()

    records = []

    print(
        "\nGenerating GRU test probabilities..."
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

        return_5m = np.load(
            SEQUENCE_TEST_DIR
            / f"{symbol}_return_5m.npy"
        )

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

        target_returns = (
            return_5m[
                valid_indices
            ]
        )

        endpoint_times = (
            times[
                valid_indices
            ]
        )

        # Remove exact-zero targets.
        nonzero_mask = (
            target_returns != 0
        )

        valid_indices = (
            valid_indices[
                nonzero_mask
            ]
        )

        endpoint_times = (
            endpoint_times[
                nonzero_mask
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
                    "symbol": symbol,
                    "open_time": pd.to_datetime(
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
            f"{len(probabilities):,} sequences"
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
    # >= 0.50 -> long
    # <  0.50 -> short
    df[
        "always_trade_signal"
    ] = np.where(
        df[
            "ensemble_probability"
        ] >= 0.50,
        1,
        -1,
    )

    # Confidence filtered:
    # >= 0.55 -> long
    # <= 0.45 -> short
    # otherwise -> no trade
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
            (
                1.0
                - CONFIDENCE_THRESHOLD
            ),
            -1,
            0,
        ),
    )

    # Inverse momentum baseline.
    df[
        "inverse_5m_signal"
    ] = np.where(
        df[
            "return_5m"
        ] < 0,
        1,
        -1,
    )

    return df


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_symbol(
    symbol_df: pd.DataFrame,
    signal_column: str,
    symbol_initial_capital: float,
    round_trip_cost: float,
    strategy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    symbol_df = (
        symbol_df
        .sort_values(
            "open_time"
        )
        .reset_index(
            drop=True
        )
    )

    capital = (
        symbol_initial_capital
    )

    equity_records = []

    trades = []

    i = 0

    while i < len(symbol_df):

        row = symbol_df.iloc[i]

        timestamp = row[
            "open_time"
        ]

        equity_records.append(
            {
                "open_time":
                    timestamp,
                "symbol":
                    row["symbol"],
                "equity":
                    capital,
            }
        )

        signal = int(
            row[
                signal_column
            ]
        )

        if signal == 0:
            i += 1
            continue

        if pd.isna(
            row[
                "future_return_5m"
            ]
        ):
            i += 1
            continue

        gross_return = (
            signal
            *
            float(
                row[
                    "future_return_5m"
                ]
            )
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

        exit_time = (
            timestamp
            + pd.Timedelta(
                minutes=5
            )
        )

        trades.append(
            {
                "strategy":
                    strategy_name,
                "symbol":
                    row["symbol"],
                "entry_time":
                    timestamp,
                "exit_time":
                    exit_time,
                "direction":
                    (
                        "LONG"
                        if signal == 1
                        else "SHORT"
                    ),
                "future_return_5m":
                    float(
                        row[
                            "future_return_5m"
                        ]
                    ),
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

        capital = capital_after

        # Prevent overlapping trades.
        # Move to the first row at or after the
        # five-minute exit.
        i += 1

        while i < len(symbol_df):

            if (
                symbol_df.iloc[i][
                    "open_time"
                ]
                >= exit_time
            ):
                break

            i += 1

    equity_records.append(
        {
            "open_time":
                symbol_df.iloc[-1][
                    "open_time"
                ],
            "symbol":
                symbol_df.iloc[-1][
                    "symbol"
                ],
            "equity":
                capital,
        }
    )

    return (
        pd.DataFrame(
            trades
        ),
        pd.DataFrame(
            equity_records
        ),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_capital: float,
) -> dict:

    final_equity = float(
        equity[
            "portfolio_equity"
        ].iloc[-1]
    )

    total_return = (
        final_equity
        /
        initial_capital
        - 1.0
    )

    if len(trades) == 0:

        return {
            "trades": 0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "total_return":
                total_return,
            "max_drawdown": 0.0,
            "ending_capital":
                final_equity,
        }

    trade_returns = (
        trades[
            "net_return"
        ].to_numpy(
            dtype=float
        )
    )

    wins = (
        trade_returns > 0
    )

    profits = trade_returns[
        trade_returns > 0
    ]

    losses = trade_returns[
        trade_returns < 0
    ]

    if len(losses) == 0:
        profit_factor = np.inf
    else:
        profit_factor = (
            profits.sum()
            /
            abs(
                losses.sum()
            )
        )

    peak = (
        equity[
            "portfolio_equity"
        ]
        .cummax()
    )

    drawdown = (
        equity[
            "portfolio_equity"
        ]
        /
        peak
        - 1.0
    )

    return {
        "trades":
            int(len(trades)),
        "win_rate":
            float(wins.mean()),
        "profit_factor":
            float(profit_factor),
        "total_return":
            float(total_return),
        "max_drawdown":
            float(drawdown.min()),
        "ending_capital":
            final_equity,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CORRECTED PORTFOLIO BACKTEST V2"
    )
    print("=" * 70)

    print(
        f"Initial capital: "
        f"{INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Capital per coin: "
        f"{CAPITAL_PER_SYMBOL:,.2f}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    feature_columns = (
        load_feature_columns()
    )

    print(
        "\nLoading train / validation / test..."
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
    # Prepare model data
    # --------------------------------------------------------

    X_train, y_train, _ = (
        make_binary_dataset(
            train_df,
            feature_columns,
        )
    )

    X_validation, y_validation, _ = (
        make_binary_dataset(
            validation_df,
            feature_columns,
        )
    )

    X_test, y_test, test_clean = (
        make_binary_dataset(
            test_df,
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

    # --------------------------------------------------------
    # CatBoost
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

    test_clean[
        "catboost_probability"
    ] = catboost_probability

    # --------------------------------------------------------
    # GRU
    # --------------------------------------------------------

    gru_predictions = (
        generate_gru_predictions()
    )

    test_clean[
        "open_time"
    ] = pd.to_datetime(
        test_clean[
            "open_time"
        ],
        utc=True,
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = test_clean[
        [
            "symbol",
            "open_time",
            "close",
            "return_5m",
            "future_return_5m",
            "catboost_probability",
        ]
    ].merge(
        gru_predictions,
        on=[
            "symbol",
            "open_time",
        ],
        how="inner",
    )

    print(
        f"\nMatched rows: "
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
            "always_trade_signal",
        "Confidence_Filtered_Ensemble":
            "confidence_signal",
        "Inverse_5m":
            "inverse_5m_signal",
    }

    summary_rows = []

    all_trade_rows = []

    all_equity_rows = []

    for cost in (
        ROUND_TRIP_COSTS
    ):

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

        for strategy_name, signal_column in (
            strategies.items()
        ):

            symbol_final_equities = []

            strategy_trade_frames = []

            strategy_equity_frames = []

            for symbol in SYMBOLS:

                symbol_df = merged[
                    merged["symbol"]
                    == symbol
                ].copy()

                trades, equity = (
                    simulate_symbol(
                        symbol_df,
                        signal_column,
                        CAPITAL_PER_SYMBOL,
                        cost,
                        strategy_name,
                    )
                )

                final_equity = float(
                    equity[
                        "equity"
                    ].iloc[-1]
                )

                symbol_final_equities.append(
                    final_equity
                )

                if len(trades) > 0:

                    trades[
                        "round_trip_cost"
                    ] = cost

                    strategy_trade_frames.append(
                        trades
                    )

                equity[
                    "strategy"
                ] = strategy_name

                equity[
                    "round_trip_cost"
                ] = cost

                strategy_equity_frames.append(
                    equity
                )

            # ------------------------------------------------
            # Combine five independent portfolios
            # ------------------------------------------------

            portfolio_final_equity = float(
                np.sum(
                    symbol_final_equities
                )
            )

            initial_portfolio = (
                INITIAL_CAPITAL
            )

            portfolio_return = (
                portfolio_final_equity
                /
                initial_portfolio
                - 1.0
            )

            if strategy_trade_frames:

                trades = pd.concat(
                    strategy_trade_frames,
                    ignore_index=True,
                )

            else:

                trades = pd.DataFrame()

            # Build synchronized portfolio equity.
            equity_events = []

            for equity in (
                strategy_equity_frames
            ):

                temp = equity[
                    [
                        "open_time",
                        "symbol",
                        "equity",
                    ]
                ].copy()

                equity_events.append(
                    temp
                )

            all_equity = pd.concat(
                equity_events,
                ignore_index=True,
            )

            portfolio_equity = (
                all_equity
                .groupby(
                    "open_time"
                )[
                    "equity"
                ]
                .sum()
                .sort_index()
                .rename(
                    "portfolio_equity"
                )
                .reset_index()
            )

            if len(portfolio_equity) > 0:

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

                max_drawdown = (
                    float(
                        drawdown.min()
                    )
                )

            else:

                max_drawdown = 0.0

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

                    profit_factor = (
                        profits.sum()
                        /
                        abs(
                            losses.sum()
                        )
                    )

            else:

                win_rate = np.nan
                profit_factor = np.nan

            summary_rows.append(
                {
                    "strategy":
                        strategy_name,
                    "round_trip_cost":
                        cost,
                    "cost_bps":
                        cost * 10000,
                    "trades":
                        int(
                            len(trades)
                        ),
                    "win_rate":
                        win_rate,
                    "profit_factor":
                        float(
                            profit_factor
                        ),
                    "total_return":
                        portfolio_return,
                    "max_drawdown":
                        max_drawdown,
                    "ending_capital":
                        portfolio_final_equity,
                }
            )

            print(
                f"{strategy_name:30s} "
                f"Trades={len(trades):6d} "
                f"Win={win_rate * 100:6.2f}% "
                f"Return={portfolio_return * 100:8.2f}% "
                f"MaxDD={max_drawdown * 100:8.2f}% "
                f"PF={profit_factor:7.3f}"
            )

            if len(trades) > 0:

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

    buy_hold_returns = []

    for symbol in SYMBOLS:

        symbol_df = merged[
            merged["symbol"]
            == symbol
        ].sort_values(
            "open_time"
        )

        first_price = float(
            symbol_df.iloc[0][
                "close"
            ]
        )

        last_price = float(
            symbol_df.iloc[-1][
                "close"
            ]
        )

        buy_hold_return = (
            last_price
            /
            first_price
            - 1.0
        )

        buy_hold_returns.append(
            buy_hold_return
        )

    equal_weight_buy_hold = float(
        np.mean(
            buy_hold_returns
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Equal-weight Buy & Hold: "
        f"{equal_weight_buy_hold * 100:.2f}%"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Save reports
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
        / "binary_strategy_backtest_v2.csv"
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
            / "binary_strategy_backtest_trades_v2.csv"
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
            / "binary_strategy_backtest_equity_v2.csv"
        )

        equity_df.to_csv(
            equity_path,
            index=False,
        )

    else:

        equity_path = None

    metadata = {
        "version":
            "backtest_v2",
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
        "round_trip_costs":
            ROUND_TRIP_COSTS,
        "methodology":
            {
                "separate_capital_per_symbol":
                    True,
                "overlapping_positions_same_symbol":
                    False,
                "holding_period_minutes":
                    5,
                "long_short":
                    True,
            },
        "buy_hold_equal_weight_return":
            equal_weight_buy_hold,
        "research_note":
            (
                "The strategy return is based on the existing "
                "future_return_5m target. It is therefore a "
                "research backtest of the predictive signal, "
                "not yet a deployment-grade execution simulator."
            ),
    }

    metadata_path = (
        REPORTS_DIR
        / "binary_strategy_backtest_metadata_v2.json"
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