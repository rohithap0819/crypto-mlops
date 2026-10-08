"""
Backtest the frozen CatBoost + GRU crypto direction strategy.

Frozen from validation:
    CatBoost weight = 0.55
    GRU weight      = 0.45
    Confidence      >= 0.55

Strategies:
    1. Buy & Hold
    2. Always-trade ensemble
    3. Confidence-filtered ensemble
    4. Inverse previous-5m baseline

Backtest design:
    - Five symbols
    - Equal capital allocation across symbols
    - Long/short positions
    - Maximum one open trade per symbol
    - Fixed 5-minute holding period
    - No leverage
    - Configurable round-trip transaction cost
    - No funding cost modeled
    - No future information used for signal generation

IMPORTANT:
    The test set has previously been used for model evaluation,
    so this should be described as a final comparative backtest,
    not a pristine untouched holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from catboost import CatBoostClassifier


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

CATBOOST_WEIGHT = 0.55
GRU_WEIGHT = 0.45

CONFIDENCE_THRESHOLD = 0.55

HOLDING_MINUTES = 5

# We will evaluate several cost scenarios.
ROUND_TRIP_COSTS = [
    0.0000,  # 0 bps
    0.0010,  # 10 bps
    0.0020,  # 20 bps
    0.0030,  # 30 bps
]

INITIAL_CAPITAL = 100_000.0

# Equal allocation among five coins.
CAPITAL_PER_SYMBOL = (
    INITIAL_CAPITAL / len(SYMBOLS)
)


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


# ============================================================
# METRICS
# ============================================================

def calculate_max_drawdown(
    equity: pd.Series,
) -> float:

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
        - 1.0
    )

    return float(
        drawdown.min()
    )


def calculate_profit_factor(
    trade_returns: np.ndarray,
) -> float:

    profits = trade_returns[
        trade_returns > 0
    ]

    losses = trade_returns[
        trade_returns < 0
    ]

    if len(losses) == 0:
        return float("inf")

    return float(
        profits.sum()
        /
        abs(losses.sum())
    )


def calculate_trade_metrics(
    trades: pd.DataFrame,
    starting_capital: float,
) -> dict:

    if len(trades) == 0:

        return {
            "trades": 0,
            "win_rate": np.nan,
            "average_trade_return": np.nan,
            "profit_factor": np.nan,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "ending_capital": starting_capital,
        }

    returns = trades[
        "net_return"
    ].to_numpy(
        dtype=float
    )

    equity = (
        starting_capital
        * (
            1.0
            +
            pd.Series(
                returns
            ).cumsum()
        )
    )

    # The above is simple cumulative P&L based on a fixed
    # capital-per-trade allocation. We also calculate total
    # compounded return separately below.
    compounded_equity = (
        starting_capital
        *
        np.cumprod(
            1.0 + returns
        )
    )

    ending_capital = float(
        compounded_equity[-1]
    )

    total_return = (
        ending_capital
        /
        starting_capital
        - 1.0
    )

    max_drawdown = calculate_max_drawdown(
        pd.Series(
            compounded_equity
        )
    )

    return {
        "trades": int(
            len(trades)
        ),
        "win_rate": float(
            np.mean(
                returns > 0
            )
        ),
        "average_trade_return":
            float(
                returns.mean()
            ),
        "median_trade_return":
            float(
                np.median(
                    returns
                )
            ),
        "profit_factor":
            calculate_profit_factor(
                returns
            ),
        "total_return":
            float(
                total_return
            ),
        "max_drawdown":
            float(
                max_drawdown
            ),
        "ending_capital":
            ending_capital,
    }


# ============================================================
# STRATEGY SIGNALS
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

    # Always-trade:
    # p >= 0.50 -> LONG
    # p < 0.50  -> SHORT
    df[
        "always_trade_signal"
    ] = np.where(
        df[
            "ensemble_probability"
        ] >= 0.50,
        1,
        -1,
    )

    confidence = np.maximum(
        df[
            "ensemble_probability"
        ],
        1.0
        -
        df[
            "ensemble_probability"
        ],
    )

    df[
        "confidence"
    ] = confidence

    # Selective:
    # >= 0.55 -> direction
    # otherwise -> 0 / no trade
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
                -
                CONFIDENCE_THRESHOLD
            ),
            -1,
            0,
        ),
    )

    # Inverse momentum:
    # previous 5-minute return < 0 -> LONG
    # previous 5-minute return > 0 -> SHORT
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
# STRATEGY BACKTEST
# ============================================================

def backtest_strategy(
    symbol_df: pd.DataFrame,
    signal_column: str,
    round_trip_cost: float,
    capital: float,
    strategy_name: str,
) -> pd.DataFrame:

    symbol_df = symbol_df.sort_values(
        "open_time"
    ).reset_index(drop=True)

    trades = []

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

        # Need a realized 5-minute outcome.
        if pd.isna(
            row["future_return_5m"]
        ):
            i += 1
            continue

        entry_time = row[
            "open_time"
        ]

        entry_price = float(
            row["close"]
        )

        realized_market_return = float(
            row[
                "future_return_5m"
            ]
        )

        # Direction-adjusted gross return.
        gross_return = (
            signal
            *
            realized_market_return
        )

        # Round-trip costs are applied once.
        net_return = (
            gross_return
            -
            round_trip_cost
        )

        exit_time = (
            entry_time
            +
            pd.Timedelta(
                minutes=HOLDING_MINUTES
            )
        )

        trades.append(
            {
                "strategy":
                    strategy_name,
                "symbol":
                    row["symbol"],
                "entry_time":
                    entry_time,
                "exit_time":
                    exit_time,
                "direction":
                    "LONG"
                    if signal == 1
                    else "SHORT",
                "entry_price":
                    entry_price,
                "future_market_return":
                    realized_market_return,
                "gross_return":
                    gross_return,
                "cost":
                    round_trip_cost,
                "net_return":
                    net_return,
                "capital_before":
                    capital,
            }
        )

        # One open trade per symbol.
        #
        # Move forward by approximately five minutes.
        # This avoids overlapping positions.
        current_time = entry_time

        i += 1

        while i < len(symbol_df):

            next_time = symbol_df.iloc[
                i
            ][
                "open_time"
            ]

            if (
                next_time
                >= exit_time
            ):
                break

            i += 1

    return pd.DataFrame(
        trades
    )


# ============================================================
# BUY & HOLD
# ============================================================

def calculate_buy_hold(
    symbol_df: pd.DataFrame,
) -> float:

    symbol_df = symbol_df.sort_values(
        "open_time"
    )

    if len(symbol_df) < 2:
        return np.nan

    first_price = float(
        symbol_df.iloc[0]["close"]
    )

    last_price = float(
        symbol_df.iloc[-1]["close"]
    )

    return (
        last_price
        /
        first_price
        - 1.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRYPTO ENSEMBLE BACKTEST")
    print("=" * 70)

    feature_columns = (
        load_feature_columns()
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

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
    # Prepare model datasets
    # --------------------------------------------------------

    X_train, y_train, train_clean = (
        make_binary_dataset(
            train_df,
            feature_columns,
        )
    )

    X_validation, y_validation, validation_clean = (
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

    # --------------------------------------------------------
    # Load saved GRU test predictions
    #
    # We use the prediction file produced by the previous
    # frozen-confidence evaluator if available.
    # Otherwise, explicitly tell the user to run that stage.
    # --------------------------------------------------------

    confidence_results_file = (
        REPORTS_DIR
        / "binary_ensemble_confidence_test_v1.json"
    )

    if not confidence_results_file.exists():

        raise FileNotFoundError(
            "Frozen confidence test report not found.\n"
            "Run:\n"
            "python -m src.models.evaluate_ensemble_confidence_test"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Recreate GRU predictions by importing the existing
    # evaluator helper would couple scripts unnecessarily.
    #
    # Instead, use the previously generated ensemble test
    # comparison process through the per-row predictions.
    #
    # We expect the confidence evaluator to have already
    # generated no row-level file, so we construct GRU
    # predictions here using the same trained checkpoint.
    # --------------------------------------------------------

    from src.models.binary_sequence_models import BinaryGRU

    sequence_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "sequence_v1"
        / "test"
    )

    gru_model = BinaryGRU(
        input_size=61,
        hidden_size=64,
        dropout=0.2,
        num_symbols=len(SYMBOLS),
        symbol_embedding_dim=4,
    )

    checkpoint = torch.load(
        PROJECT_ROOT
        / "models"
        / "binary_sequence_v1"
        / "gru_binary_best.pt",
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        gru_model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )
    else:
        gru_model.load_state_dict(
            checkpoint
        )

    gru_model.eval()


    gru_records = []

    print(
        "\nGenerating GRU test probabilities..."
    )

    for symbol_id, symbol in enumerate(
        SYMBOLS
    ):

        X = np.load(
            sequence_dir
            / f"{symbol}_X.npy"
        )

        valid_indices = np.load(
            sequence_dir
            / f"{symbol}_valid_indices.npy"
        ).astype(np.int64)

        return_5m = np.load(
            sequence_dir
            / f"{symbol}_return_5m.npy"
        )

        times = np.load(
            sequence_dir
            / f"{symbol}_time.npy"
        )

        valid_mask = (
            valid_indices
            >= 59
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
                    i - 59:
                    i + 1
                ]
                for i in valid_indices
            ],
            dtype=np.float32,
        )

        symbol_ids = np.full(
            len(sequences),
            symbol_id,
            dtype=np.int64,
        )

        predictions = []

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

                xb = torch.tensor(
                    sequences[start:end],
                    dtype=torch.float32,
                )

                sb = torch.tensor(
                    symbol_ids[start:end],
                    dtype=torch.long,
                )

                logits = gru_model(
                    xb,
                    sb,
                )

                predictions.append(
                    torch.sigmoid(
                        logits
                    ).numpy()
                )

        predictions = np.concatenate(
            predictions
        )

        sequence_times = pd.to_datetime(
            endpoint_times,
            unit="us",
            utc=True,
        )

        gru_records.append(
            pd.DataFrame(
                {
                    "symbol":
                        symbol,
                    "open_time":
                        sequence_times,
                    "gru_probability":
                        predictions,
                }
            )
        )

    gru_predictions = pd.concat(
        gru_records,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Prepare test rows
    # --------------------------------------------------------

    test_clean = test_clean.copy()

    test_clean[
        "catboost_probability"
    ] = catboost_probability

    test_clean[
        "open_time"
    ] = pd.to_datetime(
        test_clean[
            "open_time"
        ],
        utc=True,
    )

    # --------------------------------------------------------
    # Merge with GRU
    # --------------------------------------------------------

    merged = test_clean[
        [
            "symbol",
            "open_time",
            "close",
            "return_5m",
            "future_return_5m",
        ]
        + [
            "catboost_probability"
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
        f"\nMatched test rows: "
        f"{len(merged):,}"
    )

    if len(merged) < 100_000:

        raise ValueError(
            "Too few rows matched between "
            "CatBoost and GRU."
        )

    merged = build_signals(
        merged
    )

    # --------------------------------------------------------
    # Backtest all strategies under all costs
    # --------------------------------------------------------

    summary_rows = []
    all_trades = []

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

        strategies = {
            "Always_Trade_Ensemble":
                "always_trade_signal",
            "Confidence_Filtered_Ensemble":
                "confidence_signal",
            "Inverse_5m":
                "inverse_5m_signal",
        }

        for strategy_name, signal_column in (
            strategies.items()
        ):

            strategy_trades = []

            for symbol in SYMBOLS:

                symbol_df = merged[
                    merged["symbol"]
                    == symbol
                ].copy()

                symbol_trades = (
                    backtest_strategy(
                        symbol_df,
                        signal_column,
                        cost,
                        CAPITAL_PER_SYMBOL,
                        strategy_name,
                    )
                )

                strategy_trades.append(
                    symbol_trades
                )

            if strategy_trades:

                trades = pd.concat(
                    strategy_trades,
                    ignore_index=True,
                )

            else:

                trades = pd.DataFrame()

            if len(trades) > 0:

                # Equal capital per symbol means each trade
                # return is applied to its own 20% capital bucket.
                #
                # For summary purposes we calculate trade-level
                # statistics and aggregate P&L contribution.
                capital_weight = (
                    CAPITAL_PER_SYMBOL
                    /
                    INITIAL_CAPITAL
                )

                weighted_trade_returns = (
                    trades[
                        "net_return"
                    ]
                    * capital_weight
                )

                ending_capital = (
                    INITIAL_CAPITAL
                    *
                    np.prod(
                        1.0
                        +
                        weighted_trade_returns.to_numpy()
                    )
                )

                total_return = (
                    ending_capital
                    /
                    INITIAL_CAPITAL
                    - 1.0
                )

                win_rate = float(
                    np.mean(
                        trades[
                            "net_return"
                        ]
                        > 0
                    )
                )

                profit_factor = (
                    calculate_profit_factor(
                        trades[
                            "net_return"
                        ].to_numpy()
                    )
                )

                max_drawdown = 0.0

                if len(
                    weighted_trade_returns
                ) > 0:

                    equity = (
                        INITIAL_CAPITAL
                        *
                        np.cumprod(
                            1.0
                            +
                            weighted_trade_returns.to_numpy()
                        )
                    )

                    max_drawdown = (
                        calculate_max_drawdown(
                            pd.Series(
                                equity
                            )
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
                        "trades":
                            len(trades),
                        "win_rate":
                            win_rate,
                        "profit_factor":
                            profit_factor,
                        "total_return":
                            total_return,
                        "max_drawdown":
                            max_drawdown,
                        "ending_capital":
                            ending_capital,
                    }
                )

                trades[
                    "round_trip_cost"
                ] = cost

                all_trades.append(
                    trades
                )

                print(
                    f"{strategy_name:30s} "
                    f"Trades={len(trades):5d} "
                    f"Win={win_rate * 100:6.2f}% "
                    f"Return={total_return * 100:8.2f}% "
                    f"MaxDD={max_drawdown * 100:8.2f}% "
                    f"PF={profit_factor:6.3f}"
                )

    # --------------------------------------------------------
    # Buy & Hold diagnostics
    # --------------------------------------------------------

    buy_hold_rows = []

    for symbol in SYMBOLS:

        symbol_df = merged[
            merged["symbol"]
            == symbol
        ].sort_values(
            "open_time"
        )

        buy_hold_return = (
            calculate_buy_hold(
                symbol_df
            )
        )

        buy_hold_rows.append(
            {
                "symbol": symbol,
                "buy_hold_return":
                    buy_hold_return,
            }
        )

    buy_hold_df = pd.DataFrame(
        buy_hold_rows
    )

    equal_weight_buy_hold = float(
        buy_hold_df[
            "buy_hold_return"
        ].mean()
    )

    print(
        "\nEqual-weight Buy & Hold "
        f"return: "
        f"{equal_weight_buy_hold * 100:.2f}%"
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
        / "binary_strategy_backtest_v1.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    if all_trades:

        trades_df = pd.concat(
            all_trades,
            ignore_index=True,
        )

        trades_path = (
            REPORTS_DIR
            / "binary_strategy_backtest_trades_v1.csv"
        )

        trades_df.to_csv(
            trades_path,
            index=False,
        )

    else:

        trades_path = None

    buy_hold_path = (
        REPORTS_DIR
        / "binary_strategy_backtest_buy_hold_v1.csv"
    )

    buy_hold_df.to_csv(
        buy_hold_path,
        index=False,
    )

    metadata = {
        "catboost_weight":
            CATBOOST_WEIGHT,
        "gru_weight":
            GRU_WEIGHT,
        "confidence_threshold":
            CONFIDENCE_THRESHOLD,
        "holding_minutes":
            HOLDING_MINUTES,
        "initial_capital":
            INITIAL_CAPITAL,
        "capital_per_symbol":
            CAPITAL_PER_SYMBOL,
        "symbols":
            SYMBOLS,
        "round_trip_costs_tested":
            ROUND_TRIP_COSTS,
        "buy_hold_equal_weight_return":
            equal_weight_buy_hold,
        "methodology_note":
            (
                "One open trade per symbol. "
                "Each trade holds for five minutes. "
                "Long/short positions are equally sized. "
                "Funding is not modeled."
            ),
        "test_usage_note":
            (
                "The test period has previously been used "
                "for model evaluation; this is a comparative "
                "backtest, not a pristine untouched holdout."
            ),
    }

    metadata_path = (
        REPORTS_DIR
        / "binary_strategy_backtest_metadata_v1.json"
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

    print(
        buy_hold_path
    )

    print(
        metadata_path
    )


if __name__ == "__main__":
    main()