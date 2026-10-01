"""
Diagnose predictability at different forecast horizons.

Horizons:
    5 minutes
    15 minutes
    30 minutes
    60 minutes

Uses the existing V2 validation split only.

For each horizon:
- Future return
- Zero-return baseline
- Same-horizon momentum baseline
- RMSE
- MAE
- R2
- Target standard deviation
- Correlation between current and future same-horizon return

All calculations are performed separately within each symbol
to prevent any cross-asset contamination.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits_v2"
    / "validation.parquet"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

HORIZONS = [
    5,
    15,
    30,
    60,
]

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]


# ============================================================
# METRICS
# ============================================================

def regression_metrics(
    actual,
    predicted,
):
    """
    Calculate regression metrics.
    """

    return {
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "r2": float(
            r2_score(
                actual,
                predicted,
            )
        ),
    }


def safe_correlation(
    first,
    second,
):
    """
    Calculate Pearson correlation safely.
    """

    if len(first) < 2:
        return float("nan")

    if np.std(first) == 0:
        return float("nan")

    if np.std(second) == 0:
        return float("nan")

    return float(
        np.corrcoef(
            first,
            second,
        )[0, 1]
    )


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(
        f"Validation file not found:\n"
        f"{VALIDATION_PATH}"
    )

print(
    f"Loading validation data:\n"
    f"{VALIDATION_PATH}"
)

df = pd.read_parquet(
    VALIDATION_PATH
)

print(
    f"Rows loaded: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "open_time",
    "close",
    "symbol",
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: "
            f"{column}"
        )


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = (
    df.sort_values(
        [
            "symbol",
            "open_time",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# STORAGE
# ============================================================

symbol_results = []


# ============================================================
# PER SYMBOL
# ============================================================

for symbol in SYMBOLS:

    print()
    print("=" * 70)
    print(
        f"SYMBOL: {symbol}"
    )
    print("=" * 70)

    symbol_df = (
        df[
            df["symbol"] == symbol
        ]
        .copy()
        .reset_index(drop=True)
    )

    if symbol_df.empty:

        print(
            "No rows found."
        )

        continue

    close = (
        symbol_df["close"]
        .astype(float)
    )

    print(
        f"Rows: {len(symbol_df):,}"
    )

    # --------------------------------------------------------
    # TEST EVERY HORIZON
    # --------------------------------------------------------

    for horizon in HORIZONS:

        # ----------------------------------------------------
        # CURRENT SAME-HORIZON RETURN
        #
        # Example for 15m:
        #
        # current return =
        # close[t] / close[t-15] - 1
        # ----------------------------------------------------

        current_return = (
            close
            / close.shift(horizon)
            - 1.0
        )

        # ----------------------------------------------------
        # FUTURE SAME-HORIZON RETURN
        #
        # Example for 15m:
        #
        # future return =
        # close[t+15] / close[t] - 1
        # ----------------------------------------------------

        future_return = (
            close.shift(-horizon)
            / close
            - 1.0
        )

        horizon_df = pd.DataFrame(
            {
                "current_return": (
                    current_return
                ),
                "future_return": (
                    future_return
                ),
            }
        )

        horizon_df = (
            horizon_df
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        current = (
            horizon_df[
                "current_return"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        future = (
            horizon_df[
                "future_return"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        # ----------------------------------------------------
        # ZERO-RETURN BASELINE
        # ----------------------------------------------------

        zero_prediction = (
            np.zeros_like(
                future
            )
        )

        zero_metrics = (
            regression_metrics(
                future,
                zero_prediction,
            )
        )

        # ----------------------------------------------------
        # SAME-HORIZON MOMENTUM BASELINE
        # ----------------------------------------------------

        momentum_prediction = (
            current
        )

        momentum_metrics = (
            regression_metrics(
                future,
                momentum_prediction,
            )
        )

        # ----------------------------------------------------
        # CORRELATION
        # ----------------------------------------------------

        current_future_corr = (
            safe_correlation(
                current,
                future,
            )
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        symbol_results.append(
            {
                "symbol": symbol,
                "horizon_minutes": horizon,
                "samples": len(future),
                "target_mean": float(
                    np.mean(future)
                ),
                "target_std": float(
                    np.std(future)
                ),
                "zero_rmse": (
                    zero_metrics["rmse"]
                ),
                "zero_mae": (
                    zero_metrics["mae"]
                ),
                "zero_r2": (
                    zero_metrics["r2"]
                ),
                "momentum_rmse": (
                    momentum_metrics["rmse"]
                ),
                "momentum_mae": (
                    momentum_metrics["mae"]
                ),
                "momentum_r2": (
                    momentum_metrics["r2"]
                ),
                "current_future_correlation": (
                    current_future_corr
                ),
            }
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print()
        print(
            f"Horizon: {horizon} minutes"
        )

        print(
            f"  Samples: "
            f"{len(future):,}"
        )

        print(
            f"  Target std: "
            f"{np.std(future):.8f}"
        )

        print(
            f"  Zero baseline:"
        )

        print(
            f"    RMSE: "
            f"{zero_metrics['rmse']:.8f}"
        )

        print(
            f"    MAE:  "
            f"{zero_metrics['mae']:.8f}"
        )

        print(
            f"    R2:   "
            f"{zero_metrics['r2']:.6f}"
        )

        print(
            f"  Momentum baseline:"
        )

        print(
            f"    RMSE: "
            f"{momentum_metrics['rmse']:.8f}"
        )

        print(
            f"    MAE:  "
            f"{momentum_metrics['mae']:.8f}"
        )

        print(
            f"    R2:   "
            f"{momentum_metrics['r2']:.6f}"
        )

        print(
            f"  Current/Future correlation: "
            f"{current_future_corr:.6f}"
        )


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    symbol_results
)


# ============================================================
# POOLED RESULTS
# ============================================================

pooled_results = []

for horizon in HORIZONS:

    horizon_parts = []

    for symbol in SYMBOLS:

        symbol_df = (
            df[
                df["symbol"] == symbol
            ]
            .copy()
        )

        close = (
            symbol_df["close"]
            .astype(float)
        )

        current_return = (
            close
            / close.shift(horizon)
            - 1.0
        )

        future_return = (
            close.shift(-horizon)
            / close
            - 1.0
        )

        temp = pd.DataFrame(
            {
                "current_return": (
                    current_return
                ),
                "future_return": (
                    future_return
                ),
            }
        )

        temp = (
            temp
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        horizon_parts.append(
            temp
        )

    pooled = pd.concat(
        horizon_parts,
        ignore_index=True,
    )

    current = (
        pooled[
            "current_return"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    future = (
        pooled[
            "future_return"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    zero_prediction = (
        np.zeros_like(
            future
        )
    )

    momentum_prediction = (
        current
    )

    zero_metrics = (
        regression_metrics(
            future,
            zero_prediction,
        )
    )

    momentum_metrics = (
        regression_metrics(
            future,
            momentum_prediction,
        )
    )

    pooled_results.append(
        {
            "horizon_minutes": horizon,
            "samples": len(future),
            "target_mean": float(
                np.mean(future)
            ),
            "target_std": float(
                np.std(future)
            ),
            "zero_rmse": (
                zero_metrics["rmse"]
            ),
            "zero_mae": (
                zero_metrics["mae"]
            ),
            "zero_r2": (
                zero_metrics["r2"]
            ),
            "momentum_rmse": (
                momentum_metrics["rmse"]
            ),
            "momentum_mae": (
                momentum_metrics["mae"]
            ),
            "momentum_r2": (
                momentum_metrics["r2"]
            ),
            "current_future_correlation": (
                safe_correlation(
                    current,
                    future,
                )
            ),
        }
    )


pooled_df = pd.DataFrame(
    pooled_results
)


# ============================================================
# SAVE REPORTS
# ============================================================

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

per_symbol_path = (
    REPORTS_DIR
    / "forecast_horizon_diagnostics_v1.csv"
)

pooled_path = (
    REPORTS_DIR
    / "forecast_horizon_pooled_v1.csv"
)

results_df.to_csv(
    per_symbol_path,
    index=False,
)

pooled_df.to_csv(
    pooled_path,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print(
    "POOLED HORIZON SUMMARY"
)
print("=" * 70)

print()

print(
    pooled_df[
        [
            "horizon_minutes",
            "samples",
            "target_std",
            "zero_rmse",
            "zero_mae",
            "zero_r2",
            "momentum_rmse",
            "momentum_r2",
            "current_future_correlation",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 70)
print(
    "REPORTS SAVED"
)
print("=" * 70)

print(
    f"Per-symbol:"
)

print(
    per_symbol_path
)

print()

print(
    f"Pooled:"
)

print(
    pooled_path
)