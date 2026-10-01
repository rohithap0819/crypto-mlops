from pathlib import Path

import numpy as np
import pandas as pd


V1_FILE = Path(
    "data/processed/crypto_ml_dataset.parquet"
)

V2_FILE = Path(
    "data/processed/crypto_ml_dataset_v2.parquet"
)


def create_direction(
    future_return,
    volatility,
    multiplier,
):
    threshold = (
        volatility * multiplier
    )

    return np.select(
        [
            future_return > threshold,
            future_return < -threshold,
        ],
        [
            "UP",
            "DOWN",
        ],
        default="NEUTRAL",
    )


def main():

    print("=" * 70)
    print("V1 TARGET THRESHOLD DIAGNOSTIC")
    print("=" * 70)

    v1 = pd.read_parquet(V1_FILE)
    v2 = pd.read_parquet(V2_FILE)

    columns = [
        "open_time",
        "symbol",
        "future_return_5m",
        "volatility_15m",
        "target_direction_5m",
    ]

    v1 = v1[columns].copy()

    v2 = v2[columns].copy()

    v1 = v1.rename(
        columns={
            "future_return_5m":
                "future_return",
            "volatility_15m":
                "volatility",
            "target_direction_5m":
                "v1_direction",
        }
    )

    v2 = v2.rename(
        columns={
            "future_return_5m":
                "future_return_v2",
            "volatility_15m":
                "volatility_v2",
            "target_direction_5m":
                "v2_direction",
        }
    )

    merged = v1.merge(
        v2,
        on=[
            "open_time",
            "symbol",
        ],
        how="inner",
    )

    print()
    print(
        f"Matching rows: {len(merged):,}"
    )

    print()
    print(
        "Testing volatility multipliers..."
    )

    results = []

    multipliers = np.arange(
        0.10,
        2.01,
        0.05,
    )

    for multiplier in multipliers:

        predicted = create_direction(
            merged["future_return"],
            merged["volatility"],
            multiplier,
        )

        match_rate = (
            predicted
            == merged["v1_direction"]
        ).mean()

        results.append(
            {
                "multiplier": multiplier,
                "match_rate": match_rate,
            }
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "match_rate",
            ascending=False,
        )
    )

    print()
    print(
        results_df.head(10).to_string(
            index=False
        )
    )

    best_multiplier = (
        results_df.iloc[0]["multiplier"]
    )

    best_match = (
        results_df.iloc[0]["match_rate"]
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"Best multiplier: "
        f"{best_multiplier:.2f}"
    )

    print(
        f"Label match rate: "
        f"{best_match * 100:.4f}%"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Compare common candidate thresholds
    # --------------------------------------------------------

    print()
    print(
        "COMMON THRESHOLD CHECK"
    )

    for multiplier in [
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
    ]:

        predicted = create_direction(
            merged["future_return"],
            merged["volatility"],
            multiplier,
        )

        match_rate = (
            predicted
            == merged["v1_direction"]
        ).mean()

        print(
            f"Multiplier {multiplier:.2f}: "
            f"{match_rate * 100:.4f}% match"
        )

    print()
    print(
        "Current V2 multiplier is 0.50."
    )


if __name__ == "__main__":
    main()