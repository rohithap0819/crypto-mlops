from pathlib import Path

import numpy as np
import pandas as pd


V1_FILE = Path(
    "data/processed/crypto_ml_dataset.parquet"
)


def main():

    print("=" * 70)
    print("DERIVE V1 DIRECTION THRESHOLD")
    print("=" * 70)

    df = pd.read_parquet(V1_FILE)

    required = [
        "future_return_5m",
        "volatility_15m",
        "target_direction_5m",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[
        required
    ].dropna()

    ratio = (
        df["future_return_5m"].abs()
        / df["volatility_15m"]
    )

    ratio = ratio.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    direction = (
        df.loc[
            ratio.index,
            "target_direction_5m"
        ]
    )

    neutral_ratios = ratio[
        direction == "NEUTRAL"
    ]

    directional_ratios = ratio[
        direction.isin(
            ["UP", "DOWN"]
        )
    ]

    if neutral_ratios.empty:
        raise ValueError(
            "No NEUTRAL samples found."
        )

    if directional_ratios.empty:
        raise ValueError(
            "No UP/DOWN samples found."
        )

    # For NEUTRAL:
    # abs(return) / volatility must be
    # <= multiplier.
    neutral_lower_bound = (
        neutral_ratios.max()
    )

    # For UP/DOWN:
    # abs(return) / volatility must be
    # > multiplier.
    directional_upper_bound = (
        directional_ratios.min()
    )

    print()
    print(
        f"V1 rows analysed: "
        f"{len(df):,}"
    )

    print()
    print(
        "Threshold constraints:"
    )

    print(
        f"NEUTRAL lower bound: "
        f"{neutral_lower_bound:.12f}"
    )

    print(
        f"UP/DOWN upper bound: "
        f"{directional_upper_bound:.12f}"
    )

    print()

    if (
        neutral_lower_bound
        < directional_upper_bound
    ):

        midpoint = (
            neutral_lower_bound
            + directional_upper_bound
        ) / 2

        print(
            "A single volatility multiplier "
            "can reproduce all V1 labels."
        )

        print()
        print(
            f"Valid multiplier range: "
            f"[{neutral_lower_bound:.12f}, "
            f"{directional_upper_bound:.12f})"
        )

        print(
            f"Suggested multiplier: "
            f"{midpoint:.12f}"
        )

    else:

        print(
            "No single constant multiplier "
            "can reproduce all V1 labels."
        )

        print(
            "This means the original V1 "
            "target logic used another condition."
        )

    # --------------------------------------------------------
    # Check 1.1 explicitly
    # --------------------------------------------------------

    multiplier = 1.1

    threshold = (
        df["volatility_15m"]
        * multiplier
    )

    recreated = np.select(
        [
            df["future_return_5m"]
            > threshold,

            df["future_return_5m"]
            < -threshold,
        ],
        [
            "UP",
            "DOWN",
        ],
        default="NEUTRAL",
    )

    match_rate = (
        recreated
        == df["target_direction_5m"]
    ).mean()

    print()
    print(
        f"Explicit 1.10 match rate: "
        f"{match_rate * 100:.6f}%"
    )


if __name__ == "__main__":
    main()