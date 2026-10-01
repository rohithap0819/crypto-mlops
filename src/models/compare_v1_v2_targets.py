from pathlib import Path

import numpy as np
import pandas as pd


V1_FILE = Path(
    "data/processed/crypto_ml_dataset.parquet"
)

V2_FILE = Path(
    "data/processed/crypto_ml_dataset_v2.parquet"
)

OUTPUT_FILE = Path(
    "reports/v1_v2_target_comparison.csv"
)


def main():

    print("=" * 70)
    print("V1 VS V2 TARGET CONSISTENCY CHECK")
    print("=" * 70)

    v1 = pd.read_parquet(V1_FILE)
    v2 = pd.read_parquet(V2_FILE)

    required = [
        "open_time",
        "symbol",
        "future_return_5m",
        "target_direction_5m",
        "volatility_15m",
    ]

    for column in required:

        if column not in v1.columns:
            raise ValueError(
                f"V1 missing column: {column}"
            )

        if column not in v2.columns:
            raise ValueError(
                f"V2 missing column: {column}"
            )

    v1 = v1[
        required
    ].copy()

    v2 = v2[
        required
    ].copy()

    v1 = v1.rename(
        columns={
            "future_return_5m":
                "v1_future_return_5m",
            "target_direction_5m":
                "v1_direction",
            "volatility_15m":
                "v1_volatility_15m",
        }
    )

    v2 = v2.rename(
        columns={
            "future_return_5m":
                "v2_future_return_5m",
            "target_direction_5m":
                "v2_direction",
            "volatility_15m":
                "v2_volatility_15m",
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

    print(
        f"V1 rows: {len(v1):,}"
    )

    print(
        f"V2 rows: {len(v2):,}"
    )

    # --------------------------------------------------------
    # Future return comparison
    # --------------------------------------------------------

    return_difference = (
        merged["v1_future_return_5m"]
        - merged["v2_future_return_5m"]
    ).abs()

    volatility_difference = (
        merged["v1_volatility_15m"]
        - merged["v2_volatility_15m"]
    ).abs()

    print()
    print("FUTURE RETURN COMPARISON")
    print(
        f"Mean absolute difference: "
        f"{return_difference.mean():.12f}"
    )

    print(
        f"Maximum absolute difference: "
        f"{return_difference.max():.12f}"
    )

    print()
    print("VOLATILITY COMPARISON")
    print(
        f"Mean absolute difference: "
        f"{volatility_difference.mean():.12f}"
    )

    print(
        f"Maximum absolute difference: "
        f"{volatility_difference.max():.12f}"
    )

    # --------------------------------------------------------
    # Direction comparison
    # --------------------------------------------------------

    direction_match = (
        merged["v1_direction"]
        == merged["v2_direction"]
    )

    direction_match_rate = (
        direction_match.mean()
    )

    print()
    print("DIRECTION COMPARISON")
    print(
        f"Direction match rate: "
        f"{direction_match_rate * 100:.4f}%"
    )

    print()

    confusion = pd.crosstab(
        merged["v1_direction"],
        merged["v2_direction"],
        rownames=["V1"],
        colnames=["V2"],
    )

    print("DIRECTION CROSS-TAB")
    print(confusion)

    # --------------------------------------------------------
    # Save detailed comparison
    # --------------------------------------------------------

    merged["return_difference"] = (
        merged["v1_future_return_5m"]
        - merged["v2_future_return_5m"]
    )

    merged["volatility_difference"] = (
        merged["v1_volatility_15m"]
        - merged["v2_volatility_15m"]
    )

    merged["direction_match"] = (
        direction_match
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Detailed comparison saved to: "
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("TARGET CONSISTENCY CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()