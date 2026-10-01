from src.models.tune_optuna import (
    calculate_sample_weights,
    load_feature_columns,
)


def test_optuna_feature_columns():

    columns = load_feature_columns()

    assert len(columns) == 61


def test_sample_weights():

    import pandas as pd

    y = pd.Series(
        [0, 0, 0, 1, 1, 2]
    )

    weights = calculate_sample_weights(
        y
    )

    assert len(weights) == 6
    assert all(
        weight > 0
        for weight in weights
    )