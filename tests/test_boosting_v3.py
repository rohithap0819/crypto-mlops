import pandas as pd

from src.models.train_boosting_v3 import (
    get_classification_models,
    get_regression_models,
    load_feature_columns,
    prepare_features,
)


def test_v3_feature_columns():
    columns = load_feature_columns()

    assert len(columns) == 61


def test_v3_prepare_features():
    columns = load_feature_columns()

    data = {
        "symbol": ["BTCUSDT"],
    }

    for column in columns:
        data[column] = [0.01]

    df = pd.DataFrame(data)

    X = prepare_features(
        df,
        columns,
    )

    assert not X.empty
    assert X.shape[1] == len(columns) + 5


def test_v3_regression_models():
    models = get_regression_models()

    assert len(models) == 3
    assert "XGBoost" in models
    assert "LightGBM" in models
    assert "CatBoost" in models


def test_v3_classification_models():
    models = get_classification_models()

    assert len(models) == 3
    assert "XGBoost" in models
    assert "LightGBM" in models
    assert "CatBoost" in models