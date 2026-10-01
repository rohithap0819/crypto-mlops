import pandas as pd
import pytest

from src.models.train_baselines import (
    FEATURE_COLUMNS,
    get_classification_models,
    get_regression_models,
    prepare_features,
)


def test_feature_columns_exist():
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            **{
                column: [0.01]
                for column in FEATURE_COLUMNS
            },
        }
    )

    X = prepare_features(df)

    assert not X.empty
    assert X.shape[1] == len(FEATURE_COLUMNS) + 5


def test_regression_models():
    models = get_regression_models()

    assert len(models) == 3

    assert "LinearRegression" in models
    assert "RandomForestRegressor" in models
    assert "XGBRegressor" in models


def test_classification_models():
    models = get_classification_models()

    assert len(models) == 3

    assert "LogisticRegression" in models
    assert "RandomForestClassifier" in models
    assert "XGBClassifier" in models