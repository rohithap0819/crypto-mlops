from src.models.train_tuned_candidates import (
    calculate_sample_weights,
    encode_target,
)


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


def test_encode_target():

    import pandas as pd

    y_train = pd.Series(
        ["DOWN", "NEUTRAL", "UP"]
    )

    y_validation = pd.Series(
        ["UP", "DOWN", "NEUTRAL"]
    )

    (
        train_encoded,
        validation_encoded,
        classes,
    ) = encode_target(
        y_train,
        y_validation,
    )

    assert classes == [
        "DOWN",
        "NEUTRAL",
        "UP",
    ]

    assert list(
        train_encoded
    ) == [0, 1, 2]

    assert list(
        validation_encoded
    ) == [2, 0, 1]