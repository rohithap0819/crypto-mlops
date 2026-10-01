import numpy as np
import pandas as pd

from src.models.prepare_sequence_data import (
    build_valid_end_indices,
)


def test_valid_contiguous_sequence():

    timestamps = pd.date_range(
        "2026-01-01",
        periods=70,
        freq="min",
    )

    indices = build_valid_end_indices(
        timestamps,
        sequence_length=60,
        forecast_horizon=5,
    )

    # With 70 continuous minutes:
    # 60-minute history + 5-minute future horizon
    # gives valid endpoints 59 through 64.
    expected_indices = np.arange(
        59,
        65,
        dtype=np.int32,
    )

    np.testing.assert_array_equal(
        indices,
        expected_indices,
    )


def test_gap_breaks_sequence():

    first_block = pd.date_range(
        "2026-01-01",
        periods=40,
        freq="min",
    )

    second_block = pd.date_range(
        "2026-01-01 02:00:00",
        periods=40,
        freq="min",
    )

    timestamps = list(
        first_block
    ) + list(
        second_block
    )

    indices = build_valid_end_indices(
        timestamps,
        sequence_length=20,
        forecast_horizon=5,
    )

    # First block:
    # valid endpoints = 19 through 34
    #
    # Second block:
    # valid endpoints = 59 through 74
    #
    # No valid sequence may cross the two-hour gap.

    expected_indices = (
        set(range(19, 35))
        | set(range(59, 75))
    )

    assert set(
        indices.tolist()
    ) == expected_indices


def test_future_horizon_is_required():

    timestamps = pd.date_range(
        "2026-01-01",
        periods=62,
        freq="min",
    )

    indices = build_valid_end_indices(
        timestamps,
        sequence_length=60,
        forecast_horizon=5,
    )

    # 60 history observations plus 5 future minutes
    # require at least 65 observations.
    assert len(indices) == 0


def test_sequence_index_type():

    timestamps = pd.date_range(
        "2026-01-01",
        periods=100,
        freq="min",
    )

    indices = build_valid_end_indices(
        timestamps,
        sequence_length=60,
        forecast_horizon=5,
    )

    assert isinstance(
        indices,
        np.ndarray,
    )

    assert indices.dtype == np.int32