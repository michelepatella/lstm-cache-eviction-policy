"""test_feature_engineering.py

This module contains parameterized unit tests for the feature engineering
components, specifically validating trigonometric time encoding and local
statistical calculations for request sequences.

It ensures that cyclical time representations and rolling window metrics
(frequency and recency) are computed correctly across various input scenarios.

Functions:
    test_encode_time_trigonometrically(
        timestamps: np.array,
        expected_sin: list[float],
        expected_cos: list[float]
    ) -> None:
        Verifies the sine and cosine encoding of timestamps using parameterization.
    test_calculate_local_frequencies(
        requests: list[int],
        seq_len: int,
        expected_local_frequencies: list[float]
    ) -> None:
        Validates the rolling window frequency calculation logic with multiple cases.
    test_calculate_local_recencies(
        requests: list[int],
        seq_len: int,
        expected_local_recencies: list[float]
    ) -> None:
        Validates the rolling window recency logic with multiple cases.
"""

import numpy as np
import pytest

from components.const import TIME_HOURS_IN_DAY
from components.dataset.features.local.local_frequencies_calculator import (
    calculate_local_frequencies,
)
from components.dataset.features.local.local_recencies_calculator import (
    calculate_local_recencies,
)
from components.time.transforms.trig_encoder import (
    encode_time_trigonometrically,
)


@pytest.mark.code_encode_time_trigonometrically
@pytest.mark.parametrize(
    "timestamps, expected_sin, expected_cos",
    [
        (
            np.array([0.0, 6.0, 12.0, 18.0]),
            [0.0, 1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0, 0.0],
        ),
        (np.array([0.0, 12.0]), [0.0, 0.0], [1.0, -1.0]),
        (np.array([12.0, 24.0]), [0.0, 0.0], [-1.0, 1.0]),
    ],
)
def test_encode_time_trigonometrically(
    timestamps: np.array,
    expected_sin: list[float],
    expected_cos: list[float],
) -> None:
    """Tests the trigonometric encoding of timestamps.

    This test verifies that timestamps are correctly mapped to
    sine and cosine values.

    Args:
        timestamps (np.ndarray): Input time array.
        expected_sin (list[float]): Expected sine components.
        expected_cos (list[float]): Expected cosine components.

    Returns:
        None
    """
    # Encode timestamps trigonometrically
    # producing sine and cosine components
    sin_time, cos_time = encode_time_trigonometrically(
        timestamps,
        TIME_HOURS_IN_DAY,
    )

    # Assert that sine and cosine time values
    # are those expected
    assert np.allclose(sin_time, expected_sin)
    assert np.allclose(cos_time, expected_cos)

    # Assert that sine and cosine time shapes
    # are those expected
    assert sin_time.shape == timestamps.shape
    assert cos_time.shape == timestamps.shape

    # Assert that sin and cosine components
    # are of expected types
    assert isinstance(sin_time, np.ndarray)
    assert isinstance(cos_time, np.ndarray)


@pytest.mark.code_calculate_local_frequencies
@pytest.mark.parametrize(
    "requests, seq_len, expected_local_frequencies",
    [
        ([1, 2, 1, 3, 1], 2, [0.5, 0.5, 0.5, 0.5, 0.5]),
        ([1, 1, 1], 3, [0.3333333, 0.6666666, 1.0]),
        ([1, 2, 3], 1, [1.0, 1.0, 1.0]),
    ],
)
def test_calculate_local_frequencies(
    requests: list[int],
    seq_len: int,
    expected_local_frequencies: list[float],
) -> None:
    """Validates the normalized local frequency calculation
    using multiple scenarios.

    This test ensures that for various sequences and window lengths, the
    frequency of each key is correctly counted and normalized.

    Args:
        requests (list[int]): Sequence of requested keys.
        seq_len (int): Rolling window size.
        expected_local_frequencies (list[float]): Expected normalized frequencies.

    Returns:
        None
    """
    # Calculate the local frequencies
    # given the requests and a sequence length
    local_frequencies = calculate_local_frequencies(requests, seq_len)

    # Assert that the result matches with
    # the expected one
    assert np.allclose(local_frequencies, expected_local_frequencies)

    # Assert that the result is of expected type
    assert isinstance(local_frequencies, np.ndarray)


@pytest.mark.code_calculate_local_recencies
@pytest.mark.parametrize(
    "requests, seq_len, expected_local_recencies",
    [
        ([1, 2, 1, 1], 3, [0.0, 0.0, 1.0 - (1 / 3), 1.0]),
        ([1, 1, 1], 2, [0.0, 1.0, 1.0]),
        ([1, 2, 3], 3, [0.0, 0.0, 0.0]),
    ],
)
def test_calculate_local_recencies(
    requests: list[int],
    seq_len: int,
    expected_local_recencies: list[float],
) -> None:
    """Validates the local recency calculation logic using multiple scenarios.

    This test checks the reverse distance from the last occurrence.
    It covers cases where items are repeated immediately, repeated within
    the window, or not present at all.

    Args:
        requests (list[int]): Sequence of requested keys.
        seq_len (int): Rolling window size.
        expected_local_recencies (list[float]): Expected normalized recencies.

    Returns:
        None
    """
    # Calculate the local recencies
    # given the requests and a sequence length
    local_recencies = calculate_local_recencies(requests, seq_len)

    # Assert that the result matches with
    # the expected one
    assert np.allclose(local_recencies, expected_local_recencies)

    # Assert that the result is of expected type
    assert isinstance(local_recencies, np.ndarray)
