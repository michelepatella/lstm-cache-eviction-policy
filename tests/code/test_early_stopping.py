"""test_early_stopping.py

This module contains a unit test for the EarlyStopping utility class used
during model training.

It validates the logic for tracking the best average loss, resetting
counters upon improvement, and correctly triggering the early stopping
flag when the specified patience is exceeded.

Functions:
    test_early_stopping_logic(
        patience: int,
        delta: float,
        losses: list[float],
        expected_stop: bool
    ) -> None:
        Validates the triggering logic across different loss sequences.
"""

import pytest

from components.const import (
    EARLY_STOPPING_TRIGGERED,
    EARLY_STOPPING_UNTRIGGERED,
)
from components.training.callbacks.early_stopping import EarlyStopping


@pytest.mark.code_early_stopping_logic
@pytest.mark.parametrize(
    "patience, delta, losses, expected_stop",
    [
        # Case 1: Constant improvement, should not stop
        (3, 0.0, [10.0, 9.0, 8.0, 7.0], EARLY_STOPPING_UNTRIGGERED),
        # Case 2: No improvement, should stop after 3 epochs
        (3, 0.0, [10.0, 10.0, 10.0, 10.0], EARLY_STOPPING_TRIGGERED),
        # Case 3: Improvement is less than delta, should count as no improvement
        (2, 0.5, [10.0, 9.8, 9.6], EARLY_STOPPING_TRIGGERED),
        # Case 4: Reset counter on significant improvement
        (2, 0.1, [10.0, 10.0, 9.0, 9.0], EARLY_STOPPING_UNTRIGGERED),
        # Case 5: Stop exactly at patience limit
        (2, 0.0, [10.0, 11.0, 12.0], EARLY_STOPPING_TRIGGERED),
    ],
)
def test_early_stopping_logic(
    patience: int,
    delta: float,
    losses: list[float],
    expected_stop: bool,
) -> None:
    """Tests the early stopping trigger logic with various loss sequences.

    This test ensures that the internal counter increments correctly when
    the loss does not improve beyond the delta, and that the early_stop
    flag is set only when the counter reaches the patience threshold.

    Args:
        patience (int): Number of epochs to wait.
        delta (float): Minimum improvement threshold.
        losses (list[float]): Sequence of average losses to simulate epochs.
        expected_stop (bool): Expected final state of the early_stop flag.

    Returns:
        None
    """
    # Initialize early stopping object
    es = EarlyStopping(patience=patience, delta=delta)

    # Check early stopping over all
    # the losses
    for loss in losses:
        es(loss)

    # Assert that early stopping triggers
    # when expected
    assert es.early_stop == expected_stop
