"""test_class_weight_calculation.py

This module contains a parameterized unit test for the clas weight
calculation logic.

It ensures that class weights are correctly computed for imbalanced
datasets and that the function handles edge cases.

Functions:
    test_calculate_class_weight_logic(
        targets: torch.tensor,
        num_classes: int,
        weight_type: str,
        expected_higher_idx: int,
        expected_lower_idx: int
    ) -> None:
        Validates weight values, array shapes, and balancing logic.
"""

import numpy as np
import pytest
import torch

from components.loss.class_weight.calculator import calculate_class_weight


@pytest.mark.code_calculate_class_weight_logic
@pytest.mark.parametrize(
    "targets, num_classes, weight_type, expected_higher_idx, expected_lower_idx",
    [
        # Case 1: Imbalanced batch. Class 0 is rare (1), Class 1 is frequent (3).
        # Class 0 should have a higher weight than Class 1.
        (torch.tensor([0, 1, 1, 1]), 2, "balanced", 0, 1),
        # Case 2: More classes than present in batch.
        # Classes 2 and 3 are missing, should default to 1.0.
        (torch.tensor([0, 0, 1]), 4, "balanced", 1, 0),
    ],
)
def test_calculate_class_weight_logic(
    targets: torch.tensor,
    num_classes: int,
    weight_type: str,
    expected_higher_idx: int,
    expected_lower_idx: int,
) -> None:
    """Test the class weight calculation.

    Args:
        targets (torch.Tensor): Input labels.
        num_classes (int): Total classes.
        weight_type (str): Weighting strategy.
        expected_higher_idx (int): Index expected to have a larger weight.
        expected_lower_idx (int): Index expected to have a smaller weight.
    """
    # Calculate class weights
    class_weight = calculate_class_weight(targets, num_classes, weight_type)

    # Assert that the class weight is
    # of expected type
    assert isinstance(class_weight, np.ndarray)

    # Assert that class weight has expected shape
    assert class_weight.shape == (num_classes,)

    # Assert that class weight has expected dtype
    assert class_weight.dtype == np.float32

    # Assert that, in balanced mode, the rarer classes
    # have a larger weight
    assert class_weight[expected_higher_idx] > class_weight[expected_lower_idx]

    # Assert that all class weight
    # elements are positive
    assert np.all(class_weight > 0)
