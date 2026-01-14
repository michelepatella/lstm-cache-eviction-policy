"""test_data_splitting.py

This module contains a unit test for the dataset splitting logic.

It validates the function for splitting data, ensuring that DataFrames are
correctly sliced at the specified index and that the appropriate portion
is returned based on the user's requirements.

Functions:
    test_split_dataset_data_logic(
        rows: int,
        split_idx: int,
        take_first: bool,
        expected_len: int
    ) -> None:
        Validates the splitting logic with various indices and portions.
"""

import pandas as pd
import pytest

from components.const import LIST_FIRST_IDX
from components.dataset.splits.data_splitter import split_dataset_data


@pytest.mark.code_split_dataset_data_logic
@pytest.mark.parametrize(
    "rows, split_idx, take_first, expected_len",
    [
        # Case 1: Take the first 3 rows of 10
        (10, 3, True, 3),
        # Case 2: Take the second part starting from index 3 (7 rows remaining)
        (10, 3, False, 7),
        # Case 3: Split at 0 taking the first part (empty)
        (5, 0, True, 0),
        # Case 4: Split at 0 taking the second part (all)
        (5, 0, False, 5),
        # Case 5: Split at index equal to length taking first part (all)
        (5, 5, True, 5),
    ],
)
def test_split_dataset_data_logic(
    rows: int,
    split_idx: int,
    take_first: bool,
    expected_len: int,
) -> None:
    """Tests the logic of splitting a DataFrame into parts.

    This test ensures that the slicing logic correctly partitions the
    DataFrame rows according to the provided index and boolean flag.

    Args:
        rows (int): Number of rows to generate in the test DataFrame.
        split_idx (int): The index at which to split.
        take_first (bool): Flag to select the first or second portion.
        expected_len (int): Expected number of rows in the resulting DataFrame.

    Returns:
        None
    """
    # Create a dummy DataFrame
    df = pd.DataFrame({"col": range(rows)})

    # Split dataset
    data = split_dataset_data(df, split_idx, take_first)

    # Assert that the split data is still
    # a DataFrame
    assert isinstance(data, pd.DataFrame)

    # Assert that the length of split
    # data is the expected one
    assert len(data) == expected_len

    if expected_len > 0:
        if take_first:
            # Assert if it actually took the beginning
            assert data.iloc[LIST_FIRST_IDX]["col"] == LIST_FIRST_IDX
        else:
            # Assert if it actually started from the split index
            assert data.iloc[LIST_FIRST_IDX]["col"] == split_idx
