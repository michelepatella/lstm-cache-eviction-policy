"""helpers.py

This module provides utility functions for generating synthetic and
dataset-derived input data for testing the API.

It includes logic for sampling sequences of accesses from a processed
dataset and optionally introducing data corruptions to validate the
robustness of the API.

Functions:
    get_random_api_input(
        dataset: DataFrame,
        seq_len: int,
        corrupt: bool = False
    ) -> tuple[list[tuple[float, int]], list[int]]:
        Extracts a random sequence of accesses and current cache keys.
"""

import math
import random

from pandas import DataFrame

from components.dataset.rows.extractions.lasts_extractor import (
    extract_last_rows_from_dataset,
)


def get_random_api_input(
    dataset: DataFrame,
    seq_len: int,
    corrupt: bool = False,
) -> tuple[list[tuple[float, int]], list[int]]:
    """Generates a random set of input data for the API from a dataset.

    This function selects a random point in the provided dataset and extracts
    a sequence of recent accesses. It also simulates the current state of
    the cache by identifying unique keys within that sequence. Optionally,
    it can corrupt the timestamps to test API error handling.

    Args:
        dataset (DataFrame): The processed dataset containing access logs.
        seq_len (int): The required sequence length for the model input.
        corrupt (bool): If True, replaces all timestamps in the access
                        sequence with NaN values.

    Returns:
        tuple[list[tuple[float, int]], list[int]]:
            - last_accesses: A list of (timestamp, key) tuples.
            - keys_in_cache: A list of unique integer keys currently
                             assumed to be in the cache.
    """
    # Pick a dataset index at random
    # (ensuring enough past history)
    idx = random.randint(seq_len, len(dataset) - 1)

    # Extract last accesses
    last_accesses = extract_last_rows_from_dataset(
        idx,
        seq_len + seq_len,
        dataset,
    )

    # (Optionally) Corrupt last accesses
    if corrupt:
        last_accesses = [(math.nan, key) for _, key in last_accesses]

    # Assume keys in cache are all those
    # appearing in the last accesses
    keys_in_cache = [key for _, key in last_accesses]

    return last_accesses, keys_in_cache
