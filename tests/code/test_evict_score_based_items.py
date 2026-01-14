"""test_evict_score_based_items.py

This module contains a unit test for the score-based cache eviction logic.

It validates the score-based item eviction function, ensuring that items with
the lowest scores are prioritized for eviction, excluded keys are respected,
and the requested number of evictions is correctly handled even with constraints.

Functions:
    test_evict_score_based_items_logic(
        keys_in_cache: list[int],
        key_scores: list[float],
        excluded_keys: list[int],
        num_evictions: int,
        expected_evicted: list[int]
    ) -> None:
        Validates the core eviction logic using parameterized scenarios.
"""

import pytest

from components.caches.implementations.items.evictions.score_based_evictor import (
    evict_score_based_items,
)


@pytest.mark.code_evict_score_based_items_logic
@pytest.mark.parametrize(
    "keys_in_cache, key_scores, excluded_keys, num_evictions, expected_evicted",
    [
        # Case 1: Basic eviction (lowest scores first: 20 and 40)
        (
            [10, 20, 30, 40],
            [
                0.0
                if i not in [10, 20, 30, 40]
                else {10: 0.9, 20: 0.1, 30: 0.5, 40: 0.2}[i]
                for i in range(50)
            ],
            [],
            2,
            [20, 40],
        ),
        # Case 2: With exclusions (skip key 1 even if it's the lowest)
        ([1, 2, 3], [0.0, 0.1, 0.2, 0.3], [1], 1, [2]),
        # Case 3: Overflow (request more evictions than items available)
        (
            [100, 101],
            [
                0.5 if i == 100 else 0.6 if i == 101 else 0.0
                for i in range(200)
            ],
            [],
            10,
            [100, 101],
        ),
        # Case 4: Empty cache
        ([], [0.1, 0.2], [], 1, []),
    ],
)
def test_evict_score_based_items_logic(
    keys_in_cache: list[int],
    key_scores: list[float],
    excluded_keys: list[int],
    num_evictions: int,
    expected_evicted: list[int],
) -> None:
    """Tests various eviction scenarios using parameterization.

    Args:
        keys_in_cache (list[int]): Keys currently in the cache.
        key_scores (list[float]): Scores for all possible keys.
        excluded_keys (list[int]): Keys to protect from eviction.
        num_evictions (int): Number of items to remove.
        expected_evicted (list[int]): The expected list of evicted keys.
    """
    # Get the keys to evict
    evicted = evict_score_based_items(
        keys_in_cache,
        key_scores,
        excluded_keys,
        num_evictions,
    )

    # Assert that the number of eviction
    # matches the expected one
    assert len(evicted) == len(expected_evicted)

    # Assert that the evicted keys match
    # the expected ones
    assert evicted == expected_evicted
