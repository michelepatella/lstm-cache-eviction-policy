from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def evict_score_based_items(
    keys_in_cache: list[int],
    key_scores: list[float],
    excluded_keys: list[int],
    num_evictions: int,
) -> list[int]:
    """Decide which items to evict from cache.

    This function determines which items should be evicted
    from the cache based on their associated scores. Lower
    scores indicate less relevant keys and are prioritized
    for eviction. Excluded keys are filtered out from the
    eviction candidates.

    Args:
        keys_in_cache (list[int]): List of keys currently in the
                                   cache.
        key_scores (list[float]): Mapping from keys to their
                                  scores.
        excluded_keys (list[int]): List of keys to exclude from eviction.
        num_evictions (int): Number of keys to evict.

    Returns:
        list[int]: List of evicted keys.

    Raises:
        RuntimeError: If eviction decision fails due to:
            * Missing or invalid key scores (KeyError, ValueError,
              TypeError).
            * Empty or malformed cache key list (ValueError).
    """
    try:
        debug(
            "Score based eviction started",
            extra={
                "keys_in_cache_num": len(keys_in_cache),
                "key_scores_num": len(key_scores),
                "keys_excluded": excluded_keys,
                "evictions_requested_num": num_evictions,
                "context": "Score based eviction",
            },
        )

        # The excluded keys are those indicated and
        # present in the cache at moment
        excluded_keys = set(excluded_keys) & set(keys_in_cache)

        # Select candidate keys as those
        # currently in cache and different
        # from excluded ones
        candidate_keys = {
            k: key_scores[k]
            for k in range(len(key_scores))
            if k in keys_in_cache and k not in excluded_keys
        }

        # Select a number of keys to be evicted
        # equals the provided number of evictions, by
        # reading scores in descending order and picking
        # up the last ones
        num_evictions = min(num_evictions, len(candidate_keys))
        sorted_candidate_keys = sorted(
            candidate_keys,
            key=lambda k: candidate_keys[k],
        )
        keys_to_evict = sorted_candidate_keys[:num_evictions]

        debug(
            "Score based eviction completed",
            extra={
                "keys_to_evict": keys_to_evict,
                "keys_to_evict_num": len(keys_to_evict),
                "context": "Score based eviction",
            },
        )

        return keys_to_evict
    except (KeyError, ValueError, TypeError) as e:
        msg = "Score based eviction failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "keys_in_cache_num": len(keys_in_cache)
                if keys_in_cache
                else 0,
                "key_scores_num": len(key_scores) if key_scores else 0,
                "context": "Score based eviction",
            },
        )
        raise RuntimeError(msg) from e
