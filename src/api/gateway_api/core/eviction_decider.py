from fastapi import HTTPException, status

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from api.kwargs.APIKwargs import APIKwargs


def decide_eviction(
    keys_in_cache: list[int],
    key_scores: dict[int, float],
    api_kwargs: APIKwargs,
) -> list[int]:
    """Decide which keys to evict from cache.

    This function determines which keys should be evicted
    from the cache based on their associated scores. Lower
    scores indicate less relevant keys and are prioritized
    for eviction. Excluded keys are filtered out from the
    eviction candidates.

    Args:
        keys_in_cache (list[int]): List of keys currently in the
                                   cache.
        key_scores (dict[int, float]): Mapping from keys to their
                                       scores.
        api_kwargs (APIKwargs): API configuration.

    Returns:
        list[int]: List of evicted keys.

    Raises:
        HTTPException: If eviction decision fails due to:
            * Missing or invalid key scores (KeyError, ValueError,
              TypeError).
            * Empty or malformed cache key list (ValueError).
    """
    try:
        debug(
            "Eviction decision started",
            extra={
                "keys_in_cache_num": len(keys_in_cache),
                "key_scores_num": len(key_scores),
                "keys_excluded": api_kwargs.excluded_keys,
                "evictions_requested_num": api_kwargs.num_evictions,
                "context": "Eviction decision",
            },
        )

        # The excluded keys are those indicated and
        # present in the cache at moment
        excluded_keys = set(api_kwargs.excluded_keys) & set(keys_in_cache)

        # Select candidate keys as those
        # currently in cache and different
        # from excluded ones
        candidate_keys = {
            int(k): score
            for k, score in key_scores.items()
            if int(k) in keys_in_cache and int(k) not in excluded_keys
        }

        # Select a number of keys to be evicted
        # equals the provided number of evictions, by
        # reading scores in descending order and picking
        # up the last ones
        num_evictions = min(api_kwargs.num_evictions, len(candidate_keys))
        sorted_candidate_keys = sorted(
            candidate_keys,
            key=lambda k: candidate_keys[k],
        )
        keys_to_evict = sorted_candidate_keys[:num_evictions]

        debug(
            "Eviction decision completed",
            extra={
                "keys_to_evict": keys_to_evict,
                "keys_to_evict_num": len(keys_to_evict),
                "context": "Eviction decision",
            },
        )

        return keys_to_evict
    except (KeyError, ValueError, TypeError) as e:
        error(
            "Eviction decision failed",
            extra={
                "exception": str(e),
                "keys_in_cache_num": len(keys_in_cache)
                if keys_in_cache
                else 0,
                "key_scores_num": len(key_scores) if key_scores else 0,
                "context": "Eviction decision",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
