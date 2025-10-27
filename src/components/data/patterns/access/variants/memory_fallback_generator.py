import numpy as np

from components.logs.levels.error_logger import error


def generate_memory_fallback_pattern(
    memory_interval: int,
    memory_offset: int,
    requests: list[int],
    requests_count: int,
    keys_range: np.ndarray,
    zipf_probs: np.ndarray,
) -> int:
    """Determine the next key to be accessed according to the memory/fallback
       pattern.

    This function simulates accesses based on a memory interval, or zipfian
    accesses. If the current request index matches the memory  interval and
    enough history exists, it selects a key previously accessed memory offset
    steps back. Otherwise, it  selects a key randomly according to the provided
    Zipfian probabilities.

    Args:
        memory_interval (int): Number of requests between memory-based
                               accesses.
        memory_offset (int): Steps back in request history to pick the key.
        requests (list[int]): List of keys requested so far.
        requests_count (int): Number of requests generated so far.
        keys_range (np.ndarray): List of available keys.
        zipf_probs (np.ndarray): List of normalized Zipfian probabilities
                                 corresponding to keys range.

    Returns:
        int: Index of the next accessed key.

    Raises:
        RuntimeError: If generating the memory/fallback pattern fails:
            * Accessing the request history due to empty or too short
              list (IndexError, ValueError).
            * Random selection from Zipfian probabilities due to invalid
              shape or normalization (ValueError).
            * Using invalid arguments or data types (TypeError).
    """
    try:
        # Based on the memory interval
        if (requests_count % memory_interval == 0) and (
            requests_count > memory_offset
        ):
            # Requested key as the one requested
            # memory offset steps back in the history
            requested_key = requests[-memory_offset]
        else:
            # Determine the requested key randomly,
            # according to Zipfian key probabilities
            requested_key = np.random.choice(keys_range, p=zipf_probs)

        return requested_key
    except (IndexError, TypeError, ValueError) as e:
        msg = "Memory/fallback pattern generation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "memory_interval": memory_interval,
                "memory_offset": memory_offset,
                "requests_count": requests_count,
                "requests_num": len(requests) if requests else 0,
                "keys_range_len": (
                    len(keys_range) if keys_range is not None else 0
                ),
                "zipf_probs_shape": (
                    zipf_probs.shape if zipf_probs is not None else None
                ),
                "context": "Memory/fallback pattern generation",
            },
        )
        raise RuntimeError(msg) from e
