import numpy as np

from components.logs.levels.error_logger import error


def generate_repetition_pattern(
    repetition_interval: int,
    repetition_offset: int,
    requests_count: int,
    requests: list[int],
    keys_range: np.ndarray,
    num_keys: int,
) -> int:
    """Determine the next key to be accessed according to repetition pattern.

    This function determines which is the next key to be accessed according
    to repetition pattern. This pattern simulates some keys are accessed
    regularly, while other accesses involve keys randomly selected from a
    subset of initial keys.

    Args:
        repetition_interval (int): Integer indicating how many times
                                   there will be a regular key access.
        repetition_offset (int): Integer determining how many steps back
                                 in requests to pick the next key.
        requests_count (int): Number of requests generated so far.
        requests (list[int]): List of requests generated so far.
        keys_range (np.array): List of keys.
        num_keys (int): Total number of keys.

    Returns:
          int: Index of the next accessed key.

    Raises:
        RuntimeError: If generating the repetition pattern fails:
            * Accessing the request history due to empty or too short list
              (IndexError, ValueError).
            * Random selection from keys due to invalid shape or empty
              subset (ValueError).
            * Using invalid arguments or data types (TypeError).
    """
    try:
        # Based on a repetition interval
        if requests_count % repetition_interval == 0:
            # Regular access to a key
            requested_key = requests[-repetition_offset]
        else:
            slice_length = max(1, (num_keys // repetition_offset))
            # Random access to a key taken from
            # a subset of initial keys
            requested_key = np.random.choice(keys_range[:slice_length])

        return requested_key
    except (IndexError, TypeError, ValueError) as e:
        msg = "Repetition pattern generation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "repetition_interval": repetition_interval,
                "repetition_offset": repetition_offset,
                "requests_count": requests_count,
                "requests_len": len(requests) if requests else 0,
                "keys_range_len": (
                    len(keys_range) if keys_range is not None else 0
                ),
                "keys_num": num_keys,
                "context": "Repetition pattern generation",
            },
        )
        raise RuntimeError(msg) from e
