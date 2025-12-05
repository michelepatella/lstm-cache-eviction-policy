"""local_recencies_calculator.py

Module dedicated to calculating the local recency (reverse distance from
the last occurrence) of requested items within a rolling time window.

The module processes a sequential list of requests and computes, for
each request, how many steps have passed since the same key was last seen
within the preceding sequence of a specified length. This calculation
provides a measure of temporal distance and memory decay of item access
at a specific point in the request stream. The more a key has been recently
requested, the higher the corresponding local recency value.

Functions:
    calculate_local_recency(
        requests: list[int],
        seq_len: int
    ) -> np.ndarray
        Calculates the reverse distance from the last occurrence for each request.
"""

import numpy as np

from components.const import LIST_LAST_IDX
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def calculate_local_recencies(requests: list[int], seq_len: int) -> np.ndarray:
    """Calculates the recency from the last occurrence of each key
    within a rolling window.

    This function, for every request, determines the number of steps elapsed
    since the same key was last seen within the lookback window. The result is
    normalized by the sequence length and reversed according to the concept
    of local recency.

    Args:
        requests (list[int]): A list of requested keys.
        seq_len (int): The maximum length of the rolling window to consider for
                       recency calculation.

    Returns:
        np.ndarray: An array of integers representing the normalized recency
                    value for the corresponding request key.

    Raises:
        RuntimeError: If local recency calculation fails:
            * Incorrect input types (TypeError).
            * List indexing fails (IndexError).
            * Data values are unsuitable (ValueError).
    """
    try:
        debug(
            "Local recency calculation started",
            extra={
                "seq_len": seq_len,
                "requests_num": len(requests),
                "context": "Local recency calculation",
            },
        )

        # For each key, calculate its local
        # recency according to the sequence length
        recencies = []
        for i in range(len(requests)):
            # Consider a window of requests
            # equals the sequence length
            window_start = max(0, i - seq_len + 1)
            window = requests[window_start:i]

            # Find all indices in window where current key appears
            occurrences = [j for j, k in enumerate(window) if k == requests[i]]

            if not occurrences:
                # Key not found in the window:
                # assign max recency
                recency = seq_len
            else:
                # Distance from last occurrence
                # as recency
                recency = len(window) - 1 - occurrences[LIST_LAST_IDX]

            # Reverse and normalize
            recencies.append(1.0 - (recency / seq_len))

        debug(
            "Local recency calculation completed",
            extra={
                "recencies_len": len(recencies),
                "context": "Local recency calculation",
            },
        )

        return np.array(recencies)
    except (TypeError, IndexError, ValueError) as e:
        msg = "Local recency calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "requests_len": len(requests)
                if requests is not None
                else None,
                "seq_len": seq_len,
                "context": "Local recency calculation",
            },
        )
        raise RuntimeError(msg) from e
