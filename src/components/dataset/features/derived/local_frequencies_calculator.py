"""local_frequencies_calculator.py

Module dedicated to calculating the local frequency of requested keys
within a rolling time window.

The module processes a sequential list of requests and computes, for
each request, its normalized frequency relative to the preceding sequence
of a specified length. This calculation provides a measure of
recency and popularity of a key at a specific point in time within the
request stream.

Functions:
    calculate_local_frequency(
        requests: list[int],
        seq_len: int
    ) -> np.ndarray
        Calculates the normalized local frequency for each request.
"""

import numpy as np

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def calculate_local_frequencies(
    requests: list[int], seq_len: int
) -> np.ndarray:
    """Calculates the normalized local frequency for each
    request key.

    For every request in the list, this function determines how many
    times that specific key appears in the rolling window immediately
    preceding and including the current request. The result
    is normalized by the sequence length.

    Args:
        requests (list[int]): A list of requested keys.
        seq_len (int): The length of the rolling window to consider for local
                       frequency calculation.

    Returns:
        np.ndarray: An array of floats where each element represents the
                    normalized local frequency of the corresponding request key.

    Raises:
        RuntimeError: If local frequency calculation fails:
            * Incorrect input types (TypeError).
            * List indexing fails (IndexError).
            * Data values are unsuitable (ValueError).
    """
    try:
        debug(
            "Local frequency calculation started",
            extra={
                "seq_len": seq_len,
                "requests_num": len(requests),
                "context": "Local frequency calculation",
            },
        )

        # For each key, calculate its
        # local frequency according to
        # the sequence length
        frequencies = []
        for i in range(len(requests)):
            # Consider a window of requests
            # equals the sequence length
            window_start = max(0, i - seq_len + 1)
            window = requests[window_start : i + 1]

            # Count how many times the current
            # key appears in the last sequence
            # length requests and normalize this
            # frequency w.r.t. the sequence length
            count = sum(k == requests[i] for k in window)
            frequencies.append(count / seq_len)

        debug(
            "Local frequency calculation completed",
            extra={
                "frequencies_len": len(frequencies),
                "context": "Local frequency calculation",
            },
        )

        return np.array(frequencies)
    except (TypeError, IndexError, ValueError) as e:
        msg = "Local frequency calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "requests_len": len(requests)
                if requests is not None
                else None,
                "seq_len": seq_len,
                "context": "Local frequency calculation",
            },
        )
        raise RuntimeError(msg) from e
