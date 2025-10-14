from typing import Dict, List

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info
from utils.math.percentage_calculator import calculate_percentage


def calculate_eviction_mistake_rate(
    evicted_items: Dict[int, List[float]],
    access_events_dict: Dict[int, List[float]],
    mistake_window: int
) -> float:
    """
    Calculate the eviction mistake rate.

    This function calculates the eviction mistake rate based on provided
    evicted keys and their access events. An eviction mistake occurs
    when a key that was evicted is accessed again within the specified
    temporal window.

    Args:
        evicted_items (Dict[int, List[float]]): Dictionary mapping keys
                                                to their eviction times.
        access_events_dict (Dict[int, List[float]]): Dictionary mapping keys to their
                                                     access timestamps.
        mistake_window (int): Temporal window to consider accesses as mistakes.

    Returns:
        float: Eviction mistake rate in percentage.
    """
    debug(f"Eviction mistake rate window: {mistake_window}")

    try:
        # Initialize counter
        tot_eviction_mistakes = 0
        tot_eviction_events = 0

        # For each eviction
        for key, eviction_times in evicted_items.items():
            # Get its access times
            access_times = access_events_dict.get(key, [])

            for eviction_time in eviction_times:
                # Increase the eviction events by one
                tot_eviction_events += 1

                # Check whether an access time lies
                # within the mistake window
                future_accesses = [
                    t for t in access_times
                    if eviction_time < t <= eviction_time + mistake_window
                ]

                # If there is at least one future
                # access lying within the mistake window
                if future_accesses:
                    # Increase the eviction mistakes by one
                    tot_eviction_mistakes += 1

                    debug(
                        f"Eviction mistake detected for key: {key}, "
                        f"eviction time: {eviction_time}, "
                        f"future access(es): {future_accesses}"
                    )

        # Calculate eviction mistake rate
        eviction_mistake_rate = calculate_percentage(tot_eviction_mistakes, tot_eviction_events)

        info(f"Eviction mistake rate calculated: {eviction_mistake_rate}")

        return eviction_mistake_rate
    except (AttributeError, TypeError, ValueError) as e:
        msg = "Failed to calculate eviction mistake rate"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e