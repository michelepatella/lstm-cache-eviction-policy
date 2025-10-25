from typing import Dict, List

from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.math.percentage_calculator import calculate_percentage


def calculate_eviction_mistake_rate(
    evicted_items: Dict[int, List[float]],
    access_events_dict: Dict[int, List[float]],
    mistake_window: int,
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

    Raises:
        RuntimeError: If eviction mistake rate calculation fails:
            * Evicted items or access events are not dictionaries (TypeError).
            * Eviction times or access times contain invalid values
              (TypeError, ValueError).
            * Attributes of inputs are missing (AttributeError).
    """
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
                    t
                    for t in access_times
                    if eviction_time < t <= eviction_time + mistake_window
                ]

                # If there is at least one future
                # access lying within the mistake window
                if future_accesses:
                    # Increase the eviction mistakes by one
                    tot_eviction_mistakes += 1

        # Calculate eviction mistake rate
        eviction_mistake_rate = calculate_percentage(
            tot_eviction_mistakes, tot_eviction_events
        )

        info(
            "Eviction mistake rate calculated",
            extra={
                "eviction_mistake_rate": eviction_mistake_rate,
                "total_eviction_events": tot_eviction_events,
                "total_eviction_mistakes": tot_eviction_mistakes,
                "num_evicted_keys": len(evicted_items),
                "num_keys_with_accesses": sum(
                    1 for key in evicted_items if access_events_dict.get(key)
                ),
                "mistake_window": mistake_window,
                "context": "Eviction mistake rate calculation",
            },
        )

        return eviction_mistake_rate
    except (AttributeError, TypeError, ValueError) as e:
        msg = "Eviction mistake rate calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "num_evicted_keys": (
                    len(evicted_items)
                    if isinstance(evicted_items, dict)
                    else None
                ),
                "num_access_keys": (
                    len(access_events_dict)
                    if isinstance(access_events_dict, dict)
                    else None
                ),
                "mistake_window": mistake_window,
                "context": "Eviction mistake rate calculation",
            },
        )
        raise RuntimeError(msg) from e
