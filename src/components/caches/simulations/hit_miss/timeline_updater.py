from typing import Dict, List

from components.const import (
    SIMULATIONS_METRICS_TIMELINE_INDEX_NAME,
    SIMULATIONS_METRICS_TIMELINE_INSTANT_HIT_RATE_NAME,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.math.percentage_calculator import calculate_percentage
from const import (
    SIMULATIONS_METRICS_HIT_COUNTER_NAME,
    SIMULATIONS_METRICS_MISS_COUNTER_NAME,
)


def update_hit_miss_timeline(
    idx: int,
    counters: Dict[str, int],
    timeline: List[Dict[str, int]],
    timeline_index_name: str = SIMULATIONS_METRICS_TIMELINE_INDEX_NAME,
    timeline_instant_hit_rate_name: str = SIMULATIONS_METRICS_TIMELINE_INSTANT_HIT_RATE_NAME,
    hit_counter_name: str = SIMULATIONS_METRICS_HIT_COUNTER_NAME,
    miss_counter_name: str = SIMULATIONS_METRICS_MISS_COUNTER_NAME,
) -> List[Dict[str, int]]:
    """
    Update the timeline with the current hit and miss counts.

    This function calculates the instant hit rate based on the
    cumulative number of hits and the number of requests processed
    so far, then appends a new entry to the timeline with
    the current statistics.

    Args:
        counters (Dict[str, int]): Dictionary containing current hit and miss
                                   counters.
        idx (int): Current request index.
        timeline (List[Dict[str, int]]): List storing the timeline of hit
                                         and miss statistics.
        timeline_index_name (str): Key name for storing the request index in the
                                   timeline entry.
        timeline_instant_hit_rate_name (str): Key name for storing the instant hit rate.
        hit_counter_name (str): Key name of the hit counter in the counters'
                                dictionary.
        miss_counter_name (str): Key name of the miss counter in the counters'
                                 dictionary.

    Returns:
        List[Dict[str, int]]: Updated timeline including the latest hit and miss
                              statistics.

    Raises:
        RuntimeError: If updating the hit and miss timeline fails:
            * Hit and miss counters missing in the dictionary (KeyError).
            * Timeline data structure is invalid (TypeError, AttributeError).
    """
    try:
        # Calculate instant hit rate
        instant_hit_rate = calculate_percentage(
            counters[hit_counter_name], idx + 1
        )

        # Append current metrics to the timeline
        timeline.append(
            {
                timeline_index_name: idx,
                timeline_instant_hit_rate_name: instant_hit_rate,
                hit_counter_name: counters[hit_counter_name],
                miss_counter_name: counters[miss_counter_name],
            }
        )

        debug(
            f"Timeline updated for request index {idx}:\n"
            f"hits: {counters[hit_counter_name]}\n"
            f"misses: {counters[miss_counter_name]}\n"
            f"instant hit rate: {instant_hit_rate}"
        )

        return timeline
    except (KeyError, TypeError, AttributeError) as e:
        msg = "Failed to update hit and miss timeline"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
