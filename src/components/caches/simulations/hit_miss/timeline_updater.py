from typing import Dict, List

from components.const import (
    SIMULATIONS_METRICS_TIMELINE_INDEX_NAME,
    SIMULATIONS_METRICS_TIMELINE_INSTANT_HIT_RATE_NAME,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.math.percentage_calculator import calculate_percentage
from src.const import (
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

        return timeline
    except (KeyError, TypeError, AttributeError) as e:
        msg = "Tit/miss timeline updating failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "request_idx": idx,
                "timeline_type": (
                    type(timeline).__name__ if timeline is not None else None
                ),
                "timeline_len": (
                    len(timeline)
                    if hasattr(timeline, "__len__") and timeline
                    else 0
                ),
                "counters_type": (
                    type(counters).__name__ if counters is not None else None
                ),
                "counters_keys": list(counters.keys()) if counters else None,
                "context": "Hit/miss timeline updating",
            },
        )
        raise RuntimeError(msg) from e
