from typing import Dict, List

from pipeline.const import (
    HIT_COUNTER_NAME,
    MISS_COUNTER_NAME,
    TIMELINE_INDEX_NAME,
    TIMELINE_INSTANT_HIT_RATE_NAME,
)
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info
from utils.math.percentage_calculator import calculate_percentage


def update_hit_miss_timeline(
    idx: int,
    counters: Dict[str, int],
    timeline: List[Dict[str, int]],
) -> List[Dict[str, int]]:
    """
    Update the timeline with the current hit
    and miss counts.

    This function calculates the instant hit rate based on the
    cumulative number of hits and the number of requests processed
    so far, then appends a new entry to the timeline with
    the current statistics.

    Args:
        counters (Dict[str, int]): Dictionary containing current
                                   hit and miss counters.
        idx (int): Current request index.
        timeline (List[Dict[str, int]]): List storing the timeline of
                                         hit/miss statistics.

    Returns:
        List[Dict[str, int]]: Updated timeline including the latest
                              hit/miss statistics.
    """
    try:
        # Calculate instant hit rate
        instant_hit_rate = calculate_percentage(counters[HIT_COUNTER_NAME], idx + 1)

        # Append current metrics to the timeline
        timeline.append(
            {
                TIMELINE_INDEX_NAME: idx,
                TIMELINE_INSTANT_HIT_RATE_NAME: instant_hit_rate,
                HIT_COUNTER_NAME: counters[HIT_COUNTER_NAME],
                MISS_COUNTER_NAME: counters[MISS_COUNTER_NAME],
            }
        )

        info(
            f"Timeline updated for request index {idx}:\n"
            f"hits: {counters[HIT_COUNTER_NAME]}\n"
            f"misses: {counters[MISS_COUNTER_NAME]}\n"
            f"instant hit rate: {instant_hit_rate}"
        )
    except KeyError as e:
        msg = "Failed to update timeline"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    return timeline
