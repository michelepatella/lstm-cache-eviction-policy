from simulation.caches.utils.classes.CacheMetricsLogger import (
    CacheMetricsLogger,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def calculate_eviction_mistake_rate(
    metrics_logger: CacheMetricsLogger, mistake_window: int
) -> float:
    """
    Calculate the eviction mistake rate within a
    given temporal window.

    This function calculates the eviction mistake rate,
    within a provided temporal window. An eviction mistake
    occurs when a key that was evicted from the cache
    is accessed again within the specified temporal window.

    Args:
        metrics_logger (CacheMetricsLogger): Object tracking cache events
                                             (evictions and accesses).
        mistake_window (int): Temporal window to consider future
                              accesses as mistakes.

    Returns:
        float: Eviction mistake rate.
    """
    debug(f"Eviction mistake rate with window: {mistake_window}")

    try:
        # Initialize counters
        tot_eviction_mistakes = 0
        tot_eviction_events = 0

        # Iterate over all evicted keys and their eviction times
        for key, eviction_times in metrics_logger.evicted_keys.items():
            for eviction_time in eviction_times:
                # Increase eviction events by one
                tot_eviction_events += 1

                # Find future accesses within
                # the mistake window
                future_accesses = [
                    t
                    for t in metrics_logger.access_events.get(key, [])
                    if eviction_time < t <= eviction_time + mistake_window
                ]

                # If there is at least one
                # future access within the
                # mistake window
                if future_accesses:
                    # Increase eviction mistake by one
                    tot_eviction_mistakes += 1

                    debug(
                        f"Eviction mistake detected for key: {key}, "
                        f"eviction time: {eviction_time}, "
                        f"future access(es): {future_accesses}"
                    )

        # Compute eviction mistake rate
        eviction_mistake_rate = (
            tot_eviction_mistakes / tot_eviction_events
            if tot_eviction_events > 0
            else 0.0
        )

        info(f"Eviction mistake rate calculated: {eviction_mistake_rate}")

        return eviction_mistake_rate
    except (AttributeError, TypeError, ValueError) as e:
        msg = "Failed to calculate eviction mistake rate"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
