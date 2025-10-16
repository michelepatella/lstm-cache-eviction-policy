from typing import Any, Dict, List

from pipeline.const import (
    AVG_CACHE_LATENCY_NAME,
    EVICTION_MISTAKE_RATE_NAME,
    HIT_RATE_NAME,
    MISS_RATE_NAME,
    POLICY_NAME,
)
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def generate_simulation_report(
    results: List[Dict[str, Any]],
) -> None:
    """
    Generate a report for cache simulation evaluation results.

    This function reports the evaluation results for each simulated
    cache eviction policy, including hit/miss rates, average latency,
    and eviction mistake rate, using informational logs.

    Args:
        results (List[Dict[str, Any]]): List of dictionaries containing
                                        cache policy metrics.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while generating simulation
                      report e.g.:
                        * If metric results are missed.
                        * If results are of wrong type.
    """
    info("=== Simulation Report ===")

    try:
        # Iterate through each cache policy result
        for res in results:
            # Get the eviction policy name
            policy_name = res.get(POLICY_NAME)

            # Retrieve all cache eviction policy
            # metrics collected during simulation
            hit_rate = res.get(HIT_RATE_NAME)
            miss_rate = res.get(MISS_RATE_NAME)
            avg_latency = res.get(AVG_CACHE_LATENCY_NAME)
            eviction_mistake_rate = res.get(EVICTION_MISTAKE_RATE_NAME)

            # Show cache eviction policy metrics
            info(
                f"{policy_name}: Hit Rate: {hit_rate}%, "
                f"Miss Rate: {miss_rate}%, "
                f"Average Cache Latency: {avg_latency}µs, "
                f"Eviction Mistake Rate: {eviction_mistake_rate}%"
            )
    except (TypeError, ValueError) as e:
        msg = "Failed to generate simulation report"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("=== End of Simulation Report ===")
