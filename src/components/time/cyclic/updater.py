from typing import Tuple

from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info


def update_cyclic_time(
    current_time_in_cycle: float,
    cycle_count: int,
    period: float,
    delta_t: float,
) -> Tuple[float, int]:
    """
    Update time in a cyclic period according to a delta.

    This function updates the current time within a cycle and
    increments the cycle count if the updated time exceeds the period.

    Args:
        current_time_in_cycle (float): Current time within the cycle.
        cycle_count (int): Current cycle count.
        period (float): Length of the cycle.
        delta_t (float): Time increment to apply.

    Returns:
        Tuple[float, int]:
            - current_time_in_cycle: Updated time within the current cycle after applying delta_t.
            - cycle_count: Updated cycle count after overflow beyond the period.
    """
    debug(
        f"Current time in cycle: {current_time_in_cycle}, "
        f"cycle count: {cycle_count} to be updated"
    )
    debug(f"Cycle period: {period}")
    debug(f"Delta time: {delta_t}")

    # Update time and cycle count if necessary
    if current_time_in_cycle + delta_t > period:
        cycle_count += 1
        current_time_in_cycle = (current_time_in_cycle + delta_t) - period
    else:
        current_time_in_cycle += delta_t

    info(
        f"Time updated (current time in cycle: {current_time_in_cycle}, "
        f"cycle count: {cycle_count})"
    )

    return current_time_in_cycle, cycle_count
