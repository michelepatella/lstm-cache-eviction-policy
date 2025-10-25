from typing import Tuple

from components.logs.levels.error_logger import error


def update_cyclic_time(
    current_time_in_cycle: float,
    cycle_count: int,
    period: float,
    delta_t: float,
) -> Tuple[float, int]:
    """
    Update time in a cyclic period according to a delta.

    This function updates the current time within a cycle and increments
    the cycle count if the updated time exceeds the period.

    Args:
        current_time_in_cycle (float): Current time within the cycle.
        cycle_count (int): Current cycle count.
        period (float): Length of the cycle.
        delta_t (float): Time increment to apply.

    Returns:
        Tuple[float, int]:
            - current_time_in_cycle: Updated time within the current cycle after
                                     applying delta time.
            - cycle_count: Updated cycle count after overflow beyond the period.

    Raises:
        RuntimeError: If updating the cyclic time fails:
            * Input values are not numeric and addition/comparison
              operations fail (TypeError).
    """
    try:
        # Update time and cycle count if the new time
        # (given by adding delta to the current one) exceeds
        # the provided period
        if current_time_in_cycle + delta_t > period:
            # A new cycle is just started
            cycle_count += 1
            current_time_in_cycle = (current_time_in_cycle + delta_t) - period
        else:
            # Still in the current cycle
            current_time_in_cycle += delta_t

        return current_time_in_cycle, cycle_count
    except TypeError as e:
        msg = "Cyclic time updating failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "current_time_in_cycle": current_time_in_cycle,
                "cycle_count": cycle_count,
                "period": period,
                "delta_t": delta_t,
                "context": "Cyclic time updating",
            },
        )
        raise RuntimeError(msg) from e
