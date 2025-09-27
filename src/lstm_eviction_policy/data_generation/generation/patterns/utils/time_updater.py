from lstm_eviction_policy.utils.logs.levels.debug_logger import debug
from lstm_eviction_policy.utils.logs.levels.info_logger import info


def update_time(
    current_seconds_in_day: float,
    current_day: int,
    period: float,
    delta_t: float,
) -> tuple[float, int]:
    """
    Update time (current seconds in day and
    current day) according to delta time.

    This function updates requests time (current
    seconds in day and current day), based on delta time
    generated, which represents the gap between the current
    request and the next one.

    Parameters:
        current_seconds_in_day (float): Current seconds in day to be updated.
        current_day (int): Current day to be updated (if needed).
        period (float): Time period used for data generation.
        delta_t (float): Gap between the current request and the next one.

    Returns:
        tuple[float, int]: Tuple of current seconds in day and
                           current day updated.
    """
    debug(
        f"Current seconds in day: {current_seconds_in_day}, and current day: {current_day} to be updated"
    )
    debug(f"Period used to update time: {period}")
    debug(f"Delta time used to update time: {delta_t}")

    # Update current day (if needed) and
    # current seconds in day, according to
    # delta time generated
    if current_seconds_in_day + delta_t > period:
        # Update current day and current seconds
        # in day, knowing that a new day is just started
        # as the current seconds in day, added to delta time,
        # exceeds the period
        current_day += 1
        current_seconds_in_day = (current_seconds_in_day + delta_t) - period
    else:
        # Update only current seconds in
        # day as a new day is not started yet
        current_seconds_in_day += delta_t

    info(
        f"Time updated (current seconds in day: {current_seconds_in_day}, current day: {current_day})"
    )

    return current_seconds_in_day, current_day
