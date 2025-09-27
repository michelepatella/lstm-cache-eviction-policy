from lstm_eviction_policy.utils.logs.levels.debug_logger import debug
from lstm_eviction_policy.utils.logs.levels.info_logger import info


def set_bursty_scale(
    burst_high: float,
    burst_low: float,
    burst_start_hour: int,
    burst_end_hour: int,
    current_hour_in_day: float,
) -> float:
    """
    Set bursty scale for the current hour in day.

    This function sets the bursty scale — for
    the current hour in day — equal to
    burst high/low, depending on the current hour in day
    with respect to the burst band of the day.

    Parameters:
        burst_high (float): Burst high value.
        burst_low (float): Burst low value.
        burst_start_hour (int): Hour in day when the burst starts.
        burst_end_hour (int): Hour in day when the burst ends.
        current_hour_in_day (float): Current hour in day for which
                                     bursty scale is to be calculated.

    Returns:
        float: Bursty scale for current hour in day.
    """
    debug(f"Burst band of day: {burst_start_hour} - {burst_end_hour}")
    debug(f"Burst high/low: {burst_high}, {burst_low}")

    # Set bursty scale to burst high or low,
    # depending on the current hour in day
    # (if it is within the bursty band of the day)
    if burst_start_hour <= current_hour_in_day <= burst_end_hour:
        bursty_scale = burst_high
    else:
        bursty_scale = burst_low

    info(
        f"Bursty scale: {bursty_scale}, set for current hour in day: {current_hour_in_day}"
    )

    return bursty_scale
