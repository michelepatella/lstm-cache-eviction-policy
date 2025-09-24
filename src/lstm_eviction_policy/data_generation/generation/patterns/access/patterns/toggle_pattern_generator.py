from lstm_eviction_policy.utils.logs.log_utils import (
    debug,
    info,
)


def generate_toggle_pattern(
    toggle_interval: int,
    toggle_forward: int,
    toggle_backward: int,
    toggle_first_base_request: int,
    toggle_second_base_request: int,
    requests: list[int],
    requests_count: int,
    first_key: int,
    keys_range_size: int,
) -> int:
    """
    Determine the next key to be accessed
    according to toggle pattern.

    This function determines which is the next
    key to be accessed according to toggle
    pattern. This pattern simulates two kinds of
    key access patterns, based on the toggle state.
    The result is an oscillating sequence of key
    requests.

    Parameters:
        toggle_interval (int): Number of requests between state switches.
        toggle_forward (int): Offset to apply in the forward toggle state.
        toggle_backward (int): Offset to apply in the backward toggle state.
        toggle_first_base_request (int): Steps back to pick the key
                                         for forward state.
        toggle_second_base_request (int): Steps back to pick the key
                                          for backward state.
        requests (list[int]): List of keys requested so far.
        requests_count (int): Number of requests generated so far.
        first_key (int): Minimum key in the range.
        keys_range_size (int): Total number of available keys.

    Returns:
        int: Index of the next accessed key.
    """
    debug(f"Toggle interval to determine the toggle state: {toggle_interval}")

    # Get the toggle state (binary)
    toggle = (requests_count // toggle_interval) % 2

    debug(f"Toggle state: {toggle}")
    debug(
        f"Toggle forward offset: {toggle_forward}, backward offset: {toggle_backward}"
    )
    debug(
        f"Toggle base requests: {toggle_first_base_request} (forward), {toggle_second_base_request} (backward)"
    )

    # Determine the requested key
    # based on the toggle state
    if toggle == 0:
        # Requested key as the one requested
        # first base request steps before, adding
        # forward offset
        requested_key = (
            (requests[-toggle_first_base_request] - first_key + toggle_forward)
            % keys_range_size
        ) + first_key

        debug("Forward toggle state access")
    else:
        # Requested key as the one requested
        # second base request steps before, adding
        # backward offset
        requested_key = (
            (
                requests[-toggle_second_base_request]
                - first_key
                + toggle_backward
            )
            % keys_range_size
        ) + first_key

        debug("Backward toggle state access")

    info(f"(Toggle pattern) Key requested: {requested_key}")

    return requested_key
