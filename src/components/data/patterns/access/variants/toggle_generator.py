from components.logs.levels.error_logger import error


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
    """Determine the next key to be accessed according to toggle pattern.

    This function determines which is the next key to be accessed according
    to toggle pattern. This pattern simulates two kinds of key access patterns,
    based on the toggle state. The result is an oscillating sequence of key
    requests.

    Args:
        toggle_interval (int): Number of requests between state switches.
        toggle_forward (int): Offset to apply in the forward toggle state.
        toggle_backward (int): Offset to apply in the backward toggle state.
        toggle_first_base_request (int): Steps back to pick the key for forward
                                         state.
        toggle_second_base_request (int): Steps back to pick the key for
                                          backward state.
        requests (list[int]): List of keys requested so far.
        requests_count (int): Number of requests generated so far.
        first_key (int): Minimum key in the range.
        keys_range_size (int): Total number of available keys.

    Returns:
        int: Index of the next accessed key.

    Raises:
        RuntimeError: If generating the toggle pattern fails:
            * Base request indices exceed the length of the requests history
              (ValueError, IndexError).
            * Using invalid arguments or data types (TypeError).
    """
    try:
        # Check whether first and second toggle
        # base requests are greater than the total
        # number of requests
        if (toggle_first_base_request > len(requests)) or (
            toggle_second_base_request > len(requests)
        ):
            msg = (
                f"Toggle first and second base request "
                f"({toggle_first_base_request}, {toggle_second_base_request}) "
                f"must be <= number of requests ({len(requests)})."
            )
            error(
                msg,
                extra={
                    "exception": msg,
                    "toggle_interval": toggle_interval,
                    "toggle_forward": toggle_forward,
                    "toggle_backward": toggle_backward,
                    "toggle_first_base_request": toggle_first_base_request,
                    "toggle_second_base_request": toggle_second_base_request,
                    "requests_count": requests_count,
                    "requests_num": len(requests),
                    "first_key": first_key,
                    "keys_range_len": keys_range_size,
                    "context": "Toggle pattern generation",
                },
            )
            error("%s", msg)
            raise ValueError(msg)

        # Get the toggle state (binary)
        toggle = (requests_count // toggle_interval) % 2

        # Determine the requested key
        # based on the toggle state
        if toggle == 0:
            # Requested key as the one requested
            # first base request steps before, adding
            # forward offset
            requested_key = (
                (
                    requests[-toggle_first_base_request]
                    - first_key
                    + toggle_forward
                )
                % keys_range_size
            ) + first_key
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

        return requested_key
    except (ValueError, IndexError, TypeError) as e:
        msg = "Toggle pattern generation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "toggle_interval": toggle_interval,
                "toggle_forward": toggle_forward,
                "toggle_backward": toggle_backward,
                "toggle_first_base_request": toggle_first_base_request,
                "toggle_second_base_request": toggle_second_base_request,
                "requests_count": requests_count,
                "requests_num": len(requests) if requests else 0,
                "first_key": first_key,
                "keys_range_len": keys_range_size,
                "context": "Toggle pattern generation",
            },
        )
        raise RuntimeError(msg) from e
