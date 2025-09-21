from datetime import timedelta

import numpy as np

from lstm_eviction_policy.config.classes.Config import Config


def generate_pattern_requests(
    keys_range: np.ndarray,
    zipf_probs: np.ndarray,
    config: Config,
) -> tuple[list[int], list[float]]:
    # Initialize data
    timestamps_seconds = [0.0]
    requests = []
    day = 0
    time_in_day = 0.0

    # Define day as period in seconds
    # (24 hours, 60 minutes, 60 seconds)
    period = timedelta(days=1).total_seconds()

    try:
        # to make the process deterministic
        np.random.seed(config.seed)

        # for each request
        for i in range(num_requests):
            # generate the delta time
            delta_t = generate_temporal_pattern(
                [timestamps_seconds[-1] % period], period, config
            )

            if time_in_day + delta_t > period:
                day += 1
                time_in_day = (time_in_day + delta_t) - period
            else:
                time_in_day += delta_t
            total_time = day * period + time_in_day

            # generate request
            request = generate_access_pattern(
                zipf_probs,
                keys_range,
                total_time,
                requests,
                config,
            )

            # store data
            requests.append(request)
            timestamps_seconds.append(total_time)

            # debugging
            debug(f"⚙️ Number of request generated: {i+1}.")
            debug(f"⚙️ Request generated: {request}.")
            debug(f"⚙️ Timestamps generated: {timestamps_seconds}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except IndexError as e:
        raise IndexError(f"IndexError: {e}.")
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"ZeroDivisionError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except MemoryError as e:
        raise MemoryError(f"MemoryError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info(f"🟢 Pattern requests generated.")

    return requests, timestamps_seconds
