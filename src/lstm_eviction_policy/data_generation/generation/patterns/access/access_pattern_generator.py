import numpy as np

from lstm_eviction_policy.config.classes.Config import Config


def generate_access_pattern(
    zipf_probs: np.ndarray,
    keys_range: np.ndarray,
    current_abs_seconds: float,
    requests: list[int],
    config: Config,
):
    # set data
    hour = (current_abs_seconds / 3600.0) % 24
    base = config.first_key
    keys = list(range(base, config.last_key))
    n_keys = len(keys)
    range_size = config.last_key - config.first_key
    noise_low, noise_high = config.noise_range

    if len(requests) < 5:
        return np.random.choice(keys)

    idx = len(requests)

    # 05:00 - 09:00 repetition pattern
    if 5 <= hour < 9:
        if idx % config.repetition_interval == 0:
            new_key = requests[-config.repetition_offset]
        else:
            new_key = np.random.choice(keys[: n_keys // 3])

    # 09:00 - 12:00 toggle pattern
    elif 9 <= hour < 12:
        toggle = (idx // config.toggle_interval) % 2
        if toggle == 0:
            new_key = ((requests[-1] - base + 1) % range_size) + base
        else:
            new_key = ((requests[-2] - base - 1) % range_size) + base

    # 12:00 - 18:00 cyclic scanning
    elif 12 <= hour < 18:
        cycle_length = (
            config.cycle_base + (idx // config.cycle_divisor) % config.cycle_mod
        )
        cycle = keys[:cycle_length]
        new_key = cycle[idx % cycle_length]

    # 18:00 - 23:00 distorted history
    elif 18 <= hour < 23:
        if idx % config.distortion_interval == 0:
            new_key = ((requests[-4] - base + 2) % range_size) + base
        else:
            noise = np.random.randint(noise_low, noise_high + 1)
            new_key = ((requests[-1] - base + noise) % range_size) + base

    # 23:00 - 05:00 pattern
    else:
        if idx % config.memory_interval == 0:
            new_key = requests[-config.memory_offset]
        else:
            new_key = np.random.choice(keys_range, p=zipf_probs)

    # show a successful message
    info(f"🟢 Requests access pattern generated.")

    return new_key
