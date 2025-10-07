from typing import Any

from lstm_cache_eviction_policy.autoregression import autoregressive_rollout
from lstm_cache_eviction_policy.confidence_interval_calculator import (
    calculate_confidence_interval,
)
from lstm_cache_eviction_policy.key_candidates_finder import (
    find_key_candidates,
)
from utils.logs.levels.info_logger import info


def manage_lstm_cache(
    cache: Any,
    seed_seq,
):
    # compute rollout
    (all_outputs, all_vars) = autoregressive_rollout(
        model,
        seed_seq,
        device,
        config_settings,
    )

    # calculate CIs related to the predictions
    (lower_ci, upper_ci) = calculate_confidence_interval(
        all_outputs,
        all_vars,
        config_settings,
    )

    # identify keys and scores thereof
    (keys, scores) = find_key_candidates(all_outputs, upper_ci, lower_ci)

    info("LSTM-based cache policy management completed.")
