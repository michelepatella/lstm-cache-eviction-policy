from typing import Any, Tuple, List

from torch import Tensor

from config.classes.Config import Config
from lstm_eviction_policy.autoregression import autoregressive_rollout
from lstm_eviction_policy.confidence_interval_calculator import (
    calculate_confidence_interval,
)
from lstm_eviction_policy.key_candidates_finder import (
    find_key_candidates,
)
from utils.logs.levels.info_logger import info
from utils.model.initialization.trained_model_initializer import (
    initialize_trained_model,
)


def manage_lstm_eviction_policy(
    keys_in_cache: List[Any],
    last_accesses: Tuple[Tensor, Tensor, Tensor],
    prediction_window: int,
    num_mc_samples: int,
    confidence_level: float,
):
    # compute rollout
    all_outputs, all_vars = autoregressive_rollout(
        model,
        last_accesses,
        device,
        config,
    )

    # calculate CIs related to the predictions
    lower_ci, upper_ci = calculate_confidence_interval(
        all_outputs,
        all_vars,
        config,
    )

    # identify keys and scores thereof
    keys, scores = find_key_candidates(all_outputs, upper_ci, lower_ci)

    info("LSTM-based cache policy management completed.")
