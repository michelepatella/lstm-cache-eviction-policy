import numpy as np
from torch.nn.functional import softmax

from eviction_policy_api.key_score_calculator import (
    calculate_key_scores,
)
from pipeline.utils.logs.levels.info_logger import info


def find_key_candidates(all_outputs, upper_ci, lower_ci):
    # build two matrices
    # probability matrix: [t, k] -> probability of using k at time t
    # confidence matrix: [t, k] -> confidence of the prediction k at time t
    num_steps = len(all_outputs)
    num_keys = len(all_outputs[0])
    prob_matrix = np.zeros((num_steps, num_keys))
    conf_matrix = np.zeros((num_steps, num_keys))

    # fill the matrices
    for t in range(num_steps):
        # take the probabilities at time step t
        probs = softmax(all_outputs[t], dim=0).cpu().numpy()
        prob_matrix[t] = probs

        # calculate the confidence at time step t
        conf = (1 / (upper_ci[t] - lower_ci[t] + 1e-6)).cpu().numpy()
        conf_matrix[t] = conf

    # normalize confidence matrix to [0,1]
    min_conf = np.min(conf_matrix)
    max_conf = np.max(conf_matrix)
    conf_range = max_conf - min_conf if max_conf != min_conf else 1.0
    conf_matrix = (conf_matrix - min_conf) / conf_range

    # calculate scores for the keys
    scores = calculate_key_scores(
        num_keys,
        num_steps,
        prob_matrix,
        conf_matrix,
    )

    # select the keys
    keys = list(scores.keys())

    info("Key candidates found.")

    return keys, scores
