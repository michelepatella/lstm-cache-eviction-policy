import torch
from scipy.stats import norm

from components.logs.levels.debug_logger import debug


def calculate_confidence_interval(all_outputs, all_variances, config_settings):
    # calculate the z-score
    z_score = norm.ppf(
        1 - (1 - config_settings.inference.confidence_intervals.level) / 2,
    )

    # calculate the standard deviation
    outputs_tensor = torch.stack(all_outputs)
    outputs_std = torch.sqrt(torch.stack(all_variances))

    # calculate lower and upper CIs boundaries
    lower_ci = outputs_tensor - z_score * outputs_std
    upper_ci = outputs_tensor + z_score * outputs_std

    debug("Confidence intervals calculated.")

    return lower_ci, upper_ci
