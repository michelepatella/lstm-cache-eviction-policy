import math

import torch

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info
from utils.mc.forward_runner import (
    mc_forward_passes,
)


def autoregressive_rollout(model, seed_sequence, device, config_settings):
    # prepare data
    x_features_seq, x_keys_seq, _ = seed_sequence
    x_features_seq = x_features_seq.unsqueeze(0).to(device)
    x_keys_seq = x_keys_seq.unsqueeze(0).to(device)
    all_outputs = []
    all_vars = []

    # initialize last time to current time
    last_sin = x_features_seq[0, -1, 0].item()
    last_cos = x_features_seq[0, -1, 1].item()
    last_time = math.atan2(last_sin, last_cos) % (2 * math.pi)

    # increase the temporal features (by 2 minutes)
    delta_t = (2 / 1440) * (2 * math.pi)

    # for each future sequence
    for i in range(config_settings.simulation.lstm.prediction.interval):
        # compute MC forward pass
        (outputs_mean, outputs_var, _) = mc_forward_passes(
            model,
            (x_features_seq, x_keys_seq),
            device,
            config_settings,
            config_settings.inference.mc_dropout.samples.count,
        )

        # store outputs and variances
        all_outputs.append(outputs_mean.squeeze(0))
        if outputs_var is not None:
            all_vars.append(outputs_var.squeeze(0))
        else:
            all_vars.append(torch.zeros_like(outputs_mean.squeeze(0)))

        # get the predicted key as the most probable one
        pred_key = outputs_mean.argmax(dim=-1).unsqueeze(1)

        # add a new step using the predicted key
        x_keys_seq = torch.cat(
            [x_keys_seq[:, 1:], pred_key],
            dim=1,
        )

        # update features
        last_time = (last_time + delta_t) % (2 * math.pi)
        new_cos = math.cos(last_time)
        new_sin = math.sin(last_time)
        new_feature = torch.tensor(
            [[new_sin, new_cos]],
            device=device,
        ).unsqueeze(0)
        x_features_seq = torch.cat(
            [
                x_features_seq[:, 1:, :],
                new_feature,
            ],
            dim=1,
        )

    info("Autoregressive rollout completed.")

    return all_outputs, all_vars
