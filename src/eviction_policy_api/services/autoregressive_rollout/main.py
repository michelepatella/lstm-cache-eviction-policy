import math
from typing import List, Tuple, Dict

import torch
from fastapi import FastAPI

from eviction_policy_api.const import AUTOREGRESSIVE_ROLLOUT_SERVICE_ENDPOINT
from eviction_policy_api.services.autoregressive_rollout.initialization.initializer import (
    initialize_autoregressive_rollout,
)
from utils.logs.levels.info_logger import info
from utils.mc.forward_runner import mc_forward_passes
from utils.model.initialization.components.device_mover import move_to_device

app = FastAPI()


@app.post(AUTOREGRESSIVE_ROLLOUT_SERVICE_ENDPOINT)
def autoregressive_rollout_service(
    last_accesses: List[Tuple[float, int]],
    model_path: str,
    model_params: Dict[str, int | float | bool],
    device_type: str,
    min_key: int,
    max_key: int,
    num_features: int,
    embedding_dim: int,
    rollout_horizon: int,
    mc_dropout_samples: int,
    time_step_increment: float,
):
    try:
        # Setup for autoregressive rollout service:
        # Prepare model and device for inference
        model, device = initialize_autoregressive_rollout(
            model_path,
            model_params,
            device_type,
            min_key,
            max_key,
            num_features,
            embedding_dim,
        )

        # Prepare initial sequence from last accesses
        x_features_seq = []
        x_keys_seq = []

        for hour, key in last_accesses:
            angle = (hour / 24) * 2 * math.pi
            x_features_seq.append([math.sin(angle), math.cos(angle)])
            x_keys_seq.append(key)

        x_features_seq = torch.tensor(
            x_features_seq, dtype=torch.float
        ).unsqueeze(0)
        move_to_device(x_features_seq, device)

        x_keys_seq = torch.tensor(x_keys_seq, dtype=torch.long).unsqueeze(0)
        move_to_device(x_keys_seq, device)

        all_outputs = []
        all_vars = []

        last_sin = x_features_seq[0, -1, 0].item()
        last_cos = x_features_seq[0, -1, 1].item()
        last_time = math.atan2(last_sin, last_cos) % (2 * math.pi)
        delta_t = (time_step_increment / 24) * (2 * math.pi)

        for _ in range(rollout_horizon):
            outputs_mean, outputs_var, _ = mc_forward_passes(
                model,
                (x_features_seq, x_keys_seq),
                device,
                x_features_seq.shape[-1],  # number of features
                mc_dropout_samples,
            )

            all_outputs.append(outputs_mean.squeeze(0))
            if outputs_var is not None:
                all_vars.append(outputs_var.squeeze(0))
            else:
                all_vars.append(torch.zeros_like(outputs_mean.squeeze(0)))

            pred_key = outputs_mean.argmax(dim=-1).unsqueeze(1)
            x_keys_seq = torch.cat([x_keys_seq[:, 1:], pred_key], dim=1)

            last_time = (last_time + delta_t) % (2 * math.pi)
            new_feature = torch.tensor(
                [[math.sin(last_time), math.cos(last_time)]], device=device
            ).unsqueeze(0)
            x_features_seq = torch.cat(
                [x_features_seq[:, 1:, :], new_feature], dim=1
            )

        info("Autoregressive rollout completed.")

        return {
            "outputs": [o.tolist() for o in all_outputs],
            "variances": [v.tolist() for v in all_vars],
        }

    except Exception as e:
        return {"error": str(e)}
