from typing import Tuple

import torch
from box import Box
from torch.nn import Module

from eviction_policy_api.services.autoregressive_rollout.initialization.utils.device_selector import (
    select_device,
)
from eviction_policy_api.services.autoregressive_rollout.initialization.utils.model_builder import (
    build_model,
)
from eviction_policy_api.services.autoregressive_rollout.initialization.utils.model_state_dict_loader import (
    load_model_state_dict,
)
from eviction_policy_api.services.autoregressive_rollout.initialization.utils.model_to_device_mover import (
    move_model_to_device,
)


def initialize_autoregressive_rollout(api_config: Box) -> tuple[Module, torch.device, Box]:
    """
    Initialize the autoregressive rollout service.

    This function performs all steps required to
    prepare the autoregressive rollout service:
        - Selects the computation device.
        - Builds the LSTM model instance.
        - Moves the model to the selected device.
        - Loads pre-trained model weights.

    Args:
        api_config (Box): API configuration object.

    Returns:
        Tuple[Module, torch.device, Box]: Tuple containing the instantiated
                                          LSTM model, the device where the model
                                          resides, and the Box object containing
                                          the API configuration.
    """
    # Select device according to device type
    device = select_device(device_type)

    # Build model to be used for inference
    model = build_model(
        model_params, min_key, max_key, embedding_dim, num_features
    )

    # Move model built to
    # selected device
    model = move_model_to_device(model, device)

    # Load state dictionary and
    # apply it to the model
    load_model_state_dict(model_path, model, device)

    return model, device, api_config
