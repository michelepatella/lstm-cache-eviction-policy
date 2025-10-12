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


def initialize_api() -> tuple[Module, torch.device, Box]:
    """
    Initialize the eviction policy API.

    This function performs all steps required to
    prepare the eviction policy API:
        - Loads API configuration from JSON file.
        - Selects the computation device.
        - Builds the LSTM model instance.
        - Moves the model to the selected device.
        - Loads pre-trained model weights.

    Returns:
        Tuple[Module, torch.device, Box]: Tuple containing the instantiated
                                          LSTM model, the device where the model
                                          resides, and the Box object containing
                                          the API configuration.
    """
    # Extract API configurations needed
    # for next steps
    device_type = api_config.hardware.device_type
    model_params = api_config.model.params
    model_path = api_config.model.path
    min_key = api_config.model.keys.min
    max_key = api_config.model.keys.max
    num_features = api_config.model.num_features
    embedding_dim = api_config.model.embedding_dim

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
