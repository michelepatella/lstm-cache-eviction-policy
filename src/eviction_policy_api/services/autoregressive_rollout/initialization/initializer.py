from typing import Tuple, Dict

import torch

from utils.model.initialization.components.device_selector import select_device
from utils.model.initialization.components.model_builder import build_model
from utils.model.initialization.components.model_state_dict_loader import load_model_state_dict
from utils.model.initialization.components.device_mover import move_to_device


def initialize_autoregressive_rollout(
    model_path: str,
    model_params: Dict[str, int | float | bool],
    device_type: str,
    min_key: int,
    max_key: int,
    num_features: int,
    embedding_dim: int,
) -> Tuple[torch.nn.Module, torch.device]:
    """
    Initialize the autoregressive rollout service.

    This function performs all steps required to prepare
    for autoregressive rollout:
        1. Selects the computation device.
        2. Builds the model instance.
        3. Moves the model to the selected device.
        4. Loads pre-trained model weights.

    Args:
        model_path (str): Path to the saved model weights.
        model_params (Dict[str, int | float | bool]): Model hyperparameters.
        device_type (str): Device type to use.
        min_key (int): Minimum key.
        max_key (int): Maximum key.
        num_features (int): Number of input features for the model.
        embedding_dim (int): Dimension of embedding layer.

    Returns:
        Tuple[torch.nn.Module, torch.device]: The initialized model and
                                              the device it resides on.
    """
    # Select computation device
    device = select_device(device_type)

    # Build the model
    model = build_model(
        model_params,
        min_key,
        max_key,
        embedding_dim,
        num_features,
    )

    # Move model to device
    model = move_to_device(model, device)

    # Load pre-trained weights
    load_model_state_dict(model_path, model, device)

    return model, device