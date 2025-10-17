from typing import Tuple

import torch

from components.device.mover import (
    move_to_device,
)
from components.device.selector import (
    select_device,
)
from components.logs.levels.info_logger import info
from components.loss.builder import build_loss
from components.model.builder import (
    build_model,
)
from pipeline.config.pydantic.config import Config
from pipeline.config.pydantic.sections.model_config import ModelParamsConfig


def initialize_model_environment(
    model_params: ModelParamsConfig,
    config: Config,
    targets: torch.Tensor = None,
) -> Tuple[torch.device, torch.nn.Module, torch.nn.Module]:
    """
    Set up the model environment.

    This function performs all steps required to prepare
    model environment:
        - Selects device
        - Defines the criterion
        - Instantiates the PyTorch model and moves it to the device

    Args:
        model_params (ModelParamsConfig): Dictionary containing model
                                          hyperparameters.
        config (Config): Configuration object.
        targets (torch.Tensor): Tensor of target labels for
                                computing class weights.

    Returns:
        Tuple[torch.device, torch.nn.Module, torch.nn.Module]:
            - device: The device used for computations.
            - criterion: Loss function initialized with class weights.
            - model: Pre-trained model ready for inference or testing.
    """
    # Prepare configuration
    device_type = config.hardware.device
    min_key = config.data.keys.min
    max_key = config.data.keys.max
    num_keys = max_key - min_key + 1
    embedding_dim = config.model.sequence.embedding.dimension
    num_features = config.model.general.features

    # Define the device
    device = select_device(device_type)

    criterion = None
    if targets is not None:
        # Build criterion
        criterion = build_loss(targets, num_keys, device)

    # Instantiate LSTM model
    model = build_model(
        model_params, min_key, max_key, embedding_dim, num_features, config
    )

    # Move model to device
    model = move_to_device(model, device)

    info("Model components initialization completed")

    return device, criterion, model
