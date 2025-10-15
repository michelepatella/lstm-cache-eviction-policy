from typing import Tuple

import torch

from pipeline.config.classes.Config import Config
from pipeline.config.classes.ModelConfig import ModelParamsConfig
from utils.criterion.builder import build_criterion
from utils.device.mover import (
    move_to_device,
)
from utils.device.selector import (
    select_device,
)
from utils.logs.levels.info_logger import info
from utils.model.building.builder import (
    build_model,
)


def initialize_model_components(
    model_params: ModelParamsConfig,
    config: Config,
    targets: torch.Tensor = None,
) -> Tuple[torch.device, torch.nn.Module, torch.nn.Module]:
    """
    Set up the model components.

    This function performs all steps required to prepare
    model components:
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
        Tuple[torch.device, torch.nn.Module, torch.nn.Module]: Tuple containing the device
                                                               used for computations, criterion,
                                                               and instantiated model
                                                               on the selected device.
    """
    # Prepare configuration
    device_type = config.hardware.device
    min_key = config.data.generation.keys.min
    max_key = config.data.generation.keys.max
    num_keys = max_key - min_key + 1
    embedding_dim = config.model.sequence.embedding.dimension
    num_features = config.model.general.features

    # Define the device
    device = select_device(device_type)

    criterion = None
    if targets is not None:
        # Build criterion
        criterion = build_criterion(targets, num_keys, device)

    # Instantiate LSTM model
    model = build_model(
        model_params, min_key, max_key, embedding_dim, num_features, config
    )

    # Move model to device
    model = move_to_device(model, device)

    info("Model components initialization completed")

    return device, criterion, model
