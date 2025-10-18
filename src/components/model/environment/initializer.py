from typing import Tuple, Optional, Dict, Union

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
    model_params: Union[ModelParamsConfig, Dict[str, Union[int, float, bool]]],
    config: Config,
    targets: Optional[torch.Tensor],
) -> Tuple[torch.device, Optional[torch.nn.Module], torch.nn.Module]:
    """
    Set up the model environment.

    This function performs all steps required to prepare
    model environment:
        - Selects device
        - Defines the criterion
        - Instantiates the PyTorch model and moves it to the device

    Args:
        model_params (Union[ModelParamsConfig, Dict[str, Union[int, float, bool]]]):
            Model parameters.
        config (Config): Configuration object.
        targets (Optional[torch.Tensor]): Target labels for computing class weights.

    Returns:
        Tuple[torch.device, Optional[torch.nn.Module], torch.nn.Module]:
            - device: The device used for computations.
            - criterion: Loss function initialized with class weights (None if no targets
              provided).
            - model: PyTorch model.
    """
    device_type = config.hardware.device
    min_key = config.data.keys.min
    max_key = config.data.keys.max
    num_keys = max_key - min_key + 1
    num_features = config.model.general.features
    embedding_dim = config.model.sequence.embedding.dimension

    # Define the device for computations
    device = select_device(device_type)

    # Build criterion if targets
    # have been provided
    criterion = None
    if targets is not None:
        criterion = build_loss(targets, num_keys, device)

    # Instantiate model
    model = build_model(
        model_params, min_key, max_key, embedding_dim, num_features, config
    )

    # Move model to device
    model = move_to_device(model, device)

    info("Model components initialization completed")

    return device, criterion, model
