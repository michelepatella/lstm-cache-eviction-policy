from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from components.data_loader.targets.extractor import (
    extract_targets_from_data_loader,
)
from components.logs.levels.debug_logger import debug
from components.model.environment.initializer import (
    initialize_model_environment,
)
from components.model.io.locator import get_model_abs_path
from components.model.state_dict.loader import (
    load_model_state_dict,
)
from pipeline.config.pydantic.config import Config
from pipeline.config.pydantic.sections.model_config import ModelParamsConfig


def initialize_best_model(
    model_params: ModelParamsConfig,
    data_distribution_mode: str,
    config: Config,
    data_loader: Optional[DataLoader],
) -> Tuple[torch.device, nn.Module, nn.Module]:
    """
    Prepare a trained PyTorch model.

    This function extracts the target labels, sets up the PyTorch model environment,
    and loads pre-trained weights referring to the best PyTorch model.

    Args:
        model_params (ModelParamsConfig): Model hyperparameters.
        data_distribution_mode (str): Data distribution mode to determine the path
                                      of the trained model.
        config (Config): Configuration object.
        data_loader (Optional[DataLoader]): DataLoader containing the dataset to be
                                            used (if any).

    Returns:
        Tuple[torch.device, nn.Module, nn.Module]:
            - device: The device on which the model is loaded.
            - criterion: Loss function initialized with class weights.
            - model: Pre-trained model ready for inference.
    """
    # Get the model path
    model_path = get_model_abs_path(data_distribution_mode)

    # Extract targets from
    # provided data loader
    targets = extract_targets_from_data_loader(data_loader)

    # Setup for model environment
    device, criterion, model = initialize_model_environment(
        model_params, config, targets
    )

    # Load the trained model
    model = load_model_state_dict(model_path, model, device)

    debug(
        "Best model initialization executed",
        extra={
            "model_type": type(model).__name__,
            "device": str(device),
            "targets_num": len(targets) if targets is not None else None,
            "model_path": model_path,
            "context": "Best model initialization",
        },
    )

    return device, criterion, model
