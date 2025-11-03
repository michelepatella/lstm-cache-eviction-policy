from typing import Any

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
from components.model.io.loader import (
    load_model,
)
from components.model.io.locator import get_model_abs_path


def initialize_best_model(
    data_distribution_mode: str,
    device_type: str,
    config: Any,
    data_loader: DataLoader | None,
) -> tuple[torch.device, nn.Module, nn.Module]:
    """Prepare a trained PyTorch model.

    This function extracts the target labels, sets up the PyTorch
    model environment, and loads pre-trained weights referring to
    the best PyTorch model.

    Args:
        data_distribution_mode (str): Data distribution mode to
                                      determine the path
                                      of the trained model.
        device_type (str): Device type to be used.
        config (Any): Configuration object.
        data_loader (DataLoader | None): DataLoader containing the dataset
                                         to be used (if any).

    Returns:
        tuple[torch.device, nn.Module, nn.Module]:
            - device: The device on which the model is loaded.
            - criterion: Loss function initialized with class weights.
            - model: Pre-trained model ready for inference.
    """
    # Get the model path
    model_path = get_model_abs_path(data_distribution_mode)

    # Extract targets from
    # provided data loader
    targets = extract_targets_from_data_loader(data_loader)

    # Load the trained model
    model = load_model(model_path)

    # Setup for model environment
    device, criterion, model = initialize_model_environment(
        targets,
        device_type,
        config,
        model=model,
    )

    debug(
        "Best model initialization executed",
        extra={
            "model_type": type(model).__name__,
            "device": str(device),
            "targets_num": len(targets) if targets is not None else None,
            "model_path": str(model_path),
            "context": "Best model initialization",
        },
    )

    return device, criterion, model
