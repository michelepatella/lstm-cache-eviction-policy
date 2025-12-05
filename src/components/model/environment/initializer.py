"""initializer.py

Utility module for setting up the PyTorch model environment.

This module provides the `initialize_model_environment` function, which
handles device selection, criterion definition based on target labels,
model instantiation, and moving the model to the selected device.

Functions:
    initialize_model_environment(
        targets: torch.Tensor | None,
        device_type: str,
        config: Any,
        model: torch.nn.Module = None,
        model_params: Any | dict[str, int | float | bool] = None
    ) -> tuple[torch.device, torch.nn.Module | None, torch.nn.Module]
        Prepares the device, criterion, and PyTorch model for training or
        inference.
"""

from typing import Any

import torch

from components.const import DATASET_FEATURE_COLUMNS
from components.device.mover import (
    move_to_device,
)
from components.device.selector import (
    select_device,
)
from components.logs.levels.debug_logger import debug
from components.loss.builder import build_loss
from components.model.builder import (
    build_model,
)


def initialize_model_environment(
    targets: torch.Tensor | None,
    device_type: str,
    config: Any,
    model: torch.nn.Module = None,
    model_params: Any | dict[str, int | float | bool] = None,
) -> tuple[torch.device, torch.nn.Module | None, torch.nn.Module]:
    """Set up the model environment.

    This function performs all steps required to prepare
    model environment:
        - Selects device
        - Defines the criterion
        - Instantiates the PyTorch model and moves it to the device

    Args:
        targets (torch.Tensor | None): Target labels for computing
                                         class weights.
        device_type (str): Device type to use.
        config (Any): Configuration object.
        model (torch.nn.Module): A PyTorch model to configure environment for.
        model_params (Any | dict[str, int | float | bool]):
            Model parameters.

    Returns:
        tuple[torch.device, torch.nn.Module | None, torch.nn.Module]:
            - device: The device used for computations.
            - criterion: Loss function initialized with class weights
                         (None if no targets provided).
            - model: PyTorch model.
    """
    min_key = config.data.general.keys.min
    max_key = config.data.general.keys.max
    num_keys = max_key - min_key + 1
    embedding_dim = config.model.sequence.embedding.dimension
    class_weight_type = config.loss.class_weight.type
    num_features = len(DATASET_FEATURE_COLUMNS)

    # Define the device for computations
    device = select_device(device_type)

    # Build criterion if targets
    # have been provided
    criterion = None
    if targets is not None:
        criterion = build_loss(targets, num_keys, class_weight_type, device)

    if model is None:
        # Instantiate model
        model = build_model(
            model_params,
            min_key,
            max_key,
            embedding_dim,
            num_features,
            config,
        )

    # Move model to device
    model = move_to_device(model, device)

    debug(
        "Model environment initialization executed",
        extra={
            "model_type": type(model).__name__,
            "device": str(device),
            "criterion_defined": criterion is not None,
            "keys_num": num_keys,
            "features_keys": num_features,
            "embedding_dim": embedding_dim,
            "context": "Model environment initialization",
        },
    )

    return device, criterion, model
