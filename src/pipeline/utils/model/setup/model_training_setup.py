from typing import Dict

import torch
from torch import nn
from torch.optim import Optimizer

from pipeline.config.classes.Config import Config
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info
from pipeline.utils.model.utils.class_weights_calculator import (
    calculate_class_weights,
)
from pipeline.utils.model.utils.LSTM import LSTM
from pipeline.utils.model.utils.optimizer_builder import build_optimizer


def model_training_setup(
    model_params: Dict[int | float | bool],
    learning_rate: float,
    targets: torch.Tensor,
    config: Config,
) -> tuple[torch.device, nn.Module, nn.Module, Optimizer]:
    """
    Set up the model training process.

    This function performs all steps required to prepare
    a PyTorch LSTM model for training:
        - Selects device
        - Defines the CrossEntropyLoss with computed class weights
        - Instantiates the LSTM model and moves it to the device
        - Builds the optimizer for the model parameters

    Parameters:
        model_params (Dict[int | float | bool]): Dictionary containing model
                                                 hyperparameters.
        learning_rate (float): Learning rate for the optimizer.
        targets (torch.Tensor): Tensor of target labels for
                                computing class weights.
        config (Config): Configuration object.

    Returns:
        tuple[torch.device, nn.Module, nn.Module, Optimizer]: Tuple containing the device
                                                              used for computations, loss
                                                              function with class weights
                                                              applied, instantiated LSTM
                                                              model on the selected device,
                                                              and optimizer configured for
                                                              the model parameters.
    """
    # Prepare configuration
    device_type = config.hardware.device
    optimizer_type = config.training.optimizer.type
    num_keys = config.data.general.keys.max - config.data.general.keys.min + 1

    try:
        # Define the device
        device = torch.device(device_type)

        debug(f"Device selected: {device}")
    except (RuntimeError, ValueError) as e:
        msg = "Failed to select device"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Compute class weights
    class_weights = calculate_class_weights(targets, num_keys)

    debug(f"Class weights computed: {class_weights}, for {num_keys} keys")

    try:
        # Define loss function
        weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)

        debug(f"{criterion} initialized as loss function with class weights")
    except (RuntimeError, TypeError) as e:
        msg = "Failed to define loss function"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Instantiate LSTM model
    model = LSTM(model_params, config)

    try:
        # Move model to device
        model.to(device)
    except RuntimeError as e:
        msg = "Failed to move model to device"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Build optimizer
    optimizer = build_optimizer(model, learning_rate, optimizer_type)

    debug(f"Optimizer created: {optimizer}")

    info("Model training setup completed")

    return device, criterion, model, optimizer
