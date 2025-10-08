import torch
from torch import nn
from torch.optim import Optimizer

from config.classes.Config import Config
from config.classes.ModelConfig import ModelParamsConfig
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info
from utils.model.initialization.utils.class_weights_calculator import (
    calculate_class_weight,
)
from utils.model.initialization.utils.optimizer_builder import build_optimizer
from utils.model.LSTM import LSTM


def initialize_model_components(
    model_params: ModelParamsConfig,
    learning_rate: float,
    config: Config,
    targets: torch.Tensor = None,
) -> tuple[torch.device, nn.Module, nn.Module, Optimizer]:
    """
    Set up the model components for
    further usage.

    This function performs all steps required to prepare
    PyTorch LSTM model components:
        - Selects device
        - Defines the CrossEntropyLoss with computed class weights
        - Instantiates the LSTM model and moves it to the device
        - Builds the optimizer for the model parameters

    Args:
        model_params (ModelParamsConfig): Dictionary containing model
                                                 hyperparameters.
        learning_rate (float): Learning rate for the optimizer.
        config (Config): Configuration object.
        targets (torch.Tensor): Tensor of target labels for
                                computing class weights.

    Returns:
        tuple[
        torch.device, nn.Module, nn.Module, Optimizer
        ]: Tuple containing the device
           used for computations, loss
           function with class weights
           applied, instantiated LSTM
           model on the selected device,
           and optimizer configured for
           the model parameters.
    Raises:
        RuntimeError: If an error occurs while
                      setting up the model components, e.g.:
            * Failed to select device due to unsupported device type.
            * Failed to define the loss function due to incompatible
              class weights or device issues.
            * Failed to move the model to the device.
    """
    # Prepare configuration
    device_type = config.hardware.device
    optimizer_type = config.training.optimizer.type
    min_key = config.data.generation.keys.min
    max_key = config.data.generation.keys.max
    num_keys = max_key - min_key + 1
    embedding_dim = config.model.sequence.embedding.dimension
    num_features = config.model.general.features

    try:
        # Define the device
        device = torch.device(device_type)

        debug(f"Device selected: {device}")
    except (RuntimeError, ValueError) as e:
        msg = "Failed to select device"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    criterion = None
    if targets is not None:
        # Compute class weights
        class_weights = calculate_class_weight(targets, num_keys)

        debug(f"Class weights computed: {class_weights}, for {num_keys} keys")

        try:
            # Define loss function
            weight = torch.tensor(class_weights, dtype=torch.float32).to(
                device
            )
            criterion = nn.CrossEntropyLoss(weight=weight)

            debug(
                f"{criterion} initialized as loss function with class weights"
            )
        except (RuntimeError, TypeError) as e:
            msg = "Failed to define loss function"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    # Instantiate LSTM model
    model = LSTM(
        model_params, min_key, max_key, embedding_dim, num_features, config
    )

    try:
        # Move model to device
        model.to(device)
    except RuntimeError as e:
        msg = "Failed to move model to device"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Build optimizer
    optimizer = build_optimizer(model, learning_rate, optimizer_type, config)

    debug(f"Optimizer created: {optimizer}")

    info("Model training initialization completed")

    return device, criterion, model, optimizer
