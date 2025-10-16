from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from pipeline.config.pydantic.config import Config
from pipeline.config.pydantic.sections.model_config import ModelParamsConfig
from components.data_loader.targets.extractor import (
    extract_targets_from_data_loader,
)
from components.logs.levels.info_logger import info
from components.model.environment.initializer import initialize_model_environment
from components.model.io.locator import get_model_abs_path
from components.model.state_dict.loader import (
    load_model_state_dict,
)


def initialize_best_model(
    model_params: ModelParamsConfig,
    data_distribution_mode: str,
    config: Config,
    data_loader: DataLoader = None,
) -> Tuple[torch.device, nn.Module, nn.Module]:
    """
    Prepare a trained model for further usage.

    This function extracts the target labels, sets up the model components,
    and loads pre-trained weights.

    Args:
        model_params (ModelParamsConfig): Model hyperparameters.
        data_distribution_mode (str): Mode to determine the path of the trained model.
        config (Config): Configuration object.
        data_loader (DataLoader | None): DataLoader containing the
                                         dataset to be used.

    Returns:
        Tuple[torch.device, nn.Module, nn.Module]:
            - device: The device on which the model is loaded.
            - criterion: Loss function initialized with class weights.
            - model: Pre-trained model ready for inference or testing.
    """
    # Get the model path
    model_path = get_model_abs_path(data_distribution_mode)

    # Extract targets from
    # provided data loader
    targets = extract_targets_from_data_loader(data_loader)

    # Setup for model components
    device, criterion, model = initialize_model_environment(
        model_params, config, targets
    )

    # Load the trained model
    model = load_model_state_dict(model_path, model, device)

    info("Trained model initialization completed")

    return device, criterion, model
